//! `diff`: are two COPC files the same point set? Exact comparison over a set of
//! fields, independent of point order and octree layout.
//!
//! **Pass 1** hashes every point's normalised record (xxh3-128) and sums the
//! hashes per tile: a wrapping 128-bit sum, so order doesn't matter and
//! duplicates count. The tiles are a cut through a cube covering both files,
//! sized by both files' points (so a crop compared with the full cloud doesn't
//! pile the rest into a few border tiles). A point's tile comes from its
//! normalised position, so equal points land in the same tile in both files.
//! Tiles whose sums or counts differ hold the differences.
//!
//! **Pass 2** decodes only the nodes that put points into those tiles, sorts the
//! tiles' records and merges them. Leftover records with the same position (and
//! GPS time, if compared) are paired up as changed points, with the fields that
//! differ. The rest exist in only one file.
//!
//! Positions are compared as raw integers if both files have the same scale and
//! offset on an axis, and otherwise on the coarser file's grid. A node with the
//! same key and byte-identical compressed chunk in two files with the same
//! record layout decodes to the same points, so it's skipped in both.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{Result, bail, ensure};
use copc_streaming::{Fields, HierarchyEntry, VoxelKey};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use rayon::prelude::*;
use xxhash_rust::xxh3::xxh3_128;

use crate::extra_bytes::{Descriptor, parse_descriptors};
use crate::input::{CopcFile, key_tuple};
use crate::pipeline::ordered_parallel;
use crate::tiles::Cut;

/// Default target X + Y points per tile: how finely pass 1 localises
/// differences, and how much pass 2 decodes per differing tile.
pub const DEFAULT_TILE_POINTS: u64 = 1_000_000;

/// The name for X's and Y's position: always compared.
pub const POSITION: &str = "xyz";
/// Selects every extra bytes field in `--fields` / `--ignore`.
pub const EXTRA: &str = "extra";

pub struct Options {
    /// Compare only these fields besides the position. `None`: every field both
    /// files have.
    pub fields: Option<Vec<String>>,
    /// Don't compare these fields.
    pub ignore: Vec<String>,
    pub tile_points: u64,
    /// Stop after pass 1: whether and where the files differ, not which points.
    pub quick: bool,
    /// Examples to print per kind of difference.
    pub limit: usize,
    pub progress: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self { fields: None, ignore: Vec::new(), tile_points: DEFAULT_TILE_POINTS, quick: false, limit: 10, progress: false }
    }
}

#[derive(Debug, Default)]
pub struct Report {
    /// Same point set over the compared fields.
    pub equal: bool,
    pub compared: Vec<String>,
    /// Field name and why it isn't compared.
    pub not_compared: NotCompared,
    pub tiles: usize,
    pub differing_tiles: usize,
    /// Nodes (in each file) and points skipped as byte-identical.
    pub identical_nodes: usize,
    pub identical_points: u64,
    /// Points decoded in pass 1, X and Y.
    pub decoded: [u64; 2],
    /// Pass 2 ran (not `quick`).
    pub detailed: bool,
    /// Points only in X, only in Y.
    pub only: [u64; 2],
    /// Points with the same position (and GPS time) whose other fields differ.
    pub changed: u64,
    /// For changed points: how many differ in each field.
    pub changed_by_field: Vec<(String, u64)>,
}

#[derive(Clone, Debug)]
enum Show {
    U8,
    U16,
    Returns,
    Channel,
    Flags,
    ScanAngle,
    Gps,
    Rgb,
    Extra(Descriptor),
}

impl Show {
    fn show(&self, b: &[u8]) -> String {
        let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
        match self {
            Show::U8 => b[0].to_string(),
            Show::U16 => u16_at(0).to_string(),
            Show::Returns => format!("{}/{}", b[0] & 0xF, b[0] >> 4),
            Show::Channel => ((b[0] >> 4) & 3).to_string(),
            Show::Flags => format!("{:#04x}", b[0]),
            Show::ScanAngle => format!("{:.3}°", f64::from(u16_at(0) as i16) * 0.006),
            Show::Gps => f64::from_le_bytes(b.try_into().unwrap()).to_string(),
            Show::Rgb => format!("({},{},{})", u16_at(0), u16_at(2), u16_at(4)),
            Show::Extra(d) => d.show(b),
        }
    }
}

/// One compared field: where it sits in each file's records and in the
/// normalised record.
#[derive(Clone, Debug)]
struct Field {
    name: String,
    at: [usize; 2],
    len: usize,
    /// For fields sharing a byte: the bits that belong to the field.
    mask: u8,
    decode: Fields,
    show: Show,
    extra: bool,
    /// Offset in the normalised record.
    pos: usize,
}

/// A standard PF 6–8 field.
struct StdField {
    name: &'static str,
    at: usize,
    len: usize,
    mask: u8,
    decode: Fields,
    show: Show,
    /// Point formats that have it.
    formats: &'static [u8],
}

fn standard() -> Vec<StdField> {
    const ALL: &[u8] = &[6, 7, 8];
    let f = |name, at, len, mask, decode, show, formats| StdField { name, at, len, mask, decode, show, formats };
    vec![
        f("gps_time", 22, 8, 0xFF, Fields::GPS_TIME, Show::Gps, ALL),
        f("intensity", 12, 2, 0xFF, Fields::INTENSITY, Show::U16, ALL),
        f("returns", 14, 1, 0xFF, Fields::empty(), Show::Returns, ALL),
        // The scanner channel is in the base layer, but shares byte 15 with the
        // flags; decode the flags layer too rather than rely on what a skipped
        // layer leaves there.
        f("channel", 15, 1, 0x30, Fields::FLAGS, Show::Channel, ALL),
        f("flags", 15, 1, 0xCF, Fields::FLAGS, Show::Flags, ALL),
        f("classification", 16, 1, 0xFF, Fields::CLASSIFICATION, Show::U8, ALL),
        f("user_data", 17, 1, 0xFF, Fields::USER_DATA, Show::U8, ALL),
        f("scan_angle", 18, 2, 0xFF, Fields::SCAN_ANGLE, Show::ScanAngle, ALL),
        f("point_source_id", 20, 2, 0xFF, Fields::POINT_SOURCE_ID, Show::U16, ALL),
        f("rgb", 30, 6, 0xFF, Fields::RGB, Show::Rgb, &[7, 8]),
        f("nir", 36, 2, 0xFF, Fields::NIR, Show::U16, &[8]),
    ]
}

/// Field name and why it isn't compared.
type NotCompared = Vec<(String, String)>;

/// The fields both files have, filtered by the options, with `gps_time` first;
/// and the fields not compared, with the reason.
fn select_fields(f: [&CopcFile; 2], opt: &Options) -> Result<(Vec<Field>, NotCompared)> {
    let std_fields = standard();
    let only = |side: usize| if side == 0 { "only in X" } else { "only in Y" };
    let mut fields = Vec::new();
    let mut skipped = NotCompared::new();
    for s in &std_fields {
        match f.map(|f| s.formats.contains(&f.point_format())) {
            [true, true] => fields.push(Field {
                name: s.name.to_string(),
                at: [s.at; 2],
                len: s.len,
                mask: s.mask,
                decode: s.decode,
                show: s.show.clone(),
                extra: false,
                pos: 0,
            }),
            [true, false] => skipped.push((s.name.to_string(), only(0).into())),
            [false, true] => skipped.push((s.name.to_string(), only(1).into())),
            [false, false] => {}
        }
    }

    // Extra bytes by name. Repeated names get #2, #3; a name that clashes with
    // a standard field gets an `extra:` prefix.
    let mut extras: [Vec<(String, Descriptor, usize)>; 2] = Default::default();
    for side in 0..2 {
        let n_extra = usize::from(f[side].las_header().point_format().extra_bytes);
        let base = f[side].record_len() - n_extra;
        let mut seen: HashMap<String, usize> = HashMap::new();
        for d in parse_descriptors(f[side].extra_bytes_vlr(), n_extra)? {
            let mut name = d.name.clone();
            if std_fields.iter().any(|s| s.name == name) || name == POSITION || name == EXTRA {
                name = format!("extra:{name}");
            }
            let n = seen.entry(name.clone()).or_default();
            *n += 1;
            if *n > 1 {
                name = format!("{name}#{n}");
            }
            let at = base + d.start;
            extras[side].push((name, d, at));
        }
    }
    for (name, dx, at_x) in &extras[0] {
        match extras[1].iter().find(|(n, ..)| n == name) {
            Some((_, dy, at_y)) if dx.same_encoding(dy) => fields.push(Field {
                name: name.clone(),
                at: [*at_x, *at_y],
                len: dx.len,
                mask: 0xFF,
                decode: Fields::EXTRA_BYTES,
                show: Show::Extra(dx.clone()),
                extra: true,
                pos: 0,
            }),
            Some(_) => skipped.push((name.clone(), "encoded differently".into())),
            None => skipped.push((name.clone(), only(0).into())),
        }
    }
    for (name, ..) in &extras[1] {
        if !extras[0].iter().any(|(n, ..)| n == name) {
            skipped.push((name.clone(), only(1).into()));
        }
    }

    // The user's selection.
    let mut known: Vec<String> = vec![POSITION.into(), EXTRA.into()];
    known.extend(std_fields.iter().map(|s| s.name.to_string()));
    known.extend(extras.iter().flatten().map(|(n, ..)| n.clone()));
    for n in opt.fields.iter().flatten().chain(&opt.ignore) {
        if !known.contains(n) {
            bail!("unknown field '{n}'; known: {}", known.join(", "));
        }
    }
    ensure!(!opt.ignore.iter().any(|n| n == POSITION), "the position ({POSITION}) is always compared");
    let named = |list: &[String], f: &Field| list.iter().any(|n| *n == f.name || (f.extra && n == EXTRA));
    let (kept, dropped): (Vec<Field>, Vec<Field>) = fields
        .into_iter()
        .partition(|f| opt.fields.as_deref().is_none_or(|l| named(l, f)) && !named(&opt.ignore, f));
    skipped.extend(dropped.into_iter().map(|f| (f.name, "not selected".into())));

    let mut fields = kept;
    let mut pos = 24;
    for f in &mut fields {
        f.pos = pos;
        pos += f.len;
    }
    Ok((fields, skipped))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Axis {
    /// The raw integer is the normalised coordinate.
    Raw,
    /// `round((raw * s + d) / q)`.
    Scaled { s: f64, d: f64 },
}

/// The common position grid: spacing `q` from `origin`, per axis.
#[derive(Debug)]
struct Grid {
    q: [f64; 3],
    origin: [f64; 3],
    axes: [[Axis; 3]; 2],
}

impl Grid {
    fn new(f: [&CopcFile; 2]) -> Self {
        let so = f.map(|f| f.scale_offset());
        let mut g = Grid { q: [0.0; 3], origin: [0.0; 3], axes: [[Axis::Raw; 3]; 2] };
        for a in 0..3 {
            let (s, o) = ([so[0].0[a], so[1].0[a]], [so[0].1[a], so[1].1[a]]);
            // The coarser file's grid (X's on a tie); its integers are used as is.
            let coarse = usize::from(s[1] > s[0]);
            let fine = 1 - coarse;
            g.q[a] = s[coarse];
            g.origin[a] = o[coarse];
            if s[fine] != s[coarse] || o[fine] != o[coarse] {
                g.axes[fine][a] = Axis::Scaled { s: s[fine], d: o[fine] - o[coarse] };
            }
        }
        g
    }

    fn raw(&self) -> bool {
        self.axes.iter().flatten().all(|a| *a == Axis::Raw)
    }

    #[inline]
    fn canon(&self, side: usize, raw: &[u8]) -> [i64; 3] {
        std::array::from_fn(|a| {
            let r = i32::from_le_bytes(raw[4 * a..4 * a + 4].try_into().unwrap());
            match self.axes[side][a] {
                Axis::Raw => i64::from(r),
                Axis::Scaled { s, d } => ((f64::from(r) * s + d) / self.q[a]).round() as i64,
            }
        })
    }

    #[inline]
    fn world(&self, c: [i64; 3]) -> [f64; 3] {
        std::array::from_fn(|a| c[a] as f64 * self.q[a] + self.origin[a])
    }

    /// Decimals that show one grid step.
    fn decimals(&self) -> usize {
        self.q.iter().map(|q| (-q.log10()).ceil().max(0.0) as usize).max().unwrap_or(3).min(9)
    }
}

/// Per tile: wrapping sum of record hashes and point count.
type TileSum = (u128, u64);

struct Diff<'a> {
    files: [&'a CopcFile; 2],
    fields: Vec<Field>,
    /// Normalised record: position (3 × 8 bytes, big-endian with the sign bit
    /// flipped, so byte order is numeric order), then the fields.
    width: usize,
    /// The sort key that identifies a point: position and GPS time if compared.
    key_len: usize,
    decode: Fields,
    grid: Grid,
    cut: Cut,
}

impl Diff<'_> {
    /// Write `raw`'s normalised record into `out`; returns the position.
    #[inline]
    fn record(&self, side: usize, raw: &[u8], out: &mut [u8]) -> [i64; 3] {
        let c = self.grid.canon(side, raw);
        for a in 0..3 {
            out[8 * a..8 * a + 8].copy_from_slice(&((c[a] as u64) ^ (1 << 63)).to_be_bytes());
        }
        for f in &self.fields {
            let at = f.at[side];
            out[f.pos..f.pos + f.len].copy_from_slice(&raw[at..at + f.len]);
            out[f.pos] &= f.mask;
        }
        c
    }

    fn position(&self, r: &[u8]) -> [f64; 3] {
        self.grid.world(std::array::from_fn(|a| {
            (u64::from_be_bytes(r[8 * a..8 * a + 8].try_into().unwrap()) ^ (1 << 63)) as i64
        }))
    }

    /// Visit every point of a node: normalised record and tile.
    fn for_each_point(&self, side: usize, e: &HierarchyEntry, mut visit: impl FnMut(&[u8], usize)) -> Result<()> {
        let f = self.files[side];
        let chunk = f.decode(e, self.decode)?;
        let mut buf = vec![0u8; self.width];
        let mut hint = 0;
        for raw in chunk.cloud().raw_bytes().chunks_exact(f.record_len()) {
            let c = self.record(side, raw, &mut buf);
            hint = self.cut.locate_near(self.grid.world(c), hint);
            visit(&buf, hint);
        }
        Ok(())
    }

    /// Pass 1 for one node: (tile, hash sum, count) for each tile it reaches.
    fn hash_node(&self, side: usize, e: &HierarchyEntry) -> Result<Vec<(u32, u128, u64)>> {
        let mut acc: HashMap<usize, TileSum> = HashMap::new();
        // Consecutive points mostly share a tile: sum runs before the map.
        let (mut cur, mut sum, mut n) = (0usize, 0u128, 0u64);
        let mut flush = |cur: usize, sum: u128, n: u64| {
            if n > 0 {
                let a = acc.entry(cur).or_default();
                a.0 = a.0.wrapping_add(sum);
                a.1 += n;
            }
        };
        self.for_each_point(side, e, |r, t| {
            if t != cur {
                flush(cur, sum, n);
                (cur, sum, n) = (t, 0, 0);
            }
            sum = sum.wrapping_add(xxh3_128(r));
            n += 1;
        })?;
        flush(cur, sum, n);
        let mut v: Vec<_> = acc.into_iter().map(|(t, (s, n))| (t as u32, s, n)).collect();
        v.sort_unstable_by_key(|x| x.0);
        Ok(v)
    }

    /// Pass 2 for one tile, from the nodes that reached it in pass 1.
    fn diff_tile(&self, t: usize, nodes: &[Vec<u32>; 2], items: &[(usize, &HierarchyEntry)], expect: [u64; 2], limit: usize) -> Result<TileDiff> {
        let mut recs: [Vec<u8>; 2] = Default::default();
        for side in 0..2 {
            for &i in &nodes[side] {
                let (s, e) = items[i as usize];
                debug_assert_eq!(s, side);
                self.for_each_point(side, e, |r, pt| {
                    if pt == t {
                        recs[side].extend_from_slice(r);
                    }
                })?;
            }
            ensure!(
                (recs[side].len() / self.width) as u64 == expect[side],
                "tile {:?}: pass 2 found {} points of {}, pass 1 {}",
                self.cut.tiles[t],
                recs[side].len() / self.width,
                ["X", "Y"][side],
                expect[side]
            );
        }
        let rows = recs.each_ref().map(|r| {
            let mut v: Vec<&[u8]> = r.chunks_exact(self.width).collect();
            v.sort_unstable();
            v
        });
        let (changed, only) = pair_by_key(unmatched(&rows[0], &rows[1]), self.key_len);
        ensure!(
            !changed.is_empty() || only.iter().any(|v| !v.is_empty()),
            "tile {:?}: hash sums differ but the records match",
            self.cut.tiles[t]
        );

        let mut d = TileDiff {
            only: only.each_ref().map(|v| v.len() as u64),
            changed: changed.len() as u64,
            by_field: vec![0; self.fields.len()],
            examples: Default::default(),
        };
        for (a, b) in &changed {
            for (i, f) in self.fields.iter().enumerate() {
                if a[f.pos..f.pos + f.len] != b[f.pos..f.pos + f.len] {
                    d.by_field[i] += 1;
                }
            }
        }
        for (e, rows) in d.examples.iter_mut().zip(&only) {
            *e = rows.iter().take(limit).map(|r| self.describe(r)).collect();
        }
        d.examples[2] = changed.iter().take(limit).map(|(a, b)| self.describe_change(a, b)).collect();
        Ok(d)
    }

    fn describe(&self, r: &[u8]) -> String {
        let p = self.position(r);
        let d = self.grid.decimals();
        let mut s = format!("({:.d$}, {:.d$}, {:.d$})", p[0], p[1], p[2]);
        for f in &self.fields {
            s += &format!(" {}={}", f.name, f.show.show(&r[f.pos..f.pos + f.len]));
        }
        s
    }

    fn describe_change(&self, a: &[u8], b: &[u8]) -> String {
        let p = self.position(a);
        let d = self.grid.decimals();
        let mut s = format!("({:.d$}, {:.d$}, {:.d$})", p[0], p[1], p[2]);
        for f in &self.fields {
            let (va, vb) = (&a[f.pos..f.pos + f.len], &b[f.pos..f.pos + f.len]);
            if f.pos < self.key_len {
                s += &format!(" {}={}", f.name, f.show.show(va));
            } else if va != vb {
                s += &format!(" {} {} -> {}", f.name, f.show.show(va), f.show.show(vb));
            }
        }
        s
    }
}

struct TileDiff {
    only: [u64; 2],
    changed: u64,
    by_field: Vec<u64>,
    /// Only in X, only in Y, changed.
    examples: [Vec<String>; 3],
}

/// Rows of normalised records.
type Rows<'r> = Vec<&'r [u8]>;
/// Changed pairs (X, Y), and the rows only in X and only in Y.
type Paired<'r> = (Vec<(&'r [u8], &'r [u8])>, [Rows<'r>; 2]);

/// Multiset difference of two sorted lists: the rows of each without a match in
/// the other, still sorted.
fn unmatched<'r>(a: &[&'r [u8]], b: &[&'r [u8]]) -> [Rows<'r>; 2] {
    let (mut i, mut j) = (0, 0);
    let mut out: [Vec<&[u8]>; 2] = Default::default();
    while i < a.len() && j < b.len() {
        match a[i].cmp(b[j]) {
            std::cmp::Ordering::Equal => (i, j) = (i + 1, j + 1),
            std::cmp::Ordering::Less => {
                out[0].push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out[1].push(b[j]);
                j += 1;
            }
        }
    }
    out[0].extend_from_slice(&a[i..]);
    out[1].extend_from_slice(&b[j..]);
    out
}

/// Pair unmatched rows of X and Y with the same key prefix (in sorted order);
/// returns the pairs and the rows left over.
fn pair_by_key(u: [Rows<'_>; 2], key_len: usize) -> Paired<'_> {
    let [a, b] = u;
    let (mut i, mut j) = (0, 0);
    let mut pairs = Vec::new();
    let mut only: [Vec<&[u8]>; 2] = Default::default();
    while i < a.len() && j < b.len() {
        match a[i][..key_len].cmp(&b[j][..key_len]) {
            std::cmp::Ordering::Equal => {
                pairs.push((a[i], b[j]));
                (i, j) = (i + 1, j + 1);
            }
            std::cmp::Ordering::Less => {
                only[0].push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                only[1].push(b[j]);
                j += 1;
            }
        }
    }
    only[0].extend_from_slice(&a[i..]);
    only[1].extend_from_slice(&b[j..]);
    (pairs, only)
}

/// Same record layout: identical compressed chunks decode to identical records.
fn same_layout(x: &CopcFile, y: &CopcFile) -> bool {
    let laz = |f: &CopcFile| f.vlrs.iter().find(|v| v.is("laszip encoded", 22204)).map(|v| v.data.clone());
    x.point_format() == y.point_format()
        && x.record_len() == y.record_len()
        && x.scale_offset() == y.scale_offset()
        && x.extra_bytes_vlr() == y.extra_bytes_vlr()
        && laz(x) == laz(y)
}

fn bar(len: u64, unit: &str, show: bool) -> ProgressBar {
    if !show {
        return ProgressBar::hidden();
    }
    let bar = ProgressBar::with_draw_target(Some(len), ProgressDrawTarget::stderr());
    bar.set_style(
        ProgressStyle::with_template(&format!(
            "{{elapsed_precise}} [{{wide_bar}}] {{percent:>3}}% {{human_pos}}/{{human_len}} {unit}, ETA {{eta}}"
        ))
        .expect("valid template")
        .progress_chars("=> "),
    );
    bar
}

fn describe_file(name: &str, f: &CopcFile) {
    let (s, o) = f.scale_offset();
    println!("{name}: {}", f.path.display());
    println!(
        "   {} points, format {}, {} extra bytes, scale {s:?}, offset {o:?}",
        f.las_header().number_of_points(),
        f.point_format(),
        f.las_header().point_format().extra_bytes
    );
}

/// Compare the point sets of `x` and `y`, print what differs, and return the
/// report.
pub fn run(x: &CopcFile, y: &CopcFile, opt: &Options) -> Result<Report> {
    let files = [x, y];
    let (fields, not_compared) = select_fields(files, opt)?;
    describe_file("X", x);
    describe_file("Y", y);
    println!(
        "CRS: {}",
        match (x.wkt(), y.wkt()) {
            (Some(a), Some(b)) if a == b => "same",
            (Some(_), Some(_)) => "differs (not part of the comparison)",
            (None, None) => "none declared",
            _ => "declared in only one file (not part of the comparison)",
        }
    );

    let grid = Grid::new(files);
    let width = fields.last().map_or(24, |f| f.pos + f.len);
    let key_len = fields.iter().find(|f| f.name == "gps_time").map_or(24, |f| f.pos + f.len);
    let decode = fields.iter().fold(Fields::Z, |m, f| m | f.decode);
    if grid.raw() {
        println!("position: raw integers (same scale and offset)");
    } else {
        println!("position: on a common grid of {:?} from {:?} (the coarser scale per axis)", grid.q, grid.origin);
    }
    let compared: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
    println!("compared: {POSITION}{}", compared.iter().map(|n| format!(", {n}")).collect::<String>());
    if !not_compared.is_empty() {
        let list: Vec<String> = not_compared.iter().map(|(n, why)| format!("{n} ({why})")).collect();
        println!("not compared: {}", list.join(", "));
    }

    let t0 = Instant::now();
    let cut = Cut::covering(&files, opt.tile_points);
    let d = Diff { files, fields, width, key_len, decode, grid, cut };

    // Byte-identical nodes.
    let mut identical: HashSet<(i32, i32, i32, i32)> = HashSet::new();
    let mut identical_points = 0u64;
    if same_layout(x, y) {
        let xs: Vec<&HierarchyEntry> = x.data_entries().collect();
        let same: Vec<bool> = xs
            .par_iter()
            .map(|e| y.get(&e.key).is_some_and(|f| f.point_count == e.point_count && x.compressed(e) == y.compressed(f)))
            .collect();
        for (e, s) in xs.iter().zip(same) {
            if s {
                identical.insert(key_tuple(&e.key));
                identical_points += u64::from(e.point_count);
            }
        }
    }
    let is_identical = |k: &VoxelKey| identical.contains(&key_tuple(k));

    // Pass 1.
    let mut items: Vec<(usize, &HierarchyEntry)> = Vec::new();
    let mut decoded = [0u64; 2];
    for (side, f) in files.iter().enumerate() {
        for e in f.data_entries().filter(|e| !is_identical(&e.key)) {
            items.push((side, e));
            decoded[side] += u64::from(e.point_count);
        }
    }
    let pb = bar(decoded[0] + decoded[1], "points", opt.progress);
    let per_node: Vec<Vec<(u32, u128, u64)>> = items
        .par_iter()
        .map(|&(side, e)| {
            let r = d.hash_node(side, e);
            pb.inc(u64::from(e.point_count));
            r
        })
        .collect::<Result<_>>()?;
    pb.finish_and_clear();
    let n_tiles = d.cut.tiles.len();
    let mut sums: Vec<[TileSum; 2]> = vec![[(0, 0); 2]; n_tiles];
    let mut tile_nodes: Vec<[Vec<u32>; 2]> = vec![Default::default(); n_tiles];
    for (i, ((side, _), contrib)) in items.iter().zip(&per_node).enumerate() {
        for &(t, h, n) in contrib {
            let s = &mut sums[t as usize][*side];
            s.0 = s.0.wrapping_add(h);
            s.1 += n;
            tile_nodes[t as usize][*side].push(i as u32);
        }
    }
    let differing: Vec<usize> = (0..n_tiles).filter(|&t| sums[t][0] != sums[t][1]).collect();
    println!(
        "pass 1: {n_tiles} tiles, decoded {} + {} points; {} nodes ({identical_points} points) byte-identical, not decoded; {:.1} s",
        decoded[0],
        decoded[1],
        identical.len(),
        t0.elapsed().as_secs_f64()
    );
    let mut report = Report {
        equal: differing.is_empty(),
        compared,
        not_compared,
        tiles: n_tiles,
        differing_tiles: differing.len(),
        identical_nodes: identical.len(),
        identical_points,
        decoded,
        detailed: false,
        only: [0; 2],
        changed: 0,
        changed_by_field: Vec::new(),
    };
    if !differing.is_empty() {
        println!("{} tiles differ:", differing.len());
        for &t in differing.iter().take(opt.limit) {
            let k = &d.cut.tiles[t];
            let b = d.cut.tile_bounds(t);
            println!(
                "  tile {}/{}/{}/{} [{:.2}, {:.2}, {:.2}]–[{:.2}, {:.2}, {:.2}]: X {} points, Y {} (outside byte-identical nodes)",
                k.level, k.x, k.y, k.z, b.min[0], b.min[1], b.min[2], b.max[0], b.max[1], b.max[2], sums[t][0].1, sums[t][1].1
            );
        }
        if differing.len() > opt.limit {
            println!("  ...");
        }
    }

    // Pass 2.
    if !differing.is_empty() && !opt.quick {
        let t1 = Instant::now();
        let pb = bar(differing.len() as u64, "tiles", opt.progress);
        let mut by_field = vec![0u64; d.fields.len()];
        let mut examples: [Vec<String>; 3] = Default::default();
        ordered_parallel(
            &differing,
            rayon::current_num_threads(),
            |_, &t| d.diff_tile(t, &tile_nodes[t], &items, [sums[t][0].1, sums[t][1].1], opt.limit),
            |_, td| {
                report.only[0] += td.only[0];
                report.only[1] += td.only[1];
                report.changed += td.changed;
                for (a, b) in by_field.iter_mut().zip(&td.by_field) {
                    *a += b;
                }
                for (e, new) in examples.iter_mut().zip(td.examples) {
                    let room = opt.limit.saturating_sub(e.len());
                    e.extend(new.into_iter().take(room));
                }
                pb.inc(1);
                Ok(())
            },
        )?;
        pb.finish_and_clear();
        report.detailed = true;
        report.changed_by_field =
            d.fields.iter().zip(by_field).filter(|(_, n)| *n > 0).map(|(f, n)| (f.name.clone(), n)).collect();
        println!("pass 2: {:.1} s", t1.elapsed().as_secs_f64());
        let key = if key_len > 24 { "position and GPS time" } else { "position" };
        println!("only in X: {} points", report.only[0]);
        println!("only in Y: {} points", report.only[1]);
        let changed: Vec<String> = report.changed_by_field.iter().map(|(n, c)| format!("{n} {c}")).collect();
        println!("changed (same {key}): {} points{}", report.changed, if changed.is_empty() { String::new() } else { format!(": {}", changed.join(", ")) });
        for (label, e) in ["only in X", "only in Y", "changed, X -> Y"].iter().zip(&examples) {
            if !e.is_empty() {
                println!("examples, {label}:");
                for s in e {
                    println!("  {s}");
                }
            }
        }
    }
    println!("{}", if report.equal { "point sets are identical" } else { "point sets differ" });
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiset_difference_and_pairing() {
        // Key = first byte.
        let a: Vec<&[u8]> = vec![&[1, 0], &[1, 0], &[2, 5], &[3, 0], &[4, 1]];
        let b: Vec<&[u8]> = vec![&[1, 0], &[2, 6], &[3, 0], &[5, 0]];
        let u = unmatched(&a, &b);
        assert_eq!(u[0], vec![&[1u8, 0][..], &[2, 5], &[4, 1]]);
        assert_eq!(u[1], vec![&[2u8, 6][..], &[5, 0]]);
        let (pairs, only) = pair_by_key(u, 1);
        assert_eq!(pairs, vec![(&[2u8, 5][..], &[2u8, 6][..])]);
        assert_eq!(only, [vec![&[1u8, 0][..], &[4, 1]], vec![&[5u8, 0][..]]]);
    }

    #[test]
    fn position_bytes_sort_numerically() {
        let enc = |v: i64| ((v as u64) ^ (1 << 63)).to_be_bytes();
        let vals = [i64::MIN, -300, -1, 0, 1, 255, 256, 1 << 40];
        for w in vals.windows(2) {
            assert!(enc(w[0]) < enc(w[1]), "{} vs {}", w[0], w[1]);
        }
    }
}
