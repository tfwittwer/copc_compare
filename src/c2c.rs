//! `c2c`: distance from every point of the compared file A to the reference file
//! B, written out as A plus scaled-i32 fields:
//!
//! - `c2c_dist`: nearest-neighbour distance (always).
//! - `c2c_dx`, `c2c_dy`, `c2c_dz` (`--components`): A − nearest B point, so
//!   positive dz means A lies above B.
//! - `c2c_plane` (`--model plane`): signed distance to the least-squares plane
//!   through the k nearest B points within max-dist, normal oriented to +z.
//!
//! Points without a B point within max-dist get 5 × max-dist in every field.

use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use copc_streaming::{Fields, HierarchyEntry, VoxelKey};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use rayon::prelude::*;

use crate::extra_bytes::ScaledField;
use crate::input::CopcFile;
use crate::pipeline::ordered_parallel;
use crate::tiles::{Cut, contains, grow};
use crate::write::{CopcWriter, RecordLayout};

pub const FIELD_NAME: &str = "c2c_dist";
pub const FIELD_DESCRIPTION: &str = "C2C dist m; >max = 5*max";

/// Planes whose middle eigenvalue is below this fraction of the largest are
/// treated as degenerate (neighbours nearly collinear): fall back to NN.
const DEGENERATE: f64 = 1e-3;

/// Peak bytes per compared + reference point of a tile in flight: reference
/// positions (f64, 24 B), kd-tree (f32 + index, ~16 B), compressed output and
/// per-node buffers. Measured ~56 B/pt on the real pair (RSS minus mapped files).
pub const BYTES_PER_TILE_POINT: u64 = 60;
pub const DEFAULT_TILE_POINTS: u64 = 8_000_000;

pub struct Options {
    /// Neighbours farther than this count as "none"; also the tile buffer.
    pub max_dist: f64,
    /// Field scale; default the finer of A's and B's coordinate scales.
    pub dist_scale: Option<f64>,
    /// Also write the NN offset components.
    pub components: bool,
    /// Also write the plane-model distance, fitted to this many neighbours.
    pub plane_k: Option<usize>,
    /// Target compared + reference points per tile.
    pub tile_points: u64,
    /// Tiles processed at once (bounds memory).
    pub tiles_in_flight: usize,
    /// Run despite differing CRS.
    pub force: bool,
    /// Draw a progress bar (only shown on a terminal).
    pub progress: bool,
}

impl Options {
    /// Options with the default tiling for `threads` workers.
    pub fn new(max_dist: f64) -> Self {
        Self {
            max_dist,
            dist_scale: None,
            components: false,
            plane_k: None,
            tile_points: DEFAULT_TILE_POINTS,
            tiles_in_flight: 8,
            force: false,
            progress: false,
        }
    }
}

/// Tile size and tiles in flight for a memory budget: as many default-size
/// tiles as fit (at most one per thread); smaller tiles only if not even one fits.
/// Mapped input files don't count: they live in the page cache.
pub fn tiling_for_memory(limit_bytes: u64, threads: usize) -> (u64, usize) {
    let usable = limit_bytes / 4 * 3;
    let per_tile = DEFAULT_TILE_POINTS * BYTES_PER_TILE_POINT;
    let in_flight = ((usable / per_tile) as usize).clamp(1, threads.max(1));
    let tile_points = (usable / (in_flight as u64 * BYTES_PER_TILE_POINT)).clamp(100_000, DEFAULT_TILE_POINTS);
    (tile_points, in_flight)
}

/// Half the physical memory, or 8 GiB if that can't be read.
pub fn default_memory_limit() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            let kb: u64 = m.lines().find(|l| l.starts_with("MemTotal:"))?.split_whitespace().nth(1)?.parse().ok()?;
            Some(kb * 1024 / 2)
        })
        .unwrap_or(8 << 30)
}

/// Output field layout.
struct Layout {
    components: Option<usize>,
    plane: Option<usize>,
    n: usize,
}

impl Layout {
    fn new(opt: &Options) -> Self {
        let mut n = 1;
        let components = opt.components.then(|| {
            n += 3;
            1
        });
        let plane = opt.plane_k.map(|_| {
            n += 1;
            n - 1
        });
        Self { components, plane, n }
    }

    fn fields(&self, scale: f64) -> Result<Vec<ScaledField>> {
        let mut f = vec![ScaledField::new(FIELD_NAME, FIELD_DESCRIPTION, scale)?];
        if self.components.is_some() {
            for axis in ["x", "y", "z"] {
                f.push(ScaledField::new(&format!("c2c_d{axis}"), &format!("NN d{axis} m, A-B; >max = 5*max"), scale)?);
            }
        }
        if self.plane.is_some() {
            f.push(ScaledField::new("c2c_plane", "Plane dist m,+z; >max = 5*max", scale)?);
        }
        Ok(f)
    }
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub a_points: u64,
    pub shared_points: u64,
    /// Points with no reference neighbour within `max_dist`.
    pub beyond_cap: u64,
    /// Plane model: points that fell back to the NN distance.
    pub plane_fallbacks: u64,
    /// Owned points farther than eps outside their tile box (corrupt input).
    pub outside_tile: u64,
    pub b_decoded: u64,
    pub b_in_trees: u64,
    pub max_b_in_tile: u64,
    pub tiles_run: usize,
    /// (min, max) raw value per output field.
    pub ranges: Vec<(i32, i32)>,
    /// Wall time summed over tiles: reference decode, kd-tree build, compared
    /// decode + query + encode.
    pub secs_b_decode: f64,
    pub secs_tree: f64,
    pub secs_a: f64,
    /// Wall time of the slowest tile.
    pub secs_max_tile: f64,
}

impl Stats {
    fn new(n_fields: usize) -> Self {
        Self { ranges: vec![(i32::MAX, i32::MIN); n_fields], ..Default::default() }
    }

    fn merge(&mut self, o: &Stats) {
        self.a_points += o.a_points;
        self.shared_points += o.shared_points;
        self.beyond_cap += o.beyond_cap;
        self.plane_fallbacks += o.plane_fallbacks;
        self.outside_tile += o.outside_tile;
        self.b_decoded += o.b_decoded;
        self.b_in_trees += o.b_in_trees;
        self.max_b_in_tile = self.max_b_in_tile.max(o.max_b_in_tile);
        self.tiles_run += o.tiles_run;
        for (r, o) in self.ranges.iter_mut().zip(&o.ranges) {
            *r = (r.0.min(o.0), r.1.max(o.1));
        }
        self.secs_b_decode += o.secs_b_decode;
        self.secs_tree += o.secs_tree;
        self.secs_a += o.secs_a;
        self.secs_max_tile = self.secs_max_tile.max(o.secs_max_tile);
    }

    #[inline]
    fn add(&mut self, vals: &[i32], m: Measured) {
        self.a_points += 1;
        self.beyond_cap += u64::from(m.beyond);
        self.plane_fallbacks += u64::from(m.plane_fallback);
        for (r, &v) in self.ranges.iter_mut().zip(vals) {
            *r = (r.0.min(v), r.1.max(v));
        }
    }
}

#[derive(Debug)]
pub struct Summary {
    pub stats: Stats,
    pub tiles: usize,
    pub scale: f64,
    pub b_points: u64,
    pub fields: Vec<String>,
}

/// The WKT CRS (VLR or EVLR LASF_Projection/2112), trailing NULs and whitespace trimmed.
fn wkt(f: &CopcFile) -> Option<&[u8]> {
    f.vlrs.iter().chain(&f.evlrs).find(|v| v.is("LASF_Projection", 2112)).map(|v| {
        let d = &v.data[..];
        let end = d.iter().rposition(|&c| c != 0 && !c.is_ascii_whitespace()).map_or(0, |i| i + 1);
        &d[..end]
    })
}

fn check_crs(a: &CopcFile, b: &CopcFile, force: bool) -> Result<()> {
    match (wkt(a), wkt(b)) {
        (Some(wa), Some(wb)) if wa != wb => {
            let msg = format!(
                "CRS differs between {} and {}:\n  {}\n  {}",
                a.path.display(),
                b.path.display(),
                String::from_utf8_lossy(&wa[..wa.len().min(120)]),
                String::from_utf8_lossy(&wb[..wb.len().min(120)])
            );
            if force {
                eprintln!("warning: {msg}\n(continuing because of --force)");
            } else {
                bail!("{msg}\nUse --force to run anyway.");
            }
        }
        (Some(_), None) | (None, Some(_)) => {
            eprintln!("warning: only one of the two files declares a CRS; assuming they match");
        }
        _ => {}
    }
    Ok(())
}

struct SharedPoint {
    node: u32,
    index: u32,
    pos: [f64; 3],
}

/// One encoded node: key, point count, compressed chunk.
type NodeChunk = (VoxelKey, u32, Vec<u8>);

struct TileOut {
    chunks: Vec<NodeChunk>,
    /// `n_fields` values per shared point of the tile.
    shared_values: Vec<i32>,
    stats: Stats,
}

#[derive(Clone, Copy, Default)]
struct Measured {
    beyond: bool,
    plane_fallback: bool,
}

/// Everything a tile needs, shared read-only across workers.
struct Ctx<'a> {
    a: &'a CopcFile,
    b: &'a CopcFile,
    cut: &'a Cut,
    shared_by_tile: &'a [Vec<SharedPoint>],
    record: &'a RecordLayout,
    fields: &'a [ScaledField],
    layout: Layout,
    plane_k: Option<NonZeroUsize>,
    max_dist: f64,
    cap: i32,
    eps: f64,
}

/// Neighbour lookup for one tile: kd-tree in f32 relative to the tile centre,
/// distances recomputed in f64 from the original coordinates.
struct Neighbours {
    tree: Option<ImmutableKdTree<f32, 3>>,
    points: Vec<[f64; 3]>,
    center: [f64; 3],
}

#[inline]
fn dist(p: [f64; 3], q: [f64; 3]) -> f64 {
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
}

impl Neighbours {
    fn new(mut points: Vec<[f64; 3]>, center: [f64; 3]) -> Result<Self> {
        // Exact duplicates can't change a nearest distance (or a plane fit to the
        // distinct points), but they bloat the kd-tree leaves (unsplittable).
        // Dense mobile-mapping data can be half duplicates at 1 mm; dropping
        // them made queries ~35-45 % faster.
        points.par_sort_unstable_by_key(|p| p.map(f64::to_bits));
        points.dedup_by_key(|p| p.map(f64::to_bits));
        let rel: Vec<[f32; 3]> = points.iter().map(|p| std::array::from_fn(|a| (p[a] - center[a]) as f32)).collect();
        let tree = if rel.is_empty() {
            None
        } else {
            Some(ImmutableKdTree::new_from_slice_parallel(&rel).map_err(|e| anyhow!("kd-tree: {e:?}"))?)
        };
        Ok(Self { tree, points, center })
    }

    #[inline]
    fn query_point(&self, p: [f64; 3]) -> [f32; 3] {
        std::array::from_fn(|a| (p[a] - self.center[a]) as f32)
    }

    /// The nearest point: (distance, index).
    #[inline]
    fn nearest(&self, p: [f64; 3]) -> Option<(f64, usize)> {
        let tree = self.tree.as_ref()?;
        let nn = tree.query(&self.query_point(p)).nearest_one::<SquaredEuclidean<f32>>().execute();
        let i = nn.item as usize;
        Some((dist(p, self.points[i]), i))
    }

    /// Up to `k` nearest points within `max_dist`, nearest first: (distance, index).
    fn within(&self, p: [f64; 3], k: NonZeroUsize, max_dist: f64) -> Vec<(f64, usize)> {
        let Some(tree) = self.tree.as_ref() else { return Vec::new() };
        // A hair wider in f32, then exact in f64: every B point within max_dist
        // is in this tile's tree, so the result doesn't depend on the tiling.
        let r = (max_dist * (1.0 + 1e-4) + 1e-6) as f32;
        let mut out: Vec<(f64, usize)> = tree
            .query(&self.query_point(p))
            .nearest_n::<SquaredEuclidean<f32>>(k)
            .within(r * r)
            .execute()
            .into_iter()
            .map(|n| {
                let i = n.item as usize;
                (dist(p, self.points[i]), i)
            })
            .filter(|&(d, _)| d <= max_dist)
            .collect();
        out.sort_unstable_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)));
        out
    }

    /// Signed distance from `p` to the least-squares plane through `nbrs`, with
    /// the normal oriented to +z. `None` for fewer than 3 points or a degenerate
    /// (nearly collinear) set. Computed relative to `p` for precision.
    fn plane_distance(&self, p: [f64; 3], nbrs: &[(f64, usize)]) -> Option<f64> {
        if nbrs.len() < 3 {
            return None;
        }
        let rel = |i: usize| Vector3::from(self.points[i]) - Vector3::from(p);
        let n = nbrs.len() as f64;
        let c = nbrs.iter().map(|&(_, i)| rel(i)).sum::<Vector3<f64>>() / n;
        let mut cov = Matrix3::zeros();
        for &(_, i) in nbrs {
            let d = rel(i) - c;
            cov += d * d.transpose();
        }
        let eig = SymmetricEigen::new(cov / n);
        let mut order = [0, 1, 2];
        order.sort_by(|&x, &y| eig.eigenvalues[x].total_cmp(&eig.eigenvalues[y]));
        let (mid, max) = (eig.eigenvalues[order[1]], eig.eigenvalues[order[2]]);
        if max.is_nan() || max <= 0.0 || mid <= DEGENERATE * max {
            return None;
        }
        let mut normal = eig.eigenvectors.column(order[0]).into_owned();
        if normal.z < 0.0 {
            normal = -normal;
        }
        // p is the origin here: its signed distance from the plane through c.
        Some(-c.dot(&normal))
    }
}

impl Ctx<'_> {
    /// Fill `out` (one value per field) for point `p`.
    #[inline]
    fn measure(&self, nb: &Neighbours, p: [f64; 3], out: &mut [i32]) -> Measured {
        let f = &self.fields[0];
        let (nn, plane) = match self.plane_k {
            Some(k) => {
                let nbrs = self.within(nb, p, k);
                (nbrs.first().copied(), Some(nbrs))
            }
            None => (nb.nearest(p).filter(|&(d, _)| d <= self.max_dist), None),
        };
        let Some((d, i)) = nn else {
            out.fill(self.cap);
            return Measured { beyond: true, plane_fallback: false };
        };
        out[0] = f.encode(d);
        if let Some(c) = self.layout.components {
            let q = nb.points[i];
            for a in 0..3 {
                out[c + a] = f.encode(p[a] - q[a]);
            }
        }
        let mut m = Measured::default();
        if let (Some(slot), Some(nbrs)) = (self.layout.plane, plane) {
            out[slot] = match nb.plane_distance(p, &nbrs) {
                Some(dp) => f.encode(dp),
                None => {
                    m.plane_fallback = true;
                    out[0]
                }
            };
        }
        m
    }

    fn within(&self, nb: &Neighbours, p: [f64; 3], k: NonZeroUsize) -> Vec<(f64, usize)> {
        nb.within(p, k, self.max_dist)
    }

    fn tile(&self, t: usize) -> Result<TileOut> {
        let owned = &self.cut.owned[t];
        let shared = &self.shared_by_tile[t];
        let nf = self.layout.n;
        let mut stats = Stats::new(nf);
        if owned.is_empty() && shared.is_empty() {
            return Ok(TileOut { chunks: Vec::new(), shared_values: Vec::new(), stats });
        }
        stats.tiles_run = 1;
        let t0 = Instant::now();

        let tb = self.cut.tile_bounds(t);
        let center: [f64; 3] = std::array::from_fn(|a| 0.5 * (tb.min[a] + tb.max[a]));
        // Every B point within max_dist of a tile point (which may sit eps outside
        // the box) lies in `reach`; B points may sit eps outside their node box.
        let reach = grow(&tb, self.max_dist + self.eps);
        let node_reach = grow(&reach, self.eps);
        let b_root = self.b.root_bounds();
        let b_nodes: Vec<&HierarchyEntry> =
            self.b.data_entries().filter(|e| e.key.bounds(&b_root).intersects(&node_reach)).collect();
        stats.b_decoded = b_nodes.iter().map(|e| u64::from(e.point_count)).sum();
        let per_node: Vec<Vec<[f64; 3]>> = b_nodes
            .par_iter()
            .map(|e| {
                let c = self.b.decode(e, Fields::Z)?;
                Ok(c.positions().expect("decoded with Z").filter(|&p| contains(&reach, p)).collect())
            })
            .collect::<Result<_>>()?;
        // Move into one Vec, freeing each node's as we go (not `concat`, which
        // holds both at once).
        let mut b_points = Vec::with_capacity(per_node.iter().map(Vec::len).sum());
        for v in per_node {
            b_points.extend(v);
        }
        let t1 = Instant::now();
        let nb = Neighbours::new(b_points, center)?;
        let t2 = Instant::now();
        stats.b_in_trees = nb.points.len() as u64;
        stats.max_b_in_tile = stats.b_in_trees;

        let inside = grow(&tb, self.eps);
        let encoded: Vec<(NodeChunk, Stats)> = owned
            .par_iter()
            .map(|e| {
                let c = self.a.decode(e, Fields::ALL)?;
                let mut s = Stats::new(nf);
                let mut vals = vec![0; c.point_count() * nf];
                for (p, out) in c.positions().expect("decoded with ALL").zip(vals.chunks_exact_mut(nf)) {
                    let m = self.measure(&nb, p, out);
                    s.add(out, m);
                    s.outside_tile += u64::from(!contains(&inside, p));
                }
                let bytes = self.record.encode(c.cloud().raw_bytes(), &vals)?;
                Ok(((e.key, e.point_count, bytes), s))
            })
            .collect::<Result<_>>()?;
        let mut chunks = Vec::with_capacity(encoded.len());
        for (c, s) in encoded {
            stats.merge(&s);
            chunks.push(c);
        }

        let mut shared_values = vec![0; shared.len() * nf];
        let measured: Vec<Measured> = shared
            .par_iter()
            .zip(shared_values.par_chunks_exact_mut(nf))
            .map(|(sp, out)| self.measure(&nb, sp.pos, out))
            .collect();
        for (m, out) in measured.into_iter().zip(shared_values.chunks_exact(nf)) {
            stats.add(out, m);
            stats.shared_points += 1;
        }
        let t3 = Instant::now();
        stats.secs_b_decode = (t1 - t0).as_secs_f64();
        stats.secs_tree = (t2 - t1).as_secs_f64();
        stats.secs_a = (t3 - t2).as_secs_f64();
        stats.secs_max_tile = (t3 - t0).as_secs_f64();
        Ok(TileOut { chunks, shared_values, stats })
    }
}

pub fn run(a: &CopcFile, b: &CopcFile, out: &Path, opt: &Options) -> Result<Summary> {
    let t0 = Instant::now();
    ensure!(opt.max_dist.is_finite() && opt.max_dist > 0.0, "--max-dist must be positive");
    if let Some(k) = opt.plane_k {
        ensure!(k >= 3, "--k must be at least 3 to fit a plane");
    }
    check_crs(a, b, opt.force)?;

    let scales = |f: &CopcFile| {
        let h = &f.raw_header;
        [h.x_scale_factor, h.y_scale_factor, h.z_scale_factor]
    };
    let all_scales: Vec<f64> = scales(a).into_iter().chain(scales(b)).collect();
    let scale = opt.dist_scale.unwrap_or_else(|| all_scales.iter().copied().fold(f64::INFINITY, f64::min));
    let layout = Layout::new(opt);
    let fields = layout.fields(scale)?;
    let nf = layout.n;
    // Components and plane distances lie within ±max_dist, the cap is 5 × max_dist.
    fields[0].check_range(5.0 * opt.max_dist)?;
    let cap = fields[0].encode(5.0 * opt.max_dist);
    // A few coordinate scale units: points may sit just outside their node box.
    let eps = 4.0 * all_scales.iter().copied().fold(0.0, f64::max);

    let cut = Cut::new(a, Some(b), opt.tile_points, 4.0 * opt.max_dist);

    // Shared nodes: decode once, assign each point to the tile that contains it.
    let shared_chunks = cut.shared.par_iter().map(|e| a.decode(e, Fields::ALL)).collect::<Result<Vec<_>>>()?;
    let mut shared_by_tile: Vec<Vec<SharedPoint>> = (0..cut.tiles.len()).map(|_| Vec::new()).collect();
    for (node, c) in shared_chunks.iter().enumerate() {
        for (index, pos) in c.positions().expect("decoded with ALL").enumerate() {
            shared_by_tile[cut.locate(pos)].push(SharedPoint { node: node as u32, index: index as u32, pos });
        }
    }
    let mut shared_values: Vec<Vec<i32>> = shared_chunks.iter().map(|c| vec![cap; c.point_count() * nf]).collect();
    let t_setup = t0.elapsed();

    let mut writer = CopcWriter::create(out, a, &fields)?;
    let record = writer.layout().clone();
    let ctx = Ctx {
        a,
        b,
        cut: &cut,
        shared_by_tile: &shared_by_tile,
        record: &record,
        fields: &fields,
        layout,
        plane_k: opt.plane_k.and_then(NonZeroUsize::new),
        max_dist: opt.max_dist,
        cap,
        eps,
    };

    let n_points = a.las_header().number_of_points();
    let bar = if opt.progress {
        let bar = ProgressBar::with_draw_target(Some(n_points), ProgressDrawTarget::stderr());
        bar.set_style(
            ProgressStyle::with_template("{elapsed_precise} [{wide_bar}] {percent:>3}% {human_pos}/{human_len} points, ETA {eta}")
                .expect("valid template")
                .progress_chars("=> "),
        );
        bar
    } else {
        ProgressBar::hidden()
    };

    let mut stats = Stats::new(nf);
    let tile_ids: Vec<usize> = (0..cut.tiles.len()).collect();
    ordered_parallel(
        &tile_ids,
        opt.tiles_in_flight,
        |_, &t| ctx.tile(t).with_context(|| format!("tile {:?}", cut.tiles[t])),
        |t, out| {
            for (key, n, bytes) in &out.chunks {
                writer.write_chunk(*key, *n, bytes)?;
            }
            for (sp, v) in shared_by_tile[t].iter().zip(out.shared_values.chunks_exact(nf)) {
                let at = sp.index as usize * nf;
                shared_values[sp.node as usize][at..at + nf].copy_from_slice(v);
            }
            stats.merge(&out.stats);
            bar.inc(out.stats.a_points);
            Ok(())
        },
    )?;
    bar.finish_and_clear();
    let t_tiles = t0.elapsed();

    let shared_encoded: Vec<Vec<u8>> = shared_chunks
        .par_iter()
        .zip(&shared_values)
        .map(|(c, v)| record.encode(c.cloud().raw_bytes(), v))
        .collect::<Result<_>>()?;
    for (e, bytes) in cut.shared.iter().zip(&shared_encoded) {
        writer.write_chunk(e.key, e.point_count, bytes)?;
    }
    let written = writer.finish(a, Some(&stats.ranges))?;
    let t_end = t0.elapsed();

    ensure!(stats.a_points == n_points, "computed {} distances for {n_points} points", stats.a_points);
    if stats.outside_tile > 0 {
        eprintln!(
            "warning: {} points of {} lie outside their octree node; their distances may miss neighbours",
            stats.outside_tile,
            a.path.display()
        );
    }

    let b_points = b.las_header().number_of_points();
    let n = stats.a_points as f64;
    let secs = t_end.as_secs_f64();
    let names: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
    let plane_note = match opt.plane_k {
        Some(k) => format!(
            "\n  plane model: k = {k}, {} fallbacks to NN ({:.2} %)",
            stats.plane_fallbacks,
            100.0 * stats.plane_fallbacks as f64 / n.max(1.0)
        ),
        None => String::new(),
    };
    eprintln!(
        "{} -> {}\n  \
         fields {}: i32 × {scale} m; beyond {} m stored as {} m{plane_note}\n  \
         tiles: {} ({} with points, levels {}–{}), {} shared nodes ({} points); {} M points/tile, {} in flight\n  \
         reference: {} points, {} decoded (re-decode factor {:.2}), {} in kd-trees, max {} per tile\n  \
         beyond max-dist: {} ({:.2} %)\n  \
         time: setup {:.2} s, tiles {:.2} s, shared + finish {:.2} s, total {secs:.2} s = {:.1} M pts/s\n  \
         tile time (summed): reference decode {:.1} s, kd-tree {:.1} s, compared {:.1} s; slowest tile {:.1} s\n  \
         output {:.1} MB ({:+.3} B/pt vs compared file)",
        a.path.display(),
        out.display(),
        names.join(", "),
        opt.max_dist,
        5.0 * opt.max_dist,
        cut.tiles.len(),
        stats.tiles_run,
        cut.tiles.iter().map(|k| k.level).min().unwrap_or(0),
        cut.tiles.iter().map(|k| k.level).max().unwrap_or(0),
        cut.shared.len(),
        stats.shared_points,
        opt.tile_points as f64 / 1e6,
        opt.tiles_in_flight,
        b_points,
        stats.b_decoded,
        stats.b_decoded as f64 / b_points.max(1) as f64,
        stats.b_in_trees,
        stats.max_b_in_tile,
        stats.beyond_cap,
        100.0 * stats.beyond_cap as f64 / n.max(1.0),
        t_setup.as_secs_f64(),
        (t_tiles - t_setup).as_secs_f64(),
        (t_end - t_tiles).as_secs_f64(),
        n / secs / 1e6,
        stats.secs_b_decode,
        stats.secs_tree,
        stats.secs_a,
        stats.secs_max_tile,
        written.bytes as f64 / 1e6,
        (written.bytes as f64 - a.file_size() as f64) / n.max(1.0),
    );
    Ok(Summary { stats, tiles: cut.tiles.len(), scale, b_points, fields: names })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nb(points: &[[f64; 3]]) -> Neighbours {
        Neighbours::new(points.to_vec(), [0.0; 3]).unwrap()
    }

    #[test]
    fn within_uses_the_radius_exactly() {
        // Distances 0.5, 0.8, 1.2 from the origin.
        let n = nb(&[[0.5, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.2]]);
        let k = NonZeroUsize::new(8).unwrap();
        let d = |r: f64| n.within([0.0; 3], k, r).iter().map(|x| x.0).collect::<Vec<_>>();
        assert_eq!(d(0.4), Vec::<f64>::new());
        assert_eq!(d(0.6), vec![0.5]);
        assert_eq!(d(1.0), vec![0.5, 0.8]);
        assert_eq!(d(2.0), vec![0.5, 0.8, 1.2]);
        assert_eq!(n.within([0.0; 3], NonZeroUsize::new(2).unwrap(), 2.0).len(), 2);
    }

    #[test]
    fn plane_distance_signed_and_oriented_up() {
        // Plane z = 0.5 x tilted 26.6°; query 0.3 above it along the normal.
        let pts: Vec<[f64; 3]> =
            (-3..=3).flat_map(|i| (-3..=3).map(move |j| [i as f64 * 0.1, j as f64 * 0.1, 0.05 * i as f64])).collect();
        let n = nb(&pts);
        let len = (1.0f64 + 0.25).sqrt();
        let normal = [-0.5 / len, 0.0, 1.0 / len];
        for t in [0.3, -0.2] {
            let p = normal.map(|v| v * t);
            let nbrs = n.within(p, NonZeroUsize::new(12).unwrap(), 1.0);
            let d = n.plane_distance(p, &nbrs).unwrap();
            assert!((d - t).abs() < 1e-12, "{d} vs {t}");
        }
    }

    #[test]
    fn plane_degenerate_falls_back() {
        let line = nb(&[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]);
        let nbrs = line.within([0.1, 0.1, 0.0], NonZeroUsize::new(8).unwrap(), 1.0);
        assert_eq!(nbrs.len(), 4);
        assert!(line.plane_distance([0.1, 0.1, 0.0], &nbrs).is_none());
        assert!(line.plane_distance([0.1, 0.1, 0.0], &nbrs[..2]).is_none());
    }

    #[test]
    fn tiling_for_memory_fits() {
        let gb = 1u64 << 30;
        assert_eq!(tiling_for_memory(64 * gb, 32), (DEFAULT_TILE_POINTS, 32));
        assert_eq!(tiling_for_memory(4 * gb, 32), (DEFAULT_TILE_POINTS, 6));
        let (t, n) = tiling_for_memory(gb / 2, 32);
        assert_eq!(n, 1);
        assert!(t < DEFAULT_TILE_POINTS && t * BYTES_PER_TILE_POINT <= gb / 2);
        assert_eq!(tiling_for_memory(64 * gb, 4).1, 4);
    }
}
