//! `c2c`: nearest-neighbour distance from every point of the compared file A to
//! the reference file B, written out as A plus a scaled-i32 field.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use copc_streaming::{Fields, HierarchyEntry, VoxelKey};
use kiddo::{ImmutableKdTree, SquaredEuclidean};
use rayon::prelude::*;

use crate::extra_bytes::ScaledField;
use crate::input::CopcFile;
use crate::pipeline::ordered_parallel;
use crate::tiles::{Cut, contains, grow};
use crate::write::{CopcWriter, RecordLayout};

pub const FIELD_NAME: &str = "c2c_dist";
pub const FIELD_DESCRIPTION: &str = "C2C dist m; >max = 5*max";

pub struct Options {
    /// Neighbours farther than this count as "none"; also the tile buffer.
    pub max_dist: f64,
    /// Field scale; default the finer of A's and B's coordinate scales.
    pub dist_scale: Option<f64>,
    /// Target compared + reference points per tile.
    pub tile_points: u64,
    /// Tiles processed at once (bounds memory).
    pub tiles_in_flight: usize,
    /// Run despite differing CRS.
    pub force: bool,
}

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub a_points: u64,
    pub shared_points: u64,
    /// Points with no reference neighbour within `max_dist`.
    pub beyond_cap: u64,
    /// Owned points farther than eps outside their tile box (corrupt input).
    pub outside_tile: u64,
    pub b_decoded: u64,
    pub b_in_trees: u64,
    pub max_b_in_tile: u64,
    pub tiles_run: usize,
    pub min: i32,
    pub max: i32,
    /// Wall time summed over tiles: reference decode, kd-tree build, compared
    /// decode + query + encode.
    pub secs_b_decode: f64,
    pub secs_tree: f64,
    pub secs_a: f64,
    /// Wall time of the slowest tile.
    pub secs_max_tile: f64,
}

impl Stats {
    fn new() -> Self {
        Self { min: i32::MAX, max: i32::MIN, ..Default::default() }
    }

    fn merge(&mut self, o: &Stats) {
        self.a_points += o.a_points;
        self.shared_points += o.shared_points;
        self.beyond_cap += o.beyond_cap;
        self.outside_tile += o.outside_tile;
        self.b_decoded += o.b_decoded;
        self.b_in_trees += o.b_in_trees;
        self.max_b_in_tile = self.max_b_in_tile.max(o.max_b_in_tile);
        self.tiles_run += o.tiles_run;
        self.min = self.min.min(o.min);
        self.max = self.max.max(o.max);
        self.secs_b_decode += o.secs_b_decode;
        self.secs_tree += o.secs_tree;
        self.secs_a += o.secs_a;
        self.secs_max_tile = self.secs_max_tile.max(o.secs_max_tile);
    }

    #[inline]
    fn add(&mut self, v: i32, beyond: bool) {
        self.a_points += 1;
        self.beyond_cap += u64::from(beyond);
        self.min = self.min.min(v);
        self.max = self.max.max(v);
    }
}

#[derive(Debug)]
pub struct Summary {
    pub stats: Stats,
    pub tiles: usize,
    pub scale: f64,
    pub b_points: u64,
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
    shared_values: Vec<i32>,
    stats: Stats,
}

/// Everything a tile needs, shared read-only across workers.
struct Ctx<'a> {
    a: &'a CopcFile,
    b: &'a CopcFile,
    cut: &'a Cut,
    shared_by_tile: &'a [Vec<SharedPoint>],
    layout: &'a RecordLayout,
    field: &'a ScaledField,
    max_dist: f64,
    cap: i32,
    eps: f64,
}

/// Nearest-neighbour lookup for one tile: kd-tree in f32 relative to the tile
/// centre, distance recomputed in f64.
struct Neighbours {
    tree: Option<ImmutableKdTree<f32, 3>>,
    points: Vec<[f64; 3]>,
    center: [f64; 3],
}

impl Neighbours {
    fn new(mut points: Vec<[f64; 3]>, center: [f64; 3]) -> Result<Self> {
        // Exact duplicates can't change a nearest distance, but they bloat the
        // kd-tree leaves (unsplittable). Dense mobile-mapping data can be half
        // duplicates at 1 mm; dropping them made queries ~35-45 % faster.
        points.par_sort_unstable_by_key(|p| p.map(f64::to_bits));
        points.dedup_by_key(|p| p.map(f64::to_bits));
        let rel: Vec<[f32; 3]> =
            points.iter().map(|p| std::array::from_fn(|a| (p[a] - center[a]) as f32)).collect();
        let tree = if rel.is_empty() {
            None
        } else {
            Some(ImmutableKdTree::new_from_slice_parallel(&rel).map_err(|e| anyhow!("kd-tree: {e:?}"))?)
        };
        Ok(Self { tree, points, center })
    }

    /// Distance to the nearest point, or `None` if there are none.
    #[inline]
    fn nearest(&self, p: [f64; 3]) -> Option<f64> {
        let tree = self.tree.as_ref()?;
        let q: [f32; 3] = std::array::from_fn(|a| (p[a] - self.center[a]) as f32);
        let nn = tree.query(&q).nearest_one::<SquaredEuclidean<f32>>().execute();
        let b = self.points[nn.item as usize];
        Some(((p[0] - b[0]).powi(2) + (p[1] - b[1]).powi(2) + (p[2] - b[2]).powi(2)).sqrt())
    }
}

impl Ctx<'_> {
    #[inline]
    fn value(&self, nb: &Neighbours, p: [f64; 3]) -> (i32, bool) {
        match nb.nearest(p) {
            Some(d) if d <= self.max_dist => (self.field.encode(d), false),
            _ => (self.cap, true),
        }
    }

    fn tile(&self, t: usize) -> Result<TileOut> {
        let owned = &self.cut.owned[t];
        let shared = &self.shared_by_tile[t];
        let mut stats = Stats::new();
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
        let b_points: Vec<Vec<[f64; 3]>> = b_nodes
            .par_iter()
            .map(|e| {
                let c = self.b.decode(e, Fields::Z)?;
                Ok(c.positions().expect("decoded with Z").filter(|&p| contains(&reach, p)).collect())
            })
            .collect::<Result<_>>()?;
        let t1 = Instant::now();
        let nb = Neighbours::new(b_points.concat(), center)?;
        let t2 = Instant::now();
        stats.b_in_trees = nb.points.len() as u64;
        stats.max_b_in_tile = stats.b_in_trees;

        let inside = grow(&tb, self.eps);
        let encoded: Vec<(NodeChunk, Stats)> = owned
            .par_iter()
            .map(|e| {
                let c = self.a.decode(e, Fields::ALL)?;
                let mut s = Stats::new();
                let mut vals = Vec::with_capacity(c.point_count());
                for p in c.positions().expect("decoded with ALL") {
                    let (v, beyond) = self.value(&nb, p);
                    s.add(v, beyond);
                    s.outside_tile += u64::from(!contains(&inside, p));
                    vals.push(v);
                }
                let bytes = self.layout.encode(c.cloud().raw_bytes(), &vals)?;
                Ok(((e.key, e.point_count, bytes), s))
            })
            .collect::<Result<_>>()?;
        let mut chunks = Vec::with_capacity(encoded.len());
        for (c, s) in encoded {
            stats.merge(&s);
            chunks.push(c);
        }

        let shared_values: Vec<(i32, bool)> = shared.par_iter().map(|sp| self.value(&nb, sp.pos)).collect();
        for &(v, beyond) in &shared_values {
            stats.add(v, beyond);
            stats.shared_points += 1;
        }
        let t3 = Instant::now();
        stats.secs_b_decode = (t1 - t0).as_secs_f64();
        stats.secs_tree = (t2 - t1).as_secs_f64();
        stats.secs_a = (t3 - t2).as_secs_f64();
        stats.secs_max_tile = (t3 - t0).as_secs_f64();
        Ok(TileOut { chunks, shared_values: shared_values.into_iter().map(|(v, _)| v).collect(), stats })
    }
}

pub fn run(a: &CopcFile, b: &CopcFile, out: &Path, opt: &Options) -> Result<Summary> {
    let t0 = Instant::now();
    ensure!(opt.max_dist.is_finite() && opt.max_dist > 0.0, "--max-dist must be positive");
    check_crs(a, b, opt.force)?;

    let scales = |f: &CopcFile| {
        let h = &f.raw_header;
        [h.x_scale_factor, h.y_scale_factor, h.z_scale_factor]
    };
    let all_scales: Vec<f64> = scales(a).into_iter().chain(scales(b)).collect();
    let scale = opt.dist_scale.unwrap_or_else(|| all_scales.iter().copied().fold(f64::INFINITY, f64::min));
    let field = ScaledField::new(FIELD_NAME, FIELD_DESCRIPTION, scale)?;
    field.check_range(5.0 * opt.max_dist)?;
    let cap = field.encode(5.0 * opt.max_dist);
    // A few coordinate scale units: points may sit just outside their node box.
    let eps = 4.0 * all_scales.iter().copied().fold(0.0, f64::max);

    let cut = Cut::new(a, Some(b), opt.tile_points, 4.0 * opt.max_dist);

    // Shared nodes: decode once, assign each point to the tile that contains it.
    let shared_chunks = cut
        .shared
        .par_iter()
        .map(|e| a.decode(e, Fields::ALL))
        .collect::<Result<Vec<_>>>()?;
    let mut shared_by_tile: Vec<Vec<SharedPoint>> = (0..cut.tiles.len()).map(|_| Vec::new()).collect();
    for (node, c) in shared_chunks.iter().enumerate() {
        for (index, pos) in c.positions().expect("decoded with ALL").enumerate() {
            shared_by_tile[cut.locate(pos)].push(SharedPoint { node: node as u32, index: index as u32, pos });
        }
    }
    let mut shared_values: Vec<Vec<i32>> = shared_chunks.iter().map(|c| vec![cap; c.point_count()]).collect();
    let t_setup = t0.elapsed();

    let mut writer = CopcWriter::create(out, a, std::slice::from_ref(&field))?;
    let layout = writer.layout().clone();
    let ctx = Ctx {
        a,
        b,
        cut: &cut,
        shared_by_tile: &shared_by_tile,
        layout: &layout,
        field: &field,
        max_dist: opt.max_dist,
        cap,
        eps,
    };
    let mut stats = Stats::new();
    let tile_ids: Vec<usize> = (0..cut.tiles.len()).collect();
    ordered_parallel(
        &tile_ids,
        opt.tiles_in_flight,
        |_, &t| ctx.tile(t).with_context(|| format!("tile {:?}", cut.tiles[t])),
        |t, out| {
            for (key, n, bytes) in &out.chunks {
                writer.write_chunk(*key, *n, bytes)?;
            }
            for (sp, v) in shared_by_tile[t].iter().zip(&out.shared_values) {
                shared_values[sp.node as usize][sp.index as usize] = *v;
            }
            stats.merge(&out.stats);
            Ok(())
        },
    )?;
    let t_tiles = t0.elapsed();

    let shared_encoded: Vec<Vec<u8>> = shared_chunks
        .par_iter()
        .zip(&shared_values)
        .map(|(c, v)| layout.encode(c.cloud().raw_bytes(), v))
        .collect::<Result<_>>()?;
    for (e, bytes) in cut.shared.iter().zip(&shared_encoded) {
        writer.write_chunk(e.key, e.point_count, bytes)?;
    }
    let written = writer.finish(a, Some(&[(stats.min, stats.max)]))?;
    let t_end = t0.elapsed();

    ensure!(
        stats.a_points == a.las_header().number_of_points(),
        "computed {} distances for {} points",
        stats.a_points,
        a.las_header().number_of_points()
    );
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
    eprintln!(
        "{} -> {}\n  \
         field {FIELD_NAME}: i32 × {scale} m; beyond {} m stored as {} m\n  \
         tiles: {} ({} with points, levels {}–{}), {} shared nodes ({} points)\n  \
         reference: {} points, {} decoded (re-decode factor {:.2}), {} in kd-trees, max {} per tile\n  \
         beyond max-dist: {} ({:.2} %)\n  \
         time: setup {:.2} s, tiles {:.2} s, shared + finish {:.2} s, total {secs:.2} s = {:.1} M pts/s\n  \
         tile time (summed): reference decode {:.1} s, kd-tree {:.1} s, compared {:.1} s; slowest tile {:.1} s\n  \
         output {:.1} MB ({:+.3} B/pt vs compared file)",
        a.path.display(),
        out.display(),
        opt.max_dist,
        5.0 * opt.max_dist,
        cut.tiles.len(),
        stats.tiles_run,
        cut.tiles.iter().map(|k| k.level).min().unwrap_or(0),
        cut.tiles.iter().map(|k| k.level).max().unwrap_or(0),
        cut.shared.len(),
        stats.shared_points,
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
    Ok(Summary { stats, tiles: cut.tiles.len(), scale, b_points })
}
