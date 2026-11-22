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
use copc_streaming::{Aabb, Fields, HierarchyEntry, VoxelKey};
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

/// Peak bytes per compared + reference point of a running tile: reference
/// positions (f64, 24 B), kd-tree (f64 + index, ~28 B), compressed output and
/// per-node buffers (~56 B/pt measured with an f32 tree, RSS minus mapped
/// files, so ~70 B/pt with f64); plus up to two finished tiles per running one
/// waiting in the reorder buffer with their compressed output (~12 B per
/// compared point for PF7).
pub const BYTES_PER_TILE_POINT: u64 = 96;
pub const DEFAULT_TILE_POINTS: u64 = 8_000_000;
/// Tiles get as large as the memory budget allows, up to this: larger tiles
/// re-decode fewer reference nodes at their borders.
pub const MAX_TILE_POINTS: u64 = 64_000_000;
/// Below this the border re-decodes dominate; run fewer tiles at once instead.
pub const MIN_TILE_POINTS: u64 = 1_000_000;
/// Tiles are also kept small enough that each tile in flight gets this many on
/// average, so the run doesn't end with a few big tiles running alone.
pub const TILES_PER_SLOT: u64 = 2;

/// The default distance scale is the finer coordinate scale of the two files,
/// but not below this: some converters store absurd scales (e.g. 6 nm), which
/// carry no information, compress badly and overflow i32 for larger max-dist.
pub const MIN_DEFAULT_SCALE: f64 = 1e-4;

/// Default distance scale for files with these coordinate scales.
pub fn default_scale(coordinate_scales: &[f64]) -> f64 {
    coordinate_scales.iter().copied().fold(f64::INFINITY, f64::min).max(MIN_DEFAULT_SCALE)
}

#[derive(Clone, Debug)]
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
    /// Re-encode A's chunks even where they could be spliced (for comparisons).
    pub reencode: bool,
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
            reencode: false,
        }
    }
}

/// Tile size and tiles in flight for a memory budget: one tile per thread, as
/// large as the budget allows (at most `MAX_TILE_POINTS`); fewer tiles at once
/// if they would get smaller than `MIN_TILE_POINTS`. Mapped input files don't
/// count: they live in the page cache.
pub fn tiling_for_memory(limit_bytes: u64, threads: usize) -> (u64, usize) {
    let usable = limit_bytes / 4 * 3;
    let per_point = |n: usize| usable / (n as u64 * BYTES_PER_TILE_POINT);
    let mut in_flight = threads.max(1);
    if per_point(in_flight) < MIN_TILE_POINTS {
        in_flight = ((usable / (MIN_TILE_POINTS * BYTES_PER_TILE_POINT)) as usize).clamp(1, in_flight);
    }
    (per_point(in_flight).clamp(100_000, MAX_TILE_POINTS), in_flight)
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
    /// A's chunks spliced rather than re-encoded.
    pub spliced: u64,
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
        self.spliced += o.spliced;
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

fn check_crs(a: &CopcFile, b: &CopcFile, force: bool) -> Result<()> {
    match (a.wkt(), b.wkt()) {
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

/// Neighbour lookup for one tile: kd-tree in f64 relative to the tile centre,
/// distances recomputed in f64 from the original coordinates.
struct Neighbours {
    tree: Option<ImmutableKdTree<f64, 3>>,
    points: Vec<[f64; 3]>,
    center: [f64; 3],
    occupancy: Occupancy,
}

/// Conservative "is any point within max_dist?" test: a bitmap of occupied
/// cells at least max_dist wide over the tile's reach box. If none of the 27
/// cells around a query holds a point, nothing is within max_dist. Without it
/// such queries (often ~5 % of points) search out to their true, useless,
/// nearest neighbour, which in dense data costs ~10 µs each.
struct Occupancy {
    min: [f64; 3],
    cell: f64,
    dims: [i64; 3],
    bits: Vec<u64>,
}

impl Occupancy {
    /// At most this many cells per axis (bitmap ≤ 8 MB); cells grow to fit.
    const MAX_DIM: f64 = 400.0;

    fn new(points: &[[f64; 3]], reach: &Aabb, max_dist: f64) -> Self {
        let size = (0..3).map(|a| reach.max[a] - reach.min[a]).fold(0.0, f64::max);
        let cell = max_dist.max(size / Self::MAX_DIM);
        let dims: [i64; 3] = std::array::from_fn(|a| ((reach.max[a] - reach.min[a]) / cell).floor() as i64 + 1);
        let n = (dims[0] * dims[1] * dims[2]) as usize;
        let mut o = Self { min: reach.min, cell, dims, bits: vec![0; n.div_ceil(64)] };
        for p in points {
            if let Some(i) = o.index(o.cell_of(*p)) {
                o.bits[i / 64] |= 1 << (i % 64);
            }
        }
        o
    }

    #[inline]
    fn cell_of(&self, p: [f64; 3]) -> [i64; 3] {
        std::array::from_fn(|a| ((p[a] - self.min[a]) / self.cell).floor() as i64)
    }

    #[inline]
    fn index(&self, c: [i64; 3]) -> Option<usize> {
        (0..3).all(|a| (0..self.dims[a]).contains(&c[a])).then(|| ((c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]) as usize)
    }

    /// False if certainly no point lies within max_dist of `p`.
    #[inline]
    fn maybe_within(&self, p: [f64; 3]) -> bool {
        let c = self.cell_of(p);
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(i) = self.index([c[0] + dx, c[1] + dy, c[2] + dz])
                        && self.bits[i / 64] & (1 << (i % 64)) != 0
                    {
                        return true;
                    }
                }
            }
        }
        false
    }
}

#[inline]
fn dist(p: [f64; 3], q: [f64; 3]) -> f64 {
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
}

impl Neighbours {
    /// `reach`: a box containing all `points` and every query point grown by
    /// `max_dist`.
    fn new(mut points: Vec<[f64; 3]>, center: [f64; 3], reach: &Aabb, max_dist: f64) -> Result<Self> {
        // Exact duplicates can't change a nearest distance (or a plane fit to the
        // distinct points), but they bloat the kd-tree leaves (unsplittable).
        // Dense mobile-mapping data can be half duplicates at 1 mm; dropping
        // them made queries ~35-45 % faster.
        points.par_sort_unstable_by_key(|p| p.map(f64::to_bits));
        points.dedup_by_key(|p| p.map(f64::to_bits));
        // f64, not f32: with f32 the neighbour picked among near-ties depended on
        // the tile centre, so different tilings (thread count, memory limit)
        // gave 1-unit differences in ~1e-7 of the points. f64 queries were
        // measured as fast as f32 ones.
        let rel: Vec<[f64; 3]> = points.iter().map(|p| std::array::from_fn(|a| p[a] - center[a])).collect();
        let tree = if rel.is_empty() {
            None
        } else {
            Some(ImmutableKdTree::new_from_slice_parallel(&rel).map_err(|e| anyhow!("kd-tree: {e:?}"))?)
        };
        let occupancy = Occupancy::new(&points, reach, max_dist);
        Ok(Self { tree, points, center, occupancy })
    }

    #[inline]
    fn query_point(&self, p: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|a| p[a] - self.center[a])
    }

    /// The nearest point if one lies within `max_dist`: (distance, index).
    /// `nearest_one` has kiddo's SIMD leaf scan (feature `simd`); the occupancy
    /// test keeps it from searching far for points without close neighbours.
    #[inline]
    fn nearest_within(&self, p: [f64; 3], max_dist: f64) -> Option<(f64, usize)> {
        if !self.occupancy.maybe_within(p) {
            return None;
        }
        let tree = self.tree.as_ref()?;
        let nn = tree.query(&self.query_point(p)).nearest_one::<SquaredEuclidean<f64>>().execute();
        let i = nn.item as usize;
        Some((dist(p, self.points[i]), i)).filter(|&(d, _)| d <= max_dist)
    }

    /// Up to `k` nearest points within `max_dist`, nearest first: (distance, index).
    fn within(&self, p: [f64; 3], k: NonZeroUsize, max_dist: f64) -> Vec<(f64, usize)> {
        let Some(tree) = self.tree.as_ref().filter(|_| self.occupancy.maybe_within(p)) else { return Vec::new() };
        // A hair wider (relative coordinates round differently), then exact on
        // the absolute coordinates: every B point within max_dist
        // is in this tile's tree, so the result doesn't depend on the tiling.
        let r = max_dist * (1.0 + 1e-9) + 1e-9;
        let mut out: Vec<(f64, usize)> = tree
            .query(&self.query_point(p))
            .nearest_n::<SquaredEuclidean<f64>>(k)
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
                let nbrs = nb.within(p, k, self.max_dist);
                (nbrs.first().copied(), Some(nbrs))
            }
            None => (nb.nearest_within(p, self.max_dist), None),
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
        let nb = Neighbours::new(b_points, center, &reach, self.max_dist)?;
        let t2 = Instant::now();
        stats.b_in_trees = nb.points.len() as u64;
        stats.max_b_in_tile = stats.b_in_trees;

        let inside = grow(&tb, self.eps);
        let encoded: Vec<(NodeChunk, Stats)> = owned
            .par_iter()
            .map(|e| {
                let compressed = self.a.compressed(e);
                let c = self.a.decode(e, self.record.decode_fields(compressed))?;
                let mut s = Stats::new(nf);
                let mut vals = vec![0; c.point_count() * nf];
                for (p, out) in c.positions().expect("decoded with Z").zip(vals.chunks_exact_mut(nf)) {
                    let m = self.measure(&nb, p, out);
                    s.add(out, m);
                    s.outside_tile += u64::from(!contains(&inside, p));
                }
                s.spliced = u64::from(self.record.splices(compressed));
                let bytes = self.record.encode(compressed, &c, &vals)?;
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
    let scale = opt.dist_scale.unwrap_or_else(|| default_scale(&all_scales));
    let layout = Layout::new(opt);
    let fields = layout.fields(scale)?;
    let nf = layout.n;
    // Components and plane distances lie within ±max_dist, the cap is 5 × max_dist.
    fields[0].check_range(5.0 * opt.max_dist)?;
    let cap = fields[0].encode(5.0 * opt.max_dist);
    // A few coordinate scale units: points may sit just outside their node box.
    let eps = 4.0 * all_scales.iter().copied().fold(0.0, f64::max);

    // Load balance: at least TILES_PER_SLOT tiles per tile in flight, so the
    // last tiles don't run alone (but never below MIN_TILE_POINTS).
    let total = a.las_header().number_of_points() + b.las_header().number_of_points();
    let balanced = (total / (TILES_PER_SLOT * opt.tiles_in_flight.max(1) as u64)).max(MIN_TILE_POINTS);
    let tile_points = opt.tile_points.min(balanced);
    let cut = Cut::new(a, Some(b), tile_points, 4.0 * opt.max_dist);

    let mut writer = CopcWriter::create(out, a, &fields)?;
    let mut record = writer.layout().clone();
    if opt.reencode {
        record = record.reencoding();
    }

    // Shared nodes: decode once, assign each point to the tile that contains it.
    let shared_chunks = cut
        .shared
        .par_iter()
        .map(|e| a.decode(e, record.decode_fields(a.compressed(e))))
        .collect::<Result<Vec<_>>>()?;
    let mut shared_by_tile: Vec<Vec<SharedPoint>> = (0..cut.tiles.len()).map(|_| Vec::new()).collect();
    for (node, c) in shared_chunks.iter().enumerate() {
        for (index, pos) in c.positions().expect("decoded with Z").enumerate() {
            shared_by_tile[cut.locate(pos)].push(SharedPoint { node: node as u32, index: index as u32, pos });
        }
    }
    let mut shared_values: Vec<Vec<i32>> = shared_chunks.iter().map(|c| vec![cap; c.point_count() * nf]).collect();
    let t_setup = t0.elapsed();

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

    let shared_encoded: Vec<Vec<u8>> = (&cut.shared, &shared_chunks, &shared_values)
        .into_par_iter()
        .map(|(e, c, v)| record.encode(a.compressed(e), c, v))
        .collect::<Result<_>>()?;
    stats.spliced += cut.shared.iter().filter(|e| record.splices(a.compressed(e))).count() as u64;
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
         tiles: {} ({} with points, levels {}–{}), {} shared nodes ({} points); {:.1} M points/tile, {} in flight\n  \
         reference: {} points, {} decoded (re-decode factor {:.2}), {} in kd-trees, max {} per tile\n  \
         beyond max-dist: {} ({:.2} %)\n  \
         time: setup {:.2} s, tiles {:.2} s, shared + finish {:.2} s, total {secs:.2} s = {:.1} M pts/s\n  \
         tile time (summed): reference decode {:.1} s, kd-tree {:.1} s, compared {:.1} s; slowest tile {:.1} s\n  \
         output {:.1} MB ({:+.3} B/pt vs compared file), {} of {} compared chunks spliced",
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
        tile_points as f64 / 1e6,
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
        stats.spliced,
        written.chunks,
    );
    Ok(Summary { stats, tiles: cut.tiles.len(), scale, b_points, fields: names })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nb(points: &[[f64; 3]]) -> Neighbours {
        let reach = Aabb { min: [-10.0; 3], max: [10.0; 3] };
        Neighbours::new(points.to_vec(), [0.0; 3], &reach, 2.0).unwrap()
    }

    #[test]
    fn occupancy_is_conservative() {
        let reach = Aabb { min: [0.0; 3], max: [100.0; 3] };
        let pts = [[50.0, 50.0, 50.0]];
        for max_dist in [0.3, 1.0] {
            let o = Occupancy::new(&pts, &reach, max_dist);
            // Everything within max_dist passes, whatever the cell size.
            for d in [[0.29, 0.0, 0.0], [0.0, -0.29, 0.0], [0.17, 0.17, -0.17]] {
                assert!(o.maybe_within([50.0 + d[0], 50.0 + d[1], 50.0 + d[2]]));
            }
            // Far away is rejected.
            assert!(!o.maybe_within([50.0 + 3.0 * o.cell, 50.0, 50.0]));
            assert!(!o.maybe_within([0.0, 0.0, 0.0]));
        }
    }

    #[test]
    fn nearest_within_caps() {
        let n = nb(&[[0.5, 0.0, 0.0], [0.0, 0.8, 0.0]]);
        assert_eq!(n.nearest_within([0.0; 3], 0.6).map(|x| x.0), Some(0.5));
        assert_eq!(n.nearest_within([0.0; 3], 0.4), None);
        assert_eq!(n.nearest_within([9.0, 9.0, 9.0], 2.0), None);
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
    fn default_scale_is_finer_but_floored() {
        assert_eq!(default_scale(&[0.001, 0.001, 0.001, 0.0005, 0.0005, 0.0005]), 0.0005);
        assert_eq!(default_scale(&[0.001, 0.001, 0.001, 1.3e-7, 1.0e-7, 6.0e-9]), MIN_DEFAULT_SCALE);
    }

    #[test]
    fn tiling_for_memory_fits() {
        let gb = 1u64 << 30;
        let fits = |limit: u64, (t, n): (u64, usize)| t * n as u64 * BYTES_PER_TILE_POINT <= limit;
        // Plenty: one tile per thread, capped size.
        assert_eq!(tiling_for_memory(512 * gb, 8), (MAX_TILE_POINTS, 8));
        // 31 GB, 8 threads: 8 tiles of ~37 M points.
        let r = tiling_for_memory(31 * gb, 8);
        assert_eq!(r.1, 8);
        assert!(r.0 > 30_000_000 && fits(31 * gb, r));
        // 4 GB, 32 threads: 32 tiles of just over 1 M points.
        let r = tiling_for_memory(4 * gb, 32);
        assert!(r.1 == 32 && r.0 >= MIN_TILE_POINTS && fits(4 * gb, r));
        // 2 GB, 32 threads: tiles would be < 1 M, so fewer at once.
        let r = tiling_for_memory(2 * gb, 32);
        assert!(r.1 < 32 && r.0 >= MIN_TILE_POINTS && fits(2 * gb, r));
        // Tiny budget: one small tile.
        let r = tiling_for_memory(gb / 16, 32);
        assert!(r.1 == 1 && r.0 < MIN_TILE_POINTS && fits(gb / 16, r));
    }
}
