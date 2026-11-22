//! Phase 2: tiling + nearest-neighbour distances, checked against an independent
//! brute-force oracle on points read back through copc-reader.

mod common;

use std::path::{Path, PathBuf};

use copc_compare::c2c::{self, Options};
use copc_compare::extra_bytes::DESCRIPTOR_SIZE;
use copc_compare::input::CopcFile;
use copc_compare::tiles::Cut;
use copc_streaming::Fields;

use common::{Oracle, cloud, convert, copc, fixture, read_back, scratch, validator, write_las};

fn opts(max_dist: f64, tile_points: u64) -> Options {
    Options { tile_points, tiles_in_flight: 3, ..Options::new(max_dist) }
}

fn run(a: &Path, b: &Path, out: &Path, o: &Options) -> c2c::Summary {
    let a = CopcFile::open(a).unwrap();
    let b = CopcFile::open(b).unwrap();
    c2c::run(&a, &b, out, o).unwrap()
}

/// (position, our `n` raw field values) for every output point, via copc-reader.
fn fields_of(path: &Path, n: usize) -> Vec<([f64; 3], Vec<i32>)> {
    read_back(path)
        .into_iter()
        .map(|(p, eb)| {
            let ours = &eb[eb.len() - 4 * n..];
            (p, ours.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect())
        })
        .collect()
}

/// (position, c2c_dist raw value) for every output point of a NN-only run.
fn values(path: &Path) -> Vec<([f64; 3], i32)> {
    fields_of(path, 1).into_iter().map(|(p, v)| (p, v[0])).collect()
}

/// Compare every output value with the oracle's. Returns (points, beyond cap).
fn check_against_oracle(out: &Path, reference: &Path, max_dist: f64, scale: f64) -> (usize, usize) {
    let oracle = Oracle::new(read_back(reference).into_iter().map(|(p, _)| p), max_dist);
    let cap = (5.0 * max_dist / scale).round() as i32;
    let vals = values(out);
    let mut beyond = 0;
    for &(p, v) in &vals {
        let want = match oracle.nearest(p) {
            Some(d) if (d - max_dist).abs() < 1e-9 => continue, // either side of the cap is fine
            Some(d) => (d / scale).round() as i32,
            None => cap,
        };
        beyond += usize::from(want == cap);
        assert!((v - want).abs() <= 1, "point {p:?}: c2c_dist {v}, oracle {want} (scale {scale})");
    }
    (vals.len(), beyond)
}

/// Assert two outputs hold identical records per node (chunk order may differ).
fn assert_same_nodes(x: &Path, y: &Path) {
    let (x, y) = (CopcFile::open(x).unwrap(), CopcFile::open(y).unwrap());
    assert_eq!(x.entries().len(), y.entries().len());
    for (ex, ey) in x.data_entries().zip(y.data_entries()) {
        assert_eq!(ex.key, ey.key);
        let (cx, cy) = (x.decode(ex, Fields::ALL).unwrap(), y.decode(ey, Fields::ALL).unwrap());
        assert!(cx.cloud().raw_bytes() == cy.cloud().raw_bytes(), "node {:?} differs", ex.key);
    }
}

fn stats_of(path: &Path) -> (i64, i64) {
    let f = CopcFile::open(path).unwrap();
    let v = f.vlrs.iter().find(|v| v.is("LASF_Spec", 4)).unwrap();
    let d = &v.data[v.data.len() - DESCRIPTOR_SIZE..];
    let at = |o: usize| i64::from_le_bytes(d[o..o + 8].try_into().unwrap());
    (at(64), at(88))
}

#[test]
fn self_distance_is_zero() {
    let u = fixture("untwine_reference.copc.laz");
    for (name, tile_points) in [("self_one_tile", 10_000_000), ("self_small_tiles", 30_000)] {
        let out = scratch(&format!("{name}.copc.laz"));
        let s = run(&u, &u, &out, &opts(0.5, tile_points));
        if tile_points < 100_000 {
            assert!(s.tiles > 8 && s.stats.shared_points > 0, "expected many tiles and shared nodes");
        }
        assert_eq!(s.stats.beyond_cap, 0);
        let vals = values(&out);
        assert_eq!(vals.len(), 829_570);
        assert!(vals.iter().all(|&(_, v)| v == 0), "{name}: nonzero self-distance");
        assert_eq!(stats_of(&out), (0, 0));
        common::assert_no_new_issues(&u, &out);
    }
}

/// B: 0.1 m grid on the plane z = 0 over 60 × 60 m. A: the same grid over 70 ×
/// 60 m, each point at a known height h in [0, 2.99]. Above B, d = h (or the cap
/// for h > max_dist); beyond B's edge the oracle decides. Different scales and
/// offsets, RD-sized coordinates.
fn plane_pair() -> (PathBuf, PathBuf) {
    // Built once per test binary: several tests use it and run in parallel.
    static PAIR: std::sync::OnceLock<(PathBuf, PathBuf)> = std::sync::OnceLock::new();
    PAIR.get_or_init(make_plane_pair).clone()
}

fn make_plane_pair() -> (PathBuf, PathBuf) {
    let (x0, y0) = (121_000.0, 486_000.0);
    let h = |i: usize, j: usize| ((i * 7 + j * 13) % 300) as f64 / 100.0;
    let b: Vec<[f64; 3]> = (0..=600)
        .flat_map(|i| (0..=600).map(move |j| [x0 + i as f64 * 0.1, y0 + j as f64 * 0.1, 0.0]))
        .collect();
    let a: Vec<[f64; 3]> = (0..700)
        .flat_map(|i| (0..=600).map(move |j| [x0 + i as f64 * 0.1, y0 + j as f64 * 0.1, h(i, j)]))
        .collect();
    let (a_las, b_las) = (scratch("plane_a.las"), scratch("plane_b.las"));
    write_las(&a_las, 0.001, [120_000.0, 480_000.0, 0.0], &a);
    write_las(&b_las, 0.0005, [119_000.0, 479_000.0, 0.0], &b);
    let (a_copc, b_copc) = (scratch("plane_a.copc.laz"), scratch("plane_b.copc.laz"));
    convert(&[a_las], &a_copc, None);
    convert(&[b_las], &b_copc, None);
    (a_copc, b_copc)
}

#[test]
fn plane_known_heights() {
    let (a, b) = plane_pair();
    let max_dist = 2.0;
    let mut outs = Vec::new();
    for tile_points in [40_000, 100_000_000] {
        let out = scratch(&format!("plane_out_{tile_points}.copc.laz"));
        let s = run(&a, &b, &out, &opts(max_dist, tile_points));
        assert_eq!(s.scale, 0.0005, "default scale is the finer of the two");
        validator::assert_valid_copc(&out);

        // Exact heights where B lies below.
        let cap = (5.0 * max_dist / s.scale).round() as i32;
        let mut checked = 0;
        for (p, v) in values(&out) {
            if p[0] - 121_000.0 <= 60.0 + 1e-6 {
                let want = if p[2] <= max_dist { (p[2] / s.scale).round() as i32 } else { cap };
                assert_eq!(v, want, "point {p:?}");
                checked += 1;
            }
        }
        assert_eq!(checked, 601 * 601);

        let (n, beyond) = check_against_oracle(&out, &b, max_dist, s.scale);
        assert_eq!(n, 700 * 601);
        assert_eq!(beyond as u64, s.stats.beyond_cap);
        outs.push(out);
    }
    // Small tiles (owned + shared, many B re-decodes) and one tile agree exactly.
    assert_same_nodes(&outs[0], &outs[1]);
}

/// splitmix64 in [0, 1).
fn rand01(s: &mut u64) -> f64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// Random clouds: A 250 k, B 150 k shifted 5 m (so a strip has no neighbours),
/// A at 1 mm, B at 1 cm. Built once per test binary.
fn random_pair() -> (PathBuf, PathBuf) {
    static PAIR: std::sync::OnceLock<(PathBuf, PathBuf)> = std::sync::OnceLock::new();
    PAIR.get_or_init(|| {
        let mut s = 42;
        let cloud = |s: &mut u64, n: usize, x0: f64| -> Vec<[f64; 3]> {
            (0..n).map(|_| [x0 + 40.0 * rand01(s), 480_000.0 + 40.0 * rand01(s), 4.0 * rand01(s)]).collect()
        };
        let a = cloud(&mut s, 250_000, 120_000.0);
        let b = cloud(&mut s, 150_000, 120_005.0); // overlaps A on 35 of 40 m
        let (a_las, b_las) = (scratch("rand_a.las"), scratch("rand_b.las"));
        write_las(&a_las, 0.001, [120_000.0, 480_000.0, 0.0], &a);
        write_las(&b_las, 0.01, [100_000.0, 400_000.0, 0.0], &b);
        let (a_copc, b_copc) = (scratch("rand_a.copc.laz"), scratch("rand_b.copc.laz"));
        convert(&[a_las], &a_copc, None);
        convert(&[b_las], &b_copc, None);
        (a_copc, b_copc)
    })
    .clone()
}

/// The result must not depend on the tiling (which follows from --threads and
/// --memory-limit): at a 1 µm distance scale, where any difference in the
/// neighbour picked among near-ties would show, two tilings give identical nodes.
#[test]
fn tiling_independent_at_fine_scale() {
    let (a, b) = random_pair();
    let mut outs = Vec::new();
    for tile_points in [20_000, 10_000_000] {
        let out = scratch(&format!("rand_fine_{tile_points}.copc.laz"));
        let o = Options { dist_scale: Some(1e-6), components: true, ..opts(0.5, tile_points) };
        let s = run(&a, &b, &out, &o);
        assert!(tile_points > 1_000_000 || s.tiles >= 8);
        outs.push(out);
    }
    assert_same_nodes(&outs[0], &outs[1]);
}

#[test]
fn random_clouds_match_oracle() {
    let (a_copc, b_copc) = random_pair();

    let max_dist = 0.5;
    let out = scratch("rand_out.copc.laz");
    let sm = run(&a_copc, &b_copc, &out, &opts(max_dist, 20_000));
    assert!(sm.tiles >= 8 && sm.stats.shared_points > 0, "expected tiles and shared nodes: {sm:?}");
    assert_eq!(sm.scale, 0.001);
    validator::assert_valid_copc(&out);
    let (n, beyond) = check_against_oracle(&out, &b_copc, max_dist, sm.scale);
    assert_eq!(n, 250_000);
    assert!(beyond > 10_000, "the 5 m strip without reference points is beyond the cap");
    assert_eq!(beyond as u64, sm.stats.beyond_cap);

    // Same inputs, same output bytes.
    let again = scratch("rand_out_again.copc.laz");
    run(&a_copc, &b_copc, &again, &opts(max_dist, 20_000));
    assert!(std::fs::read(&out).unwrap() == std::fs::read(&again).unwrap(), "output not deterministic");
}

#[test]
fn cut_partitions_cube_and_nodes() {
    let a = CopcFile::open(&fixture("untwine_reference.copc.laz")).unwrap();
    for (tile_points, with_b) in [1, 30_000, 200_000, u64::MAX].into_iter().flat_map(|t| [(t, false), (t, true)]) {
        let cut = Cut::new(&a, with_b.then_some(&a), tile_points, 0.5);
        if with_b && tile_points == 200_000 {
            // Counting the reference too (here: the same file) splits further.
            assert!(cut.tiles.len() > Cut::new(&a, None, tile_points, 0.5).tiles.len());
        }
        // Tiles: no tile is an ancestor of another, and their volumes sum to the cube.
        let vol: f64 = cut.tiles.iter().map(|k| 8f64.powi(-k.level)).sum();
        assert!((vol - 1.0).abs() < 1e-12, "tiles cover {vol} of the cube");
        for t in &cut.tiles {
            let mut k = t.parent();
            while let Some(p) = k {
                assert!(!cut.tiles.contains(&p), "{t:?} lies under tile {p:?}");
                k = p.parent();
            }
        }
        // Data nodes: each exactly once, owned or shared.
        let mut seen: Vec<_> = cut.owned.iter().flatten().chain(&cut.shared).map(|e| e.key).collect();
        seen.sort_by_key(|k| (k.level, k.x, k.y, k.z));
        let all: Vec<_> = a.data_entries().map(|e| e.key).collect();
        assert_eq!(seen, all);
        // Every point locates to a tile whose box (plus a hair) contains it.
        for e in &cut.shared {
            let c = a.decode(e, Fields::ALL).unwrap();
            for p in c.positions().unwrap() {
                let b = cut.tile_bounds(cut.locate(p));
                assert!((0..3).all(|i| p[i] >= b.min[i] - 1e-6 && p[i] <= b.max[i] + 1e-6));
            }
        }
    }
}

#[test]
fn crs_mismatch_needs_force() {
    let u = fixture("untwine_reference.copc.laz");
    let a = CopcFile::open(&u).unwrap();
    let mut b = CopcFile::open(&u).unwrap();
    let wkt = b.vlrs.iter_mut().find(|v| v.is("LASF_Projection", 2112)).expect("fixture has a WKT");
    wkt.data[20] ^= 0x01;
    let out = scratch("crs.copc.laz");
    let err = c2c::run(&a, &b, &out, &opts(0.5, u64::MAX)).unwrap_err();
    assert!(err.to_string().contains("CRS differs"), "{err}");
    assert!(!out.exists(), "nothing written on a refused run");
    let forced = Options { force: true, ..opts(0.5, u64::MAX) };
    assert_eq!(c2c::run(&a, &b, &out, &forced).unwrap().stats.beyond_cap, 0);
}

/// Flat grid with --components and --model plane: above B, dx = dy = 0,
/// dz = c2c_plane = h. Field order: dist, dx, dy, dz, plane.
#[test]
fn plane_pair_components_and_plane_model() {
    let (a, b) = plane_pair();
    let max_dist = 2.0;
    let mut outs = Vec::new();
    for tile_points in [40_000, 100_000_000] {
        let out = scratch(&format!("plane_full_{tile_points}.copc.laz"));
        let o = Options { components: true, plane_k: Some(8), ..opts(max_dist, tile_points) };
        let s = run(&a, &b, &out, &o);
        assert_eq!(s.fields, ["c2c_dist", "c2c_dx", "c2c_dy", "c2c_dz", "c2c_plane"]);
        validator::assert_valid_copc(&out);
        let cap = (5.0 * max_dist / s.scale).round() as i32;
        let (mut above, mut edge) = (0, 0);
        for (p, v) in fields_of(&out, 5) {
            let h = (p[2] / s.scale).round() as i32;
            if p[0] - 121_000.0 <= 60.0 + 1e-6 {
                let want = if p[2] <= max_dist { vec![h, 0, 0, h, h] } else { vec![cap; 5] };
                assert_eq!(v, want, "point {p:?}");
                above += 1;
            } else if v[0] == cap {
                assert_eq!(v, vec![cap; 5]);
            } else {
                // Beyond B's edge: the offset points back towards the edge.
                assert!(v[1] > 0 && v[2] == 0, "point {p:?}: {v:?}");
                edge += 1;
            }
        }
        assert_eq!(above, 601 * 601);
        // Only beyond the edge can the neighbours be collinear (one grid column).
        assert!(s.stats.plane_fallbacks > 0 && s.stats.plane_fallbacks <= edge, "{} fallbacks, {edge} edge points", s.stats.plane_fallbacks);
        outs.push(out);
    }
    assert_same_nodes(&outs[0], &outs[1]);
}

/// Tilted planar B; A points off the grid, offset by a known t along the plane
/// normal. c2c_plane = t, while c2c_dist (to the nearest grid point) is larger.
#[test]
fn tilted_plane_model() {
    let (x0, y0) = (121_000.0, 486_000.0);
    let z = |x: f64, y: f64| 0.2 * x + 0.1 * y;
    let b: Vec<[f64; 3]> = (0..=300)
        .flat_map(|i| (0..=300).map(move |j| (i as f64 * 0.1, j as f64 * 0.1)))
        .map(|(x, y)| [x0 + x, y0 + y, z(x, y)])
        .collect();
    let len = (0.2f64 * 0.2 + 0.1 * 0.1 + 1.0).sqrt();
    let normal = [-0.2 / len, -0.1 / len, 1.0 / len];
    let mut seed = 7;
    let mut a = Vec::new();
    let mut offsets = Vec::new();
    for _ in 0..20_000 {
        // Away from the edges, between grid nodes, at t in [-0.15, 0.15] m.
        let (x, y) = (2.0 + 26.0 * rand01(&mut seed), 2.0 + 26.0 * rand01(&mut seed));
        let t = 0.3 * rand01(&mut seed) - 0.15;
        a.push([x0 + x + t * normal[0], y0 + y + t * normal[1], z(x, y) + t * normal[2]]);
        offsets.push(t);
    }
    let (a_las, b_las) = (scratch("tilt_a.las"), scratch("tilt_b.las"));
    write_las(&a_las, 0.0001, [121_000.0, 486_000.0, 0.0], &a);
    write_las(&b_las, 0.0001, [121_000.0, 486_000.0, 0.0], &b);
    let (a_copc, b_copc) = (scratch("tilt_a.copc.laz"), scratch("tilt_b.copc.laz"));
    convert(&[a_las], &a_copc, None);
    convert(&[b_las], &b_copc, None);

    let out = scratch("tilt_out.copc.laz");
    let o = Options { components: true, plane_k: Some(8), ..opts(0.5, 10_000) };
    let s = run(&a_copc, &b_copc, &out, &o);
    assert_eq!(s.scale, 0.0001);
    // Match outputs to the generated offsets by position.
    // Raw (quantised) coordinates as written: offset (121000, 486000, 0), scale 0.1 mm.
    let raw = |p: &[f64; 3]| -> [i64; 3] {
        let off = [121_000.0, 486_000.0, 0.0];
        std::array::from_fn(|i| ((p[i] - off[i]) * 1e4).round() as i64)
    };
    let want: std::collections::HashMap<[i64; 3], f64> = a.iter().zip(&offsets).map(|(p, &t)| (raw(p), t)).collect();
    let mut worst = 0.0f64;
    for (p, v) in fields_of(&out, 5) {
        let t = want[&raw(&p)];
        let plane = f64::from(v[4]) * s.scale;
        worst = worst.max((plane - t).abs());
        // NN is never closer than the plane, and the components add up to it.
        assert!(f64::from(v[0]) * s.scale >= t.abs() - 2e-4, "{p:?}: dist {} < |t| {t}", v[0]);
        let norm = (f64::from(v[1]).powi(2) + f64::from(v[2]).powi(2) + f64::from(v[3]).powi(2)).sqrt();
        assert!((norm - f64::from(v[0])).abs() <= 2.0, "{p:?}: |components| {norm} vs dist {}", v[0]);
    }
    // Coordinates are quantised to 0.1 mm, so the plane isn't exact to the bit.
    assert!(worst < 3e-4, "plane distance off by up to {worst} m");
}

#[test]
fn splice_equals_reencode() {
    // laz-rs input (copc-converter) with random scanner channels and shared
    // nodes: splicing writes the same file as re-encoding.
    let a = copc("c2c_splice_a", 7, [0.001; 3], [0.0; 3], &cloud(11, 60_000));
    let b = copc("c2c_splice_b", 7, [0.001; 3], [0.0; 3], &cloud(12, 60_000));
    let o = Options { components: true, ..opts(0.5, 10_000) };
    let (sp, re) = (scratch("c2c_splice.copc.laz"), scratch("c2c_reenc.copc.laz"));
    let s = run(&a, &b, &sp, &o);
    let r = run(&a, &b, &re, &Options { reencode: true, ..o.clone() });
    let nodes = CopcFile::open(&a).unwrap().data_entries().count() as u64;
    assert!(s.stats.shared_points > 0, "test needs shared nodes");
    assert_eq!((s.stats.spliced, r.stats.spliced), (nodes, 0));
    assert!(std::fs::read(&sp).unwrap() == std::fs::read(&re).unwrap(), "spliced and re-encoded files differ");

    // Untwine's fixture, written by lazperf: it encodes these chunks exactly as
    // laz-rs does, so the files are identical too.
    let u = fixture("untwine_reference.copc.laz");
    let (sp, re) = (scratch("c2c_splice_u.copc.laz"), scratch("c2c_reenc_u.copc.laz"));
    let s = run(&u, &u, &sp, &o);
    run(&u, &u, &re, &Options { reencode: true, ..o });
    assert_eq!(s.stats.spliced, CopcFile::open(&u).unwrap().data_entries().count() as u64);
    assert!(std::fs::read(&sp).unwrap() == std::fs::read(&re).unwrap(), "untwine: spliced and re-encoded files differ");
    common::assert_no_new_issues(&u, &sp);
}

#[test]
fn distribution_matches_read_back() {
    // The statistics printed after a run equal those of the values read back,
    // computed independently (sorted median, two-pass variance).
    let a = copc("dist_a", 6, [0.001; 3], [0.0; 3], &cloud(21, 30_000));
    let b = copc("dist_b", 6, [0.001; 3], [0.0; 3], &cloud(22, 30_000));
    let out = scratch("dist_out.copc.laz");
    let o = Options { components: true, plane_k: Some(8), ..opts(1.0, 10_000) };
    let s = run(&a, &b, &out, &o);
    assert!(s.stats.shared_points > 0, "test needs shared nodes");
    let cap = (5.0 / s.scale).round() as i32;
    let vals: Vec<Vec<i32>> = fields_of(&out, 5).into_iter().map(|(_, v)| v).filter(|v| v[0] != cap).collect();
    assert!(!vals.is_empty() && (vals.len() as u64) < s.stats.a_points, "test needs points on both sides of the cap");
    assert_eq!(s.distribution.len(), 5);
    for (f, got) in s.distribution.iter().enumerate() {
        let got = got.as_ref().unwrap();
        let mut x: Vec<i64> = vals.iter().map(|v| i64::from(v[f])).collect();
        x.sort_unstable();
        let n = x.len();
        let mean = x.iter().sum::<i64>() as f64 / n as f64;
        let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n as f64;
        let median = (x[(n - 1) / 2] + x[n / 2]) as f64 / 2.0;
        let sc = s.scale;
        assert_eq!(got.n, n as u64, "field {f}");
        assert_eq!((got.min, got.max, got.median), (x[0] as f64 * sc, x[n - 1] as f64 * sc, median * sc), "field {f}");
        assert!((got.mean - mean * sc).abs() < 1e-12, "field {f}: mean {} vs {}", got.mean, mean * sc);
        assert!((got.stddev - var.sqrt() * sc).abs() < 1e-9, "field {f}: stddev {} vs {}", got.stddev, var.sqrt() * sc);
        assert_eq!(got.median_error, 0.0);
    }
}
