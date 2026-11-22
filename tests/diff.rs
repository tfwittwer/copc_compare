//! `diff`: planted differences in synthetic clouds, layouts that differ, and c2c
//! outputs.

mod common;

use std::path::Path;

use common::{cloud, convert, copc, fixture, point, read_back, rng, scratch};
use copc_compare::c2c;
use copc_compare::diff::{self, Options, Report};
use copc_compare::input::CopcFile;
use las::point::Classification;

fn run(x: &Path, y: &Path, opt: Options) -> Report {
    let (x, y) = (CopcFile::open(x).unwrap(), CopcFile::open(y).unwrap());
    diff::run(&x, &y, &opt).unwrap()
}

/// Small tiles, so the differences spread over a cut with shared nodes.
fn opts() -> Options {
    Options { tile_points: 5_000, ..Options::default() }
}

fn by_field(r: &Report) -> Vec<(&str, u64)> {
    r.changed_by_field.iter().map(|(n, c)| (n.as_str(), *c)).collect()
}

#[test]
fn identical_files() {
    let f = fixture("untwine_reference.copc.laz");
    let r = run(&f, &f, opts());
    let nodes = CopcFile::open(&f).unwrap().data_entries().count();
    assert!(r.equal && r.differing_tiles == 0);
    assert_eq!((r.identical_nodes, r.decoded), (nodes, [0, 0]), "every node byte-identical, nothing decoded");

    // Same points, another converter: other octree and order, all decoded.
    let re = scratch("untwine_reconverted.copc.laz");
    convert(std::slice::from_ref(&f), &re, None);
    let r = run(&f, &re, opts());
    assert!(r.equal, "{r:?}");
    assert_eq!(r.decoded, [829_570; 2]);
    assert!(r.not_compared.is_empty(), "{:?}", r.not_compared);
}

#[test]
fn planted_differences() {
    // Enough points for an octree with several levels.
    let n = 200_000;
    let xs = cloud(1, n);
    let mut ys = xs.clone();
    ys[1].classification = Classification::new(if u8::from(xs[1].classification) == 3 { 4 } else { 3 }).unwrap();
    ys[2].scanner_channel = (xs[2].scanner_channel + 1) % 4;
    ys[3].is_withheld = !xs[3].is_withheld;
    ys[5].gps_time = Some(-1.0);
    ys.push(xs[4].clone()); // duplicate
    ys.push(point(n, &mut rng(99))); // new
    ys.remove(0);
    ys.reverse();
    let x = copc("planted_x", 7, [0.001; 3], [0.0; 3], &xs);
    let y = copc("planted_y", 7, [0.001; 3], [0.0; 3], &ys);

    let r = run(&x, &y, opts());
    assert!(!r.equal && r.detailed);
    let shared = CopcFile::open(&x).unwrap().data_entries().count();
    assert!(r.tiles > 8 && shared > 8, "{} tiles, {shared} nodes", r.tiles);
    // 0 removed, 5's GPS time changed (a different point); 4 doubled, one new.
    assert_eq!(r.only, [2, 3]);
    assert_eq!(r.changed, 3);
    assert_eq!(by_field(&r), [("channel", 1), ("flags", 1), ("classification", 1)]);
    assert!((1..=7).contains(&r.differing_tiles), "{} tiles differ", r.differing_tiles);

    let r = run(&x, &y, Options { ignore: vec!["classification".into()], ..opts() });
    assert_eq!((r.only, r.changed), ([2, 3], 2));
    assert!(r.not_compared.contains(&("classification".into(), "not selected".into())));

    // Position only: 5 matches again (same position), and so do 1–3.
    let r = run(&x, &y, Options { fields: Some(vec![]), ..opts() });
    assert_eq!((r.only, r.changed), ([1, 2], 0));
    let r = run(&x, &y, Options { fields: Some(vec!["classification".into()]), ..opts() });
    assert_eq!((r.only, r.changed), ([1, 2], 1));
    assert_eq!(by_field(&r), [("classification", 1)]);

    let q = run(&x, &y, Options { quick: true, ..opts() });
    assert!(!q.equal && !q.detailed && q.only == [0, 0]);
    assert_eq!(q.differing_tiles, run(&x, &y, opts()).differing_tiles);

    // Unknown names and ignoring the position are errors.
    let (fx, fy) = (CopcFile::open(&x).unwrap(), CopcFile::open(&y).unwrap());
    assert!(diff::run(&fx, &fy, &Options { ignore: vec!["colour".into()], ..opts() }).is_err());
    assert!(diff::run(&fx, &fy, &Options { ignore: vec!["xyz".into()], ..opts() }).is_err());
}

#[test]
fn layouts_differ() {
    let pts = cloud(2, 20_000);
    let x = copc("layout_x", 7, [0.001; 3], [0.0; 3], &pts);
    // Finer scale, other offsets, no RGB.
    let y = copc("layout_y", 6, [0.0005, 0.001, 0.00025], [500.0, -300.0, 10.0], &pts);
    let r = run(&x, &y, opts());
    assert!(r.equal, "{r:?}");
    assert_eq!(r.not_compared, [("rgb".to_string(), "only in X".to_string())]);
    assert_eq!(r.identical_nodes, 0);

    let mut moved = pts.clone();
    moved[7].x += 0.001;
    let y = copc("layout_moved", 6, [0.0005, 0.001, 0.00025], [500.0, -300.0, 10.0], &moved);
    let r = run(&x, &y, opts());
    assert_eq!((r.only, r.changed), ([1, 1], 0));
}

#[test]
fn crop_against_full() {
    // X's root cube is a small corner of Y's. Y's points outside it must not
    // pile into X's border tiles: the cut covers both cubes.
    let full = cloud(5, 200_000);
    let crop: Vec<las::Point> = full.iter().filter(|p| p.x < 1010.0 && p.y < 2010.0).cloned().collect();
    let x = copc("crop_x", 7, [0.001; 3], [0.0; 3], &crop);
    let y = copc("crop_y", 7, [0.001; 3], [0.0; 3], &full);
    let r = run(&x, &y, opts());
    assert_eq!((r.only, r.changed), ([0, (full.len() - crop.len()) as u64], 0));
    // About 200k points at 5k per tile.
    assert!(r.tiles >= 40, "{} tiles", r.tiles);
}

#[test]
fn c2c_outputs() {
    let a = copc("c2c_a", 6, [0.001; 3], [0.0; 3], &cloud(3, 20_000));
    let b = copc("c2c_b", 6, [0.001; 3], [0.0; 3], &cloud(4, 20_000));
    let (fa, fb) = (CopcFile::open(&a).unwrap(), CopcFile::open(&b).unwrap());
    let out = |name: &str, opt: c2c::Options| {
        let p = scratch(name);
        c2c::run(&fa, &fb, &p, &opt).unwrap();
        p
    };
    let o1 = out("c2c_1m.copc.laz", c2c::Options::new(1.0));
    let o05 = out("c2c_05m.copc.laz", c2c::Options::new(0.5));
    let coarse = out("c2c_1m_cm.copc.laz", c2c::Options { dist_scale: Some(0.01), ..c2c::Options::new(1.0) });

    // The output holds the compared file's points.
    let r = run(&o1, &a, opts());
    assert!(r.equal);
    assert_eq!(r.not_compared, [("c2c_dist".to_string(), "only in X".to_string())]);
    let r = run(&o1, &coarse, opts());
    assert!(r.equal);
    assert_eq!(r.not_compared, [("c2c_dist".to_string(), "encoded differently".to_string())]);

    // Distances in (0.5, 1] and the cap differ. Same octree and order, so the
    // expected count comes from zipping the records.
    let (r1, r05) = (read_back(&o1), read_back(&o05));
    assert!(r1.iter().zip(&r05).all(|(p, q)| p.0 == q.0));
    let expected = r1.iter().zip(&r05).filter(|(p, q)| p.1 != q.1).count() as u64;
    assert!(expected > 1000, "{expected}");
    let r = run(&o1, &o05, opts());
    assert_eq!((r.only, r.changed), ([0, 0], expected));
    assert_eq!(by_field(&r), [("c2c_dist", expected)]);
}
