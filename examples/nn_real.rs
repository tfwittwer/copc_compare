//! kd-tree query cost on real data: B points in a box (+1 m) of REFERENCE, queried
//! with A points of COMPARED in the box, in node order and spatially sorted.
//!
//! cargo run --release --example nn_real -- COMPARED REFERENCE BOX_SIDE_M
use std::collections::HashSet;
use std::time::Instant;

use anyhow::Result;
use copc_compare::input::CopcFile;
use copc_compare::tiles::{contains, grow};
use copc_streaming::{Aabb, Fields};
use kiddo::{ImmutableKdTree, SquaredEuclidean};

fn points_in(f: &CopcFile, b: &Aabb) -> Result<Vec<[f64; 3]>> {
    let root = f.root_bounds();
    let mut out = Vec::new();
    for e in f.data_entries().filter(|e| e.key.bounds(&root).intersects(b)) {
        out.extend(f.decode(e, Fields::Z)?.positions().unwrap().filter(|&p| contains(b, p)));
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (a, b) = (CopcFile::open(args[1].as_ref())?, CopcFile::open(args[2].as_ref())?);
    let side: f64 = args[3].parse()?;
    // Centre on the densest node of A's deepest level.
    let deepest = a.data_entries().map(|e| e.key.level).max().unwrap();
    let e = a.data_entries().filter(|e| e.key.level == deepest).max_by_key(|e| e.point_count).unwrap();
    let nb = e.key.bounds(&a.root_bounds());
    let c: [f64; 3] = std::array::from_fn(|i| 0.5 * (nb.min[i] + nb.max[i]));
    let bx = Aabb { min: [c[0] - side / 2.0, c[1] - side / 2.0, -1e9], max: [c[0] + side / 2.0, c[1] + side / 2.0, 1e9] };
    let qa = points_in(&a, &bx)?;
    let pad: f64 = args.get(4).map_or(Ok(1.0), |s| s.parse())?;
    let pb = points_in(&b, &grow(&bx, pad))?;
    let rel = |p: &[f64; 3]| -> [f32; 3] { std::array::from_fn(|i| (p[i] - c[i]) as f32) };
    let distinct: HashSet<[i64; 3]> = pb.iter().map(|p| p.map(|v| (v * 1000.0).round() as i64)).collect();
    println!("box {side} m: {} compared, {} reference ({} distinct at 1 mm)", qa.len(), pb.len(), distinct.len());

    let rb: Vec<[f32; 3]> = pb.iter().map(rel).collect();
    let t = Instant::now();
    let tree: ImmutableKdTree<f32, 3> = ImmutableKdTree::new_from_slice(&rb).unwrap();
    println!("  build {:.0} ns/pt", t.elapsed().as_secs_f64() / rb.len() as f64 * 1e9);

    let time = |name: &str, qs: &[[f32; 3]]| {
        let t = Instant::now();
        let mut acc = 0.0f64;
        for q in qs {
            acc += f64::from(tree.query(q).nearest_one::<SquaredEuclidean<f32>>().execute().distance).sqrt();
        }
        println!("  {name:<22} {:>6.0} ns/query (mean d {:.4} m)", t.elapsed().as_secs_f64() / qs.len() as f64 * 1e9, acc / qs.len() as f64);
    };
    let q: Vec<[f32; 3]> = qa.iter().map(rel).collect();
    time("node order", &q);
    drop(tree);
    // Exact duplicates can't change a nearest distance: drop them.
    let t = Instant::now();
    let mut keyed: Vec<[i64; 3]> = pb.iter().map(|p| p.map(|v| (v * 1000.0).round() as i64)).collect();
    keyed.sort_unstable();
    keyed.dedup();
    let dedup: Vec<[f32; 3]> = keyed.iter().map(|k| rel(&k.map(|v| v as f64 / 1000.0))).collect();
    let tree: ImmutableKdTree<f32, 3> = ImmutableKdTree::new_from_slice(&dedup).unwrap();
    println!("  dedup + build {:.0} ns/pt", t.elapsed().as_secs_f64() / rb.len() as f64 * 1e9);
    let time = |name: &str, qs: &[[f32; 3]]| {
        let t = Instant::now();
        let mut acc = 0.0f64;
        for q in qs {
            acc += f64::from(tree.query(q).nearest_one::<SquaredEuclidean<f32>>().execute().distance).sqrt();
        }
        println!("  {name:<22} {:>6.0} ns/query (mean d {:.4} m)", t.elapsed().as_secs_f64() / qs.len() as f64 * 1e9, acc / qs.len() as f64);
    };
    time("dedup, node order", &q);
    // Same, as an f64 tree (coordinates relative to the centre in f64).
    {
        let d64: Vec<[f64; 3]> = keyed.iter().map(|k| std::array::from_fn(|i| k[i] as f64 / 1000.0 - c[i])).collect();
        let t = Instant::now();
        let tree64: ImmutableKdTree<f64, 3> = ImmutableKdTree::new_from_slice(&d64).unwrap();
        let build = t.elapsed().as_secs_f64() / d64.len() as f64 * 1e9;
        let q64: Vec<[f64; 3]> = qa.iter().map(|p| std::array::from_fn(|i| p[i] - c[i])).collect();
        let t = Instant::now();
        let mut acc = 0.0f64;
        for p in &q64 {
            acc += tree64.query(p).nearest_one::<SquaredEuclidean<f64>>().execute().distance.sqrt();
        }
        println!(
            "  {:<22} {:>6.0} ns/query (mean d {:.4} m), build {build:.0} ns/pt",
            "f64 tree, node order",
            t.elapsed().as_secs_f64() / q64.len() as f64 * 1e9,
            acc / q64.len() as f64
        );
    }
    // Bounded by max-dist (1 m): queries without a neighbour that close stop early.
    let t = Instant::now();
    let mut beyond = 0usize;
    for p in &q {
        let r = tree.query(p).nearest_n::<SquaredEuclidean<f32>>(std::num::NonZeroUsize::new(1).unwrap()).within(1.0).execute();
        beyond += usize::from(r.is_empty());
    }
    println!(
        "  {:<22} {:>6.0} ns/query ({:.1} % beyond 1 m)",
        "dedup, within 1 m",
        t.elapsed().as_secs_f64() / q.len() as f64 * 1e9,
        100.0 * beyond as f64 / q.len() as f64
    );
    let mut sorted = q.clone();
    sorted.sort_by_key(|p| ((p[0] * 2.0).floor() as i32, (p[1] * 2.0).floor() as i32, (p[2] * 2.0).floor() as i32));
    time("sorted by 0.5 m cells", &sorted);
    Ok(())
}
