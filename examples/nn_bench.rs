//! Single-thread kiddo cost: build an f32 ImmutableKdTree on N points of a
//! 2.5D-like cloud (a wavy surface over 200 × 200 m) and query N nearby points.
use std::time::Instant;
use kiddo::{ImmutableKdTree, SquaredEuclidean};

fn main() {
    let n: usize = std::env::args().nth(1).map_or(10_000_000, |s| s.parse().unwrap());
    let mut s = 1u64;
    let mut r = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s >> 11) as f32 / (1u64 << 53) as f32 };
    let surf = |x: f32, y: f32| 2.0 * (x / 7.0).sin() + (y / 5.0).cos();
    let pts: Vec<[f32; 3]> = (0..n).map(|_| { let (x, y) = (200.0 * r() - 100.0, 200.0 * r() - 100.0); [x, y, surf(x, y)] }).collect();
    let qs: Vec<[f32; 3]> = (0..n).map(|_| { let (x, y) = (200.0 * r() - 100.0, 200.0 * r() - 100.0); [x, y, surf(x, y) + 0.05 * r()] }).collect();
    let t = Instant::now();
    let tree: ImmutableKdTree<f32, 3> = ImmutableKdTree::new_from_slice(&pts).unwrap();
    let build = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut acc = 0.0f32;
    for q in &qs { acc += tree.query(q).nearest_one::<SquaredEuclidean<f32>>().execute().distance; }
    let query = t.elapsed().as_secs_f64();
    // Same queries in tree-friendly (sorted) order, as consecutive scan points are near.
    let mut sorted = qs.clone();
    sorted.sort_by(|a, b| ((a[0] / 2.0) as i32, a[1]).partial_cmp(&((b[0] / 2.0) as i32, b[1])).unwrap());
    let t = Instant::now();
    for q in &sorted { acc += tree.query(q).nearest_one::<SquaredEuclidean<f32>>().execute().distance; }
    let query_sorted = t.elapsed().as_secs_f64();
    println!("n {n}: build {:.2} s ({:.0} ns/pt), query random order {:.0} ns, spatially sorted {:.0} ns (acc {acc:.1})",
        build, build / n as f64 * 1e9, query / n as f64 * 1e9, query_sorted / n as f64 * 1e9);
}
