//! Independent spot check of a c2c output too large for the full validator or
//! CloudCompare: in a few boxes, read the output's points and the reference
//! points within box + max-dist through copc-reader (not the code under test),
//! recompute every nearest distance by brute force over a bucket grid, and
//! compare with `c2c_dist`.
//!
//! cargo run --release --example spot_check -- OUTPUT REFERENCE MAX_DIST BOX_SIDE [N_BOXES]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use copc_compare::input::{CopcFile, las_str};
use copc_core::Bounds;
use copc_reader::{BoundsSelection, CopcReader, LodSelection};

fn read(path: &Path, b: Bounds) -> Result<Vec<([f64; 3], Vec<u8>)>> {
    let mut r = CopcReader::from_path(path)?;
    let inside = |p: &[f64; 3]| {
        p[0] >= b.min.0 && p[0] <= b.max.0 && p[1] >= b.min.1 && p[1] <= b.max.1 && p[2] >= b.min.2 && p[2] <= b.max.2
    };
    let mut out = Vec::new();
    for p in r.points(LodSelection::All, BoundsSelection::Within(b))? {
        let p = p?;
        let xyz = [p.x, p.y, p.z];
        if inside(&xyz) {
            out.push((xyz, p.extra_bytes));
        }
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() >= 5, "usage: OUTPUT REFERENCE MAX_DIST BOX_SIDE [N_BOXES]");
    let (out_path, ref_path) = (Path::new(&args[1]), Path::new(&args[2]));
    let max_dist: f64 = args[3].parse()?;
    let side: f64 = args[4].parse()?;
    let n_boxes: usize = args.get(5).map_or(Ok(5), |s| s.parse())?;

    // Where c2c_dist sits in the extra bytes, and its scale.
    let out = CopcFile::open(out_path)?;
    let eb = out.vlrs.iter().find(|v| v.is("LASF_Spec", 4)).context("no Extra Bytes VLR")?;
    let descs: Vec<&[u8]> = eb.data.chunks_exact(192).collect();
    let ours = descs.iter().filter(|d| las_str(&d[4..36]).starts_with("c2c_")).count();
    let first = descs.iter().position(|d| las_str(&d[4..36]) == "c2c_dist").context("no c2c_dist")?;
    ensure!(first == descs.len() - ours, "c2c fields expected last");
    let scale = f64::from_le_bytes(descs[first][112..120].try_into()?);
    let cap = (5.0 * max_dist / scale).round() as i64;

    // Boxes centred on evenly spaced nodes (in key order, so spread over the
    // area) of the level with the most nodes.
    let mut per_level: HashMap<i32, usize> = HashMap::new();
    for e in out.data_entries() {
        *per_level.entry(e.key.level).or_default() += 1;
    }
    let level = per_level.iter().max_by_key(|&(l, n)| (*n, *l)).map(|(l, _)| *l).unwrap();
    let nodes: Vec<_> = out.data_entries().filter(|e| e.key.level == level).collect();
    let root = out.root_bounds();
    let (mut total, mut beyond, mut worst) = (0usize, 0usize, 0i64);
    for i in 0..n_boxes {
        let e = nodes[i * nodes.len() / n_boxes];
        let nb = e.key.bounds(&root);
        let c = [0.5 * (nb.min[0] + nb.max[0]), 0.5 * (nb.min[1] + nb.max[1])];
        let a_box = Bounds::new((c[0] - side / 2.0, c[1] - side / 2.0, -1e4), (c[0] + side / 2.0, c[1] + side / 2.0, 1e4));
        let b_box = Bounds::new(
            (a_box.min.0 - max_dist, a_box.min.1 - max_dist, -1e4),
            (a_box.max.0 + max_dist, a_box.max.1 + max_dist, 1e4),
        );
        let a = read(out_path, a_box)?;
        let b = read(ref_path, b_box)?;

        // Exact nearest within max_dist in two steps. Fine 5 cm cells searched
        // shell by shell (after shell s nothing unvisited is closer than
        // s × FINE) settle almost every point within 4 shells; the rest are
        // brute-forced over the 27 max_dist cells around them, which hold every
        // point within max_dist.
        const FINE: f64 = 0.05;
        const FINE_SHELLS: i64 = 4;
        let cell_at = |p: &[f64; 3], c: f64| p.map(|v| (v / c).floor() as i64);
        let mut fine: HashMap<[i64; 3], Vec<[f64; 3]>> = HashMap::new();
        let mut coarse: HashMap<[i64; 3], Vec<[f64; 3]>> = HashMap::new();
        for (p, _) in &b {
            fine.entry(cell_at(p, FINE)).or_default().push(*p);
            coarse.entry(cell_at(p, max_dist)).or_default().push(*p);
        }
        let d = |p: &[f64; 3], q: &[f64; 3]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
        let mut box_worst = 0;
        for (p, eb) in &a {
            let o = eb.len() - 4 * ours;
            let v = i64::from(i32::from_le_bytes(eb[o..o + 4].try_into()?));
            let k = cell_at(p, FINE);
            let mut best = f64::INFINITY;
            let mut settled = false;
            for sh in 0..=FINE_SHELLS {
                for dx in -sh..=sh {
                    for dy in -sh..=sh {
                        for dz in -sh..=sh {
                            if dx.abs().max(dy.abs()).max(dz.abs()) == sh {
                                for q in fine.get(&[k[0] + dx, k[1] + dy, k[2] + dz]).into_iter().flatten() {
                                    best = best.min(d(p, q));
                                }
                            }
                        }
                    }
                }
                if best <= sh as f64 * FINE {
                    settled = true;
                    break;
                }
            }
            if !settled {
                let k = cell_at(p, max_dist);
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            for q in coarse.get(&[k[0] + dx, k[1] + dy, k[2] + dz]).into_iter().flatten() {
                                best = best.min(d(p, q));
                            }
                        }
                    }
                }
            }
            if (best - max_dist).abs() < 1e-9 {
                continue;
            }
            let want = if best <= max_dist { (best / scale).round() as i64 } else { cap };
            beyond += usize::from(want == cap);
            let diff = (v - want).abs();
            ensure!(diff <= 1, "point {p:?}: c2c_dist {v}, brute force {want} (scale {scale})");
            box_worst = box_worst.max(diff);
        }
        worst = worst.max(box_worst);
        total += a.len();
        println!(
            "box {i} at ({:.1}, {:.1}), {side} m: {} points vs {} reference points, max diff {box_worst} units",
            c[0],
            c[1],
            a.len(),
            b.len()
        );
    }
    println!("all {total} points match brute force (max diff {worst} × {scale} m, {beyond} beyond max-dist)");
    Ok(())
}
