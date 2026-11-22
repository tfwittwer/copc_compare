//! Compare two COPC files node by node, independent of chunk order: same keys,
//! same point counts, same original records. Compressed chunks are compared
//! first (LAZ compression is deterministic); a chunk that differs is decoded.
//! Differences in the trailing i32 fields (the c2c output) are tallied per
//! field as a histogram of |difference| instead of failing.
//!
//! cargo run --release --example same_nodes -- X.copc.laz Y.copc.laz [N_FIELDS]

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail, ensure};
use copc_compare::input::CopcFile;
use copc_streaming::Fields;
use rayon::prelude::*;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() >= 3, "usage: X Y [N_FIELDS]");
    let (x, y) = (CopcFile::open(args[1].as_ref())?, CopcFile::open(args[2].as_ref())?);
    let n_fields: usize = args.get(3).map_or(Ok(1), |s| s.parse())?;
    ensure!(x.entries().len() == y.entries().len(), "{} vs {} hierarchy entries", x.entries().len(), y.entries().len());
    ensure!(x.record_len() == y.record_len(), "record lengths differ");
    let rec = x.record_len();
    let fields_at = rec - 4 * n_fields;
    let (same_bytes, decoded, differing_points) = (AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0));
    // Per field: |difference| -> count, and a few examples.
    let hist: Mutex<Vec<BTreeMap<i64, u64>>> = Mutex::new(vec![BTreeMap::new(); n_fields]);
    let examples: Mutex<Vec<String>> = Mutex::new(Vec::new());
    x.entries().par_iter().zip(y.entries()).try_for_each(|(ex, ey)| -> Result<()> {
        ensure!(ex.key == ey.key && ex.point_count == ey.point_count, "{:?} vs {:?}", ex, ey);
        if ex.point_count == 0 {
            return Ok(());
        }
        if x.compressed(ex) == y.compressed(ey) {
            same_bytes.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }
        decoded.fetch_add(1, Ordering::Relaxed);
        let (cx, cy) = (x.decode(ex, Fields::ALL)?, y.decode(ey, Fields::ALL)?);
        let mut local = vec![BTreeMap::new(); n_fields];
        for (i, (rx, ry)) in cx.cloud().raw_bytes().chunks_exact(rec).zip(cy.cloud().raw_bytes().chunks_exact(rec)).enumerate() {
            if rx[..fields_at] != ry[..fields_at] {
                bail!("node {:?} point {i}: original record differs", ex.key);
            }
            if rx == ry {
                continue;
            }
            differing_points.fetch_add(1, Ordering::Relaxed);
            for (f, hist) in local.iter_mut().enumerate() {
                let at = fields_at + 4 * f;
                let v = |r: &[u8]| i64::from(i32::from_le_bytes(r[at..at + 4].try_into().unwrap()));
                let d = (v(rx) - v(ry)).abs();
                if d != 0 {
                    *hist.entry(d).or_insert(0u64) += 1;
                    let mut e = examples.lock().unwrap();
                    if e.len() < 8 {
                        e.push(format!("node {:?} point {i} field {f}: {} vs {}", ex.key, v(rx), v(ry)));
                    }
                }
            }
        }
        let mut h = hist.lock().unwrap();
        for (h, l) in h.iter_mut().zip(local) {
            for (d, n) in l {
                *h.entry(d).or_insert(0) += n;
            }
        }
        Ok(())
    })?;
    println!(
        "{} points; {} byte-identical chunks, {} decoded; original records identical everywhere",
        x.las_header().number_of_points(),
        same_bytes.load(Ordering::Relaxed),
        decoded.load(Ordering::Relaxed)
    );
    println!("points with differing fields: {}", differing_points.load(Ordering::Relaxed));
    for (f, h) in hist.into_inner().unwrap().iter().enumerate() {
        println!("  field {f}: |diff| -> count {h:?}");
    }
    for e in examples.into_inner().unwrap() {
        println!("  e.g. {e}");
    }
    Ok(())
}
