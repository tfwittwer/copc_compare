//! `info` subcommand: header, VLRs, COPC info, per-level hierarchy statistics, and
//! optionally a timed full decode (`Fields::ALL` vs `Fields::Z`).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Result, ensure};
use copc_streaming::Fields;
use rayon::prelude::*;

use crate::input::{CopcFile, las_str};

pub fn run(path: &Path, bench: bool) -> Result<()> {
    let t0 = Instant::now();
    let f = CopcFile::open(path)?;
    let open_ms = t0.elapsed().as_secs_f64() * 1e3;

    let h = &f.raw_header;
    let lh = f.las_header();
    let fmt = lh.point_format();
    println!("{}", path.display());
    println!(
        "  {:.1} MB, header + full hierarchy loaded in {open_ms:.0} ms",
        f.file_size() as f64 / 1e6
    );
    println!(
        "  LAS {}.{}, point format {} (raw byte {:#04x}), record length {}, extra bytes {}",
        h.version.major,
        h.version.minor,
        fmt.to_u8()?,
        h.point_data_record_format,
        h.point_data_record_length,
        fmt.extra_bytes
    );
    println!("  points {}", lh.number_of_points());
    println!(
        "  software '{}', system '{}', global encoding {:#06x}",
        las_str(&h.generating_software),
        las_str(&h.system_identifier),
        h.global_encoding
    );
    println!("  scale  {} {} {}", h.x_scale_factor, h.y_scale_factor, h.z_scale_factor);
    println!("  offset {} {} {}", h.x_offset, h.y_offset, h.z_offset);
    println!(
        "  bounds x [{}, {}] y [{}, {}] z [{}, {}]",
        h.min_x, h.max_x, h.min_y, h.max_y, h.min_z, h.max_z
    );
    println!(
        "  header size {}, offset to point data {}, EVLRs start {}",
        h.header_size,
        h.offset_to_point_data,
        h.evlr.map_or(0, |e| e.start_of_first_evlr)
    );

    println!("VLRs ({}):", f.vlrs.len());
    for v in &f.vlrs {
        println!(
            "  {:<16} {:>5} {:>7} B  {}",
            las_str(&v.user_id),
            v.record_id,
            v.data.len(),
            las_str(&v.description)
        );
    }
    println!("EVLRs ({}):", f.evlrs.len());
    for v in &f.evlrs {
        println!(
            "  {:<16} {:>5} {:>7} B  {}",
            las_str(&v.user_id),
            v.record_id,
            v.data.len(),
            las_str(&v.description)
        );
    }
    // What copc-streaming's `las::Header` keeps, for the writer's VLR handling.
    let kept: Vec<String> =
        lh.vlrs().iter().map(|v| format!("{}/{}", v.user_id, v.record_id)).collect();
    println!(
        "  las::Header: {} VLRs [{}], {} EVLRs",
        kept.len(),
        kept.join(", "),
        lh.evlrs().len()
    );

    let laz = f.laz_vlr();
    let items: Vec<String> = laz
        .items()
        .iter()
        .map(|i| format!("{:?}(size {}, v{})", i.item_type(), i.size(), i.version()))
        .collect();
    println!(
        "LAZ: chunk size {}{}, items {}",
        laz.chunk_size(),
        if laz.uses_variable_size_chunks() { " (variable)" } else { "" },
        items.join(" ")
    );

    let ci = f.copc_info();
    println!(
        "COPC info: center {:?}, halfsize {}, spacing {}, root hierarchy page @{} ({} B = {} entries), gps time [{}, {}]",
        ci.center,
        ci.halfsize,
        ci.spacing,
        ci.root_hier_offset,
        ci.root_hier_size,
        ci.root_hier_size / 32,
        ci.gpstime_minimum,
        ci.gpstime_maximum
    );

    level_table(&f);

    let sum: u64 = f.data_entries().map(|e| u64::from(e.point_count)).sum();
    ensure!(
        sum == lh.number_of_points(),
        "hierarchy point counts sum to {sum}, header says {}",
        lh.number_of_points()
    );
    println!("check: hierarchy point counts sum to the header's point count ({sum})");

    if bench {
        bench_decode(&f, Fields::ALL, "ALL")?;
        bench_decode(&f, Fields::Z, "Z")?;
        // Base layer only (x, y, returns, channel): the floor for any decode.
        bench_decode(&f, Fields::empty(), "XY")?;
    }
    Ok(())
}

#[derive(Default)]
struct LevelStats {
    nodes: u64,
    empty: u64,
    points: u64,
    bytes: u64,
    max_points: u32,
}

fn level_table(f: &CopcFile) {
    let mut levels: BTreeMap<i32, LevelStats> = BTreeMap::new();
    for e in f.entries() {
        let s = levels.entry(e.key.level).or_default();
        if e.point_count == 0 {
            s.empty += 1;
        } else {
            s.nodes += 1;
            s.points += u64::from(e.point_count);
            s.bytes += u64::from(e.byte_size);
            s.max_points = s.max_points.max(e.point_count);
        }
    }
    let side0 = 2.0 * f.copc_info().halfsize;
    let rec = f.record_len() as f64;
    println!("Hierarchy ({} entries):", f.entries().len());
    println!(
        "  {:>5} {:>9} {:>6} {:>12} {:>9} {:>8} {:>6} {:>10} {:>10}",
        "level", "nodes", "empty", "points", "MB", "B/pt", "ratio", "max pts", "side m"
    );
    let mut tot = LevelStats::default();
    for (l, s) in &levels {
        let bpp = if s.points > 0 { s.bytes as f64 / s.points as f64 } else { 0.0 };
        println!(
            "  {:>5} {:>9} {:>6} {:>12} {:>9.1} {:>8.2} {:>6.2} {:>10} {:>10.3}",
            l,
            s.nodes,
            s.empty,
            s.points,
            s.bytes as f64 / 1e6,
            bpp,
            if bpp > 0.0 { rec / bpp } else { 0.0 },
            s.max_points,
            side0 / f64::from(1u32 << l)
        );
        tot.nodes += s.nodes;
        tot.empty += s.empty;
        tot.points += s.points;
        tot.bytes += s.bytes;
        tot.max_points = tot.max_points.max(s.max_points);
    }
    let bpp = tot.bytes as f64 / tot.points.max(1) as f64;
    println!(
        "  {:>5} {:>9} {:>6} {:>12} {:>9.1} {:>8.2} {:>6.2} {:>10}",
        "all",
        tot.nodes,
        tot.empty,
        tot.points,
        tot.bytes as f64 / 1e6,
        bpp,
        rec / bpp,
        tot.max_points
    );
}

/// Decode every node with `fields` on the current rayon pool and report pts/s.
fn bench_decode(f: &CopcFile, fields: Fields, name: &str) -> Result<()> {
    let entries: Vec<_> = f.data_entries().collect();
    let points = AtomicU64::new(0);
    let t = Instant::now();
    // The Z sum keeps the position decode from being optimised away and
    // includes the scale/offset step B needs anyway.
    let zsum: f64 = entries
        .par_iter()
        .map(|e| -> Result<f64> {
            let chunk = f.decode(e, fields)?;
            points.fetch_add(chunk.point_count() as u64, Ordering::Relaxed);
            Ok(chunk.positions().map_or(0.0, |p| p.map(|[_, _, z]| z).sum()))
        })
        .try_reduce(|| 0.0, |a, b| Ok(a + b))?;
    let secs = t.elapsed().as_secs_f64();
    let n = points.load(Ordering::Relaxed);
    println!(
        "decode Fields::{name:<3}: {n} points in {secs:.2} s = {:.1} M pts/s on {} threads ({:.2} M pts/s/thread), mean z {:.3}",
        n as f64 / secs / 1e6,
        rayon::current_num_threads(),
        n as f64 / secs / 1e6 / rayon::current_num_threads() as f64,
        zsum / n.max(1) as f64
    );
    Ok(())
}
