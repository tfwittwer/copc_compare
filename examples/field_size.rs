//! Phase 1 measurement: compressed bytes per point that one extra field costs, by
//! storage type and value range. Uses real nodes of FILE with random values.
//!
//! cargo run --release --example field_size -- FILE [EVERY_NTH_NODE]

use anyhow::Result;
use copc_compare::input::CopcFile;
use copc_compare::write::compress;
use copc_streaming::Fields;
use laz::LazVlrBuilder;

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let a = CopcFile::open(args[1].as_ref())?;
    let every: usize = args.get(2).map_or(Ok(10), |s| s.parse())?;
    let fmt = a.las_header().point_format().to_u8()?;
    let base_extra = a.las_header().point_format().extra_bytes;
    let rec = a.record_len();
    let vlr = |n: u16| LazVlrBuilder::default().with_point_format(fmt, base_extra + n).unwrap().with_variable_chunk_size().build();

    // (label, bytes per value, max raw value): 0–20 mm at 1 mm and 0.1 mm scale.
    let variants: [(&str, u16, u64); 5] =
        [("u8  0..20", 1, 20), ("i16 0..20", 2, 20), ("i32 0..20", 4, 20), ("i16 0..200", 2, 200), ("i32 0..200", 4, 200)];
    let mut base = 0usize;
    let mut sizes = [0usize; 5];
    let mut points = 0usize;
    for e in a.data_entries().step_by(every) {
        let c = a.decode(e, Fields::ALL)?;
        let raw = c.cloud().raw_bytes();
        let n = c.point_count();
        points += n;
        base += compress(raw, &vlr(0))?.len();
        for (v, (_, width, max)) in variants.iter().enumerate() {
            let mut seed = e.offset;
            let mut buf = Vec::with_capacity(n * (rec + *width as usize));
            for r in raw.chunks_exact(rec) {
                buf.extend_from_slice(r);
                let x = splitmix(&mut seed) % (max + 1);
                buf.extend_from_slice(&x.to_le_bytes()[..*width as usize]);
            }
            sizes[v] += compress(&buf, &vlr(*width))?.len();
        }
    }
    println!("{points} points, base {:.3} B/pt", base as f64 / points as f64);
    for ((label, _, max), s) in variants.iter().zip(sizes) {
        let added = (s - base) as f64 / points as f64;
        println!(
            "  {label:<11} +{added:.3} B/pt = {:.2} bits (entropy {:.2})",
            added * 8.0,
            ((max + 1) as f64).log2()
        );
    }
    Ok(())
}
