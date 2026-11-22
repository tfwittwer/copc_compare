//! Shared test helpers: fixtures, conversion to COPC, record comparison.
#![allow(dead_code)]

pub mod validator;

use std::path::{Path, PathBuf};

use copc_compare::input::CopcFile;
use copc_converter::{NodeStorage, Pipeline, PipelineConfig, TempCompression};
use copc_streaming::Fields;
use las::point::{Classification, ScanDirection};

/// A file from copc-converter's test data: `$COPC_CONVERTER_DATA`, else a
/// checkout at `../tools/copc-converter`.
pub fn fixture(name: &str) -> PathBuf {
    let dir = std::env::var_os("COPC_CONVERTER_DATA").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/copc-converter/tests/data"),
        PathBuf::from,
    );
    let p = dir.join(name);
    assert!(
        p.exists(),
        "test fixture {} not found: set COPC_CONVERTER_DATA to copc-converter's tests/data (see README)",
        p.display()
    );
    p
}

/// A fresh scratch path for this test.
pub fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("copc_compare_tests");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join(name);
    let _ = std::fs::remove_file(&p);
    p
}

/// Convert LAS/LAZ files to COPC with copc-converter.
pub fn convert(inputs: &[PathBuf], output: &Path, temporal_index: Option<u32>) {
    let tmp = scratch(&format!("{}.tmpdir", output.file_name().unwrap().to_string_lossy()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let config = PipelineConfig {
        memory_budget: 1 << 30,
        temp_dir: Some(tmp.clone()),
        temporal_index,
        progress: None,
        chunk_target_override: None,
        temp_compression: TempCompression::None,
        node_storage: NodeStorage::Files,
    };
    Pipeline::scan(inputs, config)
        .and_then(|p| p.validate())
        .and_then(|p| p.distribute())
        .and_then(|p| p.build())
        .and_then(|p| p.write(output))
        .unwrap_or_else(|e| panic!("converting {inputs:?}: {e}"));
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Assert that every node of `out` holds `a`'s records, in the same order, each
/// followed by the 4-byte LE values `expected(node_key, point_index)`.
pub fn assert_records_extended(
    a: &CopcFile,
    out: &CopcFile,
    expected: impl Fn(&copc_streaming::VoxelKey, usize) -> Vec<i32>,
) {
    assert_eq!(a.entries().len(), out.entries().len(), "hierarchy entry count");
    for (ea, eo) in a.entries().iter().zip(out.entries()) {
        assert_eq!(ea.key, eo.key);
        assert_eq!(ea.point_count, eo.point_count, "node {:?}", ea.key);
        if ea.point_count == 0 {
            assert_eq!((eo.offset, eo.byte_size), (0, 0), "empty node {:?}", eo.key);
            continue;
        }
        let ca = a.decode(ea, Fields::ALL).unwrap();
        let co = out.decode(eo, Fields::ALL).unwrap();
        let (ra, ro) = (a.record_len(), out.record_len());
        for (i, (pa, po)) in ca
            .cloud()
            .raw_bytes()
            .chunks_exact(ra)
            .zip(co.cloud().raw_bytes().chunks_exact(ro))
            .enumerate()
        {
            assert_eq!(pa, &po[..ra], "node {:?} point {i}: original record changed", ea.key);
            let want: Vec<u8> = expected(&ea.key, i).iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(&po[ra..], &want[..], "node {:?} point {i}: appended fields", ea.key);
        }
    }
}

/// Validator issues without their "(first: ...)" detail, which depends on chunk order.
fn issue_kinds(path: &Path) -> Vec<String> {
    let mut v: Vec<String> = validator::validate_copc(path)
        .into_iter()
        .map(|i| i.split(" (first:").next().unwrap().to_owned())
        .collect();
    v.sort();
    v
}

/// The output may only have validator issues that the input already had (e.g.
/// points a rounding error outside their node box, which we can't move). The
/// legacy point count issue the writer fixes.
pub fn assert_no_new_issues(input: &Path, output: &Path) {
    let before = issue_kinds(input);
    let new: Vec<String> = issue_kinds(output).into_iter().filter(|i| !before.contains(i)).collect();
    assert!(new.is_empty(), "{} has new COPC issues:\n  - {}", output.display(), new.join("\n  - "));
    let after = issue_kinds(output);
    assert!(
        !after.iter().any(|i| i.starts_with("legacy point counts")),
        "legacy point counts not zeroed"
    );
}

/// Write `points` to an uncompressed LAS 1.4 PF6 file with the given scale and
/// offset (same on all axes). GPS time is the point index.
pub fn write_las(path: &Path, scale: f64, offset: [f64; 3], points: &[[f64; 3]]) {
    let mut b = las::Builder::from((1, 4));
    b.point_format = las::point::Format::new(6).unwrap();
    b.transforms = las::Vector {
        x: las::Transform { scale, offset: offset[0] },
        y: las::Transform { scale, offset: offset[1] },
        z: las::Transform { scale, offset: offset[2] },
    };
    let mut w = las::Writer::from_path(path, b.into_header().unwrap()).unwrap();
    for (i, p) in points.iter().enumerate() {
        w.write_point(las::Point {
            x: p[0],
            y: p[1],
            z: p[2],
            return_number: 1,
            number_of_returns: 1,
            gps_time: Some(i as f64),
            ..Default::default()
        })
        .unwrap();
    }
    w.close().unwrap();
}

/// Write full `las::Point`s to an uncompressed LAS 1.4 file of point format
/// `format`, with the given scale and offset per axis.
pub fn write_las_points(path: &Path, format: u8, scale: [f64; 3], offset: [f64; 3], points: &[las::Point]) {
    let mut b = las::Builder::from((1, 4));
    b.point_format = las::point::Format::new(format).unwrap();
    let t = |a: usize| las::Transform { scale: scale[a], offset: offset[a] };
    b.transforms = las::Vector { x: t(0), y: t(1), z: t(2) };
    let mut w = las::Writer::from_path(path, b.into_header().unwrap()).unwrap();
    for p in points {
        w.write_point(p.clone()).unwrap();
    }
    w.close().unwrap();
}

pub fn rng(mut s: u64) -> impl FnMut() -> u64 {
    move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// A point on the mm grid in a 100 × 100 × 10 m box, every field random
/// (including the scanner channel, so the LAZ contexts switch often).
pub fn point(i: usize, r: &mut impl FnMut() -> u64) -> las::Point {
    let mut mm = |range: u64| (r() % range) as f64 * 0.001;
    let (x, y, z) = (1000.0 + mm(100_000), 2000.0 + mm(100_000), mm(10_000));
    las::Point {
        x,
        y,
        z,
        intensity: r() as u16,
        return_number: 1 + (r() % 3) as u8,
        number_of_returns: 3,
        classification: Classification::new((r() % 10) as u8).unwrap(),
        scanner_channel: (r() % 4) as u8,
        is_synthetic: r().is_multiple_of(3),
        is_key_point: r().is_multiple_of(11),
        is_withheld: r().is_multiple_of(7),
        is_overlap: r().is_multiple_of(5),
        scan_direction: if r().is_multiple_of(2) { ScanDirection::RightToLeft } else { ScanDirection::LeftToRight },
        is_edge_of_flight_line: r().is_multiple_of(13),
        scan_angle: ((r() % 2000) as f32 - 1000.0) * 0.006,
        user_data: r() as u8,
        point_source_id: (r() % 100) as u16,
        gps_time: Some(i as f64 * 0.5),
        color: Some(las::Color { red: r() as u16, green: r() as u16, blue: r() as u16 }),
        nir: Some(r() as u16),
        ..Default::default()
    }
}

pub fn cloud(seed: u64, n: usize) -> Vec<las::Point> {
    let mut r = rng(seed);
    (0..n).map(|i| point(i, &mut r)).collect()
}

/// Write `points` as LAS of point format `format` (dropping colour or NIR it
/// lacks) and convert to COPC.
pub fn copc(name: &str, format: u8, scale: [f64; 3], offset: [f64; 3], points: &[las::Point]) -> PathBuf {
    let las = scratch(&format!("{name}.las"));
    let mut points = points.to_vec();
    for p in &mut points {
        if format < 7 {
            p.color = None;
        }
        if format < 8 {
            p.nir = None;
        }
    }
    write_las_points(&las, format, scale, offset, &points);
    let out = scratch(&format!("{name}.copc.laz"));
    convert(&[las], &out, None);
    out
}

/// Points of a COPC file read through copc-reader: position and extra bytes.
pub fn read_back(path: &Path) -> Vec<([f64; 3], Vec<u8>)> {
    let mut r = copc_reader::CopcReader::from_path(path).unwrap();
    r.points(copc_reader::LodSelection::All, copc_reader::BoundsSelection::All)
        .unwrap()
        .map(|p| {
            let p = p.unwrap();
            ([p.x, p.y, p.z], p.extra_bytes)
        })
        .collect()
}

/// Brute-force nearest neighbour within `max_dist`, bucketed by `max_dist` cells.
/// Independent of the code under test.
pub struct Oracle {
    cell: f64,
    buckets: std::collections::HashMap<[i64; 3], Vec<[f64; 3]>>,
}

impl Oracle {
    pub fn new(points: impl IntoIterator<Item = [f64; 3]>, max_dist: f64) -> Self {
        let mut buckets: std::collections::HashMap<[i64; 3], Vec<[f64; 3]>> = Default::default();
        for p in points {
            buckets.entry(p.map(|v| (v / max_dist).floor() as i64)).or_default().push(p);
        }
        Self { cell: max_dist, buckets }
    }

    /// Distance to the nearest point if it is within `max_dist`.
    pub fn nearest(&self, p: [f64; 3]) -> Option<f64> {
        let c = p.map(|v| (v / self.cell).floor() as i64);
        let mut best = f64::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(v) = self.buckets.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                        for q in v {
                            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt();
                            best = best.min(d);
                        }
                    }
                }
            }
        }
        (best <= self.cell).then_some(best)
    }
}
