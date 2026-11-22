//! Phase 1: the pass-through writer keeps A's octree and records, appends the new
//! fields, and produces COPC that independent readers accept.

mod common;

use copc_compare::extra_bytes::{DESCRIPTOR_SIZE, ScaledField};
use copc_compare::input::{CopcFile, las_str};
use copc_compare::pipeline::passthrough;
use copc_reader::{BoundsSelection, CopcReader, LodSelection};
use copc_streaming::VoxelKey;

use common::{assert_no_new_issues, assert_records_extended, cloud, convert, copc, fixture, scratch, validator};
use copc_compare::splice::Splice;

fn field(name: &str, scale: f64) -> ScaledField {
    ScaledField::new(name, "C2C dist m; >max = 5*max", scale).unwrap()
}

/// The descriptors of the output's Extra Bytes VLR.
fn descriptors(f: &CopcFile) -> Vec<Vec<u8>> {
    let v = f.vlrs.iter().find(|v| v.is("LASF_Spec", 4)).expect("Extra Bytes VLR");
    v.data.chunks_exact(DESCRIPTOR_SIZE).map(<[u8]>::to_vec).collect()
}

fn i64_at(d: &[u8], off: usize) -> i64 {
    i64::from_le_bytes(d[off..off + 8].try_into().unwrap())
}

/// Read every point back through copc-reader (roteiro) and return the extra bytes.
fn extra_bytes_via_copc_reader(path: &std::path::Path) -> Vec<Vec<u8>> {
    let mut r = CopcReader::from_path(path).expect("copc-reader opens the output");
    r.points(LodSelection::All, BoundsSelection::All)
        .unwrap()
        .map(|p| p.expect("copc-reader decodes every point").extra_bytes)
        .collect()
}

#[test]
fn constant_field() {
    let a = CopcFile::open(&fixture("untwine_reference.copc.laz")).unwrap();
    // Untwine's chunks (lazperf) have the layout the splice expects.
    let s = Splice::new(&a, 4).unwrap();
    assert!(a.data_entries().all(|e| s.fits(a.compressed(e))));
    let out = scratch("constant.copc.laz");
    let fields = [field("c2c_dist", 0.001)];
    passthrough(&a, &out, &fields, false, |_, c| vec![12; c.point_count()]).unwrap();

    assert_no_new_issues(&a.path, &out);
    let o = CopcFile::open(&out).unwrap();
    assert_records_extended(&a, &o, |_, _| vec![12]);

    // Header: same points, bounds and scales; longer records.
    let (ha, ho) = (&a.raw_header, &o.raw_header);
    assert_eq!(ho.point_data_record_length, ha.point_data_record_length + 4);
    assert_eq!(ho.number_of_point_records, 0, "legacy count zeroed");
    assert_eq!(ho.large_file, ha.large_file);
    assert_eq!(ho.large_file.unwrap().number_of_point_records, a.las_header().number_of_points());
    assert_eq!((ho.min_x, ho.max_x, ho.min_z, ho.max_z), (ha.min_x, ha.max_x, ha.min_z, ha.max_z));
    assert_eq!(ho.x_scale_factor, ha.x_scale_factor);
    assert_eq!(ho.guid, ha.guid);
    assert_eq!(las_str(&ho.generating_software), concat!("copc_compare ", env!("CARGO_PKG_VERSION")));

    // Paged hierarchy: the fixture reaches level 4, so level-3 subtrees get pages.
    let pointers = o.evlrs[0].data.chunks_exact(32).filter(|e| i32::from_le_bytes(e[28..32].try_into().unwrap()) == -1).count();
    let level3 = a.entries().iter().filter(|e| e.key.level == 3).count();
    assert_eq!(pointers, level3, "one child page per level-3 subtree");

    // A's other VLRs (here the WKT) survive verbatim.
    let wkt = |f: &CopcFile| f.vlrs.iter().find(|v| v.is("LASF_Projection", 2112)).unwrap().data.clone();
    assert_eq!(wkt(&a), wkt(&o));
    assert!(o.vlrs[0].is("copc", 1) && o.vlrs[1].is("laszip encoded", 22204));

    // Descriptor: i32, scale 0.001, min = max = 12.
    let d = descriptors(&o);
    assert_eq!(d.len(), 1);
    assert_eq!(las_str(&d[0][4..36]), "c2c_dist");
    assert_eq!(d[0][2], 6);
    assert_eq!(d[0][3], 8 | 2 | 4);
    assert_eq!(f64::from_le_bytes(d[0][112..120].try_into().unwrap()), 0.001);
    assert_eq!((i64_at(&d[0], 64), i64_at(&d[0], 88)), (12, 12));

    let eb = extra_bytes_via_copc_reader(&out);
    assert_eq!(eb.len() as u64, a.las_header().number_of_points());
    assert!(eb.iter().all(|b| b[..] == 12i32.to_le_bytes()));
}

#[test]
fn reencode_without_fields_is_lossless() {
    let a = CopcFile::open(&fixture("untwine_reference.copc.laz")).unwrap();
    let out = scratch("nofields.copc.laz");
    passthrough(&a, &out, &[], true, |_, _| Vec::new()).unwrap();
    assert_no_new_issues(&a.path, &out);
    let o = CopcFile::open(&out).unwrap();
    assert_records_extended(&a, &o, |_, _| Vec::new());
    assert!(descriptors(&o).is_empty());
}

/// A deterministic, varied test value per point and field.
fn value(k: &VoxelKey, i: usize, field: usize) -> i32 {
    let h = (k.level * 1_000_003 + k.x * 7919 + k.y * 104_729 + k.z * 31) as i64;
    ((h + i as i64 * 37 + field as i64 * 1_000_000) % 20_000 - 10_000) as i32
}

#[test]
fn existing_extra_bytes_and_offset_evlr() {
    // A converted with copc-converter: has its own extra bytes, and a temporal
    // index EVLR, which holds absolute offsets and must be dropped.
    let a_path = scratch("eb_input.copc.laz");
    convert(&[fixture("extra_bytes_input.laz")], &a_path, Some(1000));
    let a = CopcFile::open(&a_path).unwrap();
    let a_desc = descriptors(&a);
    assert!(!a_desc.is_empty(), "fixture should have extra bytes descriptors");
    assert!(a.evlrs.iter().any(|v| !v.is("copc", 1000)), "fixture should have a temporal index EVLR");

    let out = scratch("eb_output.copc.laz");
    let fields = [field("c2c_dist", 0.001), field("c2c_dz", 0.01)];
    passthrough(&a, &out, &fields, false, |e, c| {
        (0..c.point_count()).flat_map(|i| [value(&e.key, i, 0), value(&e.key, i, 1)]).collect()
    })
    .unwrap();

    validator::assert_valid_copc(&out);
    let o = CopcFile::open(&out).unwrap();
    assert_records_extended(&a, &o, |k, i| vec![value(k, i, 0), value(k, i, 1)]);

    // A's descriptors come first and unchanged, then ours.
    let d = descriptors(&o);
    assert_eq!(d.len(), a_desc.len() + 2);
    assert_eq!(&d[..a_desc.len()], &a_desc[..]);
    assert_eq!(las_str(&d[a_desc.len() + 1][4..36]), "c2c_dz");
    assert_eq!(f64::from_le_bytes(d[a_desc.len() + 1][112..120].try_into().unwrap()), 0.01);

    // Only the rebuilt hierarchy is left.
    assert_eq!(o.evlrs.len(), 1);
    assert!(o.evlrs[0].is("copc", 1000));

    let eb = extra_bytes_via_copc_reader(&out);
    assert_eq!(eb.len() as u64, a.las_header().number_of_points());
    assert!(eb.iter().all(|b| b.len() == a.las_header().point_format().extra_bytes as usize + 8));
}

/// `n` fields of varied values, as the passthrough's `values` closure.
fn values_for(n: usize) -> impl Fn(&copc_streaming::HierarchyEntry, &copc_streaming::Chunk) -> Vec<i32> + Sync {
    move |e, c| (0..c.point_count()).flat_map(|i| (0..n).map(move |f| value(&e.key, i, f))).collect::<Vec<_>>()
}

#[test]
fn splice_equals_reencode() {
    // Random scanner channels on every point: the LAZ contexts switch all the
    // time, also back to channels seen before. A's chunks are laz-rs's (via
    // copc-converter), so a full re-encode reproduces them and splicing must
    // give byte-identical files, here and on top of existing extra bytes.
    let pts = cloud(7, 40_000);
    for format in [6, 7, 8] {
        let a_path = copc(&format!("splice_pf{format}"), format, [0.001; 3], [0.0; 3], &pts);
        let a = CopcFile::open(&a_path).unwrap();
        let mut a2_path = None;
        for n in [1, 3] {
            let fields: Vec<ScaledField> = (0..n).map(|i| field(&format!("f{i}"), 0.001)).collect();
            let (sp, re) = (scratch(&format!("splice_pf{format}_{n}.copc.laz")), scratch(&format!("reenc_pf{format}_{n}.copc.laz")));
            passthrough(&a, &sp, &fields, false, values_for(n)).unwrap();
            passthrough(&a, &re, &fields, true, values_for(n)).unwrap();
            let o = CopcFile::open(&sp).unwrap();
            let s = Splice::new(&a, 4 * n).unwrap();
            assert!(a.data_entries().all(|e| s.fits(a.compressed(e))), "PF{format}: every chunk spliced");
            assert!(std::fs::read(&sp).unwrap() == std::fs::read(&re).unwrap(), "PF{format}, {n} fields: files differ");
            assert_records_extended(&a, &o, |k, i| (0..n).map(|f| value(k, i, f)).collect());
            a2_path = Some(sp);
        }

        // A with 12 extra bytes of its own (laz-rs encoded): 2 more fields.
        let a2 = CopcFile::open(a2_path.as_ref().unwrap()).unwrap();
        Splice::new(&a2, 8).unwrap();
        let fields = [field("g0", 0.01), field("g1", 0.01)];
        let (sp, re) = (scratch(&format!("splice2_pf{format}.copc.laz")), scratch(&format!("reenc2_pf{format}.copc.laz")));
        let shifted = |e: &copc_streaming::HierarchyEntry, c: &copc_streaming::Chunk| {
            (0..c.point_count()).flat_map(|i| [value(&e.key, i, 5), value(&e.key, i, 6)]).collect()
        };
        passthrough(&a2, &sp, &fields, false, shifted).unwrap();
        passthrough(&a2, &re, &fields, true, shifted).unwrap();
        assert!(std::fs::read(&sp).unwrap() == std::fs::read(&re).unwrap(), "PF{format} on extra bytes: files differ");
        validator::assert_valid_copc(&sp);
        let o = CopcFile::open(&sp).unwrap();
        assert_records_extended(&a2, &o, |k, i| vec![value(k, i, 5), value(k, i, 6)]);
    }
}
