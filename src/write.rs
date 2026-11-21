//! COPC writer that keeps the compared file's octree: same keys, same point counts,
//! longer records. Glue between `las::raw` (header, VLRs), `laz` (chunks, chunk
//! table) and `copc-core` (CopcInfo, hierarchy entries).
//!
//! File layout:
//! ```text
//! header (375)                          A's raw header, patched
//! VLR copc/1 (payload at 429)           CopcInfo, root hierarchy offset patched at the end
//! VLR laszip encoded/22204              rebuilt for the longer records
//! A's other VLRs, verbatim
//! VLR LASF_Spec/4                       A's descriptors + ours
//! i64 offset of the chunk table         patched at the end
//! chunks, in arrival order
//! LAZ chunk table
//! EVLR copc/1000                        hierarchy pages, A's keys (root page first)
//! A's LASF_Projection / LASF_Spec EVLRs, verbatim
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use copc_streaming::VoxelKey;
use laz::laszip::{ChunkTable, ChunkTableEntry};
use laz::record::{LayeredPointRecordCompressor, RecordCompressor};
use laz::{LazVlr, LazVlrBuilder};

use crate::extra_bytes::{self, ScaledField};
use crate::input::{CopcFile, RawVlr, key_tuple, las_str};

const HEADER_SIZE: u16 = 375;
const VLR_HEADER_SIZE: u64 = RawVlr::HEADER_SIZE;
const EVLR_HEADER_SIZE: u64 = RawVlr::EVLR_HEADER_SIZE;
/// COPC requires the CopcInfo payload right after the header and the first VLR header.
const COPC_INFO_POS: u64 = HEADER_SIZE as u64 + VLR_HEADER_SIZE;
const COPC_INFO_SIZE: usize = 160;

/// Per-node output layout: A's record followed by the new i32 fields.
#[derive(Clone)]
pub struct RecordLayout {
    pub laz_vlr: LazVlr,
    pub in_len: usize,
    pub n_fields: usize,
}

impl RecordLayout {
    pub fn out_len(&self) -> usize {
        self.in_len + self.n_fields * ScaledField::BYTES
    }

    /// Append `values` (point-major, `n_fields` per point) to A's raw records and
    /// LAZ-compress the result as one COPC chunk.
    pub fn encode(&self, raw: &[u8], values: &[i32]) -> Result<Vec<u8>> {
        ensure!(raw.len().is_multiple_of(self.in_len), "raw records not a multiple of {}", self.in_len);
        let n = raw.len() / self.in_len;
        ensure!(
            values.len() == n * self.n_fields,
            "{} values for {n} points × {} fields",
            values.len(),
            self.n_fields
        );
        let mut buf = Vec::with_capacity(n * self.out_len());
        if self.n_fields == 0 {
            buf.extend_from_slice(raw);
        } else {
            for (rec, vals) in raw.chunks_exact(self.in_len).zip(values.chunks_exact(self.n_fields)) {
                buf.extend_from_slice(rec);
                for v in vals {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        compress(&buf, &self.laz_vlr)
    }
}

/// LAZ-compress raw records as one independent chunk.
pub fn compress(records: &[u8], laz_vlr: &LazVlr) -> Result<Vec<u8>> {
    let mut c = LayeredPointRecordCompressor::new(Vec::with_capacity(records.len() / 4));
    c.set_fields_from(laz_vlr.items())?;
    c.compress_many(records)?;
    c.done()?;
    Ok(c.into_inner())
}

#[derive(Clone, Copy)]
struct Written {
    offset: u64,
    byte_size: u32,
    point_count: u32,
}

pub struct CopcWriter {
    path: PathBuf,
    w: BufWriter<File>,
    /// Current write position.
    pos: u64,
    header: las::raw::Header,
    copc_info: copc_core::CopcInfo,
    layout: RecordLayout,
    chunk_table: ChunkTable,
    written: HashMap<(i32, i32, i32, i32), Written>,
    kept_evlrs: Vec<RawVlr>,
    /// Our descriptors and the file offset of the first, for patching stats.
    descriptors: Vec<[u8; extra_bytes::DESCRIPTOR_SIZE]>,
    descriptors_pos: u64,
}

#[derive(Debug)]
pub struct WriteSummary {
    pub bytes: u64,
    pub chunks: usize,
    pub points: u64,
    pub point_data_bytes: u64,
}

fn fixed<const N: usize>(s: &str) -> [u8; N] {
    let mut out = [0u8; N];
    let n = s.len().min(N);
    out[..n].copy_from_slice(&s.as_bytes()[..n]);
    out
}

fn new_vlr(template: Option<&RawVlr>, user_id: &str, record_id: u16, description: &str, data: Vec<u8>) -> Result<RawVlr> {
    ensure!(data.len() <= usize::from(u16::MAX), "VLR {user_id}/{record_id} too large");
    Ok(RawVlr {
        reserved: 0,
        user_id: fixed(user_id),
        record_id,
        description: template.map_or_else(|| fixed(description), |t| t.description),
        data,
    })
}

/// (day of year, year) in UTC, for the header's file creation date.
fn today() -> (u16, u16) {
    let mut days = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() / 86_400);
    let mut year = 1970u16;
    loop {
        let leap = (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400);
        let len = if leap { 366 } else { 365 };
        if days < len {
            return (days as u16 + 1, year);
        }
        days -= len;
        year += 1;
    }
}

impl CopcWriter {
    /// Create `path` and write the header, VLRs and the chunk table placeholder.
    pub fn create(path: &Path, a: &CopcFile, fields: &[ScaledField]) -> Result<Self> {
        let fmt = a.las_header().point_format();
        let fmt_id = fmt.to_u8()?;
        let a_extra = usize::from(fmt.extra_bytes);
        let new_extra = a_extra + fields.len() * ScaledField::BYTES;
        let in_len = a.record_len();
        let out_len = u16::try_from(in_len + fields.len() * ScaledField::BYTES)
            .context("output record length exceeds 65535 bytes")?;
        ensure!(
            a.raw_header.header_size == HEADER_SIZE && a.raw_header.padding.is_empty(),
            "compared file has a {}-byte header; COPC requires {HEADER_SIZE}",
            a.raw_header.header_size
        );

        let laz_vlr = LazVlrBuilder::default()
            .with_point_format(fmt_id, u16::try_from(new_extra)?)?
            .with_variable_chunk_size()
            .build();
        let mut laz_bytes = Vec::new();
        laz_vlr.write_to(&mut laz_bytes)?;

        let find = |u: &str, r: u16| a.vlrs.iter().find(|v| v.is(u, r));
        let a_eb = find(extra_bytes::USER_ID, extra_bytes::RECORD_ID);
        let eb_payload = extra_bytes::output_descriptors(a_eb.map(|v| &v.data[..]), a_extra, fields)?;
        let our_descriptors_offset = eb_payload.len() - fields.len() * extra_bytes::DESCRIPTOR_SIZE;

        let mut vlrs = vec![
            new_vlr(find("copc", 1), "copc", 1, "copc info", vec![0; COPC_INFO_SIZE])?,
            new_vlr(find("laszip encoded", 22204), "laszip encoded", 22204, "laz variable chunks", laz_bytes)?,
        ];
        vlrs.extend(
            a.vlrs
                .iter()
                .filter(|v| {
                    !v.is("copc", 1)
                        && !v.is("laszip encoded", 22204)
                        && !v.is(extra_bytes::USER_ID, extra_bytes::RECORD_ID)
                })
                .cloned(),
        );
        vlrs.push(new_vlr(a_eb, extra_bytes::USER_ID, extra_bytes::RECORD_ID, "Extra Bytes", eb_payload)?);

        // EVLRs without file offsets survive; the hierarchy is rebuilt; the rest
        // (e.g. a temporal index with absolute offsets) would break, so drop it.
        let mut kept_evlrs = Vec::new();
        for v in &a.evlrs {
            let user = las_str(&v.user_id);
            if v.is("copc", 1000) {
                continue;
            } else if user == "LASF_Projection" || user == "LASF_Spec" {
                kept_evlrs.push(v.clone());
            } else {
                eprintln!(
                    "warning: dropping EVLR {user}/{} ('{}') of {}: it may hold file offsets",
                    v.record_id,
                    las_str(&v.description),
                    a.path.display()
                );
            }
        }

        let vlr_bytes: u64 = vlrs.iter().map(|v| VLR_HEADER_SIZE + v.data.len() as u64).sum();
        let offset_to_point_data = u64::from(HEADER_SIZE) + vlr_bytes;
        let mut header = a.raw_header.clone();
        header.point_data_record_length = out_len;
        header.number_of_variable_length_records = u32::try_from(vlrs.len())?;
        header.offset_to_point_data =
            u32::try_from(offset_to_point_data).context("VLRs too large")?;
        header.generating_software = fixed(concat!("copc_compare ", env!("CARGO_PKG_VERSION")));
        (header.file_creation_day_of_year, header.file_creation_year) = today();
        // LAS 1.4 R15: formats 6+ keep point counts only in the 64-bit fields
        // and must zero the legacy ones (some writers, e.g. untwine, fill both).
        let lf = header.large_file.get_or_insert_with(Default::default);
        if lf.number_of_point_records == 0 {
            lf.number_of_point_records = u64::from(header.number_of_point_records);
        }
        if lf.number_of_points_by_return.iter().all(|&n| n == 0) {
            for (l, &n) in lf.number_of_points_by_return.iter_mut().zip(&header.number_of_points_by_return) {
                *l = u64::from(n);
            }
        }
        header.number_of_point_records = 0;
        header.number_of_points_by_return = [0; 5];
        header.evlr = Some(las::raw::header::Evlr {
            start_of_first_evlr: 0,
            number_of_evlrs: 1 + u32::try_from(kept_evlrs.len())?,
        });

        let ci = a.copc_info();
        let copc_info = copc_core::CopcInfo {
            center: (ci.center[0], ci.center[1], ci.center[2]),
            halfsize: ci.halfsize,
            spacing: ci.spacing,
            root_hier_offset: 0,
            root_hier_size: 0,
            gpstime_min: ci.gpstime_minimum,
            gpstime_max: ci.gpstime_maximum,
        };

        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut w = BufWriter::with_capacity(8 << 20, file);
        header.write_to(&mut w)?;
        for v in &vlrs {
            v.write_to(&mut w, false)?;
        }
        w.write_all(&0i64.to_le_bytes())?;
        let pos = w.stream_position()?;
        ensure!(pos == offset_to_point_data + 8, "layout error: at {pos}, expected {}", offset_to_point_data + 8);
        let eb_payload_pos = offset_to_point_data - vlrs.last().map_or(0, |v| v.data.len() as u64);

        Ok(Self {
            path: path.to_owned(),
            w,
            pos,
            header,
            copc_info,
            layout: RecordLayout { laz_vlr, in_len, n_fields: fields.len() },
            chunk_table: ChunkTable::default(),
            written: HashMap::new(),
            kept_evlrs,
            descriptors: fields.iter().map(ScaledField::descriptor).collect(),
            descriptors_pos: eb_payload_pos + our_descriptors_offset as u64,
        })
    }

    /// The record layout workers need to encode chunks for this file.
    pub fn layout(&self) -> &RecordLayout {
        &self.layout
    }

    /// Append one node's compressed chunk.
    pub fn write_chunk(&mut self, key: VoxelKey, point_count: u32, bytes: &[u8]) -> Result<()> {
        ensure!(point_count > 0, "node {key:?}: empty chunk");
        let byte_size = u32::try_from(bytes.len())
            .ok()
            .filter(|&n| n <= i32::MAX as u32)
            .with_context(|| format!("node {key:?}: chunk of {} bytes exceeds i32", bytes.len()))?;
        ensure!(point_count <= i32::MAX as u32, "node {key:?}: {point_count} points exceeds i32");
        let prev = self.written.insert(key_tuple(&key), Written { offset: self.pos, byte_size, point_count });
        ensure!(prev.is_none(), "node {key:?} written twice");
        self.w.write_all(bytes)?;
        self.pos += bytes.len() as u64;
        self.chunk_table.push(ChunkTableEntry { point_count: u64::from(point_count), byte_count: bytes.len() as u64 });
        Ok(())
    }

    /// Write the chunk table and hierarchy, then patch the header, CopcInfo and
    /// (if given, one `(min, max)` per field) the descriptor stats.
    pub fn finish(mut self, a: &CopcFile, stats: Option<&[(i32, i32)]>) -> Result<WriteSummary> {
        let mut entries = Vec::with_capacity(a.entries().len());
        let mut points = 0u64;
        for e in a.entries() {
            let k = copc_core::VoxelKey { level: e.key.level, x: e.key.x, y: e.key.y, z: e.key.z };
            let entry = if e.point_count == 0 {
                copc_core::Entry { key: k, offset: 0, byte_size: 0, point_count: 0 }
            } else {
                let Some(wr) = self.written.remove(&key_tuple(&e.key)) else {
                    bail!("node {:?} was never written", e.key);
                };
                ensure!(
                    wr.point_count == e.point_count,
                    "node {:?}: wrote {} points, compared file has {}",
                    e.key,
                    wr.point_count,
                    e.point_count
                );
                points += u64::from(wr.point_count);
                copc_core::Entry {
                    key: k,
                    offset: wr.offset,
                    byte_size: wr.byte_size as i32,
                    point_count: wr.point_count as i32,
                }
            };
            entries.push(entry);
        }
        if let Some(k) = self.written.keys().next() {
            bail!("node {k:?} isn't in the compared file's hierarchy");
        }
        let point_data_bytes = self.pos - u64::from(self.header.offset_to_point_data);

        let chunk_table_pos = self.pos;
        let mut ct = Vec::new();
        self.chunk_table.write_to(&mut ct, &self.layout.laz_vlr)?;
        self.w.write_all(&ct)?;
        self.pos += ct.len() as u64;

        let evlr_start = self.pos;
        let (page, root_size) = hierarchy_pages(entries, evlr_start + EVLR_HEADER_SIZE)?;
        self.copc_info.root_hier_offset = evlr_start + EVLR_HEADER_SIZE;
        self.copc_info.root_hier_size = root_size;
        let hier = RawVlr {
            reserved: 0,
            user_id: fixed("copc"),
            record_id: 1000,
            description: fixed("EPT hierarchy"),
            data: page,
        };
        hier.write_to(&mut self.w, true)?;
        for v in &self.kept_evlrs {
            v.write_to(&mut self.w, true)?;
        }
        if let Some(e) = self.header.evlr.as_mut() {
            e.start_of_first_evlr = evlr_start;
        }

        let info = self.copc_info.write_le_bytes()?;
        self.w.flush()?;
        let mut f = self.w.into_inner().map_err(|e| e.into_error())
            .with_context(|| format!("writing {}", self.path.display()))?;
        let bytes = f.stream_position()?;
        f.seek(SeekFrom::Start(0))?;
        self.header.write_to(&mut f)?;
        f.seek(SeekFrom::Start(COPC_INFO_POS))?;
        f.write_all(&info)?;
        f.seek(SeekFrom::Start(u64::from(self.header.offset_to_point_data)))?;
        f.write_all(&(chunk_table_pos as i64).to_le_bytes())?;
        if let Some(stats) = stats {
            ensure!(
                stats.len() == self.descriptors.len(),
                "{} stats for {} fields",
                stats.len(),
                self.descriptors.len()
            );
            f.seek(SeekFrom::Start(self.descriptors_pos))?;
            for (d, &(min, max)) in self.descriptors.iter_mut().zip(stats) {
                extra_bytes::set_i32_stats(d, min, max);
                f.write_all(d)?;
            }
        }
        Ok(WriteSummary { bytes, chunks: self.chunk_table.len(), points, point_data_bytes })
    }
}

/// One hierarchy page before layout: its node entries and its child pages
/// (subtree root key, page index).
#[derive(Default)]
struct Page {
    entries: Vec<copc_core::Entry>,
    children: Vec<(copc_core::VoxelKey, usize)>,
}

fn ancestor_at(k: copc_core::VoxelKey, level: i32) -> copc_core::VoxelKey {
    let sh = k.level - level;
    copc_core::VoxelKey { level, x: k.x >> sh, y: k.y >> sh, z: k.z >> sh }
}

/// Split `entries` into pages: entries above `boundaries[depth]` stay here; each
/// subtree rooted at the boundary level goes to its own page (recursively).
fn build_page(entries: Vec<copc_core::Entry>, boundaries: &[i32], depth: usize, pages: &mut Vec<Page>) -> usize {
    let idx = pages.len();
    pages.push(Page::default());
    let Some(&b) = boundaries.get(depth) else {
        pages[idx].entries = entries;
        return idx;
    };
    let mut groups: std::collections::BTreeMap<(i32, i32, i32, i32), Vec<copc_core::Entry>> = Default::default();
    let mut mine = Vec::new();
    for e in entries {
        if e.key.level < b {
            mine.push(e);
        } else {
            let r = ancestor_at(e.key, b);
            groups.entry((r.level, r.x, r.y, r.z)).or_default().push(e);
        }
    }
    for ((level, x, y, z), group) in groups {
        let child = build_page(group, boundaries, depth + 1, pages);
        pages[idx].children.push((copc_core::VoxelKey { level, x, y, z }, child));
    }
    pages[idx].entries = mine;
    idx
}

/// Serialize the hierarchy as nested pages laid out from file offset `base`,
/// root page first, then the others depth-first. Page boundaries every 3 levels
/// from level 3, as copc-converter does, so a streaming reader fetches only the
/// pages for the region it shows. A child page is referenced from its parent by
/// an entry with the subtree root's key and `point_count = -1`; the subtree
/// root's own entry lives in the child page.
///
/// Returns the payload and the root page size.
pub fn hierarchy_pages(entries: Vec<copc_core::Entry>, base: u64) -> Result<(Vec<u8>, u64)> {
    let max_level = entries.iter().map(|e| e.key.level).max().unwrap_or(0);
    let boundaries: Vec<i32> = (1..).map(|i| 3 * i).take_while(|&l| l < max_level).collect();
    let mut pages = Vec::new();
    build_page(entries, &boundaries, 0, &mut pages);

    let size = |p: &Page| ((p.entries.len() + p.children.len()) * copc_core::HIERARCHY_ENTRY_BYTES) as u64;
    let mut offsets = Vec::with_capacity(pages.len());
    let mut at = base;
    for p in &pages {
        offsets.push(at);
        at += size(p);
    }
    let mut out = Vec::with_capacity((at - base) as usize);
    for p in &pages {
        let mut all = p.entries.clone();
        for &(key, child) in &p.children {
            let byte_size = i32::try_from(size(&pages[child])).context("hierarchy page too large")?;
            all.push(copc_core::Entry { key, offset: offsets[child], byte_size, point_count: -1 });
        }
        out.extend_from_slice(&copc_core::HierarchyPage::new(all).write_le_bytes()?);
    }
    Ok((out, size(&pages[0])))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(level: i32, x: i32, y: i32, z: i32) -> copc_core::Entry {
        copc_core::Entry { key: copc_core::VoxelKey { level, x, y, z }, offset: 1000, byte_size: 10, point_count: 5 }
    }

    /// Root + a chain down to level 7: boundaries at 3 and 6, so three pages.
    #[test]
    fn pages_every_three_levels() {
        let mut entries: Vec<_> = (0..=7).map(|l| entry(l, 0, 0, 0)).chain([entry(4, 1, 0, 0)]).collect();
        entries.sort_by_key(|e| (e.key.level, e.key.x, e.key.y, e.key.z)); // as CopcFile::entries()
        let (data, root) = hierarchy_pages(entries, 5000).unwrap();
        let page = |off: u64, size: u64| {
            copc_core::HierarchyPage::from_le_bytes(&data[(off - 5000) as usize..][..size as usize])
                .unwrap()
                .into_entries()
        };
        let p0 = page(5000, root);
        assert_eq!(p0.iter().map(|e| (e.key.level, e.point_count)).collect::<Vec<_>>(), [(0, 5), (1, 5), (2, 5), (3, -1)]);
        let ptr = p0[3];
        let p1 = page(ptr.offset, ptr.byte_size as u64);
        // Level 3 subtree: levels 3-5 (incl. the sibling at 4) plus a pointer at 6.
        assert_eq!(p1.iter().map(|e| (e.key.level, e.point_count)).collect::<Vec<_>>(), [(3, 5), (4, 5), (4, 5), (5, 5), (6, -1)]);
        let p2 = page(p1[4].offset, p1[4].byte_size as u64);
        assert_eq!(p2.iter().map(|e| e.key.level).collect::<Vec<_>>(), [6, 7]);
        assert_eq!(data.len(), (4 + 5 + 2) * 32);
    }

    #[test]
    fn shallow_tree_is_one_page() {
        let (data, root) = hierarchy_pages((0..=3).map(|l| entry(l, 0, 0, 0)).collect(), 0).unwrap();
        assert_eq!((data.len() as u64, root), (4 * 32, 4 * 32));
    }
}
