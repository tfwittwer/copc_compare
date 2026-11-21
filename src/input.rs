//! Opening a COPC file: mmap, copc-streaming reader, and byte-exact copies of the
//! raw header, VLRs and EVLRs (the writer re-emits those verbatim).

use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use copc_streaming::{
    Aabb, ByteSource, Chunk, CopcError, CopcInfo, CopcStreamingReader, Fields, HierarchyEntry,
    VoxelKey,
};
use laz::LazVlr;
use memmap2::Mmap;

/// `ByteSource` over a shared mmap. Only used for the header and hierarchy pages;
/// chunks are decoded straight from the mapped slice without a copy.
#[derive(Clone)]
pub struct MmapSource(Arc<Mmap>);

fn range(data: &[u8], offset: u64, length: u64) -> Result<&[u8], CopcError> {
    usize::try_from(offset)
        .ok()
        .zip(usize::try_from(length).ok())
        .and_then(|(o, l)| data.get(o..o.checked_add(l)?))
        .ok_or_else(|| {
            CopcError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("range {offset}+{length} beyond file size {}", data.len()),
            ))
        })
}

impl ByteSource for MmapSource {
    async fn read_range(&self, offset: u64, length: u64) -> Result<Vec<u8>, CopcError> {
        range(&self.0, offset, length).map(<[u8]>::to_vec)
    }

    async fn size(&self) -> Result<Option<u64>, CopcError> {
        Ok(Some(self.0.len() as u64))
    }
}

/// A VLR or EVLR as stored. Unlike `las::raw::Vlr` it is `Clone` and doesn't
/// fix VLR vs EVLR at read time: the record length is written from `data`.
#[derive(Clone, Debug)]
pub struct RawVlr {
    pub reserved: u16,
    pub user_id: [u8; 16],
    pub record_id: u16,
    pub description: [u8; 32],
    pub data: Vec<u8>,
}

impl From<las::raw::Vlr> for RawVlr {
    fn from(v: las::raw::Vlr) -> Self {
        Self {
            reserved: v.reserved,
            user_id: v.user_id,
            record_id: v.record_id,
            description: v.description,
            data: v.data,
        }
    }
}

impl RawVlr {
    pub const HEADER_SIZE: u64 = 54;
    pub const EVLR_HEADER_SIZE: u64 = 60;

    pub fn is(&self, user_id: &str, record_id: u16) -> bool {
        self.record_id == record_id && las_str(&self.user_id) == user_id
    }

    /// Write as a VLR (u16 length) or, with `extended`, an EVLR (u64 length).
    pub fn write_to(&self, w: &mut impl std::io::Write, extended: bool) -> Result<()> {
        w.write_all(&self.reserved.to_le_bytes())?;
        w.write_all(&self.user_id)?;
        w.write_all(&self.record_id.to_le_bytes())?;
        if extended {
            w.write_all(&(self.data.len() as u64).to_le_bytes())?;
        } else {
            let n = u16::try_from(self.data.len()).with_context(|| {
                format!("VLR {}/{} longer than 65535 bytes", las_str(&self.user_id), self.record_id)
            })?;
            w.write_all(&n.to_le_bytes())?;
        }
        w.write_all(&self.description)?;
        w.write_all(&self.data)?;
        Ok(())
    }
}

/// An opened COPC file with its full hierarchy loaded.
pub struct CopcFile {
    pub path: PathBuf,
    data: Arc<Mmap>,
    reader: CopcStreamingReader<MmapSource>,
    /// The header exactly as stored (copc-streaming's `las::Header` is rebuilt
    /// and would recompute bounds and offsets on `into_raw`).
    pub raw_header: las::raw::Header,
    pub vlrs: Vec<RawVlr>,
    pub evlrs: Vec<RawVlr>,
    /// All hierarchy entries, including `point_count == 0` ones, sorted by
    /// (level, x, y, z).
    entries: Vec<HierarchyEntry>,
}

// Tiles decode from one shared `CopcFile` on many rayon threads.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CopcFile>();
};

impl CopcFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: read-only mapping. The file must not be truncated while we run.
        let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
        let data = Arc::new(mmap);

        let mut reader = pollster::block_on(CopcStreamingReader::open(MmapSource(data.clone())))
            .with_context(|| format!("reading COPC header of {}", path.display()))?;
        pollster::block_on(reader.load_all_hierarchy())
            .with_context(|| format!("loading hierarchy of {}", path.display()))?;

        let mut cur = Cursor::new(&data[..]);
        let raw_header = las::raw::Header::read_from(&mut cur)?;
        cur.set_position(u64::from(raw_header.header_size));
        let vlrs = (0..raw_header.number_of_variable_length_records)
            .map(|_| las::raw::Vlr::read_from(&mut cur, false).map(RawVlr::from))
            .collect::<Result<Vec<_>, _>>()
            .context("reading VLRs")?;
        let evlrs = match raw_header.evlr {
            Some(e) if e.number_of_evlrs > 0 => {
                cur.set_position(e.start_of_first_evlr);
                (0..e.number_of_evlrs)
                    .map(|_| las::raw::Vlr::read_from(&mut cur, true).map(RawVlr::from))
                    .collect::<Result<Vec<_>, _>>()
                    .context("reading EVLRs")?
            }
            _ => Vec::new(),
        };

        let mut entries: Vec<HierarchyEntry> = reader.entries().map(|(_, e)| e.clone()).collect();
        entries.sort_unstable_by_key(|e| key_tuple(&e.key));

        let f = Self { path: path.to_owned(), data, reader, raw_header, vlrs, evlrs, entries };
        f.validate()?;
        Ok(f)
    }

    fn validate(&self) -> Result<()> {
        let fmt = self.las_header().point_format().to_u8()?;
        if !(6..=8).contains(&fmt) {
            bail!("{}: point format {fmt}, COPC requires 6, 7 or 8", self.path.display());
        }
        ensure!(
            !self.reader.has_pending_pages(),
            "{}: hierarchy pages left unloaded",
            self.path.display()
        );
        for e in self.data_entries() {
            range(&self.data, e.offset, u64::from(e.byte_size))
                .with_context(|| format!("{}: node {:?}", self.path.display(), e.key))?;
        }
        Ok(())
    }

    pub fn las_header(&self) -> &las::Header {
        self.reader.header().las_header()
    }

    pub fn copc_info(&self) -> &CopcInfo {
        self.reader.copc_info()
    }

    pub fn laz_vlr(&self) -> &LazVlr {
        self.reader.header().laz_vlr()
    }

    pub fn file_size(&self) -> u64 {
        self.data.len() as u64
    }

    pub fn root_bounds(&self) -> Aabb {
        self.copc_info().root_bounds()
    }

    /// Bytes per point record, extra bytes included.
    pub fn record_len(&self) -> usize {
        usize::from(self.las_header().point_format().len())
    }

    /// All hierarchy entries (also empty ones), sorted by (level, x, y, z).
    pub fn entries(&self) -> &[HierarchyEntry] {
        &self.entries
    }

    /// Entries that hold points.
    pub fn data_entries(&self) -> impl Iterator<Item = &HierarchyEntry> {
        self.entries.iter().filter(|e| e.point_count > 0)
    }

    pub fn get(&self, key: &VoxelKey) -> Option<&HierarchyEntry> {
        self.reader.get(key)
    }

    /// The node's compressed chunk bytes.
    pub fn compressed(&self, entry: &HierarchyEntry) -> &[u8] {
        range(&self.data, entry.offset, u64::from(entry.byte_size))
            .expect("chunk ranges are checked in open()")
    }

    /// Decode one node. Sync, so it runs on any rayon thread.
    pub fn decode(&self, entry: &HierarchyEntry, fields: Fields) -> Result<Chunk> {
        let h = self.reader.header();
        let chunk =
            Chunk::decompress(self.compressed(entry), entry, h.laz_vlr(), h.las_header(), fields)
                .with_context(|| format!("{}: decoding node {:?}", self.path.display(), entry.key))?;
        ensure!(
            chunk.cloud().raw_bytes().len() == entry.point_count as usize * self.record_len(),
            "{}: node {:?} decoded to {} bytes, expected {} points × {}",
            self.path.display(),
            entry.key,
            chunk.cloud().raw_bytes().len(),
            entry.point_count,
            self.record_len()
        );
        Ok(chunk)
    }
}

pub fn key_tuple(k: &VoxelKey) -> (i32, i32, i32, i32) {
    (k.level, k.x, k.y, k.z)
}

/// A fixed-size LAS string field (user id, description, software) up to its first NUL.
pub fn las_str(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}
