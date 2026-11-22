//! Splicing new extra bytes into the compared file's compressed chunks, instead
//! of decoding every field and re-encoding the whole record.
//!
//! A COPC node is one LAS 1.4 layered LAZ chunk (item version 3):
//! ```text
//! first point, raw            record_len bytes, item after item
//! u32 point count
//! u32 byte count per layer    Point14: 9, RGB14: 1, RGBNIR14: 2, Byte14(n): n
//! the layers                  same order
//! ```
//! Each extra byte is a layer of its own, with its own arithmetic coder and
//! models, and the extra bytes are the last item. Appending k bytes to every
//! record therefore leaves all of A's layers valid. The new chunk is A's first
//! point plus k raw bytes, A's count, A's layer sizes plus k, A's layers plus k.
//! Only the new layers are encoded, by laz's own extra-bytes coder, so they come
//! out exactly as a full re-encode writes them.
//!
//! The one coupling is the coder's context, one of 4 model sets chosen by the
//! scanner channel. The LAZ v3 record coder (laz-rs, like LASzip) resets the
//! context to 0 before every point; the Point14 coder sets it to the channel
//! only on points whose channel differs from the previous point's. [`contexts`]
//! reproduces that from the channel bits, which the base layer decodes, so
//! decoding A with `Fields::Z` is enough.

use anyhow::{Result, ensure};
use laz::las::v3::LasExtraByteCompressor;
use laz::record::LayeredFieldCompressor;
use laz::{LazItemRecordBuilder, LazItemType};

use crate::input::CopcFile;

/// LAZ VLR compressor id of layered chunks (point formats 6+).
const LAYERED_CHUNKED: u16 = 3;

/// The byte of a PF 6–8 record that holds the scanner channel (bits 4–5).
const CHANNEL_BYTE: usize = 15;

/// How A's chunks get `new_bytes` more extra bytes per record.
#[derive(Clone, Debug)]
pub struct Splice {
    /// A's record length, which is also the size of the raw first point.
    record_len: usize,
    /// Layers in each of A's chunks.
    layers: usize,
    new_bytes: usize,
}

impl Splice {
    /// `Err` says why A's chunks can't be spliced: they must use layered chunks
    /// with exactly the version 3 items laz writes for A's point format, as the
    /// output does.
    pub fn new(a: &CopcFile, new_bytes: usize) -> std::result::Result<Self, String> {
        let vlr = a.vlrs.iter().find(|v| v.is("laszip encoded", 22204)).ok_or("no LAZ VLR")?;
        let compressor = vlr.data.get(..2).map(|b| u16::from_le_bytes([b[0], b[1]]));
        if compressor != Some(LAYERED_CHUNKED) {
            return Err(format!("LAZ compressor {compressor:?}, not layered chunks"));
        }
        let extra = a.las_header().point_format().extra_bytes;
        let expected =
            LazItemRecordBuilder::default_for_point_format_id(a.point_format(), extra).map_err(|e| e.to_string())?;
        let items = a.laz_vlr().items();
        if *items != expected {
            return Err(format!("LAZ items {items:?}, expected {expected:?}"));
        }
        let mut layers = 0;
        for i in items {
            layers += match i.item_type() {
                LazItemType::Point14 => 9,
                LazItemType::RGB14 => 1,
                LazItemType::RGBNIR14 => 2,
                LazItemType::Byte14(n) => usize::from(n),
                t => return Err(format!("LAZ item {t:?}")),
            };
        }
        Ok(Self { record_len: a.record_len(), layers, new_bytes })
    }

    /// Whether `chunk`'s layer table accounts for exactly its bytes. Anything
    /// else (trailing bytes, a different layout) is re-encoded instead.
    pub fn fits(&self, chunk: &[u8]) -> bool {
        let data = self.record_len + 4 + 4 * self.layers;
        chunk.get(self.record_len + 4..data).is_some_and(|sizes| {
            let layers: u64 =
                sizes.chunks_exact(4).map(|b| u64::from(u32::from_le_bytes(b.try_into().unwrap()))).sum();
            data as u64 + layers == chunk.len() as u64
        })
    }

    /// One of A's chunks with `extra` (`new_bytes` per point) appended to its
    /// records. `records` are the chunk's decoded records; only their scanner
    /// channels are used.
    pub fn splice(&self, chunk: &[u8], records: &[u8], extra: &[u8]) -> Result<Vec<u8>> {
        let k = self.new_bytes;
        ensure!(self.fits(chunk), "chunk's layer table doesn't match its size");
        ensure!(
            !records.is_empty() && records.len().is_multiple_of(self.record_len),
            "{} record bytes for {}-byte records",
            records.len(),
            self.record_len
        );
        let n = records.len() / self.record_len;
        ensure!(extra.len() == n * k, "{} extra bytes for {n} points × {k}", extra.len());
        if k == 0 {
            return Ok(chunk.to_vec());
        }
        let (first, rest) = chunk.split_at(self.record_len);
        let (count, rest) = rest.split_at(4);
        let (sizes, layers) = rest.split_at(4 * self.layers);

        let mut coder = LasExtraByteCompressor::new(k);
        let mut out = Vec::with_capacity(chunk.len() + 5 * k + n * k / 2);
        out.extend_from_slice(first);
        let mut contexts = contexts(records.chunks_exact(self.record_len).map(|r| (r[CHANNEL_BYTE] >> 4) & 3));
        let mut points = extra.chunks_exact(k);
        let (p0, mut c0) = (points.next().expect("n > 0"), contexts.next().expect("n > 0"));
        coder.init_first_point(&mut out, p0, &mut c0)?;
        for (p, mut c) in points.zip(contexts) {
            LayeredFieldCompressor::<Vec<u8>>::compress_field_with(&mut coder, p, &mut c)?;
        }
        out.extend_from_slice(count);
        out.extend_from_slice(sizes);
        coder.write_layers_sizes(&mut out)?;
        out.extend_from_slice(layers);
        coder.write_layers(&mut out)?;
        Ok(out)
    }
}

/// The context the LAZ v3 record coder gives each point's extra bytes: the
/// first point's channel, then the channel on points where it changes, else 0.
pub fn contexts(channels: impl Iterator<Item = u8>) -> impl Iterator<Item = usize> {
    let mut prev = None;
    channels.map(move |c| {
        let ctx = if prev.is_some_and(|p| p == c) { 0 } else { c };
        prev = Some(c);
        usize::from(ctx)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_resets_unless_the_channel_changes() {
        let got: Vec<usize> = contexts([2, 2, 3, 3, 0, 1, 1, 2, 2].into_iter()).collect();
        assert_eq!(got, [2, 0, 3, 0, 0, 1, 0, 2, 0]);
    }
}
