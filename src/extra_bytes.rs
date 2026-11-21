//! LAS 1.4 Extra Bytes descriptors (VLR `LASF_Spec`/4) for the output fields, and
//! scaled-integer encoding of distances.
//!
//! Descriptor layout (LAS 1.4 R15, table 24), 192 bytes:
//! `reserved[2] data_type options name[32] unused[4] no_data[3] min[3] max[3]
//!  scale[3] offset[3] description[32]`, with the three-element arrays at 40, 64,
//! 88, 112 and 136 (only element 0 is used since R14 deprecated array types).

use anyhow::{Result, bail, ensure};

use crate::input::las_str;

pub const DESCRIPTOR_SIZE: usize = 192;
pub const USER_ID: &str = "LASF_Spec";
pub const RECORD_ID: u16 = 4;

const DATA_TYPE_UNDOCUMENTED: u8 = 0;
const DATA_TYPE_I32: u8 = 6;
const OPT_MIN: u8 = 1 << 1;
const OPT_MAX: u8 = 1 << 2;
const OPT_SCALE: u8 = 1 << 3;

const OFF_DATA_TYPE: usize = 2;
const OFF_OPTIONS: usize = 3;
const OFF_NAME: usize = 4;
const OFF_MIN: usize = 64;
const OFF_MAX: usize = 88;
const OFF_SCALE: usize = 112;
const OFF_DESCRIPTION: usize = 160;

/// One output field: an i32 with descriptor scale `scale` and offset 0, so the
/// value a reader shows is `raw * scale`.
#[derive(Clone, Debug)]
pub struct ScaledField {
    pub name: String,
    pub description: String,
    pub scale: f64,
}

impl ScaledField {
    pub const BYTES: usize = 4;

    pub fn new(name: &str, description: &str, scale: f64) -> Result<Self> {
        ensure!(name.len() < 32, "field name '{name}' longer than 31 bytes");
        ensure!(description.len() < 32, "field description '{description}' longer than 31 bytes");
        ensure!(scale.is_finite() && scale > 0.0, "field scale must be positive, got {scale}");
        Ok(Self { name: name.into(), description: description.into(), scale })
    }

    /// The stored integer for value `v`. The caller has checked the range with
    /// [`ScaledField::check_range`].
    #[inline]
    pub fn encode(&self, v: f64) -> i32 {
        (v / self.scale).round() as i32
    }

    /// Error unless every value in `[-max_abs, max_abs]` fits in an i32.
    pub fn check_range(&self, max_abs: f64) -> Result<()> {
        let n = (max_abs / self.scale).round();
        ensure!(
            n <= f64::from(i32::MAX),
            "field {}: {max_abs} / scale {} = {n} doesn't fit in an i32; use a coarser scale",
            self.name,
            self.scale
        );
        Ok(())
    }

    pub fn descriptor(&self) -> [u8; DESCRIPTOR_SIZE] {
        let mut d = [0u8; DESCRIPTOR_SIZE];
        d[OFF_DATA_TYPE] = DATA_TYPE_I32;
        d[OFF_OPTIONS] = OPT_SCALE;
        d[OFF_NAME..OFF_NAME + self.name.len()].copy_from_slice(self.name.as_bytes());
        d[OFF_SCALE..OFF_SCALE + 8].copy_from_slice(&self.scale.to_le_bytes());
        d[OFF_DESCRIPTION..OFF_DESCRIPTION + self.description.len()]
            .copy_from_slice(self.description.as_bytes());
        d
    }
}

/// Set the min/max stats (options bits 1 and 2) of an i32 descriptor. Integer
/// stats are stored as i64 "anytype".
pub fn set_i32_stats(d: &mut [u8], min: i32, max: i32) {
    debug_assert_eq!(d.len(), DESCRIPTOR_SIZE);
    debug_assert_eq!(d[OFF_DATA_TYPE], DATA_TYPE_I32);
    d[OFF_OPTIONS] |= OPT_MIN | OPT_MAX;
    d[OFF_MIN..OFF_MIN + 8].copy_from_slice(&i64::from(min).to_le_bytes());
    d[OFF_MAX..OFF_MAX + 8].copy_from_slice(&i64::from(max).to_le_bytes());
}

/// Bytes per point that a descriptor covers.
fn descriptor_len(d: &[u8]) -> Result<usize> {
    const BASE: [usize; 10] = [1, 1, 2, 2, 4, 4, 8, 8, 4, 8];
    let t = d[OFF_DATA_TYPE];
    Ok(match t {
        DATA_TYPE_UNDOCUMENTED => usize::from(d[OFF_OPTIONS]),
        1..=10 => BASE[usize::from(t - 1)],
        // Deprecated 2- and 3-element arrays of types 1..=10.
        11..=20 => 2 * BASE[usize::from(t - 11)],
        21..=30 => 3 * BASE[usize::from(t - 21)],
        _ => bail!("extra bytes descriptor with unknown data type {t}"),
    })
}

/// The output Extra Bytes VLR payload: A's descriptors, then ours.
///
/// If A has extra bytes its descriptors don't cover (or none at all), an
/// "undocumented" descriptor fills the gap so ours line up with the right bytes.
pub fn output_descriptors(
    a_payload: Option<&[u8]>,
    a_extra_bytes: usize,
    fields: &[ScaledField],
) -> Result<Vec<u8>> {
    let a_payload = a_payload.unwrap_or_default();
    ensure!(
        a_payload.len().is_multiple_of(DESCRIPTOR_SIZE),
        "compared file's Extra Bytes VLR is {} bytes, not a multiple of {DESCRIPTOR_SIZE}",
        a_payload.len()
    );
    let mut covered = 0;
    let mut names = Vec::new();
    for d in a_payload.chunks_exact(DESCRIPTOR_SIZE) {
        covered += descriptor_len(d)?;
        names.push(las_str(&d[OFF_NAME..OFF_NAME + 32]));
    }
    ensure!(
        covered <= a_extra_bytes,
        "compared file's Extra Bytes descriptors cover {covered} bytes, but its records have \
         only {a_extra_bytes} extra bytes"
    );
    for f in fields {
        ensure!(
            !names.contains(&f.name),
            "compared file already has an extra bytes field named '{}'",
            f.name
        );
    }

    let mut out = a_payload.to_vec();
    let mut gap = a_extra_bytes - covered;
    while gap > 0 {
        // An undocumented descriptor carries its byte count in the u8 options.
        let n = gap.min(255);
        let mut d = [0u8; DESCRIPTOR_SIZE];
        d[OFF_DATA_TYPE] = DATA_TYPE_UNDOCUMENTED;
        d[OFF_OPTIONS] = n as u8;
        let name = b"undocumented";
        d[OFF_NAME..OFF_NAME + name.len()].copy_from_slice(name);
        out.extend_from_slice(&d);
        gap -= n;
    }
    for f in fields {
        out.extend_from_slice(&f.descriptor());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_layout() {
        let f = ScaledField::new("c2c_dist", "C2C dist m; >max = 5*max", 0.001).unwrap();
        let d = f.descriptor();
        assert_eq!(d[2], 6);
        assert_eq!(d[3], 8);
        assert_eq!(&d[4..12], b"c2c_dist");
        assert_eq!(d[12], 0);
        assert_eq!(f64::from_le_bytes(d[112..120].try_into().unwrap()), 0.001);
        assert_eq!(&d[136..144], &[0; 8]);
        assert_eq!(las_str(&d[160..192]), "C2C dist m; >max = 5*max");
        assert_eq!(descriptor_len(&d).unwrap(), 4);
    }

    #[test]
    fn encode_rounds() {
        let f = ScaledField::new("d", "", 0.001).unwrap();
        assert_eq!(f.encode(0.0124), 12);
        assert_eq!(f.encode(0.0126), 13);
        assert_eq!(f.encode(-0.0126), -13);
        assert!(f.check_range(2_000_000.0).is_ok());
        assert!(f.check_range(3_000_000.0).is_err());
    }

    #[test]
    fn fills_undocumented_gap() {
        let f = ScaledField::new("d", "", 0.01).unwrap();
        let out = output_descriptors(None, 3, std::slice::from_ref(&f)).unwrap();
        assert_eq!(out.len(), 2 * DESCRIPTOR_SIZE);
        assert_eq!(out[2], 0);
        assert_eq!(out[3], 3);
        assert_eq!(out[DESCRIPTOR_SIZE + 2], 6);
    }

    #[test]
    fn rejects_duplicate_name() {
        let f = ScaledField::new("d", "", 0.01).unwrap();
        let a = f.descriptor();
        assert!(output_descriptors(Some(&a), 4, std::slice::from_ref(&f)).is_err());
    }
}
