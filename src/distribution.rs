//! Distribution of the c2c output values over the points with a neighbour
//! within max-dist: count, min, max, mean, standard deviation and RMS, exact
//! from integer sums of the raw values, and the median from a histogram of them.

use std::fmt::Write;

/// Histogram bins per field, at most. Raw values span `2 · bound + 1` steps; the
/// bin width is the smallest power of two that keeps them within this many bins.
/// One bin per value (an exact median) up to bound 32767, e.g. max-dist 3.2 m at
/// a 0.1 mm scale.
const MAX_BINS: u64 = 1 << 16;

#[derive(Clone, Debug)]
pub struct Distribution {
    /// Raw value of the first bin's start.
    lo: i64,
    /// log2 of the bin width.
    shift: u32,
    bins: usize,
    fields: Vec<Field>,
}

#[derive(Clone, Debug)]
struct Field {
    n: u64,
    min: i32,
    max: i32,
    sum: i128,
    sum_sq: i128,
    /// Allocated on the first value, so empty distributions are cheap.
    hist: Vec<u64>,
}

/// One field's distribution, in metres.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldSummary {
    pub n: u64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    /// Population standard deviation (spread around the mean).
    pub stddev: f64,
    /// Root mean square (spread around 0): `sqrt(mean² + stddev²)`.
    pub rms: f64,
    /// How far the median may be off: 0 with one histogram bin per raw value.
    pub median_error: f64,
}

impl Distribution {
    /// For `n_fields` fields whose raw values lie within `-bound..=bound`
    /// (values outside count in the edge bins for the median).
    pub fn new(n_fields: usize, bound: i64) -> Self {
        let span = 2 * bound.max(0) as u64 + 1;
        let shift = span.div_ceil(MAX_BINS).next_power_of_two().trailing_zeros();
        let bins = span.div_ceil(1 << shift) as usize;
        let field = Field { n: 0, min: i32::MAX, max: i32::MIN, sum: 0, sum_sq: 0, hist: Vec::new() };
        Self { lo: -bound.max(0), shift, bins, fields: vec![field; n_fields] }
    }

    /// Add points' values (point-major, one per field), skipping points whose
    /// first value is `cap`: those have no neighbour within max-dist.
    pub fn add(&mut self, vals: &[i32], cap: i32) {
        let nf = self.fields.len();
        if nf == 0 || vals.is_empty() {
            return;
        }
        for f in &mut self.fields {
            if f.hist.is_empty() {
                f.hist = vec![0; self.bins];
            }
        }
        let last = self.bins as i64 - 1;
        for p in vals.chunks_exact(nf) {
            if p[0] == cap {
                continue;
            }
            for (f, &v) in self.fields.iter_mut().zip(p) {
                f.n += 1;
                f.min = f.min.min(v);
                f.max = f.max.max(v);
                let w = i128::from(v);
                f.sum += w;
                f.sum_sq += w * w;
                f.hist[((i64::from(v) - self.lo) >> self.shift).clamp(0, last) as usize] += 1;
            }
        }
    }

    pub fn merge(&mut self, o: &Distribution) {
        debug_assert_eq!((self.lo, self.shift, self.fields.len()), (o.lo, o.shift, o.fields.len()));
        for (f, g) in self.fields.iter_mut().zip(&o.fields) {
            if g.n == 0 {
                continue;
            }
            if f.hist.is_empty() {
                f.hist = vec![0; self.bins];
            }
            f.n += g.n;
            f.min = f.min.min(g.min);
            f.max = f.max.max(g.max);
            f.sum += g.sum;
            f.sum_sq += g.sum_sq;
            for (a, b) in f.hist.iter_mut().zip(&g.hist) {
                *a += b;
            }
        }
    }

    /// Per field, its distribution with raw values times `scale`; `None` if no
    /// point had a neighbour.
    pub fn summaries(&self, scale: f64) -> Vec<Option<FieldSummary>> {
        let width = 1i64 << self.shift;
        self.fields
            .iter()
            .map(|f| {
                if f.n == 0 {
                    return None;
                }
                let n = f.n as f64;
                // The raw value at 0-based rank k: exact, or its bin's centre
                // (kept within min..max).
                let at_rank = |k: u64| {
                    let mut seen = 0;
                    let bin = f.hist.iter().position(|&c| {
                        seen += c;
                        seen > k
                    });
                    let start = self.lo + bin.expect("rank below count") as i64 * width;
                    (start as f64 + (width - 1) as f64 / 2.0).clamp(f64::from(f.min), f64::from(f.max))
                };
                let median = (at_rank((f.n - 1) / 2) + at_rank(f.n / 2)) / 2.0;
                // n·Σx² − (Σx)², exact while it fits in i128.
                let var = i128::from(f.n)
                    .checked_mul(f.sum_sq)
                    .zip(f.sum.checked_mul(f.sum))
                    .and_then(|(a, b)| a.checked_sub(b))
                    .map_or_else(
                        || f.sum_sq as f64 / n - (f.sum as f64 / n).powi(2),
                        |num| num as f64 / (n * n),
                    )
                    .max(0.0);
                Some(FieldSummary {
                    n: f.n,
                    min: f64::from(f.min) * scale,
                    max: f64::from(f.max) * scale,
                    mean: f.sum as f64 / n * scale,
                    median: median * scale,
                    stddev: var.sqrt() * scale,
                    rms: (f.sum_sq as f64 / n).sqrt() * scale,
                    median_error: if width == 1 { 0.0 } else { width as f64 / 2.0 * scale },
                })
            })
            .collect()
    }
}

/// The distribution as a table, one row per field: min and max with as many
/// decimals as `scale` has, mean, median, stddev and RMS with one more (they
/// fall between the steps, and a small mean offset is often the point).
pub fn table(names: &[String], summaries: &[Option<FieldSummary>], scale: f64) -> String {
    let decimals = (-scale.log10() - 1e-9).ceil().clamp(0.0, 9.0) as usize;
    let header = ["field", "min", "max", "mean", "median", "stddev", "rms"];
    let mut rows: Vec<[String; 7]> = vec![header.map(String::from)];
    for (name, s) in names.iter().zip(summaries) {
        let (v, w) = (|x: f64| format!("{x:.decimals$}"), |x: f64| format!("{x:.0$}", decimals + 1));
        rows.push(match s {
            Some(s) => [name.clone(), v(s.min), v(s.max), w(s.mean), w(s.median), w(s.stddev), w(s.rms)],
            None => std::array::from_fn(|c| if c == 0 { name.clone() } else { "-".into() }),
        });
    }
    let widths: Vec<usize> = (0..header.len()).map(|c| rows.iter().map(|r| r[c].len()).max().unwrap_or(0)).collect();
    let mut out = String::new();
    for r in &rows {
        let _ = write!(out, "    {:<w$}", r[0], w = widths[0]);
        for (cell, &w) in r[1..].iter().zip(&widths[1..]) {
            let _ = write!(out, "  {cell:>w$}");
        }
        out.push('\n');
    }
    if let Some(e) = summaries.iter().flatten().map(|s| s.median_error).find(|&e| e > 0.0) {
        let _ = writeln!(out, "    (median to within ±{e:.decimals$} m: histogram bins of {:.decimals$} m)", 2.0 * e);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_with_one_bin_per_value() {
        let cap = 5000;
        let mut d = Distribution::new(2, 1000);
        // Two fields per point; the third point has no neighbour.
        d.add(&[1, -4, 2, 0, cap, cap, 4, 4], cap);
        let mut e = Distribution::new(2, 1000);
        e.add(&[3, -1], cap);
        d.merge(&e);
        let s = d.summaries(0.5);
        let a = s[0].as_ref().unwrap();
        assert_eq!((a.n, a.min, a.max, a.median, a.median_error), (4, 0.5, 2.0, 1.25, 0.0));
        assert!((a.mean - 1.25).abs() < 1e-12);
        assert!((a.stddev - 0.5 * 1.25f64.sqrt()).abs() < 1e-12);
        assert!((a.rms - 0.5 * 7.5f64.sqrt()).abs() < 1e-12); // sqrt((1 + 4 + 9 + 16) / 4) raw
        assert!((s[1].as_ref().unwrap().rms - 0.5 * 8.25f64.sqrt()).abs() < 1e-12); // sqrt((16 + 0 + 16 + 1) / 4) raw
        let b = s[1].as_ref().unwrap();
        assert_eq!((b.min, b.max, b.median), (-2.0, 2.0, -0.25));
        assert!(Distribution::new(1, 10).summaries(1.0)[0].is_none());
    }

    #[test]
    fn wide_ranges_bin_the_median() {
        // ±1e6 raw: 2e6 + 1 values over at most 65536 bins → width 32.
        let mut d = Distribution::new(1, 1_000_000);
        let vals: Vec<i32> = (0..1001).map(|i| i * 1000 - 500_000).collect();
        d.add(&vals, i32::MAX);
        let s = d.summaries(1.0)[0].clone().unwrap();
        assert_eq!(s.median_error, 16.0);
        assert!((s.median - 0.0).abs() <= 16.0, "median {}", s.median);
        assert_eq!((s.min, s.max, s.mean), (-500_000.0, 500_000.0, 0.0));
    }

    #[test]
    fn table_aligns_columns() {
        let s = FieldSummary { n: 3, min: -0.25, max: 1.0, mean: 0.125, median: 0.0, stddev: 0.5, rms: 0.515, median_error: 0.0 };
        let t = table(&["c2c_dist".into(), "c2c_dz".into()], &[Some(s), None], 0.001);
        let lines: Vec<&str> = t.lines().collect();
        assert_eq!(lines[0], "    field        min    max    mean  median  stddev     rms");
        assert_eq!(lines[1], "    c2c_dist  -0.250  1.000  0.1250  0.0000  0.5000  0.5150");
        assert_eq!(lines[2], "    c2c_dz         -      -       -       -       -       -");
    }
}
