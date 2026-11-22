use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use copc_compare::extra_bytes::ScaledField;
use copc_compare::input::CopcFile;
use copc_compare::{c2c, diff, info, pipeline};

#[derive(Parser)]
#[command(version, about = "Cloud-to-cloud distances between COPC files, written out as COPC")]
struct Cli {
    /// Worker threads (default: all cores).
    #[arg(long, global = true)]
    threads: Option<usize>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Distance from every point of the compared file to the reference file.
    /// Writes the compared file (same octree) with `c2c_dist` added, and
    /// optionally `c2c_dx`/`c2c_dy`/`c2c_dz` and `c2c_plane`.
    C2c {
        /// File whose points get a distance (its octree is kept).
        #[arg(long)]
        compared: PathBuf,
        /// File the distances are measured to.
        #[arg(long)]
        reference: PathBuf,
        /// Output COPC file.
        #[arg(short, long)]
        output: PathBuf,
        /// Maximum distance in metres. Points without a reference neighbour this
        /// close get 5 × max-dist in every field.
        #[arg(long)]
        max_dist: f64,
        /// Distance resolution in metres (default: the finer coordinate scale of
        /// the two files, but at least 0.1 mm).
        #[arg(long)]
        dist_scale: Option<f64>,
        /// Also write the offset to the nearest neighbour, compared minus
        /// reference, as c2c_dx, c2c_dy, c2c_dz (positive dz: compared lies above).
        #[arg(long)]
        components: bool,
        /// `plane` also writes c2c_plane: the signed distance to a least-squares
        /// plane through the k nearest reference points within max-dist (normal
        /// pointing up). Less sensitive to the reference's density and noise.
        #[arg(long, value_enum, default_value_t = Model::Nn)]
        model: Model,
        /// Neighbours for the plane model.
        #[arg(long, default_value_t = 8)]
        k: usize,
        /// Memory budget, e.g. 16G or 4096M (default: half the RAM). Sets the
        /// tile size and how many tiles run at once.
        #[arg(long, value_parser = parse_size)]
        memory_limit: Option<u64>,
        /// Run even if the two files declare different CRSs.
        #[arg(long)]
        force: bool,
        /// No progress bar.
        #[arg(long)]
        no_progress: bool,
        /// Target number of compared + reference points per tile (overrides the
        /// memory-derived value).
        #[arg(long, hide = true)]
        tile_points: Option<u64>,
        /// Tiles processed at once (overrides the memory-derived value).
        #[arg(long, hide = true)]
        tiles_in_flight: Option<usize>,
    },
    /// Whether two COPC files hold the same points, independent of point order
    /// and octree. Compares the position and every field both files have, and
    /// lists the points that differ. Exit status 0 if the point sets are
    /// identical, 1 if not, 2 on error.
    Diff {
        x: PathBuf,
        y: PathBuf,
        /// Compare only these fields besides the position (comma-separated):
        /// gps_time, intensity, returns, channel, flags, classification,
        /// user_data, scan_angle, point_source_id, rgb, nir, extra bytes field
        /// names, or `extra` for all extra bytes fields. `--fields xyz` compares
        /// the position only.
        #[arg(long, value_delimiter = ',')]
        fields: Option<Vec<String>>,
        /// Don't compare these fields (comma-separated, names as for --fields).
        #[arg(long, value_delimiter = ',')]
        ignore: Vec<String>,
        /// Only say whether and where (which tiles) the files differ, without
        /// finding the points.
        #[arg(long)]
        quick: bool,
        /// Examples to show per kind of difference.
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// No progress bar.
        #[arg(long)]
        no_progress: bool,
        /// Target X + Y points per tile.
        #[arg(long, hide = true, default_value_t = diff::DEFAULT_TILE_POINTS)]
        tile_points: u64,
    },
    /// Show header, VLRs, COPC info and per-level hierarchy statistics.
    Info {
        file: PathBuf,
        /// Also time a full decode of every node with all fields and with XYZ only.
        #[arg(long)]
        bench: bool,
    },
    /// Development: rewrite a COPC file keeping its octree, adding scaled-i32
    /// fields that hold a constant or random values. Tests the writer.
    #[command(hide = true)]
    Passthrough {
        input: PathBuf,
        output: PathBuf,
        /// Number of fields to add (0 re-encodes without adding any).
        #[arg(long, default_value_t = 1)]
        fields: usize,
        /// Field scale.
        #[arg(long, default_value_t = 0.001)]
        scale: f64,
        /// Constant value for every point.
        #[arg(long, default_value_t = 0.0, conflicts_with = "random_max")]
        value: f64,
        /// Uniform random values in [0, RANDOM_MAX] instead, seeded per node.
        #[arg(long)]
        random_max: Option<f64>,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Model {
    /// Nearest neighbour only.
    Nn,
    /// Nearest neighbour plus a local plane fit.
    Plane,
}

/// A byte count with an optional K/M/G/T suffix (powers of 1024).
fn parse_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    let (num, mul) = match s.char_indices().last() {
        Some((i, c)) if c.is_ascii_alphabetic() => {
            let mul = match c.to_ascii_uppercase() {
                'K' => 1u64 << 10,
                'M' => 1 << 20,
                'G' => 1 << 30,
                'T' => 1 << 40,
                _ => return Err(format!("unknown size suffix in '{s}'")),
            };
            (&s[..i], mul)
        }
        _ => (s, 1),
    };
    let v: f64 = num.trim().parse().map_err(|_| format!("not a size: '{s}'"))?;
    if v.is_nan() || v <= 0.0 {
        return Err(format!("size must be positive: '{s}'"));
    }
    Ok((v * mul as f64) as u64)
}

/// splitmix64, for reproducible stub values.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    // diff follows cmp/diff: 1 means "different", so errors exit with 2.
    let error_code = if matches!(cli.cmd, Cmd::Diff { .. }) { 2 } else { 1 };
    match run(cli) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("Error: {e:?}");
            ExitCode::from(error_code)
        }
    }
}

/// Ok(false) only for a diff that found differences.
fn run(cli: Cli) -> Result<bool> {
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;
    }
    match cli.cmd {
        Cmd::C2c {
            compared,
            reference,
            output,
            max_dist,
            dist_scale,
            components,
            model,
            k,
            memory_limit,
            force,
            no_progress,
            tile_points,
            tiles_in_flight,
        } => {
            let a = CopcFile::open(&compared)?;
            let b = CopcFile::open(&reference)?;
            let limit = memory_limit.unwrap_or_else(c2c::default_memory_limit);
            let (t, n) = c2c::tiling_for_memory(limit, rayon::current_num_threads());
            let opt = c2c::Options {
                max_dist,
                dist_scale,
                components,
                plane_k: matches!(model, Model::Plane).then_some(k),
                tile_points: tile_points.unwrap_or(t),
                tiles_in_flight: tiles_in_flight.unwrap_or(n),
                force,
                progress: !no_progress,
            };
            c2c::run(&a, &b, &output, &opt)?;
            Ok(true)
        }
        Cmd::Diff { x, y, fields, ignore, quick, limit, no_progress, tile_points } => {
            let (x, y) = (CopcFile::open(&x)?, CopcFile::open(&y)?);
            let opt = diff::Options { fields, ignore, tile_points, quick, limit, progress: !no_progress };
            Ok(diff::run(&x, &y, &opt)?.equal)
        }
        Cmd::Info { file, bench } => info::run(&file, bench).map(|()| true),
        Cmd::Passthrough { input, output, fields, scale, value, random_max } => {
            let a = CopcFile::open(&input)?;
            let fields: Vec<ScaledField> = (0..fields)
                .map(|i| ScaledField::new(&format!("stub_{i}"), "passthrough test value", scale))
                .collect::<Result<_>>()?;
            for f in &fields {
                f.check_range(random_max.unwrap_or(value).abs())?;
            }
            let n = fields.len();
            pipeline::passthrough(&a, &output, &fields, |e, chunk| {
                let k = &e.key;
                let mut seed = ((k.level as u64) << 48) ^ ((k.x as u64) << 32) ^ ((k.y as u64) << 16) ^ (k.z as u64);
                (0..chunk.point_count() * n)
                    .map(|_| match random_max {
                        Some(m) => fields[0].encode((splitmix(&mut seed) >> 11) as f64 / (1u64 << 53) as f64 * m),
                        None => fields[0].encode(value),
                    })
                    .collect()
            })?;
            Ok(true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_size;

    #[test]
    fn sizes() {
        assert_eq!(parse_size("16G"), Ok(16 << 30));
        assert_eq!(parse_size("512m"), Ok(512 << 20));
        assert_eq!(parse_size("1.5G"), Ok(3 << 29));
        assert_eq!(parse_size("1000"), Ok(1000));
        assert!(parse_size("12X").is_err() && parse_size("-1G").is_err() && parse_size("G").is_err());
    }
}
