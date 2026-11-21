use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use copc_compare::extra_bytes::ScaledField;
use copc_compare::input::CopcFile;
use copc_compare::{c2c, info, pipeline};

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
    /// Nearest-neighbour distance from every point of the compared file to the
    /// reference file. Writes the compared file with a `c2c_dist` field added.
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
        /// close get 5 × max-dist.
        #[arg(long)]
        max_dist: f64,
        /// Distance resolution in metres (default: the finer coordinate scale of
        /// the two files).
        #[arg(long)]
        dist_scale: Option<f64>,
        /// Run even if the two files declare different CRSs.
        #[arg(long)]
        force: bool,
        /// Target number of compared + reference points per tile.
        #[arg(long, default_value_t = 8_000_000, hide = true)]
        tile_points: u64,
        /// Tiles processed at once.
        #[arg(long, default_value_t = 8, hide = true)]
        tiles_in_flight: usize,
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

/// splitmix64, for reproducible stub values.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(n) = cli.threads {
        rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;
    }
    match cli.cmd {
        Cmd::Info { file, bench } => info::run(&file, bench),
        Cmd::C2c { compared, reference, output, max_dist, dist_scale, force, tile_points, tiles_in_flight } => {
            let a = CopcFile::open(&compared)?;
            let b = CopcFile::open(&reference)?;
            let opt = c2c::Options { max_dist, dist_scale, tile_points, tiles_in_flight, force };
            c2c::run(&a, &b, &output, &opt)?;
            Ok(())
        }
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
            Ok(())
        }
    }
}
