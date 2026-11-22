# copc_compare

Cloud-to-cloud (C2C) distances between two [COPC](https://copc.io) files, written
out as COPC again. Both clouds may be larger than RAM.

```
copc_compare c2c --compared A.copc.laz --reference B.copc.laz -o out.copc.laz --max-dist 1
```

For every point of A (the compared file), this finds the nearest point of B (the
reference file). It writes A back out with the distance added as an Extra Bytes
field. The output keeps A's octree, points and point order, so it streams and
displays like A. The new fields are spliced into A's compressed chunks as extra
LAZ layers, so A's own fields are never re-encoded.

On a 32-core machine, 3.05 billion compared points against 789 million reference
points take 136 s.

A second command, `diff`, checks whether two COPC files hold the same points,
regardless of point order and octree layout.

## Download

Prebuilt binaries are on the
[Releases](https://github.com/tfwittwer/copc_compare/releases) page:

- **Linux x86-64:** needs glibc 2.35 or newer and a CPU with AVX2 (x86-64-v3).
- **macOS Apple Silicon:** unsigned. If macOS blocks it, run
  `xattr -d com.apple.quarantine copc_compare`.
- **Windows x86-64:** needs a CPU with AVX2 (x86-64-v3).

For other machines, build from source.

## Build

```
cargo build --release
```

- `.cargo/config.toml` sets `target-cpu=native`, so the binary is built for the
  machine it's compiled on.
- The default `simd` feature uses kiddo's handwritten SIMD kd-tree scans, which
  need a nightly toolchain. On stable, build with `--no-default-features`.

## `c2c`

| Option | |
|---|---|
| `--max-dist D` | Required. Points without a reference point within D metres get 5 × D in every field. D also sets the tile overlap. |
| `--components` | Also write `c2c_dx`, `c2c_dy`, `c2c_dz`. |
| `--model plane [--k 8]` | Also write `c2c_plane`. |
| `--dist-scale S` | Distance resolution in metres. Default: the finer coordinate scale of the two files, but at least 0.1 mm. |
| `--memory-limit 16G` | Memory budget. Default: half the RAM on Linux, 8 GiB on macOS and Windows. It sets the tile size and how many tiles run at once. |
| `--threads N` | Worker threads. Default: all cores. |
| `--force` | Run even if the two files declare different CRSs. |

### Output fields

The fields are appended after any extra bytes A already has:

| Field | Written | Meaning (metres) |
|---|---|---|
| `c2c_dist` | always | Distance to the nearest reference point |
| `c2c_dx`, `c2c_dy`, `c2c_dz` | `--components` | A minus the nearest reference point; dz > 0 means A lies above |
| `c2c_plane` | `--model plane` | Signed distance to a least-squares plane through the k nearest reference points within max-dist, normal pointing up. Falls back to `c2c_dist` where no plane can be fitted. |

- **Storage:** each field is an i32 with a scale in its Extra Bytes descriptor, so
  the value in metres is raw × scale. The descriptor also holds the field's min
  and max.
- **Beyond max-dist:** points with no reference point within max-dist store
  5 × max-dist in every field instead of a no-data value, so they stay visible in
  viewers. The descriptor's description string says so.
- **Viewers:** CloudCompare shows the fields as scalar fields in metres.

### How it works

- **Tiles:** A's octree is cut into tiles sized by both files' point counts. Each
  tile decodes the reference points within reach and builds an f64 kd-tree. A
  coarse occupancy grid skips queries that have nothing within max-dist.
- **Output:** tiles run in parallel with bounded memory and are written in a fixed
  order, so the output is byte-identical across runs.
- **Header:** A's header, VLRs, CRS and octree are kept. The hierarchy is written
  as pages, every 3 levels.

## `diff`

```
copc_compare diff X.copc.laz Y.copc.laz [--fields LIST] [--ignore LIST] [--quick] [--limit 10]
```

Exit status: 0 if the two point sets are identical, 1 if not, 2 on error.

- **Fields:** positions and every field both files have are compared, extra bytes
  fields matched by name. If the scales differ, positions are compared on the
  coarser grid.
- **Method:** a hash sum per tile finds the regions that differ. Only those are
  decoded again and matched point by point.
- **Report:** points only in X, points only in Y, and points that changed, with
  counts per field and examples.

## `info`

```
copc_compare info FILE [--bench]
```

Prints the header, VLRs, COPC info and per-level hierarchy statistics. `--bench`
also times a full decode.

## Tests

```
cargo test --release
```

- The tests build their synthetic clouds with
  [copc-converter](https://github.com/360-geo/copc-converter), fetched by Cargo.
- They also read two files from copc-converter's `tests/data`, which isn't in its
  crates.io package. Either check copc-converter out at `../tools/copc-converter`,
  or point `COPC_CONVERTER_DATA` at a checkout's `tests/data`.
- Outputs are read back through copc-streaming and copc-reader and checked with a
  COPC validator. Distances are compared against a brute-force oracle.

## License

MIT
