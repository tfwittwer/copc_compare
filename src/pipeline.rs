//! Bounded parallel scheduling with in-order delivery, and the pass-through run.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Result, anyhow};
use copc_streaming::{Chunk, HierarchyEntry};

use crate::extra_bytes::ScaledField;
use crate::input::CopcFile;
use crate::write::{CopcWriter, WriteSummary};

/// Run `work` on every item on the rayon pool and hand the results to `sink` on
/// the calling thread, in item order. At most `max_running` items run at once,
/// and items are started at most `2 × max_running` ahead of the next one to
/// deliver, so up to `2 × max_running - 1` finished results can wait in the
/// reorder buffer behind a slow item instead of stalling the others. Both bound
/// memory. The first error stops new work and is returned once running items
/// have finished.
///
/// Must not be called from inside a rayon worker: the calling thread blocks
/// while it waits for results.
pub fn ordered_parallel<T, R>(
    items: &[T],
    max_running: usize,
    work: impl Fn(usize, &T) -> Result<R> + Sync,
    mut sink: impl FnMut(usize, R) -> Result<()>,
) -> Result<()>
where
    T: Sync,
    R: Send,
{
    let max_running = max_running.max(1);
    let window = 2 * max_running;
    let (tx, rx) = mpsc::channel::<(usize, Result<R>)>();
    let work = &work;
    rayon::in_place_scope(|s| {
        let mut next_spawn = 0;
        let mut next_emit = 0;
        let mut pending: BTreeMap<usize, R> = BTreeMap::new();
        let mut in_flight = 0usize;
        while next_emit < items.len() {
            while next_spawn < items.len() && in_flight < max_running && next_spawn < next_emit + window {
                let tx = tx.clone();
                let i = next_spawn;
                s.spawn(move |_| {
                    // A panic must still send, or the receive below never returns.
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        work(i, &items[i])
                    }))
                    .unwrap_or_else(|_| Err(anyhow!("worker panicked on item {i}")));
                    let _ = tx.send((i, r));
                });
                next_spawn += 1;
                in_flight += 1;
            }
            let (i, r) = rx.recv().map_err(|_| anyhow!("worker channel closed"))?;
            in_flight -= 1;
            pending.insert(i, r?);
            while let Some(r) = pending.remove(&next_emit) {
                sink(next_emit, r)?;
                next_emit += 1;
            }
        }
        debug_assert_eq!(in_flight, 0);
        Ok(())
    })
}

/// Phase 1 pass-through: decode every node of `a`, append the i32 values that
/// `values` returns for it (point-major, one per field), splice or (with
/// `reencode`, or if A's chunks can't be spliced) re-encode, and write.
/// Exercises the writer without any distance code.
pub fn passthrough(
    a: &CopcFile,
    out: &Path,
    fields: &[ScaledField],
    reencode: bool,
    values: impl Fn(&HierarchyEntry, &Chunk) -> Vec<i32> + Sync,
) -> Result<WriteSummary> {
    let t = Instant::now();
    let mut writer = CopcWriter::create(out, a, fields)?;
    let mut layout = writer.layout().clone();
    if reencode {
        layout = layout.reencoding();
    }
    let nodes: Vec<&HierarchyEntry> = a.data_entries().collect();
    let mut stats = vec![(i32::MAX, i32::MIN); fields.len()];
    ordered_parallel(
        &nodes,
        4 * rayon::current_num_threads(),
        |_, e| {
            let compressed = a.compressed(e);
            let chunk = a.decode(e, layout.decode_fields(compressed))?;
            let vals = values(e, &chunk);
            let bytes = layout.encode(compressed, &chunk, &vals)?;
            let mut s = vec![(i32::MAX, i32::MIN); fields.len()];
            for p in vals.chunks_exact(fields.len().max(1)) {
                for (s, &v) in s.iter_mut().zip(p) {
                    *s = (s.0.min(v), s.1.max(v));
                }
            }
            Ok((bytes, s))
        },
        |i, (bytes, s)| {
            for (a, b) in stats.iter_mut().zip(s) {
                *a = (a.0.min(b.0), a.1.max(b.1));
            }
            writer.write_chunk(nodes[i].key, nodes[i].point_count, &bytes)
        },
    )?;
    let summary = writer.finish(a, (!fields.is_empty()).then_some(&stats[..]))?;
    eprintln!(
        "wrote {} points in {} chunks, {:.1} MB, in {:.2} s",
        summary.points,
        summary.chunks,
        summary.bytes as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn delivers_in_order_with_bounded_running() {
        let items: Vec<usize> = (0..200).collect();
        let running = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let mut got = Vec::new();
        ordered_parallel(
            &items,
            5,
            |i, &x| {
                let r = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(r, Ordering::SeqCst);
                // Later items finish first, to exercise the reorder buffer.
                std::thread::sleep(std::time::Duration::from_micros(((200 - i) % 7) as u64 * 50));
                running.fetch_sub(1, Ordering::SeqCst);
                Ok(x * 2)
            },
            |i, r| {
                assert_eq!(r, i * 2);
                got.push(r);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(got, (0..200).map(|x| x * 2).collect::<Vec<_>>());
        assert!(peak.load(Ordering::SeqCst) <= 5);
    }

    #[test]
    fn propagates_errors_and_panics() {
        let items: Vec<usize> = (0..50).collect();
        let e = ordered_parallel(&items, 4, |_, &x| if x == 17 { Err(anyhow!("bad {x}")) } else { Ok(x) }, |_, _| Ok(()));
        assert_eq!(e.unwrap_err().to_string(), "bad 17");
        let e = ordered_parallel(&items, 4, |_, &x| if x == 9 { panic!("boom") } else { Ok(x) }, |_, _| Ok(()));
        assert!(e.unwrap_err().to_string().contains("panicked on item 9"));
        let e = ordered_parallel(&items, 4, |_, &x| Ok(x), |i, _| if i == 3 { Err(anyhow!("sink")) } else { Ok(()) });
        assert_eq!(e.unwrap_err().to_string(), "sink");
    }
}
