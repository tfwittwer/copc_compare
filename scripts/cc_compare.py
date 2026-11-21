"""Join CloudCompare's C2C export with ours on coordinates and compare distances."""
import sys
cc_path, ours_path, scale, max_dist = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])

def load(path, want):
    with open(path) as f:
        cols = f.readline().lstrip('/').split()
        i = next(k for k, c in enumerate(cols) if want(c))
        out = {}
        for line in f:
            v = line.split()
            out.setdefault((v[0], v[1], v[2]), float(v[i]))
        return cols[i], out

cc_name, cc = load(cc_path, lambda c: c.startswith('C2C'))
our_name, ours = load(ours_path, lambda c: c == 'c2c_dist')
common = cc.keys() & ours.keys()
print(f"columns: CC '{cc_name}', ours '{our_name}'; points CC {len(cc)}, ours {len(ours)}, joined {len(common)}")
print("sample coordinates:", next(iter(common)))
diffs, beyond_both, mismatch = [], 0, []
for k in common:
    c, o = cc[k], ours[k]
    if c >= max_dist - 1e-6 or o > max_dist:
        if c >= max_dist - 1e-6 and o > max_dist:
            beyond_both += 1
        else:
            mismatch.append((k, c, o))
        continue
    diffs.append(abs(c - o))
diffs.sort()
n = len(diffs)
tol = scale / 2 + 1e-4  # our rounding + CC's 4-decimal export
print(f"within max-dist: {n}, beyond in both: {beyond_both}, cap disagreements: {len(mismatch)} {mismatch[:3]}")
print(f"|ours - CC|: median {diffs[n//2]:.6f}, p99.9 {diffs[int(n*0.999)]:.6f}, max {diffs[-1]:.6f}; "
      f"over tolerance {tol}: {sum(d > tol for d in diffs)}")
