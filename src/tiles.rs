//! Tiles: a cut through the compared file's octree that partitions its root cube.
//!
//! Recurse from the root: a node is a tile if its subtree holds no compared
//! points, or at most `max_points` compared + reference points, or if its children
//! would be narrower than `min_side`. Otherwise all 8 children (with or without
//! entries) become candidates, so the tiles always cover the whole cube. Visiting
//! children in octant order yields Morton order.
//!
//! Reference points count because a tile decodes them and builds its kd-tree on
//! them: with a denser reference, sizing by the compared points alone gives a
//! few very large tiles. Each reference node is attributed to the compared key
//! that contains its centre at the finest level at least as wide as the node.
//!
//! A data node is *owned* by the tile at or above it, or *shared* if it lies above
//! the cut (a strict ancestor of tiles). Shared points are assigned to tiles by
//! position.

use std::collections::HashMap;

use copc_streaming::{Aabb, HierarchyEntry, VoxelKey};

use crate::input::{CopcFile, key_tuple};

type Key = (i32, i32, i32, i32);

pub struct Cut {
    /// Tile keys, in Morton order.
    pub tiles: Vec<VoxelKey>,
    /// A's data nodes in each tile's subtree, sorted by (level, x, y, z).
    pub owned: Vec<Vec<HierarchyEntry>>,
    /// A's data nodes above the cut, sorted by (level, x, y, z).
    pub shared: Vec<HierarchyEntry>,
    index: HashMap<Key, usize>,
    deepest: i32,
    root: Aabb,
}

impl Cut {
    /// `b`, if given, is the reference file, whose points count towards
    /// `max_points` too.
    pub fn new(a: &CopcFile, b: Option<&CopcFile>, max_points: u64, min_side: f64) -> Self {
        let root = a.root_bounds();
        let side0 = root.max[0] - root.min[0];
        // Points per subtree: (compared, reference).
        let mut subtree: HashMap<Key, (u64, u64)> = HashMap::new();
        let mut add = |key: VoxelKey, n: u64, is_b: bool| {
            let mut k = Some(key);
            while let Some(key) = k {
                let s = subtree.entry(key_tuple(&key)).or_default();
                if is_b { s.1 += n } else { s.0 += n }
                k = key.parent();
            }
        };
        for e in a.data_entries() {
            add(e.key, u64::from(e.point_count), false);
        }
        if let Some(b) = b {
            let b_root = b.root_bounds();
            for e in b.data_entries() {
                let bb = e.key.bounds(&b_root);
                let c: [f64; 3] = std::array::from_fn(|i| 0.5 * (bb.min[i] + bb.max[i]));
                if !contains(&root, c) {
                    continue;
                }
                let b_side = bb.max[0] - bb.min[0];
                let level = ((side0 / b_side).log2().floor().max(0.0) as i32).min(30);
                let n = 1i64 << level;
                let k: [i32; 3] = std::array::from_fn(|i| {
                    (((c[i] - root.min[i]) / (side0 / n as f64)).floor() as i64).clamp(0, n - 1) as i32
                });
                add(VoxelKey { level, x: k[0], y: k[1], z: k[2] }, u64::from(e.point_count), true);
            }
        }

        let mut tiles = Vec::new();
        let mut stack = vec![VoxelKey::ROOT];
        while let Some(k) = stack.pop() {
            let (sa, sb) = subtree.get(&key_tuple(&k)).copied().unwrap_or((0, 0));
            let child_side = side0 / f64::from(1u32 << (k.level + 1));
            if sa == 0 || sa + sb <= max_points || child_side < min_side || k.level >= 30 {
                tiles.push(k);
            } else {
                // Reversed, so octant 0 is popped (visited) first.
                stack.extend(k.children().into_iter().rev());
            }
        }

        let index: HashMap<Key, usize> = tiles.iter().enumerate().map(|(i, k)| (key_tuple(k), i)).collect();
        let mut owned = vec![Vec::new(); tiles.len()];
        let mut shared = Vec::new();
        for e in a.data_entries() {
            let mut k = Some(e.key);
            let tile = loop {
                match k {
                    Some(key) => match index.get(&key_tuple(&key)) {
                        Some(&t) => break Some(t),
                        None => k = key.parent(),
                    },
                    None => break None,
                }
            };
            match tile {
                Some(t) => owned[t].push(e.clone()),
                None => shared.push(e.clone()),
            }
        }
        let deepest = tiles.iter().map(|k| k.level).max().unwrap_or(0);
        Self { tiles, owned, shared, index, deepest, root }
    }

    /// The tile that contains `p`. Points outside the root cube go to the nearest
    /// boundary tile.
    pub fn locate(&self, p: [f64; 3]) -> usize {
        // One key at the deepest tile level, shifted for coarser levels, so the
        // descent is consistent even for points on cell boundaries.
        let n = 1i64 << self.deepest;
        let side = (self.root.max[0] - self.root.min[0]) / n as f64;
        let fine: [i64; 3] =
            std::array::from_fn(|a| (((p[a] - self.root.min[a]) / side).floor() as i64).clamp(0, n - 1));
        for level in 0..=self.deepest {
            let sh = self.deepest - level;
            let key = (level, (fine[0] >> sh) as i32, (fine[1] >> sh) as i32, (fine[2] >> sh) as i32);
            if let Some(&t) = self.index.get(&key) {
                return t;
            }
        }
        unreachable!("tiles partition the root cube")
    }

    pub fn tile_bounds(&self, t: usize) -> Aabb {
        self.tiles[t].bounds(&self.root)
    }
}

/// `b` grown by `d` on every side.
pub fn grow(b: &Aabb, d: f64) -> Aabb {
    Aabb { min: b.min.map(|v| v - d), max: b.max.map(|v| v + d) }
}

pub fn contains(b: &Aabb, p: [f64; 3]) -> bool {
    (0..3).all(|a| p[a] >= b.min[a] && p[a] <= b.max[a])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut_from(keys: &[(i32, i32, i32, i32)], deepest: i32) -> Cut {
        let tiles: Vec<VoxelKey> =
            keys.iter().map(|&(level, x, y, z)| VoxelKey { level, x, y, z }).collect();
        Cut {
            index: tiles.iter().enumerate().map(|(i, k)| (key_tuple(k), i)).collect(),
            owned: vec![Vec::new(); tiles.len()],
            shared: Vec::new(),
            tiles,
            deepest,
            root: Aabb { min: [0.0; 3], max: [8.0; 3] },
        }
    }

    #[test]
    fn locate_mixed_levels() {
        // Root split once; octant 0 split again, the rest are level-1 tiles.
        let mut keys: Vec<_> = VoxelKey::ROOT.child(0).children().iter().map(key_tuple).collect();
        keys.extend((1..8).map(|o| key_tuple(&VoxelKey::ROOT.child(o))));
        let c = cut_from(&keys, 2);
        assert_eq!(c.tiles[c.locate([0.5, 0.5, 0.5])], VoxelKey { level: 2, x: 0, y: 0, z: 0 });
        assert_eq!(c.tiles[c.locate([3.9, 2.1, 0.0])], VoxelKey { level: 2, x: 1, y: 1, z: 0 });
        assert_eq!(c.tiles[c.locate([5.0, 1.0, 1.0])], VoxelKey { level: 1, x: 1, y: 0, z: 0 });
        // On and beyond the boundary: clamped into the cube.
        assert_eq!(c.tiles[c.locate([8.0, 8.0, 8.0])], VoxelKey { level: 1, x: 1, y: 1, z: 1 });
        assert_eq!(c.tiles[c.locate([-0.001, 0.0, 0.0])], VoxelKey { level: 2, x: 0, y: 0, z: 0 });
        assert_eq!(c.tiles[c.locate([4.0, 0.0, 0.0])], VoxelKey { level: 1, x: 1, y: 0, z: 0 });
    }
}
