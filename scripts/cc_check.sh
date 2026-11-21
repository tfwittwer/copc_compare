#!/bin/bash
# CloudCompare cross-check of copc_compare c2c on a crop of cyl (compared) vs points (reference).
# Then: python3 scripts/cc_compare.py OUTDIR/cc_a.asc OUTDIR/ours.asc 0.001 1.0
# Usage: cc_check.sh OURS_OUTPUT OUTDIR
set -euo pipefail
OURS=$1; O=$2
D=${CC_DATA_DIR:?set CC_DATA_DIR to the directory with cyl.copc.laz and points.copc.laz}
CC=/usr/local/bin/CloudCompare
SHIFT="-GLOBAL_SHIFT -89000 -432000 0"
COMMON="-SILENT -AUTO_SAVE OFF -NO_TIMESTAMP -C_EXPORT_FMT ASC -ADD_HEADER -PREC 4"
# Local coordinates after the shift: A crop x 400..460, y 0..60; B crop 2 m wider.
A_BOX=400:0:-100:460:60:100
B_BOX=398:-2:-100:462:62:100
$CC $COMMON -O $SHIFT $D/cyl.copc.laz -CROP $A_BOX -O $SHIFT $D/points.copc.laz -CROP $B_BOX \
    -C2C_DIST -MAX_DIST 1 -SAVE_CLOUDS FILE "$O/cc_a.asc $O/cc_b.asc" > $O/cc_c2c.log 2>&1
$CC $COMMON -O $SHIFT $OURS -CROP $A_BOX -SAVE_CLOUDS FILE $O/ours.asc > $O/cc_ours.log 2>&1
