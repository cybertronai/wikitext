#!/bin/bash
# Per-K learning curves. Run sequentially so we don't thrash memory.
set -e
cd "$(dirname "$0")/../.."

for K in 4 5 7; do
    echo "=========================================================="
    echo "=== K=$K"
    echo "=========================================================="
    # Arena sizes scale with K. K=4: ~200K nodes/10M. K=5: ~700K. K=7: ~14M.
    case $K in
        4) NODES=10000000 ; ENT=100000000 ; EVAL=10000000 ;;
        5) NODES=20000000 ; ENT=200000000 ; EVAL=20000000 ;;
        7) NODES=60000000 ; ENT=600000000 ; EVAL=20000000 ;;
    esac
    python3 experiments/ppm_c/run.py --K $K \
        --max-seconds 290 \
        --max-nodes $NODES \
        --max-entries $ENT \
        --chunk-bytes 20000000 \
        --eval-every-bytes $EVAL \
        --no-online \
        2>&1 | grep -E "^\[curve|^\[train\] done|^  K |val_char_acc"
done
