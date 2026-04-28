#!/bin/bash
# Standardized Baseline Runner - 5 organs × 3 views = 15 experiments
# Each experiment: 3000 iterations, eval at 1000/2000/3000

PYTHON=/home/qyhu/anaconda3/envs/cu116torch112_1/bin/python
BASE_DIR=/home/qyhu/SPAGS
DATA_DIR=$BASE_DIR/data/spags_format
OUTPUT_DIR=$BASE_DIR/output/baseline

mkdir -p $OUTPUT_DIR

# Dataset configurations: organ view_count
datasets=(
    "chest 3"
    "chest 6"
    "chest 9"
    "foot 3"
    "foot 6"
    "foot 9"
    "head 3"
    "head 6"
    "head 9"
    "jaw 3"
    "jaw 6"
    "jaw 9"
    "pancreas 3"
    "pancreas 6"
    "pancreas 9"
)

for ds in "${datasets[@]}"; do
    read -r organ views <<< "$ds"
    name="${organ}_50_${views}views"
    echo "[$(date '+%H:%M')] 🚀 STARTING Baseline $name"
    
    cd $BASE_DIR
    $PYTHON train.py \
        -s "$DATA_DIR/$name" \
        -m "$OUTPUT_DIR/$name" \
        --iterations 3000 \
        --test_iterations 1000 2000 3000 \
        --eval \
        > "$OUTPUT_DIR/${name}.log" 2>&1
    
    EXIT=$?
    if [ $EXIT -eq 0 ]; then
        # Extract PSNR
        YML="$OUTPUT_DIR/$name/eval/iter_003000/eval2d_render_test.yml"
        if [ -f "$YML" ]; then
            PSNR=$(grep "psnr_2d:" "$YML" | head -1 | awk '{print $2}')
            SSIM=$(grep "ssim_2d:" "$YML" | head -1 | awk '{print $2}')
            echo "[$(date '+%H:%M')] ✅ $name: PSNR=$PSNR, SSIM=$SSIM"
            echo -e "$(cd $BASE_DIR && git rev-parse --short HEAD)\tbaseline_$name\t$name\t$PSNR\t$SSIM\t-\t-\t-\t-\tkeep\tbaseline $name 3000it" >> $BASE_DIR/cc-agent/autoresearch/results.tsv
        else
            echo "[$(date '+%H:%M')] ⚠️ $name: no eval results"
        fi
    else
        echo "[$(date '+%H:%M')] ❌ $name: FAILED (exit=$EXIT)"
        tail -5 "$OUTPUT_DIR/${name}.log"
    fi
done

echo ""
echo "=========================================="
echo "✅ ALL 15 BASELINES COMPLETE!"
echo "=========================================="
echo ""
echo "Results:"
grep "psnr_2d:" $(find $OUTPUT_DIR -name "eval2d_render_test.yml" 2>/dev/null) | sort
