#!/bin/bash
#SBATCH --job-name create-annotation-dataset-tel
#SBATCH --account=project_462000449
#SBATCH --partition=small     
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=10G
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error

module load LUMI/24.03  partition/C
module load zstd/1.5.5-cpeGNU-24.03
module load cray-python
set -euo pipefail
export input_root=$1
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S"), folder $input_root"
echo "Folder size $(du -h $input_root)"
input_root=$1
echo "Decompressing..."
find $input_root -name "*.zst" -print0 | xargs -P $SLURM_CPUS_PER_TASK -0 -I {} zstd -d -f {}
echo "Done"
# Target total lines
TARGET_LINES=500000  
# Initialize total line count
total_lines=0
# Loop through JSONL files and count lines
declare -A line_counts
for file in "$input_root"/*.jsonl; do
    if [[ -f "$file" ]]; then
        line_count=$(wc -l < "$file")
        line_counts["$(basename "$file")"]=$line_count
        total_lines=$((total_lines + line_count))
    fi
done

# Calculate the fraction
if (( total_lines <= TARGET_LINES )); then
    fraction=1.0
else
    fraction=$(echo "scale=6; $TARGET_LINES / $total_lines" | bc)
fi

echo "Line counts per file:"
for file in "${!line_counts[@]}"; do
    echo "$file: ${line_counts[$file]}"
done
echo "Total lines: $total_lines"
echo "Sampling fraction: $fraction"
#sample 500K lines from files
mkdir -p "$input_root/downsampled"
export fraction
mkdir -p "$input_root/downsampled"
find "$input_root" -name '*.jsonl' -type f | xargs -n 1 -P $SLURM_CPUS_PER_TASK -I {} bash -c '
    file="{}"
    output_filename=$(basename "$file" .jsonl)
    srun perl -pe '\''$_="" unless rand()<'"$fraction"'\'' "$file" > "$input_root/downsampled/${output_filename}-downsampled.jsonl"
'
#suffle the lines from chards and combine
shuffle_output_file=$(basename $input_root)
srun python ../src/shuffle_rows.py --input_files $(find "$input_root/downsampled" -type f -name "*.jsonl") --output "$input_root/downsampled/$shuffle_output_file-combined-shuffled.jsonl"
echo "Done shuffling"
#split into train val test
mkdir -p "$input_root/splitted"
srun python ../src/train_val_test_split.py --input_file "$input_root/downsampled/$shuffle_output_file-combined-shuffled.jsonl" --output-path "$input_root/splitted"
echo "Done splitting"
echo "Removing temp files.."
rm $input_root/*.jsonl
rm -rf $input_root/downsampled
echo "End $(date +"%Y-%-m-%d-%H:%M:%S")"
