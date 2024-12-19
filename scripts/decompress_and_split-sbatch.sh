#!/bin/bash
#SBATCH --job-name decompress_and_split
#SBATCH --account=project_462000449
#SBATCH --partition=small     
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --mem=10G
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error

module load LUMI/24.03  partition/C
module load zstd/1.5.5-cpeGNU-24.03
input_root=$1
set -euo pipefail
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S"), folder $input_root"
echo "Decompressing..."
find $input_root -name "*.zst" -print0 | xargs -P $SLURM_CPUS_PER_TASK -0 -I {} zstd -d  -f {}
echo "Done"
echo "Splitting..."
mkdir -p "$input_root/500K"
find "$input_root/" -name "*.jsonl" -print0 | xargs -P $SLURM_CPUS_PER_TASK -I{} bash -c 'split -l 500000 "{}" "$input_root/500K/$(basename "{}" .jsonl).split.jsonl"'
echo "End $(date +"%Y-%-m-%d-%H:%M:%S")"
