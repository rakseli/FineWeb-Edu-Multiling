#!/bin/bash
#SBATCH --job-name download_hplt
#SBATCH --account=project_462000449
#SBATCH --partition=small     
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error

module load LUMI/24.03  partition/C
module load wget/1.21.4-cpeGNU-24.03

echo "Starting $(date)"
xargs -P $SLURM_CPUS_PER_TASK -n 1 wget -nH -q -x --cut-dirs=2 --directory-prefix /scratch/project_462000353/data/hplt/all_languages  < /scratch/project_462000353/data/hplt/all_languages/hplt_monolingual_map_cleaned_2.0.txt
echo "End $(date)"
