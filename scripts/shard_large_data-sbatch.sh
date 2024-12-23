#!/bin/bash
#SBATCH --job-name shard_file
#SBATCH --account=project_462000449
#SBATCH --partition=small     
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --mem=10G
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error
module load cray-python
srun python ../src/shard_large_files.py --input-file $1 --output-root /scratch/project_462000353/data/hplt/sample/per_lang_1M/500K/large_sharded
