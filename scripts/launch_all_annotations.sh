#!/bin/bash
BASE_DIR="/scratch/project_462000353/data/hplt/all_languages"
jobs_submitted=0
for dir in $(find "$BASE_DIR" -maxdepth 1 -type d); do
    if [ -d "$dir" ]; then
        if [ -d "$dir/splitted" ]; then
            echo "Skipping $dir: splitted already exists"
        else
            echo "Submitting job for $dir"
            # sbatch create_annotation_datasets-sbatch.sh $dir
            ((jobs_submitted++))
        fi
    else
        echo "Not proper dir: $dir"
    fi
done
echo "Total jobs submitted: $jobs_submitted"