#!/bin/bash
#SBATCH --job-name test-vllm-hplt-70B-128
#SBATCH --account=project_462000449 #Change this
#SBATCH --partition=dev-g          
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --mem=480G
#SBATCH --exclusive 
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error

ml use /appl/local/csc/modulefiles/ #Access to the modules below
module load pytorch/2.4

#VLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn #Only method that works on rocm atm
export PYTHONWARNINGS=ignore #Remove annoying logs
export VLLM_USE_TRITON_FLASH_ATTN=0 #Use Rocm version of FA


#DISTRIBUTED
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999

#MISC
export TRANSFORMERS_CACHE=$HF_HOME #Set this explicitly to some directory if you dont have HF_HOME set up

#######
#--tensor-parallel-size should be equal to the amount of GPU's on 1 node
#--max-num-batched-tokens should be increased for performance gain
#--disable-custom-all-reduce Not sure if this is needed. Seems to get rid of some slowness
#######
#See the README for links to all of the parameters and documentation

##NOTE the following srun expects you to launch the script from the root of this repository, so ****/benchmarks

srun --label python ../src/run_vllm.py \
    --dataset /scratch/project_462000353/akselir/FineWeb-Edu-Multiling/data/hpltv2-shuffled-train.jsonl \
    --tensor-parallel-size 8 \
    --model /scratch/project_462000353/pyysalos/models/Llama-3.1-70B-Instruct\
    --num-prompts 1000 \
    --max-num-seqs 128 \
    --distributed_executor_backend mp \
    --disable-custom-all-reduce \
    --enable-chunked-prefill False \
    --output-json /scratch/project_462000353/akselir/FineWeb-Edu-Multiling/results/vllm_output_71B-hplt-128.json \
    --output-file /scratch/project_462000353/akselir/FineWeb-Edu-Multiling/results/vllm_annotations-71B-hplt-128.jsonl \
    --test 