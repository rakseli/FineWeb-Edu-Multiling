import os
import time
import subprocess
import argparse
import glob
parser = argparse.ArgumentParser()
ap = argparse.ArgumentParser()
ap.add_argument('--input-root',help="file to process")
ap.add_argument('--time',help="time for processing",default="2:00:00")
ap.add_argument('--partition',help="slurm partition",default="dev-g")
ap.add_argument('--rep-penalty',help="repetition penalty",default=1.0)
ap.add_argument('--dry-run', action='store_true', help="Don't submit any jobs, just print what would be done.")
ap.add_argument('--test', action='store_true', help="launch one test job")


def create_slurm_scripts(input_file,args):
    """Creates a slurm script in right string format

    Args:
        script_name (str): name for log files and slurm 
        lang (str): lang to dedup
        log_path (str): path where logs are saved. Defaults to 
        account (str): billing account Defaults to "".
        cpus_per_task (int): n cpus. Defaults to 1.
        time (str): running time give in HH:MM:SS format. Defaults to "00:05:00".
        mem_per_cpu (int): mem per cpu in mb. Defaults to 100.
        partition (str): partition to run the scirpt. Defaults to 'small'.
    Returns:
    - str: the script
    """
       
    
    basename_without_suffix = os.path.splitext(os.path.basename(input_file))[0]
    max_seq = 64
    model = "/scratch/project_462000353/models/Llama-3.3-70B-Instruct"
    model_name = os.path.basename(model)
    script_content = f"""#!/bin/bash
#SBATCH --job-name vllm-{basename_without_suffix}-{model_name}
#SBATCH --account=project_462000449
#SBATCH --partition={args.partition}         
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --time={args.time}
#SBATCH --mem=480G
#SBATCH --exclusive 
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error

ml use /appl/local/csc/modulefiles/ 
module load pytorch/2.4
source /scratch/project_462000353/akselir/venvs/transformers-4.47.1/bin/activate
#VLLM
export VLLM_WORKER_MULTIPROC_METHOD=spawn #Only method that works on rocm atm
export PYTHONWARNINGS=ignore #Remove annoying logs
export VLLM_USE_TRITON_FLASH_ATTN=0 #Use Rocm version of FA


#DISTRIBUTED
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999

#MISC
export TRANSFORMERS_CACHE=$HF_HOME #Set this explicitly to some directory if you dont have HF_HOME set up
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"

srun --label python ../src/run_vllm.py \
    --dataset {input_file} \
    --tensor-parallel-size 4 \
    --model {model} \
    --max-num-seqs {max_seq} \
    --distributed-executor-backend mp \
    --disable-custom-all-reduce \
    --enable-chunked-prefill False \
    --output-json ../results/hplt-annotations/vllm-throughput-{model_name}-{basename_without_suffix}-{max_seq}-rep-penalty-{args.rep_penalty}.json \
    --output-file ../results/hplt-annotations/vllm-annotations-{model_name}-{basename_without_suffix}-{max_seq}-rep-penalty-{args.rep_penalty}.jsonl \
    --max-model-len 50000 \
    --rep-penalty {args.rep_penalty} \
    --test
echo "End $(date +"%Y-%-m-%d-%H:%M:%S")"
""" 
    if os.path.exists(f"../results/hplt-annotations/vllm-annotations-{model_name}-{basename_without_suffix}-{max_seq}-rep-penalty-{args.rep_penalty}.jsonl"):
        return None
    return script_content

if __name__ == '__main__':
    args = ap.parse_args()
    input_files = glob.glob(f"{args.input_root}/*.jsonl")
    job_count = 0
    for i,f in enumerate(input_files,start=1):
        command = create_slurm_scripts(f,args)
        if command is None:
            print(f"Output file for {f} already exists, skipping...")
            continue
        if args.dry_run:
            print(command)
        else:
            temp_file_name = f"{os.getcwd()}/temp_slurm_job.sh"
            with open(temp_file_name,"w") as temp_file:
                temp_file.write(command)
                # Submit the SLURM job using sbatch with the temporary file
            result=subprocess.run(["sbatch", temp_file_name], text=True)
            print(result)
            time.sleep(1)
            os.remove(temp_file_name)
            job_count+=1
            if args.test:
                break
    print(f"Launched {job_count} jobs")
    
