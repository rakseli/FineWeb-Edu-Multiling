import os
import time
import subprocess
import argparse

parser = argparse.ArgumentParser()
ap = argparse.ArgumentParser()
ap.add_argument('--input-root',help="root for files")
ap.add_argument('--time',help="time for processing",default="24:00:00")
ap.add_argument('--partition',help="slurm partition",default="small")
ap.add_argument('--dry-run', action='store_true', help="Don't submit any jobs, just print what would be done.")
"""TODO
- make the script work dir wise to that 

"""

def create_slurm_scripts(args):
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
       
    
    
    basename_without_suffix = os.path.splitext(os.path.basename(args.input_file))[0]
    base_dir = os.path.basename(args.input_file)
    script_content = f"""#!/bin/bash
#SBATCH --job-name sample-{base_dir}-{basename_without_suffix}
#SBATCH --account=project_462000449
#SBATCH --partition={args.partition}         
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=3
#SBATCH --time={args.time}
#SBATCH --mem=3G
#SBATCH --exclusive 
#SBATCH --output=../logs/%x_%j.output
#SBATCH --error=../logs/%x_%j.error

module load LUMI/24.03  partition/C
module load zstd/1.5.5-cpeGNU-24.03
set -euo pipefail
echo "Start $(date +"%Y-%-m-%d-%H:%M:%S")"
echo "Decopmpress file {args.input_file}
srun zstd -d --rm {args.input_file}
# Calculate the number of rows in the file
line_count=$(wc -l < {base_dir}/{basename_without_suffix}.jsonl)
# Determine the fraction
if [ "$line_count" -lt 500000 ]; then
  fraction=1
else
  # Calculate the fraction to get approximately 500K lines
  fraction=$(echo "scale=8; 500000 / $line_count" | bc)
fi
srun perl -pe '$_="" unless rand()<'"$fraction" "{base_dir}/{basename_without_suffix}.jsonl" > "{base_dir}/{basename_without_suffix}-downsampled.jsonl"
echo "End $(date +"%Y-%-m-%d-%H:%M:%S")"
""" 

    return script_content

if __name__ == '__main__':
    args = ap.parse_args()
    command = create_slurm_scripts(args)
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
    
