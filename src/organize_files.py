import os
import re
import shutil
import argparse

# Define the regex pattern
shard_pattern = re.compile(r"vllm-annotations-Llama-3.3-70B-Instruct-(?P<lang>\w+)_(?P<script>\w+)\.split-(?:shard-(?P<shard>\d+))?-64-rep-penalty-1.0\.jsonl$")
pattern = re.compile(r"vllm-annotations-Llama-3.3-70B-Instruct-(?P<lang>\w+)_(?P<script>\w+)\.split-64-rep-penalty-1.0.jsonl$")

parser = argparse.ArgumentParser()
ap = argparse.ArgumentParser()
ap.add_argument('--input-root',help="file root",default="../results/hplt-annotations")
ap.add_argument('--test',action="store_true")
print_output = False
if __name__ == '__main__':
    args = ap.parse_args()
    for filename in os.listdir(args.input_root):
        file_path = os.path.join(args.input_root, filename)
        if not os.path.isfile(file_path):
            continue
        output_dir = f"{args.input_root}/data"
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        match = shard_pattern.match(filename)
        if not match:
            match = pattern.match(filename)
        if match:
            lang = match.group("lang")
            script = match.group("script")
            if "shard" in filename:
                shard = match.group("shard") or "unknown"
            else:
                shard = 0
            dir_name = f"{lang}-{script}"
            target_dir = os.path.join(output_dir, dir_name)
            new_filename = f"{script}-{lang}-annotations-llama3.3-70b-instruct-shard-{shard}.jsonl"
            target_path = os.path.join(target_dir, new_filename)
            if not args.test:
                os.makedirs(target_dir, exist_ok=True)
                shutil.move(file_path, target_path)
                print(f"Moved: {filename} -> {target_path}")

            else:
                if print_output:
                    print(f"Would have done dir: {target_dir}")
                    print(f"Would have moved: {filename} -> {target_path}")

    print("Organization complete.")