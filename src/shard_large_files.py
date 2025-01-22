import argparse
import os
from filelock import FileLock
parser = argparse.ArgumentParser()
ap = argparse.ArgumentParser()
ap.add_argument('--input-file',help="file to process")
ap.add_argument('--output-root',help="output root path")
ap.add_argument('--split-small',action='store_true',help="split input file into max size 100K")


def count_lines(input_file):
    """Counts the number of lines in the input file."""
    with open(input_file, 'r') as infile:
        return sum(1 for _ in infile)

def split_file(args):
    print(f"Starting to process file {args.input_file}")
    max_lines_per_shard=100000
    total_lines = count_lines(args.input_file)
    if total_lines != 500000:
        print(f"Warning: The file {args.input_file} does not contain 500,000 lines. It has {total_lines} lines.")
        return args.input_file
    basename_without_suffix = os.path.splitext(os.path.basename(args.input_file))[0]

    print(f"The file contains exactly 500,000 lines. Proceeding to split it into shards...")
    
    with open(args.input_file, 'r') as infile:
        shard_number = 1
        line_count = 0
        output_file = None
        if os.path.exists(f"{args.output_root}/{basename_without_suffix}-shard-{shard_number}.jsonl"):
            print("Shards already exists")
            return None
        for line in infile:
            if line_count == 0:
                output_file = open(f"{args.output_root}/{basename_without_suffix}-shard-{shard_number}.jsonl", 'w')
            
            
            output_file.write(line)
            line_count += 1
            
           
            if line_count >= max_lines_per_shard:
                output_file.close()  
                shard_number += 1     
                line_count = 0       
        
        if output_file:
            output_file.close()
        
        print(f"File split into {shard_number - 1} shards.")
    return None

def split_file_max_100k(args):
    basename_without_suffix = os.path.splitext(os.path.basename(args.input_file))[0]
    max_lines_per_shard=100000
    total_lines = count_lines(args.input_file)
    if total_lines <= 100000:
        return args.input_file
    with open(args.input_file, 'r') as infile:
        shard_number = 1
        line_count = 0
        output_file = None
        if os.path.exists(f"{args.output_root}/{basename_without_suffix}-shard-{shard_number}.jsonl"):
            print("Shards already exists")
            return None
        for line in infile:
            if line_count == 0:
                output_file = open(f"{args.output_root}/{basename_without_suffix}-shard-{shard_number}.jsonl", 'w')
            
            output_file.write(line)
            line_count += 1
            
            if line_count >= max_lines_per_shard:
                output_file.close()  
                shard_number += 1    
                line_count = 0       
        
      
        if output_file:
            output_file.close()
    return None

if __name__ == '__main__':
    args = ap.parse_args()
    if args.split_small:
        return_val = split_file_max_100k(args)
        if return_val is not None:
            lock = FileLock("langs_under_100K_lines.txt.lock")
            if os.path.exists("langs_under_100K_lines.txt"):
                mode = "a"
            else:
                mode = "w"
            with lock:
                with open("langs_under_100K_lines.txt",mode) as out_file:
                    out_file.write(return_val + '\n')
    else:
        return_val = split_file(args)
        if return_val is not None:
            lock = FileLock("langs_under_500K_lines.txt.lock")
            if os.path.exists("langs_under_500K_lines.txt"):
                mode = "a"
            else:
                mode = "w"
            with lock:
                with open("langs_under_500K_lines.txt",mode) as out_file:
                    out_file.write(return_val + '\n')
            
