import argparse
import os
parser = argparse.ArgumentParser()
ap = argparse.ArgumentParser()
ap.add_argument('--input-file',help="file to process")
ap.add_argument('--output-root',help="output root path")

def count_lines(input_file):
    """Counts the number of lines in the input file."""
    with open(input_file, 'r') as infile:
        return sum(1 for _ in infile)

def split_file(args):
    print(f"Starting to process file {args.input_file}")
    max_lines_per_shard=100000
    # Count the number of lines in the file
    total_lines = count_lines(args.input_file)
    # Check if the file has 500,000 lines
    if total_lines != 500000:
        print(f"Warning: The file {args.input_file} does not contain 500,000 lines. It has {total_lines} lines.")
        return None
    basename_without_suffix = os.path.splitext(os.path.basename(args.input_file))[0]

    # Proceed with splitting the file if it contains 500,000 lines
    print(f"The file contains exactly 500,000 lines. Proceeding to split it into shards...")
    
    # Open the input file for reading
    with open(args.input_file, 'r') as infile:
        # Initialize variables to track the shard number and line count
        shard_number = 1
        line_count = 0
        output_file = None
        
        # Loop through each line in the input file
        for line in infile:
            # If this is the first line or a new shard is needed
            if line_count == 0:
                # Open a new shard file for writing
                output_file = open(f"{args.output_root}/{basename_without_suffix}-shard-{shard_number}.jsonl", 'w')
            
            # Write the current line to the current shard file
            output_file.write(line)
            line_count += 1
            
            # If we've reached the max lines per shard, close the current shard file and move to the next shard
            if line_count >= max_lines_per_shard:
                output_file.close()  # Close the current shard
                shard_number += 1     # Increment the shard number
                line_count = 0       # Reset the line count for the new shard
        
        # Close the last shard file if it's open
        if output_file:
            output_file.close()
        
        print(f"File split into {shard_number - 1} shards.")

# Example usage:
if __name__ == '__main__':
    args = ap.parse_args()
    split_file(args)