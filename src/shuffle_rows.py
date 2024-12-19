import json
import random
import argparse

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def main(input_files, output_file):
    """Merge and shuffle JSONL files."""
    merged_data = []
    seed = 42
    random.seed(42)
    # Load and combine data from all input files
    for file in input_files:
        merged_data.extend(load_jsonl(file))
    
    # Shuffle the combined data
    random.shuffle(merged_data)
    
    # Write the shuffled data to the output file
    with open(output_file, "w") as f:
        for entry in merged_data:
            f.write(json.dumps(entry,ensure_ascii=False) + "\n")
    
    print(f"Merged and shuffled data written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and shuffle JSONL files.")
    parser.add_argument("--input_files", nargs="+", help="Paths to input JSONL files.")
    parser.add_argument("--output", required=True, help="Path to output JSONL file.")
    args = parser.parse_args()
    main(args.input_files, args.output)

