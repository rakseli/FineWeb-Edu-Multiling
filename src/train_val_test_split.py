import argparse
import json
import os
from shuffle_rows import load_jsonl


def split_jsonl(input_file,output_path,train_ratio, val_ratio):
    lines = load_jsonl(input_file)
    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    val_end = train_end + int(total_lines * val_ratio)

    train_data = lines[:train_end]
    val_data = lines[train_end:val_end]
    test_data = lines[val_end:]
    
    file_name = os.path.basename(input_file)
    file_prefix = os.path.splitext(file_name)[0]
    
    with open(f"{output_path}/{file_prefix}-train.jsonl", "w") as f:
        for entry in train_data:
            f.write(json.dumps(entry,ensure_ascii=False) + "\n")

    with open(f"{output_path}/{file_prefix}-val.jsonl", "w") as f:
        for entry in val_data:
            f.write(json.dumps(entry,ensure_ascii=False) + "\n")

    with open(f"{output_path}/{file_prefix}-test.jsonl", "w") as f:
        for entry in test_data:
            f.write(json.dumps(entry,ensure_ascii=False) + "\n")

    print(f"Data split into:")
    print(f" - Training set: {len(train_data)} lines")
    print(f" - Validation set: {len(val_data)} lines")
    print(f" - Test set: {len(test_data)} lines")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a JSONL file into training, validation, and test sets.")
    parser.add_argument("--input_file", help="Path to the input JSONL file.")
    parser.add_argument("--output-path",help="output path prefix")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion of data for the training set (default: 0.8).")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion of data for the validation set (default: 0.1).")
    args = parser.parse_args()
    # Perform the split
    split_jsonl(args.input_file, args.output_path,args.train_ratio, args.val_ratio)
