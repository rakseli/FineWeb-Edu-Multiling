import json
import argparse
import textwrap
import os

def filter_questions(input_file):
    selected_rows = []
    basename_without_suffix = os.path.splitext(os.path.basename(args.input_file))[0]
    base_dir = os.path.dirname(input_file)
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            print("---------------------------------------------------------------")
            print(f"Sample {row['idx']} promt:",  textwrap.fill(row.get("prompt", "No prompt field found"), width=50))
            keep = input("Enter 2 to skip row,1 to continue, 0 to interup: ").strip()
            # Only save if user entered '1'
            if keep == '0':
                break
            if keep == '2':
                continue
            print(f"Sample {row['idx']} generated text:", textwrap.fill(row.get("generated_text", "No generated_text field found"), width=50))
            print(f"Sample {row['idx']} score:", row.get("score", "No score field found"))
            annot = input("Agree with score? 1 yes , 0 no: ").strip()
            com = input("Comment?").strip()
            d = {"idx":row['idx'],"agree_with_llama":int(annot),"comment":com}
            selected_rows.append(d)
    output_file = f"{base_dir}/{basename_without_suffix}-manual-annotation-{len(selected_rows)}-rows.jsonl"
    with open(output_file, "w", encoding='utf-8') as f:
        for row in selected_rows:
            f.write(json.dumps(row,ensure_ascii=False) + "\n")

    print(f"Selected questions saved to {output_file}")

parser = argparse.ArgumentParser()
ap = argparse.ArgumentParser()
ap.add_argument('--input-file')
if __name__ == "__main__":
    args = ap.parse_args()
    filter_questions(args.input_file)
