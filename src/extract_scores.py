import re
import argparse
import json
import os 
parser = argparse.ArgumentParser()
ap = argparse.ArgumentParser()
ap.add_argument('--input-file',help="file where to add score field")
ap.add_argument('--text-field',help="field where to extract score",default="generated_text")
ap.add_argument('--test',help="wheter to test regex",action='store_true')
RE = re.compile(r"\**Education(al)? [Ss]core\s*[:=-]\**\s*\**(\d+)\**")
RE_alternative = re.compile(r"The final answer is: \$\\boxed{(\d+)}\$")
def extract_score(sample,test=False):
    match = RE.search(sample)
    if test:
        if match:
            return int(match.group(2)),match
        alt_match = RE_alternative.search(sample)
        if alt_match:
            return int(alt_match.group(1)),alt_match
        return -1, None
    if match:
        return int(match.group(2))
    alt_match = RE_alternative.search(sample)
    if alt_match:
        return int(alt_match.group(1))
    return -1

if __name__ == "__main__":
    args = ap.parse_args()
    basename_without_suffix = os.path.splitext(os.path.basename(args.input_file))[0]
    dir_name = os.path.dirname(args.input_file)
    with open(args.input_file, 'r', encoding='utf-8') as s, open(f"{dir_name}/{basename_without_suffix}-scores.jsonl", 'w', encoding='utf-8') as outfile:
        for line in s:
            data = json.loads(line)
            if not args.test:
                data["score"]=extract_score(data[args.text_field])
                outfile.write(json.dumps(data,ensure_ascii=False) + '\n')
            else:
                res,match = extract_score(data[args.text_field],test=args.test)
                print(match,res)
