import json
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from statistics import mean
from collections import Counter
from sklearn.metrics import confusion_matrix

def load_jsonl(file_path):
    data = {}
    if 'manual-annotation' in file_path:
        score_field = 'agree_with_llama'
    else:
        score_field = 'score'
    with open(file_path, 'r') as f:
        for i,line in enumerate(f):
            item = json.loads(line)
            try:
                idx = item['idx']
            except:
                idx = i
            data[idx] = item[score_field]
    return data

def compare_full(result_file):
    data = []
    with open(result_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    matches = 0
    differences = []
    scores_1 = []
    scores_2 = []
    for row in data:
        score1 = row["hf_score"]
        score2 = row["new_score"]
        scores_1.append(score1)
        scores_2.append(score2)
        if score1 == score2:
            matches += 1
        differences.append(abs(score1 - score2))
    cf_matrix = confusion_matrix(scores_1, scores_2)
    result = {
        'average_match_rate': matches / len(data),
        'average_difference': mean(differences) if differences else 0,
        'confusion_matrix': cf_matrix
    }
    return result

def compare_annots(result_files):
    data1 = load_jsonl(result_files[0])
    data2 = load_jsonl(result_files[1])
    common_indices = set(data1.keys()) & set(data2.keys())
    matches = 0
    scores_1 = []
    scores_2 = []
    for idx in common_indices:
        score1 = data1[idx]
        score2 = data2[idx]
        scores_1.append(score1)
        scores_2.append(score2)
        if score1 == score2:
            matches += 1
    cf_matrix = confusion_matrix(scores_1, scores_2)
    result = {
        'average_match_rate': matches / len(common_indices) if common_indices else 0,
        'total_common_indices': len(common_indices),
        'confusion_matrix': cf_matrix,
        'total_agreement_1':sum(scores_1)/20,
        'total_agreement_1.2':sum(scores_2)/20
    }
    return result

def compare_scores(files):
    if len(files) != 2:
        raise ValueError("Only two files should be given")
    data1 = load_jsonl(files[0])
    data2 = load_jsonl(files[1])
    common_indices = set(data1.keys()) & set(data2.keys())

    matches = 0
    differences = []
    scores_1 = []
    scores_2 = []
    for idx in common_indices:
        score1 = data1[idx]
        score2 = data2[idx]
        scores_1.append(score1)
        scores_2.append(score2)
        if score1 == score2:
            matches += 1
        differences.append(abs(score1 - score2))
    cf_matrix = confusion_matrix(scores_1, scores_2)
    result = {
        'average_match_rate': matches / len(common_indices) if common_indices else 0,
        'average_difference': mean(differences) if differences else 0,
        'total_common_indices': len(common_indices),
        'confusion_matrix': cf_matrix
    }
    return result
parser = argparse.ArgumentParser()
ap = argparse.ArgumentParser()
ap.add_argument('--input-files',nargs='+',help="file where to add score field")
if __name__ == "__main__":
    args = ap.parse_args()
    if "edu_replicate" in args.input_files[0]:
        results = compare_full(args.input_files[0])
    elif "manual-annotation" in args.input_files[0]:
        results = compare_annots(args.input_files)
    else:
        results = compare_scores(args.input_files)
    print("Comparison Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

# Save the figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'],fmt='d', annot=True)
    plt.xlabel('Annotation scores rep 1')
    plt.ylabel('Annotation socres rep 1.2')
    plt.title('Confusion Matrix')
    plt.savefig('../results/confusion_matrix_manual_annot.png')  
