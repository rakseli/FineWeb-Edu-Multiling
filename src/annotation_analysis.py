import os
import re
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import euclidean
from itertools import combinations

parser = argparse.ArgumentParser()
ap = argparse.ArgumentParser()
ap.add_argument('--input-root',help="file root",default="../results/hplt-annotations")
ap.add_argument('--output-root',help="output root",default="../results/hplt-plots")

def read_scores_from_shards(args):
    """Reads scores from all shards in a given directory."""
    scores_data = defaultdict(list)
    shard_pattern = re.compile(r"vllm-annotations-Llama-3.3-70B-Instruct-(?P<lang>\w+)_(?P<script>\w+)\.split-(?:shard-\d+)?-64-rep-penalty-1.0\.jsonl$")
    
    for file in os.listdir(args.input_root):
        match = shard_pattern.match(file)
        if match:
            lang_script = f"{match['lang']}_{match['script']}"
            with open(os.path.join(args.input_root, file), 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    scores_data[lang_script].append(data.get("score", 0))  # Default to 0 if 'score' is missing
    return scores_data

def compute_mean_scores(scores_data):
    """Computes mean scores per lang_script."""
    mean_scores = {key: np.mean(scores) for key, scores in scores_data.items()}
    return mean_scores

def compute_median_scores(scores_data):
    """Computes mean scores per lang_script."""
    mean_scores = {key: np.median(scores) for key, scores in scores_data.items()}
    return mean_scores

def analyze_scores(score_dict, language_param,small_lang_mode=False):
    freq_dict = {}
        
    for lang, scores in score_dict.items():
        if len(scores)<500000:
            if small_lang_mode:
                freq_dict[lang] = [scores.count(i) for i in range(6)]
            else:
                print(f"lang {lang} has less than 500K rows, skipping from comparison...")
                continue
        else:
            freq_dict[lang] = [scores.count(i) for i in range(6)]

    norm_freq_dict = {}
    for lang, freqs in freq_dict.items():
        total = sum(freqs)
        norm_freq_dict[lang] = [f / total for f in freqs] if total > 0 else [0] * 6


    if language_param not in norm_freq_dict:
        raise ValueError(f"Language '{language_param}' not found in the dataset.")

    param_distances = {
        lang: euclidean(norm_freq_dict[language_param], freqs)
        for lang, freqs in norm_freq_dict.items()
        if lang != language_param
    }

    most_similar = min(param_distances, key=param_distances.get)
    most_dissimilar = max(param_distances, key=param_distances.get)

    return {
        "frequency_counts": freq_dict,
        "normalized_frequencies": norm_freq_dict,
        "most_similar": most_similar,
        "most_dissimilar": most_dissimilar,
    }
    

def plot_scores(descreptive_scores,score_type,args):
    """Plots mean scores as a bar plot."""
    os.makedirs(args.output_root, exist_ok=True)
    output_path = os.path.join(args.output_root, f"hplt_{score_type}_scores.png")
    lang_scripts = list(descreptive_scores.keys())
    scores = list(descreptive_scores.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(lang_scripts, scores, color="skyblue")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Language_Script")
    plt.ylabel(f"{score_type} Score")
    plt.title(f"{score_type} per Language and Script")
    plt.xticks(fontsize=5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_combined_score_distribution_plot(scores_data, language_families,plot_type, args):
    """Saves a combined grid plot of score distributions for all language families."""
    os.makedirs(args.output_root, exist_ok=True)
    output_path = os.path.join(args.output_root, f"combined_{plot_type}_distributions.png")
    
    num_families = len(language_families)
    cols = 3  # Number of columns in the grid
    rows = (num_families + cols - 1) // cols  # Calculate rows to fit all families

    plt.figure(figsize=(15, 5 * rows))
    
    for idx, (family, languages) in enumerate(language_families.items(), start=1):
        plt.subplot(rows, cols, idx)
        for lang_script in languages:
            if lang_script in scores_data:
                scores = scores_data[lang_script]
                plt.hist(scores, bins=np.arange(0, 6, 0.5), alpha=0.5, label=lang_script, density=True)
        
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.title(f"{family} Languages")
        plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    
if __name__ == '__main__':
    args = ap.parse_args()
    scores_data = read_scores_from_shards(args)
    mean_scores = compute_mean_scores(scores_data)
    median_scores = compute_median_scores(scores_data)
    most_similar_and_dissimilar_to_eng = analyze_scores(scores_data,"eng_Latn")
    plot_scores(mean_scores,'mean',args)
    plot_scores(median_scores,'median',args)


    language_families = {
    "Indo-European": ["eng_Latn", "fra_Latn"],
    "Sino-Tibetan": ["zho_Hant", "mya_Mymr"],
    "Afro-Asiatic": ["urd_Arab", "ara_Arab"],
    }
    
    en_comparison = {
    "Most similar": ["eng_Latn", most_similar_and_dissimilar_to_eng["most_similar"]],
    "Most dissimilar": ["eng_Latn", most_similar_and_dissimilar_to_eng["most_dissimilar"]],
    }
    
    save_combined_score_distribution_plot(scores_data, language_families,"language_family",args)
    save_combined_score_distribution_plot(scores_data, en_comparison,"most_similar",args)