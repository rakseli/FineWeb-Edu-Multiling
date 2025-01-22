import argparse
import dataclasses
import json
import random
import time
import os 

from typing import List, Optional, Tuple
from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.utils import FlexibleArgumentParser
from datasets import load_dataset,disable_caching,Dataset
from extract_scores import extract_score
from vllm import LLM, SamplingParams

disable_caching()


n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK",1))
fineweb_prompt = """Below is an extract from a web page. Evaluate whether the page has a high educational
value and could be useful in an educational setting for teaching from primary school to
grade school levels using the additive 5-point scoring system described below. Points are
accumulated based on the satisfaction of each criterion:
- Add 1 point if the extract provides some basic information relevant to educational top-
ics, even if it includes some irrelevant or non-academic content like advertisements and
promotional material.
- Add another point if the extract addresses certain elements pertinent to education but
does not align closely with educational standards. It might mix educational content with
non-educational material, offering a superficial overview of potentially useful topics, or
presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key
concepts relevant to school curricula. It is coherent though it may not be comprehensive
or could include some extraneous information. It may resemble an introductory section of
a textbook or a basic tutorial that is suitable for learning but has notable limitations like
treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes
for a level not higher than grade school, exhibiting a clear and consistent writing style. It
could be similar to a chapter from a textbook or a tutorial, offering substantial educational
content, including exercises and solutions, with minimal irrelevant information, and the
concepts aren’t too advanced for grade school students. The content is coherent, focused,
and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for
teaching either at primary school or grade school. It follows detailed reasoning, the writing
style is easy to follow and offers profound and thorough insights into the subject matter,
devoid of any non-educational or complex content.
The extract: {example}.
After examining the extract:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: "Educational score: <total points>"
"""

def prepare_data(args,tokenizer,max_len=1500):
    fineweb_prompt_tokenized = tokenizer(fineweb_prompt)
    fineweb_prompt_len = len(fineweb_prompt_tokenized['input_ids'])
    
    def truncate(example):
        
        example_len = len(example['text'])
        example['text'] = example['text'][:min(max_len, example_len)]
        example_tokens = tokenizer(example['text'])
        example['extract_len']=len(example_tokens.input_ids)
        return example
    
    def add_prompt(example):
        example['text']=fineweb_prompt.format(example=example['text'])
        example['text'] = tokenizer.apply_chat_template([{"role": "user", "content": example['text']}], tokenize=False)
        return example

    if 'Llama-3' in args.model:
        stop_tokens = ["<|end_of_text|>", "<|eot_id|>"]
    elif 'Qwen' in args.model:
        stop_tokens = ['<|endoftext|>']
    else:
        raise ValueError("Only qwen or llama3 are supported!")
    print(f"Safe mode: {args.safe_mode}")
    if args.safe_mode == 'true':
        with open(args.dataset,'r',encoding='utf-8') as f:
            data_as_json = [json.loads(line) for line in f]
        filtered_list = [{"text": json_obj["text"]} for json_obj in data_as_json if "text" in json_obj]
        dataset = Dataset.from_list(filtered_list)
    else:
        dataset = load_dataset("json", data_files=args.dataset,num_proc=n_cpus,split='train',)
        dataset = dataset.remove_columns(column_names=["f","o","id","filter","ts","pii", "s","rs","u","c","collection","lang","prob","seg_langs","robotstxt","doc_scores"])
    dataset = dataset.map(truncate,num_proc=n_cpus)
    dataset = dataset.map(add_prompt,num_proc=n_cpus)
    dataset = dataset.add_column("promt_len", [fineweb_prompt_len] * len(dataset))
    sampling_params = [SamplingParams(
                n=args.n,
                temperature=0.6,
                repetition_penalty=args.rep_penalty,
                top_p=0.95,
                top_k=50,
                max_tokens=1500,
                stop=stop_tokens
            ) for i in range(len(dataset))]
    assert isinstance(sampling_params[0],SamplingParams) == True
    if args.test:
        dataset = dataset.select(list(range(args.num_prompts)))
        sampling_params = sampling_params[:args.num_prompts]
    print(f"Example data point: {dataset[0]}",flush=True)
    print(f"Dataset len: {len(dataset)}",flush=True)
    return dataset,sampling_params

def prepare_original(args,tokenizer):
    dataset = load_dataset("json", data_files=args.dataset,num_proc=n_cpus,split='train')
    def count_tokens(example):
        total_tokens = tokenizer(example['prompt'])
        example['n_tokens']=len(total_tokens.input_ids)
        return example
    
    dataset = dataset.map(count_tokens,num_proc=n_cpus)
    sampling_params = [SamplingParams(
                n=1,
                temperature=0.6,
                repetition_penalty=args.rep_penalty,
                top_p=0.95,
                top_k=50,
                max_tokens=1500,
                stop=["<|end_of_text|>", "<|eot_id|>"]
            ) for i in range(len(dataset))]
    assert isinstance(sampling_params[0],SamplingParams) == True
    print(f"Example data point: {dataset[0]}",flush=True)
    print(f"Dataset len: {len(dataset)}",flush=True)
    return dataset,sampling_params

def run_vllm(dataset,sampling_params,args,engine_args: EngineArgs,use_beam_search=False):
    llm = LLM(**dataclasses.asdict(engine_args))
    # Add the requests to the engine.
    if "fineweb" in args.dataset:
        prompt_col = "prompt"
    else:
        prompt_col = "text"
    start = time.perf_counter()        
    output = llm.generate(dataset[prompt_col],sampling_params, use_tqdm=True)
    end = time.perf_counter()
    elapsed_time = end - start
    return elapsed_time, output


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Sample the requests.
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=args.trust_remote_code)
    if "fineweb" in args.dataset:
        dataset, sampling_params = prepare_original(args,tokenizer)
        total_num_tokens = sum(dataset["n_tokens"])
    else:
        dataset,sampling_params = prepare_data(args,tokenizer)
        total_num_tokens = sum(dataset["promt_len"]) + sum(dataset['extract_len'])
        
    elapsed_time,outputs = run_vllm(dataset,sampling_params,args,
                                EngineArgs.from_cli_args(args))    
    print(f"Throughput: {len(dataset) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} total tokens/s, ")

    # Output JSON results if specified
    if args.output_json:
        if "fineweb" in args.dataset:
            results = {
                "elapsed_time": elapsed_time,
                "num_requests": len(dataset),
                "total_num_tokens": total_num_tokens,
                "requests_per_second": len(dataset) / elapsed_time,
                "tokens_per_second": total_num_tokens / elapsed_time,
            }
        else:
            results = {
                "elapsed_time": elapsed_time,
                "num_requests": len(dataset),
                "total_num_tokens": total_num_tokens,
                "extract_tokens":sum(dataset['extract_len']),
                "avg_extract_tokes_per_page":sum(dataset['extract_len'])/len(dataset),
                "requests_per_second": len(dataset) / elapsed_time,
                "tokens_per_second": total_num_tokens / elapsed_time,
            }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
    if args.output_file is not None:
        with open(args.output_file,"w") as f:
            for i,output in enumerate(outputs):
                prompt = output.prompt
                generated_text = output.outputs[0].text
                score = extract_score(generated_text)
                l = {"idx":i,'prompt':prompt,'generated_text':generated_text,"score":score}
                json_line = json.dumps(l,ensure_ascii=False)
                f.write(json_line + '\n')
            

if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--dataset",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument("--safe-mode",
                        type=str,
                        help="use safe mode to load dataset")
    parser.add_argument("--rep-penalty",
                        type=float,
                        default=1.0,
                        help="repetition penalty")
    parser.add_argument("--test",action="store_true")
    parser.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the throughput results in JSON format.')
    parser.add_argument("--async-engine",
                        action='store_true',
                        default=False,
                        help="Use vLLM async engine rather than LLM class.")
    parser.add_argument("--disable-frontend-multiprocessing",
                        action='store_true',
                        default=False,
                        help="Disable decoupled async engine frontend.")
    parser.add_argument("--output-file",
                        help="output path prefix")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    main(args)