import argparse
import json
import os
import re
import random
import time
from pprint import pprint
from tqdm import tqdm

import jsonlines
import parmap
from datasets import Dataset

from utils import openai_generation, get_prompt, get_all_keys_and_values

PROMPT_PTH = {
    "gpt": "prompt/rm_gpt.txt",
}

def match_dset(question, target_list):
    for sample in target_list:
        if sample["question"] == question:
            return sample 
    return None

def parse_index(string):
    temp_split_idx = 3

    prelim = string.split()[0]
    prelim_dot, prelim_digit = 0, 0
    safe = True
    for alphabet in prelim:
        if alphabet.isdigit():
            prelim_digit += 1
        elif alphabet == ".":
            prelim_dot += 1
        else:
            safe = False
    if safe and (prelim_digit != 0):
        if (prelim_dot == prelim_digit) or (prelim_dot + 1 == prelim_digit):
            index = string.split()[0].replace(".","").replace(" ","")
            parsed = " ".join(string.split()[1:])
            return index, parsed

    p = re.compile("\.\d\s|\d\.|\d\s") # \.\s|
    temp_str_front = " ".join(string.split()[:temp_split_idx])
    temp_str_back = " ".join(string.split()[temp_split_idx:])
    filtered = p.findall(temp_str_front)
    print(filtered)
    parsed = " ".join([temp_str_front.split(filtered[-1])[-1].strip(), temp_str_back])
    for _ in range(3):
        parsed = parsed[1:].strip() if parsed[0].isdigit() else parsed.strip()
    
    index = string.split(parsed)[0].replace(" ","").replace(".","")
    return index, parsed

def evaluate(sample, prompt, save_pth):
    result = {
        "question": sample["question"], "instruction": sample["instruction"], "eval": []
    }
    
    query = " ".join([sample["question"], sample["instruction"]])
    for cand in sample["candidates"]:
        subt = cand["subtree"]
        if type(subt) == list:
            subt_nodes = subt # random baseline
        elif type(subt) != dict:
            subt = eval(subt)
            subt_nodes = get_all_keys_and_values(subt)
        else:
            subt_nodes = get_all_keys_and_values(subt)

        context = ""
        for nd in subt_nodes:
            _, sq = parse_index(nd)
            context += f"- {sq}\n"
        
        eval_input = prompt.substitute(query=query, subqueries=context)
        max_retry = 5
        for _ in range(max_retry):
            try:
                output = openai_generation(eval_input, "gpt-4-0125-preview", 1.0)
                output = output.replace("json", "").replace("```","").strip()
                break
            except:
                pass
        
        eval_result = dict(eval(output))
        result["eval"].append({
            "subtree": subt,
            "score": eval_result["score"],
            "rationale": eval_result["rationale"],
        })
    
    with open(save_pth, "a+", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False) 
        f.write("\n") 
    time.sleep(0.3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_pth", type=str, default="dataset/train.jsonl")
    parser.add_argument("--save_pth", type=str, default="dataset/train_rm_gpt.jsonl")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--generate_score", action="store_true", default=False)
    parser.add_argument("--generate_dpo", action="store_true", default=False)
    cfg = parser.parse_args()
    
    print(f"### NUMBER OF PROCESSES: {cfg.num_proc}")

    rm_prompt = get_prompt(PROMPT_PTH, "gpt")

    # Load original dataset
    with jsonlines.open(cfg.dset_pth) as f:
        dataset = [line for line in f.iter()]
    print(f"# of full dataset: {len(dataset)}")

    # Generate eval set
    if not os.path.exists(cfg.save_pth):
        with open(cfg.save_pth, "a+", encoding="utf-8") as f:
            pass
    with jsonlines.open(cfg.save_pth) as f:
        cache = [line for line in f.iter()]
    cache = {item["question"]: None for item in cache}

    if cfg.generate_score:
        dataset = [item for item in dataset if item["question"] not in cache]
        print(f"# of dataset already generated: {len(cache)}")
        print(f"# of dataset to be generated: {len(dataset)}")

        if cfg.num_proc == 1:
            for sample in tqdm(dataset, desc="Sample"):
                evaluate(sample, rm_prompt, cfg.save_pth)

        elif cfg.num_proc > 1:
            parmap.map(evaluate, dataset, rm_prompt, cfg.save_pth)
    
    if cfg.generate_dpo:
        # Generate SFT / DPO dataset
        with jsonlines.open(cfg.save_pth) as f:
            dataset_eval = [line for line in f.iter()]

        skip = 0
        exception = 0
        dpo_inputs = []

        global_inst = "Create a three-level deep query graph that expands on the knowledge related to the provided question. Next, identify and extract a subsection of this graph that most effectively answers the question, ensuring this subsection retains a tree-like structure and includes four distinct nodes."
        for sample in tqdm(dataset_eval, desc="Ranking"):
            usr_query = " ".join([sample["question"], sample["instruction"]])

            sc_max, sc_min = 0, 100
            max_indice, min_indice = [], []
            for idx, cand in enumerate(sample["eval"]):
                sc = cand["score"]
                if sc > sc_max:
                    sc_max = sc
                    max_indice = [idx]
                elif sc == sc_max:
                    max_indice.append(idx)

                if sc < sc_min:
                    sc_min = sc
                    min_indice = [idx]
                elif sc == sc_min:
                    min_indice.append(idx)
            
            # DPO 
            if sc_max == sc_min:
                skip += 1
                continue
            max_idx = random.choice(max_indice)
            min_idx = random.choice(min_indice)

            raw = match_dset(sample["question"], dataset)
            if raw is None:
                continue
            if sample["eval"][max_idx]["subtree"] != raw["candidates"][max_idx]["subtree"]:
                exception += 1
                continue

            chosen = [
                {"role": "user", "content": f"{global_inst}\nQuestion: {usr_query}"}, 
                {"role": "assistant", "content": f"<TREE>{raw['tree']}</TREE> <SUBTREE>{sample['eval'][max_idx]['subtree']}</SUBTREE>"}
            ]
            rejected = [
                {"role": "user", "content": f"{global_inst}\nQuestion: {usr_query}"}, 
                {"role": "assistant", "content": f"<TREE>{raw['tree']}</TREE> <SUBTREE>{sample['eval'][min_idx]['subtree']}</SUBTREE>"}
            ]
            dpo_input = {"chosen": chosen, "rejected": rejected}
            dpo_inputs.append(dpo_input)
        
        # Save
        dset_dpo = Dataset.from_list(dpo_inputs) # 8568
        dset_dpo.to_parquet("dataset/train_dpo_gpt.parquet")

        print(f"Skipped samples: {skip}") # 2010
        print(f"Exceptions: {exception}") # 2 
        
