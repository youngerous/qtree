import argparse
import json
import jsonlines
import copy
import random
import time
import parmap
from collections import Counter, defaultdict
from typing import Dict, List, Optional
from tqdm import tqdm

from serpapi import GoogleSearch
from utils import openai_generation, get_prompt, get_all_keys_and_values

PROMPT_PTH = {
    "generate_tree": "prompt/generate_tree.txt",
    "generate_inst": "prompt/generate_inst.txt",
    "generate_subtree": "prompt/generate_subtree.txt",
    "writing": "prompt/writing.txt",
}
BACKGROUND_DEPTH = [1, 2, 3] 
INTENTION = {
    "include": "The user want to know about the answer specifically related to the given subquestion. Therefore, generated instructions MUST ask to include topics of the subquestion.",
    "exclude": "The user already know about the answer to the given subquestion. Therefore, generated instructions MUST require to exclude topics of the subquestion.",
}
NUM_CANDS=3
NUM_NODES=4

################################################
# Pipeline
################################################

def data_pipeline(
    data, ir_topk, save_pth, openai_model, openai_temp, serpapi_key
):     
    question = data["question"]
    result = {"question": question}

    # Generate full tree
    retry, max_retry = 0, 2
    qc_pass_tree = False
    while retry < max_retry:
        tree = generate_tree(
            question=question, model=openai_model, temp=openai_temp
        )
        # QC_TREE
        if quality_check_tree(tree):
            qc_pass_tree = True
            break
        retry += 1
    if (tree is None) or (not qc_pass_tree):
        print("Error at tree generation.")
        return
    
    # Extract background knowledge and generate corresponding instructions
    depth = random.choice(BACKGROUND_DEPTH)
    knowledge = extract_background_knowledge(tree=tree, depth=depth)
    intention = random.choice(list(INTENTION)) 
    # print(intention, knowledge)
    instructions = generate_instructions(
        question=question, 
        subq=knowledge, 
        intention=INTENTION[intention], 
        model=openai_model,
        temp=openai_temp,
    )
    try:
        instructions = [inst.strip() for inst in instructions.split("&&")]
    except Exception as e:
        print(f"Error at instruction generation: {e}")
        return

    ## FOR TEST SET GENERATION
    # result["tree"] = tree
    # result["background"] = knowledge
    # result["intention"] = intention
    # result["instruction"] = instructions[0]
    # with open(save_pth, "a+", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False) 
    #     f.write("\n") 
    # return

    # Parse candidate subtrees
    final_inst, final_cand = None, None
    for inst in tqdm(instructions, desc="Instruction"):
        # print(inst)
        candidates = set()
        retry = 0
        max_retry = 2
        qc_pass_subtree = False

        while (len(candidates) < NUM_CANDS) and (retry < max_retry):
            subtrees = generate_subtree(
                question=question,
                tree=tree,
                instruction=inst,
                intent=intention,
                background=knowledge,
                model=openai_model,
                temp=openai_temp,
            )
            for subt in subtrees.split("&&"):
                try:
                    # print(subt)
                    graph = eval(subt.replace("```", "").replace("json", "").replace("\n","").strip())
                except Exception as e:
                    print(f"Error at subtree conversion: {e}")
                    continue

                # QC_SUBQ_1
                if not quality_check_overlap_cand(cache=candidates, cur_graph=graph):
                    # print("QC Failed: Overlap")
                    continue
                # print("QC Passed: Overlap")
                # QC_SUBQ_2
                nodenum = quality_check_node_num(tr=graph)
                # print(f"QC NodeNum: {nodenum}")
                if nodenum == NUM_NODES:
                    pass # Passed node number check
                elif nodenum > NUM_NODES:
                    try:
                        while len(get_all_keys_and_values(copy.deepcopy(graph))) != NUM_NODES:
                            graph = heuristic_revise_node_num(
                                wrong_subtree=graph, 
                                background=knowledge, 
                                intention=intention,
                            )
                            nodenum -= 1
                            # print(f"QC Revised: {nodenum+1} -> {nodenum}")
                            if nodenum < NUM_NODES:
                                raise Exception("Do not need to revise node number anymore.")
                    except Exception as e:
                        print(f"Failed: {e}") 
                        continue
                elif nodenum < NUM_NODES:
                    continue
                else:
                    raise NotImplementedError()
                
                # QC_SUBQ_3
                if not quality_check_hierarchy(tr=graph):
                    # print("QC Failed: Hierarchy")
                    continue
                # QC_SUBQ_4
                if quality_check_intention(
                    subt=graph, intention=intention, background=knowledge
                ):
                    candidates.add(str(graph)) # SUCCESS !
                    # print(f"QC Passed: {len(candidates)} / 3")
            retry += 1

            if len(candidates) >= NUM_CANDS:
                qc_pass_subtree = True
                final_inst = inst
                final_cand = [eval(cd) for cd in list(candidates)]
                break
        
        if qc_pass_subtree:
            break
    
    if final_inst is None:
        # Exceptional heuristic
        if (depth == 3) and (intention == "include"):
            final_inst = random.choice(instructions)
            final_cand = heuristic_depth3_include(
                full_tree=tree, bg_node=knowledge
            )
        else:
            print(f"Error at subtree generation.")
            return 

    # Postprocess candidates
    if len(final_cand) > NUM_CANDS:
        final_cand = random.sample(final_cand, NUM_CANDS)

    # IR & Writing
    result["candidates"] = []
    sources = None # Do not limit sources
    for subtree in tqdm(final_cand, desc="Writing"):
        queries = [question] + get_all_keys_and_values(subtree)
        try:
            cxt, ans = generate_ir_answer(
                queries=queries,
                instruction=final_inst,
                source=sources,
                ir_topk=ir_topk,
                model=openai_model,
                temp=openai_temp,
                serpapi_key=serpapi_key,
            )
        except Exception as e:
            print(f"Error at API module: {e}")
            exit(1)
        
        result["candidates"].append({
            "subtree": subtree,
            "context": cxt,
            "response": ans,
        })

    result["tree"] = tree
    result["background"] = knowledge
    result["intention"] = intention
    result["instruction"] = final_inst
    
    # Save
    with open(save_pth, "a+", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False) 
        f.write("\n") 

def generate_tree(question, model, temp) -> Optional[dict]:
    tree_prompt = get_prompt(PROMPT_PTH, "generate_tree")
    api_input = tree_prompt.substitute(question=question)
    decomp = openai_generation(api_input, model, temp)
    retry, max_retry, graph = 0, 3, None
    while retry < max_retry:
        try:
            graph = eval(decomp.replace("```", "").replace("json", "").replace("\n",""))
            break
        except:
            decomp = openai_generation(api_input, model, temp)
            retry += 1
    return graph

def extract_background_knowledge(tree: dict, depth: int) -> str:
    same_hierarchy = []
    if depth == 1:
        same_hierarchy.extend(list(tree))
    elif depth == 2:
        for depth2 in tree:
            same_hierarchy.extend(tree[depth2])
    elif depth == 3:
        for depth2 in tree:
            for depth3 in tree[depth2]:
                same_hierarchy.extend(tree[depth2][depth3])
    else:
        raise NotImplementedError()
    knowledge = random.choice(same_hierarchy)
    return knowledge

def generate_instructions(question, subq, intention, model, temp) -> str:
    context_prompt = get_prompt(PROMPT_PTH, "generate_inst")
    api_input = context_prompt.substitute(question=question, subquestion=subq, intention=intention)
    instructions = openai_generation(api_input, model, temp)
    return instructions

def generate_subtree(question, tree, instruction, intent, background, model, temp) -> str:
    subtree_parsing_prompt = get_prompt(PROMPT_PTH, "generate_subtree")
    rule = f"{intent.capitalize()} the node '{background}' in the subtree."
    api_input = subtree_parsing_prompt.substitute(question=question, tree=tree, inst=instruction, rule=rule)
    subtrees = openai_generation(api_input, model, temp)
    return subtrees

def search_evidence(
    question: str, serpapi_key: str, topk: int, source: str
) -> List[Dict[str, str]]:
    if source is not None:
        if not source: # Search from all sources
            pass
        elif "&&" not in source: # Search from multiple sources
            question = question + f" site:{source}"
        else: # Search from single source
            srcs = [f"site:{src.strip()}" for src in source.split("&&")]
            srcs = " OR ".join(srcs)
            question = question + f" {srcs}"

    params = {"engine": "duckduckgo", "q": question, "kl": "us-en", "api_key": serpapi_key}
    max_request, cnt_request = 3, 0
    while True:
        result = GoogleSearch(params).get_dict()
        try:
            return result["organic_results"][:topk]
        except Exception as e:
            # 'error': "DuckDuckGo hasn't returned any results for this query."
            print(result)
            time.sleep(1)
            cnt_request += 1
            if cnt_request > max_request:
                print("EXCEED MAX REQUEST")
                return None

def compose_context(q: str, evidence: List[Dict[str, str]], qidx = None):
    evidence = [(evi["title"], evi["snippet"]) for evi in evidence]
    context = ""
    for idx, evi in enumerate(evidence):
        context += f"TITLE: {evi[0]}\n"
        context += f"CONTENT: {evi[1]}\n"
        if qidx is not None:
            context += f"QUERY: Q{qidx}\n\n" 
    return context

def generate_ir_answer(
    queries, instruction, source, ir_topk, model, temp, serpapi_key
    ):
    context = "Search queries used to find the answer to the question: \n"
    for idx, subq in enumerate(queries):
        if idx != 0:
            subq = " ".join(subq.split()[1:]).replace('"', '')
        context += f"- [Q{idx+1}] {subq}\n"
    context += "\nEvidence:\n\n"
    
    # IR
    non_evidence_cnt = 0
    for idx, subq in enumerate(queries):
        if idx != 0:
            subq = " ".join(subq.split()[1:]).replace('"', '')
        evidence = search_evidence(
            question=subq,
            serpapi_key=serpapi_key,
            topk=2,
            source=source,
        )
        if evidence is None:
            non_evidence_cnt += 1
            continue
        context += compose_context(subq, evidence, qidx=idx+1)
    if non_evidence_cnt == len(queries):
        raise Exception("Error at IR API module")
    
    writing_prompt = get_prompt(PROMPT_PTH, "writing")
    writing_input = writing_prompt.substitute(
        question = f"{queries[0]} {instruction}", 
        context = context
    )
    answer = openai_generation(writing_input, model, temp)
    
    return context, answer

################################################
# Quality Check
################################################

def quality_check_tree(tree: dict, gold_depth: int=3) -> bool:
    # Check 3-depth with completed structure
    if len(tree) != gold_depth:
        return False
    
    for firstkey, firstval in tree.items():
        if not isinstance(firstval, dict):
            return False
        if len(firstval) != gold_depth:
            return False
        for secondkey, secondval in firstval.items():
            if not isinstance(secondval, list):
                return False
            if len(secondval) != gold_depth:
                return False    
    
    node_list = get_all_keys_and_values(tree)
    
    # Check correct numbering
    for item in node_list:
        if not item[0].isdigit():
            return False
    
    # Check overlapped content
    node_list = [" ".join(each.split()[1:]) for each in node_list]
    if len(node_list) != len(set(node_list)):
        return False
    
    return True

def quality_check_overlap_cand(cache, cur_graph): 
    # Check overlap within current graph
    if len(cur_graph) != len(set(cur_graph)):
        return False
    
    # Check overlap with cached subtree 
    cnt_cur = Counter(get_all_keys_and_values(cur_graph))
    for cc in cache:
        cnt_cache = Counter(get_all_keys_and_values(eval(cc)))
        overlapped = list((cnt_cache & cnt_cur).elements())
        if len(overlapped) > 2: # if more than 2/4 overlapped, then reject
            return False
    return True

def quality_check_node_num(tr):
    nodes = get_all_keys_and_values(tr)
    return len(nodes)

def quality_check_hierarchy(tr):
    # Check: Connectivity of nodes in the tree
    for firstkey in tr:
        firstidx = firstkey.split()[0].replace(".","")
        if isinstance(tr[firstkey], dict):
            for secondkey in tr[firstkey]:
                secondidx = secondkey.split()[0].replace(".","")
                if firstidx != secondidx[:-1]:
                    return False
                if not isinstance(tr[firstkey][secondkey], list):
                    return False
                for ex in tr[firstkey][secondkey]:
                    lastidx = ex.split()[0].replace(".","")
                    if secondidx != lastidx[:-1]:
                        return False
        elif isinstance(tr[firstkey], list):
            for ex in tr[firstkey]:
                exidx = ex.split()[0].replace(".","")
                if firstidx != exidx[:-1]:
                    return False
                
    # Check: First node in the same hierarchy
    # Check: Same parent node in depth 2
    for idx, firstkey in enumerate(tr):
        if idx == 0:
            criteria = firstkey.split()[0].replace(".","")
            continue
        firstidx = firstkey.split()[0].replace(".","")
        if len(firstidx) != criteria:
            return False
        
        if (len(firstidx) > 1) and (criteria[0] != firstidx[0]):
            return False    
    return True

def quality_check_intention(subt, intention, background):
    nodes = get_all_keys_and_values(subt)
    if intention == "include":
        return True if background in nodes else False
    elif intention == "exclude":
        return False if background in nodes else True
    else:
        raise NotImplementedError()

################################################
# Heuristics
################################################

def heuristic_depth3_include(full_tree, bg_node):
    def _search_node(index):
        nodes = get_all_keys_and_values(full_tree)
        for node in nodes:
            node_idx = node.split()[0].replace(".","")
            if index == node_idx:
                return node
        return None
        
    bg_idx = bg_node.split()[0].replace(".","")
    parent_node = _search_node(bg_idx[:2])
    
    # Candidate 1: (a.b - a.b.c / a.b.d / a.b.e)
    neighbor = ["1","2","3"]
    neighbor.remove(bg_idx[-1])
    neighbor_nodes = [_search_node(f"{bg_idx[:-1]}{nb}") for nb in neighbor]
    cand1 = {
        parent_node: sorted([bg_node] + neighbor_nodes)
    }
    
    # Candidate 2: (a - a.b - a.b.c) & (d)
    grandparent_node = _search_node(bg_idx[0])
    depth1 = ["1","2","3"]
    depth1.remove(bg_idx[0])
    depth1_other = _search_node(random.choice(depth1))
    cand2 = {
        grandparent_node: {parent_node: [bg_node]},
        depth1_other: {}
    }
    
    # Candidate 3: (a.b - a.b.c) & (a.d - a.d.e)
    depth2 = ["1","2","3"]
    depth2.remove(bg_idx[1])
    parent_neighbor_idx = random.choice(depth2)
    parent_neighbor = _search_node(f"{bg_idx[0]}{parent_neighbor_idx}")
    parent_neighbor_child = _search_node(f"{bg_idx[0]}{parent_neighbor_idx}{random.choice(['1','2','3'])}")
    cand3 = {
        parent_neighbor: [parent_neighbor_child],
        parent_node: [bg_node],
    }
    return [cand1, cand2, cand3]

def heuristic_revise_node_num(wrong_subtree, background, intention):
    # Find the deepest depth
    nodes = get_all_keys_and_values(wrong_subtree)
    max_depth = max([len(nd.split()[0].replace(".","")) for nd in nodes])
    
    # Select candidate to remove
    cand_remove = [nd for nd in nodes if len(nd.split()[0].replace(".","")) == max_depth]
    if background in cand_remove:
        if intention == "include":
            cand_remove.remove(background)
        elif intention == "exclude":
            cand_remove = [background]
        else:
            raise NotImplementedError()

    final_cand_remove = random.choice(cand_remove)
    nodes.remove(final_cand_remove) 
    
    # list -> dict (note: we need more fancy code)
    sort_by_depth = defaultdict(list)
    for nd in nodes:
        depth = len(nd.split()[0].replace(".",""))
        sort_by_depth[depth].append(nd)
    
    reconstructed = {}
    for idx, dep in enumerate(sort_by_depth):
        if idx == 0:
            for uppernode in sort_by_depth[dep]:
                reconstructed[uppernode] = {} if dep == 1 else []
        else:
            if dep == 2:
                for midnode in sort_by_depth[dep]:
                    upper_idx = midnode[0]
                    match_key = None
                    for uppernode in sort_by_depth[1]:
                        if uppernode[0] == upper_idx:
                            match_key = uppernode
                    if match_key is not None:
                        reconstructed[match_key][midnode] = []
                    else:
                        reconstructed[midnode] = []
            elif dep == 3:
                for leafnode in sort_by_depth[dep]:
                    upper_idx = leafnode[0]
                    mid_idx = leafnode[2]
                    match_key_upper, match_key_mid = None, None
                    for uppernode in sort_by_depth[1]:
                        if uppernode[0] == upper_idx:
                            match_key_upper = uppernode
                            
                    if match_key_upper is not None:
                        for midnode in sort_by_depth[2]:
                            if (midnode[0] == match_key_upper[0]) and (midnode[2] == mid_idx):
                                match_key_mid = midnode
                        assert match_key_mid is not None
                        reconstructed[match_key_upper][match_key_mid].append(leafnode)
                    else:
                        for midnode in sort_by_depth[2]:
                            if midnode[2] == mid_idx:
                                match_key_mid = midnode
                        reconstructed[match_key_mid].append(leafnode)
            else:
                NotImplementedError()
    return reconstructed    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--ir_topk", type=int, default=2)
    parser.add_argument("--openai_model", type=str, default="gpt-4-0125-preview")
    parser.add_argument("--openai_temp", type=float, default=1.0)
    parser.add_argument("--serpapi_key", type=str)
    parser.add_argument("--save_pth", type=str, default="dataset/test.jsonl")

    cfg = parser.parse_args()

    print(f"### PROCESS: {cfg.num_proc}")
    
    dpath = {
        "asqa": "dataset/test_q_asqa.jsonl", ##
        "longform": "dataset/test_q_lf.jsonl", ##
        "expertqa": "dataset/test_q_expert.jsonl", ##
    }
    dset = []
    for bench in dpath:
        with jsonlines.open(dpath[bench]) as f:
            data = [line for line in f.iter()]
        dset.extend(data)
        print(f"Dataset: {bench} / Amount: {len(data)}")

    with jsonlines.open(cfg.save_pth) as f:
        saved = [line for line in f.iter()]
    print(f"# of cached dataset: {len(saved)}")
    cache = {item["question"]: None for item in saved} if len(saved) else None

    print(f"# of total raw dataset: {len(dset)}")

    if cfg.num_proc == 1:
        for sample in tqdm(dset, desc="Sample"):
            if (cache is not None) and (sample["question"] in cache):
                continue
            data_pipeline(
                data=sample,
                ir_topk=cfg.ir_topk,
                save_pth=cfg.save_pth,
                openai_model=cfg.openai_model,
                openai_temp=cfg.openai_temp,
                serpapi_key=cfg.serpapi_key,
            )
    else:
        if cache is not None:
            remaining_set = [
                datum for datum in dset if datum["question"] not in cache
            ]
        else:
            remaining_set = dset
        print(f"# of remaining dataset: {len(remaining_set)}")
        parmap.map(
            data_pipeline,
            remaining_set,
            ir_topk=cfg.ir_topk,
            save_pth=cfg.save_pth,
            openai_model=cfg.openai_model,
            openai_temp=cfg.openai_temp,
            serpapi_key=cfg.serpapi_key,
            pm_pbar=True,
            pm_processes=cfg.num_proc,   
        )
        
# python src/generate_tree.py --num_proc 1

    