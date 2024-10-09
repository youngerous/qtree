import time
from string import Template
from openai import OpenAI

API_KEY = "YOUR_KEY"
client = OpenAI(api_key=API_KEY)

def openai_generation(api_input: str, model_name: str, temp: float) -> str:
    sleep_time = 3
    sleep_cnt = 0
    result = None
    while result is None:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": api_input}],
                temperature=temp
            )
            result = response.choices[0].message.content
        except Exception as e:
            print(e)
            print("Sleep time: ", sleep_time)
            time.sleep(sleep_time)
            sleep_cnt += 1
            if sleep_cnt > 30:
                exit()
    return result

def get_prompt(src: dict, prompt_type: str):
    with open(src[prompt_type], "r") as f:
        prompt_raw = f.readlines()
    return Template(" ".join(prompt_raw))

def get_all_keys_and_values(graph):
    def _get_nodes_unsorted(g, memory):
        for key, val in g.items():
            memory.append(key)
            if isinstance(val, dict):
                _get_nodes_unsorted(val, memory)
            elif isinstance(val, list):
                for v in val:
                    memory.append(v)
        return memory

    unsorted = _get_nodes_unsorted(graph, [])
    return sorted(unsorted, key=lambda e: (len(e.split()[0].replace(".","")), e[0], e[2], e[4]))