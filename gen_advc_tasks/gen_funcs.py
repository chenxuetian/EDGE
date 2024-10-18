import os
import json
from typing import Literal
from utils.anno_proc import raw_anno_2_text
from utils.call_api import call_vision_api
from utils.select_anno import select_samples
from utils.multi_threads import multi_threads_gen

with open("prompt/system_prompt.txt") as f:
    system_prompt = f.read()
    
with open("prompt/share_prompt.txt") as f:
    share_prompt = f.read()

with open('prompt/intention_prompt.txt', 'r') as f:
    intention_prompt = f.read()
    
with open('prompt/function_prompt.txt', 'r') as f:
    function_prompt = f.read()
    
with open('prompt/detail_prompt.txt', 'r') as f:
    detail_prompt = f.read()


def gen_intention(model, page_dir, sample_name):
    if os.path.exists(os.path.join(page_dir, "intention", f"{sample_name}.json")) or \
        os.path.exists(os.path.join(page_dir, "intention", f"{sample_name}.txt")):
        return {"prompt": 0, "completion": 0}
    
    anno_path = os.path.join(page_dir, "anno", f"{sample_name}.json")
    img_path = os.path.join(page_dir, "som", f"{sample_name}.png")
    with open(anno_path, 'r', encoding='utf-8') as f:
        img_anno = json.load(f)
    
    sample_annotation = raw_anno_2_text(
        img_anno, normalize=True,
        show_index=True,
        show_title_and_description=True
    )
    user_prompt = f"{share_prompt}\n{intention_prompt}\nPrompt ends.\n\nSample annotation:\n{sample_annotation}Annotation ends.\n\nGenerated QA pairs:\n"
    # print(user_prompt)
    # response_content, token_usage = call_vision_api(model, system_prompt, user_prompt, img_path)
    response_content = call_vision_api(model, system_prompt, user_prompt, img_path)
    
    try:
        result = json.loads(response_content)
        with open(os.path.join(page_dir, "intention", f"{sample_name}.json"), "w", encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(e)
        with open(os.path.join(page_dir, "intention", f"{sample_name}.txt"), "w", encoding='utf-8') as f:
            f.write(response_content)
    # return token_usage

def gen_intention_all(model, page_dir, *, max_id=9999, page_part: Literal['top', 'mid', 'btm']):
    sample_names = select_samples(page_dir, page_part=page_part, max_id=max_id)

    os.makedirs(os.path.join(page_dir, "intention"), exist_ok=True)
    token_usages = multi_threads_gen(
        model, page_dir, target_func=gen_intention, sample_names=sample_names, 
    )
    # print_total_cost(token_usages)


def gen_detail(model, page_dir, sample_name):
    if os.path.exists(os.path.join(page_dir, "detail", f"{sample_name}.txt")):
        return {"prompt": 0, "completion": 0}
    
    anno_path = os.path.join(page_dir, "anno", f"{sample_name}.json")
    img_path = os.path.join(page_dir, "raw", f"{sample_name}.png")
    with open(anno_path, 'r', encoding='utf-8') as f:
        img_anno = json.load(f)
    
    sample_annotation = raw_anno_2_text(
        img_anno, normalize=True,
        show_index=False,
        show_title_and_description=True
    )
    user_prompt = f"{share_prompt}\n{detail_prompt}\nPrompt ends.\n\nSample annotation:\n{sample_annotation}Annotation ends.\n\nGenerated detailed description:\n"
    # print(user_prompt)
    # response_content, token_usage = call_vision_api(model, system_prompt, user_prompt, img_path)
    response_content = call_vision_api(model, system_prompt, user_prompt, img_path)
    
    with open(os.path.join(page_dir, "detail", f"{sample_name}.txt"), "w", encoding='utf-8') as f:
        f.write(response_content)
    # return token_usage

def gen_detail_all(model, page_dir, *, max_id=100000, page_part: Literal['top', 'mid', 'btm']):
    sample_names = select_samples(page_dir, page_part=page_part, max_id=max_id)

    os.makedirs(os.path.join(page_dir, 'detail'), exist_ok=True)
    token_usages = multi_threads_gen(
        model, page_dir, target_func=gen_detail, sample_names=sample_names, 
    )
    # print_total_cost(token_usages)


def gen_function(model, page_dir, sample_name):
    if os.path.exists(os.path.join(page_dir, "function", f"{sample_name}.txt")):
        return {"prompt": 0, "completion": 0}
    
    anno_path = os.path.join(page_dir, "anno", f"{sample_name}.json")
    img_path = os.path.join(page_dir, "raw", f"{sample_name}.png")
    with open(anno_path, 'r', encoding='utf-8') as f:
        img_anno = json.load(f)
    
    sample_annotation = raw_anno_2_text(
        img_anno, normalize=True,
        show_index=False,
        show_title_and_description=True
    )
    user_prompt = f"{share_prompt}\n{function_prompt}\nPrompt ends.\n\nSample annotation:\n{sample_annotation}Annotation ends.\n\nGenerated function inference:\n"
    # print(user_prompt)
    # response_content, token_usage = call_vision_api(model, system_prompt, user_prompt, img_path)
    response_content = call_vision_api(model, system_prompt, user_prompt, img_path)
    
    with open(os.path.join(page_dir, "function", f"{sample_name}.txt"), "w", encoding='utf-8') as f:
        f.write(response_content)
    # return token_usage

def gen_function_all(model, page_dir, *, max_id=1000000, page_part: Literal['top', 'mid', 'btm']):
    sample_names = select_samples(page_dir, page_part=page_part, max_id=max_id)
    
    os.makedirs(os.path.join(page_dir, "function"), exist_ok=True)
    token_usages = multi_threads_gen(
        model, page_dir, target_func=gen_function, sample_names=sample_names, 
    )
    # print_total_cost(token_usages)


if __name__ == '__main__':
    if 'test_webpage' not in os.listdir():
        exit('To test this script, prepare a directory named `test_webpage` with annotated pages in it.')
        
    gen_detail_all(
        'claude-3-5-sonnet-20240620', 'test_webpage', page_part='top'
    )