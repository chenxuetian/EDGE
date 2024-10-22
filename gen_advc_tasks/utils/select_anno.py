import os
import json
from typing import Literal, Set


def box_is_valid(bbox, size=[1, 1]):
    assert isinstance(bbox, list)
    W, H = size
    return (0 <= bbox[0] <= bbox[2] <= W) and (0 <= bbox[1] <= bbox[3] <= H)

def is_valid(page_anno: dict):
    if not isinstance(page_anno["description"], str):
        return False
    # if not (5 <= len(page_anno["elements"]) <= 50):
    #     return False
    for element in page_anno["elements"]:
        if not (element and isinstance(element, dict) and {"bbox", "text", "types"}.issubset(element)):
            return False
        if not (isinstance(element["bbox"], list) and all(isinstance(coord, int) for coord in element["bbox"])):
            return False
        if not box_is_valid(element["bbox"], page_anno["viewport"]):
            return False
        if not isinstance(element["types"], list):
            return False
        if len(element["types"]) == 0 or not isinstance(element["types"][0], str):
            return False
    return True


def select_samples(
    page_dir,
    *, page_part: Set[Literal['top', 'mid', 'btm']], max_id=1000000,
    cover_exist=False, task_name: Literal['function', 'detail', 'intention']
):
    """Select samples from the given page directory based on the given page parts (`page_part`), the maximum ID (`max_id`),
    and whether to cover the existing corresponding result file(s) (`cover_exist`).
    """

    print(f'Selecting samples in `{page_dir}` for task "{task_name}"... (page_part={page_part}, cover_exist={cover_exist}, max_id={max_id})')

    anno_names = []
    dismiss_parts = {'top', 'mid', 'btm'} - page_part
    
    for filename in os.listdir(os.path.join(page_dir, "anno")):
        if not filename.endswith(".json"):
            continue
        anno_name = filename[:-5]
        if not cover_exist:
            if os.path.exists(os.path.join(page_dir, task_name, f"{anno_name}.json")) or os.path.exists(os.path.join(page_dir, task_name, f"{anno_name}.txt")):
                continue

        number = int(anno_name.split("_")[0] if "_" in anno_name else anno_name.split(".")[0])
        if number > max_id:
            continue
        if len(anno_name) >= 4 and anno_name[-4] == '_' and anno_name[-3:] in dismiss_parts:
            continue
        if 'top' in dismiss_parts and '_' not in anno_name:
            continue
        if not os.path.exists(os.path.join(page_dir, "som", f"{anno_name}.png")):
            continue
        try:
            with open(os.path.join(page_dir, "anno", filename)) as f:
                page_anno = json.load(f)
            if is_valid(page_anno):
                anno_names.append(anno_name)
        except:
            continue

    print(f'Done! Selected {len(anno_names)} webpage(s):')
    if len(anno_names) < 4:
        for anno_name in anno_names:
            print('\t', anno_name)
    else:
        print('\t', anno_names[:2], '\n...\n', '\t', anno_names[-2:])
    print()

    return anno_names


if __name__ == '__main__':
    import os

    if os.getcwd().endswith('/utils'):
        os.chdir('..')
    if 'test_webpage' not in os.listdir():
        exit('To test this script, prepare a directory named "test_webpage" with annotated pages in it.')

    print(select_samples('test_webpage', page_part={'mid', 'btm'}, task_name='function', cover_exist=False))