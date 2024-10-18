import os
import json
from typing import Literal, Optional

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

def select_samples(page_dir, *, page_part: Literal['top', 'mid', 'btm'], max_id=100000):
    """Select samples from the given page directory based on the page part (`page_part`) and the maximum ID (`max_id`)."""

    print(f'Selecting samples in `{page_dir}`...')

    anno_names = []
    dismiss_parts = ['_top', '_mid', '_btm']
    dismiss_parts.remove(f'_{page_part}')
    for filename in os.listdir(os.path.join(page_dir, "anno")):
        if not filename.endswith(".json"):
            continue
        anno_name = filename[:-5]
        number = int(anno_name.split("_")[0] if "_" in anno_name else anno_name.split(".")[0])
        if number > max_id:
            continue
        if anno_name[-4:] in dismiss_parts:
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

    print(select_samples('test_webpage', page_part='top'))