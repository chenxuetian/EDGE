import numpy as np
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from copy import copy
import re
import os
import random

import torch
import torch.nn.functional as F
from transformers.generation import GenerationConfig

from monkey_model.modeling_monkey import MonkeyLMHeadModel
from monkey_model.tokenization_qwen import QWenTokenizer
from monkey_model.configuration_monkey import MonkeyConfig
from utils.utils_data import BoxUtils
from edge_dataset.dataset import EDGETensorDataset


POINT_PATTERN = re.compile(r"\(\d\.\d+, \d\.\d+\)")
BBOX_PATTERN = re.compile(r"\(\d\.\d+, \d\.\d+, \d\.\d+, \d\.\d+\)")
GRD_OUTPUT_BBOX_PATTERN = re.compile(r"\[0\.(\d{1,4}), 0\.(\d{1,4}), 0\.(\d{1,4}), 0\.(\d{1,4})\]")
BPOINTS_PATTERN = re.compile(r"\((\d+),(\d+)\),\((\d+),(\d+)\)")


def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_tokenizer(path):
    tokenizer = QWenTokenizer.from_pretrained(
        path,
        model_max_length=2048,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    return tokenizer


def load_model(path):
    config = MonkeyConfig.from_pretrained(path)
    print("Start loading model...")
    model = MonkeyLMHeadModel.from_pretrained(path, config=config, device_map=0)
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(path, trust_remote_code=True)
    return model


def plot_anything_to_image(img_pil, targets):
    W, H = img_pil.size
    img_pil = copy(img_pil)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.load_default()
    radius = 10
    width = 5
    for target in targets:
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        if "point" in target:
            point = target["point"]
            x, y = round(point[0] * W), round(point[1] * H)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        elif "bbox" in target:
            bbox = list(target["bbox"])
            x1, y1, x2, y2 = BoxUtils.round_to_int(BoxUtils.denormalize(bbox, (W, H)))
            draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
            if "label" in target:
                label = target["label"]
                text_bbox = draw.textbbox((x1, y1), str(label), font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1, y1), str(label), fill="white")
        else:
            raise ValueError(f"Unexcepted! target={target}")
    return img_pil


def process_plotting(question, answer=None, response=None):
    targets = []

    if result := POINT_PATTERN.findall(question):
        for point in result:
            targets.append({"point": eval(point), "label": "Referred"})
            
    if answer and (result := POINT_PATTERN.findall(answer)):
        for point in result:
            targets.append({"point": eval(point), "label": "GroundTruth"})
    
    if response and (result := POINT_PATTERN.findall(response)):
        for point in result:
            targets.append({"point": eval(point), "label": "Prediction"})
    
    if result := BPOINTS_PATTERN.findall(question):
        for bbox in result:
            bbox = BoxUtils.normalize(list(map(int, bbox)), size=[1000, 1000])
            targets.append({"bbox": bbox, "label": "Referred"})
            
    if answer and (result := BPOINTS_PATTERN.findall(answer)):
        for bbox in result:
            bbox = BoxUtils.normalize(list(map(int, bbox)), size=[1000, 1000])
            targets.append({"bbox": bbox, "label": "GroundTruth"})
    
    if response and (result := BPOINTS_PATTERN.findall(response)):
        for bbox in result:
            bbox = BoxUtils.normalize(list(map(int, bbox)), size=[1000, 1000])
            targets.append({"bbox": bbox, "label": "Prediction"})
            
    if result := BBOX_PATTERN.findall(question):
        for bbox in result:
            bbox = eval(bbox)
            targets.append({"bbox": bbox, "label": "Referred"})
            
    if answer and (result := BBOX_PATTERN.findall(answer)):
        for bbox in result:
            bbox = eval(bbox)
            targets.append({"bbox": bbox, "label": "GroundTruth"})
    
    if response and (result := BBOX_PATTERN.findall(response)):
        for bbox in result:
            bbox = eval(bbox)
            targets.append({"bbox": bbox, "label": "Prediction"})
    return targets
    

def display_samples(dataset, index, task=None, elem_task=None):
    item = dataset.get_raw_qa(index, task, elem_task)
    img_path = item["images"][0]
    print(f"Image: {img_path}")
    print(f"Task: {task}:{elem_task}")

    img = Image.open(img_path)
    for i in range(0, len(item["messages"]), 2):
        question = item["messages"][i]["content"]
        answer = item["messages"][i+1]["content"]
        targets = process_plotting(question, answer)
        img_plotted = plot_anything_to_image(img, targets)
        display(img_plotted)
        print(f"******************************\nQuestion: \n{question}\n")
        print(f"******************************\nGround truth: {answer}\n")
    

def generate_freeform(model, tokenizer, img_path, query, need_display=True):
    img = Image.open(img_path)
    images = EDGETensorDataset.read_and_split_img(img_path).unsqueeze(0)
    query = tokenizer.from_list_format([
        {'image': img_path},
        {'text': query},
    ])
    response, history = model.chat(tokenizer, query=query, history=None, images=images)
    
    if need_display:
        img = Image.open(img_path)
        targets = process_plotting(query, None, response)
        img = plot_anything_to_image(img, targets)
        display(img)
        print(f"Query: {query}\n")
        print(f"Response: \n{response}\n")

    return response


def predict_sample_from_dataset(model, tokenizer, dataset, index, task=None, elem_task=None):
    
    item = dataset.get_raw_qa(index, task, elem_task)
    img_path = item["images"][0]
    
    print(f"Image: {img_path}")
    print(f"Task: {task}:{elem_task}")

    img = Image.open(img_path)
    images = EDGETensorDataset.read_and_split_img(img_path).unsqueeze(0)
    history = None
    for i in range(0, len(item["messages"]), 2):
        question = item["messages"][i]["content"]
        answer = item["messages"][i+1]["content"]
        
        query = tokenizer.from_list_format([
            {'text': question},
        ])
        response, history = model.chat(tokenizer, query=query, history=history, images=images)
        
        targets = process_plotting(question, answer, response)
        img_plotted = plot_anything_to_image(img, targets)
        display(img_plotted)
        print(f"******************************\nQuery: \n{question}\n")
        print(f"******************************\nResponse: \n{response}\n")
        print(f"******************************\nGround truth: \n{answer}\n")
        