import os
import re
import json
import random
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from transformers.trainer_pt_utils import LabelSmoother

from utils.utils_data import BoxUtils, TextUtils
from utils.utils_ddp import rank0_print
from .anno_reader import EDGEAnnotationReader
from . import prompts


intention_pattern = re.compile(r"({\d+: )?[\[\(](\d\.\d+, \d\.\d+, \d\.\d+, \d\.\d+)[\]\)]}?")

class LegalTasks:
    BASIC = {"grounding", "ocr"}
    ACCESSIBILITY = {"general_acb", "image_alt"}
    CAPTIONING = {"title", "description", "keywords"}
    ICON_MIXED = {"icon_grounding", "icon_referring", "icon_all_grounding"}
    SOM = {"som_general", "som_icon"}
    ICON_DESC = {"icon_desc"}
    RICO_TASKS = {"ricosca", "widget-grounding", "widget-caption", "screen2words"}
    SCREEN_AGENT = {"screen_agent"}
    ADVANCED_TASKS = {"intention", "function", "detail"}
    MONKEY_TRAINING = {"monkey_training"}
    LLAVA_INSTRUCT = {"llava_instruct"}


class GeneralDataset:
    legal_elem_tasks = {}
    
    @staticmethod
    def create_vqa_text_dataset(task, task_meta):
        if task == "basic" or task == "basic_cropped":
            return BasicDataset(task, task_meta)
        elif task == "accessibility":
            return AccessibilityDataset(task, task_meta)
        elif task == "captioning":
            return CaptioningDataset(task, task_meta)
        elif task == "icon_mixed" or task == "icon_mixed_cropped":
            return IconMixedDataset(task, task_meta)
        elif task == "som" or task == "som_cropped":
            return SoMDataset(task, task_meta)
        elif task == "icon_desc":
            return IconDescDataset(task, task_meta)
        elif task == "rico_tasks":
            return RicoTasksDataset(task, task_meta)
        elif task == "advanced_tasks":
            return AdvancedTasksDataset(task, task_meta)
        elif task == "monkey_training":
            return MonkeyTrainingDataset(task, task_meta)
        elif task == "llava_instruct":
            return LLaVAInstructDataset(task, task_meta)
        else:
            raise ValueError(f"Invalid task type: {task}!")
    
    def __init__(self, task, task_meta):
        self.task = task
        self.data_dir = task_meta["data_dir"]
        self.elem_tasks = set(task_meta["elem_tasks"])
        self.max_items = task_meta.get("max_items", 99999999)
        self.repeated_time = task_meta.get("repeated_time", 1)
        assert self.elem_tasks.issubset(self.legal_elem_tasks)
        self.read_annos()
        self.qa_data = {}

    def __len__(self):
        return sum(len(self.qa_data[subtask]) for subtask in self.qa_data.keys())
    
    @staticmethod
    def get_num_qas_from_item(item):
        return len(item["messages" if "messages" in item else "conversations"]) // 2 
    
    def get_elem_task_count(self, elem_task):
        elem_task_data = self.qa_data[elem_task]
        num_images = len(elem_task_data)
        num_total = sum(self.get_num_qas_from_item(page_qas) for page_qas in elem_task_data)
        return num_images, num_total
    
    def get_all_count(self):
        num_images = 0
        num_total = 0
        for elem_task_data in self.qa_data.values():
            num_images += len(elem_task_data)
            num_total += sum(self.get_num_qas_from_item(page_qas) for page_qas in elem_task_data)
        return num_images, num_total
        
    def read_annos(self):
        return NotImplementedError
    
    def read_done_message(self):
        return NotImplementedError
    
    def create_qa_data(self):
        return NotImplementedError
    
    def create_done_message(self):
        num_images, num_total = self.get_all_count()
        message = f"{num_images} images ({num_total} QAs), where"
        for elem_task in self.elem_tasks:
            num_images, num_total = self.get_elem_task_count(elem_task)
            message += f"\n\t{elem_task}: {num_images} ({num_total})"
        return message
    
    def add_item(self, elem_task, img_path, sys_prompt, questions, answers):
        if not os.path.exists(img_path):
            print(f"Image: {img_path} not found in task {elem_task}! Skip it.")
            return
        assert len(questions) == len(answers)
        
        if sys_prompt:
            sys_prompt += "\n"
        
        messages = []
        for question, answer in zip(questions, answers):
            if not "<img>" in question and not "<img>" in answer:
                messages.append({"content": question, "role": "user"})
                messages.append({"content": answer, "role": "assistant"})
        
        if len(messages) > 0:
            messages[0]["content"] = f"<image>{sys_prompt}" + messages[0]["content"]
            item = {"id": f"{self.task};{elem_task}", "messages": messages, "images": [img_path]}
            self.qa_data[elem_task].append(item)
    
    @staticmethod
    def format_bbox(bbox_format, bbox, random=False):
        if bbox_format == "bbox":
            bbox = BoxUtils.round(bbox, ndigits=4) if not random else BoxUtils.round(bbox, ndigits="random")
            bbox = tuple(bbox)
        elif bbox_format == "point":
            bbox = BoxUtils.get_cxcy(bbox) if not random else BoxUtils.point_sampling(bbox, ndigits="random") 
            bbox = tuple(bbox)
        else:
            raise ValueError(f"Invalid bbox_format: {bbox_format}!")
        return str(bbox)
    

class BasicDataset(GeneralDataset):
    legal_elem_tasks = LegalTasks.BASIC
    def __init__(self, task, task_meta):
        self.page_annos = []
        EDGEAnnotationReader.max_pages = task_meta.pop("max_items", 9999999)
        super().__init__(task, task_meta)

        self.num_per_page = task_meta.pop("num_per_page", 99)
    
    def read_annos(self):
        for data_dir in self.data_dir:
            self.page_annos.extend(EDGEAnnotationReader.get_dir_annos(data_dir))
        
    def read_done_message(self):
        num_pages = len(self.page_annos)
        num_elements = sum(len(page_anno["elements"]) for page_anno in self.page_annos)
        return f"Successfully read {num_elements} valid elements in {num_pages} pages."
        
    def create_qa_data(self):
        for elem_task in self.elem_tasks:
            self.qa_data[elem_task] = []
        
        for page_anno in self.page_annos:
            img_path, elements = page_anno["img_path"], page_anno["elements"]
            elements = list(filter(
                lambda element: len(element["text"]) > 0 \
                    and BoxUtils.is_valid(element["bbox"], size=page_anno["viewport"]) \
                    and len(element["types"] & {"Image", "Icon"}) == 0,
                elements
            ))
            
            elem_task = random.sample(self.elem_tasks, 1)[0]
            bbox_format = random.choice(prompts.bbox_formats)
            sys_prompt = prompts.add_bbox_suffix(random.choice(prompts.basic[elem_task]), bbox_format)
            questions = []
            answers = []
            for element in random.sample(elements, min(len(elements), self.num_per_page)):
                text, bbox = element["text"], element["bbox"]
                text = TextUtils.truncate_both(text, max_words=30, max_len=300)
                bbox = BoxUtils.normalize(bbox, page_anno["viewport"])
                
                if elem_task == "grounding":
                    question = text
                    answer = self.format_bbox(bbox_format, bbox, random=False)
                elif elem_task == "ocr":
                    question = self.format_bbox(bbox_format, bbox, random=True)
                    answer = text
                questions.append(question)
                answers.append(answer)
            self.add_item(elem_task, img_path, sys_prompt, questions, answers)


class AccessibilityDataset(GeneralDataset):
    legal_elem_tasks = LegalTasks.ACCESSIBILITY
    def __init__(self, task, task_meta):
        self.page_annos = []
        super().__init__(task, task_meta)
        
        self.num_per_page = task_meta.pop("num_per_page", 99)
    def read_annos(self):
        for data_dir in self.data_dir:
            self.page_annos.extend(EDGEAnnotationReader.get_dir_annos(data_dir))
    
    def read_done_message(self):
        return f"Successfully read {len(self.page_annos)} pages."
    
    def create_qa_data(self):
        for elem_task in self.elem_tasks:
            self.qa_data[elem_task] = []

        for page_anno in self.page_annos:
            img_path, elements = page_anno["img_path"], page_anno["elements"]
            random.shuffle(elements)
            
            elem_task = random.sample(self.elem_tasks, 1)[0]
            bbox_format = random.choice(prompts.bbox_formats)
            sys_prompt = prompts.add_bbox_suffix(random.choice(prompts.accessibility[elem_task]), bbox_format)
            questions = []
            answers = []
            num_cur_page = 0
            for element in elements:
                if num_cur_page == self.num_per_page:
                    break
                
                text, bbox, types = element["text"], element["bbox"], element["types"]
                bbox = BoxUtils.normalize(bbox, page_anno["viewport"])

                if elem_task == "general_acb":
                    acb_text = element["ariaLabel"] if not element["title"] else element["title"]
                    if len(acb_text) == 0 or acb_text == text:
                        continue
                    question = self.format_bbox(bbox_format, bbox, random=True)
                    answer = acb_text
                elif elem_task == "image_alt":
                    if len(types & {"Image", "Icon"}) == 0 or len(text) == 0:
                        continue
                    question = self.format_bbox(bbox_format, bbox, random=True)
                    answer = text
                questions.append(question)
                answers.append(answer)
                num_cur_page += 1
            if len(self.qa_data[elem_task]) < self.max_items:
                self.add_item(elem_task, img_path, sys_prompt, questions, answers)


class CaptioningDataset(GeneralDataset):
    legal_elem_tasks = LegalTasks.CAPTIONING
    def __init__(self, task, task_meta):
        self.page_annos = []
        super().__init__(task, task_meta)

    def read_annos(self):
        for data_dir in self.data_dir:
            self.page_annos.extend(EDGEAnnotationReader.get_dir_annos(data_dir))
    
    def read_done_message(self):
        return f"Successfully read {len(self.page_annos)} pages."
    
    def create_qa_data(self):
        for elem_task in self.elem_tasks:
            self.qa_data[elem_task] = []

        for page_anno in self.page_annos:
            img_path = page_anno["img_path"]
            elem_task = random.sample(self.elem_tasks, 1)[0]
            
            caption = page_anno[elem_task]
            if len(caption) == 0:
                continue
            question = random.choice(prompts.captioning[elem_task])
            answer = caption
            self.add_item(elem_task, img_path, "", [question], [answer])


class IconMixedDataset(GeneralDataset):
    legal_elem_tasks = LegalTasks.ICON_MIXED
    def __init__(self, task, task_meta):
        self.img_annos = []
        super().__init__(task, task_meta)
        self.num_per_page = task_meta.pop("num_per_page", 99)

    def read_annos(self):
        with open(os.path.join(self.data_dir, "record.json")) as f:
            self.img_annos = json.load(f)
        random.shuffle(self.img_annos)
    
    def read_done_message(self):
        return f"Successfully read 1 file with {len(self.img_annos)} images."
        
    def create_qa_data(self):
        rand_etasks = list(self.elem_tasks & {"icon_grounding", "icon_referring"})
        for elem_task in self.elem_tasks:
            self.qa_data[elem_task] = []
            
        for img_anno in self.img_annos[:self.max_items]:
            key, size, icons_desc = img_anno["image"], img_anno["size"], img_anno["icons_desc"]
            img_path = os.path.join(self.data_dir, "images", key)
            bboxes = [BoxUtils.normalize(bbox, size=size) for bbox in img_anno["bboxes"]]
            assert len(bboxes) == len(icons_desc)
            num_icons = len(bboxes)

            questions = []
            answers = []
            if "icon_all_grounding" not in self.elem_tasks or num_icons == 1 or random.random() < 0.7:
                elem_task = random.choice(rand_etasks)
                bbox_format = random.choice(prompts.bbox_formats)
                sys_prompt = prompts.add_bbox_suffix(random.choice(prompts.icon_mixed[elem_task]), bbox_format)
                for i in range(min(num_icons, self.num_per_page)):
                    bbox, icon_desc = bboxes[i], icons_desc[i]
                    if elem_task == "icon_grounding":
                        question = icon_desc
                        answer = self.format_bbox(bbox_format, bbox, random=False)
                    elif elem_task == "icon_referring":
                        question = self.format_bbox(bbox_format, bbox, random=True)
                        answer = icon_desc
                    questions.append(question)
                    answers.append(answer)
                self.add_item(elem_task, img_path, sys_prompt, questions, answers)
            else:
                elem_task = "icon_all_grounding"
                bbox_format = random.choice(prompts.bbox_formats)
                question = prompts.add_bbox_suffix(random.choice(prompts.icon_mixed[elem_task]), bbox_format)
                answer = f"There are {num_icons} pasted icons in this screenshot:"
                if size[0] > size[1]:
                    sort_key = lambda x: x[0]
                else:
                    sort_key = lambda x: (x[0][1], x[0][0])
                bboxes_with_desc = sorted(zip(bboxes, icons_desc), key=sort_key)

                for i in range(num_icons):
                    bbox, desc = bboxes_with_desc[i]
                    bbox = self.format_bbox(bbox_format, bbox, random=False)
                    punct = "," if i < num_icons-1 else "."
                    answer += f" {desc} {bbox}" + punct

                self.add_item(elem_task, img_path, "", [question], [answer])


class SoMDataset(GeneralDataset):
    legal_elem_tasks = LegalTasks.SOM
    def __init__(self, task, task_meta):
        self.img_annos = {}
        super().__init__(task, task_meta)

    def read_annos(self):
        with open(os.path.join(self.data_dir, "record_general.json")) as f:
            self.img_annos["som_general"] = json.load(f)
        with open(os.path.join(self.data_dir, "record_icon_mixed.json")) as f:
            self.img_annos["som_icon"] = json.load(f)
        random.shuffle(self.img_annos["som_general"])
        random.shuffle(self.img_annos["som_icon"])
    
    def read_done_message(self):
        return f"Successfully read 2 file with {len(self.img_annos['som_general'])} general and {len(self.img_annos['som_icon'])} icon_mixed images."
    
    def create_qa_data(self):
        for elem_task in self.elem_tasks:
            self.qa_data[elem_task] = []
            for img_anno in self.img_annos[elem_task][:self.max_items]:
                img_name, text, bbox = img_anno["image"], img_anno["text"], img_anno["bbox"]
                img_dir = "images_" + ("general" if elem_task == "som_general" else "icon_mixed")
                img_path = os.path.join(self.data_dir, img_dir, img_name)
                assert os.path.exists(img_path), img_path
                if not BoxUtils.is_valid(bbox, size=img_anno["size"]):
                    continue
                bbox = BoxUtils.normalize(bbox, size=img_anno["size"])
                bbox_format = random.choice(prompts.bbox_formats)
                bbox = self.format_bbox(bbox_format, bbox, random=False)

                if elem_task == "som_general":
                    types = img_anno["types"]
                    assert len(text) > 0 and (types is None or (len(types) > 0 and isinstance(types[0], str)))
                    if len({"Image", "Icon"} & set(types)) > 0:
                        continue
                
                question = prompts.add_bbox_suffix(random.choice(prompts.som), bbox_format)
                answer = f"{text} {bbox}"
                self.add_item(elem_task, img_path, "", [question], [answer])


class IconDescDataset(GeneralDataset):
    legal_elem_tasks = LegalTasks.ICON_DESC
    def __init__(self, task, task_meta):
        self.annos = {}
        self.max_items = task_meta.get("max_items", 99999)
        super().__init__(task, task_meta)

    def read_annos(self):
        for data_dir in self.data_dir:
            with open(os.path.join(data_dir, "icon_desc.json")) as f:
                icon_anno = json.load(f)
            self.annos[data_dir] = icon_anno
            
    def read_done_message(self):
        return f"Successfully read {len(self.annos)} parts of icon annotations."

    def create_qa_data(self):
        elem_task = next(iter(self.elem_tasks))
        self.qa_data[elem_task] = []
        for data_dir, icon_anno in self.annos.items():
            for icon_item in icon_anno[:self.max_items]:
                img_path = os.path.join(data_dir, "pngs_bg", icon_item["image"])
                assert os.path.exists(img_path)
                question = random.choice(prompts.icon_description)
                answer = icon_item["desc"]
                self.add_item(elem_task, img_path, "", [question], [answer])


class RicoTasksDataset(GeneralDataset):
    legal_elem_tasks = LegalTasks.RICO_TASKS
    def __init__(self, task, task_meta):
        self.annos = {}
        super().__init__(task, task_meta)
        self.img_dir = task_meta.get("img_dir", "datasets/Rico/combined")
    
    def read_annos(self):
        for elem_task in self.elem_tasks:
            elem_task_dir = os.path.join(self.data_dir, "widget-caption" if elem_task == "widget-grounding" else elem_task)
            assert os.path.isdir(elem_task_dir)
            elem_task_anno = {}
            for fname in filter(lambda x: x.endswith(".json"), os.listdir(elem_task_dir)):
                key = fname[:-5]
                with open(os.path.join(elem_task_dir, fname), "r", encoding="utf-8") as f:
                    annotation = json.load(f)
                elem_task_anno[key] = annotation
            self.annos[elem_task] = elem_task_anno
    
    def read_done_message(self):
        message = ""
        for elem_task, elem_task_anno in self.annos.items():
            num_elem_task_files = len(elem_task_anno)
            message += f"\n\t{elem_task}: Successfully read {num_elem_task_files} files."
        return message
    
    def create_qa_data(self):
        for elem_task in self.elem_tasks:
            self.qa_data[elem_task] = []
            assert len(self.annos[elem_task]) == 1
            annotation = next(iter(self.annos[elem_task].values()))
            random.shuffle(annotation)
            for i, item in enumerate(annotation):
                img_path = os.path.join(self.img_dir, item["img_filename"])
                bbox_format = random.choice(prompts.bbox_formats)
                if elem_task == "screen2words":
                    caption = random.choice(item["captions"])
                    question = random.choice(prompts.rico_tasks[elem_task])
                    answer = caption
                elif elem_task == "widget-grounding":
                    instruction = item["instruction"]
                    question = prompts.add_bbox_suffix(random.choice(prompts.rico_tasks[elem_task]).format(instruction=instruction), bbox_format)
                    answer = self.format_bbox(bbox_format, item["bbox"], random=False)
                elif elem_task == "widget-caption":
                    instruction = item["instruction"]
                    bbox = self.format_bbox(bbox_format, item["bbox"], random=True)
                    question = random.choice(prompts.rico_tasks[elem_task]).format(bbox=bbox)
                    answer = instruction
                elif elem_task == "ricosca":
                    instruction = item["instruction"]
                    question = prompts.add_bbox_suffix(random.choice(prompts.rico_tasks["widget-grounding"]).format(instruction=instruction), bbox_format)
                    answer = self.format_bbox(bbox_format, item["bbox"], random=False)
                else: 
                    continue
                if len(self.qa_data[elem_task]) < self.max_items:
                    self.add_item(elem_task, img_path, "", [question], [answer])


class AdvancedTasksDataset(GeneralDataset):
    legal_elem_tasks = LegalTasks.ADVANCED_TASKS
    def __init__(self, task, task_meta):
        self.tasks_annos = {}
        self.max_items = task_meta.get("max_items", 99999)
        super().__init__(task, task_meta)
        
    def read_annos(self):
        for elem_task in self.elem_tasks:
            suffix = ".json" if elem_task == "intention" else ".txt"
            elem_task_data = {}
            for data_dir in self.data_dir:
                elem_task_dir = os.path.join(data_dir, elem_task)
                if os.path.exists(elem_task_dir):
                    for page_name in filter(lambda x: x.endswith(suffix), os.listdir(elem_task_dir)):
                        key = f"{data_dir};{page_name[:-len(suffix)]}"
                        with open(os.path.join(elem_task_dir, page_name)) as f:
                            page_anno = json.load(f) if elem_task == "intention" else f.read()
                        elem_task_data[key] = page_anno
            if elem_task_data:
                self.tasks_annos[elem_task] = elem_task_data
    
    def read_done_message(self):
        message = "Successfully read: "
        for elem_task in self.elem_tasks:
            num_data_task = len(self.tasks_annos[elem_task])
            message += f"\n\t{elem_task}: {num_data_task}"
        return message
    
    def create_qa_data(self):
        for elem_task in self.elem_tasks:
            self.qa_data[elem_task] = []
            for key, page_anno in list(self.tasks_annos[elem_task].items())[:self.max_items]:
                data_dir, page_name = key.split(";")
                img_path = f"{data_dir}/raw/{page_name}.png"
                for _ in range(self.repeated_time):
                    if elem_task == "intention":
                        questions = []
                        answers = []
                        for qa in page_anno:
                            answer = qa["System"]
                            if not intention_pattern.search(answer):
                                continue
                            answer = intention_pattern.sub(r"(\2)", answer)
                            questions.append(qa["User"])
                            answers.append(answer)
                        self.add_item(elem_task, img_path, "", questions, answers)
                    elif elem_task == "detail" or elem_task == "function":
                        question = random.choice(prompts.advanced_tasks[elem_task])
                        answer = page_anno
                        self.add_item(elem_task, img_path, "", [question], [answer])


class MonkeyTrainingDataset(GeneralDataset):
    legal_elem_tasks = LegalTasks.MONKEY_TRAINING
    def __init__(self, task, task_meta):
        self.annos = []
        super().__init__(task, task_meta)
        
        assert len(self.elem_tasks) == 1
        self.img_tag = re.compile(r"</img> ")
        self.answer_suffix = re.compile(r" ?Answer: ?")
        self.img_dir = task_meta.get("img_dir", "datasets/MonkeyData/Monkey_Train_data")
        self.desc_ratio = 0.75
        self.total_num_desc = int(self.max_items * self.desc_ratio)
        self.total_num_vqa = self.max_items - self.total_num_desc
    
    def read_annos(self):
        with open(os.path.join(self.data_dir, "train_monkey.json")) as f:
            self.annos = json.load(f)
        random.shuffle(self.annos)

    def read_done_message(self):
        return f"Successfully read {len(self.annos)} qa_pairs."

    def create_qa_data(self):
        elem_task = next(iter(self.elem_tasks))
        self.qa_data[elem_task] = []
        num_desc, num_vqa = 0, 0
        for qa_pair in self.annos:
            if num_desc + num_vqa == self.max_items:
                break
            img_path = os.path.join(self.img_dir, qa_pair['id'])
            conversation = qa_pair["conversations"]
            assert len(conversation) == 2
            question = conversation[0]["value"]
            answer = conversation[1]["value"]

            if question.endswith(" Answer: "):
                if num_vqa == self.total_num_vqa:
                    continue
                question = question[:-9]
                search_result = self.img_tag.search(question)
                question = question[search_result.end():]
                num_vqa += 1
            elif question.endswith("in English: "):
                if num_desc == self.total_num_desc:
                    continue
                question = random.choice(prompts.monkey_training)
                num_desc += 1
            else:
                raise ValueError(f"Question no formatted: {question}")
            self.add_item(elem_task, img_path, "", [question], [answer])


class LLaVAInstructDataset(GeneralDataset):
    legal_elem_tasks = LegalTasks.LLAVA_INSTRUCT
    def __init__(self, task, task_meta):
        self.img_prefix = task_meta.get("img_prefix", "datasets/MonkeyData/Monkey_Train_data/COCO2014/train2014/COCO_train2014_")
        self.annos = []
        super().__init__(task, task_meta)
        assert len(self.elem_tasks) == 1
    
    def read_annos(self):
        with open(os.path.join(self.data_dir, "llava_instruct_150k.json")) as f:
            self.annos = json.load(f)
        random.shuffle(self.annos)

    def read_done_message(self):
        return f"Successfully read {len(self.annos)} images."

    def create_qa_data(self):
        elem_task = next(iter(self.elem_tasks))
        self.qa_data[elem_task] = []
        for item in self.annos[:self.max_items]:
            img_name = item["image"]
            img_path = self.img_prefix + img_name
            conversations = item["conversations"]
            assert len(conversations) % 2 == 0
            questions = []
            answers = []
            
            for i in range(0, len(conversations)-1, 2):
                assert conversations[i]["from"] == "human" and conversations[i+1]["from"] == "gpt"
                question = conversations[i]["value"]
                answer = conversations[i+1]["value"]
                if i == 0:
                    if question.startswith("<image>\n"):
                        question = question[8:]
                    elif question.endswith("\n<image>"):
                        question = question[:-8]
                    else:
                        raise Exception
                questions.append(question)
                answers.append(answer)
            self.add_item(elem_task, img_path, "", questions, answers)


class EDGETensorDataset(Dataset):

    image_transform = transforms.Compose([
        transforms.Resize((896, 896), interpolation=TF.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    tokenized = None
    ignore_token_id = LabelSmoother.ignore_index
    
    def __init__(self, tokenizer, dataset_meta=None, items_filepath=None):
        assert dataset_meta or items_filepath, "dataset_meta and items_filepath can not be None simultaneously!"
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = tokenizer.model_max_length
        self.ignore_token_id = LabelSmoother.ignore_index
        self.pad_token_id = tokenizer.pad_token_id
        
        self.items = []
        if items_filepath:
            self.task2vqas = defaultdict(lambda: defaultdict(list))
            self.items = self.load_jsonl_items(items_filepath)
        else:
            self.task2vqas = self.create_vqa_text_dataset(dataset_meta)
            self.fill_img_path(self.task2vqas)
            self.items = sum((item for task in dataset_meta.keys() for item in self.task2vqas[task].values()), start=[])
            random.shuffle(self.items)
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = item["images"][0]
        inputs = self.preprocess([item], self.tokenizer)
        img_patches = self.read_and_split_img(img_path)
        return dict(
            input_ids=inputs["input_ids"][0],
            labels=inputs["labels"][0],
            attention_mask=inputs["attention_mask"][0],
            images=img_patches
        )
        
    @staticmethod
    def create_vqa_text_dataset(dataset_meta):
        task_data_readers = {}
        for task, task_meta in dataset_meta.items():
            task_data_reader = GeneralDataset.create_vqa_text_dataset(task, task_meta)
            task_data_readers[task] = task_data_reader
            rank0_print(f"Task **{task}**: {task_data_reader.read_done_message()}")
        
        num_images = 0
        num_total = 0
        task2vqas = {}
        for task, task_data_reader in task_data_readers.items():
            task_data_reader.create_qa_data()
            task2vqas[task] = task_data_reader.qa_data
            task_num_images, task_num_total = task_data_reader.get_all_count()
            num_images += task_num_images
            num_total += task_num_total
            rank0_print(f"Task **{task}{task_data_reader.elem_tasks}**: {task_data_reader.create_done_message()}")
        
        rank0_print(f"Successfully create {num_images} images ({num_total} QAs) in total.")
        return task2vqas
    
    @staticmethod
    def fill_img_path(task2vqas):
        for task_data in task2vqas.values():
            for elem_task_data in task_data.values():
                for item in elem_task_data:
                    first_content = item["messages"][0]["content"]
                    assert first_content.startswith("<image>")
                    item["messages"][0]["content"] = f"Picture 1: <img>{item['images'][0]}</img>\n" + first_content[7:]
    
    def dump_jsonl_items(self, items_filepath):
        assert items_filepath.endswith("jsonl")
        num_encode_error = 0
        with open(items_filepath, 'w', encoding='utf-8') as f:
            for item in tqdm(self.items):
                try:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + '\n')
                except UnicodeEncodeError:
                    num_encode_error += 1
                    continue
        rank0_print(f"Successfully dump {len(self.items)} items, with {num_encode_error} errors.")

    def load_jsonl_items(self, items_filepath):
        rank0_print("Loading data items ...", end="")
        items = []
        assert items_filepath.endswith("jsonl")
        num_total = 0
        with open(items_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    task, elem_task = item["id"].split(";")
                    self.task2vqas[task][elem_task].append(item)
                    items.append(item)
                    num_total += GeneralDataset.get_num_qas_from_item(items[-1])
        rank0_print(f"Done!\nSuccessfully read {len(items)} items ({num_total} QAs) in total.")
        return items
    
    def get_raw_qa(self, idx, task=None, elem_task=None):
        if task is None or elem_task is None:
            return self.items[idx]
        return self.task2vqas[task][elem_task][idx]

    @staticmethod
    def sliding_window(img, window_size, stride):
        _, h, w = img.shape
        window_rows = (h - window_size[0]) // stride + 1
        window_cols = (w - window_size[1]) // stride + 1
        windows = [
            img[:, i*stride:i*stride+window_size[0],  j*stride:j*stride+window_size[1]]
            for i in range(window_rows) for j in range(window_cols)
        ]
        return windows
    
    @classmethod
    def read_and_split_img(cls, img_path):
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = cls.image_transform(img)
            windows = cls.sliding_window(img, window_size=(448, 448), stride=448)
            img_448 = F.interpolate(img.unsqueeze(0), size=(448, 448), mode='bicubic').squeeze(0)
            img_patches = torch.stack(windows + [img_448])
        return img_patches
    
    @classmethod
    def preprocess(cls, items, tokenizer, max_len=2048):
        if not cls.tokenized:
            cls.tokenized ={
                "new_line": tokenizer("\n").input_ids,
                "system": tokenizer("system").input_ids,
                "system_message": tokenizer("You are a helpful assistant.").input_ids,
                '<|im_start|>user': tokenizer("<|im_start|>user").input_ids,
                "<|im_start|>assistant": tokenizer("<|im_start|>assistant").input_ids,
            }
            cls.roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
        
        im_start = tokenizer.im_start_id
        im_end = tokenizer.im_end_id
        nl_tokens = cls.tokenized["new_line"]
        _system = cls.tokenized["system"] + nl_tokens
        
        if "messages" in items[0]:
            role_tag = "role"
            content_tag = "content"
            message_tag = "messages"
        else:
            role_tag = "from"
            content_tag = "value"
            message_tag = "conversations"
        
        
        # Apply prompt templates
        input_ids, targets = [], []
        for item in items:
            assert item[message_tag][0][role_tag] == "user"

            input_id, target = [], []
            system = [im_start] + _system + cls.tokenized["system_message"] + [im_end] + nl_tokens
            input_id += system
            target += [im_start] + [cls.ignore_token_id] * (len(system)-3) + [im_end] + nl_tokens
            assert len(input_id) == len(target)
            for message in item[message_tag]:
                role = cls.roles[message[role_tag]]
                _input_id = cls.tokenized[role] + nl_tokens + \
                    tokenizer(message[content_tag]).input_ids + [im_end] + nl_tokens
                input_id += _input_id
                if role == '<|im_start|>user':
                    _target = [im_start] + [cls.ignore_token_id] * (len(_input_id)-3) + [im_end] + nl_tokens
                elif role == '<|im_start|>assistant':
                    _target = [im_start] + [cls.ignore_token_id] * len(cls.tokenized[role]) + \
                        _input_id[len(cls.tokenized[role])+1:-2] + [im_end] + nl_tokens
                else:
                    raise NotImplementedError
                target += _target
            assert len(input_id) == len(target)
            input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
            target += [cls.ignore_token_id] * (max_len - len(target))
            input_ids.append(input_id[:max_len])
            targets.append(target[:max_len])
        input_ids = torch.tensor(input_ids, dtype=torch.int)
        targets = torch.tensor(targets, dtype=torch.int)

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
        )
