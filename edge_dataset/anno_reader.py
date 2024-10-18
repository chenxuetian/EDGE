import os
import json
import numpy as np
from collections import Counter
from tqdm import tqdm

from utils.utils_ddp import rank0_print
from utils.utils_data import BoxUtils, TextUtils


class EDGEAnnotationReader:
    read_done_dirs = {}
    max_pages = 9999999

    @classmethod
    def get_dir_annos(cls, data_dir):
        data_dir_name = data_dir.split("/")[-1]
        if data_dir_name not in cls.read_done_dirs:
            cls.read_anno_dir(data_dir)
        return cls.read_done_dirs[data_dir_name]

    @classmethod
    def read_anno_dir(cls, data_dir):
        # TODO: 适应新的标注格式
        page_annos = []
        num_filtered_elements = 0
        anno_dir = os.path.join(data_dir, "anno")
        page_anno_fnames = [fname for fname in os.listdir(anno_dir) if fname.endswith(".json")]
        for page_anno_fname in tqdm(page_anno_fnames):
            with open(os.path.join(anno_dir, page_anno_fname), "r", encoding="utf-8") as f:
                page_anno = json.load(f)
            num_raw_elements = len(page_anno["elements"])
            valid = cls.reformat_and_filter(page_anno)
            if valid:
                num_filtered_elements += num_raw_elements - len(page_anno["elements"])
                page_annos.append(dict(
                    img_path=os.path.join(data_dir, "raw", page_anno["image"]), 
                    viewport=page_anno["viewport"],
                    title=page_anno["title"], 
                    description=page_anno["description"], 
                    keywords=page_anno["keywords"],
                    elements=page_anno["elements"]
                ))
                if len(page_annos) == cls.max_pages:
                    break
        
        data_dir_name = data_dir.split("/")[-1]
        num_filtered_pages = len(page_anno_fnames) - len(page_annos)
        num_valid_elements = sum(len(page_anno["elements"]) for page_anno in page_annos)
        rank0_print(f"Read {data_dir_name}: {num_filtered_pages} pages filtered, {len(page_annos)} left. Additionaly {num_filtered_elements} items filtered, {num_valid_elements} left.")
        cls.read_done_dirs[data_dir_name] = page_annos

    @staticmethod
    def is_valid(page_anno):
        if not isinstance(page_anno["description"], str):
            return False
        if not (5 <= len(page_anno["elements"]) <= 50):
            return False
        for element in page_anno["elements"]:
            if not (element and isinstance(element, dict) and {"bbox", "text", "types"}.issubset(element)):
                return False
            if not (isinstance(element["bbox"], list) and all(isinstance(coord, int) for coord in element["bbox"])):
                return False
            if not BoxUtils.is_valid(element["bbox"], page_anno["viewport"]):
                return False
            if not isinstance(element["types"], list):
                return False
            if len(element["types"]) == 0 or not isinstance(element["types"][0], str):
                return False
        return True

    @staticmethod
    def reformat_and_filter(page_anno):
        if page_anno["description"] is None:
            page_anno["description"] = ""
        if page_anno["keywords"] is None:
            page_anno["keywords"] = ""
        assert isinstance(page_anno["description"], str) and isinstance(page_anno["keywords"], str)
        
        viewport = page_anno["viewport"]
        W, H = viewport
        for element in page_anno["elements"]:
            if not (element and isinstance(element, dict) and {"bbox", "text", "types"}.issubset(element)):
                return False
            bbox, types = element["bbox"], element["types"]
            assert isinstance(bbox, list) and all(isinstance(coord, int) for coord in bbox)
            # if not BoxUtils.is_valid(bbox, viewport):
            #     return False
            if not ((0 <= bbox[0] <= bbox[2] <= W) and (0 <= bbox[1] <= bbox[3] <= H)):
                return False
            
            
            assert isinstance(types, list)
            if len(types) == 0 or not isinstance(types[0], str):
                return False
            element["types"] = set(types)
            element["text"] = TextUtils.replace_space(element["text"])
        
        valid_elements = EDGEAnnotationReader.filter_elements(page_anno["elements"], viewport)
        if len(valid_elements) > 0:
            page_anno["elements"] = valid_elements
            return True
        
        return False

    @staticmethod
    def filter_elements(elements, viewport):
        # TODO: 适应新的标注格式
        valid_elements = []
        ### 1 过滤质量不高的网页
        # 1.1 标注不能太多或太少
        if len(elements) < 5 or len(elements) > 60:
            return valid_elements
        
        # 1.2 标注框必须都有效，且不能大部分都是特别扁长形的框
        bboxes = np.array([element["bbox"] for element in elements])
        width, height = BoxUtils.get_wh_array(bboxes)
        if (width == 0).any() or (height == 0).any():
            return valid_elements
        vw, vh = viewport
        if len(elements) > 5 and (width / height > 20).mean() > 0.5:
            return valid_elements
        
        # 1.3 标注框不能大部分都太小
        if (width * height < vw * vh / 50**2).mean() > 0.4:
            return valid_elements
        
        # 1.4 不能大部分都是文本
        text_elements = [element for element in elements if element["types"] == {"Text"}]
        # word_count = sum(TextUtils.word_count(element["text"]) for element in text_elements)
        word_count = sum(len(element["text"].split()) for element in text_elements)
        if len(elements) > 5 and len(text_elements) / len(elements) > 0.8 or (word_count > 300 and word_count / len(text_elements) > 30):
            return valid_elements
            
        ### 2 过滤网页中的极端标注
        text2cnt = Counter([element["text"] for element in elements])
        for element in elements:
            text, bbox, types = element["text"], element["bbox"], element["types"]
            width, height = BoxUtils.get_wh(bbox)
            # 2.1 文本不能太长
            if len(text) > 300 or len(text.split()) > 60:
                continue 
            
            # 2.2 边界框长宽比例不能太极端
            if height == 0 or width / height > 30 or width / height < 1/30:
                continue
            
            # 2.3 元素不能太小（长、宽、面积）
            if width < 20 or height < 20 or width * height < 500:
                continue
            
            # 2.3 不能无类别
            if len(types) == 0:
                continue
            
            # 2.4 不能有重复项
            if len(text) > 0 and text2cnt[text] > 1:
                continue
            
            valid_elements.append(element)
            
        return valid_elements
