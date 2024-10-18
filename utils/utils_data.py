import re
import random
import numpy as np


class BoxUtils:
    @staticmethod
    def is_valid(bbox, size=[1, 1]):
        assert isinstance(bbox, list)
        W, H = size
        return (0 <= bbox[0] <= bbox[2] <= W) and (0 <= bbox[1] <= bbox[3] <= H)

    @staticmethod
    def round(bbox, ndigits=4):
        assert isinstance(bbox, list)
        assert ndigits == "random" or isinstance(ndigits, int) and ndigits > 0
        return list(map(lambda x: round(x, ndigits=random.randint(2, 4) if ndigits == "random" else ndigits), bbox))
        
    @staticmethod
    def round_to_int(bbox):
        assert isinstance(bbox, list)
        return list(map(lambda x: int(round(x)), bbox))
    
    @staticmethod
    def normalize(bbox, size):
        assert isinstance(bbox, list)
        W, H = size
        x1, y1, x2, y2 = bbox
        bbox_norm = [x1/W, y1/H, x2/W, y2/H]
        return bbox_norm
    
    @staticmethod
    def denormalize(bbox, size, round=True):
        assert isinstance(bbox, list)
        W, H = size
        x1, y1, x2, y2 = bbox
        bbox_denorm = [x1*W, y1*H, x2*W, y2*H]
        if round:
            bbox_denorm = BoxUtils.round_to_int(bbox_denorm)
        return bbox_denorm
    
    @staticmethod
    def xyxy_to_cxcywh(bbox):
        assert isinstance(bbox, list)
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    
    @staticmethod
    def cxcywh_to_xyxy(bbox):
        assert isinstance(bbox, list)
        cx, cy, w, h = bbox
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
    
    @staticmethod
    def x0y0wh_to_xyxy(bbox):
        assert isinstance(bbox, list)
        x0, y0, w, h = bbox
        return [x0, y0, x0+w, y0+h]
    
    @staticmethod
    def get_wh(bbox):
        assert isinstance(bbox, list)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    @staticmethod
    def get_cxcy(bbox, ndigits=4):
        assert isinstance(bbox, list)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if ndigits == "random":
            return [round(cx, random.randint(2, 4)), round(cy, random.randint(2, 4))]
        else:
            return [round(cx, ndigits), round(cy, ndigits)]
    
    @staticmethod
    def get_wh_array(bbox):
        assert isinstance(bbox, np.ndarray)
        return bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]
    
    @staticmethod
    def point_sampling(bbox, margin_ratio=0.3, ndigits=4):
        assert ndigits == "random" or isinstance(ndigits, int) and ndigits > 0
        assert isinstance(bbox, list)
        cx, cy, w, h = BoxUtils.xyxy_to_cxcywh(bbox)
        x_offset = (random.random() - 0.5) * w * (1 - margin_ratio)
        y_offset = (random.random() - 0.5) * h * (1 - margin_ratio)
        x = cx + x_offset
        y = cy + y_offset
        if ndigits == "random":
            return [round(x, random.randint(2, 4)), round(y, random.randint(2, 4))]
        else:
            return [round(x, ndigits), round(y, ndigits)]
    
    @staticmethod
    def point_sampling_cxcywh(bbox, margin_ratio=0.3, ndigit=-1):
        if isinstance(bbox, list):
            cx, cy, w, h = bbox
            x_offset = (random.random() - 0.5) * w * (1 - margin_ratio)
            y_offset = (random.random() - 0.5) * h * (1 - margin_ratio)
            x = cx + x_offset
            y = cy + y_offset
            if ndigit >= 0:
                return [round(x, 4), round(y, 4)]
        else:
            raise TypeError("The bounding box should be a list!")
    

class TextUtils:
    space_pattern = re.compile(
        r'[\u0020\u00A0\u202F\uFEFF\u205F\u3000'
        r'\u2000-\u200B\u200E-\u200F\u2028-\u2029\u2060-\u2064'
        r'\s]+'
    )
    
    @staticmethod
    def word_count(text):
        return len(text.split())
    
    @staticmethod
    def truncate_len(text, max_len=100):
        if len(text) <= max_len:
            return text
        else:
            return text[:max_len] + " ..."
    
    @staticmethod
    def truncate_words(text, max_words=5):
        words = text.split()
        if len(words) <= max_words:
            return text
        else:
            return ' '.join(words[:max_words]) + " ..."
        
    @staticmethod
    def truncate_both(text, max_words=5, max_len=30):
        words = text.split()
        if len(words) <= max_words and len(text) <= max_len:
            return text.strip()
        else:
            return ' '.join(words[:max_words])[:max_len].strip() + " ..."
        
    @classmethod
    def replace_space(cls, text):
        return cls.space_pattern.sub(' ', text)
                