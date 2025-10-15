import torch
from torch.utils.data import Dataset
import os, sys
import random
import pandas as pd
from .masker import mask
LABEL_DICT = {"SUPPORTS":0, "REFUTES":1, "NOT ENOUGH INFO":2}
ANTONYM_PAIRS = {
    "tốt": ["xấu", "kém", "tồi", "dở"],
    "xấu": ["đẹp", "tốt", "dễ nhìn"],
    "đẹp": ["xấu", "tệ", "kinh khủng"],
    "cao": ["thấp", "lùn", "ngắn"],
    "thấp": ["cao", "dài"],
    "mạnh": ["yếu", "mềm"],
    "yếu": ["mạnh", "cứng", "rắn"],
    "đúng": ["sai", "nhầm", "trật"],
    "sai": ["đúng", "chuẩn"],
    "giàu": ["nghèo", "bần", "khó khăn"],
    "nghèo": ["giàu", "sung túc"],
    "vui": ["buồn", "chán", "phiền"],
    "buồn": ["vui", "hạnh phúc"],
    "dễ": ["khó", "phức tạp"],
    "khó": ["dễ", "đơn giản"],
    "lớn": ["nhỏ", "bé"],
    "nhỏ": ["lớn", "to"],
    "nhiều": ["ít", "thiếu"],
    "ít": ["nhiều", "dư"],
    "thắng": ["thua", "bại"],
    "thua": ["thắng", "chiến thắng"],
    "tăng": ["giảm", "hạ"],
    "giảm": ["tăng", "cao"],
    "mở": ["đóng", "khóa"],
    "đóng": ["mở", "bật"],
    "tích cực": ["tiêu cực", "bị động"],
    "tiêu cực": ["tích cực", "chủ động"],
    "sáng": ["tối", "mờ"],
    "tối": ["sáng", "chói"],
    "dày": ["mỏng", "thưa"],
    "mỏng": ["dày", "đặc"],
    "nhanh": ["chậm", "từ tốn"],
    "chậm": ["nhanh", "gấp"],
    "xa": ["gần", "cận"],
    "gần": ["xa", "xa xôi"],
    "mới": ["cũ", "lỗi thời"],
    "cũ": ["mới", "hiện đại"],
    "trẻ": ["già", "cao tuổi"],
    "già": ["trẻ", "non"],
    "nóng": ["lạnh", "mát"],
    "lạnh": ["nóng", "ấm"],
    "đông": ["vắng", "ít người"],
    "vắng": ["đông", "tấp nập"],
    "sạch": ["bẩn", "dơ"],
    "bẩn": ["sạch", "ngăn nắp"],
    "an toàn": ["nguy hiểm", "rủi ro"],
    "nguy hiểm": ["an toàn", "ổn định"],
    "ổn định": ["bất ổn", "dao động"],
    "bất ổn": ["ổn định"],
    "dài": ["ngắn", "cụt"],
    "ngắn": ["dài", "kéo dài"],
    "sớm": ["muộn", "trễ"],
    "muộn": ["sớm", "đúng giờ"],
    "dễ chịu": ["khó chịu", "bực bội"],
    "khó chịu": ["dễ chịu", "thoải mái"],
    "thật": ["giả", "dối"],
    "giả": ["thật", "xịn"],
    "cao cấp": ["thấp cấp", "bình dân"],
    "thấp cấp": ["cao cấp"],
    "bận": ["rảnh", "nhàn"],
    "rảnh": ["bận", "bận rộn"],
    "đầy": ["rỗng", "trống"],
    "rỗng": ["đầy", "chặt"],
    "dày đặc": ["thưa thớt"],
    "thưa thớt": ["dày đặc"],
    "thông minh": ["ngu", "đần", "ngốc"],
    "ngu": ["thông minh", "lanh lợi"],
    "may mắn": ["xui xẻo", "rủi ro"],
    "xui": ["may", "may mắn"],
    "cao quý": ["hèn hạ", "thấp kém"],
    "hèn hạ": ["cao quý", "đáng kính"],
    "thực tế": ["ảo tưởng", "phi thực tế"],
    "ảo": ["thật", "hiện thực"],
    "thân thiện": ["thù địch", "xa cách"],
    "thù địch": ["thân thiện", "bạn bè"],
    "trung thực": ["dối trá", "gian dối"],
    "gian dối": ["trung thực", "thật thà"],
    "vui vẻ": ["u sầu", "buồn rầu"],
    "tự tin": ["rụt rè", "mặc cảm"],
    "rụt rè": ["tự tin", "mạnh dạn"],
    "tự do": ["giam cầm", "ràng buộc"],
    "giam cầm": ["tự do"],
    "thành công": ["thất bại"],
    "thất bại": ["thành công"],
    "chấp nhận": ["từ chối"],
    "từ chối": ["chấp nhận", "đồng ý"],
    "bắt đầu": ["kết thúc", "dừng"],
    "kết thúc": ["bắt đầu", "mở đầu"],
    "ra ngoài": ["vào trong"],
    "vào trong": ["ra ngoài"],
    "đứng": ["ngồi", "nằm"],
    "ngồi": ["đứng", "đi"],
    "đi": ["đứng", "ở lại"],
    "ở lại": ["đi", "rời đi"],
    "đến": ["đi", "rời"],
    "rời": ["đến"],
    "đầy đủ": ["thiếu thốn"],
    "thiếu thốn": ["đầy đủ"],
    "mềm": ["cứng", "rắn"],
    "cứng": ["mềm"],
}

class CRDataset(Dataset):
    """
        This class is used to handle fact verification task and can either:
        1. Tokenize input (claim <sep> evidence).
        2. Support load data from dataset or list, dicts.
    """

    def __init__(self, data_source, mask_ratio = 0.15, tokenizer = None, max_len = None, transform = None, src_column = "original_VI", tgt_column = "original_VI", evidence_column = "gold_evidence_VI", label_column = "labels", selected_label = 1, is_inference = False):
        self.mask_ratio = mask_ratio
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform

        self.src_column = src_column
        self.tgt_column = tgt_column
        self.evidence_column = evidence_column
        self.label_column = label_column
        self.label_map = LABEL_DICT

        self.selected_label = selected_label
        self.is_inference = is_inference
        
        if isinstance(data_source, str):
            if data_source.endswith('.json'):
                df = pd.read_json(data_source)
            elif data_source.endswith('.parquet'):
                df = pd.read_parquet(data_source)
            elif data_source.endswith('.csv'):
                df = pd.read_csv(data_source)
            self.data = df.to_dict(orient = 'records')
        else:
            self.data = data_source

        filtered_data = []

        for item in self.data:
            raw_label = item[self.label_column]

            if raw_label in [0, 1, 2]:
                mapped_label = raw_label
            elif raw_label in self.label_map:
                mapped_label = self.label_map[raw_label]
            else:
                raise ValueError(f"Label {raw_label} không có trong label_map!")

            if mapped_label == selected_label:
                item[self.label_column] = mapped_label
                filtered_data.append(item)

        self.data = filtered_data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        instance = self.data[idx]
        src = instance[self.src_column]
        tgt = instance[self.tgt_column]

        src_encoding = self.tokenizer(
            src,
            max_length=self.max_len,
            truncation=True,
            padding=False,
            add_special_tokens=True,
            return_tensors=None
        )
        src_ids = torch.tensor(src_encoding["input_ids"], dtype=torch.long)

        sample = {
            "src_tokenization": src_ids,
            "idx": idx
        }

        if not self.is_inference:
            tgt_encoding = self.tokenizer(
                tgt,
                max_length=self.max_len,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_tensors=None
            )
            tgt_ids = torch.tensor(tgt_encoding["input_ids"], dtype=torch.long)
            sample["tgt_tokenization"] = tgt_ids

        return sample
