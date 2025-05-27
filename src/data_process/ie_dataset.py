import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional

try:
    from data_process.loader import HateOriDataset
    from data_process.preprocessor import GENERIC_IOB_LABEL_TO_ID, GENERIC_ID_TO_IOB_LABEL, IGNORE_INDEX
except ImportError:
    try:
        from .loader import HateOriDataset
        from .preprocessor import GENERIC_IOB_LABEL_TO_ID, GENERIC_ID_TO_IOB_LABEL, IGNORE_INDEX
    except ImportError:
        print("CRITICAL ERROR: Could not import HateOriDataset or GENERIC IOB label definitions from preprocessor.")
        print("Please ensure loader.py and preprocessor.py are in the correct path and contain GENERIC_IOB definitions.")
        class HateOriDataset: pass # Fallback
        GENERIC_IOB_LABEL_TO_ID = {"O": 0, "B": 1, "I": 2} # Fallback
        GENERIC_ID_TO_IOB_LABEL = {v: k for k, v in GENERIC_IOB_LABEL_TO_ID.items()} # Fallback
        IGNORE_INDEX = -100 # Fallback


class HateIEDataset(Dataset):
    def __init__(self,
                 original_json_path: str,
                 tokenizer_name_or_path: str,
                 max_seq_length: int = 512,
                 # max_char_overlap_ratio_with_other_ta_pairs (已移除)
                 ):
        self.original_dataset = HateOriDataset(original_json_path, is_pred=False)
        if not hasattr(self.original_dataset, '__getitem__') or not hasattr(self.original_dataset, '__len__'):
             raise ImportError("Failed to properly load HateOriDataset. Check loader.py.")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_seq_length = max_seq_length
        
        self.iob_label_to_id = GENERIC_IOB_LABEL_TO_ID
        self.id_to_iob_label = GENERIC_ID_TO_IOB_LABEL

        self.processed_data = self._preprocess_data()
        if not self.processed_data:
            print(f"警告: _preprocess_data for {original_json_path} returned no samples. "
                  "Check data or preprocessing logic.")

    def _find_first_occurrence(self, text: str, subtext: str) -> Optional[Tuple[int, int]]:
        """查找subtext在text中第一次出现的位置。"""
        if not subtext or subtext == "NULL":
            return None
        
        pos = text.find(subtext)
        if pos != -1:
            return (pos, pos + len(subtext))
        return None

    def _map_char_span_to_token_span_robust(self,
                                            char_start: int,
                                            char_end: int,
                                            offset_mapping: List[Tuple[int, int]]
                                            ) -> Tuple[int, int]:
        """
        更鲁棒地将字符span映射到token span。
        返回 (token_start_index, token_end_index) 或 (-1, -1) 如果失败。
        """
        token_start, token_end = -1, -1
        for i, (offset_s, offset_e) in enumerate(offset_mapping):
            if offset_s == offset_e == 0: continue
            if token_start == -1 and offset_s <= char_start < offset_e:
                token_start = i
            if token_start != -1 and offset_s < char_end and offset_e >= char_start:
                token_end = i
        if token_start != -1 and token_end == -1:
             s_offset, e_offset = offset_mapping[token_start]
             if s_offset <= char_start and e_offset >= char_end: token_end = token_start
        if token_start != -1 and token_end != -1 and token_start <= token_end:
            actual_char_s = offset_mapping[token_start][0]
            actual_char_e = offset_mapping[token_end][1]
            if actual_char_s <= char_start and actual_char_e >= char_end:
                return token_start, token_end
        return -1, -1

    def _apply_iob_labels_to_sequence(self,
                                      token_start_idx: int,
                                      token_end_idx: int,
                                      labels_sequence: List[int]
                                     ):
        """将B和I标签应用到给定的labels_sequence的指定token span上。"""
        if token_start_idx != -1 and token_end_idx != -1 and token_start_idx <= token_end_idx:
            b_id = self.iob_label_to_id["B"]
            i_id = self.iob_label_to_id["I"]
            if labels_sequence[token_start_idx] != IGNORE_INDEX:
                labels_sequence[token_start_idx] = b_id
            for k in range(token_start_idx + 1, token_end_idx + 1):
                if labels_sequence[k] != IGNORE_INDEX:
                    labels_sequence[k] = i_id

    def _preprocess_data(self) -> List[Dict]:
        processed_samples = []
        for i in range(len(self.original_dataset)):
            item_id, content, ta_pairs = self.original_dataset[i]

            if not ta_pairs and not self.original_dataset.is_pred:
                continue

            tokenized_output = self.tokenizer(
                content, max_length=self.max_seq_length, padding="max_length",
                truncation=True, return_offsets_mapping=True,
                return_attention_mask=True, return_token_type_ids=True
            )
            input_ids = tokenized_output["input_ids"]
            attention_mask = tokenized_output["attention_mask"]
            token_type_ids = tokenized_output.get("token_type_ids")
            offset_mapping = tokenized_output["offset_mapping"]
            
            labels_target = [self.iob_label_to_id["O"]] * len(input_ids)
            labels_argument = [self.iob_label_to_id["O"]] * len(input_ids)

            for token_idx, offset in enumerate(offset_mapping):
                if offset == (0, 0) or attention_mask[token_idx] == 0:
                    labels_target[token_idx] = IGNORE_INDEX
                    labels_argument[token_idx] = IGNORE_INDEX
            
            # 不再使用全局字符占用跟踪器进行预过滤

            for ta_pair_idx, ta_pair in enumerate(ta_pairs):
                target_text = ta_pair["target"]
                argument_text = ta_pair["argument"]
                
                # --- 1. 标记Target层 ---
                target_labeled_this_ta_pair = False
                if target_text and target_text != "NULL":
                    # 查找Target的第一个出现
                    chosen_t_char_span = self._find_first_occurrence(content, target_text)
                    if chosen_t_char_span:
                        t_tok_start, t_tok_end = self._map_char_span_to_token_span_robust(
                            chosen_t_char_span[0], chosen_t_char_span[1], offset_mapping
                        )
                        if t_tok_start != -1:
                            # 对于两层标签，我们直接在对应层打标，不检查是否已被另一层占用
                            # 因为模型有两个独立的头来预测这两层
                            self._apply_iob_labels_to_sequence(
                                t_tok_start, t_tok_end, labels_target
                            )
                            target_labeled_this_ta_pair = True
                
                if not target_labeled_this_ta_pair and target_text and target_text != "NULL":
                    # 调试信息：打印实体是否在content中，以及offset_mapping
                    found_raw = content.find(target_text) != -1
                    print(f"警告: 样本ID {item_id} (TA对 {ta_pair_idx}), 未能为Target '{target_text}' 生成Target层token标签. "
                          f"在Content中找到: {found_raw}. Content: '{content[:70]}...'")
                    # if found_raw: # 如果找到了但映射失败，打印offset
                    #     print(f"Offset Mapping (sample): {offset_mapping[:20]}")


                # --- 2. 标记Argument层 ---
                argument_labeled_this_ta_pair = False
                if argument_text and argument_text != "NULL":
                    chosen_a_char_span = self._find_first_occurrence(content, argument_text)
                    if chosen_a_char_span:
                        a_tok_start, a_tok_end = self._map_char_span_to_token_span_robust(
                            chosen_a_char_span[0], chosen_a_char_span[1], offset_mapping
                        )
                        if a_tok_start != -1:
                            self._apply_iob_labels_to_sequence(
                                a_tok_start, a_tok_end, labels_argument
                            )
                            argument_labeled_this_ta_pair = True
                
                if not argument_labeled_this_ta_pair and argument_text and argument_text != "NULL":
                     found_raw = content.find(argument_text) != -1
                     print(f"警告: 样本ID {item_id} (TA对 {ta_pair_idx}), 未能为Argument '{argument_text}' 生成Argument层token标签. "
                           f"在Content中找到: {found_raw}. Content: '{content[:70]}...'")
                     # if found_raw:
                     #    print(f"Offset Mapping (sample): {offset_mapping[:20]}")
            
            sample = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels_target": torch.tensor(labels_target, dtype=torch.long),
                "labels_argument": torch.tensor(labels_argument, dtype=torch.long)
            }
            if token_type_ids is not None and len(token_type_ids) > 0:
                 sample["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)
            processed_samples.append(sample)
        return processed_samples

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


# --- 示例用法 ---
if __name__ == "__main__":
    import json
    import os

    if 'GENERIC_IOB_LABEL_TO_ID' not in globals():
        print("Running ie_dataset.py directly, using inline GENERIC_IOB definitions for testing.")
        GENERIC_IOB_LABELS = ["O", "B", "I"]
        GENERIC_IOB_LABEL_TO_ID = {"O": 0, "B": 1, "I": 2}
        GENERIC_ID_TO_IOB_LABEL = {i: label for i, label in enumerate(GENERIC_IOB_LABELS)}
        IGNORE_INDEX = -100
        HateIEDataset.iob_label_to_id = GENERIC_IOB_LABEL_TO_ID
        HateIEDataset.id_to_iob_label = GENERIC_ID_TO_IOB_LABEL

    temp_original_data = [
        { "id": 3534, "content": "有三个豪爽的地方很特别，与众不同。可以黑别人，别人不能说它们。😆", "output": "它们 | 可以黑别人，别人不能说它们 | Region | hate [END]"},
        { "id": 5672, "content": "除非是想德艾滋病。", "output": "除非是想德艾滋病 | 除非是想德艾滋病 | non-hate | non-hate [END]"},
        { "id": 1, "content": "张三打李四因为李四偷东西。", "output": "张三 | 打李四 | LGBTQ | Hateful [SEP] 李四 | 偷东西 | LGBTQ | Hateful [END]"},
        { "id": 7620, "content": "给他脸了，这不110", "output": "他 | 给他脸了 | non-hate | non-hate [END]"},
        { "id": 3828, "content": "她没资格评价我，她自己什么样心里清楚", "output": "她 | 没资格评价我 | LGBTQ | Hateful [SEP] 她自己 | 什么样心里清楚 | LGBTQ | Hateful [END]"}, # "她" 出现多次
        { "id": 5208, "content": "同性取向不是病，恐同才是病，同性取向是天生的。", "output": "同性取向 | 不是病 | LGBTQ | NonHateful [SEP] 恐同 | 才是病 | LGBTQ | Hateful [SEP] 同性取向 | 是天生的 | LGBTQ | NonHateful [END]"}, # "同性取向" 出现多次
        { "id": 6459, "content": "河南的电信诈骗全国闻名，怎么洗都没用，上次去河南的洛阳玩差点被偷了钱包。", "output": "河南的 | 电信诈骗全国闻名 | Region | Hateful [END]"}
    ]
    temp_json_path = "temp_ie_simplified_filter_test_data.json"
    with open(temp_json_path, 'w', encoding='utf-8') as f:
        json.dump(temp_original_data, f, ensure_ascii=False, indent=2)

    tokenizer_path = "models/chinese-roberta-wwm-ext-large"
    # tokenizer_path = "hfl/chinese-roberta-wwm-ext" 

    print(f"使用Tokenizer: {tokenizer_path}")
    print("-" * 30)

    try:
        # 移除了 max_char_overlap_ratio_with_other_ta_pairs 参数，因为它不再被使用
        ie_dataset = HateIEDataset(
            original_json_path=temp_json_path,
            tokenizer_name_or_path=tokenizer_path,
            max_seq_length=128 # 增加max_seq_length以容纳更长的content
        )
    except Exception as e:
        print(f"初始化 HateIEDataset 失败: {e}")
        import traceback
        traceback.print_exc()
        if os.path.exists(temp_json_path): os.remove(temp_json_path)
        exit()

    print(f"HateIEDataset 大小: {len(ie_dataset)}")
    print("-" * 30)

    if not ie_dataset:
        print("未能生成任何处理后的样本。")
    else:
        for i in range(len(ie_dataset)):
            sample = ie_dataset[i]
            original_sample_info = ie_dataset.original_dataset[i] # (id, content, ta_pairs_dicts)
            print(f"\n--- 处理后样本 {i} (原始ID: {original_sample_info[0]}) ---")
            print(f"原始Content: {original_sample_info[1]}")
            print(f"原始TA对: {original_sample_info[2]}")


            original_tokens = ie_dataset.tokenizer.convert_ids_to_tokens(sample['input_ids'].tolist())
            
            readable_labels_tgt = []
            for token_idx, label_id in enumerate(sample['labels_target'].tolist()):
                if label_id != IGNORE_INDEX:
                    readable_labels_tgt.append(f"{original_tokens[token_idx]}({ie_dataset.id_to_iob_label.get(label_id, 'UNK')})")
            print(f"Readable Labels Target: {' '.join(readable_labels_tgt)}")

            readable_labels_arg = []
            for token_idx, label_id in enumerate(sample['labels_argument'].tolist()):
                if label_id != IGNORE_INDEX:
                    readable_labels_arg.append(f"{original_tokens[token_idx]}({ie_dataset.id_to_iob_label.get(label_id, 'UNK')})")
            print(f"Readable Labels Argument: {' '.join(readable_labels_arg)}")
    
    if os.path.exists(temp_json_path): os.remove(temp_json_path)
    print("\n简化过滤的 HateIEDataset 测试运行完成。")