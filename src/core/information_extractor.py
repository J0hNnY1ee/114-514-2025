# information_extractor.py
import torch
from transformers import AutoTokenizer, AutoConfig # 需要 AutoConfig
from typing import List, Dict, Optional, Tuple, Union
import os

try:
    from models.ie_model import RobertaForMultiLabelTokenClassification 
    from data_process.preprocessor import GENERIC_ID_TO_IOB_LABEL, GENERIC_IOB_LABELS, IGNORE_INDEX
except ImportError as e:
    print(f"ImportError in InformationExtractor: {e}. 确保 model.py 和 preprocessor.py 可访问。")
    GENERIC_ID_TO_IOB_LABEL = {0: "O", 1: "B", 2: "I"}
    GENERIC_IOB_LABELS = ["O", "B", "I"]
    IGNORE_INDEX = -100
    class RobertaForMultiLabelTokenClassification: # Placeholder
        def __init__(self, model_name_or_path, num_single_layer_labels, dropout_prob=0.1): pass
        def load_state_dict(self, state_dict): pass
        def to(self, device): pass
        def eval(self): pass


class InformationExtractor:
    def __init__(
        self,
        trained_model_checkpoint_path: str,  # 训练好的模型权重pytorch_model.bin所在的目录
        base_model_name_or_path: str,       # 用于加载config和tokenizer的基础模型路径/名称
        device: Union[str, torch.device] = "cpu",
        max_seq_length: int = 512,
        id_to_iob_label_map: Optional[Dict[int, str]] = None,
        num_iob_labels_per_layer: Optional[int] = None,
    ):
        """
        初始化信息抽取器。

        参数:
            trained_model_checkpoint_path (str): 包含训练好的模型权重 (pytorch_model.bin) 的目录路径。
            base_model_name_or_path (str): 基础预训练模型（如 'models/chinese-roberta-wwm-ext-large' 或 
                                           Hugging Face Hub上的名称）的路径或名称。
                                           将从此路径加载 AutoConfig 和 AutoTokenizer。
            device (Union[str, torch.device]): "cpu" 或 "cuda" 或 torch.device 对象。
            max_seq_length (int): 输入序列的最大长度。
            id_to_iob_label_map (Dict[int, str], optional): 通用IOB标签ID到标签字符串的映射。
            num_iob_labels_per_layer (int, optional): 每个独立标注层的IOB标签数量 (通常是3: O,B,I)。
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Tokenizer 和 Config 从 base_model_name_or_path 加载
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.max_seq_length = max_seq_length

        self.id_to_iob_label = id_to_iob_label_map if id_to_iob_label_map else GENERIC_ID_TO_IOB_LABEL
        self.num_iob_labels = num_iob_labels_per_layer if num_iob_labels_per_layer else len(GENERIC_IOB_LABELS)

        try:
            # 1. 实例化模型结构，使用 base_model_name_or_path 来获取配置
            # RobertaForMultiLabelTokenClassification 的 model_name_or_path 参数用于加载其内部的 AutoModel 和 AutoConfig
            self.model = RobertaForMultiLabelTokenClassification(
                model_name_or_path=base_model_name_or_path, # 用于内部的 AutoModel 和 AutoConfig
                num_single_layer_labels=self.num_iob_labels
            )
            
            # 2. 加载训练好的权重
            model_weights_path = os.path.join(trained_model_checkpoint_path, "pytorch_model.bin")
            if not os.path.exists(model_weights_path):
                raise FileNotFoundError(
                    f"训练好的模型权重 'pytorch_model.bin' 未在 '{trained_model_checkpoint_path}' 找到。"
                )
            
            print(f"尝试从 '{model_weights_path}' 加载权重...")
            self.model.load_state_dict(
                torch.load(model_weights_path, map_location=self.device)
            )
            
            self.model.to(self.device)
            self.model.eval()
            print(f"多标签信息抽取模型已从 '{trained_model_checkpoint_path}' (权重) "
                  f"和 '{base_model_name_or_path}' (配置/基础模型) 加载到 {self.device}")
        except Exception as e:
            print(f"从 '{trained_model_checkpoint_path}' 加载IE模型时出错: {e}")
            print("请确保：")
            print(f"  1. '{base_model_name_or_path}' 是一个有效的基础模型路径/名称，包含config.json。")
            print(f"  2. '{trained_model_checkpoint_path}' 目录中包含 'pytorch_model.bin' 文件。")
            print(f"  3. 模型结构 (RobertaForMultiLabelTokenClassification) 与保存的权重匹配。")
            raise

    # _decode_iob_to_typed_entities 方法保持不变
    def _decode_iob_to_typed_entities(
        self, tokens: List[str], iob_tags: List[str], entity_main_type: str
    ) -> List[Dict[str, str]]:
        entities = []
        current_entity_tokens = []
        for i, tag in enumerate(iob_tags):
            token = tokens[i]
            if tag == "B":
                if current_entity_tokens:
                    entities.append({"entity_type": entity_main_type, "text": "".join(current_entity_tokens)})
                current_entity_tokens = [token]
            elif tag == "I":
                if current_entity_tokens:
                    current_entity_tokens.append(token)
                else: # I without B
                    current_entity_tokens = [token] # Treat as B
            elif tag == "O":
                if current_entity_tokens:
                    entities.append({"entity_type": entity_main_type, "text": "".join(current_entity_tokens)})
                current_entity_tokens = []
        if current_entity_tokens:
            entities.append({"entity_type": entity_main_type, "text": "".join(current_entity_tokens)})
        return entities

    # _pair_targets_and_arguments_advanced 方法保持不变
    def _pair_targets_and_arguments_advanced(
        self, 
        target_entities: List[Dict[str, str]], 
        argument_entities: List[Dict[str, str]],
        original_text: str 
    ) -> List[Dict[str, str]]:
        if not target_entities and not argument_entities: return []
        if not target_entities:
            return [{"target": "NULL", "argument": arg_e["text"]} for arg_e in argument_entities]
        if not argument_entities:
            return [{"target": tgt_e["text"], "argument": "NULL"} for tgt_e in target_entities]

        def get_char_spans(text, entity_list):
            spans = []
            # 简单地为每个解码出的实体文本在原文中查找第一次出现
            # 注意：如果实体文本在原文中多次出现，这可能不准确地反映原始TA对的位置
            for entity in entity_list:
                entity_text_to_find = entity.get("text", "")
                if not entity_text_to_find: # 跳过空实体文本
                    spans.append({"text": "", "start": -1, "end": -1, "original_entity": entity})
                    continue
                try:
                    start = text.find(entity_text_to_find) 
                    if start != -1:
                        spans.append({"text": entity_text_to_find, "start": start, "end": start + len(entity_text_to_find), "original_entity": entity})
                    else:
                        spans.append({"text": entity_text_to_find, "start": -1, "end": -1, "original_entity": entity})
                except TypeError:
                     spans.append({"text": str(entity_text_to_find), "start": -1, "end": -1, "original_entity": entity})
            spans.sort(key=lambda x: x["start"] if x["start"] != -1 else float('inf'))
            return spans

        target_spans_with_info = get_char_spans(original_text, target_entities)
        argument_spans_with_info = get_char_spans(original_text, argument_entities)
        
        pairs = []
        used_arg_indices = [False] * len(argument_spans_with_info)

        for t_info in target_spans_with_info:
            current_target_text = t_info["text"]
            if t_info["start"] == -1 and current_target_text != "NULL": # 如果Target文本有效但找不到位置
                print(f"警告: 在配对时，Target实体 '{current_target_text}' 未在原文 '{original_text[:50]}...' 中找到。")
                # 仍然尝试为它配对，或者直接给它一个NULL argument
                # pairs.append({"target": current_target_text, "argument": "NULL"})
                # continue

            best_arg_text_match = "NULL" # 默认为NULL
            best_arg_idx = -1
            # 使用一个分数来选择最佳Argument，例如重叠度优先，其次是邻近度
            best_match_score = -1 

            for j, a_info in enumerate(argument_spans_with_info):
                if not used_arg_indices[j] and a_info["start"] != -1:
                    current_arg_text = a_info["text"]
                    score = 0
                    # 计算Target和Argument之间的关系分数
                    # 1. 重叠度
                    if t_info["start"] != -1: # 只有当Target位置已知时才能计算重叠
                        overlap_start = max(t_info["start"], a_info["start"])
                        overlap_end = min(t_info["end"], a_info["end"])
                        overlap_length = max(0, overlap_end - overlap_start)
                        if overlap_length > 0:
                            score += 100 + overlap_length # 重叠是强信号
                    
                    # 2. 邻近度 (Argument在Target之后)
                    if t_info["start"] != -1 and a_info["start"] >= t_info["end"]: # Argument在Target之后
                        distance = a_info["start"] - t_info["end"]
                        score += max(0, 50 - distance) # 越近分数越高
                    elif t_info["start"] != -1 and a_info["end"] <= t_info["start"]: # Argument在Target之前
                        distance = t_info["start"] - a_info["end"]
                        score += max(0, 20 - distance) # 也给一点分，但低于在后的

                    if score > best_match_score:
                        best_match_score = score
                        best_arg_text_match = current_arg_text
                        best_arg_idx = j
            
            pairs.append({"target": current_target_text, "argument": best_arg_text_match})
            if best_arg_idx != -1:
                used_arg_indices[best_arg_idx] = True
        
        # 添加那些没有配对上的Argument
        for j, a_info in enumerate(argument_spans_with_info):
            if not used_arg_indices[j] and a_info["start"] != -1:
                pairs.append({"target": "NULL", "argument": a_info["text"]})
        
        valid_pairs = [p for p in pairs if not (p["target"] == "NULL" and p["argument"] == "NULL")]
        if not valid_pairs and (target_entities or argument_entities): # 如果有实体但没配上对
             # 可能所有T都配了NULL A，所有A都配了NULL T，然后被过滤了
             # 至少返回一个，避免完全空，除非T和A列表都为空
            if target_entities and not argument_entities:
                return [{"target": t["text"], "argument": "NULL"} for t in target_entities]
            if argument_entities and not target_entities:
                return [{"target": "NULL", "argument": a["text"]} for a in argument_entities]

        return valid_pairs

    # extract 方法保持不变，它调用的是修改后的_decode和_pair方法
    def extract(
        self, text: Union[str, List[str]]
    ) -> Union[List[Dict[str, str]], List[List[Dict[str, str]]]]:
        is_single_input = isinstance(text, str)
        texts_to_process = [text] if is_single_input else text
        all_results = []

        for single_text in texts_to_process:
            inputs = self.tokenizer(
                single_text, max_length=self.max_seq_length, padding="max_length",
                truncation=True, return_offsets_mapping=True,
                return_attention_mask=True, return_tensors="pt"
            )
            input_ids_tensor = inputs["input_ids"].to(self.device)
            attention_mask_tensor = inputs["attention_mask"].to(self.device)
            token_type_ids_tensor = inputs.get("token_type_ids")
            if token_type_ids_tensor is not None:
                token_type_ids_tensor = token_type_ids_tensor.to(self.device)

            with torch.no_grad():
                model_outputs = self.model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                    token_type_ids=token_type_ids_tensor,
                    return_dict=True
                )
                logits_target = model_outputs["logits_target"]
                logits_argument = model_outputs["logits_argument"]
            
            preds_ids_target = torch.argmax(logits_target, dim=-1).squeeze(0).cpu().tolist()
            preds_ids_argument = torch.argmax(logits_argument, dim=-1).squeeze(0).cpu().tolist()

            tokens_for_decoding = []
            iob_tags_target_str = []
            iob_tags_argument_str = []
            
            offsets = inputs["offset_mapping"].squeeze(0).cpu().tolist()
            attention_mask_list = attention_mask_tensor.squeeze(0).cpu().tolist()

            for i in range(len(preds_ids_target)):
                if attention_mask_list[i] == 1:
                    start_char, end_char = offsets[i]
                    if start_char == end_char == 0: continue
                    
                    token_id = input_ids_tensor[0, i].item()
                    if token_id in self.tokenizer.all_special_ids:
                        continue
                    
                    tokens_for_decoding.append(single_text[start_char:end_char])
                    iob_tags_target_str.append(self.id_to_iob_label.get(preds_ids_target[i], "O"))
                    iob_tags_argument_str.append(self.id_to_iob_label.get(preds_ids_argument[i], "O"))
            
            target_entities = self._decode_iob_to_typed_entities(tokens_for_decoding, iob_tags_target_str, "TGT")
            argument_entities = self._decode_iob_to_typed_entities(tokens_for_decoding, iob_tags_argument_str, "ARG")
            
            ta_pairs = self._pair_targets_and_arguments_advanced(target_entities, argument_entities, single_text)
            all_results.append(ta_pairs)

        return all_results[0] if is_single_input else all_results


# --- 示例用法 ---
if __name__ == "__main__":
    # 指向你训练好的多标签IE模型的权重文件所在的目录
    TRAINED_MODEL_CHECKPOINT_DIR = "models/outputs/ie_multilabel/best_model_on_eval_loss" 
    # 指向用于获取config和tokenizer的基础模型
    BASE_MODEL_FOR_CONFIG_AND_TOKENIZER = "models/chinese-roberta-wwm-ext-large"
    # BASE_MODEL_FOR_CONFIG_AND_TOKENIZER = "hfl/chinese-roberta-wwm-ext" # 备用

    if 'GENERIC_ID_TO_IOB_LABEL' not in globals(): # 确保测试时定义了
        print("Running information_extractor.py directly, using inline GENERIC_IOB definitions.")
        GENERIC_IOB_LABELS = ["O", "B", "I"]
        GENERIC_ID_TO_IOB_LABEL = {0:"O", 1:"B", 2:"I"}

    if not os.path.exists(os.path.join(TRAINED_MODEL_CHECKPOINT_DIR, "pytorch_model.bin")):
        print(f"错误: 训练好的多标签IE模型权重 'pytorch_model.bin' 未在 '{TRAINED_MODEL_CHECKPOINT_DIR}' 找到。请先运行训练。")
    else:
        try:
            extractor = InformationExtractor(
                trained_model_checkpoint_path=TRAINED_MODEL_CHECKPOINT_DIR,
                base_model_name_or_path=BASE_MODEL_FOR_CONFIG_AND_TOKENIZER,
                device="cuda" if torch.cuda.is_available() else "cpu",
                id_to_iob_label_map=GENERIC_ID_TO_IOB_LABEL, # 使用通用映射
                num_iob_labels_per_layer=len(GENERIC_IOB_LABELS)
            )

            test_texts = [
                "有三个豪爽的地方很特别，与众不同。可以黑别人，别人不能说它们。😆", 
                "除非是想德艾滋病。", 
                "张三打李四因为李四偷东西。", 
                "给他脸了，这不110", 
                "这部电影真好看，主角演的也好，配角演的也好。" # T1=电影, A1=好看; T2=主角, A2=演的好; T3=配角, A3=演的好
            ]

            print(f"\n--- 开始使用多标签模型抽取 Target 和 Argument (新初始化方式) ---")
            for i, text in enumerate(test_texts):
                print(f"\n原始文本 {i+1}: {text}")
                extracted_pairs = extractor.extract(text)
                if extracted_pairs:
                    for pair_idx, pair in enumerate(extracted_pairs):
                        print(f"  抽取对 {pair_idx+1}: Target='{pair['target']}', Argument='{pair['argument']}'")
                else:
                    print("  (未抽取到 Target-Argument 对)")
        except Exception as e:
            print(f"运行示例时发生错误: {e}")
            import traceback
            traceback.print_exc()