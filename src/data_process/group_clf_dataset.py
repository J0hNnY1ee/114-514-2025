# data_process/targeted_group_clf_dataset.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional

try:
    from data_process.loader import HateOriDataset
    from data_process.preprocessor import GROUP_TO_ID, ID_TO_GROUP, ID_TO_HATEFUL, HATEFUL_TO_ID
except ImportError:
    # Fallback for direct execution or import issues
    class HateOriDataset: pass
    ID_TO_HATEFUL = {0: "non-hate", 1: "hate"} # Example
    HATEFUL_TO_ID = {"non-hate":0, "hate":1} # Example
    # Example GROUP_CATEGORIES based on your preprocessor snippet
    _fallback_group_cats = ["Sexism", "Racism", "Region", "LGBTQ", "others", "non-hate"]
    GROUP_TO_ID = {name: i for i, name in enumerate(_fallback_group_cats)}
    ID_TO_GROUP = {i: name for i, name in enumerate(_fallback_group_cats)}
    print("Warning: Using fallback label definitions in targeted_group_clf_dataset.py")


class TargetedGroupClassificationDataset(Dataset):
    def __init__(self,
                 original_json_path: str,
                 tokenizer_name_or_path: str,
                 max_seq_length: int = 256,
                 input_template: str = "[CLS] T: {target} A: {argument} H: {hateful_label} C: {content} [SEP]"
                 ):
        """
        为Targeted Group分类任务准备数据。
        直接使用 preprocessor.py 中定义的 GROUP_TO_ID 作为目标群体的标签。
        """
        self.original_dataset = HateOriDataset(original_json_path, is_pred=False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_seq_length = max_seq_length
        self.input_template = input_template
        
        self.ID_TO_HATEFUL = ID_TO_HATEFUL
        # self.hateful_label_to_id = HATEFUL_TO_ID # Potentially needed if template uses string
        self.targeted_group_to_id = GROUP_TO_ID # 使用现有的映射
        self.id_to_targeted_group = ID_TO_GROUP # 用于调试

        self.samples = self._create_classification_samples()

    def _create_classification_samples(self) -> List[Dict]:
        classification_samples = []
        skipped_samples_no_ta = 0
        skipped_samples_no_tg_label = 0

        for i in range(len(self.original_dataset)):
            original_id, original_content, ta_quadruples = self.original_dataset[i]

            if not ta_quadruples:
                skipped_samples_no_ta += 1
                continue

            for quad_info in ta_quadruples:
                target_text = quad_info["target"]
                argument_text = quad_info["argument"]
                hateful_id = quad_info["hateful_id"] # int
                original_target_group_ids = quad_info.get("target_group_ids", []) # List[int] from GROUP_TO_ID

                hateful_label_str = self.ID_TO_HATEFUL.get(hateful_id, "non-hate") # Default to non-hate if ID unknown
                
                final_targeted_group_id: Optional[int] = None

                if hateful_label_str == "non-hate":
                    # 如果是non-hate，TG标签应该对应 GROUP_TO_ID["non-hate"]
                    if "non-hate" in self.targeted_group_to_id:
                        final_targeted_group_id = self.targeted_group_to_id["non-hate"]
                    else: # 如果 "non-hate" 不在 GROUP_TO_ID 中 (不应该发生，根据你的preprocessor)
                        print(f"错误: 样本ID {original_id}, 'non-hate' 未在GROUP_TO_ID中定义!")
                        skipped_samples_no_tg_label += 1
                        continue
                else: # 如果是 hate
                    if original_target_group_ids: # 确保列表不为空
                        # 取第一个目标群体ID作为这个分类任务的标签
                        final_targeted_group_id = original_target_group_ids[0]
                        # 可选的验证：检查这个ID是否在 GROUP_TO_ID 的值中
                        if final_targeted_group_id not in self.id_to_targeted_group:
                             print(f"警告: 样本ID {original_id}, 从HateOriDataset获取的TG ID "
                                   f"'{final_targeted_group_id}' 无效 (不在ID_TO_GROUP中)。跳过此TA对。")
                             skipped_samples_no_tg_label += 1
                             continue
                    else: # Hateful 但没有提供目标群体
                        print(f"警告: 样本ID {original_id}, TA对被标记为hate但缺少target_group_ids。跳过此TA对。")
                        skipped_samples_no_tg_label += 1
                        continue
                
                if final_targeted_group_id is None: # 双重保险
                    print(f"警告: 样本ID {original_id}, 无法确定final_targeted_group_id。跳过。")
                    skipped_samples_no_tg_label +=1
                    continue

                target_for_input = "[EMPTY_TARGET]" if target_text == "NULL" or not str(target_text).strip() else str(target_text)
                argument_for_input = "[EMPTY_ARG]" if argument_text == "NULL" or not str(argument_text).strip() else str(argument_text)

                try:
                    input_text = self.input_template.format(
                        target=target_for_input,
                        argument=argument_for_input,
                        hateful_label=hateful_label_str, 
                        content=original_content
                    )
                except KeyError as e:
                    raise ValueError(f"TG输入模板 '{self.input_template}' 占位符错误: {e}.")

                tokenized_output = self.tokenizer(
                    input_text, max_length=self.max_seq_length, padding="max_length",
                    truncation=True, return_attention_mask=True, return_token_type_ids=True
                )

                classification_samples.append({
                    "input_ids": torch.tensor(tokenized_output["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(tokenized_output["attention_mask"], dtype=torch.long),
                    "token_type_ids": torch.tensor(tokenized_output.get("token_type_ids", []), dtype=torch.long),
                    "label": torch.tensor(final_targeted_group_id, dtype=torch.long),
                    "original_id": original_id,
                })
        
        if skipped_samples_no_ta > 0:
            print(f"信息: TG Dataset跳过 {skipped_samples_no_ta} 个原始样本 (因无TA对)。")
        if skipped_samples_no_tg_label > 0:
            print(f"警告: TG Dataset跳过 {skipped_samples_no_tg_label} 个TA对 (因无法确定有效TG标签)。")
        return classification_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "token_type_ids": sample["token_type_ids"],
            "labels": sample["label"]
        }

# --- 示例用法 ---
if __name__ == "__main__":
    import json
    import os
    # Fallback definitions if preprocessor.py is not in the path during direct execution
    if 'GROUP_TO_ID' not in globals(): # Check if already imported
        print("Running targeted_group_clf_dataset.py directly, using inline fallback label definitions.")
        _fallback_group_cats_main = ["Sexism", "Racism", "Region", "LGBTQ", "others", "non-hate"]
        GROUP_TO_ID = {name: i for i, name in enumerate(_fallback_group_cats_main)}
        ID_TO_GROUP = {i: name for i, name in enumerate(_fallback_group_cats_main)}
        ID_TO_HATEFUL = {0: "non-hate", 1: "hate"} # Assume 0 for non-hate, 1 for hate
        # HateClassificationDataset uses HATEFUL_TO_ID, so let's define it too for consistency in fallback
        HATEFUL_TO_ID = {v:k for k,v in ID_TO_HATEFUL.items()}


    # 模拟 HateOriDataset 的输出，特别是 target_group_ids
    # 这些ID应该基于 GROUP_TO_ID
    temp_original_data_tg = [
        { 
            "id": 1, "content": "江苏人都是骗子。", 
            "output": f"江苏人 | 都是骗子 | Region | hate [END]" # Ori output
            # HateOriDataset会解析为: target_group_ids=[GROUP_TO_ID['Region']], hateful_id=HATEFUL_TO_ID['hate']
        },
        { 
            "id": 2, "content": "这本书很好看。", 
            "output": f"这本书 | 很好看 | non-hate | non-hate [END]"
            # HateOriDataset会解析为: target_group_ids=[GROUP_TO_ID['non-hate']], hateful_id=HATEFUL_TO_ID['non-hate']
        },
        { 
            "id": 3, "content": "他这种行为很恶心。", 
            "output": f"他这种行为 | 很恶心 | others | hate [END]"
            # HateOriDataset会解析为: target_group_ids=[GROUP_TO_ID['others']], hateful_id=HATEFUL_TO_ID['hate']
        },
        { # 模拟原始数据有多个TG，HateOriDataset会返回一个ID列表
            "id": 4, "content": "黑人女性太糟糕了。", 
            "output": "黑人女性 | 太糟糕了 | Racism,Sexism | hate [END]" # 原始数据
            # HateOriDataset会解析为: target_group_ids=[GROUP_TO_ID['Racism'], GROUP_TO_ID['Sexism']], hateful_id=HATEFUL_TO_ID['hate']
            # 当前TG Dataset会取第一个ID，即 Racism 的ID。
        },
         { # 模拟Hateful但TG为空的情况 (HateOriDataset应该返回空的target_group_ids)
            "id": 5, "content": "这种人该死！",
            "output": "这种人 | 该死 | | hate [END]" # TG字段为空
        }
    ]
    temp_json_path_tg = "temp_tg_clf_simplified_test_data.json"
    with open(temp_json_path_tg, 'w', encoding='utf-8') as f:
        json.dump(temp_original_data_tg, f, ensure_ascii=False, indent=2)

    tokenizer_path_tg = "models/chinese-roberta-wwm-ext-large"
    # tokenizer_path_tg = "hfl/chinese-roberta-wwm-ext" # Fallback

    print(f"使用Tokenizer: {tokenizer_path_tg}")
    print(f"Targeted Group (GROUP_TO_ID) 标签映射: {GROUP_TO_ID}")
    print("-" * 30)

    try:
        tg_clf_dataset = TargetedGroupClassificationDataset(
            original_json_path=temp_json_path_tg,
            tokenizer_name_or_path=tokenizer_path_tg,
            max_seq_length=128,
            # Example template, can be configured
            input_template="[CLS] T: {target} A: {argument} IsHateful: {hateful_label} Context: {content} [SEP]"
        )
    except Exception as e:
        print(f"初始化 TargetedGroupClassificationDataset 失败: {e}")
        import traceback; traceback.print_exc()
        if os.path.exists(temp_json_path_tg): os.remove(temp_json_path_tg)
        exit()

    print(f"TargetedGroupClassificationDataset 大小: {len(tg_clf_dataset)}") # 预期: 1(id1) + 1(id2) + 1(id3) + 1(id4) + 0(id5被跳过) = 4
    if not tg_clf_dataset: print("未能生成样本。")

    for i in range(len(tg_clf_dataset)):
        sample = tg_clf_dataset[i] # __getitem__ returns the dict
        original_info_from_self_samples = tg_clf_dataset.samples[i] # Access internal for original_id

        print(f"\n--- TG 分类样本 {i} (源ID: {original_info_from_self_samples['original_id']}) ---")
        # 使用 self.id_to_targeted_group (即 ID_TO_GROUP) 来解码标签
        print(f"  Label ID: {sample['labels'].item()} ({tg_clf_dataset.id_to_targeted_group.get(sample['labels'].item())})")
        decoded_text = tg_clf_dataset.tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
        print(f"  Decoded Input Text (first 80 chars): {decoded_text[:80]}...")
        
    if os.path.exists(temp_json_path_tg): os.remove(temp_json_path_tg)
    print("\nTargetedGroupClassificationDataset (使用现有GROUP_TO_ID) 测试完成。")