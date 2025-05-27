# data_process/clf_dataset.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional

# 假设可以从同级或项目的其他模块导入
try:
    from data_process.loader import HateOriDataset # 从 data_process.loader 导入
    from data_process.preprocessor import HATEFUL_TO_ID # 从 data_process.preprocessor 导入
except ImportError:
    try:
        from loader import HateOriDataset
        from preprocessor import HATEFUL_TO_ID
    except ImportError:
        print("Error: Could not import HateOriDataset or HATEFUL_TO_ID. Check paths.")
        # 占位符，以便类可以定义
        class HateOriDataset: pass
        HATEFUL_TO_ID = {"hate": 0, "non-hate": 1}


class HateClassificationDataset(Dataset):
    def __init__(self,
                 original_json_path: str,
                 tokenizer_name_or_path: str,
                 max_seq_length: int = 512,
                 # 控制如何组合输入文本的模板
                 input_template: str = "[CLS] {target} [SEP] {argument} [SEP] {content} [SEP]"
                 # 或者 "[CLS] {target} [SEP] {argument} [SEP]"
                 # 或者 "[CLS] {content} [SEP] {target} [SEP] {argument} [SEP]"
                 ):
        """
        为 Hateful 分类任务准备数据的数据集类。

        参数:
            original_json_path (str): HateOriDataset 使用的原始JSON文件路径。
            tokenizer_name_or_path (str): 预训练模型的分词器名称或本地路径。
            max_seq_length (int): Token序列的最大长度。
            input_template (str): 用于构建模型输入的字符串模板。
                                  必须包含 {target}, {argument}, {content} 占位符。
                                  Tokenizer 会自动处理 [CLS] 和 [SEP] (如果模板中未显式提供但模型需要)。
                                  但为了明确，模板中包含它们更好。
        """
        self.original_dataset = HateOriDataset(original_json_path, is_pred=False) # 分类训练需要标签
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.max_seq_length = max_seq_length
        self.input_template = input_template
        self.hateful_to_id = HATEFUL_TO_ID

        self.samples = self._create_classification_samples()

    def _create_classification_samples(self) -> List[Dict]:
        """
        从原始数据中提取 Target-Argument 对，并为每个对创建分类样本。
        """
        classification_samples = []
        skipped_samples_no_ta = 0

        for i in range(len(self.original_dataset)):
            original_id, original_content, ta_quadruples = self.original_dataset[i]

            if not ta_quadruples: # 如果原始样本解析后没有四元组
                skipped_samples_no_ta +=1
                continue

            for quad_info in ta_quadruples:
                target_text = quad_info["target"]
                argument_text = quad_info["argument"]
                # quad_info["target_group_ids"] # 这个任务不需要
                hateful_label_id = quad_info["hateful_id"] # 这是已经转换好的ID

                # 跳过 "NULL" 或空的 target/argument，因为它们可能不适合直接作为分类输入
                # 或者你可以选择将它们替换为特殊标记，但这取决于你的策略
                if target_text == "NULL" or not target_text.strip():
                    target_text_for_input = "[EMPTY_TARGET]" # 或其他标记
                else:
                    target_text_for_input = target_text

                if argument_text == "NULL" or not argument_text.strip():
                    argument_text_for_input = "[EMPTY_ARGUMENT]"
                else:
                    argument_text_for_input = argument_text
                
                # 构建输入文本
                # 注意：Hugging Face Tokenizer 会自动添加 [CLS] 和 [SEP] (如果 add_special_tokens=True 默认)
                # 如果模板中已经有了，确保 tokenizer 设置不会重复添加。
                # 通常，如果模板以 [CLS] 开头，以 [SEP] 结尾，tokenizer 会识别它们。
                # 或者，一种更安全的方式是让tokenizer负责添加特殊token：
                # text_a = f"{target_text_for_input} [SEP] {argument_text_for_input}"
                # text_b = original_content
                # tokenized_input = self.tokenizer(text_a, text_b, ...)
                # 但使用模板更灵活。

                # 确保模板中的占位符被正确替换
                try:
                    input_text = self.input_template.format(
                        target=target_text_for_input,
                        argument=argument_text_for_input,
                        content=original_content
                    )
                except KeyError as e:
                    raise ValueError(f"输入模板 '{self.input_template}' 中的占位符不正确: {e}. "
                                     "请确保模板包含 {{target}}, {{argument}}, 和 {{content}}。")


                # Tokenize
                # add_special_tokens=True: 如果模板中没有 [CLS]/[SEP]，它会添加。
                # 如果模板中有了，它通常能正确处理（不会重复添加）。
                # 为了确保，可以设置 add_special_tokens=False，然后在模板中完全控制。
                # 但多数情况下，add_special_tokens=True 配合模板中的特殊token是能工作的。
                tokenized_output = self.tokenizer(
                    input_text,
                    max_length=self.max_seq_length,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                    return_token_type_ids=True # 对于BERT类型模型
                    # add_special_tokens=True (默认)
                )

                classification_samples.append({
                    "input_ids": torch.tensor(tokenized_output["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(tokenized_output["attention_mask"], dtype=torch.long),
                    "token_type_ids": torch.tensor(tokenized_output.get("token_type_ids", []), dtype=torch.long), # RoBERTa可能没有
                    "label": torch.tensor(hateful_label_id, dtype=torch.long), # 单个标签
                    "original_id": original_id, # 保留原始ID用于调试或追溯
                    "original_target": target_text,
                    "original_argument": argument_text,
                })
        
        if skipped_samples_no_ta > 0:
            print(f"警告: 跳过了 {skipped_samples_no_ta} 个原始样本，因为它们没有解析出 (Target, Argument) 对。")
        
        return classification_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        # 返回模型需要的内容
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "token_type_ids": sample["token_type_ids"],
            "labels": sample["label"] # 模型通常期望分类标签键为 "labels"
        }

# --- 示例用法 ---
if __name__ == "__main__":
    import json # 确保导入
    import os   # 确保导入
    from preprocessor import HATEFUL_ID_TO_LABEL # 用于可读性输出

    # 创建一个临时的原始数据JSON文件用于测试
    temp_original_data_clf = [
        {
            "id": 1,
            "content": "张三是个大好人，李四是个坏蛋。",
            "output": "张三 | 大好人 | non-hate | non-hate [SEP] 李四 | 坏蛋 | others | hate [END]"
        },
        {
            "id": 2,
            "content": "这部电影真精彩！主角演技爆表。",
            "output": "这部电影 | 真精彩 | non-hate | non-hate [SEP] 主角 | 演技爆表 | non-hate | non-hate [END]"
        },
        {
            "id": 3, # 没有TA对的样本
            "content": "天气不错。",
            "output": ""
        },
        {
            "id": 4,
            "content": "NULL用户是坏人。",
            "output": "NULL | 坏人 | Racism | hate [END]"
        }
    ]
    temp_json_path_clf = "temp_clf_test_data.json"
    with open(temp_json_path_clf, 'w', encoding='utf-8') as f:
        json.dump(temp_original_data_clf, f, ensure_ascii=False, indent=2)

    tokenizer_path_clf = "models/chinese-roberta-wwm-ext-large" # 你的tokenizer路径
    # tokenizer_path_clf = "hfl/chinese-roberta-wwm-ext" # 测试用

    print(f"使用Tokenizer: {tokenizer_path_clf}")
    print(f"Hateful 标签映射: {HATEFUL_TO_ID}")
    print("-" * 30)

    # 测试不同的输入模板
    templates_to_test = [
        "[CLS] {target} [SEP] {argument} [SEP] {content} [SEP]",
        "[CLS] Target: {target} Argument: {argument} Context: {content} [SEP]",
        # 一个不使用 content 的例子，只用 target 和 argument
        "[CLS] {target} [SEP] {argument} [SEP]"
    ]

    for tmpl_idx, template in enumerate(templates_to_test):
        print(f"\n--- 测试模板 {tmpl_idx+1}: '{template}' ---")
        try:
            clf_dataset = HateClassificationDataset(
                original_json_path=temp_json_path_clf,
                tokenizer_name_or_path=tokenizer_path_clf,
                max_seq_length=128,
                input_template=template
            )
        except Exception as e:
            print(f"初始化 HateClassificationDataset 失败 (模板: '{template}'): {e}")
            continue

        print(f"HateClassificationDataset 大小 (模板 '{template}'): {len(clf_dataset)}")
        # 预期大小: 样本1有2对，样本2有2对，样本3有0对(跳过)，样本4有1对 = 2+2+0+1 = 5个分类样本

        if not clf_dataset:
            print("未能生成任何处理后的分类样本。")
            continue

        for i in range(min(len(clf_dataset), 2)): # 只打印前2个样本
            sample = clf_dataset[i] # 这是 __getitem__ 返回的字典
            original_info = clf_dataset.samples[i] # 这是 _create_classification_samples 内部存储的

            print(f"\n  --- 分类样本 {i} (源ID: {original_info['original_id']}) ---")
            # print(f"  Input IDs: {sample['input_ids']}")
            # print(f"  Attention Mask: {sample['attention_mask']}")
            # if sample['token_type_ids'] is not None and len(sample['token_type_ids']) > 0:
            #     print(f"  Token Type IDs: {sample['token_type_ids']}")
            print(f"  Label ID: {sample['labels']} ({HATEFUL_ID_TO_LABEL.get(sample['labels'].item())})")
            
            # 解码输入文本以验证模板和内容
            decoded_text = clf_dataset.tokenizer.decode(sample['input_ids'], skip_special_tokens=False) # 显示特殊token
            print(f"  Decoded Input Text: {decoded_text}")
            print(f"  Original Target: '{original_info['original_target']}', Original Argument: '{original_info['original_argument']}'")

    # 清理临时文件
    if os.path.exists(temp_json_path_clf):
        os.remove(temp_json_path_clf)
    print(f"\n临时测试文件 {temp_json_path_clf} 已删除。")
    print("\nHateClassificationDataset 测试完成。")