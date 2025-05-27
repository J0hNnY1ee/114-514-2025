# prepare_eval_data.py
import json
import os # 确保导入os
from typing import List, Dict, Union, Optional

try:

    from data_process.loader import HateOriDataset # 如果 prepare_eval_data.py 在 src/ 外部，或者调整路径
    from data_process.preprocessor import ID_TO_GROUP, ID_TO_HATEFUL
except ImportError:
    try: # 尝试相对导入
        from ..data_process.loader import HateOriDataset
        from ..data_process.preprocessor import ID_TO_GROUP, ID_TO_HATEFUL
    except ImportError:
        print("Warning: Could not import HateOriDataset or preprocessor mappers for prepare_eval_data.py.")
        class HateOriDataset:
            def __init__(self, json_file_path, is_pred=False): self.data = []
            def __len__(self): return 0
            def __getitem__(self, idx): return None, "", []
        ID_TO_GROUP = {}
        ID_TO_HATEFUL = {}


def load_predictions(filepath: str) -> Dict[Union[int, str], List[Dict[str, str]]]:
    """
    加载预测文件 (标准的JSON数组文件, 包含id和structured_quadruples)。
    """
    predictions_map = {}
    if not os.path.exists(filepath):
        print(f"错误: 预测文件未找到: {filepath}")
        return predictions_map
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            # 将整个文件解析为一个JSON列表
            data_list = json.load(f) 
            if not isinstance(data_list, list):
                print(f"错误: 预测文件 '{filepath}' 的顶层不是一个JSON列表。")
                return predictions_map

            for record_num, record in enumerate(data_list):
                if not isinstance(record, dict):
                    print(f"警告: 预测文件 '{filepath}' 中的第 {record_num+1} 项不是一个字典。跳过。")
                    continue
                
                record_id = record.get("id")
                if record_id is None:
                    print(f"警告: 预测文件 '{filepath}' 第 {record_num+1} 项缺少 'id'。跳过。")
                    continue
                
                raw_quads = record.get("structured_quadruples", [])
                processed_quads = []
                if isinstance(raw_quads, list):
                    for rq_dict in raw_quads:
                        if isinstance(rq_dict, dict): # 确保rq_dict也是字典
                            processed_quads.append({
                                "target": rq_dict.get("target", "NULL"),
                                "argument": rq_dict.get("argument", "NULL"),
                                "targeted_group": rq_dict.get("targeted_group_label", "NULL"),
                                "hateful": rq_dict.get("hateful_label", "NULL")
                            })
                        else:
                             print(f"警告: ID {record_id} 的 structured_quadruples 中的元素不是字典: {rq_dict}")
                elif raw_quads is not None:
                    print(f"警告: ID {record_id} 的 'structured_quadruples' 不是列表，而是 {type(raw_quads)}。")

                predictions_map[record_id] = processed_quads
    except json.JSONDecodeError as e:
        print(f"错误: 解析预测JSON文件 '{filepath}' 失败: {e}")
    except Exception as e:
        print(f"读取预测文件 '{filepath}' 时发生错误: {e}")
    return predictions_map

# load_gold_standards 函数保持不变，因为它使用 HateOriDataset，
# 而 HateOriDataset 应该能处理标准的JSON数组文件或JSON Lines文件（取决于其_load_data实现）
# 我们假设 GOLD_STANDARD_FILE (如 test.json) 是 HateOriDataset 可以处理的格式。
def load_gold_standards(filepath: str) -> Dict[Union[int, str], List[Dict[str, str]]]:
    """使用 HateOriDataset 加载标准答案文件并解析为四元组列表。"""
    gold_map = {}
    if not os.path.exists(filepath):
        print(f"错误: 标准答案文件未找到: {filepath}")
        return gold_map
    try:
        gold_dataset = HateOriDataset(json_file_path=filepath, is_pred=False)
        if not gold_dataset or len(gold_dataset) == 0:
            print(f"HateOriDataset 未能从 '{filepath}' 加载任何标准答案数据。")
            return gold_map

        for i in range(len(gold_dataset)):
            item_id, _, parsed_quad_infos = gold_dataset[i]
            if item_id is None:
                print(f"警告: 标准答案数据索引 {i} 缺少 'id'。跳过。")
                continue

            quads_for_eval = []
            if parsed_quad_infos:
                for quad_info in parsed_quad_infos:
                    tg_label_str = "NULL"
                    if quad_info.get("target_group_ids"): 
                        first_tg_id = quad_info["target_group_ids"][0]
                        tg_label_str = ID_TO_GROUP.get(first_tg_id, "UNKNOWN_TG_ID")
                    
                    h_label_str = ID_TO_HATEFUL.get(quad_info["hateful_id"], "UNKNOWN_H_ID")

                    quads_for_eval.append({
                        "target": quad_info.get("target", "NULL"),
                        "argument": quad_info.get("argument", "NULL"),
                        "targeted_group": tg_label_str,
                        "hateful": h_label_str
                    })
            gold_map[item_id] = quads_for_eval
    except Exception as e:
        print(f"使用HateOriDataset加载标准答案文件 '{filepath}' 时发生错误: {e}")
        import traceback; traceback.print_exc()
    return gold_map


def create_paired_eval_file(predictions_filepath: str, 
                            gold_standard_filepath: str, 
                            output_paired_filepath: str):
    """
    加载预测和标准答案，按ID配对，并保存到一个新的JSON Lines文件。
    """
    print("开始加载预测数据...")
    predicted_data_map = load_predictions(predictions_filepath)
    print(f"加载了 {len(predicted_data_map)} 条预测数据。")

    print("开始加载标准答案数据...")
    gold_data_map = load_gold_standards(gold_standard_filepath)
    print(f"加载了 {len(gold_data_map)} 条标准答案数据。")

    if not predicted_data_map or not gold_data_map:
        print("预测或标准答案数据为空，无法生成配对文件。")
        return

    paired_data = []
    pred_ids = set(predicted_data_map.keys())
    gold_ids = set(gold_data_map.keys())

    common_ids = pred_ids.intersection(gold_ids)
    ids_only_in_pred = pred_ids - gold_ids
    ids_only_in_gold = gold_ids - pred_ids

    if ids_only_in_pred:
        print(f"警告: {len(ids_only_in_pred)} 个ID仅存在于预测文件中: {list(ids_only_in_pred)[:5]}...")
    if ids_only_in_gold:
        print(f"警告: {len(ids_only_in_gold)} 个ID仅存在于标准答案文件中: {list(ids_only_in_gold)[:5]}...")
    
    print(f"找到 {len(common_ids)} 个共同ID用于配对。")

    # 获取 content: 假设标准答案文件 (如 test.json) 是原始格式，包含 content
    # HateOriDataset 加载时会保留原始数据在 self.data 中
    # 或者，如果预测文件也包含content，可以直接从那里取
    
    # 重新读取gold_standard_filepath以获取content，确保与gold_quads对齐
    original_content_map = {}
    if os.path.exists(gold_standard_filepath):
        try:
            with open(gold_standard_filepath, "r", encoding="utf-8") as f:
                # 假设 gold_standard_filepath 是一个JSON数组文件
                gold_data_list_for_content = json.load(f)
                if isinstance(gold_data_list_for_content, list):
                    for record in gold_data_list_for_content:
                        if isinstance(record, dict) and "id" in record and "content" in record:
                            original_content_map[record["id"]] = record["content"]
                else:
                    print(f"警告: 标准答案文件 '{gold_standard_filepath}' 顶层不是列表，无法提取content。")
        except json.JSONDecodeError:
             print(f"警告: 解析标准答案文件 '{gold_standard_filepath}' 以获取content失败。尝试逐行。")
             try:
                 with open(gold_standard_filepath, "r", encoding="utf-8") as f_line:
                    for line in f_line:
                        try:
                            record = json.loads(line.strip())
                            if "id" in record and "content" in record:
                                original_content_map[record["id"]] = record["content"]
                        except: pass # 忽略无法解析的行
             except: pass # 忽略文件读取错误
        except Exception as e:
            print(f"读取原始content时出错: {e}")
    else:
        print(f"警告: 标准答案文件 '{gold_standard_filepath}' 未找到，无法提取content。")


    for record_id in sorted(list(common_ids)):
        paired_data.append({
            "id": record_id,
            "content": original_content_map.get(record_id, "CONTENT_NOT_FOUND_IN_GOLD_FILE"),
            "predicted_quads": predicted_data_map[record_id],
            "gold_quads": gold_data_map[record_id]
        })
    
    try:
        output_dir = os.path.dirname(output_paired_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_paired_filepath, "w", encoding="utf-8") as f:
            for item in paired_data: # 保存为JSON Lines格式
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"配对好的评估数据已保存到: {output_paired_filepath} (JSON Lines格式)")
    except Exception as e:
        print(f"保存配对评估文件时出错: {e}")

if __name__ == "__main__":
    import os 
    
    PREDICTIONS_FILE = "data/outputs/pipeline_final_predictions.json" 
    GOLD_STANDARD_FILE = "data/segment/test.json" 
    PAIRED_EVAL_OUTPUT_FILE = "data/outputs/evaluation_paired_data.jsonl" 

    if not os.path.exists(PREDICTIONS_FILE):
        print(f"错误: 预测文件 '{PREDICTIONS_FILE}' 不存在。请先运行流水线。")
    elif not os.path.exists(GOLD_STANDARD_FILE):
        print(f"错误: 标准答案文件 '{GOLD_STANDARD_FILE}' 不存在。")
    else:
        create_paired_eval_file(PREDICTIONS_FILE, GOLD_STANDARD_FILE, PAIRED_EVAL_OUTPUT_FILE)
        print("\n现在你可以使用 'evaluation_paired_data.jsonl' 文件进行评估了。")