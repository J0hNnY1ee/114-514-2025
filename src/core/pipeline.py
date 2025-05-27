# src/core/pipeline.py
import torch
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Union, Optional

try:
    from core.information_extractor import InformationExtractor
    from core.hate_classifier import HateClassifier
    from core.group_classifier import GroupClassifier
    from data_process.loader import HateOriDataset
except ImportError:
    try:
        from .information_extractor import InformationExtractor
        from .hate_classifier import HateClassifier
        from .group_classifier import GroupClassifier
        from ..data_process.loader import HateOriDataset
    except ImportError as e:
        print(f"Import Error in Pipeline: {e}. Ensure core components and HateOriDataset are accessible.")
        # Fallback placeholders
        class InformationExtractor: pass
        class HateClassifier: pass
        class GroupClassifier: pass
        class HateOriDataset:
            def __init__(self, json_file_path, is_pred=False): self.data = []
            def __len__(self): return 0
            def __getitem__(self, idx): return None, "", []


class SentimentPipeline:
    def __init__(self,
                 ie_model_checkpoint_path: str,
                 ie_base_model_name_or_path: str,
                 hate_clf_checkpoint_path: str,
                 hate_clf_base_model_name_or_path: str,
                 group_clf_checkpoint_path: str,
                 group_clf_base_model_name_or_path: str,
                 device: Union[str, torch.device] = "cpu",
                 max_seq_length_ie: int = 512,
                 max_seq_length_hate: int = 256,
                 max_seq_length_group: int = 256,
                 hate_input_template: str = "[CLS] {target} [SEP] {argument} [SEP] {content} [SEP]",
                 group_input_template: str = "[CLS] T: {target} A: {argument} IsHate: {hateful_label} C: {content} [SEP]",
                 output_intermediate_dir: str = "data/outputs/pipeline_intermediate_stages"
                 ):
        
        self.device = torch.device(device) if isinstance(device, str) else device
        self.output_intermediate_dir = output_intermediate_dir
        os.makedirs(self.output_intermediate_dir, exist_ok=True)

        print("Initializing Sentiment Pipeline...")
        print("Loading Information Extractor...")
        self.information_extractor = InformationExtractor(
            trained_model_checkpoint_path=ie_model_checkpoint_path,
            base_model_name_or_path=ie_base_model_name_or_path,
            device=self.device,
            max_seq_length=max_seq_length_ie
        )
        print("Information Extractor loaded.")
        print("Loading Hate Classifier...")
        self.hate_classifier = HateClassifier(
            trained_model_checkpoint_path=hate_clf_checkpoint_path,
            base_model_name_or_path=hate_clf_base_model_name_or_path,
            device=self.device,
            max_seq_length=max_seq_length_hate,
            input_template=hate_input_template
        )
        print("Hate Classifier loaded.")
        print("Loading Group Classifier...")
        self.group_classifier = GroupClassifier(
            trained_model_checkpoint_path=group_clf_checkpoint_path,
            base_model_name_or_path=group_clf_base_model_name_or_path,
            device=self.device,
            max_seq_length=max_seq_length_group,
            input_template=group_input_template
        )
        print("Group Classifier loaded.")
        print("Sentiment Pipeline initialized.")

    def _save_stage_results(self, data_list: List[Dict], stage_name: str, base_filename: str):
        """将某个阶段处理完整个数据集后的结果保存到一个文件 (作为JSON数组)。"""
        filepath = os.path.join(self.output_intermediate_dir, f"{base_filename}_{stage_name}.json")
        try:
            # 确保输出目录存在
            output_dir_for_file = os.path.dirname(filepath)
            if output_dir_for_file and not os.path.exists(output_dir_for_file):
                 os.makedirs(output_dir_for_file)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=4) # 直接将列表dump为JSON数组
            print(f"Results for stage '{stage_name}' (file '{base_filename}') saved to {filepath}")
        except Exception as e:
            print(f"Error saving results for stage '{stage_name}': {e}")

    def _format_quadruple_output_string(self, quadruples: List[Dict[str, str]]) -> str:
        if not quadruples: return "[END]"
        parts = []
        for quad in quadruples:
            t = quad.get("target", "NULL")
            a = quad.get("argument", "NULL")
            tg = quad.get("targeted_group_label", "NULL") 
            h = quad.get("hateful_label", "NULL")        
            parts.append(f"{t} | {a} | {tg} | {h}")
        return " [SEP] ".join(parts) + " [END]"

    def process_dataset(self, 
                        input_data_tuples: List[Tuple[Union[int, str], str]], 
                        pipeline_batch_size: int = 16, # 这个batch_size主要用于IE阶段分批
                        base_filename_for_saving: str = "dataset"
                       ) -> List[Dict]:
        if not input_data_tuples: return []
        num_items = len(input_data_tuples)
        # IE阶段的批处理
        num_ie_batches = (num_items + pipeline_batch_size - 1) // pipeline_batch_size
        print(f"Processing dataset '{base_filename_for_saving}' with {num_items} items.")
        print(f"IE stage will use {num_ie_batches} batches (batch size: {pipeline_batch_size}).")

        # --- 阶段1: 信息抽取 ---
        print(f"\n--- Stage 1: Information Extraction for '{base_filename_for_saving}' ---")
        stage1_all_outputs = [] 
        for i in range(num_ie_batches):
            batch_start = i * pipeline_batch_size
            batch_end = min((i + 1) * pipeline_batch_size, num_items)
            current_batch_tuples = input_data_tuples[batch_start:batch_end]
            batch_ids = [item[0] for item in current_batch_tuples]
            batch_contents = [item[1] for item in current_batch_tuples]
            print(f"  IE: Processing content batch {i+1}/{num_ie_batches}")
            try:
                extracted_ta_lists_for_batch = self.information_extractor.extract(batch_contents)
            except Exception as e:
                print(f"  Error in IE for content batch {i+1}: {e}")
                extracted_ta_lists_for_batch = [[{"target": "IE_ERROR", "argument": "IE_ERROR"}]] * len(batch_contents)
            for j, item_id in enumerate(batch_ids):
                stage1_all_outputs.append({
                    "id": item_id, "content": batch_contents[j],
                    "extracted_ta_pairs": extracted_ta_lists_for_batch[j] if j < len(extracted_ta_lists_for_batch) else [{"target":"IE_ALIGN_ERROR", "argument":"IE_ALIGN_ERROR"}]
                })
        self._save_stage_results(stage1_all_outputs, "1_information_extraction", base_filename_for_saving)

        # --- 阶段2: Hateful 分类 ---
        print(f"\n--- Stage 2: Hateful Classification for '{base_filename_for_saving}' ---")
        stage2_all_outputs_with_hate = [] 
        hate_clf_overall_inputs = []
        hate_clf_map_back = [] # (original_sample_idx_in_stage1, ta_pair_within_sample_idx)
        
        for sample_idx, s1_data in enumerate(stage1_all_outputs):
            # 预填充，即使没有TA对或IE出错，也保留原始id和content
            stage2_all_outputs_with_hate.append({
                "id": s1_data["id"], "content": s1_data["content"],
                "extracted_ta_pairs": s1_data["extracted_ta_pairs"],
                "hateful_predictions": [] # 初始化为空列表
            })
            if s1_data["extracted_ta_pairs"] and not (len(s1_data["extracted_ta_pairs"]) == 1 and s1_data["extracted_ta_pairs"][0].get("target","").endswith("_ERROR")):
                for ta_idx, ta_pair in enumerate(s1_data["extracted_ta_pairs"]):
                    hate_clf_overall_inputs.append({
                        "target": ta_pair.get("target", "NULL"), "argument": ta_pair.get("argument", "NULL"),
                        "content": s1_data["content"]
                    })
                    hate_clf_map_back.append((sample_idx, ta_idx))
                    stage2_all_outputs_with_hate[sample_idx]["hateful_predictions"].append({}) # 占位
            else:
                stage2_all_outputs_with_hate[sample_idx]["hateful_predictions"].append({"hateful_label": "SKIPPED_NO_TA", "hateful_score": 0.0})

        all_hateful_results_flat = []
        if hate_clf_overall_inputs:
            print(f"  Hate CLF: Processing {len(hate_clf_overall_inputs)} TA pairs in total...")
            try:
                # HateClassifier.classify_batch 应该能够处理大列表 (内部可能也分批)
                all_hateful_results_flat = self.hate_classifier.classify_batch(hate_clf_overall_inputs)
            except Exception as e:
                print(f"  Error during batch Hate Classification: {e}")
                all_hateful_results_flat = [{"hateful_label": "HATE_CLF_ERROR", "hateful_score": 0.0}] * len(hate_clf_overall_inputs)
        
        for i, hateful_res in enumerate(all_hateful_results_flat):
            sample_idx, ta_idx = hate_clf_map_back[i]
            stage2_all_outputs_with_hate[sample_idx]["hateful_predictions"][ta_idx] = hateful_res
        self._save_stage_results(stage2_all_outputs_with_hate, "2_hateful_classification", base_filename_for_saving)

        # --- 阶段3: Targeted Group 分类 ---
        print(f"\n--- Stage 3: Targeted Group Classification for '{base_filename_for_saving}' ---")
        final_pipeline_results = []
        group_clf_overall_inputs = []
        group_clf_map_back = [] # (final_result_sample_idx, quadruple_idx_in_sample)

        for sample_idx, s2_data in enumerate(stage2_all_outputs_with_hate):
            current_sample_final_output = {
                "id": s2_data["id"], "content": s2_data["content"],
                "structured_quadruples": [] 
            }
            if s2_data["extracted_ta_pairs"] and not (len(s2_data["extracted_ta_pairs"]) == 1 and s2_data["extracted_ta_pairs"][0].get("target","").endswith("_ERROR")):
                for ta_idx, ta_pair in enumerate(s2_data["extracted_ta_pairs"]):
                    hateful_pred = s2_data["hateful_predictions"][ta_idx]
                    group_clf_overall_inputs.append({
                        "target": ta_pair.get("target", "NULL"), "argument": ta_pair.get("argument", "NULL"),
                        "hateful_label": hateful_pred.get("hateful_label", "HATE_CLF_ERROR"),
                        "content": s2_data["content"]
                    })
                    group_clf_map_back.append((sample_idx, ta_idx)) # sample_idx refers to index in final_pipeline_results
                    current_sample_final_output["structured_quadruples"].append({}) # Placeholder
            final_pipeline_results.append(current_sample_final_output)

        all_group_results_flat = []
        if group_clf_overall_inputs:
            print(f"  Group CLF: Processing {len(group_clf_overall_inputs)} TA pairs for group classification...")
            try:
                all_group_results_flat = self.group_classifier.classify_batch(group_clf_overall_inputs)
            except Exception as e:
                print(f"  Error during batch Group Classification: {e}")
                all_group_results_flat = [{"targeted_group_label": "GROUP_CLF_ERROR", "targeted_group_score": 0.0}] * len(group_clf_overall_inputs)

        for i, group_res in enumerate(all_group_results_flat):
            sample_idx, ta_idx = group_clf_map_back[i]
            original_s1_data = stage1_all_outputs[sample_idx] # Get T/A from stage1
            original_s2_data = stage2_all_outputs_with_hate[sample_idx] # Get Hateful from stage2
            
            ta_pair_info = original_s1_data["extracted_ta_pairs"][ta_idx]
            hateful_info = original_s2_data["hateful_predictions"][ta_idx]

            quad = {
                "target": ta_pair_info.get("target", "NULL"),
                "argument": ta_pair_info.get("argument", "NULL"),
                "hateful_label": hateful_info.get("hateful_label", "HATE_CLF_ERROR"),
                "targeted_group_label": group_res.get("targeted_group_label", "GROUP_CLF_ERROR"),
            }
            final_pipeline_results[sample_idx]["structured_quadruples"][ta_idx] = quad
            
        for item in final_pipeline_results:
            item["output"] = self._format_quadruple_output_string(item.get("structured_quadruples", []))

        self._save_stage_results(final_pipeline_results, "3_final_output_formatted", base_filename_for_saving)
        
        print(f"Dataset '{base_filename_for_saving}' processing finished.")
        return final_pipeline_results

    def process_json_file(self, input_json_path: str, output_json_path: Optional[str] = None, 
                          pipeline_batch_size: int = 32) -> List[Dict]:
        print(f"Loading input data from '{input_json_path}' using HateOriDataset...")
        try:
            input_dataset = HateOriDataset(json_file_path=input_json_path, is_pred=True)
            if not input_dataset or len(input_dataset) == 0:
                print("输入JSON文件为空或HateOriDataset未能加载任何数据。")
                return []
        except Exception as e:
            print(f"使用HateOriDataset加载 '{input_json_path}' 时出错: {e}")
            import traceback; traceback.print_exc(); return []

        input_data_tuples = []
        for i in range(len(input_dataset)):
            item_id, content, _ = input_dataset[i]
            if item_id is None or content is None:
                print(f"警告: HateOriDataset 返回了无效的 id 或 content (索引 {i})。跳过。")
                continue
            input_data_tuples.append((item_id, content))
        
        if not input_data_tuples:
            print("未能从HateOriDataset中提取有效的 (id, content) 对。")
            return []

        base_input_filename = os.path.splitext(os.path.basename(input_json_path))[0]
        
        all_processed_results = self.process_dataset(
            input_data_tuples, 
            pipeline_batch_size=pipeline_batch_size,
            base_filename_for_saving=base_input_filename
        )

        if output_json_path:
            try:
                output_dir = os.path.dirname(output_json_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # 保存为单个JSON文件，其中包含一个JSON数组
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(all_processed_results, f, ensure_ascii=False, indent=4)
                print(f"流水线最终预测结果已保存到: {output_json_path} (作为JSON数组)")
            except Exception as e:
                print(f"保存最终输出JSON文件时出错: {e}")
        
        return all_processed_results

# --- 示例用法 ---
if __name__ == "__main__":
    IE_CHECKPOINT_PATH = "./models/outputs/ie_multilabel/best_model_on_eval_loss" 
    IE_BASE_MODEL = "models/chinese-roberta-wwm-ext-large"
    HATE_CLF_CHECKPOINT_PATH = "./models/outputs/hate_clf/best_model_on_eval_loss"
    HATE_CLF_BASE_MODEL = "models/chinese-roberta-wwm-ext-large"
    GROUP_CLF_CHECKPOINT_PATH = "./models/outputs/group_clf/best_model_on_eval_loss"
    GROUP_CLF_BASE_MODEL = "models/chinese-roberta-wwm-ext-large"

    INPUT_TEST_JSON_PATH = "data/segment/test.json" 
    PIPELINE_INTERMEDIATE_STAGES_DIR = "data/outputs/pipeline_stages_output"
    FINAL_PREDICTIONS_OUTPUT_PATH = "data/outputs/pipeline_final_predictions.json"

    print("检查模型和输入文件路径...")
    required_paths_info = {
        "IE Checkpoint Dir": (IE_CHECKPOINT_PATH, True, "pytorch_model.bin"),
        "IE Base Model Dir": (IE_BASE_MODEL, True, "config.json"),
        "Hate CLF Checkpoint Dir": (HATE_CLF_CHECKPOINT_PATH, True, "pytorch_model.bin"),
        "Hate CLF Base Model Dir": (HATE_CLF_BASE_MODEL, True, "config.json"),
        "Group CLF Checkpoint Dir": (GROUP_CLF_CHECKPOINT_PATH, True, "pytorch_model.bin"),
        "Group CLF Base Model Dir": (GROUP_CLF_BASE_MODEL, True, "config.json"),
        "Input Test JSON File": (INPUT_TEST_JSON_PATH, False, None)
    }
    all_paths_valid = True
    for name, (path, is_dir, expected_file) in required_paths_info.items():
        path_exists = os.path.exists(path)
        content_ok = True
        if path_exists and is_dir and expected_file:
            if not os.path.exists(os.path.join(path, expected_file)):
                content_ok = False; print(f"错误: {name} 目录 '{path}' 缺少 '{expected_file}'。")
        if not path_exists or not content_ok:
            if not path_exists : print(f"错误: {name} 路径 '{path}' 不存在。")
            all_paths_valid = False
            
    if not all_paths_valid:
        print("一个或多个必要路径无效或不完整，流水线无法运行。")
    else:
        print("所有路径检查通过。")
        try:
            pipeline = SentimentPipeline(
                ie_model_checkpoint_path=IE_CHECKPOINT_PATH,
                ie_base_model_name_or_path=IE_BASE_MODEL,
                hate_clf_checkpoint_path=HATE_CLF_CHECKPOINT_PATH,
                hate_clf_base_model_name_or_path=HATE_CLF_BASE_MODEL,
                group_clf_checkpoint_path=GROUP_CLF_CHECKPOINT_PATH,
                group_clf_base_model_name_or_path=GROUP_CLF_BASE_MODEL,
                device="cuda" if torch.cuda.is_available() else "cpu",
                output_intermediate_dir=PIPELINE_INTERMEDIATE_STAGES_DIR
            )

            print(f"\n--- 开始处理测试文件: {INPUT_TEST_JSON_PATH} ---")
            final_results_list = pipeline.process_json_file(
                input_json_path=INPUT_TEST_JSON_PATH, 
                output_json_path=FINAL_PREDICTIONS_OUTPUT_PATH,
                pipeline_batch_size=8 
            )
            
            if final_results_list:
                print(f"\n--- 处理完成 ---")
                print(f"最终预测结果已保存到: {FINAL_PREDICTIONS_OUTPUT_PATH}")
                print(f"共处理了 {len(final_results_list)} 条原始数据项。")
                print(f"每个阶段处理完整个数据集后的中间文件保存在: {pipeline.output_intermediate_dir} 目录中。")
            else:
                print("未能处理任何数据或生成任何结果。")

        except Exception as e:
            print(f"运行Pipeline时发生主错误: {e}")
            import traceback
            traceback.print_exc()