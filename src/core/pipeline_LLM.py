# pipeline_llm_1.py (可以放在 core 目录下)
import torch
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Union, Optional
import traceback
# 假设其他分类器和HateOriDataset的导入路径不变
try:
    # from core.information_extractor import InformationExtractor # 旧的IE
    from core.information_extractor_llm import InformationExtractor_LLM # 
    from core.hate_classifier import HateClassifier
    from core.group_classifier import GroupClassifier
    from data_process.loader import HateOriDataset
except ImportError:
    try:
        # from .information_extractor import InformationExtractor
        from .information_extractor_llm import InformationExtractor_LLM
        from .hate_classifier import HateClassifier
        from .group_classifier import GroupClassifier
        from ..data_process.loader import HateOriDataset
    except ImportError as e:
        print(f"Import Error in Pipeline_LLM_1: {e}. Ensure core components and HateOriDataset are accessible.")
        # Fallback placeholders
        class InformationExtractor_LLM: pass
        class HateClassifier: pass
        class GroupClassifier: pass
        class HateOriDataset:
            def __init__(self, json_file_path, is_pred=False): self.data = []
            def __len__(self): return 0
            def __getitem__(self, idx): return None, "", []


class SentimentPipeline_LLM:
    def __init__(self,
                 # IE 参数变为 LLM 模型的参数
                 ie_llm_base_model_name_or_path: str,      # 例如 "models/Qwen3-8B"
                 ie_llm_lora_checkpoint_path: str,         # 例如 "models/outputs/Qwen3-LoRA-TargetArgument/best_lora_adapter"
                 hate_clf_checkpoint_path: str,
                 hate_clf_base_model_name_or_path: str,
                 group_clf_checkpoint_path: str,
                 group_clf_base_model_name_or_path: str,
                 device: Union[str, torch.device] = "cpu",
                 max_seq_length_ie_llm: int = 256, # LLM生成IE结果的最大token数
                 qwen_think_end_token_id_fallback: int = 151647, # Qwen </think> 后备ID
                 max_seq_length_hate: int = 256,
                 max_seq_length_group: int = 256,
                 hate_input_template: str = "[CLS] {target} [SEP] {argument} [SEP] {content} [SEP]",
                 group_input_template: str = "[CLS] T: {target} A: {argument} IsHate: {hateful_label} C: {content} [SEP]",
                 output_intermediate_dir: str = "data/outputs/pipeline_llm1_intermediate_stages" # 修改目录名以区分
                 ):

        self.device = torch.device(device) if isinstance(device, str) else device
        self.output_intermediate_dir = output_intermediate_dir
        os.makedirs(self.output_intermediate_dir, exist_ok=True)

        print("Initializing Sentiment Pipeline with LLM for IE (Pipeline_LLM_1)...")
        print("Loading Information Extractor (LLM based)...")
        self.information_extractor = InformationExtractor_LLM( # 使用新的IE类
            base_model_name_or_path=ie_llm_base_model_name_or_path,
            trained_model_checkpoint_path=ie_llm_lora_checkpoint_path,
            device=self.device,
            max_seq_length=max_seq_length_ie_llm,
            qwen_think_end_token_id_fallback=qwen_think_end_token_id_fallback
        )
        print("Information Extractor (LLM based) loaded.")

        # 后续分类器的加载保持不变
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
        print("Sentiment Pipeline (LLM_1 for IE) initialized.")

    # _save_stage_results 和 _format_quadruple_output_string 方法保持不变
    def _save_stage_results(self, data_list: List[Dict], stage_name: str, base_filename: str):
        filepath = os.path.join(self.output_intermediate_dir, f"{base_filename}_{stage_name}.json")
        try:
            output_dir_for_file = os.path.dirname(filepath)
            if output_dir_for_file and not os.path.exists(output_dir_for_file):
                 os.makedirs(output_dir_for_file)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=4)
            print(f"Results for stage '{stage_name}' (file '{base_filename}') saved to {filepath}")
        except Exception as e:
            print(f"Error saving results for stage '{stage_name}': {e}")

    def _format_quadruple_output_string(self, quadruples: List[Dict[str, str]]) -> str:
        if not quadruples: return "[END]" # 确保没有TA对时也返回[END]
        parts = []
        for quad in quadruples:
            t = quad.get("target", "NULL")
            a = quad.get("argument", "NULL")
            # 这两个字段在初始的IE阶段还没有，会在后续阶段填充
            tg = quad.get("targeted_group_label", "NULL_TG_PENDING")
            h = quad.get("hateful_label", "NULL_H_PENDING")
            parts.append(f"{t} | {a} | {tg} | {h}") # 格式是T|A|TG|H
        return " [SEP] ".join(parts) + " [END]"


    # process_dataset 方法只需要修改IE部分的调用，其他逻辑保持不变
    def process_dataset(self,
                        input_data_tuples: List[Tuple[Union[int, str], str]],
                        pipeline_batch_size: int = 16, # IE_LLM 也可以分批处理输入文本
                        base_filename_for_saving: str = "dataset"
                       ) -> List[Dict]:
        if not input_data_tuples: return []
        num_items = len(input_data_tuples)
        num_ie_batches = (num_items + pipeline_batch_size - 1) // pipeline_batch_size
        print(f"Processing dataset '{base_filename_for_saving}' with {num_items} items using LLM_1 pipeline.")
        print(f"IE_LLM stage will use {num_ie_batches} batches (batch size: {pipeline_batch_size}).")

        # --- 阶段1: 信息抽取 (使用 InformationExtractor_LLM) ---
        print(f"\n--- Stage 1: Information Extraction (LLM) for '{base_filename_for_saving}' ---")
        stage1_all_outputs = []
        for i in range(num_ie_batches):
            batch_start = i * pipeline_batch_size
            batch_end = min((i + 1) * pipeline_batch_size, num_items)
            current_batch_tuples = input_data_tuples[batch_start:batch_end]
            batch_ids = [item[0] for item in current_batch_tuples]
            batch_contents = [item[1] for item in current_batch_tuples]

            print(f"  IE_LLM: Processing content batch {i+1}/{num_ie_batches} (size: {len(batch_contents)})")
            try:
                # InformationExtractor_LLM.extract 接收一个内容列表，返回TA对的列表的列表
                extracted_ta_lists_for_batch = self.information_extractor.extract(batch_contents)
            except Exception as e:
                print(f"  Error in IE_LLM for content batch {i+1}: {e}")
                traceback.print_exc()
                # 为批次中的每个项目填充错误信息
                extracted_ta_lists_for_batch = [[{"target": "IE_LLM_BATCH_ERROR", "argument": "IE_LLM_BATCH_ERROR"}]] * len(batch_contents)

            for j, item_id in enumerate(batch_ids):
                # 确保 extracted_ta_lists_for_batch 的长度与 batch_ids 匹配
                if j < len(extracted_ta_lists_for_batch):
                    current_item_ta_pairs = extracted_ta_lists_for_batch[j]
                else: # 如果IE返回结果少于输入（理论上不应发生，但作为保护）
                    current_item_ta_pairs = [{"target":"IE_LLM_ALIGN_ERROR", "argument":"IE_LLM_ALIGN_ERROR"}]

                stage1_all_outputs.append({
                    "id": item_id,
                    "content": batch_contents[j],
                    "extracted_ta_pairs": current_item_ta_pairs
                })
        self._save_stage_results(stage1_all_outputs, "1_information_extraction_llm", base_filename_for_saving)

        # --- 阶段2: Hateful 分类 (代码保持不变) ---
        print(f"\n--- Stage 2: Hateful Classification for '{base_filename_for_saving}' ---")
        stage2_all_outputs_with_hate = []
        hate_clf_overall_inputs = []
        hate_clf_map_back = []

        for sample_idx, s1_data in enumerate(stage1_all_outputs):
            stage2_item = {
                "id": s1_data["id"], "content": s1_data["content"],
                "extracted_ta_pairs": s1_data["extracted_ta_pairs"], # 从stage1继承
                "hateful_predictions": []
            }
            # 检查 extracted_ta_pairs 是否有效且不包含错误标记
            # (注意：IE_LLM 的错误标记可能不同，需要根据 _parse_ie_string_to_ta_pairs 的实现调整)
            is_valid_ta = False
            if s1_data["extracted_ta_pairs"]:
                first_ta = s1_data["extracted_ta_pairs"][0]
                if not (first_ta.get("target","").startswith("IE_") or \
                        first_ta.get("target","").startswith("PARSE_") or \
                        first_ta.get("argument","").startswith("IE_") or \
                        first_ta.get("argument","").startswith("PARSE_")
                        ):
                    is_valid_ta = True

            if is_valid_ta:
                for ta_idx, ta_pair in enumerate(s1_data["extracted_ta_pairs"]):
                    hate_clf_overall_inputs.append({
                        "target": ta_pair.get("target", "NULL"),
                        "argument": ta_pair.get("argument", "NULL"),
                        "content": s1_data["content"]
                    })
                    hate_clf_map_back.append((sample_idx, ta_idx))
                    stage2_item["hateful_predictions"].append({}) # Placeholder
            else: # 如果IE出错或没有有效的TA对
                stage2_item["hateful_predictions"].append({"hateful_label": "SKIPPED_INVALID_IE", "hateful_score": 0.0})
            stage2_all_outputs_with_hate.append(stage2_item)


        all_hateful_results_flat = []
        if hate_clf_overall_inputs:
            print(f"  Hate CLF: Processing {len(hate_clf_overall_inputs)} TA pairs in total...")
            try:
                all_hateful_results_flat = self.hate_classifier.classify_batch(hate_clf_overall_inputs)
            except Exception as e:
                print(f"  Error during batch Hate Classification: {e}")
                traceback.print_exc()
                all_hateful_results_flat = [{"hateful_label": "HATE_CLF_ERROR", "hateful_score": 0.0}] * len(hate_clf_overall_inputs)

        for i, hateful_res in enumerate(all_hateful_results_flat):
            sample_idx, ta_idx = hate_clf_map_back[i]
            # 确保 hateful_predictions 列表有足够的空间
            if ta_idx < len(stage2_all_outputs_with_hate[sample_idx]["hateful_predictions"]):
                stage2_all_outputs_with_hate[sample_idx]["hateful_predictions"][ta_idx] = hateful_res
            else: # 如果映射关系出错（理论上不应发生）
                print(f"  Error: ta_idx {ta_idx} out of bounds for hateful_predictions at sample_idx {sample_idx}")
                # 可以选择追加或记录错误
                stage2_all_outputs_with_hate[sample_idx]["hateful_predictions"].append(hateful_res)


        self._save_stage_results(stage2_all_outputs_with_hate, "2_hateful_classification", base_filename_for_saving)


        # --- 阶段3: Targeted Group 分类 (代码保持不变) ---
        print(f"\n--- Stage 3: Targeted Group Classification for '{base_filename_for_saving}' ---")
        final_pipeline_results = []
        group_clf_overall_inputs = []
        group_clf_map_back = []

        for sample_idx, s2_data in enumerate(stage2_all_outputs_with_hate):
            current_sample_final_output = {
                "id": s2_data["id"], "content": s2_data["content"],
                "structured_quadruples": []
            }
            # 再次检查 s2_data["extracted_ta_pairs"] 和 s2_data["hateful_predictions"] 的有效性
            is_valid_ta_for_group = False
            if s2_data["extracted_ta_pairs"] and s2_data["hateful_predictions"]:
                first_ta = s2_data["extracted_ta_pairs"][0]
                first_hate = s2_data["hateful_predictions"][0]
                if not (first_ta.get("target","").startswith("IE_") or \
                        first_ta.get("target","").startswith("PARSE_") or \
                        first_hate.get("hateful_label","").startswith("SKIPPED_") or \
                        first_hate.get("hateful_label","").startswith("HATE_CLF_ERROR")
                        ):
                    is_valid_ta_for_group = True

            if is_valid_ta_for_group:
                 # 确保 extracted_ta_pairs 和 hateful_predictions 长度一致
                num_pairs = min(len(s2_data["extracted_ta_pairs"]), len(s2_data["hateful_predictions"]))
                for ta_idx in range(num_pairs):
                    ta_pair = s2_data["extracted_ta_pairs"][ta_idx]
                    hateful_pred = s2_data["hateful_predictions"][ta_idx]

                    # 再次检查单个TA对和hateful预测的有效性
                    if ta_pair.get("target","").startswith("IE_") or \
                       ta_pair.get("target","").startswith("PARSE_") or \
                       hateful_pred.get("hateful_label","").startswith("SKIPPED_") or \
                       hateful_pred.get("hateful_label","").startswith("HATE_CLF_ERROR"):
                        # 如果单个TA对或其hateful预测无效，则为其创建一个表示错误的quadruple
                        quad = {
                            "target": ta_pair.get("target", "NULL"),
                            "argument": ta_pair.get("argument", "NULL"),
                            "hateful_label": hateful_pred.get("hateful_label", "HATE_CLF_ERROR"),
                            "targeted_group_label": "SKIPPED_INVALID_HATE_OR_IE"
                        }
                        current_sample_final_output["structured_quadruples"].append(quad)
                        continue # 跳过对此TA对的group分类

                    group_clf_overall_inputs.append({
                        "target": ta_pair.get("target", "NULL"),
                        "argument": ta_pair.get("argument", "NULL"),
                        "hateful_label": hateful_pred.get("hateful_label", "HATE_CLF_ERROR"), # 从s2获取
                        "content": s2_data["content"]
                    })
                    group_clf_map_back.append((sample_idx, ta_idx))
                    current_sample_final_output["structured_quadruples"].append({}) # Placeholder
            else: # 如果没有有效的TA对或hateful预测来进行group分类
                # 即使没有可供group分类的内容，也需要一个空的structured_quadruples或表示错误/跳过的条目
                # 为了与_format_quadruple_output_string兼容，这里保持为空列表
                pass


            final_pipeline_results.append(current_sample_final_output)


        all_group_results_flat = []
        if group_clf_overall_inputs:
            print(f"  Group CLF: Processing {len(group_clf_overall_inputs)} items for group classification...")
            try:
                all_group_results_flat = self.group_classifier.classify_batch(group_clf_overall_inputs)
            except Exception as e:
                print(f"  Error during batch Group Classification: {e}")
                traceback.print_exc()
                all_group_results_flat = [{"targeted_group_label": "GROUP_CLF_ERROR", "targeted_group_score": 0.0}] * len(group_clf_overall_inputs)

        for i, group_res in enumerate(all_group_results_flat):
            sample_idx, ta_idx = group_clf_map_back[i]

            # 从原始阶段数据中获取信息来构建完整的 quadruple
            # 注意：这里依赖于 stage1 和 stage2 的输出结构
            # 需要确保即使在错误情况下，这些结构也包含必要的键
            original_ta_pair = stage1_all_outputs[sample_idx]["extracted_ta_pairs"][ta_idx] \
                if sample_idx < len(stage1_all_outputs) and \
                   ta_idx < len(stage1_all_outputs[sample_idx]["extracted_ta_pairs"]) else \
                   {"target": "ERROR_ACCESSING_S1_TA", "argument": "ERROR_ACCESSING_S1_TA"}

            original_hateful_pred = stage2_all_outputs_with_hate[sample_idx]["hateful_predictions"][ta_idx] \
                if sample_idx < len(stage2_all_outputs_with_hate) and \
                   ta_idx < len(stage2_all_outputs_with_hate[sample_idx]["hateful_predictions"]) else \
                   {"hateful_label": "ERROR_ACCESSING_S2_HATE"}

            quad = {
                "target": original_ta_pair.get("target", "NULL"),
                "argument": original_ta_pair.get("argument", "NULL"),
                "hateful_label": original_hateful_pred.get("hateful_label", "HATE_CLF_ERROR"),
                "targeted_group_label": group_res.get("targeted_group_label", "GROUP_CLF_ERROR"),
                # 可以选择性加入scores
                # "hateful_score": original_hateful_pred.get("hateful_score", 0.0),
                # "targeted_group_score": group_res.get("targeted_group_score", 0.0)
            }
            # 确保 final_pipeline_results[sample_idx]["structured_quadruples"] 有足够的空间
            if ta_idx < len(final_pipeline_results[sample_idx]["structured_quadruples"]):
                final_pipeline_results[sample_idx]["structured_quadruples"][ta_idx] = quad
            else: # 理论上不应发生，如果发生了，可能是占位符逻辑问题
                print(f"  Error: ta_idx {ta_idx} out of bounds for structured_quadruples at sample_idx {sample_idx}")
                final_pipeline_results[sample_idx]["structured_quadruples"].append(quad)


        # 为每个条目生成最终的 "output" 字符串
        for item in final_pipeline_results:
            item["output"] = self._format_quadruple_output_string(item.get("structured_quadruples", []))

        self._save_stage_results(final_pipeline_results, "3_final_output_formatted_llm_ie", base_filename_for_saving)

        print(f"Dataset '{base_filename_for_saving}' processing finished with LLM_1 pipeline.")
        return final_pipeline_results

    # process_json_file 方法保持不变，它会调用 process_dataset
    def process_json_file(self, input_json_path: str, output_json_path: Optional[str] = None,
                          pipeline_batch_size: int = 32) -> List[Dict]:
        print(f"Loading input data from '{input_json_path}' using HateOriDataset...")
        try:
            # is_pred=True 表明我们不需要标签，只处理 id 和 content
            input_dataset = HateOriDataset(json_file_path=input_json_path, is_pred=True)
            if not input_dataset or len(input_dataset) == 0:
                print("输入JSON文件为空或HateOriDataset未能加载任何数据。")
                return []
        except Exception as e:
            print(f"使用HateOriDataset加载 '{input_json_path}' 时出错: {e}")
            import traceback; traceback.print_exc(); return []

        input_data_tuples = []
        for i in range(len(input_dataset)):
            # HateOriDataset 在 is_pred=True 时，getitem应该返回 (id, content, None) 或类似结构
            # 我们需要确保只取 id 和 content
            try:
                item_data = input_dataset[i]
                if len(item_data) >= 2: # 至少有id和content
                    item_id, content = item_data[0], item_data[1]
                else: # 结构不符合预期
                    print(f"警告: HateOriDataset 返回的数据结构不符合预期 (索引 {i}): {item_data}。跳过。")
                    continue

                if item_id is None or content is None:
                    print(f"警告: HateOriDataset 返回了无效的 id 或 content (索引 {i})。跳过。")
                    continue
                input_data_tuples.append((item_id, content))
            except Exception as e_item:
                print(f"处理来自 HateOriDataset 的条目 (索引 {i}) 时出错: {e_item}。跳过。")
                continue

        if not input_data_tuples:
            print("未能从HateOriDataset中提取有效的 (id, content) 对。")
            return []

        base_input_filename = os.path.splitext(os.path.basename(input_json_path))[0]

        all_processed_results = self.process_dataset(
            input_data_tuples,
            pipeline_batch_size=pipeline_batch_size, # IE_LLM的批处理大小
            base_filename_for_saving=base_input_filename
        )

        if output_json_path:
            try:
                output_dir = os.path.dirname(output_json_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(all_processed_results, f, ensure_ascii=False, indent=4)
                print(f"流水线最终预测结果已保存到: {output_json_path}")
            except Exception as e:
                print(f"保存最终输出JSON文件时出错: {e}")

        return all_processed_results
    
    
# 在 pipeline_llm_1.py 的末尾
if __name__ == "__main__":
    # --- LLM IE 模型路径 ---
    IE_LLM_BASE_MODEL = "models/Qwen3-8B" # 你的Qwen3基础模型路径
    IE_LLM_LORA_CHECKPOINT = "models/outputs/Qwen3-LoRA-TargetArgument/best_lora_adapter" # 你的LoRA权重路径
    QWEN_THINK_FALLBACK_ID = 151647 # 根据你的Qwen3模型确认</think>的ID

    # --- 分类器模型路径 (保持不变) ---
    HATE_CLF_CHECKPOINT_PATH = "./models/outputs/hate_clf/best_model_on_eval_loss"
    HATE_CLF_BASE_MODEL = "models/chinese-roberta-wwm-ext-large"
    GROUP_CLF_CHECKPOINT_PATH = "./models/outputs/group_clf/best_model_on_eval_loss"
    GROUP_CLF_BASE_MODEL = "models/chinese-roberta-wwm-ext-large"

    # --- 输入输出路径 ---
    INPUT_TEST_JSON_PATH = "data/segment/test.json" 
    PIPELINE_LLM1_INTERMEDIATE_STAGES_DIR = "data/outputs/pipeline_llm_stages_output" # 新的中间结果目录
    FINAL_PREDICTIONS_LLM1_OUTPUT_PATH = "data/outputs/pipeline_llm_final_predictions.json" # 新的最终输出文件

    print("检查模型和输入文件路径 (LLM_1 Pipeline)...")
    required_paths_info = {
        "IE LLM Base Model Dir": (IE_LLM_BASE_MODEL, True, "config.json"), # 检查Qwen模型目录
        "IE LLM LoRA Checkpoint Dir": (IE_LLM_LORA_CHECKPOINT, True, "adapter_model.safetensors"), # 或 adapter_model.bin
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
            # LoRA权重的检查可能需要更具体，adapter_config.json 通常存在
            if expected_file == "adapter_model.safetensors" and not os.path.exists(os.path.join(path, expected_file)) \
               and not os.path.exists(os.path.join(path, "adapter_model.bin")):
                content_ok = False; print(f"错误: {name} 目录 '{path}' 缺少 '{expected_file}' 或 'adapter_model.bin'。")
            elif not os.path.exists(os.path.join(path, expected_file)) and \
                 not (expected_file == "adapter_model.safetensors" and os.path.exists(os.path.join(path, "adapter_model.bin"))):
                content_ok = False; print(f"错误: {name} 目录 '{path}' 缺少 '{expected_file}'。")

        if not path_exists or not content_ok:
            if not path_exists : print(f"错误: {name} 路径 '{path}' 不存在。")
            all_paths_valid = False

    if not all_paths_valid:
        print("一个或多个必要路径无效或不完整，LLM流水线无法运行。")
    else:
        print("所有路径检查通过 (LLM_1 Pipeline)。")
        try:
            pipeline_llm1 = SentimentPipeline_LLM(
                ie_llm_base_model_name_or_path=IE_LLM_BASE_MODEL,
                ie_llm_lora_checkpoint_path=IE_LLM_LORA_CHECKPOINT,
                qwen_think_end_token_id_fallback=QWEN_THINK_FALLBACK_ID,
                hate_clf_checkpoint_path=HATE_CLF_CHECKPOINT_PATH,
                hate_clf_base_model_name_or_path=HATE_CLF_BASE_MODEL,
                group_clf_checkpoint_path=GROUP_CLF_CHECKPOINT_PATH,
                group_clf_base_model_name_or_path=GROUP_CLF_BASE_MODEL,
                device="cuda" if torch.cuda.is_available() else "cpu",
                output_intermediate_dir=PIPELINE_LLM1_INTERMEDIATE_STAGES_DIR,
                max_seq_length_ie_llm=256 # 根据你的IE输出长度调整
            )

            print(f"\n--- 开始处理测试文件 (LLM_1 Pipeline): {INPUT_TEST_JSON_PATH} ---")
            final_results_list = pipeline_llm1.process_json_file(
                input_json_path=INPUT_TEST_JSON_PATH,
                output_json_path=FINAL_PREDICTIONS_LLM1_OUTPUT_PATH,
                pipeline_batch_size=4 # LLM IE 可能比较慢，减小批次大小
            )

            if final_results_list:
                print(f"\n--- 处理完成 (LLM_1 Pipeline) ---")
                print(f"最终预测结果已保存到: {FINAL_PREDICTIONS_LLM1_OUTPUT_PATH}")
                print(f"共处理了 {len(final_results_list)} 条原始数据项。")
                print(f"每个阶段处理完整个数据集后的中间文件保存在: {pipeline_llm1.output_intermediate_dir} 目录中。")
            else:
                print("未能处理任何数据或生成任何结果 (LLM_1 Pipeline)。")

        except Exception as e:
            print(f"运行Pipeline_LLM_1时发生主错误: {e}")
            import traceback
            traceback.print_exc()