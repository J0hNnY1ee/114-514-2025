# information_extractor_llm.py (可以放在 core 目录下)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import traceback
from typing import List, Dict, Union

# 假设 get_analysis_prompts 在 LLM.prompt 中
try:
    from LLM.prompt import get_analysis_prompts
except ImportError:
    # 尝试相对导入（如果此文件在 core 内部，而 LLM 是 core 的同级或父级目录下的包）
    try:
        from ..LLM.prompt import get_analysis_prompts # 如果 LLM 是 core 的父目录下的包
    except ImportError:
        try:
            from ..LLM.prompt import get_analysis_prompts # 如果 LLM 是 core 的父目录的父目录下的包
        except ImportError as e_llm:
            print(f"错误: 无法从 LLM.prompt 导入 get_analysis_prompts。错误: {e_llm}")
            print("请确保 LLM 模块及其 prompt.py 文件在 Python 搜索路径中。")
            # 定义一个占位符函数，以便代码结构完整，但会报错如果实际调用
            def get_analysis_prompts(content: str):
                raise NotImplementedError("get_analysis_prompts未能成功导入")


class InformationExtractor_LLM:
    def __init__(self,
                 base_model_name_or_path: str, # 例如 "models/Qwen3-8B"
                 trained_model_checkpoint_path: str, # 例如 "models/outputs/Qwen3-LoRA-TargetArgument/best_lora_adapter"
                 device: Union[str, torch.device] = "cpu",
                 max_seq_length: int = 512, # 这里是生成IE结果的最大token数
                 # Qwen3 的 </think> token ID，需要确认
                 # Qwen1.5: 151668, Qwen2: 151647
                 qwen_think_end_token_id_fallback: int = 151647 # 默认使用Qwen2的ID作为后备
                 ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_new_tokens_for_ie = max_seq_length # 参数名统一
        self.model = None
        self.tokenizer = None
        self.think_end_token_id = None
        self.qwen_think_end_token_id_fallback = qwen_think_end_token_id_fallback

        self._load_model(base_model_name_or_path, trained_model_checkpoint_path)

    def _load_model(self, base_model_path: str, lora_adapter_path: str):
        if not os.path.isdir(base_model_path):
            raise FileNotFoundError(f"基础模型目录未找到：{base_model_path}")
        if not os.path.isdir(lora_adapter_path):
            raise FileNotFoundError(f"LoRA 适配器目录未找到：{lora_adapter_path}")

        print(f"IE_LLM: 基础模型路径: {base_model_path}")
        print(f"IE_LLM: LoRA 适配器路径: {lora_adapter_path}")
        print(f"IE_LLM: 当前使用的设备: {self.device}")

        try:
            print(f"IE_LLM: 正在从基础模型路径加载 Tokenizer: {base_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            print("IE_LLM: Tokenizer 加载完成。")

            print(f"IE_LLM: 正在从基础模型路径加载基础模型: {base_model_path}")
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype="auto",
                trust_remote_code=True
            )
            print("IE_LLM: 基础模型加载完成。")

            print(f"IE_LLM: 正在从以下路径加载 LoRA 适配器并应用到基础模型: {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(base_model_obj, lora_adapter_path)
            print("IE_LLM: LoRA 适配器层已加载。")

            print("IE_LLM: 正在合并 LoRA 适配器权重到基础模型...")
            self.model = self.model.merge_and_unload()
            print("IE_LLM: LoRA 适配器已合并。")

            print(f"IE_LLM: 将模型移动到设备: {self.device}")
            self.model = self.model.to(self.device)
            print(f"IE_LLM: 最终模型已在设备: {next(self.model.parameters()).device}")
            self.model.eval()
            print("IE_LLM: 模型已设置为评估模式。")

            think_end_token_str = "</think>"
            think_end_token_id_list = self.tokenizer.encode(think_end_token_str, add_special_tokens=False)
            if len(think_end_token_id_list) == 1:
                self.think_end_token_id = think_end_token_id_list[0]
                print(f"IE_LLM: 成功获取到 '{think_end_token_str}' 的 Token ID: {self.think_end_token_id}")
            else:
                print(f"IE_LLM: 警告: 未能可靠地获取 '{think_end_token_str}' 的单个 Token ID。编码结果为: {think_end_token_id_list}。")
                print(f"IE_LLM: 将回退到硬编码的 Token ID: {self.qwen_think_end_token_id_fallback} (请为 Qwen3-8B 核实此值)")
                self.think_end_token_id = self.qwen_think_end_token_id_fallback

        except Exception as e:
            print(f"IE_LLM: 在加载模型或 LoRA 适配器时发生错误: {e}")
            traceback.print_exc()
            raise  # 重新抛出异常，让上层知道加载失败

    def _parse_ie_string_to_ta_pairs(self, ie_result_str: str) -> List[Dict[str, str]]:
        """
        将模型输出的 "T1 | A1 [SEP] T2 | A2 [END]" 格式字符串解析为TA对列表。
        例如: "女人 | 找黑人我也会疏远 [SEP] 黑黄混血 | 我一样会远离 [END]"
        会变成:
        [
            {"target": "女人", "argument": "找黑人我也会疏远"},
            {"target": "黑黄混血", "argument": "我一样会远离"}
        ]
        如果格式不正确或为空，返回空列表或包含错误信息的单个元素列表。
        """
        pairs = []
        if not ie_result_str or not ie_result_str.strip():
            return [{"target": "PARSE_EMPTY_IE_STRING", "argument": "PARSE_EMPTY_IE_STRING"}]

        # 移除末尾的 [END] 标记（如果存在）
        content_to_parse = ie_result_str
        end_tag = "[END]" # 注意前后是否有空格
        if content_to_parse.endswith(" " + end_tag):
            content_to_parse = content_to_parse[:-len(" " + end_tag)].strip()
        elif content_to_parse.endswith(end_tag):
             content_to_parse = content_to_parse[:-len(end_tag)].strip()


        if not content_to_parse: # 如果移除 [END] 后为空
             return [{"target": "PARSE_ONLY_END_TAG", "argument": "PARSE_ONLY_END_TAG"}]

        segments = content_to_parse.split(" [SEP] ")
        for segment in segments:
            parts = segment.split(" | ")
            if len(parts) == 2:
                target = parts[0].strip()
                argument = parts[1].strip()
                if target and argument: # 确保T和A都不是空字符串
                    pairs.append({"target": target, "argument": argument})
                elif target: # 只有target
                    pairs.append({"target": target, "argument": "ARG_MISSING_IN_PARSE"})
                elif argument: # 只有argument
                    pairs.append({"target": "TARGET_MISSING_IN_PARSE", "argument": argument})
                # else: 两者都为空，忽略
            elif len(parts) == 1 and parts[0].strip(): # 只有一个部分，且不为空
                 pairs.append({"target": parts[0].strip(), "argument": "ARG_MISSING_MALFORMED"})
            elif len(parts) > 2: # 超过两个部分，也认为是格式错误
                pairs.append({"target": " | ".join(parts), "argument": "MALFORMED_TOO_MANY_PARTS"})
            # else: segment 为空或 parts 为空，忽略

        if not pairs: # 如果解析后什么都没有
            return [{"target": "PARSE_NO_VALID_PAIRS", "argument": ie_result_str[:100]}] # 返回原始IE输出的前100个字符作为参考
        return pairs

    def _generate_ie_result(self, system_prompt_str: str, user_prompt_str: str,
                            try_disable_thinking: bool = False # 与pipeline中默认行为一致
                            ) -> str:
        messages = [
            {"role": "system", "content": system_prompt_str},
            {"role": "user", "content": user_prompt_str}
        ]
        enable_think_flag = not try_disable_thinking

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_think_flag
            )
        except TypeError: # 如果 tokenizer 不支持 enable_thinking
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=self.max_new_tokens_for_ie,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        output_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:].tolist()
        actual_content = ""

        current_think_id_to_use = self.think_end_token_id
        if current_think_id_to_use is None: # 应该在_load_model中设置了后备
            current_think_id_to_use = self.qwen_think_end_token_id_fallback


        if enable_think_flag and current_think_id_to_use is not None:
            try:
                if not isinstance(current_think_id_to_use, int):
                    raise ValueError(f"current_think_id_to_use 不是整数: {current_think_id_to_use}")
                idx_think_end_in_reversed = output_ids[::-1].index(current_think_id_to_use)
                index_of_think_end_token = len(output_ids) - 1 - idx_think_end_in_reversed
                # 我们需要 </think> 之后的内容
                actual_content = self.tokenizer.decode(output_ids[index_of_think_end_token + 1:], skip_special_tokens=True).strip(" \n")
            except ValueError:
                actual_content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(" \n")
            except Exception:
                actual_content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(" \n")
        else:
            actual_content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip(" \n")
        return actual_content

    def extract(self, contents: List[str]) -> List[List[Dict[str, str]]]:
        """
        对一批文本内容进行信息抽取。
        返回一个列表，每个元素是对应输入内容的TA对列表。
        """
        all_extracted_ta_lists = []
        if not self.model or not self.tokenizer:
            print("IE_LLM: 模型或Tokenizer未加载，无法进行抽取。")
            error_result = [{"target": "IE_MODEL_NOT_LOADED", "argument": "IE_MODEL_NOT_LOADED"}]
            return [error_result for _ in contents]

        for content_idx, original_content in enumerate(contents):
            # print(f"  IE_LLM: Processing content {content_idx + 1}/{len(contents)}") # 批处理时减少打印
            if not original_content or not original_content.strip():
                all_extracted_ta_lists.append([{"target": "EMPTY_INPUT_CONTENT", "argument": "EMPTY_INPUT_CONTENT"}])
                continue
            try:
                system_p, user_p = get_analysis_prompts(original_content)
                ie_result_str = self._generate_ie_result(
                    system_p, user_p,
                    try_disable_thinking=False # 保持与pipeline一致，默认进行思考并解析
                )
                ta_pairs = self._parse_ie_string_to_ta_pairs(ie_result_str)
                all_extracted_ta_lists.append(ta_pairs)
            except Exception as e:
                print(f"  IE_LLM: Error processing content '{original_content[:50]}...': {e}")
                traceback.print_exc()
                all_extracted_ta_lists.append([{"target": "IE_RUNTIME_ERROR", "argument": str(e)[:100]}])
        return all_extracted_ta_lists