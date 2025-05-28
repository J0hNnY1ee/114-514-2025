import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import traceback
import json # 用于 JSON 文件的读写
import argparse # 用于接收输入输出文件名参数
import traceback 
# --- 1. 从 prompt.py 导入函数 ---
try:
    from LLM.prompt import get_analysis_prompts
except ImportError:
    print("错误：无法从 prompt.py 导入 get_analysis_prompts。请确保该文件存在于当前目录或PYTHONPATH中。")
    exit(1)

# --- 全局变量定义 ---
model = None
tokenizer = None
think_end_token_id = None
device = "cuda" if torch.cuda.is_available() else "cpu"


# --- 模型加载函数 (保持不变，但注意 think_end_token_id 的确认) ---
def load_model_and_tokenizer(base_model_path: str, lora_adapter_path: str):
    global model, tokenizer, think_end_token_id, device

    if not os.path.isdir(base_model_path):
        print(f"错误：未找到基础模型目录：{base_model_path}")
        return False
    if not os.path.isdir(lora_adapter_path):
        print(f"错误：未找到 LoRA 适配器目录：{lora_adapter_path}")
        return False

    print(f"基础模型路径: {base_model_path}")
    print(f"LoRA 适配器路径: {lora_adapter_path}")
    print(f"当前使用的设备: {device}")

    try:
        print(f"正在从基础模型路径加载 Tokenizer: {base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        print("Tokenizer 加载完成。")

        print(f"正在从基础模型路径加载基础模型: {base_model_path}")
        base_model_obj = AutoModelForCausalLM.from_pretrained( # 重命名以避免与全局 model 冲突
            base_model_path,
            torch_dtype="auto",
            trust_remote_code=True
        )
        print("基础模型加载完成。")

        print(f"正在从以下路径加载 LoRA 适配器并应用到基础模型: {lora_adapter_path}")
        model = PeftModel.from_pretrained(base_model_obj, lora_adapter_path)
        print("LoRA 适配器层已加载。")

        print("正在合并 LoRA 适配器权重到基础模型...")
        model = model.merge_and_unload()
        print("LoRA 适配器已合并。模型现在是标准的 Transformer 模型。")

        print(f"将模型移动到设备: {device}")
        model = model.to(device)
        print(f"最终模型已在设备: {next(model.parameters()).device}")
        model.eval()
        print("模型已设置为评估模式。")

        think_end_token_str = "</think>"
        think_end_token_id_list = tokenizer.encode(think_end_token_str, add_special_tokens=False)
        if len(think_end_token_id_list) == 1:
            think_end_token_id = think_end_token_id_list[0]
            print(f"成功获取到 '{think_end_token_str}' 的 Token ID: {think_end_token_id}")
        else:
            print(f"警告: 未能可靠地获取 '{think_end_token_str}' 的单个 Token ID。编码结果为: {think_end_token_id_list}。")

            fallback_think_id = 151647 # 假设是 Qwen2/Qwen3 风格的 ID
            print(f"将回退到硬编码的 Token ID: {fallback_think_id} (请为 Qwen3-8B 核实此值)")
            think_end_token_id = fallback_think_id
        return True
    except Exception as e:
        print(f"在加载模型或 LoRA 适配器时发生错误: {e}")
        traceback.print_exc()
        return False




# --- 生成函数 (严格按照官方示例逻辑，移除调试) ---
def generate_actual_content(
    system_prompt_str: str,
    user_prompt_str: str,
    max_new_tokens=250,
    try_disable_thinking=False # 默认进行思考
    ):
    global model, tokenizer, think_end_token_id # 确保访问全局变量

    messages = [
        {"role": "system", "content": system_prompt_str},
        {"role": "user", "content": user_prompt_str}
    ]
    enable_think_flag = not try_disable_thinking # 默认为 True

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_think_flag
        )
    except TypeError as te:
        if "enable_thinking" in str(te).lower():
            # 不再打印警告，因为我们现在遵循官方逻辑，即使参数无效，后续解析也可能工作
            # print(f"警告: Tokenizer.apply_chat_template 不支持 'enable_thinking'...")
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            raise te

    current_model_device = next(model.parameters()).device
    model_inputs = tokenizer([text], return_tensors="pt").to(current_model_device)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    output_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:].tolist()
    # thinking_content = "" # 我们不再显式分离 thinking_content 给外部
    actual_content = ""

    # 尝试使用全局 think_end_token_id，如果未设置则回退到官方示例的 151668
    current_think_id_to_use = think_end_token_id
    if current_think_id_to_use is None:
        # print("警告: 全局 think_end_token_id 未设置，将尝试使用官方示例的 151668。")
        current_think_id_to_use = 151668 # Qwen 官方示例使用的 ID

    if enable_think_flag and current_think_id_to_use is not None:
        try:
            if not isinstance(current_think_id_to_use, int):
                # 这种内部错误应该比较少见，如果发生，当作解析失败处理
                raise ValueError(f"current_think_id_to_use 不是整数: {current_think_id_to_use}")

            # 官方解析逻辑: index 指向 </think> token 本身
            idx_think_end_in_reversed = output_ids[::-1].index(current_think_id_to_use)
            index_of_think_end_token = len(output_ids) - 1 - idx_think_end_in_reversed

            # 官方示例的 content (我们称之为 actual_content) 是从 </think> token 开始的
            actual_content = tokenizer.decode(output_ids[index_of_think_end_token:], skip_special_tokens=True).strip(" \n")
            # 注意：thinking_content (即 output_ids[:index_of_think_end_token]) 在此函数中不再被使用或返回

        except ValueError: # 如果找不到 current_think_id_to_use
            # 按照官方示例，如果找不到 </think>，整个输出都作为 content
            actual_content = tokenizer.decode(output_ids, skip_special_tokens=True).strip(" \n")
        except Exception:
            # 捕获其他潜在的解析错误，并将整个输出视为内容
            # traceback.print_exc() # 在生产环境中通常不打印堆栈，除非有专门的日志系统
            actual_content = tokenizer.decode(output_ids, skip_special_tokens=True).strip(" \n")
    else: # 如果不启用思考或 think_id 无效，整个输出是 actual_content
        actual_content = tokenizer.decode(output_ids, skip_special_tokens=True).strip(" \n")

    return actual_content


# --- 主程序 ---
def main():
    parser = argparse.ArgumentParser(description="使用Qwen3-8B LoRA模型对JSON数据进行信息抽取，并将结果添加到原数据中。")
    parser.add_argument(
        "--base_model", type=str, default="models/Qwen3-8B",
        help="基础模型路径"
    )
    parser.add_argument(
        "--lora_adapter", type=str, default="models/outputs/Qwen3-LoRA-TargetArgument/best_lora_adapter",
        help="LoRA适配器路径"
    )
    parser.add_argument(
        "--input_json_file", type=str, 
        default="data/segment/test.json",
        help="包含 'content' 字段的输入JSON文件路径"
    )
    parser.add_argument(
        "--output_json_file", type=str, 
        default="data/segment/test_ie.json",
        help="保存带有 'IE_result' 字段的输出JSON文件路径"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=250,
        help="模型生成的最大新token数"
    )
    parser.add_argument(
        "--try_disable_thinking", action='store_true', default=False, 
        help="尝试禁用模型的思考过程输出 (如果模型和tokenizer支持)"
    )
    parser.add_argument(
        "--no_try_disable_thinking", action='store_false', dest='try_disable_thinking',
        help="显式启用模型的思考过程并进行解析 (会覆盖 --try_disable_thinking 的默认行为)"
    )


    args = parser.parse_args()

    # 加载模型
    if not load_model_and_tokenizer(args.base_model, args.lora_adapter):
        print("模型加载失败，程序退出。")
        return

    # 读取输入JSON文件
    try:
        with open(args.input_json_file, "r", encoding="utf-8") as infile:
            dataset = json.load(infile)
        if not isinstance(dataset, list):
            print(f"错误: 输入文件 '{args.input_json_file}' 的顶层不是一个列表。")
            return
        print(f"成功从 '{args.input_json_file}' 读取 {len(dataset)} 条数据。")
    except FileNotFoundError:
        print(f"错误: 输入文件 '{args.input_json_file}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析 '{args.input_json_file}' 为有效的JSON。")
        return
    except Exception as e:
        print(f"读取输入文件时发生错误: {e}")
        return

    processed_dataset = []
    total_items = len(dataset)

    for i, item in enumerate(dataset):
        print(f"\n--- 正在处理条目 {i+1}/{total_items} ---")
        if not isinstance(item, dict) or "content" not in item:
            print(f"警告: 条目 {i+1} 格式不正确或缺少 'content' 字段，已跳过: {item}")
            # 仍然将原始条目添加到结果中，或者你可以选择完全跳过
            processed_dataset.append(item)
            continue

        original_content = item["content"]
        print(f"原始 Content: {original_content}")

        # 1. 获取 Prompts
        system_p, user_p = get_analysis_prompts(original_content)

        # 2. 模型推理得到实际抽取内容
        ie_result = generate_actual_content(
            system_p,
            user_p,
            max_new_tokens=args.max_new_tokens,
            try_disable_thinking=args.try_disable_thinking
        )
        print(f"模型抽取的 IE 结果: {ie_result}")

        # 3. 将 IE 结果添加到原始条目中
        # 我们直接修改 item 字典，因为它会被添加到 processed_dataset
        item["IE_result"] = ie_result # 添加新字段

        processed_dataset.append(item) # 将更新后的条目添加到结果列表

    # 保存处理后的数据集到新的 JSON 文件
    output_dir = os.path.dirname(args.output_json_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    try:
        with open(args.output_json_file, "w", encoding="utf-8") as outfile:
            json.dump(processed_dataset, outfile, ensure_ascii=False, indent=4)
        print(f"\n处理完成！包含 'IE_result' 的数据集已保存到 '{args.output_json_file}'。")
    except Exception as e:
        print(f"保存输出文件时发生错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()