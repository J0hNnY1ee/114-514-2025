import json
import argparse
import os
import sys







try:
    from LLM.prompt import get_analysis_prompts
except ImportError as e:
    print(f"错误：无法从 'LLM.prompt' 导入 'get_analysis_prompts' 函数。错误信息: {e}")
    print(f"当前 sys.path: {sys.path}")
    print("请确保 'LLM' 文件夹（包含 'prompt.py'）位于Python可以找到的路径中，并且 'prompt.py' 中定义了该函数。")
    print(f"脚本期望 'LLM' 文件夹与脚本在同一级别，或者其父目录是 sys.path 的一部分。当前脚本目录: {os.path.dirname(os.path.abspath(__file__))}")
    exit(1)


def process_raw_output_to_answer(raw_output_str: str) -> str:
    """
    将原始的输出字符串（可能包含额外标签）处理成模型微调所需的 'answer' 格式。
    原始格式可能为：T1 | A1 | L1_1 | L1_2 [SEP] T2 | A2 | L2_1 | L2_2 [END]
    或 T1 | A1 [SEP] T2 | A2 [END] （如果原始数据不含标签，或标签已被去除）
    目标格式是 `Target | Argument [END]` 或 `T1 | A1 [SEP] T2 | A2 [END]`。
    此函数会移除每个 Target-Argument 对后面的两个额外标签（如果存在）。

    Args:
        raw_output_str: 原始数据集中的 "output" 字段字符串。

    Returns:
        处理后，符合模型微调期望的 "answer" 字符串。
    """
    end_tag = " [END]"
    sep_tag = " [SEP] "
    
    # 检查并移除末尾的 [END] 标签，后续会重新加上
    content_to_process = raw_output_str
    if raw_output_str.endswith(end_tag):
        content_to_process = raw_output_str[:-len(end_tag)]
    else:
        # 如果没有 [END] 标签，记录警告，但仍然处理
        print(
            f"警告: 输入的原始输出 '{raw_output_str}' 不以 '{end_tag}' 结尾。将尝试处理并追加 '{end_tag}'。"
        )

    # 按 [SEP] 分割成独立的论点单元
    argument_units_raw = content_to_process.split(sep_tag)
    
    processed_units = []
    num_labels_per_unit = 2 # 每个论点单元（T | A）后面跟的标签数量

    for unit_raw in argument_units_raw:
        if not unit_raw.strip(): # 跳过因 " [SEP]  [SEP] " 或首尾 [SEP] 产生的空单元
            continue
            
        parts = unit_raw.split(" | ")
        
        # 期望格式是 Target | Argument | Label1 | Label2 (4部分)
        # 或 Target | Argument (2部分，如果标签不存在或已被移除)
        # 我们只需要 Target | Argument
        
        if len(parts) >= 2: # 至少要有 Target 和 Argument
            # 取前两部分（Target 和 Argument）
            target_argument_pair = " | ".join(parts[:2])
            processed_units.append(target_argument_pair)
        elif len(parts) == 1: # 只有一个部分，可能是Target，或者格式错误
            print(f"警告: 论点单元 '{unit_raw}' 在 '{raw_output_str}' 中只有一个部分，将按原样保留。")
            processed_units.append(parts[0]) # 保留这一个部分
        # 如果 len(parts) == 0 (不太可能除非 unit_raw 是由多个 | 组成的空字符串) 则忽略

    # 用 [SEP] 重新组合处理过的单元，并加上 [END]
    if not processed_units: # 如果处理后什么都没有（例如，输入是空的或只有 [END]）
        print(f"警告: 原始输出 '{raw_output_str}' 处理后没有有效的论点单元。")
        return end_tag.strip() # 返回 "[END]" 或空字符串 + end_tag，取决于end_tag定义

    final_answer_base = sep_tag.join(processed_units)
    return final_answer_base + end_tag


def run_conversion(input_file: str, output_file: str):
    """
    执行从原始数据到微调数据集的转换。
    """
    fine_tuning_dataset = []
    raw_dataset = []

    print(f"正在从 '{input_file}' 读取数据...")
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            raw_dataset = json.load(infile)
            if not isinstance(raw_dataset, list):
                print(
                    f"错误: 输入的JSON文件 '{input_file}' 的顶层结构不是一个列表。"
                )
                return
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误: 无法解析 '{input_file}' 为有效的JSON。")
        return
    except Exception as e:
        print(f"读取输入文件时发生错误: {e}")
        return

    for item_number, raw_item in enumerate(raw_dataset, 1):
        if not isinstance(raw_item, dict):
            print(
                f"警告: 跳过第 {item_number} 个条目，因为它不是一个字典对象: {raw_item}"
            )
            continue

        if "content" not in raw_item or "output" not in raw_item:
            print(
                f"警告: 跳过第 {item_number} 个条目，缺少 'content' 或 'output' 字段: {raw_item}"
            )
            continue

        content_to_analyze = raw_item["content"]
        original_full_output = raw_item["output"]

        _system_prompt, user_prompt_for_tuning = get_analysis_prompts(
            content_to_analyze
        )
        answer_for_tuning = process_raw_output_to_answer(original_full_output)

        fine_tuning_item = {
            "question": user_prompt_for_tuning,
            "think": "",
            "answer": answer_for_tuning,
        }
        fine_tuning_dataset.append(fine_tuning_item)

    print(f"数据处理完成。共生成 {len(fine_tuning_dataset)} 条微调数据。")

    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir): # output_dir可能为空（如果文件名不含路径）
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")

        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(fine_tuning_dataset, outfile, ensure_ascii=False, indent=4)
        print(f"微调数据集已成功保存到 '{output_file}'。")
    except Exception as e:
        print(f"保存输出文件时发生错误: {e}")


def main():
    # --- 在这里配置默认文件名 ---
    DEFAULT_INPUT_FILE = "data/segment/test.json"  # 默认输入文件名
    DEFAULT_OUTPUT_FILE = "data/finetune/test.json" # 默认输出文件名
    # --------------------------

    parser = argparse.ArgumentParser(
        description="根据原始数据生成用于LLM微调的数据集。"
    )
    parser.add_argument(
        "--input_json_file",
        type=str,
        default=DEFAULT_INPUT_FILE, # 使用默认值
        help=f"输入的 JSON 文件路径 (默认: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output_json_file",
        type=str,
        default=DEFAULT_OUTPUT_FILE, # 使用默认值
        help=f"输出的 JSON 文件路径 (默认: {DEFAULT_OUTPUT_FILE})",
    )

    args = parser.parse_args()
    
    # 确保输入文件路径是相对于脚本位置的（如果它们是相对路径）
    # 或者，如果用户提供了绝对路径，os.path.join不会改变它
    # 如果默认路径是相对于脚本的，这样处理更健壮
    input_file_path = args.input_json_file
    output_file_path = args.output_json_file


    run_conversion(input_file_path, output_file_path)


if __name__ == "__main__":
    main()