import json
import os

def convert_jsonl_to_json(jsonl_filepath: str, json_filepath: str):
    """
    将JSON Lines文件转换为标准的JSON数组文件。

    参数:
        jsonl_filepath (str): 输入的JSON Lines文件路径 (.jsonl)。
        json_filepath (str): 输出的JSON文件路径 (.json)。
    """
    if not os.path.exists(jsonl_filepath):
        print(f"错误: 输入文件 '{jsonl_filepath}' 不存在。")
        return

    all_objects = []
    print(f"正在读取JSON Lines文件: {jsonl_filepath} ...")
    try:
        with open(jsonl_filepath, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    json_object = json.loads(line)
                    all_objects.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"警告: 无法解析第 {line_num + 1} 行: '{line}'. 错误: {e}. 跳过此行。")
    except Exception as e:
        print(f"读取文件 '{jsonl_filepath}' 时发生错误: {e}")
        return

    if not all_objects:
        print(f"未从 '{jsonl_filepath}' 中读取到任何有效的JSON对象。输出文件将为空列表。")
    
    print(f"正在写入JSON数组文件: {json_filepath} ...")
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(json_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建目录: {output_dir}")

        with open(json_filepath, 'w', encoding='utf-8') as outfile:
            json.dump(all_objects, outfile, ensure_ascii=False, indent=4) # indent=4 用于美化输出
        print(f"转换完成! {len(all_objects)} 个对象已写入 '{json_filepath}'。")
    except Exception as e:
        print(f"写入文件 '{json_filepath}' 时发生错误: {e}")


if __name__ == "__main__":

    
    convert_jsonl_to_json("data/outputs/evaluation_paired_data.jsonl", "data/outputs/evaluation_paired_data.json")