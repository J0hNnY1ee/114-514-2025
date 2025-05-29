import json
import argparse # 用于处理命令行参数

def extract_original_output(input_json_file, output_txt_file):
    """
    Reads a JSON file, extracts the "output" field from each item AS IS,
    and writes each "output" string to a new line in a text file.
    No transformation or processing is done on the "output" string.
    """
    original_outputs = []

    try:
        with open(input_json_file, 'r', encoding='utf-8') as f_json:
            data_list = json.load(f_json)
    except FileNotFoundError:
        print(f"错误：输入文件 '{input_json_file}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误：输入文件 '{input_json_file}' 不是有效的JSON格式。")
        return
    except Exception as e:
        print(f"读取输入文件时发生未知错误: {e}")
        return

    for item in data_list:
        if "output" in item:
            if isinstance(item["output"], str):
                original_outputs.append(item["output"])
            else:
                print(f"警告：ID {item.get('id', 'N/A')} 的 'output' 字段不是字符串，已跳过。值为: {item['output']}")
        else:
            # 如果某个条目没有 output 字段，可以选择跳过或记录一个空行/提示
            print(f"警告：ID {item.get('id', 'N/A')} 的条目缺少 'output' 字段，已跳过。")


    try:
        with open(output_txt_file, 'w', encoding='utf-8') as f_out:
            for line in original_outputs:
                f_out.write(line + "\n") # 每条 output 占一行
        print(f"原始 'output' 字段提取完成！结果已保存到 '{output_txt_file}'")
    except IOError:
        print(f"错误：无法写入到输出文件 '{output_txt_file}'。")
    except Exception as e:
        print(f"写入输出文件时发生未知错误: {e}")


if __name__ == "__main__":
    extract_original_output("data/original/outputs/test2.json", "data/original/outputs/test2.txt")