# 文件名，请确保它和你的实际文件名一致
filename = "test.txt"

try:
    # 使用 'r' 模式表示读取文本文件
    # encoding='utf-8' 明确指定使用UTF-8编码来解码文件内容
    # 这是读取无BOM UTF-8文件的标准方式
    with open(filename, "r", encoding='utf-8') as file:
        print(f"--- 开始读取文件: {filename} (编码: UTF-8) ---")
        
        # 一行一行读取并打印
        line_number = 0
        for line in file:
            line_number += 1
            # line 变量会包含行尾的换行符，print() 默认也会加一个换行符
            # 使用 strip() 去除行尾换行符，或者用 print(line, end='') 避免额外换行
            print(f"行 {line_number}: {line.strip()}") 
            # 或者不去除换行符，让print自己处理:
            # print(line, end='') 

        print(f"--- 文件读取结束: {filename} ---")
        print(f"总共读取了 {line_number} 行。")

except FileNotFoundError:
    print(f"错误: 文件 '{filename}' 未找到。请确保文件存在于脚本相同的目录下，或者提供正确的文件路径。")
except UnicodeDecodeError as e:
    print(f"错误: 文件 '{filename}' 在以UTF-8解码时发生错误。")
    print(f"错误详情: {e}")
    print("这表明文件内容中可能存在非UTF-8的字节序列。")
    print("如果VS Code能正常显示，可能是VS Code做了一些容错处理，")
    print("或者文件在之前的处理中并未完全转为纯净的UTF-8 (无BOM)。")
    print("你可以尝试使用之前带有 'errors=replace' 的修复脚本再处理一次原始 test.txt。")
except Exception as e:
    print(f"读取文件时发生未知错误: {e}")