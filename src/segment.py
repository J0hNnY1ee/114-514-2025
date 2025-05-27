import json
import random

# 读取原始数据文件
input_file = 'data/original/train.json'  # 替换为你自己的文件名
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 打乱数据顺序
random.shuffle(data)

# 按比例划分：80% 训练集，10% 验证集，10% 测试集
total = len(data)
train_end = int(total * 0.8)
val_end = train_end + int(total * 0.1)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]

# 保存到新文件
with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open('val.json', 'w', encoding='utf-8') as f:
    json.dump(val_data, f, ensure_ascii=False, indent=2)

with open('test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("✅ 数据划分完成！")
print(f"Total: {total}")
print(f"Train: {len(train_data)}")
print(f"Val: {len(val_data)}")
print(f"Test: {len(test_data)}")