# loader.py
import json
import os # 导入 os 模块用于文件操作
from torch.utils.data import Dataset
# 确保从 preprocessor 导入所有需要的映射表和函数
from data_process.preprocessor import (
    preprocess_target_group,
    preprocess_hateful_label,
    GROUP_TO_ID,
    ID_TO_GROUP,
    HATEFUL_TO_ID,
    ID_TO_HATEFUL,
    GROUP_CATEGORIES,
    HATEFUL_CATEGORIES
)

class HateOriDataset(Dataset):
    """
    用于读取和处理 val.json 格式数据的自定义数据集类。
    可以处理带有 'output' 字段的数据 (用于训练/验证)
    或仅有 'id' 和 'content' 的数据 (用于预测)。
    """
    def __init__(self, json_file_path: str, is_pred: bool = False):
        """
        初始化数据集。

        参数:
            json_file_path (str): 包含数据的JSON文件路径。
            is_pred (bool): 如果为 True，则表示数据集用于预测，
                            此时数据中预计没有 'output' 字段，标签将返回为空列表。
                            默认为 False。
        """
        self.data = self._load_data(json_file_path)
        self.is_pred = is_pred
        # 映射表在非预测模式下使用
        if not self.is_pred:
            self.group_to_id = GROUP_TO_ID
            self.hateful_to_id = HATEFUL_TO_ID

    def _load_data(self, json_file_path: str):
        """
        从JSON文件加载数据。
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"错误: 文件 {json_file_path} 未找到。")
            return []
        except json.JSONDecodeError:
            print(f"错误: 文件 {json_file_path} 不是有效的JSON格式。")
            return []

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        获取数据集中指定索引的样本。

        参数:
            idx (int): 样本的索引。

        返回:
            tuple:
                - item_id (int): 样本ID。
                - content (str): 评论内容。
                - labels (list):
                    - 如果 is_pred 为 True，则为空列表 []。
                    - 如果 is_pred 为 False，则为一个标签字典列表，每个字典格式为
                      {"target": Target, "argument": Argument,
                       "target_group_ids": TargetGroup_ids, "hateful_id": Hateful_id}。
        """
        if not self.data or idx >= len(self.data):
            raise IndexError("Dataset index out of range.")

        item = self.data[idx]
        item_id = item.get('id')
        content = item.get('content', '')

        if item_id is None:
            print(f"警告: 索引 {idx} 处的样本缺少 'id' 字段。将使用 None 作为 ID。")
        
        if self.is_pred:
            return item_id, content, [] # 预测模式下，标签返回空列表

        # 非预测模式 (is_pred is False)，解析 output 字段
        output_str = item.get('output', '') # 如果 'output' 字段不存在，也按空处理
        labels = []
        if not output_str: # 如果 output 为空或不存在，则标签也为空
            if 'output' not in item:
                 print(f"警告: ID {item_id if item_id is not None else '未知'} 处于非预测模式，但缺少 'output' 字段。返回空标签。")
            return item_id, content, []


        # 移除末尾的 '[END]' 标记
        if output_str.endswith("[END]"):
            output_str = output_str[:-len("[END]")].strip()

        # 按 '[SEP]' 分割得到每个四元组的字符串
        quadruplet_strs = output_str.split('[SEP]')

        for quad_str in quadruplet_strs:
            quad_str = quad_str.strip()
            if not quad_str:
                continue

            parts = [p.strip() for p in quad_str.split('|')]
            if len(parts) == 4:
                target = parts[0]
                argument = parts[1]
                target_group_str = parts[2]
                hateful_str = parts[3]

                target_group_ids = preprocess_target_group(target_group_str)
                hateful_id = preprocess_hateful_label(hateful_str)

                if hateful_id is not None:
                    labels.append({
                        "target": target,
                        "argument": argument,
                        "target_group_ids": target_group_ids,
                        "hateful_id": hateful_id
                    })
            else:
                print(f"警告: ID {item_id if item_id is not None else '未知'} 中的输出片段格式不正确: '{quad_str}'，期望4个部分，得到{len(parts)}个。")

        return item_id, content, labels

def testOriDataset():
    # --- 测试数据定义 ---
    # 1. 带有 'output' 字段的训练/验证数据
    train_val_data_content = [
        {
            "id": 7024,
            "content": "所以，你5点问题其实也就是归结为三点：1。同性恋对反对者和异性恋乱喷。这是这些乱喷者受压后不理智的错误行为。2。性倾向的不可改变性。这是医学的结论，是你自己（还有某些同性恋者）认识错误。3。疾病传播。同异同样有责任，并不是某个群体的特定问题。而且公平化和正面教育有助于解决此问题，而不是相反。",
            "output": "同性恋 | 对反对者和异性恋乱喷 | non-hate | non-hate [SEP] 性倾向 | 不可改变性 | non-hate | non-hate [SEP] 疾病传播 | 同异同样有责任 | non-hate | non-hate [END]"
        },
        {
            "id": 3007,
            "content": "他们就是典型的欺软怕硬的，又没自理能力的一族",
            "output": "他们 | 典型的欺软怕硬的，又没自理能力的一族 | Racism | hate [END]"
        },
        {
            "id": 7255,
            "content": "普通女性真的对黑人不感冒好吗？连巴基斯坦类似中亚人都不喜欢，因为真的觉得奇奇怪怪的。",
            "output": "普通女性 | 对黑人不感冒 | non-hate | non-hate [SEP] 巴基斯坦类似中亚人 | 奇奇怪怪的 | non-hate | non-hate [END]"
        }
    ]
    temp_train_val_json_file = "temp_test_train_val.json"
    with open(temp_train_val_json_file, 'w', encoding='utf-8') as f:
        json.dump(train_val_data_content, f, ensure_ascii=False, indent=2)

    # 2. 仅有 'id' 和 'content' 的预测数据
    pred_data_content = [
        {
            "id": 4169,
            "content": "飞周兄弟还不认亲？兄弟啊，还袭击？"
        },
        {
            "id": 4348,
            "content": "好久没上贴吧了，广黑吧没了吗？"
        },
        {
            "id": 4417,
            "content": "有这样的话？如果是这样的，那就是卫辉官媒的错误。但是，你是不是在造谣。请给出网址以供查阅。"
        }
    ]
    temp_pred_json_file = "temp_test_pred.json"
    with open(temp_pred_json_file, 'w', encoding='utf-8') as f:
        json.dump(pred_data_content, f, ensure_ascii=False, indent=2)

    print(f"使用的目标群体映射 (非预测模式): {GROUP_TO_ID}")
    print(f"使用的仇恨标签映射 (非预测模式): {HATEFUL_TO_ID}")
    print("=" * 40)
    print("开始测试非预测模式 (is_pred=False)")
    print("-" * 30)

    dataset_train_val = HateOriDataset(json_file_path=temp_train_val_json_file, is_pred=False)

    expected_count_train_val = len(train_val_data_content)
    assert len(dataset_train_val) == expected_count_train_val, f"非预测模式数据集大小错误: 期望 {expected_count_train_val}, 得到 {len(dataset_train_val)}"
    print(f"非预测模式数据集大小检查通过: {len(dataset_train_val)} 条")

    # 详细检查第一个带标签的样本 (ID: 7024)
    item_id_0, content_0, labels_0 = dataset_train_val[0] # 使用不同的变量名以避免覆盖
    assert item_id_0 == 7024
    assert content_0 == train_val_data_content[0]["content"]
    assert len(labels_0) == 3
    assert labels_0[0]["target_group_ids"] == [GROUP_TO_ID["non-hate"]]
    assert labels_0[0]["hateful_id"] == HATEFUL_TO_ID["non-hate"]
    print(f"样本 ID {item_id_0} (非预测模式) 详细检查通过。")
    # (可以添加更多对 train_val_data_content 的断言)

    print("\n迭代检查非预测模式样本结构和类型:")
    for i in range(len(dataset_train_val)):
        item_id, content, labels_list = dataset_train_val[i] # item_id, content, labels_list 会在循环中被重新赋值
        assert isinstance(item_id, int)
        assert isinstance(content, str)
        assert isinstance(labels_list, list)
        # 进一步检查 labels_list 内部结构
        for lbl_dict in labels_list:
            assert isinstance(lbl_dict, dict)
            assert "target_group_ids" in lbl_dict and isinstance(lbl_dict["target_group_ids"], list)
            assert "hateful_id" in lbl_dict and isinstance(lbl_dict["hateful_id"], int)
        # 为每个非预测模式样本打印通过信息
        print(f"  非预测模式样本 {i} (ID: {item_id}) 结构和类型检查通过。")
    print("非预测模式样本结构和类型迭代检查完成。")
    print("-" * 30)

    print("=" * 40)
    print("开始测试预测模式 (is_pred=True)")
    print("-" * 30)

    dataset_pred = HateOriDataset(json_file_path=temp_pred_json_file, is_pred=True)

    expected_count_pred = len(pred_data_content)
    assert len(dataset_pred) == expected_count_pred, f"预测模式数据集大小错误: 期望 {expected_count_pred}, 得到 {len(dataset_pred)}"
    print(f"预测模式数据集大小检查通过: {len(dataset_pred)} 条")

    # 检查预测模式下的样本
    print("\n迭代检查预测模式样本:") # 添加一个标题
    for i in range(len(dataset_pred)):
        item_id, content, labels = dataset_pred[i] # item_id, content, labels 会在循环中被重新赋值
        expected_item = pred_data_content[i]
        
        assert item_id == expected_item["id"], f"预测模式样本 {i} ID错误: 期望 {expected_item['id']}, 得到 {item_id}"
        assert content == expected_item["content"], f"预测模式样本 {i} content不匹配"
        assert labels == [], f"预测模式样本 {i} 标签错误: 期望 [], 得到 {labels}"
        print(f"  预测模式样本 {i} (ID: {item_id}) 检查通过 (ID, content 匹配, labels 为空)。")
    print("预测模式样本迭代检查完成。")
    print("-" * 30)

    # 清理临时文件
    if os.path.exists(temp_train_val_json_file):
        os.remove(temp_train_val_json_file)
    print(f"临时测试文件 {temp_train_val_json_file} 已删除。")
    if os.path.exists(temp_pred_json_file):
        os.remove(temp_pred_json_file)
    print(f"临时测试文件 {temp_pred_json_file} 已删除。")




if __name__ == "__main__":
    
    testOriDataset()
    print("\n所有测试断言均通过。HateOriDataset (包括预测模式) 工作符合预期！")