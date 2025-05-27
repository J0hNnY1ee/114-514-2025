# preprocessor.py

# 定义目标群体类别列表
GROUP_CATEGORIES = ["Sexism", "Racism", "Region", "LGBTQ", "others", "non-hate"]

# 定义仇恨标签类别列表
HATEFUL_CATEGORIES = ["hate", "non-hate"]

# 创建从类别名称到索引的映射字典
GROUP_TO_ID = {name: i for i, name in enumerate(GROUP_CATEGORIES)}
ID_TO_GROUP = {i: name for i, name in enumerate(GROUP_CATEGORIES)}

HATEFUL_TO_ID = {name: i for i, name in enumerate(HATEFUL_CATEGORIES)}
ID_TO_HATEFUL = {i: name for i, name in enumerate(HATEFUL_CATEGORIES)}


def preprocess_target_group(group_str: str):
    """
    将目标群体字符串（可能包含逗号分隔的多个类别）转换为主类别索引列表。
    例如 "Sexism, Racism" -> [GROUP_TO_ID["Sexism"], GROUP_TO_ID["Racism"]]
    """
    groups = [g.strip() for g in group_str.split(",")]
    group_ids = []
    for group in groups:
        if group in GROUP_TO_ID:
            group_ids.append(GROUP_TO_ID[group])
        else:
            # 如果遇到未知的类别，可以抛出错误或记录日志
            print(f"警告: 在目标群体中发现未知类别 '{group}'，将忽略。")
            # 或者可以将其映射到一个特殊的 "UNK" 类别，如果定义了的话
    return group_ids


def preprocess_hateful_label(hateful_str: str):
    """
    将仇恨标签字符串转换为其对应的索引。
    """
    if hateful_str in HATEFUL_TO_ID:
        return HATEFUL_TO_ID[hateful_str]
    else:
        # 如果遇到未知的标签，可以抛出错误或记录日志
        print(f"警告: 发现未知仇恨标签 '{hateful_str}'，将忽略。")
        return None  # 或者返回一个默认值/抛出错误


GENERIC_IOB_LABELS = ["O", "B", "I"]
GENERIC_IOB_LABEL_TO_ID = {label: i for i, label in enumerate(GENERIC_IOB_LABELS)}
GENERIC_ID_TO_IOB_LABEL = {i: label for i, label in enumerate(GENERIC_IOB_LABELS)}

# --- 原先特定于单层IE的标签 (可选，如果不再使用可以移除或注释掉) ---
# IE_LABELS = ["O", "B-TGT", "I-TGT", "B-ARG", "I-ARG"]
# IE_LABEL_TO_ID = {label: i for i, label in enumerate(IE_LABELS)}
# IE_ID_TO_LABEL = {i: label for i, label in enumerate(IE_LABELS)}

# --- 特殊标签ID ---
IGNORE_INDEX = -100  # 用于在计算损失时忽略 [CLS], [SEP], [PAD] 等


if __name__ == "__main__":
    print("--- 目标群体映射 ---")
    print("类别:", GROUP_CATEGORIES)
    print("类别到ID:", GROUP_TO_ID)
    print("ID到类别:", ID_TO_GROUP)
    print("\n--- 仇恨标签映射 ---")
    print("类别:", HATEFUL_CATEGORIES)
    print("类别到ID:", HATEFUL_TO_ID)
    print("ID到类别:", ID_TO_HATEFUL)

    print("\n--- 预处理函数测试 ---")
    test_group_str1 = "Sexism, Racism"
    print(f"'{test_group_str1}' -> {preprocess_target_group(test_group_str1)}")
    test_group_str2 = "Region"
    print(f"'{test_group_str2}' -> {preprocess_target_group(test_group_str2)}")
    test_group_str3 = "non-hate"
    print(f"'{test_group_str3}' -> {preprocess_target_group(test_group_str3)}")
    test_group_str4 = "Unknown, Sexism"
    print(f"'{test_group_str4}' -> {preprocess_target_group(test_group_str4)}")

    test_hateful_str1 = "hate"
    print(f"'{test_hateful_str1}' -> {preprocess_hateful_label(test_hateful_str1)}")
    test_hateful_str2 = "non-hate"
    print(f"'{test_hateful_str2}' -> {preprocess_hateful_label(test_hateful_str2)}")
    test_hateful_str3 = "unknown_label"
    print(f"'{test_hateful_str3}' -> {preprocess_hateful_label(test_hateful_str3)}")
    print("--- 目标群体映射 ---")
    print("类别到ID:", GROUP_TO_ID)
    print("\n--- 仇恨标签映射 ---")
    print("类别到ID:", HATEFUL_TO_ID)
    print("\n--- IE 序列标注标签映射 ---")
    print("\n--- 通用IOB 序列标注标签映射 (用于多层IE) ---")
    print("通用IOB标签:", GENERIC_IOB_LABELS)
    print("通用IOB标签到ID:", GENERIC_IOB_LABEL_TO_ID)
    print("通用ID到IOB标签:", GENERIC_ID_TO_IOB_LABEL)
    print("\n--- 特殊忽略索引 ---")
    print("IGNORE_INDEX:", IGNORE_INDEX)
