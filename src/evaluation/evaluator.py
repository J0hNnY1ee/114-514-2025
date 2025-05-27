# src/evaluation/evaluator.py
import difflib,json
from typing import List, Dict, Tuple, Set

class QuadrupleEvaluator:
    def __init__(self, similarity_threshold: float = 0.5):
        """
        初始化四元组评估器。

        参数:
            similarity_threshold (float): 软匹配中 Target 和 Argument 字符串相似度的阈值。
        """
        self.similarity_threshold = similarity_threshold

    def _normalize_quadruple(self, quad: Dict[str, str]) -> Tuple[str, str, str, str]:
        """
        将四元组字典规范化为有序元组，以便比较和哈希。
        确保字段顺序一致：(Target, Argument, Targeted Group, Hateful)
        并将所有值转为字符串并去除首尾空格。
        """
        def _get_val(value):
            if value is None:
                return ""
            return str(value).strip()

        target = _get_val(quad.get("target"))
        argument = _get_val(quad.get("argument"))
        targeted_group = _get_val(quad.get("targeted_group"))
        hateful = _get_val(quad.get("hateful"))
        return (target, argument, targeted_group, hateful)

    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        计算两个字符串的相似度，严格按照公式: Similarity = M * 2 / (len(s1) + len(s2))
        其中 M 是 s1 和 s2 之间匹配的字符总数 (所有匹配块长度之和)。
        """
        len_s1 = len(s1)
        len_s2 = len(s2)

        if len_s1 == 0 and len_s2 == 0: # 两者都为空字符串
            return 1.0
        if len_s1 == 0 or len_s2 == 0: # 其中一个为空
            return 0.0

        # 使用 difflib.SequenceMatcher
        matcher = difflib.SequenceMatcher(None, s1, s2, autojunk=False) # autojunk=False 更精确

        # M 是所有匹配块的长度之和
        # get_matching_blocks() 返回 (i, j, n) 元组列表，最后一个是 (len(a), len(b), 0)
        # 我们需要对所有 n 求和
        M = 0
        for block in matcher.get_matching_blocks():
            # The last block is a dummy value (len(a), len(b), 0)
            # We are interested in the 'n' (size of the block)
            M += block.size # block.size 是匹配块的长度 n

        similarity = (M * 2.0) / (len_s1 + len_s2)
        return similarity

    def _is_hard_match(self, pred_quad: Tuple[str, str, str, str], gold_quad: Tuple[str, str, str, str]) -> bool:
        """
        判断两个规范化的四元组是否硬匹配。
        """
        return pred_quad == gold_quad

    def _is_soft_match(self, pred_quad: Tuple[str, str, str, str], gold_quad: Tuple[str, str, str, str]) -> bool:
        """
        判断两个规范化的四元组是否软匹配。
        """
        # 1. Targeted Group 和 Hateful 必须完全一致
        if pred_quad[2] != gold_quad[2] or pred_quad[3] != gold_quad[3]:
            return False

        # 2. Target 和 Argument 字符串匹配程度超过阈值
        target_similarity = self._string_similarity(pred_quad[0], gold_quad[0])
        argument_similarity = self._string_similarity(pred_quad[1], gold_quad[1])

        return target_similarity >= self.similarity_threshold and \
               argument_similarity >= self.similarity_threshold

    def _calculate_f1(self, precision: float, recall: float) -> float:
        """计算F1分数"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _evaluate_instance(self, pred_quads_norm: Set[Tuple[str, str, str, str]],
                           gold_quads_norm: Set[Tuple[str, str, str, str]],
                           match_type: str) -> Tuple[int, int, int]:
        """
        评估单个样本的预测。
        返回 (true_positives, false_positives, false_negatives)
        """
        tp = 0
        # fp = 0 # fp 和 fn 的计算将在下面根据匹配情况调整
        # fn = 0

        if match_type == "hard":
            tp = len(pred_quads_norm.intersection(gold_quads_norm))
            fp = len(pred_quads_norm) - tp # 所有预测中未匹配上的
            fn = len(gold_quads_norm) - tp # 所有真实答案中未匹配上的
            return tp, fp, fn

        # 对于软匹配，我们需要迭代，并确保一对一匹配
        # 将集合转换为列表以便使用索引进行跟踪
        list_pred_quads = list(pred_quads_norm)
        list_gold_quads = list(gold_quads_norm)

        num_preds = len(list_pred_quads)
        num_golds = len(list_gold_quads)

        # 跟踪哪些gold四元组已经被匹配，以确保一对一
        matched_gold_indices = [False] * num_golds

        for i in range(num_preds):
            pred_q = list_pred_quads[i]
            found_match_for_this_pred = False
            for j in range(num_golds):
                if not matched_gold_indices[j]: # 如果这个gold还没有被匹配
                    gold_q = list_gold_quads[j]
                    is_match = False
                    # _is_soft_match 内部会调用我们更新后的 _string_similarity
                    if self._is_soft_match(pred_q, gold_q): # match_type 肯定是 "soft"
                        is_match = True
                    
                    if is_match:
                        tp += 1
                        matched_gold_indices[j] = True # 标记这个gold已被匹配
                        found_match_for_this_pred = True
                        break # 当前pred_q找到了匹配，不再为它找其他gold_q
            
        
        fp = num_preds - tp
        fn = num_golds - tp 

        return tp, fp, fn


    def evaluate(self, predictions: List[List[Dict[str, str]]],
                 gold_standards: List[List[Dict[str, str]]]) -> Dict[str, Dict[str, float]]:
        """
        评估预测的四元组列表与标准答案的四元组列表。
        (其余部分与之前相同)
        """
        if len(predictions) != len(gold_standards):
            raise ValueError("预测列表和标准答案列表的长度必须一致 (样本数量应相同)。")

        total_tp_hard, total_fp_hard, total_fn_hard = 0, 0, 0
        total_tp_soft, total_fp_soft, total_fn_soft = 0, 0, 0

        for i in range(len(predictions)):
            pred_list_dicts = predictions[i]
            gold_list_dicts = gold_standards[i]

            pred_quads_norm_set = {self._normalize_quadruple(q) for q in pred_list_dicts}
            gold_quads_norm_set = {self._normalize_quadruple(q) for q in gold_list_dicts}
            
            # 硬匹配评估
            tp_hard, fp_hard, fn_hard = self._evaluate_instance(pred_quads_norm_set, gold_quads_norm_set, "hard")
            total_tp_hard += tp_hard
            total_fp_hard += fp_hard
            total_fn_hard += fn_hard
            
            # 软匹配评估
            tp_soft, fp_soft, fn_soft = self._evaluate_instance(pred_quads_norm_set, gold_quads_norm_set, "soft")
            total_tp_soft += tp_soft
            total_fp_soft += fp_soft
            total_fn_soft += fn_soft

        # --- 计算硬匹配的 Precision, Recall, F1 ---
        precision_hard = total_tp_hard / (total_tp_hard + total_fp_hard) if (total_tp_hard + total_fp_hard) > 0 else 0.0
        recall_hard = total_tp_hard / (total_tp_hard + total_fn_hard) if (total_tp_hard + total_fn_hard) > 0 else 0.0
        f1_hard = self._calculate_f1(precision_hard, recall_hard)

        # --- 计算软匹配的 Precision, Recall, F1 ---
        precision_soft = total_tp_soft / (total_tp_soft + total_fp_soft) if (total_tp_soft + total_fp_soft) > 0 else 0.0
        recall_soft = total_tp_soft / (total_tp_soft + total_fn_soft) if (total_tp_soft + total_fn_soft) > 0 else 0.0
        f1_soft = self._calculate_f1(precision_soft, recall_soft)

        average_f1 = (f1_hard + f1_soft) / 2.0

        results = {
            "hard_match": {
                "precision": precision_hard,
                "recall": recall_hard,
                "f1": f1_hard,
                "tp": total_tp_hard,
                "fp": total_fp_hard,
                "fn": total_fn_hard
            },
            "soft_match": {
                "precision": precision_soft,
                "recall": recall_soft,
                "f1": f1_soft,
                "tp": total_tp_soft,
                "fp": total_fp_soft,
                "fn": total_fn_soft
            },
            "average_f1": average_f1
        }
        return results

if __name__ == "__main__":
    evaluator = QuadrupleEvaluator(similarity_threshold=0.5)

    # 示例数据 (每个内部列表代表一个样本的四元组输出)
    predictions_data = [
        [ # 样本1的预测
            {"target": "苹果公司", "argument": "发布了新手机", "targeted_group": "科技爱好者", "hateful": "non-hate"},
            {"target": "天气", "argument": "真好", "targeted_group": "通用", "hateful": "non-hate"}
        ],
        [ # 样本2的预测
            {"target": "那部电影", "argument": "非常精彩", "targeted_group": "影迷", "hateful": "non-hate"}
        ],
        [ # 样本3的预测 (空的)
        ],
        [ # 样本4的预测 (软匹配测试)
            {"target": "服务员态度", "argument": "有点差", "targeted_group": "顾客", "hateful": "hate"}
        ]
    ]

    gold_standards_data = [
        [ # 样本1的标准答案
            {"target": "苹果公司", "argument": "发布了新款手机", "targeted_group": "科技爱好者", "hateful": "non-hate"}, # argument略有不同
            {"target": "天气", "argument": "真不错", "targeted_group": "通用", "hateful": "non-hate"} # argument略有不同
        ],
        [ # 样本2的标准答案
            {"target": "那部电影", "argument": "非常精彩", "targeted_group": "影迷", "hateful": "non-hate"} # 完全匹配
        ],
        [ # 样本3的标准答案
            {"target": "某个事件", "argument": "正在发生", "targeted_group": "公众", "hateful": "non-hate"} # 预测为空，FN
        ],
        [ # 样本4的标准答案 (软匹配测试)
            {"target": "服务员的态度", "argument": "很差劲", "targeted_group": "顾客", "hateful": "hate"}
        ]
    ]
    
    print("--- 测试1: 正常数据 ---")
    results = evaluator.evaluate(predictions_data, gold_standards_data)
    print(json.dumps(results, indent=4, ensure_ascii=False))

    print("\n--- 测试2: 包含完全匹配和完全不匹配 ---")
    predictions_data_2 = [
        [{"target": "A", "argument": "B", "targeted_group": "C", "hateful": "D"}] # 完全匹配
    ]
    gold_standards_data_2 = [
        [{"target": "A", "argument": "B", "targeted_group": "C", "hateful": "D"}]
    ]
    results_2 = evaluator.evaluate(predictions_data_2, gold_standards_data_2)
    print(json.dumps(results_2, indent=4, ensure_ascii=False))

    predictions_data_3 = [
        [{"target": "X", "argument": "Y", "targeted_group": "Z", "hateful": "W"}] # 完全不匹配
    ]
    gold_standards_data_3 = [
        [{"target": "A", "argument": "B", "targeted_group": "C", "hateful": "D"}]
    ]
    results_3 = evaluator.evaluate(predictions_data_3, gold_standards_data_3)
    print(json.dumps(results_3, indent=4, ensure_ascii=False))

    print("\n--- 测试3: 软匹配阈值 (使用新公式) ---")
    evaluator_strict_soft = QuadrupleEvaluator(similarity_threshold=0.9)
    # 注意：这里的字符串与之前 difflib.ratio() 的例子可能结果不同，因为M的计算方式明确了
    pred_soft_test = [[{"target": "apple pie", "argument": "is good", "targeted_group": "Food", "hateful": "no"}]]
    gold_soft_test = [[{"target": "apple  pie", "argument": "is very good", "targeted_group": "Food", "hateful": "no"}]]
    
    print("s1='apple pie', s2='apple  pie'")
    sim_t_new = evaluator_strict_soft._string_similarity("apple pie", "apple  pie")
    print(f"Target similarity (new formula): {sim_t_new}")
    # "apple pie" vs "apple  pie"
    # s1: a p p l e   p i e (len=9)
    # s2: a p p l e     p i e (len=10)
    # Matcher: [('a', 'a'), ('p', 'p'), ('p', 'p'), ('l', 'l'), ('e', 'e'), (' ', ' '), ('p', 'p'), ('i', 'i'), ('e', 'e')]
    # M (matching blocks): 'apple ', 'pie' -> 6 + 3 = 9
    # Sim = 2 * 9 / (9 + 10) = 18 / 19 = 0.947...

    print("s1='is good', s2='is very good'")
    sim_a_new = evaluator_strict_soft._string_similarity("is good", "is very good")
    print(f"Argument similarity (new formula): {sim_a_new}")
    # "is good" vs "is very good"
    # s1: i s   g o o d (len=7)
    # s2: i s   v e r y   g o o d (len=12)
    # Matcher: 'is ', ' good'
    # M: 3 + 4 = 7
    # Sim = 2 * 7 / (7 + 12) = 14 / 19 = 0.736...

    results_soft_strict = evaluator_strict_soft.evaluate(pred_soft_test, gold_soft_test)
    print("Soft match (threshold 0.9, new formula):") # argument sim 0.736 < 0.9, so no match
    print(json.dumps(results_soft_strict, indent=4, ensure_ascii=False))

    evaluator_loose_soft = QuadrupleEvaluator(similarity_threshold=0.7) # 阈值改为0.7
    results_soft_loose = evaluator_loose_soft.evaluate(pred_soft_test, gold_soft_test)
    print("Soft match (threshold 0.7, new formula):") # argument sim 0.736 > 0.7, so should match
    print(json.dumps(results_soft_loose, indent=4, ensure_ascii=False))

    print("\n--- 测试4: 处理空字符串和 NULL ---")
    pred_null_test = [[{"target": "NULL", "argument": "", "targeted_group": "TG", "hateful": "H"}]]
    gold_null_test = [[{"target": "", "argument": "NULL", "targeted_group": "TG", "hateful": "H"}]]
    results_null = evaluator.evaluate(pred_null_test, gold_null_test)
    print("NULL and empty string test:")
    print(json.dumps(results_null, indent=4, ensure_ascii=False))

    pred_both_empty = [[{"target": "", "argument": "", "targeted_group": "TG", "hateful": "H"}]]
    gold_both_empty = [[{"target": None, "argument": None, "targeted_group": "TG", "hateful": "H"}]]
    results_both_empty = evaluator.evaluate(pred_both_empty, gold_both_empty)
    print("Both empty/None test:")
    print(json.dumps(results_both_empty, indent=4, ensure_ascii=False))