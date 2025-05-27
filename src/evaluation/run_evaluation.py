import json
from typing import List, Dict
import os 

try:
    from evaluation.evaluator import QuadrupleEvaluator
except ImportError:
    from .evaluator import QuadrupleEvaluator # Fallback for direct run

def main():
    PAIRED_DATA_FILE = "data/outputs/evaluation_paired_data.jsonl" 
    SIMILARITY_THRESHOLD = 0.5

    if not os.path.exists(PAIRED_DATA_FILE):
        print(f"错误: 配对评估文件 '{PAIRED_DATA_FILE}' 不存在。")
        print("请先运行 'prepare_eval_data.py' (或类似的脚本) 来生成此文件。")
        return

    print(f"开始从配对文件 '{PAIRED_DATA_FILE}' 加载数据进行评估...")
    
    predictions_for_eval: List[List[Dict]] = []
    gold_standards_for_eval: List[List[Dict]] = []
    processed_ids = [] # 可选，用于跟踪
    
    try:
        with open(PAIRED_DATA_FILE, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    record = json.loads(line.strip())
                    
                    preds = record.get("predicted_quads")
                    golds = record.get("gold_quads")
                    record_id = record.get("id", f"line_{line_num+1}")

                    if preds is None or golds is None: # 确保两者都存在
                        print(f"警告: 配对文件行 {line_num+1} (ID: {record_id}) "
                              f"缺少 'predicted_quads' 或 'gold_quads'。跳过。")
                        continue 
                    
                    # QuadrupleEvaluator 期望的四元组已经是正确的键名，因为
                    # prepare_eval_data.py 中的 load_predictions 和 load_gold_standards 已经处理了
                    predictions_for_eval.append(preds)
                    gold_standards_for_eval.append(golds)
                    processed_ids.append(record_id)

                except json.JSONDecodeError:
                    print(f"警告: 无法解析JSON行 (配对文件 '{PAIRED_DATA_FILE}', 行 {line_num+1}): {line.strip()}. 跳过。")
    except Exception as e:
        print(f"读取配对评估文件 '{PAIRED_DATA_FILE}' 时发生错误: {e}")
        return

    if not predictions_for_eval or not gold_standards_for_eval:
        print("未能从配对文件中加载有效数据进行评估。")
        return
    
    print(f"加载了 {len(predictions_for_eval)} 个样本进行评估。")
    
    evaluator = QuadrupleEvaluator(similarity_threshold=SIMILARITY_THRESHOLD)
    
    print("\n开始计算评估指标...")
    try:
        results = evaluator.evaluate(predictions_for_eval, gold_standards_for_eval)
        
        print("\n--- 评估结果 ---")
        print(json.dumps(results, indent=4, ensure_ascii=False))

        print("\n详细指标:")
        print(f"  硬匹配 F1: {results['hard_match']['f1']:.4f} "
              f"(P: {results['hard_match']['precision']:.4f}, R: {results['hard_match']['recall']:.4f}) "
              f"[TP: {results['hard_match']['tp']}, FP: {results['hard_match']['fp']}, FN: {results['hard_match']['fn']}]")
        print(f"  软匹配 F1: {results['soft_match']['f1']:.4f} "
              f"(P: {results['soft_match']['precision']:.4f}, R: {results['soft_match']['recall']:.4f}) "
              f"[TP: {results['soft_match']['tp']}, FP: {results['soft_match']['fp']}, FN: {results['soft_match']['fn']}]")
        print(f"  平均 F1 (硬+软)/2: {results['average_f1']:.4f}")

    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n评估脚本执行完毕。")

if __name__ == "__main__":
    main()