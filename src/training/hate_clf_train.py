# training/hate_clf_train.py
import torch
import os
import json
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import List, Dict

# 模块导入 (根据你的项目结构调整)
try:
    from data_process.preprocessor import HATEFUL_CATEGORIES, HATEFUL_TO_ID # 需要类别数量和ID到标签的映射
    from models.hate_clf_model import RobertaForHatefulClassification
    from data_process.hate_clf_dataset import HateClassificationDataset
    from training.trainer_base import GeneralTrainer
except ImportError as e:
    print(f"Import Error: {e}. Please check your PYTHONPATH and project structure.")
    try:
        from ..data_process.preprocessor import HATEFUL_CATEGORIES, HATEFUL_TO_ID
        from ..models.hate_clf_model import RobertaForHatefulClassification
        from ..data_process.hate_clf_dataset import HateClassificationDataset
        from .trainer_base import GeneralTrainer
    except ImportError:
        raise ImportError("Could not resolve imports for hate_clf_train.py.")


def compute_hateful_classification_metrics(
    list_of_batch_model_outputs_cpu: List[Dict[str, torch.Tensor]], 
    list_of_batch_inputs_cpu: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
    """
    计算二分类 (Hateful/Non-Hateful) 的评估指标。
    接收模型输出批次列表和输入批次列表。
    """
    print(f"DEBUG compute_metrics: Received {len(list_of_batch_model_outputs_cpu)} model output batches.")
    print(f"DEBUG compute_metrics: Received {len(list_of_batch_inputs_cpu)} input batches.")

    all_true_labels_flat = []
    all_predictions_flat = []

    for batch_outputs, batch_inputs in zip(list_of_batch_model_outputs_cpu, list_of_batch_inputs_cpu):
        logits = batch_outputs.get("logits") # 假设分类模型的logits键是 "logits"
        true_labels_batch = batch_inputs.get("labels") # 假设真实标签的键是 "labels"

        if logits is None or true_labels_batch is None:
            print("警告: compute_metrics 发现批次中缺少 logits 或 labels。")
            continue
        
        # logits 形状: (batch_size_this_batch, num_classes)
        # true_labels_batch 形状: (batch_size_this_batch)
        
        preds_batch = torch.argmax(logits, dim=-1) # (batch_size_this_batch)
        
        all_true_labels_flat.extend(true_labels_batch.numpy().tolist())
        all_predictions_flat.extend(preds_batch.numpy().tolist())

    if not all_true_labels_flat or not all_predictions_flat:
        print("DEBUG compute_metrics: No labels or predictions to evaluate after processing batches.")
        return {"accuracy": 0.0, "precision_hate": 0.0, "recall_hate": 0.0, "f1_hate": 0.0} # 返回默认值

    print(f"DEBUG compute_metrics: Total true labels collected: {len(all_true_labels_flat)}, Sample: {all_true_labels_flat[:10]}")
    print(f"DEBUG compute_metrics: Total predictions collected: {len(all_predictions_flat)}, Sample: {all_predictions_flat[:10]}")

    try:
        from data_process.preprocessor import HATEFUL_TO_ID # 尝试在函数内部导入，确保可用
        positive_label_id = HATEFUL_TO_ID.get("hate", 0)
    except ImportError:
        positive_label_id = 0 
        print(f"警告: compute_metrics 无法从preprocessor导入HATEFUL_TO_ID, 假设 'hate' 标签的ID为 {positive_label_id}")

    try:
        accuracy = accuracy_score(all_true_labels_flat, all_predictions_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_labels_flat, 
            all_predictions_flat, 
            average='binary', 
            pos_label=positive_label_id, 
            zero_division=0
        )
    except ValueError as ve: # 捕获可能的ValueError，例如如果标签不符合预期
        print(f"Error during sklearn metrics calculation: {ve}")
        print(f"Unique true labels: {set(all_true_labels_flat)}")
        print(f"Unique predictions: {set(all_predictions_flat)}")
        return {"accuracy": 0.0, "precision_hate_label_X": 0.0, "recall_hate_label_X": 0.0, "f1_hate_label_X": 0.0}


    return {
        "accuracy": accuracy,
        f"precision_hate_label_{positive_label_id}": precision,
        f"recall_hate_label_{positive_label_id}": recall,
        f"f1_hate_label_{positive_label_id}": f1,
    }


def train_hateful_classifier(args):
    """
    训练Hateful分类模型的主函数。
    """
    print("Starting Hateful Classification Model Training...")
    print(f"Arguments: {args}")

    # --- 1. 设置设备和随机种子 ---
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # --- 2. 准备数据集 ---
    print("Loading and preprocessing datasets for Hateful classification...")
    try:
        # 这里的 input_template 需要与 HateClassificationDataset 期望的一致
        # 你可以将其作为命令行参数，或者在此处硬编码一个经过验证的模板
        input_template_for_clf = args.input_template
        if not input_template_for_clf: # 如果命令行没提供，给个默认值
            input_template_for_clf = "[CLS] {target} [SEP] {argument} [SEP] {content} [SEP]"
            print(f"Using default input template: {input_template_for_clf}")

        train_dataset = HateClassificationDataset(
            original_json_path=args.train_file,
            tokenizer_name_or_path=args.model_name_or_path,
            max_seq_length=args.max_seq_length,
            input_template=input_template_for_clf
        )
        eval_dataset = HateClassificationDataset(
            original_json_path=args.validation_file,
            tokenizer_name_or_path=args.model_name_or_path,
            max_seq_length=args.max_seq_length,
            input_template=input_template_for_clf
        )
    except Exception as e:
        print(f"Error creating Hateful classification datasets: {e}")
        import traceback
        traceback.print_exc()
        return

    if not train_dataset or not len(train_dataset):
        print(f"Hateful training dataset at {args.train_file} is empty or could not be loaded. Aborting.")
        return
    if not eval_dataset or not len(eval_dataset):
        print(f"Hateful evaluation dataset at {args.validation_file} is empty. Consider creating one.")
        # return # 或者允许继续，但不进行评估

    print(f"Hateful Train dataset size: {len(train_dataset)} (Target-Argument pairs)")
    print(f"Hateful Evaluation dataset size: {len(eval_dataset)} (Target-Argument pairs)")

    # --- 3. 实例化模型 ---
    print(f"Loading pre-trained model from: {args.model_name_or_path} for Hateful classification")
    try:
        # num_labels 从 HATEFUL_CATEGORIES 动态获取
        model = RobertaForHatefulClassification(
            model_name_or_path=args.model_name_or_path,
            num_labels=len(HATEFUL_CATEGORIES), # 通常是2
            dropout_prob=args.dropout_prob
        )
    except Exception as e:
        print(f"Error instantiating Hateful classification model: {e}")
        return

    # --- 4. 实例化 Trainer ---
    output_dir_hate_clf = os.path.join(args.output_dir_root, "hate_clf")
    os.makedirs(output_dir_hate_clf, exist_ok=True)

    trainer = GeneralTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics_fn=compute_hateful_classification_metrics, # 传入二分类评估函数
        epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        device=device,
        output_dir=output_dir_hate_clf,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed
    )

    # --- 5. 开始训练 ---
    print("\n--- Starting Hateful Classification Model Training Phase ---")
    trainer.train()
    print("--- Hateful Classification Model Training Finished ---")

    # --- 6. 训练后在验证集上最终评估 (最佳模型已在训练中保存) ---
    best_model_path_on_eval = os.path.join(output_dir_hate_clf, "best_model_on_eval_loss")
    if os.path.exists(os.path.join(best_model_path_on_eval, "pytorch_model.bin")):
        print(f"\nLoading best Hateful model for final evaluation from: {best_model_path_on_eval}")
        final_eval_model = RobertaForHatefulClassification(
            model_name_or_path=args.model_name_or_path, # 基础模型路径
            num_labels=len(HATEFUL_CATEGORIES)
        )
        final_eval_model.load_state_dict(torch.load(os.path.join(best_model_path_on_eval, "pytorch_model.bin"), map_location=device))
        final_eval_model.to(device)
        
        evaluator_for_best_model = GeneralTrainer(
            model=final_eval_model,
            eval_dataset=eval_dataset,
            compute_metrics_fn=compute_hateful_classification_metrics,
            batch_size=args.batch_size,
            device=device,
            output_dir=output_dir_hate_clf
        )
        print("\n--- Final Evaluation on Validation Set (Best Hateful Model) ---")
        # 调整GeneralTrainer.evaluate以适应分类任务的指标收集
        # 假设 GeneralTrainer.evaluate 已经更新，或者 compute_metrics_fn 能处理其输出
        final_eval_metrics = evaluator_for_best_model.evaluate(description="Best Hateful Model on Validation Set")
        print(f"Final Validation Metrics (Best Hateful Model): {final_eval_metrics}")
        with open(os.path.join(output_dir_hate_clf, "best_model_eval_results.json"), "w", encoding="utf-8") as f:
            json.dump(final_eval_metrics, f, ensure_ascii=False, indent=4)
    else:
        print(f"Warning: Best Hateful model checkpoint not found at {best_model_path_on_eval}.")

    # --- 7. 在测试集上评估 (如果提供) ---
    if args.test_file:
        print("\nLoading and preprocessing test dataset for Hateful classification...")
        try:
            test_dataset = HateClassificationDataset(
                original_json_path=args.test_file,
                tokenizer_name_or_path=args.model_name_or_path,
                max_seq_length=args.max_seq_length,
                input_template=input_template_for_clf
            )
        except Exception as e:
            print(f"Error creating Hateful test dataset: {e}")
            test_dataset = None

        if test_dataset and len(test_dataset) > 0:
            print(f"Hateful Test dataset size: {len(test_dataset)}")
            if 'final_eval_model' in locals() and final_eval_model is not None:
                tester = GeneralTrainer(
                    model=final_eval_model,
                    eval_dataset=test_dataset,
                    compute_metrics_fn=compute_hateful_classification_metrics,
                    batch_size=args.batch_size,
                    device=device,
                    output_dir=output_dir_hate_clf
                )
                print("\n--- Evaluation on Test Set (Best Hateful Model) ---")
                test_metrics = tester.evaluate(description="Hateful Test Set Evaluation")
                print(f"Test Metrics (Best Hateful Model): {test_metrics}")
                with open(os.path.join(output_dir_hate_clf, "test_set_results.json"), "w", encoding="utf-8") as f:
                    json.dump(test_metrics, f, ensure_ascii=False, indent=4)
            else:
                print("Best Hateful model not available for test set evaluation.")
        elif args.test_file:
            print(f"Hateful test dataset at {args.test_file} is empty or could not be loaded.")

    print("Hateful classification training script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hateful Classification Model")

    # 数据路径参数 (与IE训练脚本类似，但用于Hateful分类)
    parser.add_argument("--train_file", type=str, default="data/segment/train.json", help="Path to the training data JSON file (for HateOriDataset).")
    parser.add_argument("--validation_file", type=str, default="data/segment/val.json", help="Path to the validation data JSON file (for HateOriDataset).")
    parser.add_argument("--test_file", type=str, default="data/segment/test.json", help="Optional path to the test data JSON file (for HateOriDataset).")

    # 模型和Tokenizer参数
    parser.add_argument("--model_name_or_path", type=str, default="models/chinese-roberta-wwm-ext-large", help="Path to pre-trained model or shortcut name to use as base.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum total input sequence length after tokenization for Hateful classification.")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout probability for the Hateful classification layer.")
    parser.add_argument("--input_template", type=str, default="[CLS] {target} [SEP] {argument} [SEP] {content} [SEP]", help="Template for constructing input text for Hateful classification. Use {target}, {argument}, {content}.")

    # 训练参数
    parser.add_argument("--output_dir_root", type=str, default="models/outputs", help="Root directory where model outputs will be saved.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup over warmup_steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit total amount of checkpoints.")
    parser.add_argument("--seed", type=int, default=114514, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU.")

    args = parser.parse_args()

    train_hateful_classifier(args)