# training/group_clf_train.py
import torch
import os
import json
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import List, Dict

# 模块导入
try:
    from data_process.preprocessor import GROUP_CATEGORIES, GROUP_TO_ID, ID_TO_GROUP # 需要类别和映射
    from model.group_clf_model import RobertaForTargetedGroupClassification
    from data_process.group_clf_dataset import TargetedGroupClassificationDataset
    from training.trainer_base import GeneralTrainer
except ImportError as e:
    print(f"Import Error in group_clf_train.py: {e}. Check PYTHONPATH and structure.")
    try:
        from ..data_process.preprocessor import GROUP_CATEGORIES, GROUP_TO_ID, ID_TO_GROUP
        from ..model.group_clf_model import RobertaForTargetedGroupClassification
        from ..data_process.group_clf_dataset import TargetedGroupClassificationDataset
        from .trainer_base import GeneralTrainer
    except ImportError:
        raise ImportError("Could not resolve imports for group_clf_train.py.")


def compute_group_classification_metrics(
    list_of_batch_model_outputs_cpu: List[Dict[str, torch.Tensor]], 
    list_of_batch_inputs_cpu: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
    """
    计算Targeted Group多分类任务的评估指标。
    """
    all_true_labels_flat = []
    all_predictions_flat = []

    for batch_outputs, batch_inputs in zip(list_of_batch_model_outputs_cpu, list_of_batch_inputs_cpu):
        logits = batch_outputs.get("logits")
        true_labels_batch = batch_inputs.get("labels")

        if logits is None or true_labels_batch is None:
            print("警告: compute_group_metrics 发现批次中缺少 logits 或 labels。")
            continue
        
        preds_batch = torch.argmax(logits, dim=-1)
        
        all_true_labels_flat.extend(true_labels_batch.numpy().tolist())
        all_predictions_flat.extend(preds_batch.numpy().tolist())

    if not all_true_labels_flat or not all_predictions_flat:
        print("DEBUG compute_group_metrics: No labels or predictions to evaluate.")
        return {"accuracy": 0.0, "f1_micro": 0.0, "f1_macro": 0.0}

    # 使用全局导入的 GROUP_CATEGORIES 来获取 target_names
    # target_names = GROUP_CATEGORIES # 这应该包含了 "non-hate"
    # 或者，如果只想评估实际的5个仇恨目标群体+non-hate，确保列表正确
    # 为了安全，我们从 ID_TO_GROUP 生成，确保顺序和ID对应
    
    # 获取实际参与评估的标签ID范围，以生成正确的target_names for classification_report
    # unique_labels_in_data = sorted(list(set(all_true_labels_flat) | set(all_predictions_flat)))
    # report_target_names = [ID_TO_GROUP.get(lbl_id, f"UnknownID-{lbl_id}") for lbl_id in unique_labels_in_data]
    # 如果标签ID总是从0开始连续的，可以直接用：
    report_target_names = [ID_TO_GROUP.get(i, f"ID-{i}") for i in range(len(GROUP_CATEGORIES))]


    accuracy = accuracy_score(all_true_labels_flat, all_predictions_flat)
    f1_micro = f1_score(all_true_labels_flat, all_predictions_flat, average='micro', zero_division=0)
    f1_macro = f1_score(all_true_labels_flat, all_predictions_flat, average='macro', zero_division=0)
    f1_weighted = f1_score(all_true_labels_flat, all_predictions_flat, average='weighted', zero_division=0)
    
    print("\nClassification Report (Targeted Group):")
    # try-except 以防标签不一致导致 classification_report 报错
    try:
        # Ensure labels parameter is set correctly for scikit-learn's report
        # It should be the list of all possible label indices for your task
        possible_labels = sorted(list(GROUP_TO_ID.values()))
        report_str = classification_report(
            all_true_labels_flat, 
            all_predictions_flat, 
            labels=possible_labels, # 显式提供所有可能的标签ID
            target_names=report_target_names, 
            digits=4,
            zero_division=0
        )
        print(report_str)
    except Exception as report_e:
        print(f"Error generating classification_report: {report_e}")
        print(f"Unique true labels: {set(all_true_labels_flat)}")
        print(f"Unique pred labels: {set(all_predictions_flat)}")


    return {
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }


def train_targeted_group_classifier(args):
    """
    训练Targeted Group分类模型的主函数。
    """
    print("Starting Targeted Group Classification Model Training...")
    print(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print("Loading and preprocessing datasets for Targeted Group classification...")
    try:
        input_template_for_clf = args.input_template
        if not input_template_for_clf:
            input_template_for_clf = "[CLS] T: {target} A: {argument} IsHate: {hateful_label} C: {content} [SEP]"
            print(f"Using default input template for TG classification: {input_template_for_clf}")

        train_dataset = TargetedGroupClassificationDataset(
            original_json_path=args.train_file,
            tokenizer_name_or_path=args.model_name_or_path,
            max_seq_length=args.max_seq_length,
            input_template=input_template_for_clf
        )
        eval_dataset = TargetedGroupClassificationDataset(
            original_json_path=args.validation_file,
            tokenizer_name_or_path=args.model_name_or_path,
            max_seq_length=args.max_seq_length,
            input_template=input_template_for_clf
        )
    except Exception as e:
        print(f"Error creating Targeted Group classification datasets: {e}")
        import traceback; traceback.print_exc(); return

    if not train_dataset or not len(train_dataset):
        print(f"TG training dataset at {args.train_file} is empty. Aborting.")
        return
    print(f"TG Train dataset size: {len(train_dataset)} (Target-Argument-HatefulLabel entries)")
    print(f"TG Evaluation dataset size: {len(eval_dataset) if eval_dataset else 0}")

    print(f"Loading pre-trained model from: {args.model_name_or_path} for TG classification")
    try:
        model = RobertaForTargetedGroupClassification(
            model_name_or_path=args.model_name_or_path,
            num_labels=len(GROUP_CATEGORIES), # 使用GROUP_CATEGORIES的长度
            dropout_prob=args.dropout_prob
        )
    except Exception as e:
        print(f"Error instantiating TG classification model: {e}")
        return

    output_dir_group_clf = os.path.join(args.output_dir_root, "group_clf")
    os.makedirs(output_dir_group_clf, exist_ok=True)

    trainer = GeneralTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics_fn=compute_group_classification_metrics, # 新的评估函数
        epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        # ... 其他参数与 hate_clf_train.py 类似 ...
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        device=device,
        output_dir=output_dir_group_clf, # 指定输出目录
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed
    )

    print("\n--- Starting Targeted Group Classification Model Training Phase ---")
    if trainer.train_dataloader:
        trainer.train()
        print("--- Targeted Group Classification Model Training Finished ---")

        best_model_path = os.path.join(output_dir_group_clf, "best_model_on_eval_loss")
        if os.path.exists(os.path.join(best_model_path, "pytorch_model.bin")):
            print(f"\nLoading best TG model for final evaluation from: {best_model_path}")
            final_eval_model = RobertaForTargetedGroupClassification(
                model_name_or_path=args.model_name_or_path,
                num_labels=len(GROUP_CATEGORIES)
            )
            final_eval_model.load_state_dict(torch.load(os.path.join(best_model_path, "pytorch_model.bin"), map_location=device))
            final_eval_model.to(device)
            
            evaluator = GeneralTrainer(
                model=final_eval_model, eval_dataset=eval_dataset,
                compute_metrics_fn=compute_group_classification_metrics,
                batch_size=args.batch_size, device=device, output_dir=output_dir_group_clf
            )
            print("\n--- Final Evaluation on Validation Set (Best TG Model) ---")
            final_metrics = evaluator.evaluate(description="Best TG Model on Validation Set")
            print(f"Final Validation Metrics (Best TG Model): {final_metrics}")
            with open(os.path.join(output_dir_group_clf, "best_model_eval_results.json"), "w", encoding="utf-8") as f:
                json.dump(final_metrics, f, ensure_ascii=False, indent=4)
        else:
            print(f"Warning: Best TG model checkpoint not found at {best_model_path}.")

        if args.test_file:
            print("\nLoading and preprocessing test dataset for TG classification...")
            try:
                test_dataset = TargetedGroupClassificationDataset(
                    original_json_path=args.test_file, tokenizer_name_or_path=args.model_name_or_path,
                    max_seq_length=args.max_seq_length, input_template=input_template_for_clf
                )
                if test_dataset and len(test_dataset) > 0:
                    print(f"TG Test dataset size: {len(test_dataset)}")
                    if 'final_eval_model' in locals() and final_eval_model is not None:
                        tester = GeneralTrainer(
                            model=final_eval_model, eval_dataset=test_dataset,
                            compute_metrics_fn=compute_group_classification_metrics,
                            batch_size=args.batch_size, device=device, output_dir=output_dir_group_clf
                        )
                        print("\n--- Evaluation on Test Set (Best TG Model) ---")
                        test_metrics = tester.evaluate(description="TG Test Set Evaluation")
                        print(f"Test Metrics (Best TG Model): {test_metrics}")
                        with open(os.path.join(output_dir_group_clf, "test_set_results.json"), "w", encoding="utf-8") as f:
                            json.dump(test_metrics, f, ensure_ascii=False, indent=4)
                    else: print("Best TG model not available for test set.")
                else: print(f"TG test dataset at {args.test_file} is empty or could not be loaded.")
            except Exception as e: print(f"Error creating TG test dataset: {e}")
    else:
        print("No training data for TG classification.")

    print("Targeted Group classification training script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Targeted Group Classification Model")
    # 数据路径参数 (与Hateful分类脚本类似)
    parser.add_argument("--train_file", type=str, default="data/segment/train.json", help="Path to the training data JSON file (for HateOriDataset).")
    parser.add_argument("--validation_file", type=str, default="data/segment/val.json", help="Path to the validation data JSON file (for HateOriDataset).")
    parser.add_argument("--test_file", type=str, default="data/segment/test.json", help="Optional path to the test data JSON file (for HateOriDataset).")

    # 模型和Tokenizer参数
    parser.add_argument("--model_name_or_path", type=str, default="models/chinese-roberta-wwm-ext-large", help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Max sequence length for TG classification.")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout probability for TG classification layer.")
    parser.add_argument("--input_template", type=str, 
                        default="[CLS] Target: {target} Argument: {argument} Hatefulness: {hateful_label} Context: {content} [SEP]", 
                        help="Template for constructing input text for TG classification. Use {target}, {argument}, {hateful_label}, {content}.")

    # 训练参数
    parser.add_argument("--output_dir_root", type=str, default="models/outputs", help="Root directory for model outputs.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    # ... (其他与hate_clf_train.py相同的训练参数)
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup over warmup_steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--logging_steps", type=int, default=20, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps (default: only at end).")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Limit total amount of checkpoints (only best and final).") # 通常只保存最佳
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU.")


    args = parser.parse_args()
    train_targeted_group_classifier(args)