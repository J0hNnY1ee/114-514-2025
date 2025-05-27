# training/ie_train.py
from typing import Dict, List
import torch
import os
import json
import argparse

# 导入修改后的模块
try:
    # 从 preprocessor 导入通用IOB定义
    from data_process.preprocessor import GENERIC_ID_TO_IOB_LABEL, GENERIC_IOB_LABELS, IGNORE_INDEX
    # 从 ie_model 导入新的多标签模型类
    from models.ie_model import RobertaForMultiLabelTokenClassification
    from data_process.ie_dataset import HateIEDataset # 这个应该已经是适配两层标签的版本了
    from training.trainer_base import GeneralTrainer # trainer_base 可能也需要调整evaluate
    SEQEVAL_AVAILABLE = True # 假设 seqeval 已经集成在 trainer_base 或这里
    if SEQEVAL_AVAILABLE:
        from seqeval.metrics import classification_report as seqeval_classification_report
        from seqeval.scheme import IOB2
except ImportError as e:
    print(f"Import Error in ie_train.py: {e}. Please check your PYTHONPATH and project structure.")
    # ... (备用导入，但最好修复主导入) ...
    # For now, let's assume imports will work for the rest of the example
    # Fallback for critical definitions if imports fail, to allow script to be parsed
    GENERIC_IOB_LABELS = ["O", "B", "I"]
    GENERIC_ID_TO_IOB_LABEL = {0:"O", 1:"B", 2:"I"}
    IGNORE_INDEX = -100
    class RobertaForMultiLabelTokenClassification: pass
    class HateIEDataset: pass
    class GeneralTrainer: pass
    SEQEVAL_AVAILABLE = False


# 修改后的评估函数
def compute_multilabel_ie_metrics(
    model_outputs_list: List[Dict], # Trainer.evaluate 收集的模型原始输出列表
    true_labels_batches_list: List[Dict], # Trainer.evaluate 收集的原始批次标签列表
    id_to_label_map: Dict
    ) -> Dict:
    """
    计算多标签序列标注任务的评估指标。
    model_outputs_list: 每个元素是模型forward返回的字典，包含 "logits_target", "logits_argument"
    true_labels_batches_list: 每个元素是dataloader的batch，包含 "labels_target", "labels_argument"
    """
    if not SEQEVAL_AVAILABLE:
        print("seqeval 不可用，跳过IE指标计算。")
        return {}

    all_true_tgt_sequences = []
    all_pred_tgt_sequences = []
    all_true_arg_sequences = []
    all_pred_arg_sequences = []

    for batch_outputs, batch_inputs in zip(model_outputs_list, true_labels_batches_list):
        logits_tgt = batch_outputs["logits_target"] # (batch, seq_len, num_iob_labels)
        logits_arg = batch_outputs["logits_argument"] # (batch, seq_len, num_iob_labels)
        
        labels_tgt_batch = batch_inputs["labels_target"] # (batch, seq_len)
        labels_arg_batch = batch_inputs["labels_argument"] # (batch, seq_len)
        attention_mask_batch = batch_inputs.get("attention_mask", torch.ones_like(labels_tgt_batch)) # (batch, seq_len)

        preds_tgt_batch = torch.argmax(logits_tgt, dim=-1)
        preds_arg_batch = torch.argmax(logits_arg, dim=-1)

        for i in range(labels_tgt_batch.size(0)): # Iterate over samples in batch
            # Target Layer
            true_seq_tgt = []
            pred_seq_tgt = []
            for j in range(labels_tgt_batch.size(1)): # Iterate over sequence
                if attention_mask_batch[i, j].item() == 1 and labels_tgt_batch[i, j].item() != IGNORE_INDEX:
                    true_seq_tgt.append(id_to_label_map.get(labels_tgt_batch[i, j].item(), "O"))
                    pred_seq_tgt.append(id_to_label_map.get(preds_tgt_batch[i, j].item(), "O"))
            if true_seq_tgt: # Only add if there are actual labels
                all_true_tgt_sequences.append(true_seq_tgt)
                all_pred_tgt_sequences.append(pred_seq_tgt)

            # Argument Layer
            true_seq_arg = []
            pred_seq_arg = []
            for j in range(labels_arg_batch.size(1)): # Iterate over sequence
                if attention_mask_batch[i, j].item() == 1 and labels_arg_batch[i, j].item() != IGNORE_INDEX:
                    true_seq_arg.append(id_to_label_map.get(labels_arg_batch[i, j].item(), "O"))
                    pred_seq_arg.append(id_to_label_map.get(preds_arg_batch[i, j].item(), "O"))
            if true_seq_arg:
                all_true_arg_sequences.append(true_seq_arg)
                all_pred_arg_sequences.append(pred_seq_arg)
    
    metrics = {}
    try:
        if all_true_tgt_sequences and all_pred_tgt_sequences:
            report_tgt = seqeval_classification_report(
                all_true_tgt_sequences, all_pred_tgt_sequences, output_dict=True, mode='strict', scheme=IOB2, zero_division=0
            )
            metrics["precision_tgt_micro"] = report_tgt["micro avg"]["precision"]
            metrics["recall_tgt_micro"] = report_tgt["micro avg"]["recall"]
            metrics["f1_tgt_micro"] = report_tgt["micro avg"]["f1-score"]
            if "TGT" in report_tgt : metrics["f1_TGT_entity"] = report_tgt["TGT"]["f1-score"] # if 'TGT' is used as entity type in IOB
            elif "B" in report_tgt : metrics["f1_TGT_entity_B"] = report_tgt["B"]["f1-score"] # if only B, I, O are used


        if all_true_arg_sequences and all_pred_arg_sequences:
            report_arg = seqeval_classification_report(
                all_true_arg_sequences, all_pred_arg_sequences, output_dict=True, mode='strict', scheme=IOB2, zero_division=0
            )
            metrics["precision_arg_micro"] = report_arg["micro avg"]["precision"]
            metrics["recall_arg_micro"] = report_arg["micro avg"]["recall"]
            metrics["f1_arg_micro"] = report_arg["micro avg"]["f1-score"]
            if "ARG" in report_arg : metrics["f1_ARG_entity"] = report_arg["ARG"]["f1-score"]
            elif "B" in report_arg : metrics["f1_ARG_entity_B"] = report_arg["B"]["f1-score"]

    except Exception as e:
        print(f"Error during seqeval metric calculation: {e}")
        print("Sample True TGT:", all_true_tgt_sequences[0] if all_true_tgt_sequences else "N/A")
        print("Sample Pred TGT:", all_pred_tgt_sequences[0] if all_pred_tgt_sequences else "N/A")
    
    # 计算一个总体的平均F1 (简单平均)
    f1_tgt = metrics.get("f1_tgt_micro", 0.0)
    f1_arg = metrics.get("f1_arg_micro", 0.0)
    if f1_tgt > 0 and f1_arg > 0:
        metrics["f1_overall_avg"] = (f1_tgt + f1_arg) / 2.0
    elif f1_tgt > 0:
        metrics["f1_overall_avg"] = f1_tgt
    elif f1_arg > 0:
        metrics["f1_overall_avg"] = f1_arg
    else:
        metrics["f1_overall_avg"] = 0.0
        
    return metrics


def train_information_extraction(args):
    print("Starting Information Extraction Model Training (Multi-Label IOB)...")
    print(f"Arguments: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print("Loading and preprocessing datasets...")
    try:
        train_dataset = HateIEDataset(
            original_json_path=args.train_file,
            tokenizer_name_or_path=args.model_name_or_path,
            max_seq_length=args.max_seq_length
        )
        eval_dataset = HateIEDataset(
            original_json_path=args.validation_file,
            tokenizer_name_or_path=args.model_name_or_path,
            max_seq_length=args.max_seq_length
        )
    except Exception as e:
        print(f"Error creating datasets: {e}")
        import traceback; traceback.print_exc(); return

    if not train_dataset or not len(train_dataset): # Check len as well
        print(f"Training dataset is empty. Aborting.")
        return
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset) if eval_dataset else 0}")


    print(f"Loading pre-trained model from: {args.model_name_or_path}")
    try:
        model = RobertaForMultiLabelTokenClassification( # 使用新的模型类
            model_name_or_path=args.model_name_or_path,
            num_single_layer_labels=len(GENERIC_IOB_LABELS), # 每个头输出 O,B,I
            dropout_prob=args.dropout_prob
        )
    except Exception as e:
        print(f"Error instantiating model: {e}")
        import traceback; traceback.print_exc(); return

    # 定义新的评估函数包装器
    def multilabel_ie_metrics_calculator_wrapper(model_outputs_list, true_labels_batches_list):
        return compute_multilabel_ie_metrics(
            model_outputs_list, 
            true_labels_batches_list, 
            GENERIC_ID_TO_IOB_LABEL # 使用通用的 O,B,I 映射
        )

    output_dir_ie = os.path.join(args.output_dir_root, "ie_multilabel") # 新的输出目录
    os.makedirs(output_dir_ie, exist_ok=True)

    trainer = GeneralTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics_fn=multilabel_ie_metrics_calculator_wrapper, # 使用新的评估函数
        epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        # ... 其他参数与之前类似 ...
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        device=device,
        output_dir=output_dir_ie,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed
    )

    print("\n--- Starting Multi-Label IE Model Training Phase ---")
    if trainer.train_dataloader is not None : # 确保有训练数据
        trainer.train()
        print("--- Multi-Label IE Model Training Finished ---")

        best_model_path_on_eval = os.path.join(output_dir_ie, "best_model_on_eval_loss")
        if os.path.exists(os.path.join(best_model_path_on_eval, "pytorch_model.bin")):
            print(f"\nLoading best model for final evaluation from: {best_model_path_on_eval}")
            final_eval_model = RobertaForMultiLabelTokenClassification(
                model_name_or_path=args.model_name_or_path,
                num_single_layer_labels=len(GENERIC_IOB_LABELS)
            )
            final_eval_model.load_state_dict(torch.load(os.path.join(best_model_path_on_eval, "pytorch_model.bin"), map_location=device))
            final_eval_model.to(device)
            
            evaluator_for_best_model = GeneralTrainer(
                model=final_eval_model,
                eval_dataset=eval_dataset,
                compute_metrics_fn=multilabel_ie_metrics_calculator_wrapper,
                batch_size=args.batch_size,
                device=device,
                output_dir=output_dir_ie
            )
            print("\n--- Final Evaluation on Validation Set (Best Multi-Label IE Model) ---")
            final_eval_metrics = evaluator_for_best_model.evaluate(description="Best Model on Validation Set")
            print(f"Final Validation Metrics (Best Model): {final_eval_metrics}")
            with open(os.path.join(output_dir_ie, "best_model_eval_results.json"), "w", encoding="utf-8") as f:
                json.dump(final_eval_metrics, f, ensure_ascii=False, indent=4)
        else:
            print(f"Warning: Best model checkpoint not found at {best_model_path_on_eval}.")

        if args.test_file:
            print("\nLoading and preprocessing test dataset...")
            try:
                test_dataset = HateIEDataset(
                    original_json_path=args.test_file,
                    tokenizer_name_or_path=args.model_name_or_path,
                    max_seq_length=args.max_seq_length
                )
                if test_dataset and len(test_dataset) > 0:
                    print(f"Test dataset size: {len(test_dataset)}")
                    if 'final_eval_model' in locals() and final_eval_model is not None:
                        tester = GeneralTrainer(model=final_eval_model, eval_dataset=test_dataset,
                                                compute_metrics_fn=multilabel_ie_metrics_calculator_wrapper,
                                                batch_size=args.batch_size, device=device, output_dir=output_dir_ie)
                        print("\n--- Evaluation on Test Set (Best Multi-Label IE Model) ---")
                        test_metrics = tester.evaluate(description="Test Set Evaluation")
                        print(f"Test Metrics (Best Model): {test_metrics}")
                        with open(os.path.join(output_dir_ie, "test_set_results.json"), "w", encoding="utf-8") as f:
                            json.dump(test_metrics, f, ensure_ascii=False, indent=4)
                    else: print("Best model not available for test set.")
                else: print(f"Test dataset at {args.test_file} is empty or could not be loaded.")
            except Exception as e: print(f"Error creating test dataset: {e}")
    else:
        print("No training data provided, skipping training.")

    print("Multi-Label Information Extraction training script finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Label Information Extraction Model")
    # ... (argparse 参数定义与之前类似，保持不变) ...
    # 数据路径参数
    parser.add_argument("--train_file", type=str, default="data/segment/train.json", help="Path to the training data JSON file.")
    parser.add_argument("--validation_file", type=str, default="data/segment/val.json", help="Path to the validation data JSON file.")
    parser.add_argument("--test_file", type=str, default="data/segment/test.json", help="Optional path to the test data JSON file.")
    # 模型和Tokenizer参数
    parser.add_argument("--model_name_or_path", type=str, default="models/chinese-roberta-wwm-ext-large", help="Path to pre-trained model or shortcut name.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum total input sequence length after tokenization.")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout probability for the classification layer.")
    # 训练参数
    parser.add_argument("--output_dir_root", type=str, default="models/outputs", help="Root directory where model outputs (checkpoints, results) will be saved.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate for AdamW.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup over warmup_steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit the total amount of checkpoints. Deletes the older checkpoints.")
    parser.add_argument("--seed", type=int, default=114514, help="Random seed for initialization.")
    parser.add_argument("--cpu", action="store_true", help="Force using CPU even if CUDA is available.")

    args = parser.parse_args()
    train_information_extraction(args)