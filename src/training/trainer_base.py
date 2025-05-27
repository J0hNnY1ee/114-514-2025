# trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import time  # for timing
import os
import json
from typing import Optional, Dict, List, Callable


try:
    from seqeval.metrics import classification_report as seqeval_classification_report
    from seqeval.scheme import IOB2  # 或者其他你使用的标注方案

    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False
    print(
        "警告: seqeval 库未安装，序列标注任务的实体级别评估将不可用。请运行: pip install seqeval"
    )


class GeneralTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        compute_metrics_fn: Optional[Callable] = None,  # 用于评估的函数
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,  # 梯度裁剪
        device: Optional[torch.device] = None,
        output_dir: str = "./results",
        logging_steps: int = 50,
        save_steps: int = 500,  # 每多少步保存一次checkpoint
        save_total_limit: Optional[int] = None,  # 最多保存多少个checkpoint
        seed: int = 42,
    ):
        """
        通用的训练器类。

        参数:
            model (nn.Module): 要训练的模型。
            train_dataset (Dataset, optional): 训练数据集。
            eval_dataset (Dataset, optional): 评估数据集。
            optimizer (torch.optim.Optimizer, optional): 优化器。如果为None，则会创建AdamW。
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): 学习率调度器。如果为None，则会创建线性预热调度器。
            compute_metrics_fn (Callable, optional): 一个函数，接收模型的预测和真实标签，返回评估指标字典。
            epochs (int): 训练的总轮数。
            batch_size (int): 训练和评估的批处理大小。
            learning_rate (float): 优化器的初始学习率。
            weight_decay (float): AdamW优化器的权重衰减。
            warmup_steps (int): 学习率预热的步数。
            gradient_accumulation_steps (int): 梯度累积的步数。
            max_grad_norm (float): 梯度裁剪的最大范数。
            device (torch.device, optional): 训练设备 (例如 torch.device("cuda") 或 torch.device("cpu"))。
                                          如果为None，则自动检测CUDA。
            output_dir (str): 保存模型checkpoint和结果的目录。
            logging_steps (int): 每隔多少步打印一次日志。
            save_steps (int): 每隔多少步保存一次模型checkpoint。
            save_total_limit (int, optional): 最多保存的checkpoint数量。如果为None，则保存所有。
            seed (int): 随机种子，用于可复现性。
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics_fn = compute_metrics_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.seed = seed

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # 设置随机种子
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            # np.random.seed(self.seed) # 如果使用了numpy
            # random.seed(self.seed)   # 如果使用了random

        if self.train_dataset:
            self.train_dataloader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )
            num_training_steps_per_epoch = (
                len(self.train_dataloader) // self.gradient_accumulation_steps
            )
            self.num_training_steps = num_training_steps_per_epoch * self.epochs
        else:
            self.train_dataloader = None
            self.num_training_steps = 0

        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset, batch_size=self.batch_size, shuffle=False
            )
        else:
            self.eval_dataloader = None

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if self.train_dataloader:  # 只有在训练时才需要优化器和调度器
            if self.optimizer is None:
                self.optimizer = self._create_optimizer()
            if self.lr_scheduler is None:
                self.lr_scheduler = self._create_scheduler()

        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = None  # 用于保存最佳模型的指标，例如 eval_loss 或某个 F1 分数

        os.makedirs(self.output_dir, exist_ok=True)
        print(
            f"Trainer initialized. Device: {self.device}. Output dir: {self.output_dir}"
        )

    def _create_optimizer(self):
        # 为AdamW准备参数组，以便对偏置和LayerNorm权重应用不同的权重衰减
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-8)

    def _create_scheduler(self):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_training_steps,
        )

    def _save_model(self, checkpoint_name: str = "best_model"):
        """保存模型和相关状态"""
        path = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(path, "pytorch_model.bin"))
        # 可以选择保存tokenizer和config（如果它们是从Hugging Face加载的）
        # self.model.config.save_pretrained(path)
        # tokenizer.save_pretrained(path) # 如果tokenizer是trainer的一部分
        print(f"Model saved to {path}")

        # 管理checkpoint数量
        if self.save_total_limit is not None and self.save_total_limit > 0:
            checkpoints = sorted(
                [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1]),
            )
            if len(checkpoints) > self.save_total_limit:
                for old_checkpoint in checkpoints[: -self.save_total_limit]:
                    old_checkpoint_path = os.path.join(self.output_dir, old_checkpoint)
                    # shutil.rmtree(old_checkpoint_path) # 更安全地删除目录
                    print(f"Deleting old checkpoint: {old_checkpoint_path}")
                    # 简单的 os.rmdir 可能不工作如果目录非空，需要递归删除

    def _load_model(self, checkpoint_path: str):
        """从checkpoint加载模型"""
        model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {checkpoint_path}")
        else:
            print(f"Warning: No model found at {checkpoint_path}")

    def train_epoch(self):
        """执行一个完整的训练轮次"""
        self.model.train()  # 设置模型为训练模式
        epoch_loss = 0
        epoch_start_time = time.time()

        for step, batch in enumerate(self.train_dataloader):
            # 将批次数据移动到设备
            inputs = {k: v.to(self.device) for k, v in batch.items()}

            # 模型前向传播
            outputs = self.model(
                **inputs, return_dict=True
            )  # 确保模型返回字典且包含loss
            loss = outputs["loss"]

            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            loss.backward()  # 反向传播计算梯度

            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(
                self.train_dataloader
            ):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )  # 梯度裁剪
                self.optimizer.step()  # 更新参数
                self.lr_scheduler.step()  # 更新学习率
                self.optimizer.zero_grad()  # 清空梯度
                self.global_step += 1

                if self.global_step % self.logging_steps == 0:
                    elapsed_time = time.time() - epoch_start_time
                    print(
                        f"Epoch: {self.current_epoch + 1}/{self.epochs}, Step: {self.global_step}/{self.num_training_steps}, "
                        f"LR: {self.lr_scheduler.get_last_lr()[0]:.2e}, Loss: {loss.item() * self.gradient_accumulation_steps:.4f}, "
                        f"Time: {elapsed_time:.2f}s"
                    )

                if self.global_step % self.save_steps == 0:
                    self._save_model(checkpoint_name=f"checkpoint-{self.global_step}")

            epoch_loss += loss.item() * self.gradient_accumulation_steps

        avg_epoch_loss = epoch_loss / len(self.train_dataloader)
        epoch_duration = time.time() - epoch_start_time
        print(
            f"End of Epoch {self.current_epoch + 1}. Avg Loss: {avg_epoch_loss:.4f}. Duration: {epoch_duration:.2f}s"
        )
        return avg_epoch_loss

    def evaluate(
        self, dataloader: Optional[DataLoader] = None, description: str = "Evaluation"
    ) -> Dict:
        if dataloader is None and self.eval_dataloader is None:
            print("No evaluation dataloader provided.")
            return {}

        eval_dataloader_to_use = dataloader if dataloader else self.eval_dataloader
        self.model.eval()
        total_eval_loss = 0

        # 存储每个批次的模型输出和输入，以便后续传递给 compute_metrics_fn
        list_of_batch_model_outputs_cpu = [] 
        list_of_batch_inputs_cpu = [] # 包含真实标签和attention_mask

        print(f"\n--- Running {description} ---")
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_dataloader_to_use):
                inputs_on_device = {k: v.to(self.device) for k, v in batch.items()}
                
                # 模型返回一个字典，例如: {"loss": ..., "logits_target": ..., "logits_argument": ...}
                model_outputs_dict = self.model(**inputs_on_device, return_dict=True) 
                
                # 检查是否有任何logits输出，适应单logits和多logits的情况
                has_logits = model_outputs_dict.get("logits") is not None or \
                             model_outputs_dict.get("logits_target") is not None or \
                             model_outputs_dict.get("logits_argument") is not None
                
                if not has_logits:
                    print(f"警告: 批次 {batch_idx} 的模型输出中没有有效的 'logits'。跳过此批次的指标计算。")
                    continue

                if model_outputs_dict.get("loss") is not None:
                    total_eval_loss += model_outputs_dict["loss"].item()
                
                # 将整个模型输出字典和输入批次（包含真实标签）移到CPU并存储
                list_of_batch_model_outputs_cpu.append(
                    {k: v.cpu() for k, v in model_outputs_dict.items() if isinstance(v, torch.Tensor)}
                )
                list_of_batch_inputs_cpu.append(
                    {k: v.cpu() for k, v in batch.items() if isinstance(v, torch.Tensor) or k == "input_ids"} # 保留input_ids
                )

        avg_eval_loss = (
            total_eval_loss / len(eval_dataloader_to_use)
            if len(eval_dataloader_to_use) > 0
            else 0.0
        )
        eval_duration = time.time() - start_time
        print(
            f"{description} completed in {eval_duration:.2f}s. Average Loss: {avg_eval_loss:.4f}"
        )

        metrics = {"eval_loss": avg_eval_loss}
        
        # 将收集到的批次列表直接传递给 compute_metrics_fn
        # compute_metrics_fn 需要负责从这些列表中提取和处理数据
        if self.compute_metrics_fn and list_of_batch_inputs_cpu and list_of_batch_model_outputs_cpu:
            # 确保 list_of_batch_inputs_cpu 至少包含一个带有标签的批次
            has_labels_in_inputs = any(
                ("labels" in batch_input or \
                 "labels_target" in batch_input or \
                 "labels_argument" in batch_input) 
                for batch_input in list_of_batch_inputs_cpu
            )

            if has_labels_in_inputs:
                eval_metrics_results = self.compute_metrics_fn(
                    list_of_batch_model_outputs_cpu, 
                    list_of_batch_inputs_cpu
                )
                if eval_metrics_results and isinstance(eval_metrics_results, dict):
                    metrics.update(eval_metrics_results)
                    print(f"{description} Metrics: {eval_metrics_results}")
                elif eval_metrics_results:
                    print(f"警告: compute_metrics_fn 返回了非字典类型的结果: {type(eval_metrics_results)}")
            else:
                print(f"{description}: 未在输入中找到标签，跳过指标计算。")
        
        return metrics


    def train(self):
        """完整的训练过程"""
        if not self.train_dataloader:
            print("No training dataloader provided. Cannot start training.")
            return

        print(f"***** Starting Training *****")
        print(f"  Num epochs = {self.epochs}")
        print(f"  Batch size = {self.batch_size}")
        print(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {self.num_training_steps}")
        print(f"  Warmup steps = {self.warmup_steps}")
        print(f"  Learning rate = {self.learning_rate}")

        overall_best_metric_val = None  # 用于早停或保存最佳

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            print(f"\n===== Epoch {epoch + 1}/{self.epochs} =====")
            avg_train_loss = self.train_epoch()

            if self.eval_dataloader:
                eval_metrics = self.evaluate(
                    description=f"Evaluation after Epoch {epoch + 1}"
                )

                # 简单的基于 eval_loss 保存最佳模型
                current_eval_loss = eval_metrics.get("eval_loss", float("inf"))
                if (
                    overall_best_metric_val is None
                    or current_eval_loss < overall_best_metric_val
                ):
                    overall_best_metric_val = current_eval_loss
                    print(
                        f"New best model found with eval_loss: {overall_best_metric_val:.4f}. Saving..."
                    )
                    self._save_model(checkpoint_name="best_model_on_eval_loss")

                # 你可以根据 compute_metrics_fn 返回的特定指标来保存最佳模型
                # 例如, if eval_metrics.get("f1", 0) > overall_best_f1: ...

        print("Training finished.")
        # 训练结束后可以保存最终模型
        self._save_model(checkpoint_name="final_model")

    def predict(self, test_dataset: Dataset) -> List:
        """在测试数据集上进行预测"""
        if not test_dataset:
            print("No test dataset provided for prediction.")
            return []

        test_dataloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.model.eval()
        all_predictions_ids = []
        print("\n--- Running Prediction ---")
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = {
                    k: v.to(self.device) for k, v in batch.items() if k != "labels"
                }  # 预测时通常没有labels
                outputs = self.model(**inputs, return_dict=True)
                logits = outputs["logits"]
                predictions = torch.argmax(logits, dim=-1)
                all_predictions_ids.extend(predictions.cpu().numpy().tolist())

        print("Prediction finished.")
        return all_predictions_ids


# --- 辅助函数：用于IE任务的compute_metrics ---
def compute_ie_metrics(
    predictions: List[List[int]], true_labels: List[List[int]], id_to_label_map: Dict
) -> Dict:
    """
    计算序列标注任务的评估指标 (如 F1, P, R)。
    predictions: List of lists, e.g., [[0, 1, 2, 0], [0, 3, 4, 0]] (模型预测的标签ID序列)
    true_labels: List of lists, e.g., [[0, 1, 2, 0], [0, 1, 1, 0]] (真实的标签ID序列)
    id_to_label_map: 字典，将ID映射回标签字符串 (例如 IE_ID_TO_LABEL)
    """
    if not SEQEVAL_AVAILABLE:
        print("seqeval 不可用，跳过IE指标计算。")
        return {}

    # 将ID转换回标签字符串
    true_sequences = [
        [id_to_label_map.get(lbl_id, "O") for lbl_id in seq] for seq in true_labels
    ]
    pred_sequences = [
        [id_to_label_map.get(lbl_id, "O") for lbl_id in seq] for seq in predictions
    ]

    try:
        # scheme=IOB2: 确保你的标签是严格的IOB2格式，否则seqeval会报错
        # mode='strict': 精确匹配实体类型和边界
        report = seqeval_classification_report(
            true_sequences,
            pred_sequences,
            output_dict=True,
            mode="strict",
            scheme=IOB2,  # 或者你使用的其他scheme如IOBES
        )
        # 提取我们关心的指标，例如 micro avg F1 或 specific entity F1s
        metrics = {
            "precision_micro": report["micro avg"]["precision"],
            "recall_micro": report["micro avg"]["recall"],
            "f1_micro": report["micro avg"]["f1-score"],
            "accuracy": report.get("accuracy", 0.0),  # seqeval新版本可能有accuracy
        }
        # 你也可以提取每个实体类型的F1，例如 "TGT" 和 "ARG"
        if "TGT" in report:
            metrics["f1_TGT"] = report["TGT"]["f1-score"]
        if "ARG" in report:
            metrics["f1_ARG"] = report["ARG"]["f1-score"]
        return metrics
    except Exception as e:
        print(f"使用seqeval计算指标时出错: {e}")
        print("请检查标签序列格式是否符合所选scheme (如IOB2)。")
        print("True sequences sample:", true_sequences[0] if true_sequences else "N/A")
        print("Pred sequences sample:", pred_sequences[0] if pred_sequences else "N/A")
        return {}


# --- 示例如何使用 Trainer ---
if __name__ == "__main__":
    # 这个 __main__ 块只是一个示例，你需要实际的数据集和模型来运行它
    print("Trainer class defined. This is an example of how to use it.")

    # 0. 准备：确保有 preprocessor.py, model.py, ie_dataset.py
    from data_process.preprocessor import IE_ID_TO_LABEL, IE_LABELS
    from model.ie_model import RobertaForTokenClassification
    from data_process.ie_dataset import HateIEDataset

    # 1. 定义参数
    MODEL_PATH = "models/chinese-roberta-wwm-ext-large"  # 或者 Hugging Face 模型名
    # MODEL_PATH = "hfl/chinese-roberta-wwm-ext" # 测试用
    TRAIN_JSON = "temp_ie_test_data_train.json"  # 假设你有一个训练用的JSON
    EVAL_JSON = "temp_ie_test_data_eval.json"  # 假设你有一个评估用的JSON
    OUTPUT_DIR = "./my_ie_model_results"
    NUM_EPOCHS = 1  # 示例只跑1个epoch
    BATCH_SIZE = 2  # 非常小的批次用于测试
    LEARNING_RATE = 2e-5

    # 创建临时的示例数据文件
    sample_data = [
        {
            "id": 1,
            "content": "张三对李四说你好。",
            "output": "张三 | 对李四说你好 | non-hate | non-hate [END]",
        },
        {
            "id": 2,
            "content": "这部电影真棒，演员表现出色。",
            "output": "这部电影 | 真棒 | non-hate | non-hate [SEP] 演员 | 表现出色 | non-hate | non-hate [END]",
        },
        {
            "id": 3,
            "content": "小明是学生。",
            "output": "小明 | 学生 | non-hate | non-hate [END]",
        },
        {
            "id": 4,
            "content": "小红爱唱歌，她唱得很好听。",
            "output": "小红 | 爱唱歌 | non-hate | non-hate [SEP] 她 | 唱得很好听 | non-hate | non-hate [END]",
        },
    ]
    with open(TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(sample_data[:2], f)
    with open(EVAL_JSON, "w", encoding="utf-8") as f:
        json.dump(sample_data[2:], f)

    # 2. 实例化模型
    try:
        ie_model = RobertaForTokenClassification(
            model_name_or_path=MODEL_PATH, num_labels=len(IE_LABELS)
        )
    except Exception as e:
        print(
            f"无法实例化模型: {e}. 请检查模型路径 '{MODEL_PATH}' 是否正确且包含必要文件。"
        )
        exit()

    # 3. 实例化数据集
    # 注意: HateIEDataset 内部会使用 tokenizer，确保 tokenizer_name_or_path 与模型一致
    try:
        train_ie_dataset = HateIEDataset(
            original_json_path=TRAIN_JSON,
            tokenizer_name_or_path=MODEL_PATH,
            max_seq_length=128,
        )
        eval_ie_dataset = HateIEDataset(
            original_json_path=EVAL_JSON,
            tokenizer_name_or_path=MODEL_PATH,
            max_seq_length=128,
        )
    except Exception as e:
        print(f"无法实例化数据集: {e}. 可能是tokenizer路径问题或原始json文件问题。")
        exit()

    if not train_ie_dataset or not eval_ie_dataset:
        print("训练或评估数据集为空，无法继续。")
        exit()

    # 4. 定义 compute_metrics 函数 (特定于IE任务)
    def ie_metrics_calculator(preds, labels):
        return compute_ie_metrics(preds, labels, IE_ID_TO_LABEL)

    # 5. 实例化 Trainer
    trainer = GeneralTrainer(
        model=ie_model,
        train_dataset=train_ie_dataset,
        eval_dataset=eval_ie_dataset,
        compute_metrics_fn=ie_metrics_calculator,  # 传入我们定义的IE评估函数
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        output_dir=OUTPUT_DIR,
        logging_steps=1,  # 频繁打印用于小数据集测试
        save_steps=10,  # 假设
    )

    # 6. 开始训练
    print("\n--- 开始示例训练 ---")
    trainer.train()

    # 7. 训练后评估 (可选，因为evaluate已在每个epoch后调用)
    print("\n--- 训练后最终评估 ---")
    final_eval_metrics = trainer.evaluate(description="Final Evaluation on Eval Set")
    print(f"Final Evaluation Metrics: {final_eval_metrics}")

    # 8. 预测示例 (使用评估集作为测试集)
    print("\n--- 开始示例预测 ---")
    # 预测时，HateIEDataset 通常也需要原始JSON（不含output），或只传content给模型
    # 这里我们简单地用 eval_ie_dataset，它会包含'labels'键，但predict方法会忽略它
    predictions_ids = trainer.predict(
        test_dataset=eval_ie_dataset
    )  # 返回的是标签ID列表的列表

    if predictions_ids:
        print(f"得到 {len(predictions_ids)} 条预测结果。")
        for i, pred_id_seq in enumerate(predictions_ids[:2]):  # 只打印前2条
            # 需要从 eval_ie_dataset 获取原始输入和 attention_mask 来正确解码
            # 这里只是一个简化的展示
            raw_input_example = eval_ie_dataset.original_dataset[
                i
            ]  # 获取原始HateOriDataset的项
            # 注意：上面的索引i可能不直接对应原始ID，因为HateIEDataset可能跳过一些样本
            # 更准确的做法是让predict也返回原始ID或文本

            tokens_for_pred = eval_ie_dataset.tokenizer.convert_ids_to_tokens(
                eval_ie_dataset[i]["input_ids"].tolist()
            )
            attention_mask_for_pred = eval_ie_dataset[i]["attention_mask"].tolist()

            pred_labels_str = []
            for token_idx, token_id in enumerate(pred_id_seq):
                if attention_mask_for_pred[
                    token_idx
                ] == 1 and token_id != IE_ID_TO_LABEL.get(
                    "O"
                ):  # 只显示有效且非O的预测
                    # 在实际场景中，还需要处理 IGNORE_INDEX (-100) 的情况，
                    # 但我们的 predict 返回的是 argmax 后的结果，不会直接是 -100
                    if (
                        eval_ie_dataset[i]["labels"][token_idx] != -100
                    ):  # 只考虑非特殊token的预测
                        pred_labels_str.append(
                            f"{tokens_for_pred[token_idx]}({IE_ID_TO_LABEL.get(token_id, 'UNK')})"
                        )

            print(
                f"  预测 {i} (Content: '{raw_input_example[1][:30]}...'): {' '.join(pred_labels_str)}"
            )

    # 清理临时文件
    if os.path.exists(TRAIN_JSON):
        os.remove(TRAIN_JSON)
    if os.path.exists(EVAL_JSON):
        os.remove(EVAL_JSON)
    print("\n示例完成，临时数据文件已删除。")
