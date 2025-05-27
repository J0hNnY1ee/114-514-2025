# models/ie_model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

# 从 preprocessor 导入 GENERIC_IOB_LABELS 的长度，这将是每个分类头的 num_labels
# 同时也需要 IGNORE_INDEX 用于损失计算
try:
    # 假设 preprocessor.py 在与 models/ 目录同级的 data_process/ 目录下
    from data_process.preprocessor import GENERIC_IOB_LABELS, IGNORE_INDEX
    num_generic_iob_labels = len(GENERIC_IOB_LABELS)
except ImportError:
    print("警告: 无法从 data_process.preprocessor 导入 GENERIC_IOB_LABELS 或 IGNORE_INDEX。")
    print("将使用默认的 num_labels=3 (O,B,I) 和 IGNORE_INDEX=-100。")
    num_generic_iob_labels = 3
    IGNORE_INDEX = -100


class RobertaForMultiLabelTokenClassification(nn.Module):
    def __init__(self, 
                 model_name_or_path: str, 
                 num_single_layer_labels: int = num_generic_iob_labels, # 每个独立层的标签数 (O,B,I)
                 dropout_prob: float = 0.1):
        """
        基于 RoBERTa 的多标签 Token 分类模型 (用于Target层和Argument层独立标注)。

        参数:
            model_name_or_path (str): 预训练模型的名称或本地路径。
            num_single_layer_labels (int): 每个独立标注层 (Target, Argument) 的标签数量。
                                         通常是 GENERIC_IOB_LABELS 的长度 (e.g., 3 for O, B, I)。
            dropout_prob (float): Dropout 概率。
        """
        super(RobertaForMultiLabelTokenClassification, self).__init__()
        self.num_target_labels = num_single_layer_labels
        self.num_argument_labels = num_single_layer_labels

        config = AutoConfig.from_pretrained(model_name_or_path)
        # config.num_labels 字段对于这种多头模型意义不大，但可以设置一个参考值
        # config.num_labels = num_single_layer_labels # 或者不设置

        self.roberta = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.dropout = nn.Dropout(dropout_prob if dropout_prob is not None else config.hidden_dropout_prob)

        # 分类头 - Target层
        self.target_classifier = nn.Linear(config.hidden_size, self.num_target_labels)
        # 分类头 - Argument层
        self.argument_classifier = nn.Linear(config.hidden_size, self.num_argument_labels)

        # 可选的权重初始化
        # self._init_classifier_weights(self.target_classifier)
        # self._init_classifier_weights(self.argument_classifier)

    def _init_classifier_weights(self, module):
        """ 初始化分类头权重 (可选) """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.roberta.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                labels_target: torch.Tensor = None,    # Target 层的标签
                labels_argument: torch.Tensor = None, # Argument 层的标签
                return_dict: bool = None):
        """
        模型的前向传播。

        参数:
            labels_target (torch.Tensor, optional): Target层的真实标签。形状: (batch_size, seq_length)
            labels_argument (torch.Tensor, optional): Argument层的真实标签。形状: (batch_size, seq_length)
            ... (其他参数与之前类似)
        返回:
            一个字典，包含:
            "loss": 总损失 (如果提供了labels_target和labels_argument)
            "logits_target": Target层的logits。形状: (batch_size, seq_length, num_target_labels)
            "logits_argument": Argument层的logits。形状: (batch_size, seq_length, num_argument_labels)
            ... (可选的 hidden_states, attentions)
        """
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state
        sequence_output_dropout = self.dropout(sequence_output)

        logits_target = self.target_classifier(sequence_output_dropout)
        logits_argument = self.argument_classifier(sequence_output_dropout)

        total_loss = None
        if labels_target is not None and labels_argument is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX) # 使用全局定义的IGNORE_INDEX
            
            # 计算Target层的损失
            loss_target = loss_fct(logits_target.view(-1, self.num_target_labels), labels_target.view(-1))
            
            # 计算Argument层的损失
            loss_argument = loss_fct(logits_argument.view(-1, self.num_argument_labels), labels_argument.view(-1))
            
            # 合并损失 (可以直接相加，或加权相加)
            total_loss = loss_target + loss_argument
            # 可以考虑返回独立的损失：
            # total_loss = {"target_loss": loss_target, "argument_loss": loss_argument, "total_loss": loss_target + loss_argument}
            # 但Trainer通常期望一个单一的 "loss" 键用于反向传播

        if not return_dict: # 保持与HuggingFace风格一致的非字典返回（尽管不常用）
            output = (logits_target, logits_argument)
            return ((total_loss,) + output) if total_loss is not None else output

        output_payload = {
            "logits_target": logits_target,
            "logits_argument": logits_argument,
        }
        if total_loss is not None:
            output_payload["loss"] = total_loss # Trainer将使用这个进行反向传播
            output_payload["loss_target"] = loss_target # 可选，用于监控
            output_payload["loss_argument"] = loss_argument # 可选，用于监控
        
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            output_payload["hidden_states"] = outputs.hidden_states
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            output_payload["attentions"] = outputs.attentions
            
        return output_payload


# --- 示例如何测试这个多标签模型 ---
def test_multi_label_model():
    local_model_path = "models/chinese-roberta-wwm-ext-large"
    # local_model_path = "hfl/chinese-roberta-wwm-ext"

    print(f"尝试加载多标签模型: {local_model_path}")
    print(f"每个标注层的标签数量 (O,B,I): {num_generic_iob_labels}")

    try:
        # num_single_layer_labels 对应每个独立分类头的输出维度
        model = RobertaForMultiLabelTokenClassification(
            model_name_or_path=local_model_path, 
            num_single_layer_labels=num_generic_iob_labels
        )
        model.eval()
    except Exception as e:
        print(f"加载多标签模型失败: {e}")
        return

    print("多标签模型加载成功!")

    batch_size = 2
    seq_length = 64
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    
    # 伪标签 - 两层
    labels_target = torch.randint(0, num_generic_iob_labels, (batch_size, seq_length), dtype=torch.long)
    labels_argument = torch.randint(0, num_generic_iob_labels, (batch_size, seq_length), dtype=torch.long)
    
    # 设置一些IGNORE_INDEX
    labels_target[0, 5:10] = IGNORE_INDEX
    labels_argument[1, 0] = IGNORE_INDEX 

    print(f"\n构造伪输入数据: batch_size={batch_size}, seq_length={seq_length}")

    # 测试推理模式
    print("\n测试推理模式:")
    with torch.no_grad():
        outputs_infer = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids, 
            return_dict=True
        )
    logits_tgt_infer = outputs_infer["logits_target"]
    logits_arg_infer = outputs_infer["logits_argument"]

    assert logits_tgt_infer.shape == (batch_size, seq_length, num_generic_iob_labels), "Logits_target形状错误"
    assert logits_arg_infer.shape == (batch_size, seq_length, num_generic_iob_labels), "Logits_argument形状错误"
    print(f"推理模式 Logits Target 形状: {logits_tgt_infer.shape} (通过)")
    print(f"推理模式 Logits Argument 形状: {logits_arg_infer.shape} (通过)")

    # 测试训练模式
    print("\n测试训练模式:")
    model.train()
    outputs_train = model(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        token_type_ids=token_type_ids, 
        labels_target=labels_target,
        labels_argument=labels_argument,
        return_dict=True
    )
    model.eval()

    total_loss_train = outputs_train["loss"]
    loss_tgt_train = outputs_train.get("loss_target") # 如果返回了独立损失
    loss_arg_train = outputs_train.get("loss_argument")
    logits_tgt_train = outputs_train["logits_target"]
    logits_arg_train = outputs_train["logits_argument"]

    assert logits_tgt_train.shape == (batch_size, seq_length, num_generic_iob_labels), "训练模式Logits_target形状错误"
    assert logits_arg_train.shape == (batch_size, seq_length, num_generic_iob_labels), "训练模式Logits_argument形状错误"
    print(f"训练模式 Logits Target 形状: {logits_tgt_train.shape} (通过)")
    print(f"训练模式 Logits Argument 形状: {logits_arg_train.shape} (通过)")

    assert total_loss_train is not None and total_loss_train.dim() == 0, "总损失错误"
    print(f"训练模式 总Loss: {total_loss_train.item()} (通过)")
    if loss_tgt_train is not None: print(f"训练模式 Target Loss: {loss_tgt_train.item()}")
    if loss_arg_train is not None: print(f"训练模式 Argument Loss: {loss_arg_train.item()}")

    print("\n多标签Token分类模型基本功能测试完成！")

if __name__ == "__main__":
    test_multi_label_model()