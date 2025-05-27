# models/hate_clf_model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig # AutoConfig用于加载配置


try:
    
    from data_process.preprocessor import HATEFUL_CATEGORIES
    num_hateful_labels = len(HATEFUL_CATEGORIES)
except ImportError:
    print("警告: 无法从 data_process.preprocessor 导入 HATEFUL_CATEGORIES。")
    print("将使用默认的 num_labels=2 进行 Hateful 分类。")
    num_hateful_labels = 2 # 默认 hate/non-hate


class RobertaForHatefulClassification(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int = num_hateful_labels, dropout_prob: float = 0.1):
        """
        基于 RoBERTa (或其他BERT类模型) 的序列分类模型，用于Hateful判断。

        参数:
            model_name_or_path (str): 预训练模型的名称或本地路径。
                                     例如: "models/chinese-roberta-wwm-ext-large"
            num_labels (int): 分类任务的标签数量 (对于Hateful分类，通常是2)。
            dropout_prob (float): 应用于 RoBERTa 输出和最终分类层之前的 Dropout 概率。
        """
        super(RobertaForHatefulClassification, self).__init__()
        self.num_labels = num_labels

        # 加载预训练模型的配置
        config = AutoConfig.from_pretrained(model_name_or_path)
        # 更新配置中的num_labels，有些模型实现可能会用到它
        config.num_labels = self.num_labels

        # 加载预训练的 RoBERTa 模型主体
        self.roberta = AutoModel.from_pretrained(model_name_or_path, config=config)

        # Dropout 层
        self.dropout = nn.Dropout(dropout_prob if dropout_prob is not None else config.hidden_dropout_prob)

        # 序列分类头
        # 通常使用 [CLS] token 的输出 (即 pooler_output 或 last_hidden_state[:, 0, :])
        # RoBERTa的 config.hidden_size 是其隐藏层维度
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)


    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None, # RoBERTa 通常不使用
                labels: torch.Tensor = None, # 训练时传入，用于计算损失
                return_dict: bool = None):
        """
        模型的前向传播。

        参数:
            input_ids (torch.Tensor): 输入的 token ID 序列。形状: (batch_size, seq_length)
            attention_mask (torch.Tensor, optional): Attention mask。形状: (batch_size, seq_length)
            token_type_ids (torch.Tensor, optional): Token type ID。形状: (batch_size, seq_length)
            labels (torch.Tensor, optional): 真实的标签 ID，用于计算损失。形状: (batch_size)
            return_dict (bool, optional): 是否返回一个包含所有输出的字典。

        返回:
            如果 `labels` 不为 None (训练模式):
                返回一个元组 (loss, logits) 或一个包含 "loss" 和 "logits" 的字典。
            如果 `labels` 为 None (推理模式):
                返回 logits (torch.Tensor) 或一个包含 "logits" 的字典。
                logits 形状: (batch_size, num_labels)
        """
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True # 确保返回字典以便访问特定输出
        )


        cls_token_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(cls_token_output) # 对[CLS]的表示应用dropout
        logits = self.classifier(pooled_output) # 形状: (batch_size, num_labels)

        loss = None
        if labels is not None:
            # 对于二分类或多分类，CrossEntropyLoss 期望 logits (N, C) 和 labels (N)
            # labels 应该是类别索引 (0, 1, ..., C-1)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        # 使用 Hugging Face Transformers 风格的输出字典 (可选，但推荐)
        # from transformers.modeling_outputs import SequenceClassifierOutput
        # return SequenceClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        # 简化版输出
        output_dict = {"logits": logits}
        if loss is not None:
            output_dict["loss"] = loss
        # 如果需要，可以传递模型的其他输出
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            output_dict["hidden_states"] = outputs.hidden_states
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            output_dict["attentions"] = outputs.attentions
        return output_dict


# --- 示例如何测试这个模型 ---
def test_hateful_classification_model():
    # 本地模型路径
    local_model_path = "models/chinese-roberta-wwm-ext-large"
    # 或者使用Hugging Face模型名进行快速测试
    # local_model_path = "hfl/chinese-roberta-wwm-ext"

    print(f"尝试加载模型: {local_model_path}")
    print(f"用于Hateful分类的标签数量: {num_hateful_labels}")

    try:
        model = RobertaForHatefulClassification(model_name_or_path=local_model_path, num_labels=num_hateful_labels)
        model.eval() # 设置为评估模式
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保模型路径正确，并且本地模型文件完整。")
        return

    print("模型加载成功!")

    # 构造一个伪输入数据
    batch_size = 2
    seq_length = 128 # 假设序列长度
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    
    # 伪标签 (训练时需要)，形状为 (batch_size)
    labels = torch.randint(0, num_hateful_labels, (batch_size,), dtype=torch.long)

    print(f"\n构造伪输入数据: batch_size={batch_size}, seq_length={seq_length}")

    # 测试前向传播 (推理模式，不带labels)
    print("\n测试推理模式 (不带labels):")
    with torch.no_grad():
        outputs_infer = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
    logits_infer = outputs_infer["logits"]
    assert logits_infer.shape == (batch_size, num_hateful_labels), \
        f"推理模式下Logits形状错误: 期望 ({batch_size}, {num_hateful_labels}), 得到 {logits_infer.shape}"
    print(f"推理模式 Logits 形状: {logits_infer.shape} (通过)")

    # 测试前向传播 (训练模式，带labels)
    print("\n测试训练模式 (带labels):")
    model.train()
    outputs_train = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, return_dict=True)
    model.eval()

    logits_train = outputs_train["logits"]
    loss_train = outputs_train["loss"]

    assert logits_train.shape == (batch_size, num_hateful_labels), \
        f"训练模式下Logits形状错误: 期望 ({batch_size}, {num_hateful_labels}), 得到 {logits_train.shape}"
    print(f"训练模式 Logits 形状: {logits_train.shape} (通过)")

    assert loss_train is not None and loss_train.dim() == 0, \
        f"训练模式下Loss错误: 期望一个标量, 得到 {loss_train}"
    print(f"训练模式 Loss: {loss_train.item()} (类型: {type(loss_train)}, 维度: {loss_train.dim()}) (通过)")

    print("\nHateful分类模型基本功能测试完成！")

if __name__ == "__main__":
    test_hateful_classification_model()