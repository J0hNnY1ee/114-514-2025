# src/models/group_clf_model.py
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

try:
    from data_process.preprocessor import GROUP_CATEGORIES # 使用这个作为目标群体标签
    num_group_labels = len(GROUP_CATEGORIES)
except ImportError:
    print("警告: 无法从 data_process.preprocessor 导入 GROUP_CATEGORIES。")
    print("将使用默认的 num_labels=6 进行 Targeted Group 分类。")
    num_group_labels = 6 


class RobertaForTargetedGroupClassification(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int = num_group_labels, dropout_prob: float = 0.1):
        """
        基于 RoBERTa 的序列分类模型，用于Targeted Group判断。

        参数:
            model_name_or_path (str): 预训练模型的名称或本地路径。
            num_labels (int): 分类任务的标签数量 (即 GROUP_CATEGORIES 的长度)。
            dropout_prob (float): Dropout 概率。
        """
        super(RobertaForTargetedGroupClassification, self).__init__()
        self.num_labels = num_labels

        config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = self.num_labels # 更新配置中的num_labels

        self.roberta = AutoModel.from_pretrained(model_name_or_path, config=config)
        self.dropout = nn.Dropout(dropout_prob if dropout_prob is not None else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # 可选: 初始化分类头权重
        # self._init_classifier_weights(self.classifier)

    # def _init_classifier_weights(self, module): ... (与之前的模型类似)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor = None,
                token_type_ids: torch.Tensor = None,
                labels: torch.Tensor = None, # 训练时传入，用于计算损失
                return_dict: bool = None):
        """
        模型的前向传播。

        参数:
            labels (torch.Tensor, optional): 真实的标签 ID (目标群体类别ID)。形状: (batch_size)
            ... (其他参数与序列分类模型类似)
        返回:
            与 RobertaForHatefulClassification 类似的输出字典，包含 "loss" 和 "logits"。
            logits 形状: (batch_size, num_labels) (num_labels 是目标群体的类别数)
        """
        return_dict = return_dict if return_dict is not None else self.roberta.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        # 使用 [CLS] token 的输出进行分类
        cls_token_output = outputs.last_hidden_state[:, 0, :]
        # 或者使用 pooled_output 如果可用且适合
        # pooled_output = outputs.pooler_output if outputs.pooler_output is not None else cls_token_output
        
        pooled_output = self.dropout(cls_token_output)
        logits = self.classifier(pooled_output) # 形状: (batch_size, self.num_labels)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss() # 多分类交叉熵
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        output_dict = {"logits": logits}
        if loss is not None:
            output_dict["loss"] = loss
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            output_dict["hidden_states"] = outputs.hidden_states
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            output_dict["attentions"] = outputs.attentions
        return output_dict


# --- 示例如何测试这个模型 ---
def test_targeted_group_classification_model():
    local_model_path = "models/chinese-roberta-wwm-ext-large"
    # local_model_path = "hfl/chinese-roberta-wwm-ext" # 备用

    print(f"尝试加载模型: {local_model_path}")
    print(f"用于Targeted Group分类的标签数量: {num_group_labels}")

    try:
        model = RobertaForTargetedGroupClassification(
            model_name_or_path=local_model_path, 
            num_labels=num_group_labels # 使用从preprocessor导入的数量
        )
        model.eval()
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    print("Targeted Group 分类模型加载成功!")

    batch_size = 2
    seq_length = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)
    
    # 伪标签，形状为 (batch_size)，值在 [0, num_group_labels-1] 之间
    labels = torch.randint(0, num_group_labels, (batch_size,), dtype=torch.long)

    print(f"\n构造伪输入数据: batch_size={batch_size}, seq_length={seq_length}")

    # 测试推理模式
    print("\n测试推理模式 (不带labels):")
    with torch.no_grad():
        outputs_infer = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=True)
    logits_infer = outputs_infer["logits"]
    assert logits_infer.shape == (batch_size, num_group_labels), \
        f"推理模式下Logits形状错误: 期望 ({batch_size}, {num_group_labels}), 得到 {logits_infer.shape}"
    print(f"推理模式 Logits 形状: {logits_infer.shape} (通过)")

    # 测试训练模式
    print("\n测试训练模式 (带labels):")
    model.train()
    outputs_train = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, return_dict=True)
    model.eval()

    logits_train = outputs_train["logits"]
    loss_train = outputs_train["loss"]

    assert logits_train.shape == (batch_size, num_group_labels), \
        f"训练模式下Logits形状错误: 期望 ({batch_size}, {num_group_labels}), 得到 {logits_train.shape}"
    print(f"训练模式 Logits 形状: {logits_train.shape} (通过)")

    assert loss_train is not None and loss_train.dim() == 0, \
        f"训练模式下Loss错误: 期望一个标量, 得到 {loss_train}"
    print(f"训练模式 Loss: {loss_train.item()} (类型: {type(loss_train)}, 维度: {loss_train.dim()}) (通过)")

    print("\nTargeted Group 分类模型基本功能测试完成！")

if __name__ == "__main__":
    test_targeted_group_classification_model()