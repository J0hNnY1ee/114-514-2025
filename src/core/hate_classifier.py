# src/core/hate_classifier.py
import torch,os
from transformers import AutoTokenizer, AutoConfig
from typing import List, Dict, Union, Optional

try:
    # 假设模型定义在 models.hate_clf_model
    from model.hate_clf_model import RobertaForHatefulClassification
    # 假设标签映射在 data_process.preprocessor
    from data_process.preprocessor import HATEFUL_TO_ID, ID_TO_HATEFUL, HATEFUL_CATEGORIES
except ImportError as e:
    print(f"ImportError in HateClassifier: {e}. Ensure relevant modules are accessible.")
    # Fallback definitions
    class RobertaForHatefulClassification: # Placeholder
        def __init__(self, model_name_or_path, num_labels, dropout_prob=0.1): pass
        def load_state_dict(self, state_dict): pass
        def to(self, device): pass
        def eval(self): pass
    HATEFUL_TO_ID = {"hate": 1, "non-hate": 0} # Example, ensure matches your preprocessor
    ID_TO_HATEFUL = {v: k for k, v in HATEFUL_TO_ID.items()}
    HATEFUL_CATEGORIES = ["non-hate", "hate"]


class HateClassifier:
    def __init__(self,
                 trained_model_checkpoint_path: str, # 包含 pytorch_model.bin 的目录
                 base_model_name_or_path: str,    # 用于加载config和tokenizer的基础模型
                 device: Union[str, torch.device] = "cpu",
                 max_seq_length: int = 256, # Hateful分类任务的max_seq_length
                 # 用于构建分类模型输入的模板
                 input_template: str = "[CLS] {target} [SEP] {argument} [SEP] {content} [SEP]",
                 id_to_hateful_label_map: Optional[Dict[int, str]] = None,
                 num_hateful_labels: Optional[int] = None
                 ):
        """
        初始化Hateful分类器。

        参数:
            trained_model_checkpoint_path (str): 包含训练好的分类模型权重 (pytorch_model.bin) 的目录。
            base_model_name_or_path (str): 基础预训练模型的路径/名称，用于加载Config和Tokenizer。
            device (Union[str, torch.device]): "cpu" 或 "cuda"。
            max_seq_length (int): 分类任务输入序列的最大长度。
            input_template (str): 构建模型输入的模板，应包含 {target}, {argument}, {content}。
            id_to_hateful_label_map (Dict[int, str], optional): Hateful标签ID到字符串的映射。
            num_hateful_labels (int, optional): Hateful标签的数量 (通常是2)。
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.max_seq_length = max_seq_length
        self.input_template = input_template
        
        self.id_to_label = id_to_hateful_label_map if id_to_hateful_label_map else ID_TO_HATEFUL
        self._num_labels = num_hateful_labels if num_hateful_labels else len(HATEFUL_CATEGORIES)

        try:
            # 1. 实例化模型结构
            self.model = RobertaForHatefulClassification(
                model_name_or_path=base_model_name_or_path, # 用于获取config和基础roberta
                num_labels=self._num_labels
            )
            # 2. 加载训练好的权重
            model_weights_path = os.path.join(trained_model_checkpoint_path, "pytorch_model.bin")
            if not os.path.exists(model_weights_path):
                raise FileNotFoundError(
                    f"训练好的Hateful分类模型权重 'pytorch_model.bin' 未在 '{trained_model_checkpoint_path}' 找到。"
                )
            
            print(f"尝试从 '{model_weights_path}' 加载Hateful分类模型权重...")
            self.model.load_state_dict(
                torch.load(model_weights_path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval() # 设置为评估模式
            print(f"Hateful Classification model loaded from '{trained_model_checkpoint_path}' (weights) "
                  f"and '{base_model_name_or_path}' (config/base) on {self.device}")
        except Exception as e:
            print(f"从 '{trained_model_checkpoint_path}' 加载Hateful分类模型时出错: {e}")
            raise

    def _prepare_input_text(self, target: str, argument: str, content: str) -> str:
        """根据模板准备输入文本。"""
        # 处理可能的 "NULL" 或空字符串
        target_for_input = "[EMPTY_TARGET]" if target == "NULL" or not str(target).strip() else str(target)
        argument_for_input = "[EMPTY_ARGUMENT]" if argument == "NULL" or not str(argument).strip() else str(argument)
        
        try:
            return self.input_template.format(
                target=target_for_input,
                argument=argument_for_input,
                content=content
            )
        except KeyError as e:
            raise ValueError(f"HateClassifier输入模板 '{self.input_template}' 中的占位符不正确: {e}.")


    def classify_batch(self, 
                       ta_pairs_with_content: List[Dict[str, str]]
                       ) -> List[Dict[str, Union[str, float]]]:
        """
        对一批 (Target, Argument, Content) 组合进行Hateful分类。

        参数:
            ta_pairs_with_content (List[Dict[str, str]]):
                一个列表，每个元素是一个字典，包含 "target", "argument", "content"键。
                例如: [{"target": "T1", "argument": "A1", "content": "C1"},
                       {"target": "T2", "argument": "A2", "content": "C2"}]

        返回:
            一个列表，每个元素是一个字典，包含预测的 "hateful_label" (字符串) 和 "hateful_score" (置信度)。
            例如: [{"hateful_label": "hate", "hateful_score": 0.9},
                   {"hateful_label": "non-hate", "hateful_score": 0.8}]
        """
        if not ta_pairs_with_content:
            return []

        input_texts = [
            self._prepare_input_text(item["target"], item["argument"], item["content"])
            for item in ta_pairs_with_content
        ]

        inputs = self.tokenizer(
            input_texts,
            max_length=self.max_seq_length,
            padding="max_length", # 对整个批次进行padding
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        results = []
        with torch.no_grad():
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            )
            logits = model_outputs["logits"] # 形状: (batch_size, num_hateful_labels)
            
            # 计算概率 (softmax) 和预测标签
            probabilities = torch.softmax(logits, dim=-1) # (batch_size, num_hateful_labels)
            predicted_ids = torch.argmax(probabilities, dim=-1) # (batch_size)

            for i in range(len(ta_pairs_with_content)):
                pred_id = predicted_ids[i].item()
                pred_label_str = self.id_to_label.get(pred_id, "UNK_HATEFUL")
                # 置信度是模型对预测类别的softmax概率
                confidence_score = probabilities[i, pred_id].item() 
                
                results.append({
                    "hateful_label": pred_label_str,
                    "hateful_score": float(f"{confidence_score:.4f}") # 保留4位小数
                })
        return results

    def classify_single(self, target: str, argument: str, content: str) -> Dict[str, Union[str, float]]:
        """对单个 (Target, Argument, Content) 组合进行分类。"""
        return self.classify_batch([{"target": target, "argument": argument, "content": content}])[0]


# --- 示例用法 ---
if __name__ == "__main__":
    # 假设Hateful分类模型已训练并保存在此路径
    TRAINED_HATE_CLF_MODEL_DIR = "models/outputs/hate_clf/best_model_on_eval_loss" 
    # 基础模型路径，用于加载config和tokenizer
    BASE_MODEL_PATH = "models/chinese-roberta-wwm-ext-large"
    # BASE_MODEL_PATH = "hfl/chinese-roberta-wwm-ext" # 备用

    # 确保测试时定义了 ID_TO_HATEFUL 和 HATEFUL_CATEGORIES
    if 'ID_TO_HATEFUL' not in globals():
        print("Running hate_classifier.py directly, using inline Hateful label definitions.")
        HATEFUL_CATEGORIES = ["non-hate", "hate"] # 确保顺序与ID一致
        HATEFUL_TO_ID = {label: i for i, label in enumerate(HATEFUL_CATEGORIES)}
        ID_TO_HATEFUL = {i: label for i, label in enumerate(HATEFUL_CATEGORIES)}


    if not (os.path.exists(os.path.join(TRAINED_HATE_CLF_MODEL_DIR, "pytorch_model.bin"))):
        print(f"错误: 训练好的Hateful分类模型权重 'pytorch_model.bin' 未在 '{TRAINED_HATE_CLF_MODEL_DIR}' 找到。")
        print("请先运行 hate_clf_train.py 生成模型。此示例将跳过。")
    else:
        try:
            classifier = HateClassifier(
                trained_model_checkpoint_path=TRAINED_HATE_CLF_MODEL_DIR,
                base_model_name_or_path=BASE_MODEL_PATH,
                device="cuda" if torch.cuda.is_available() else "cpu",
                id_to_hateful_label_map=ID_TO_HATEFUL, # 传递正确的映射
                num_hateful_labels=len(HATEFUL_CATEGORIES)   # 传递正确的标签数量
            )

            # 示例输入 (通常由 InformationExtractor 提供)
            sample_ta_pairs_with_content = [
                {"target": "他们", "argument": "都是坏人", "content": "我觉得他们都是坏人，应该被谴责。"},
                {"target": "这本书", "argument": "内容很精彩", "content": "这本书内容很精彩，推荐阅读。"},
                {"target": "NULL", "argument": "真差劲", "content": "服务真差劲，太慢了。"},
                {"target": "小猫", "argument": "NULL", "content": "小猫真可爱啊，不是吗？"},
                {"target": "没爹的黑孩", "argument": "到处扔", "content": "没爹的黑孩到处扔"},
            ]

            print(f"\n--- 开始Hateful分类 ---")
            
            # 测试单个分类
            print("\n测试单个分类:")
            single_result = classifier.classify_single(
                target="游戏角色", 
                argument="设计得太弱了", 
                content="这个游戏角色设计得太弱了，完全没法用。"
            )
            print(f"  Input: T='游戏角色', A='设计得太弱了'")
            print(f"  Output: Label='{single_result['hateful_label']}', Score={single_result['hateful_score']:.4f}")

            # 测试批量分类
            print("\n测试批量分类:")
            batch_results = classifier.classify_batch(sample_ta_pairs_with_content)
            
            for i, item_input in enumerate(sample_ta_pairs_with_content):
                result = batch_results[i]
                print(f"  Input {i+1}: T='{item_input['target']}', A='{item_input['argument']}'")
                print(f"  Output {i+1}: Label='{result['hateful_label']}', Score={result['hateful_score']:.4f}")

        except Exception as e:
            print(f"运行HateClassifier示例时发生错误: {e}")
            import traceback
            traceback.print_exc()