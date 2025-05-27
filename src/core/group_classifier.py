# src/core/group_classifier.py
import torch,os
from transformers import AutoTokenizer, AutoConfig
from typing import List, Dict, Union, Optional

try:

    from models.group_clf_model import RobertaForTargetedGroupClassification
    from data_process.preprocessor import GROUP_TO_ID, ID_TO_GROUP, GROUP_CATEGORIES
except ImportError as e:
    print(f"ImportError in GroupClassifier: {e}. Ensure relevant modules are accessible.")
    # Fallback definitions
    class RobertaForTargetedGroupClassification: # Placeholder
        def __init__(self, model_name_or_path, num_labels, dropout_prob=0.1): pass
        def load_state_dict(self, state_dict): pass
        def to(self, device): pass
        def eval(self): pass
    _fallback_group_cats_core = ["Sexism", "Racism", "Region", "LGBTQ", "others", "non-hate"]
    GROUP_TO_ID = {name: i for i, name in enumerate(_fallback_group_cats_core)}
    ID_TO_GROUP = {i: name for i, name in enumerate(_fallback_group_cats_core)}
    GROUP_CATEGORIES = _fallback_group_cats_core
    print("Warning: Using fallback label definitions in group_classifier.py")


class GroupClassifier:
    def __init__(self,
                 trained_model_checkpoint_path: str, # 包含 pytorch_model.bin 的目录
                 base_model_name_or_path: str,    # 用于加载config和tokenizer的基础模型
                 device: Union[str, torch.device] = "cpu",
                 max_seq_length: int = 256, 
                 # 输入模板，现在包含 {hateful_label}
                 input_template: str = "[CLS] T: {target} A: {argument} IsHate: {hateful_label} C: {content} [SEP]",
                 id_to_group_label_map: Optional[Dict[int, str]] = None,
                 group_label_to_id_map: Optional[Dict[str, int]] = None,
                 num_group_labels: Optional[int] = None
                 ):
        """
        初始化Targeted Group分类器。

        参数:
            trained_model_checkpoint_path (str): 包含训练好的TG分类模型权重 (pytorch_model.bin) 的目录。
            base_model_name_or_path (str): 基础预训练模型的路径/名称。
            device (Union[str, torch.device]): "cpu" 或 "cuda"。
            max_seq_length (int): 输入序列的最大长度。
            input_template (str): 构建模型输入的模板，应包含 {target}, {argument}, {hateful_label}, {content}。
            id_to_group_label_map (Dict[int, str], optional): TG标签ID到字符串的映射。
            group_label_to_id_map (Dict[str, int], optional): TG标签字符串到ID的映射 (主要用于获取 "non-hate" ID)。
            num_group_labels (int, optional): TG标签的数量 (GROUP_CATEGORIES的长度)。
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.max_seq_length = max_seq_length
        self.input_template = input_template
        
        self.id_to_label = id_to_group_label_map if id_to_group_label_map else ID_TO_GROUP
        self.label_to_id = group_label_to_id_map if group_label_to_id_map else GROUP_TO_ID
        self._num_labels = num_group_labels if num_group_labels else len(GROUP_CATEGORIES)

        # 获取 "non-hate" 对应的目标群体ID，用于特殊处理
        self.non_hate_group_id = self.label_to_id.get("non-hate")
        if self.non_hate_group_id is None:
            raise ValueError("'non-hate' category not found in group_label_to_id_map (GROUP_TO_ID). "
                             "Ensure it's defined in preprocessor.GROUP_CATEGORIES.")

        try:
            self.model = RobertaForTargetedGroupClassification(
                model_name_or_path=base_model_name_or_path,
                num_labels=self._num_labels
            )
            model_weights_path = os.path.join(trained_model_checkpoint_path, "pytorch_model.bin")
            if not os.path.exists(model_weights_path):
                raise FileNotFoundError(
                    f"训练好的TG分类模型权重 'pytorch_model.bin' 未在 '{trained_model_checkpoint_path}' 找到。"
                )
            
            print(f"尝试从 '{model_weights_path}' 加载TG分类模型权重...")
            self.model.load_state_dict(
                torch.load(model_weights_path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"Targeted Group Classification model loaded from '{trained_model_checkpoint_path}' (weights) "
                  f"and '{base_model_name_or_path}' (config/base) on {self.device}")
        except Exception as e:
            print(f"从 '{trained_model_checkpoint_path}' 加载TG分类模型时出错: {e}")
            raise

    def _prepare_input_text(self, target: str, argument: str, hateful_label: str, content: str) -> str:
        """根据模板准备输入文本。"""
        target_for_input = "[EMPTY_TARGET]" if target == "NULL" or not str(target).strip() else str(target)
        argument_for_input = "[EMPTY_ARG]" if argument == "NULL" or not str(argument).strip() else str(argument)
        
        # hateful_label 已经是字符串形式 ("hate" or "non-hate")
        try:
            return self.input_template.format(
                target=target_for_input,
                argument=argument_for_input,
                hateful_label=hateful_label, # 使用传入的hateful_label字符串
                content=content
            )
        except KeyError as e:
            raise ValueError(f"TG Classifier输入模板 '{self.input_template}' 中的占位符不正确: {e}.")


    def classify_batch(self, 
                       pipeline_inputs: List[Dict[str, str]]
                       ) -> List[Dict[str, Union[str, float]]]:
        """
        对一批组合输入进行Targeted Group分类。

        参数:
            pipeline_inputs (List[Dict[str, str]]):
                一个列表，每个元素是一个字典，必须包含:
                "target": 抽取的Target文本
                "argument": 抽取的Argument文本
                "hateful_label": 前一阶段预测的Hateful标签 ("hate" 或 "non-hate")
                "content": 原始文本内容

        返回:
            一个列表，每个元素是一个字典，包含预测的 "targeted_group_label" (字符串) 
            和 "targeted_group_score" (置信度)。
            例如: [{"targeted_group_label": "Region", "targeted_group_score": 0.9}, ...]
        """
        if not pipeline_inputs:
            return []

        results = []
        # 将输入分为需要模型预测的和可以直接确定的 (non-hate)
        inputs_to_model = []
        indices_to_model = [] # 记录这些输入在原始pipeline_inputs中的索引
        
        for i, item in enumerate(pipeline_inputs):
            if item.get("hateful_label", "").lower() == "non-hate":
                # 对于non-hate的情况，直接输出结果
                results.append({
                    "targeted_group_label": self.id_to_label.get(self.non_hate_group_id, "non-hate"), # 从ID转回 "non-hate"
                    "targeted_group_score": 1.0 # 置信度设为1.0
                })
            else: # hateful_label is "hate" or unknown (assume needs prediction)
                input_text = self._prepare_input_text(
                    item["target"], item["argument"], 
                    item.get("hateful_label", "hate"), # 如果没提供hateful_label，默认按hate处理
                    item["content"]
                )
                inputs_to_model.append(input_text)
                indices_to_model.append(i)
                results.append(None) # 占位，稍后填充模型预测结果
        
        if not inputs_to_model: # 如果所有输入都是non-hate
            return results # 此时results已填充完毕

        # 对需要模型预测的部分进行批量处理
        tokenized_batch = self.tokenizer(
            inputs_to_model,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = tokenized_batch["input_ids"].to(self.device)
        attention_mask = tokenized_batch["attention_mask"].to(self.device)
        token_type_ids = tokenized_batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        with torch.no_grad():
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            )
            logits = model_outputs["logits"] # 形状: (len(inputs_to_model), num_group_labels)
            
            probabilities = torch.softmax(logits, dim=-1)
            predicted_ids = torch.argmax(probabilities, dim=-1)

            for j, original_idx in enumerate(indices_to_model):
                pred_id = predicted_ids[j].item()
                pred_label_str = self.id_to_label.get(pred_id, "others") # 默认"others"如果ID未知
                confidence_score = probabilities[j, pred_id].item()
                
                results[original_idx] = { # 填充到正确的位置
                    "targeted_group_label": pred_label_str,
                    "targeted_group_score": float(f"{confidence_score:.4f}")
                }
        return results

    def classify_single(self, target: str, argument: str, hateful_label: str, content: str) -> Dict[str, Union[str, float]]:
        """对单个组合输入进行分类。"""
        return self.classify_batch([{
            "target": target, 
            "argument": argument, 
            "hateful_label": hateful_label, 
            "content": content
        }])[0]


# --- 示例用法 ---
if __name__ == "__main__":
    TRAINED_GROUP_CLF_MODEL_DIR = "./models/outputs/group_clf/best_model_on_eval_loss" 
    BASE_MODEL_PATH = "models/chinese-roberta-wwm-ext-large"
    # BASE_MODEL_PATH = "hfl/chinese-roberta-wwm-ext"

    if 'ID_TO_GROUP' not in globals(): # Fallback for direct run
        print("Running group_classifier.py directly, using inline fallback label definitions.")
        _fallback_group_cats_main_clf = ["Sexism", "Racism", "Region", "LGBTQ", "others", "non-hate"]
        GROUP_TO_ID = {name: i for i, name in enumerate(_fallback_group_cats_main_clf)}
        ID_TO_GROUP = {i: name for i, name in enumerate(_fallback_group_cats_main_clf)}
        GROUP_CATEGORIES = _fallback_group_cats_main_clf


    if not (os.path.exists(os.path.join(TRAINED_GROUP_CLF_MODEL_DIR, "pytorch_model.bin"))):
        print(f"错误: 训练好的TG分类模型权重 'pytorch_model.bin' 未在 '{TRAINED_GROUP_CLF_MODEL_DIR}' 找到。")
        print("请先运行 group_clf_train.py 生成模型。此示例将跳过。")
    else:
        try:
            classifier = GroupClassifier(
                trained_model_checkpoint_path=TRAINED_GROUP_CLF_MODEL_DIR,
                base_model_name_or_path=BASE_MODEL_PATH,
                device="cuda" if torch.cuda.is_available() else "cpu",
                # 使用与训练时一致的映射和标签数量
                id_to_group_label_map=ID_TO_GROUP, 
                group_label_to_id_map=GROUP_TO_ID,
                num_group_labels=len(GROUP_CATEGORIES)
            )

            # 示例输入 (通常由前两个阶段的流水线提供)
            pipeline_sample_inputs = [
                {"target": "江苏人", "argument": "都是骗子", "hateful_label": "hate", "content": "江苏人都是骗子，我再也不信了。"},
                {"target": "这本书", "argument": "很好看", "hateful_label": "non-hate", "content": "这本书很好看，推荐。"},
                {"target": "黑人", "argument": "智商低", "hateful_label": "hate", "content": "我觉得黑人智商低。"},
                {"target": "同性恋", "argument": "不正常", "hateful_label": "hate", "content": "同性恋不正常，应该治疗。"},
                {"target": "女人", "argument": "就该在家做饭", "hateful_label": "hate", "content": "女人就该在家做饭带孩子。"},
                {"target": "某些官员", "argument": "贪污腐败", "hateful_label": "hate", "content": "那些某些官员贪污腐败，太可恶了。"},
            ]

            print(f"\n--- 开始Targeted Group分类 ---")
            
            print("\n测试单个分类 (hate case):")
            single_res_hate = classifier.classify_single(
                target="外地人", argument="抢工作", hateful_label="hate", content="外地人来抢工作，真烦！"
            )
            print(f"  Input: T='外地人', A='抢工作', H='hate'")
            print(f"  Output: TG Label='{single_res_hate['targeted_group_label']}', Score={single_res_hate['targeted_group_score']:.4f}")

            print("\n测试单个分类 (non-hate case):")
            single_res_nonhate = classifier.classify_single(
                target="这个苹果", argument="很甜", hateful_label="non-hate", content="这个苹果很甜，很好吃。"
            )
            print(f"  Input: T='这个苹果', A='很甜', H='non-hate'")
            print(f"  Output: TG Label='{single_res_nonhate['targeted_group_label']}', Score={single_res_nonhate['targeted_group_score']:.4f}")


            print("\n测试批量分类:")
            batch_results = classifier.classify_batch(pipeline_sample_inputs)
            
            for i, item_input in enumerate(pipeline_sample_inputs):
                result = batch_results[i]
                print(f"  Input {i+1}: T='{item_input['target']}', A='{item_input['argument']}', H='{item_input['hateful_label']}'")
                print(f"  Output {i+1}: TG Label='{result['targeted_group_label']}', Score={result['targeted_group_score']:.4f}")

        except Exception as e:
            print(f"运行GroupClassifier示例时发生错误: {e}")
            import traceback
            traceback.print_exc()