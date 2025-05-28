import json
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
import sys

# 导入 PEFT 相关库
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training # kbit training for quantization (optional but good with LoRA)

# --- 配置 ---
try:
    from LLM.prompt import get_analysis_prompts # 保持原样，假设路径设置正确
except ImportError as e:
    print(f"错误：无法从 'LLM.prompt' 导入 'get_analysis_prompts' 函数。错误信息: {e}")
    print(f"当前 sys.path: {sys.path}")
    print("请确保 Python 能够找到 'LLM' 包及其中的 'prompt.py'模块。")
    exit(1)

# Swanlab 配置
SWANLAB_PROJECT_NAME = "qwen3-lora-sft-target-argument" # 修改项目名以反映LoRA
os.environ["SWANLAB_PROJECT"] = SWANLAB_PROJECT_NAME

# 模型和数据配置
MODEL_ID = "models/Qwen3-8B" # 您本地的8B模型路径
CACHE_DIR = "models/model_cache" # 如果需要下载，缓存到这里
MAX_LENGTH = 900 # 保持较小以节省显存

# --- 数据集路径配置 ---
ORIGINAL_TRAIN_DATA_PATH = "data/finetune/train.json"
ORIGINAL_EVAL_DATA_PATH = "data/finetune/val.json"
SFT_TRAIN_FORMATTED_PATH = "data/finetune/sft_lora_train_formatted.jsonl" # 文件名区分LoRA
SFT_EVAL_FORMATTED_PATH = "data/finetune/sft_lora_eval_formatted.jsonl"

# 训练参数
OUTPUT_DIR = "models/outputs/Qwen3-LoRA-TargetArgument" # 输出目录也区分LoRA
LOGGING_DIR = "./logs/Qwen3-LoRA-TargetArgument"

# --- Swanlab 初始化 ---
swanlab.init(
    project=SWANLAB_PROJECT_NAME,
    config={
        "model_id": MODEL_ID,
        "finetuning_method": "LoRA", # 添加LoRA信息
        "data_max_length": MAX_LENGTH,
        "original_train_dataset": ORIGINAL_TRAIN_DATA_PATH,
        "original_eval_dataset": ORIGINAL_EVAL_DATA_PATH,
        "learning_rate": 2e-4, # LoRA通常可以使用稍高一点的学习率
        "num_train_epochs": 3, # LoRA可能需要更多轮次或根据数据调整
        "per_device_train_batch_size": 2, # LoRA下可以尝试增加batch_size
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "lora_r": 16,             # LoRA rank (常见值 8, 16, 32, 64)
        "lora_alpha": 32,         # LoRA alpha (通常是 r 的两倍)
        "lora_dropout": 0.05,     # LoRA dropout
    }
)

# --- 辅助函数 (与之前相同) ---
def extract_content_from_user_prompt(full_user_prompt: str) -> str:
    prefix = "请严格按照你被设定的角色和指令，分析以下文本内容：\n\n"
    if full_user_prompt.startswith(prefix):
        return full_user_prompt[len(prefix):]
    else:
        print(f"警告：无法提取内容。在用户提示中未找到前缀：{full_user_prompt}")
        return full_user_prompt

def transform_dataset_for_chat_finetuning(origin_json_path: str, new_jsonl_path: str):
    formatted_data = []
    if not os.path.exists(origin_json_path):
        print(f"错误: 原始数据集文件 {origin_json_path} 未找到。"); return False
    with open(origin_json_path, "r", encoding="utf-8") as f:
        try: dataset_items = json.load(f)
        except json.JSONDecodeError as e: print(f"错误: 解析JSON文件 {origin_json_path} 失败: {e}"); return False
    for item in dataset_items:
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            print(f"警告: 在 {origin_json_path} 中发现格式不正确的数据项，已跳过: {item}"); continue
        full_user_prompt = item["question"]; assistant_response = item["answer"]
        raw_content = extract_content_from_user_prompt(full_user_prompt)
        system_prompt_for_item, _ = get_analysis_prompts(raw_content)
        formatted_data.append({"system_prompt": system_prompt_for_item, "user_content": raw_content, "assistant_response": assistant_response})
    with open(new_jsonl_path, "w", encoding="utf-8") as f:
        for entry in formatted_data: f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"已转换的数据集保存至 {new_jsonl_path}"); return True

tokenizer = None

def process_sft_data(example):
    global tokenizer; assert tokenizer is not None, "Tokenizer 未初始化。"
    system_part_str = f"<|im_start|>system\n{example['system_prompt']}<|im_end|>\n"
    user_part_str = f"<|im_start|>user\n{example['user_content']}<|im_end|>\n<|im_start|>assistant\n"
    assistant_part_str = f"{example['assistant_response']}<|im_end|>"
    system_tokens = tokenizer(system_part_str, add_special_tokens=False)
    user_tokens = tokenizer(user_part_str, add_special_tokens=False)
    assistant_tokens = tokenizer(assistant_part_str, add_special_tokens=False)
    prompt_length = len(system_tokens.input_ids) + len(user_tokens.input_ids)
    input_ids = system_tokens.input_ids + user_tokens.input_ids + assistant_tokens.input_ids
    attention_mask = [1] * len(input_ids)
    labels = ([-100] * prompt_length) + assistant_tokens.input_ids
    if len(input_ids) > MAX_LENGTH:
        input_ids, attention_mask, labels = input_ids[:MAX_LENGTH], attention_mask[:MAX_LENGTH], labels[:MAX_LENGTH]


    # # --- 调试打印 ---
    # num_valid_labels = sum(1 for label_id in labels if label_id != -100)
    # print(f"Sample - Input length: {len(input_ids)}, Label length: {len(labels)}")
    # print(f"Sample - Prompt length for masking: {prompt_length}")
    # print(f"Sample - Number of valid (non -100) labels: {num_valid_labels}")
    # if num_valid_labels == 0 and len(assistant_tokens.input_ids) > 0 :
    #     print(f"Sample - WARNING: Zero valid labels despite non-empty assistant response!")
    #     print(f"Sample - Assistant tokens: {assistant_tokens.input_ids}")
    #     print(f"Sample - Decoded assistant: {tokenizer.decode(assistant_tokens.input_ids)}")
    #     print(f"Sample - First 10 labels: {labels[:10]}")
    #     print(f"Sample - Last 10 labels: {labels[-10:]}")
    # # --- 结束调试打印 ---

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def predict_with_model(system_prompt_content, user_input_content, model_to_predict, tokenizer_ref): # Renamed model arg
    device = model_to_predict.device # PEFT model also has a device attribute
    messages = [{"role": "system", "content": system_prompt_content}, {"role": "user", "content": user_input_content}]
    text_prompt = tokenizer_ref.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer_ref([text_prompt], return_tensors="pt").to(device)
    # PEFT模型可以直接用于generate
    generated_ids = model_to_predict.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=MAX_LENGTH,
        # pad_token_id=tokenizer_ref.eos_token_id # 确保pad_token_id设置正确
    )
    response_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    response_text = tokenizer_ref.decode(response_ids, skip_special_tokens=True)
    return response_text

# --- 主脚本 ---
if __name__ == "__main__":
    # 1. 模型路径配置 (与之前类似)
    LOCAL_MODEL_PATH = MODEL_ID # 直接使用配置的MODEL_ID作为本地路径
    DOWNLOAD_IF_NOT_FOUND = False # 假设模型总是在本地
    model_load_path = None

    if os.path.isdir(LOCAL_MODEL_PATH) and os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
        print(f"发现本地模型路径: {LOCAL_MODEL_PATH}")
        model_load_path = LOCAL_MODEL_PATH
    elif DOWNLOAD_IF_NOT_FOUND:
        print(f"本地模型路径 {LOCAL_MODEL_PATH} 无效或未找到，尝试从 ModelScope 下载...")
        model_path_in_cache = os.path.join(CACHE_DIR, *MODEL_ID.split('/')) # MODEL_ID 现在是本地路径，这行可能不适用
                                                                        # 如果要下载，MODEL_ID应该是Hub上的ID
        # 如果MODEL_ID是本地路径，则不应进入下载逻辑，除非它也是Hub ID
        # 此处假设MODEL_ID就是本地路径，所以如果找不到就报错
        print(f"错误: 本地模型路径 {LOCAL_MODEL_PATH} 无效，且DOWNLOAD_IF_NOT_FOUND为True但未配置Hub ID。"); exit(1)
    else:
        print(f"错误: 本地模型路径 {LOCAL_MODEL_PATH} 无效，且未启用下载。"); exit(1)

    if model_load_path is None: print("错误: 无法确定模型加载路径。"); exit(1)
    print(f"最终模型加载路径: {model_load_path}")

    # 加载 Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_load_path, use_fast=False, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            print("Tokenizer `pad_token_id` 未设置，将其设置为 `eos_token_id`。")
            tokenizer.pad_token_id = tokenizer.eos_token_id # 很多模型推荐这样做
        print(f"Tokenizer: pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")
    except Exception as e:
        print(f"从路径 {model_load_path} 加载Tokenizer失败: {e}"); exit(1)

    # 加载基础模型
    # 对于LoRA，我们首先加载原始的预训练模型
    # 如果要进行量化 (如8-bit或4-bit)，可以在这里配置 `load_in_8bit=True` 或 `load_in_4bit=True`
    # 并使用 `prepare_model_for_kbit_training`
    print("正在加载基础模型 (用于LoRA)...")
    try:
        # load_in_8bit=True 可以进一步减少显存，但需要 bitsandbytes
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_load_path,
        #     torch_dtype=torch.bfloat16, # 或者 torch.float16
        #     trust_remote_code=True,
        #     device_map="auto", # 仍然可以使用 device_map 让基础模型分布在多卡上
        #     # load_in_8bit=True, # 可选：启用8位量化以节省更多显存
        # )

        # 如果使用 load_in_8bit 或 4bit，通常也建议用 prepare_model_for_kbit_training
        # if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
        #     print("Preparing model for k-bit training...")
        #     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # 确保梯度检查点与kbit兼容
        # else: # 如果不量化，但仍想用梯度检查点
        #     model.gradient_checkpointing_enable() # 确保启用
        
        # 简化版：先不加量化，专注于LoRA
        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            torch_dtype=torch.bfloat16, # 或者 torch.float16
            trust_remote_code=True,
            # device_map="auto" # 对于 LoRA，通常会将整个基础模型加载到一张卡或多张卡上，LoRA层很小
                              # 如果基础模型本身就放不下，device_map="auto" 仍然有帮助
        )
        # LoRA 通常不直接与 model.enable_input_require_grads() 一起使用
        # peft 库的 get_peft_model 会处理梯度的设置
        # model.enable_input_require_grads() # 通常在LoRA中不需要，由PEFT处理

    except Exception as e:
        print(f"从路径 {model_load_path} 加载基础模型失败: {e}"); exit(1)

    # 配置 LoRA
    # 目标模块通常是 'q_proj', 'k_proj', 'v_proj', 'o_proj' (线性层)
    # 对于Qwen模型，可能需要查看模型结构确定具体名称，常见的还有 'gate_proj', 'up_proj', 'down_proj'
    # 使用 `print(model)` 可以查看模型结构
    print("打印模型结构以确定LoRA目标模块:")
    # print(model) # 取消注释以查看模型结构

    # 尝试Qwen系列常见的模块名，如果您的Qwen3-8B不同，需要调整
    # Qwen2 / Qwen1.5 通常是 'q_proj', 'k_proj', 'v_proj', 'o_proj'
    # 有些模型还包括 'gate_proj', 'up_proj', 'down_proj' 在FFN层
    # 对于Qwen3，需要确认。一个通用策略是针对所有 attention 投影和部分 FFN 层
    target_modules_qwen = [
        "q_proj", "k_proj", "v_proj", "o_proj", # Attention 投影
        # "gate_proj", "up_proj", "down_proj"  # FFN 层，可选
    ]
    # 检查模型中实际存在的模块名
    found_target_modules = []
    for name, _ in model.named_modules():
        for target in target_modules_qwen:
            if target in name.split('.')[-1] and target not in found_target_modules : #确保是叶子模块
                found_target_modules.append(target)
    
    if not found_target_modules:
        print(f"警告: 未在模型中找到预定义的LoRA目标模块: {target_modules_qwen}。请检查模型结构并更新 target_modules。")
        # 可以尝试一个更通用的策略，比如所有 'Linear' 层，但这可能不是最优的
        # found_target_modules = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and "lm_head" not in name]
        # print(f"将尝试使用找到的所有Linear层 (除lm_head外) 作为LoRA目标。这可能不是最优。")
        # target_modules_qwen = list(set([name.split('.')[-1] for name in found_target_modules])) # 获取唯一的模块名

    print(f"实际用于LoRA的目标模块: {found_target_modules if found_target_modules else '将由PEFT自动推断或需要手动指定'}")


    lora_config = LoraConfig(
        r=swanlab.config.lora_r,
        lora_alpha=swanlab.config.lora_alpha,
        target_modules=found_target_modules if found_target_modules else None, # 如果为空，PEFT可能尝试自动寻找或报错
        lora_dropout=swanlab.config.lora_dropout,
        bias="none",  # "none", "all", or "lora_only"
        task_type=TaskType.CAUSAL_LM, # 任务类型是因果语言模型
    )

    # 使用 PEFT 获取 LoRA 模型
    print("正在应用LoRA配置到模型...")
    model = get_peft_model(model, lora_config)

    # 打印可训练参数的数量和比例
    model.print_trainable_parameters()

    # 如果之前启用了梯度检查点，并且使用了 kbit training (8bit/4bit量化)
    # prepare_model_for_kbit_training 已经处理了梯度检查点
    # 如果没有用kbit，但仍想用梯度检查点（LoRA时通常可以不用，因为可训练参数少）
    # if training_args.gradient_checkpointing:
    # model.enable_input_require_grads() # 确保与梯度检查点兼容 (PEFT通常会处理)
    # model.gradient_checkpointing_enable() # 对于PEFT模型，通常在基础模型上启用

    # 对于LoRA，通常不需要手动调用 enable_input_require_grads()
    # 如果在TrainingArguments中设置了gradient_checkpointing=True，Trainer会处理

    # 2. 准备和处理训练集和验证集 (与之前相同)
    print("正在转换和处理训练集...")
    if not os.path.exists(SFT_TRAIN_FORMATTED_PATH):
        if not transform_dataset_for_chat_finetuning(ORIGINAL_TRAIN_DATA_PATH, SFT_TRAIN_FORMATTED_PATH):
            print(f"错误: 转换训练集 {ORIGINAL_TRAIN_DATA_PATH} 失败。"); exit(1)
    print("正在转换和处理验证集...")
    if not os.path.exists(SFT_EVAL_FORMATTED_PATH):
        if not transform_dataset_for_chat_finetuning(ORIGINAL_EVAL_DATA_PATH, SFT_EVAL_FORMATTED_PATH):
            print(f"错误: 转换验证集 {ORIGINAL_EVAL_DATA_PATH} 失败。"); exit(1)

    print("正在加载处理后的数据集...")
    try:
        raw_train_ds = load_dataset("json", data_files=SFT_TRAIN_FORMATTED_PATH, split="train")
        raw_eval_ds = load_dataset("json", data_files=SFT_EVAL_FORMATTED_PATH, split="train")
    except Exception as e: print(f"加载格式化数据集失败: {e}"); exit(1)
    print(f"原始训练集大小: {len(raw_train_ds)}, 原始验证集大小: {len(raw_eval_ds)}")

    print("正在对数据集进行Tokenization...")
    tokenized_train_dataset = raw_train_ds.map(process_sft_data, remove_columns=raw_train_ds.column_names)
    tokenized_eval_dataset = raw_eval_ds.map(process_sft_data, remove_columns=raw_eval_ds.column_names)
    print(f"Tokenized 训练集大小: {len(tokenized_train_dataset)}, Tokenized 验证集大小: {len(tokenized_eval_dataset)}")
    if len(tokenized_train_dataset)>0: print(f"Tokenized 训练集样本 (input_ids[:50]): {tokenized_train_dataset[0]['input_ids'][:50]}")
    if len(tokenized_eval_dataset)>0: print(f"Tokenized 验证集样本 (input_ids[:50]): {tokenized_eval_dataset[0]['input_ids'][:50]}")


    # 3. 配置训练参数
    print("正在配置训练参数...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=swanlab.config.per_device_train_batch_size,
        per_device_eval_batch_size=swanlab.config.per_device_eval_batch_size,
        gradient_accumulation_steps=swanlab.config.gradient_accumulation_steps,
        eval_strategy=swanlab.config.eval_strategy,
        eval_steps=swanlab.config.eval_steps,
        logging_steps=10,
        num_train_epochs=swanlab.config.num_train_epochs,
        save_steps=200, # LoRA模型小，可以更频繁保存
        save_total_limit=3,
        learning_rate=swanlab.config.learning_rate,
        # save_on_each_node=True, # 对于LoRA，通常在主进程保存即可
        # gradient_checkpointing=True, # 如果基础模型很大且没用kbit量化，可以启用
                                    # 注意: PEFT模型使用梯度检查点需要特定设置，
                                    # 通常在 prepare_model_for_kbit_training 或基础模型上启用
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) and torch.cuda.is_available(),
        report_to="swanlab",
        run_name=f"{MODEL_ID.split('/')[-1]}-LoRA-sft",
        logging_dir=LOGGING_DIR,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # optim="adamw_bnb_8bit" # 如果想用8bit优化器，可以配合LoRA
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)

    # 使用 Trainer (不需要 CustomTrainer，因为 PEFT 模型应该能正确处理设备)
    trainer = Trainer(
        model=model, # 现在是 PEFT 模型
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )

    # 4. 开始训练
    print("开始LoRA微调 (包含评估)...")
    trainer.train()

    # 5. 保存最终模型 (LoRA适配器)
    print("正在保存最终 (或最佳) LoRA适配器...")
    # 对于PEFT模型，保存的是适配器权重，而不是整个模型
    final_adapter_path = os.path.join(OUTPUT_DIR, "best_lora_adapter" if training_args.load_best_model_at_end else "final_lora_adapter")
    model.save_pretrained(final_adapter_path) # 这会保存LoRA的 adapter_config.json 和 adapter_model.bin
    tokenizer.save_pretrained(final_adapter_path) # 也保存tokenizer配置，方便后续加载
    print(f"LoRA适配器已保存至: {final_adapter_path}")

    # 如果想保存合并后的完整模型 (可选)
    # print("正在合并LoRA权重到基础模型并保存 (可选)...")
    # merged_model = model.merge_and_unload() # 合并权重并卸载PEFT层，得到一个标准的HF模型
    # merged_model_path = os.path.join(OUTPUT_DIR, "merged_model_checkpoint")
    # merged_model.save_pretrained(merged_model_path)
    # tokenizer.save_pretrained(merged_model_path)
    # print(f"合并后的完整模型已保存至: {merged_model_path}")
    # model = merged_model # 后续预测使用合并后的模型


    # 6. 测试/预测 (使用验证集的前几个样本进行示例预测)
    # 预测时，可以直接使用PEFT模型，或者使用上面合并后的模型
    print("正在运行示例预测 (使用LoRA模型)...")
    try:
        test_df_for_prediction = pd.read_json(SFT_EVAL_FORMATTED_PATH, lines=True, nrows=3)
        test_text_list_for_swanlab = []
        if not test_df_for_prediction.empty:
            for index, row in test_df_for_prediction.iterrows():
                system_prompt_content, user_input_content, ground_truth_output = row['system_prompt'], row['user_content'], row['assistant_response']
                print(f"\n--- 正在预测验证集样本 {index+1} (LoRA) ---"); print(f"用户输入内容: {user_input_content[:200]}...")
                llm_response = predict_with_model(system_prompt_content, user_input_content, model, tokenizer) # model 现在是 PEFT 模型
                response_log = f"===== 验证集样本 {index+1} (LoRA) =====\n用户输入:\n{user_input_content}\n---\n期望输出:\n{ground_truth_output}\n---\n模型预测:\n{llm_response}\n==="
                print(response_log); test_text_list_for_swanlab.append(swanlab.Text(response_log, caption=f"验证集LoRA预测_{index+1}"))
            if test_text_list_for_swanlab: swanlab.log({"验证集LoRA推理示例": test_text_list_for_swanlab})
        else: print("警告: 未能从验证集加载样本进行预测。")
    except Exception as e: print(f"在预测或记录到 SwanLab 时发生错误: {e}")

    swanlab.finish()
    print("LoRA微调和评估完成。")