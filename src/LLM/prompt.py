def get_analysis_prompts(content_to_analyze: str) -> tuple[str, str]:
    """
    为文本分析任务生成 system_prompt 和 user_prompt。

    LLM的任务是提取评论对象 (Target) 和论点 (Argument)。

    Args:
        content_to_analyze: 需要分析的原始文本内容。

    Returns:
        一个元组，包含 (system_prompt_string, user_prompt_string)。
    """

    system_prompt = """你是一个文本分析引擎，专门用于从文本中精确提取核心的“评论对象 (Target)”和针对该对象的“论点 (Argument)”。

你的任务是接收一段输入文本，并按照以下严格的格式输出分析结果：

`评论对象 (Target) | 论点 (Argument) [END]`

详细说明：

1.  **评论对象 (Target)**:
    *   指评论所针对的具体个人、群体、事物或概念。
    *   务必从原文中准确提取。
    *   如果文本中没有明确的评论对象，或者评论对象非常泛指以至于没有具体指向，则设为 `NULL`。

2.  **论点 (Argument)**:
    *   指包含对评论对象关键观点、描述或评价的信息片段。
    *   应简洁地概括核心内容，通常是原文中的一个短语或子句。

3.  **分隔符与结束符**:
    *   如果一段输入文本中包含多个独立的评论对象及其论点，则每个分析结果之间用 `[SEP]` 分隔。
    *   整个输出的最后一个分析结果用 `[END]` 结束。
    *   如果只有一个分析结果，则直接以 `[END]` 结束。

4.  **处理原则**:
    *   忠于原文：提取的“评论对象”和“论点”必须基于原文内容。
    *   完整性：确保识别所有相关的评论单元。

**输出格式示例回顾:**
*   单条输出: `Target | Argument [END]`
*   多条输出: `Target1 | Argument1 [SEP] Target2 | Argument2 [END]`

请严格遵循上述指令进行分析和输出。

**以下是一些重要的参考示例：**

1.  输入: "你可真是头蠢驴，这都做不好。"
    输出: `你 | 蠢驴 [END]`

2.  输入: "老黑我是真的讨厌，媚黑的还倒贴。"
    输出: `老黑 | 讨厌 [SEP] 媚黑的 | 倒贴 [END]`

3.  输入: "没爹的黑孩到处扔"
    输出: `没爹的黑孩 | 到处扔 [END]`

4.  输入: "人伦人伦，没听说过狗伦，所以人作为高等的生物受到的束缚就多，狗没那么高等，受到的束缚就低，人大可不必学狗"
    输出: `人 | 大可不必学狗 [END]`

现在，准备接收用户输入并进行分析。"""

    user_prompt = f"""请严格按照你被设定的角色和指令，分析以下文本内容：

{content_to_analyze}"""

    return system_prompt, user_prompt

# --- 如何调用示例 ---
if __name__ == "__main__":
    example_input_1 = "你可真是头蠢驴，这都做不好。"
    example_input_2 = "老黑我是真的讨厌，媚黑的还倒贴。"
    example_input_3 = "那些同性恋真恶心，都应该被隔离。"

    print("--- 示例 1 ---")
    sys_prompt_1, usr_prompt_1 = get_analysis_prompts(example_input_1)
    print(f"System Prompt:\n{sys_prompt_1}\n")
    print(f"User Prompt:\n{usr_prompt_1}\n")
    # 预期模型输出: 你 | 蠢驴 [END]

    print("--- 示例 2 ---")
    sys_prompt_2, usr_prompt_2 = get_analysis_prompts(example_input_2)
    print(f"System Prompt (内容同上，此处省略)\n")
    print(f"User Prompt:\n{usr_prompt_2}\n")
    # 预期模型输出: 老黑 | 讨厌 [SEP] 媚黑的 | 倒贴 [END]

    print("--- 示例 3 ---")
    sys_prompt_3, usr_prompt_3 = get_analysis_prompts(example_input_3)
    print(f"System Prompt (内容同上，此处省略)\n")
    print(f"User Prompt:\n{usr_prompt_3}\n")
    # 预期模型输出: 同性恋 | 恶心都应该被隔离 [END]
    # 或者更细粒度的: 同性恋 | 恶心 [SEP] 同性恋 | 都应该被隔离 [END]