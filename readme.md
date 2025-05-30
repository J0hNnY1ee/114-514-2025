HITSZ-NLP-2025
=====
CCL25-Eval 任务10：细粒度中文仇恨识别评测 & 2025年HITSZ自然语言处理大作业
## 使用方法

1. 先执行

   ```
   pip install -e .
   ```

2. 用 `training` 目录下三个文件来训练模型

3. 用 `pipeline.py` 来生产四元组
4. 用 `prepare_eval_data.py`生产评估对
5. 评估模型性能 `run_evaluation.py`

## 文件组织形式

```
.
├── data
│   ├── original
│   │   ├── demo.txt
│   │   ├── test1.json
│   │   ├── test2.json
│   │   └── train.json
│   ├── outputs
│   └── segment
│       ├── test.json
│       ├── train.json
│       └── val.json
├── main.py
├── models
│   ├── chinese-roberta-wwm-ext-large
│   ├── hfd.sh
│   └── outputs
├── requirements.txt
├── setup.py
├── src
│   ├── core
│   │   ├── group_classifier.py
│   │   ├── hate_classifier.py
│   │   ├── information_extractor.py
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   ├── data_process
│   │   ├── group_clf_dataset.py
│   │   ├── hate_clf_dataset.py
│   │   ├── ie_dataset.py
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── preprocessor.py
│   ├── evaluation
│   │   ├── evaluator.py
│   │   ├── __init__.py
│   │   ├── prepare_eval_data.py
│   │   └── run_evaluation.py
│   ├── __init__.py
│   ├── models
│   │   ├── group_clf_model.py
│   │   ├── hate_clf_model.py
│   │   ├── ie_model.py
│   │   ├── __init__.py
│   ├── NLP_2025.egg-info
│   │   ├── dependency_links.txt
│   │   ├── PKG-INFO
│   │   ├── SOURCES.txt
│   │   └── top_level.txt
│   ├── segment.py
│   ├── training
│   │   ├── group_clf_train.py
│   │   ├── hate_clf_train.py
│   │   ├── ie_train.py
│   │   ├── __init__.py
│   │   └── trainer_base.py
│   └── utils
│       └── jsonljson.py
└── tree.log
```

