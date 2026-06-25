# 大数据分析课程论文数据

本仓库用于保存课程论文《大语言模型能否改变金融投诉的表达方式吗？来自 CFPB 消费者投诉数据的文本分析证据》的代码、数据、图表和 LaTeX 文稿。

## 数据来源

原始数据来自美国消费者金融保护局 CFPB 的公开数据：

- Consumer Complaint Database: https://www.consumerfinance.gov/data-research/consumer-complaints/
- CFPB 数据字段说明: https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/

仓库中的样本用于课程论文复现实验，覆盖 2015--2026 年消费者金融投诉文本、产品类别、公司、州、提交时间和企业响应结果等字段。

## 目录结构

- `data/raw/`: 原始或下载后的 CFPB 投诉数据样本。
- `data/processed/`: 清洗后特征、建模数据和主题标签数据。
- `scripts/`: 数据获取、清洗、建模和绘图脚本。
- `outputs/tables/`: 描述统计、模型指标、主题结果和特征重要性表。
- `outputs/models/`: 预测模型评估结果。
- `figures/`: 论文中使用的正式图表。
- `llm_financial_complaints.tex`: 论文 LaTeX 源文件。
- `references.bib`: 参考文献数据库。
- `llm_financial_complaints.pdf`: 当前编译后的论文 PDF。

## 复现方式

推荐使用 Python 3.10 或更高版本。可按以下顺序运行：

```bash
python scripts/run_pipeline.py --source hf
```

若需要分步运行，可执行：

```bash
python scripts/01_fetch_data.py
python scripts/02_clean_features.py
python scripts/03_profile_and_models.py
python scripts/06_draw_algorithm_figures.py
```

论文 PDF 使用 XeLaTeX 编译：

```bash
latexmk -xelatex llm_financial_complaints.tex
```

## 说明

本文的经验结果主要用于课程论文分析和可视化展示。由于公开投诉数据并非随机实验数据，论文中关于 LLM、文本表达变化与企业响应之间的关系主要解释为相关性证据，而不是严格因果结论。
