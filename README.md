# AutoResearch

一个用于自动化学术论文收集、分类和分析的智能工具。支持从多个顶级 AI/ML 会议自动爬取论文，通过大语言模型进行领域分类，并提供向量化分析和聚类功能。

## 功能特性

- **多会议支持**: 支持 ACL、EMNLP、ICLR、CVPR、ICML、ICCV、AAAI、NIPS/NeurIPS 等顶级会议
- **自动化爬取**: 从会议官网自动提取论文标题、作者、摘要和 PDF 链接
- **智能分类**: 利用大语言模型（DeepSeek/Kimi）对论文进行研究领域分类
- **向量化分析**: 使用文本嵌入模型生成论文向量表示
- **聚类分析**: 基于 K-means 算法对论文进行主题聚类
- **关键词提取**: 自动提取研究领域的关键词和核心观点

## 项目结构

```
AutoResearch/
├── api.py                  # API 封装，支持 DeepSeek 和 Kimi 模型
├── utils.py                # 工具函数，包含领域选择器 Field_Selector
├── requirements.txt        # 项目依赖
├── run.sh                  # 运行脚本示例
├── fields/                 # 研究领域定义
│   ├── fields.json         # 原始领域列表
│   ├── filtered_fields.json # 处理后的领域定义（含解释和视角）
│   └── fulfill_fields.py   # 使用 LLM 填充领域描述
├── web/                    # 论文爬取模块
│   ├── web.py              # 各会议爬虫实现
│   ├── crawl.py            # 爬取入口
│   └── info.json           # 会议 URL 配置
├── paper/                  # 论文数据处理（各 step 子目录存储中间结果）
├── scripts/                # 辅助脚本
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `openai` - 用于调用 DeepSeek 和 Kimi API
- `openreview-py` - 用于爬取 OpenReview 平台论文（ICLR、NIPS）
- `beautifulsoup4` - HTML 解析
- `requests` - HTTP 请求
- `transformers` - 文本嵌入模型
- `torch` - PyTorch 深度学习框架
- `tqdm` - 进度条显示

## 环境变量配置

在使用 API 功能前，需要设置以下环境变量：

```bash
# DeepSeek API
export DEEPSEEK_API_KEY="your_deepseek_api_key"

# Kimi API（可选）
export MOONSHOT_API_KEY="your_moonshot_api_key"

# OpenReview 账号（用于爬取 ICLR、NIPS 论文）
export OR_USERNAME="your_openreview_username"
export OR_PASSWORD="your_openreview_password"
```

## 使用说明

### 1. 爬取论文数据

```bash
python -m web.crawl --conference ICLR --year 2024
```

支持的会议：`ACL`、`EMNLP`、`ICLR`、`CVPR`、`ICML`、`ICCV`、`AAAI`、`NIPS`

### 2. 领域分类（Step 1）

使用 LLM 自动为论文标注研究领域：

```bash
python -m paper.step1.annotate_field --conference ICLR --year 2024
```

### 3. 数据标准化（Step 2）

```bash
python paper/step2/standarize.py --conference ICLR --year 2024
```

### 4. 下载嵌入模型（用于 Step 5）

项目使用 `nvidia/NV-Embed-v2` 模型生成论文嵌入向量。请先下载模型：

```bash
# 默认下载到 models/NV-Embed-v2
python scripts/download_nv_embed.py

# 或指定自定义路径
python scripts/download_nv_embed.py --output-dir /path/to/your/model
```

**环境变量**: 如果模型需要 HuggingFace token，请设置：
```bash
export HUGGINGFACE_HUB_TOKEN="your_hf_token"
# 或
export HUGGINGFACE_TOKEN="your_hf_token"
```

### 5. 后续处理步骤

项目采用流水线式处理，各步骤依次执行：

- **Step 3**: 论文摘要总结
- **Step 4**: 数据规范化处理
- **Step 5**: 生成文本嵌入向量（需要指定模型路径）
- **Step 6**: K-means 聚类分析
- **Step 7**: 关键词提取

#### Step 5 使用说明

生成文本嵌入向量（默认使用 `models/NV-Embed-v2`）：

```bash
python paper/step5/embed.py --field "AI Agents"
```

如果使用自定义模型路径：

```bash
python paper/step5/embed.py --field "AI Agents" --model-path /path/to/your/model
```

### 6. 按领域检索论文

```python
from utils import field_to_paper

# 获取指定会议年份中特定领域的论文
papers = field_to_paper("ICLR", 2024, "Robotics")
print(f"找到 {len(papers)} 篇相关论文")
```

## 项目结构补充说明

```
models/                     # 存放下载的嵌入模型（默认 NV-Embed-v2）
└── NV-Embed-v2/           # 由 scripts/download_nv_embed.py 下载
```

## 研究领域定义

项目预定义了多个 AI/ML 研究领域，存储在 `fields/filtered_fields.json` 中。每个领域包含：

- `field`: 领域名称
- `abbr`: 缩写
- `explanation`: 领域解释和关键指标
- `perspectives`: 阅读该领域论文时的核心视角

示例领域：
- Machine Learning (ML)
- Deep Learning (DL)
- Reinforcement Learning (RL)
- Natural Language Processing (NLP)
- Computer Vision (CV)
- Robotics
- ...

## API 模块说明

### API 类 (`api.py`)

封装了大语言模型调用：

```python
from api import API

# 使用 DeepSeek
api = API(model_name="deepseek")
response = api.forward("你的提示词")

# 使用 Kimi（支持文件上传）
api = API(model_name="kimi")
response = api.forward("你的提示词", file="document.pdf")
```

### Embed 类 (`api.py`)

文本嵌入生成：

```python
from api import Embed

embedder = Embed(model_path="your_embedding_model")
vectors = embedder.encode(["文本1", "文本2"])
```

## 批量处理脚本

参考 `run.sh` 编写批量处理命令：

```bash
# 批量处理多个会议和年份
python -m paper.step1.annotate_field --conference NIPS --year 2023
python -m paper.step1.annotate_field --conference NIPS --year 2024
python -m paper.step1.annotate_field --conference ACL --year 2023
# ...
```

## 数据格式

### 论文数据格式

论文数据以 JSON 格式存储，包含以下字段：
- `paper_id`: 论文唯一标识（如 "ICLR2024-00001"）
- `title`: 论文标题
- `authors`: 作者列表
- `pdf_url`: PDF 下载链接
- `abstract`: 论文摘要
- `fields`: 所属研究领域列表

## 注意事项

1. **API 限制**: 调用 DeepSeek/Kimi API 时注意速率限制和费用
2. **网络访问**: 爬取论文需要稳定的网络连接，部分会议网站可能需要科学上网
3. **OpenReview 认证**: 爬取 ICLR 和 NIPS 论文需要有效的 OpenReview 账号
4. **数据缓存**: 中间结果保存在各 step 目录中，可避免重复处理

## 许可证

本项目仅供学术研究使用。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。
