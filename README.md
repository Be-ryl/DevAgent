# DevAgent

一个用于学习最小 Tool Calling 链路的本地 Agent。当前版本可以把消息和工具
Schema 发送给 DeepSeek，在模型请求工具时执行工作区内的文件或命令工具，再将
工具结果回填给模型。每条命令都是独立请求，不包含多轮记忆。

## 本地运行

创建并启用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
```

安装项目：

```bash
python -m pip install -e .
```

从模板创建本地配置文件：

```bash
cp .env.example .env
```

然后只在本机编辑 `.env`，把占位值替换为真实密钥：

```dotenv
DEEPSEEK_API_KEY="..."
```

`.env` 已被 `.gitignore` 排除，不能提交；`.env.example` 只保留占位值，可以正常
上传 GitHub。CLI 从项目当前目录自动加载 `.env`。如果系统环境中已经设置了
`DEEPSEEK_API_KEY`，系统环境变量优先，不会被 `.env` 覆盖。

运行一条普通对话：

```bash
python -m devagent "你好"
```

指定工具工作区，并明确请求使用已有的文件工具：

```bash
python -m devagent --workspace . "请调用 read_file 工具读取 README.md，并概括内容。"
```

启用本地旅游政策知识库，让模型可以调用 `knowledge_search`：

```bash
python -m devagent \
  --workspace /Users/xiaomaibufubuzhang/devagent \
  --knowledge-dir /Users/xiaomaibufubuzhang/devagent/data/policies \
  "请查询知识库，告诉我托运行李超重如何收费，并说明信息来源。"
```

增加 `--trace` 可在 stderr 查看模型是否调用检索工具、命中的来源和分数：

```bash
python -m devagent \
  --trace \
  --workspace /Users/xiaomaibufubuzhang/devagent \
  --knowledge-dir /Users/xiaomaibufubuzhang/devagent/data/policies \
  "请查询知识库，告诉我托运行李超重如何收费，并说明信息来源。"
```

不传 `--trace` 时只输出最终回答。

安装后也可以使用等价的 `devagent` 命令。

## 测试

测试使用 Mock/Fake Provider，不需要访问真实模型 API：

```bash
python -m unittest discover -s tests -v
```

## 本地 BM25 检索演示

以下命令只读取仓库中的虚构政策数据，不调用模型 API：

```bash
python -m devagent.rag.service data/policies "酒店入住前一天取消收费吗？" --top-k 3
```

输出包含查询、命中的 Chunk、来源文件、BM25 分数和排名。BM25 分数只用于当前
查询内的相对排序，不是概率。

## BM25 离线测评

以下命令使用仓库内的人工标注题目计算来源 Recall@3 和 MRR，不调用模型 API：

```bash
python -m devagent.evaluation.retrieval_eval \
  --knowledge-dir data/policies \
  --cases data/evaluation/retrieval_cases.jsonl \
  --top-k 3
```

测评会先确认每条 `evidence_substring` 确实存在于标注的政策原文中，然后输出逐题
结果以及 exact、paraphrase、typo、noise 四类汇总指标。

当前版本的 `knowledge_search` 只使用 BM25；尚未实现 Dense Retrieval、RRF、
多轮记忆、LangChain/LangGraph、MCP、多 Agent 或云部署。
