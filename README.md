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

当前版本尚未实现完整 RAG、知识库 Tool、Dense Retrieval、RRF、多轮记忆、
LangChain/LangGraph、MCP、多 Agent 或云部署。
