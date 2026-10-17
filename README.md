# DevAgent

DevAgent 是面向复杂编码任务的多 Agent 代码执行平台。

当前同时提供可对比的 SingleAgent 与 MultiAgent 架构，并为两种模式统一提供上下文预算、
命令审批、Checkpoint/Resume、Trace 和运行指标。CLI 默认使用更稳定的 `single` 模式。

## 执行模式

### SingleAgent

`SingleAgentRunner` 只初始化一个 Agent，可按需使用 WebSearch、文件读写、局部编辑、
文本搜索和 Bash 命令工具，不会初始化 Supervisor、SearchAgent、CodeAgent 或内部
Verifier。

```bash
devagent --mode single --workspace . "读取项目并完成编码任务"
```

### MultiAgent

`MultiAgentRunner` 使用 LangGraph 组织以下流程：

```text
START → planner → VerificationExecutor → verifier → final → END
            ↑                              │
            └──────────────────────────────┘  验收失败且未达到最大尝试次数
```

Supervisor/Planner 生成 Todo、验收标准和验证命令，并按需把 SearchAgent 或 CodeAgent
作为工具调用；无需外部资料时可以跳过 SearchAgent。SearchAgent 只有 WebSearch，
CodeAgent 使用文件、文本搜索、Bash 和 Todo 工具。Planner 不能直接操作文件或执行命令。

验证命令由独立的 `VerificationExecutor` 按 Planner 的命令白名单执行。Verifier 模型
只有 `read_file` 和 `search_text`，只读取验证结果并判断是否满足验收标准，不能自行
构造或执行命令。达到最大尝试次数后工作流进入 Final 并明确报告失败。

```bash
devagent --mode multi --workspace . "实现功能并运行测试"
```

两种 Runner 共享模型 Provider、底层工具实现、workspace 安全边界、超时与循环预算、
上下文阈值、审批策略和 `RunResult` 返回结构。

## Context Engineering

`LayeredMemory` 将运行上下文划分为三层：

- `rules`：稳定规则与 workspace 边界；
- `working_memory`：任务、计划、Todo、验收标准、验证命令、来源、handoff、修改文件和最近错误；
- `history_summary`：被压缩的历史信息。

`ContextMonitor` 仅在估算 Token 超过 `DEVAGENT_CONTEXT_TOKEN_LIMIT` 时触发
`ContextCompressor`。压缩会用分层记忆真正替换旧消息，并记录
`context_compressions`。Single 与 Multi 默认使用同一个阈值：

```dotenv
DEVAGENT_CONTEXT_TOKEN_LIMIT=120000
```

## 命令审批

SingleAgent、CodeAgent 和 VerificationExecutor 共享 `ApprovalPolicy`。安装依赖、访问
网络、启动后台服务和破坏性命令被视为高风险：

```bash
devagent --approval-mode inline --workspace . "执行任务"  # 由 CLI 询问
devagent --approval-mode deny --workspace . "执行任务"    # 直接拒绝
devagent --approval-mode auto --workspace . "执行任务"    # 受控环境自动通过
```

核心审批逻辑使用可注入回调，不依赖终端输入；`input()` 只存在于 CLI 交互适配层。

## Checkpoint 与恢复

每次运行会在 workspace 中保存：

```text
.devagent/checkpoints/
├── checkpoint.json
└── RECOVERY.md
```

Checkpoint 只保存 JSON 可序列化的恢复状态，不保存 API Client、模型对象或回调函数。
中断后可从同一 workspace 恢复；未显式传入 `--mode` 时，会沿用 checkpoint 中的模式：

```bash
devagent --resume /path/to/workspace
devagent --resume /path/to/workspace --approval-mode deny
```

## Trace 与指标

Single 与 Multi 使用相同的 Trace 格式，每次运行生成独立目录：

```text
.devagent/traces/run-20261017-120000-a1b2c3d4/
├── events.jsonl
├── summary.json
└── timeline.md
```

事件可包含 `run_start`、`run_end`、`model_call`、`tool_call`、`tool_result`、
`handoff`、`approval`、`context_compression`、`verifier_result`、`checkpoint`
和 `error`。Trace 仅用于观测，不参与 Agent 决策。

`RunMetrics` 记录耗时、模型调用、工具调用、输入/输出 Token、重试次数、执行模式、
完成状态、错误、上下文压缩次数、审批请求与拒绝数、Checkpoint 数、Trace ID 和修改
文件数量。Bash 创建、修改或删除的文件通过运行前后 workspace 快照差异计入
`changed_files`。

## 工具与安全边界

- 文件工具支持读取、写入、唯一匹配局部编辑和递归文本搜索；
- 文件路径拒绝绝对路径、父目录穿越和符号链接逃逸；
- Bash 命令不经过 Shell，并以 workspace 为工作目录；
- Verifier 不持有 Bash 或写文件工具；
- `.devagent` 运行数据不计入任务修改文件并被 Git 忽略。

## 安装与配置

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
```

本地 `.env` 使用以下变量：

```dotenv
DEEPSEEK_API_KEY="..."
DEEPSEEK_BASE_URL="https://api.deepseek.com"
DEEPSEEK_MODEL="deepseek-chat"
TAVILY_API_KEY="..."
DEVAGENT_CONTEXT_TOKEN_LIMIT=120000
```

`.env` 已被 Git 忽略，不能提交真实密钥。只有实际调用 WebSearch 时才需要
`TAVILY_API_KEY`。DeepSeek 继续通过 OpenAI-compatible 接口调用。

## 公平对比原则

比较 SingleAgent 与 MultiAgent 时，应使用相同的任务文本、初始 workspace 快照、
模型配置、超时、循环预算、上下文阈值和审批模式，并为每次运行使用独立指标实例。
MultiAgent 内部 Verifier 只负责该工作流的验收回环，不作为两种架构对比的最终裁判；
项目保留独立 `Evaluator` 接口，正式评测集留到后续阶段。

## 测试

测试使用 Fake/Mock 模型、搜索与审批回调，不访问真实模型或搜索 API：

```bash
python -m unittest discover -s tests -v
git diff --check
```

当前不实现 Textual TUI、正式评测集、独立 Evaluator 或检索知识库。
