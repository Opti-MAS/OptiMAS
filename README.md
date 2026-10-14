# OptiMAS

**OptiMAS** (Optimizing Multi-Agent Systems) is a framework for automatically
evolving and optimizing multi-agent systems through a training loop that combines
forward execution, evaluation, and LLM-driven backward optimization.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     OptiMAS Client                      │
│                                                         │
│  CLI (train / test)                                     │
│    ├── MASForwardEngine  ──► ReAct Agents               │
│    │     └── MCPClient   ──────────────┐                │
│    ├── UniversalEvaluator              │                │
│    └── OptimizerAgent (backward pass)  │                │
│                                        │                │
├────────────────────────────────────────┼────────────────┤
│                     MCP (HTTP)         │                │
├────────────────────────────────────────┼────────────────┤
│                     OptiMAS Server     ▼                │
│                                                         │
│  FastMCP + FastAPI                                      │
│    ├── File Operations (read, write, edit, grep, ...)   │
│    ├── Workspace Operations (list, delete, move)                   │
│    ├── Session Management (handshake, workspace, ...)   │
│    └── WorkBench Tools (calendar, email, CRM, ...)      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

The system operates in two phases:

1. **Forward pass**: The MAS executes tasks by calling tools through an MCP
   server. Agents follow a ReAct loop (reason, act, observe) with configurable
   skills and prompts defined in YAML.

2. **Backward pass**: An optimizer agent analyzes trajectories from the forward
   pass, identifies failure patterns, and modifies the MAS configuration
   (prompts, skills, tool sets, agent architecture) to improve performance.

## Quick Start

### Prerequisites

- Python 3.10+
- An LLM API key (OpenAI, Gemini, or compatible endpoint)

### 1. Install Dependencies

```bash
# Client + shared dependencies
pip install -e .

# Server dependencies
pip install -r optimas_server/requirements.txt
```

### 2. Fetch the WorkBench Benchmark

WorkBench is third-party (MIT) and is **not** vendored into this repository. Clone it
into the location the server expects:

```bash
git clone https://github.com/olly-styles/WorkBench.git optimas_server/workbench
git -C optimas_server/workbench checkout --detach 42f1b6dc18986f47701ca83de2d63dd4f95c59a0
```

This pinned commit provides the WorkBench API and sandbox state used by this release.
Later upstream revisions change that API. The server registers 27 domain tools at
startup; OptiMAS installs their runtime dependencies.

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Start the MCP Server

```bash
cd optimas_server
python -m src.main
```

The server starts at `http://127.0.0.1:8188`. Client and server share the repository
`workspace/` directory. On separate hosts, mount the same storage at their configured
paths. The dashboard requires an API key: use any username and your MCP key as the
HTTP Basic password. Replace the development key and session secret before remote
deployment.

### 5. Run Evaluation

```bash
python -m optimas_client.cli.test \
    --mas-config optimas_client/config/mas_workbench \
    --task-id workbench \
    --mcp-key optimas-local-dev-key \
    --n-trials 1
```

### 6. Run Training (MAS Evolution)

```bash
python -m optimas_client.cli.train \
    --task-id my-experiment \
    --mcp-key optimas-local-dev-key \
    --optimizer-config optimas_client/config/optimizer_workbench/config.yaml
```

## Project Structure

```
optimas_client/         # Agent client: training, evaluation, LLM integrations
  ├── cli/              #   CLI entry points (train.py, test.py)
  ├── config/           #   YAML configurations, prompts, skills
  ├── data/             #   Benchmark datasets (JSONL)
  └── src/              #   Core source code
      ├── agent/        #     ReAct agent framework
      ├── config/       #     Configuration loader
      ├── data/         #     Dataset loaders
      ├── engine/       #     MAS forward execution engine
      ├── evaluation/   #     Pluggable evaluation backends
      ├── llm/          #     LLM provider integrations (OpenAI, Gemini)
      ├── mcp/          #     MCP client
      ├── messaging/    #     Structured logging / trajectory recording
      ├── optimizer/    #     Backward-pass optimizer agent
      └── workbench/    #     WorkBench benchmark integration

optimas_server/         # MCP tool server
  ├── src/              #   Server source code
  │   ├── config/       #     Pydantic settings
  │   ├── middleware/   #     Auth, session, logging, dashboard
  │   ├── tools/        #     Tool implementations
  │   └── utils/        #     Logging, error handling
  ├── workbench/        #   WorkBench benchmark (third-party; cloned at setup)
  ├── client/           #   Example MCP clients
  └── docs/             #   Server documentation
```

## Configuration

OptiMAS uses a layered YAML configuration system:

- **`optimas_client/config/global_config.yaml`** -- Global settings (MCP endpoint,
  messaging, API usage recording)
- **`optimas_client/config/mas_workbench/config.yaml`** -- MAS agent definitions
  (prompts, skills, tools, iteration limits)
- **`optimas_client/config/mas_workbench/agents_llm_config.yaml`** -- Per-agent
  LLM model settings (separated for easy swapping)
- **`optimas_server/config.yaml`** -- Server settings (port, auth, sessions)

Secrets (API keys, endpoints) come from the environment; everything else — model
routing, timeouts, iteration budgets, skills, tool sets — lives in the YAML files
above. See [`.env.example`](.env.example) for the environment variables that are read.

## Supported Benchmarks

| Benchmark | Type | Evaluator |
|-----------|------|-----------|
| [WorkBench](https://arxiv.org/abs/2405.00823) | Workplace tasks | State-based (official) |

> **Scope note.** The paper evaluates OptiMAS on four benchmarks (WorkBench, GAIA-text,
> BrowseComp and SWE-Bench-Verified). This release ships the **WorkBench** pipeline only:
> it is fully self-contained and needs no web-search, crawler, or Docker infrastructure,
> which makes it the one configuration anybody can reproduce from a clean checkout. The
> optimizer itself is task-agnostic — adding a benchmark means supplying a dataloader
> under `optimas_client/src/data/`, an evaluator under `optimas_client/src/evaluation/`,
> and a MAS config directory.

## Reproducing the Paper

The defaults in this repository match the paper's WorkBench setup: 10 epochs, batch
capacity 10, seed 42, and an E0 iteration budget of 40 steps per phase / 100 total
(paper Appendix C.1.1, D.1 and Table 6). The shipped split
(`workbench_train/val/test.jsonl`, 60/12/618 of 690) is the stratified `seed=42`
partition from Table 4.

```bash
python -m optimas_client.cli.train \
    --task-id workbench-repro \
    --mcp-key optimas-local-dev-key \
    --optimizer-config optimas_client/config/optimizer_workbench/config.yaml \
    --epochs 10 --batch-size 10 --seed 42
```

The complete, fixed validation split is evaluated at every step. The optimizer sees
only its accuracy and change from the previous step; validation trajectories stay
outside its workspace. Training does not load the held-out test file.

`last_mas_config/` contains the final configuration E_T returned by the paper's
algorithm. `best_mas_config/` is an additional convenience export of a configuration
that was actually evaluated on validation. Each metric refers to the configuration
before that step's update. Use the final configuration when reproducing the paper.

Two deliberate differences from the paper's runs:

- **Backbones.** Every config ships with `gpt-4o` so the quickstart works with a single
  OpenAI-compatible key. The paper uses GPT-5-Nano, Qwen3.6-35B-A3B and Gemini-3-Flash
  as MAS backbones, with Gemini-3-Flash as the optimizer (Appendix C.1.2). Set these in
  `optimas_client/config/mas_workbench/agents_llm_config.yaml` and
  `optimas_client/config/optimizer_workbench/config.yaml`.
- **Trials.** Reported test numbers are averaged over three independent runs; pass
  `--n-trials 3` to `optimas_client.cli.test`.

Prompt wording and skill organization have been simplified for this release. The
example is a runnable WorkBench configuration; reproducing reported scores requires
the paper's backbones and experimental settings.

Benchmark runs require the pinned WorkBench checkout and model credentials. API
recording contains full model messages when enabled; keep runtime directories private.

## Credits

- **WorkBench** benchmark by [MindsDB](https://mindsdb.com/) -- MIT License,
  from [olly-styles/WorkBench](https://github.com/olly-styles/WorkBench). Cloned at
  setup time rather than vendored; see
  [`optimas_server/THIRD_PARTY_NOTICE.md`](optimas_server/THIRD_PARTY_NOTICE.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
