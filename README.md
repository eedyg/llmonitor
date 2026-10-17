# llm-monitor

统一控制面 + 观测面，同时支持 **vLLM** 与 **NInfer**（通过上一层目录的 `../run_llm.sh` 编排）。

一个进程、一个网页、一套 MCP 工具：起停 / 切换引擎、看实时吞吐与延迟、看每一条请求的
prefill/thinking/output/缓存命中、看 GPU 功耗温度风扇显存降频。

```
┌──────────────────────────── browser / OpenAI clients / MCP ────────────────────────────┐
│                                                                                        │
│   http://host:9000  (UI + REST + WS)          http://host:8000  (OpenAI/Anthropic API) │
└───────────────┬───────────────────────────────────────────────┬────────────────────────┘
                │                                               │
        ┌───────▼───────────────────────────────────────────────▼───────┐
        │                     llm-monitor (python)                      │
        │  API/WS ──── State Store (ring 2h + SQLite 30d) ──── /metrics │
        │     ▲                 ▲            ▲                          │
        │  Controller     Collectors     Gateway Tap (0 buffer)         │
        │  run_llm.sh     /metrics 2s    passthrough + SSE tee          │
        │  start/stop     NVML 1s        per-request accounting         │
        │                 JSONL tail                                    │
        └───────┬──────────────┬────────────────────────────────────────┘
                │              │ docker / pid
        ┌───────▼───────┐  ┌───▼──────────────┐
        │ vllm-qwen38   │  │ ninfer-serve     │   ← 引擎只绑定 127.0.0.1:8001
        │ :8001         │  │ :8001            │
        └───────────────┘  └──────────────────┘
```

## 为什么需要它

| | vLLM 0.31 | NInfer |
|---|---|---|
| 自带 Web UI | ❌ | ❌ |
| 机器可读指标 | `GET /metrics`（Prometheus） | `GET /metrics` + 逐请求 JSONL + stderr 吞吐 |
| 逐请求明细 | **没有**（只有直方图） | ✅ `--request-log-jsonl` |
| 在飞请求阶段 | 只有聚合 | ✅ prefill/decode/paused 分开 |

两个引擎都有指标，但都没有页面；vLLM 还缺逐请求数据。所以 monitor 站在**数据路径上**：
它占住公网端口，把字节原样透传给引擎，同时在旁路解析一份。这样两个引擎在 UI 里长一个样。

## 快速开始

```bash
cd ~/workspace/llm/monitor
./run.sh                      # 默认 config.yaml
# → UI      http://<lan-ip>:9000
# → gateway http://<lan-ip>:8000   （客户端 base_url 指这里，其余不变）
```

首次启动时如果引擎还占着 `:8000`，会直接报错并给出处理办法，因为 monitor 要独占
`:8000`、引擎必须退到 `:8001`。停止旧引擎后由 UI（或 MCP）重新拉起即可：

```bash
../run_llm.sh --engine vllm --stop
./run.sh
```

之后所有起停都在 UI 里点，或者：

```bash
curl -X POST localhost:9000/api/engine/start \
  -H 'content-type: application/json' \
  -d '{"engine":"ninfer","profile":"mtp","ctx":"256k"}'
```

> 说明：monitor 启动的引擎等价于
> `HOST_PORT=8001 COMPAT=0 ../run_llm.sh --engine X --profile Y --host 127.0.0.1 -- ...`
> 不需要修改 `run_llm.sh`。

## 端口

| 端口 | 用途 | 谁在用 |
|---|---|---|
| 9000 | 控制面：网页 + REST + WebSocket | 人 / 自动化脚本 |
| 8000 | 数据面：OpenAI / Anthropic 兼容网关 | 所有 LLM 客户端 |
| 8001 | 当前引擎（仅 127.0.0.1） | monitor 内部 |
| 9001 | 可选：MCP streamable-http | agent |

客户端只认 `:8000`，切换引擎对客户端完全透明（`run_llm.sh` 的契约：同一 model 名、同一
协议、同一路径）。

## 引擎启动（P2b：已接管）

启动命令现在由声明式 spec 生成，controller **默认直接执行 `LaunchPlan`**，不再经过
`run_llm.sh`：

```bash
./run.sh engine --engine vllm --profile mtp --ctx 128k      # 打印计划（不执行）
./run.sh engine --engine ninfer --profile mtp --ctx 256k --json
./run.sh engine --engine vllm --profile stage1 --ctx 128k --kv-cache-dtype fp8_e5m2
```

参数由 spec 的 `options:` 声明（`key` 就是 launcher 变量名），UI 通过
`GET /api/engines` 拿到字段定义、通过 `POST /api/engine/plan` 预览；加字段不需要改前端。

* `monitor/engines/vllm.yaml` / `ninfer.yaml`：profiles、mounts、env、条件参数、preflight 规则、**启动阶段正则**
* `monitor/launcher.py`：`resolve(engine, opts, env) -> LaunchPlan{argv, mounts, env, labels, warnings, errors}`

回退开关：`MONITOR_ENGINE_LAUNCHER=0`（或 `engine.launcher: false`）→ 退回旧的
`../run_llm.sh` 路径。两条路径共用同一个状态机。

启动过程不再是黑盒：

* `engines/*.yaml` 的 `startup:` 正则把引擎日志变成阶段进度
  （vLLM：初始化引擎 → 加载权重 → 预热 → 捕获 CUDA Graph → 启动 API → 就绪；
  NInfer：初始化 → 加载权重 → 权重就绪 → CUDA Graph → 引擎就绪 → 监听中）
* 失败时直接显示最后一行日志 + 轨迹（`PhaseChip` 变红）
* preflight 错误（镜像缺失、ctx 超窗口、dflash 权重缺失…）在启动前返回可读原因
* 停止：容器用 `docker stop -t 25` + `rm`（优雅释放显存）；进程用 SIGTERM→SIGKILL，
  并等待监听端口真正释放；启动前自动清理同名 exited 容器和 TIME-WAIT（NInfer 不能
  rebind 的限制）

等价性由
`tools/parity.py` 证明：它把假 `docker` / 假 `ninfer-serve` 放进 PATH，让 `run_llm.sh`
走真实启动路径并抓取 argv（NUL 分隔，能正确处理 kv3 的多行 JSON），再和 `LaunchPlan.argv`
逐参数对比。

```bash
../.venv/bin/python tools/parity.py        # 31/31 cases match（launcher 与 run_llm.sh）
../.venv/bin/python tools/parity.py -v     # 逐条用例
```

实测（P2b）：`vllm → ninfer → vllm` 真实切换两次，脚本启动与 launcher 启动结果一致；
NInfer 冷启动 ≈10s，vLLM 冷启动 ≈90s。

覆盖：vllm × {stage1,mtp,dflash,kv2,kv3} × {128k,256k} × {fp8,auto}、ninfer ×
{stage1,mtp,dflash,dflash2}，以及 API_KEY / CORS / USE_PRIVILEGED / SPEC_TOKENS /
MAX_NUM_SEQS / GPU_MEM_UTIL / MAMBA_SSM_CACHE_DTYPE / VLLM_IMAGE / NINFER_* 等覆盖项；
双方都会拒绝的组合（vllm+dflash2、ninfer+kv2/kv3）也要求"同时拒绝"。

唯一有意的差异：launcher 给容器加了 `--label llm-monitor.managed=1`（对拍时单独列出），
这样 docker 面板能识别 monitor 创建的容器。

## 镜像与源（国内优先）

这台机器直连 Docker Hub / pypi / huggingface 都不稳定，统一走镜像：

* Docker：`dockerproxy.net/vllm/vllm-openai:...`（把它加进 `docker.image_repos` 就能在面板里操作）
* pip：`https://pypi.tuna.tsinghua.edu.cn/simple`
* HuggingFace：`HF_ENDPOINT=https://hf-mirror.com`
* npm：`https://registry.npmmirror.com`

`docker pull` 如果连不上会挂到 `docker.pull_timeout_s` 再被自动终止（也可以手动取消），
不会留下僵尸进程。

## 配置

见 `config.yaml`，常用项可用环境变量覆盖：

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `MONITOR_UI_PORT` | 9000 | 网页端口 |
| `MONITOR_GATEWAY_PORT` | 8000 | 网关端口 |
| `MONITOR_ENGINE_PORT` | 8001 | 引擎端口 |
| `MONITOR_AUTH_TOKEN` | 空 | 控制面共享密钥（header `x-monitor-token` 或 `?token=`） |
| `MONITOR_TEXT_KEEP` | 200 | 保留多少条请求的 prompt/输出原文 |
| `MONITOR_DB` | `./state/monitor.db` | SQLite 路径 |
| `MONITOR_CONFIG` | `config.yaml` | 配置文件 |

其它重要项：

```yaml
engine:
  bind_host: "127.0.0.1"   # 引擎绑定地址；0.0.0.0 会让客户端绕过网关（应急用）
  ninfer_jsonl: true       # 自动给 NInfer 打开 --request-log-jsonl（逐请求精确指标）
gateway:
  force_usage_chunk: false # 给流式请求注入 include_usage（拿精确 token；个别客户端会崩）
storage:
  text_keep: 200           # 只保留最近 200 条的原文，元数据保留 30 天
```

## Docker（仅 vLLM）

monitor 管 vLLM 的**容器和镜像**，不碰任何其它 docker 资源。安全模型是"读全局、写 scope"：

| 规则 | 说明 |
|---|---|
| scope | 默认 `image_repos: ["vllm/vllm-openai"]`、`containers: ["vllm-qwen38"]`，外加带 `llm-monitor.managed=1` label 的容器 |
| 命令白名单 | 只允许 `ps / inspect / image ls / image inspect / pull / rmi / rm / stop / start / logs / version / system df`；`system prune`、`image prune`、`volume rm`、`network rm` 在执行前就抛错 |
| 读全局 | 枚举**所有**容器与镜像，只为在删除前确认"没有别的项目在用" |
| 删镜像 | 有任何容器（含其他项目）在用 → 拒绝；只删 scope 内 tag；若镜像还挂着 scope 外 tag（如 `dockerproxy.net/...`），只删 scope 内 tag 并提示 |
| 悬空镜像 | 只有 monitor 自己 pull 产生的（记录在 `docker_images` ledger 里）才允许删 |
| 删容器 | 只允许 scope 内、且已停止的容器 |
| 拉取 | 流式日志 + 超时（`docker.pull_timeout_s`，默认 1h）+ 可取消，不留僵尸进程 |

能做的事：

* **镜像**：tag / 大小 / 创建时间 / 被谁使用 / 状态（当前 / 旧版本 / 悬空 / 锁定）/ 不可删除原因
* **拉取 / 检查更新**：流式日志；tag 变了会把旧 image 记为 superseded（随即变成可清理）
* **用新镜像重启**：重建容器并等 `/health`，冷启动几分钟
* **清理**：删除单个镜像、删除已停止容器、一键清理"可删除"（逐个重新校验，绝不 `prune`）

对应 MCP：`llm_docker_status`、`llm_docker_pull`、`llm_docker_job`、`llm_docker_remove_image`、
`llm_docker_remove_container`、`llm_docker_prune`、`llm_docker_restart_with`（写操作都要 `confirm=true`）。

> 如果 Docker Hub 不可达，`docker pull` 会一直挂到超时 —— 这台机器上的镜像来自
> `dockerproxy.net` 之类的镜像站，可以把它加进 `docker.image_repos` 再操作。

## 页面里有什么（4 个 tab）

页面按逻辑拆成四个 tab，状态记在 URL hash（`#overview` / `#requests` / `#engine` / `#docker`，
也支持 `?tab=requests`），刷新后停在上次的 tab。

**总览 overview** — 8 个 KPI（decode/prefill 速度、TTFT、KV、在飞、GPU、显存、功耗）、
四张 1 小时趋势图（吞吐 / 延迟 / 缓存与显存 / 功耗温度）、GPU 详情卡（利用率/功耗/温度/
风扇/时钟/PCIe/降频原因/每进程显存）。≥1900px 时一行 8 个 KPI、一行 4 张图。

**请求 requests** — 顶部汇总（总数 / 错误 / 近 1h 平均 decode、TTFT、prompt、输出）、
实时请求卡片（阶段、tokens、思考与输出尾巴、实时 tok/s）、请求历史表（筛选/分页，
点开抽屉看 thinking/content 全文与 NInfer JSONL 原始记录）。

**引擎 engine** — 引擎控制（vLLM/NInfer、profile、ctx、按引擎声明的参数如 KV 类型、
预览命令、启动/停止/重启/切换、契约探针、extended 开关）、启动阶段进度、引擎信息、
引擎日志。

**Docker docker** — 容器卡片、镜像表（大小/使用中/可删除原因）、拉取与清理、
镜像账本（monitor 拉取/取代/删除记录）。

Tab 上有状态提示：请求 tab 显示在飞数量徽标，引擎 tab 的小圆点跟随引擎状态
（绿=健康、琥珀=启动中、红=失败）。

## REST / WebSocket

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/state` | 全量快照（引擎/GPU/live/汇总） |
| GET | `/api/requests?limit&engine&status&search` | 历史列表 |
| GET | `/api/requests/{id}` | 单条详情（含原文） |
| GET | `/api/requests/live` | 在飞请求 |
| GET | `/api/series?window=3600` | 时间序列 |
| GET | `/api/logs?tail=200` | 引擎日志 |
| POST | `/api/engine/start\|stop\|restart\|switch` | 生命周期 |
| POST | `/api/engine/probe` | 契约探针 |
| GET | `/api/engines` | 每个引擎的 profiles + 可配置字段（KV 类型等） |
| POST | `/api/engine/plan` | 用任意 opts 预览 LaunchPlan |
| GET | `/api/engine/dry-run?engine=&profile=&ctx=` | 只出命令行（旧接口） |
| GET | `/metrics` | 统一命名空间的 Prometheus 导出（`llm_*`） |
| WS | `/ws` | 1 Hz 快照 + 事件（`request_done` / `engine_state`） |

`/metrics` 可以直接给 Grafana 刮；命名统一，不区分引擎。

## MCP

22 个工具，和网页按钮完全等价（同一个 Controller / 同一个 docker scope）：

`llm_status` `llm_metrics` `llm_gpu` `llm_live_requests` `llm_requests` `llm_request`
`llm_series` `llm_logs` `llm_probe` `llm_dry_run` `llm_start` `llm_stop` `llm_restart`
`llm_switch` `llm_bench`，以及 docker 的 `llm_docker_status` `llm_docker_pull` `llm_docker_job`
`llm_docker_remove_image` `llm_docker_remove_container` `llm_docker_prune` `llm_docker_restart_with`

stdio（pi / claude code / codex 直接挂）：

```json
{
  "mcpServers": {
    "llm-monitor": {
      "command": "/home/eedy/workspace/llm/.venv/bin/python",
      "args": ["-m", "monitor.mcp_server", "--url", "http://127.0.0.1:9000"],
      "cwd": "/home/eedy/workspace/llm/monitor"
    }
  }
}
```

HTTP：

```bash
../.venv/bin/python -m monitor.mcp_server --transport http --port 9001
# endpoint: http://127.0.0.1:9001/mcp
```

写操作默认只返回预览，必须 `confirm=true` 才执行。

## 逐请求数据从哪来

1. **网关旁路**（两个引擎通用）：请求体 → prompt 大小/消息数/媒体/采样参数；SSE 增量 →
   thinking 与 content 分开计数、TTFT、首字时间、实时 tok/s；末包 `usage` →
   精确 prompt/cached/completion/thinking token。
2. **NInfer JSONL**（更精确）：`x-request-id` 关联，回填 `computed_prefill_tokens`、
   `prefix_cache_hit_tokens`、`prefill/decode` 秒数、投机解码接受数、状态换出等。
3. **引擎 `/metrics`**：全局计数、KV 分页、并发、抢占、直方图分位。
4. **NVML**：功耗/温度/风扇/显存/时钟/PCIe/降频原因/每进程显存（`nvidia-smi` 兜底）。

没有 `stream_options.include_usage` 的流式请求，token 数用增量块计数（两个引擎都是
一 token 一块），prompt 用估算值（UI 里带 `~`）。想全部精确就打开
`gateway.force_usage_chunk: true`。

## 关于"专家激活"

`Qwen3.8-27B` 是 **dense** 模型（`config.json` 里没有 MoE，64 层 = 48 线性注意力 + 16 全注意力），
不存在专家路由。NInfer 只在 MoE artifact（Qwen3.5-35B-A3B / Qwen4Exp）下暴露 expert cache
占用，且没有 per-expert 激活直方图（需要给引擎打 patch）。按当前决定，**不做这个面板**；
换成 MoE 权重时再说。

## 运维

```ini
# /etc/systemd/system/llm-monitor.service
[Unit]
Description=LLM monitor (vLLM/NInfer control + observability)
After=docker.service network-online.target

[Service]
Type=simple
User=eedy
WorkingDirectory=/home/eedy/workspace/llm/monitor
ExecStart=/home/eedy/workspace/llm/.venv/bin/python -m monitor --config config.yaml
Restart=always
RestartSec=2
# docker 起停需要
SupplementaryGroups=docker

[Install]
WantedBy=multi-user.target
```

monitor 是数据面单点：它挂了，`:8000` 就没了（引擎仍在 `:8001`，可临时直连或改
`engine.bind_host`）。用 `Restart=always` 兜底。

## 开发

```bash
../.venv/bin/python -m pytest -q     # 50 个单测：prom/SSE/compat/docker scope/launcher/startup/schema
../.venv/bin/python tools/parity.py  # 31 个 launcher ⇄ run_llm.sh 对拍用例
../.venv/bin/python tools/shot.py http://127.0.0.1:9000/ /tmp/ui.png --height 3600   # 截图
node --check monitor/web/app.js      # 前端语法（无构建步骤）
```

模块：

```
monitor/
├── config.py      YAML + env 覆盖，路径解析
├── prom.py        Prometheus 文本解析 + 直方图分位
├── nvml.py        NVML(ctypes) + nvidia-smi 兜底
├── engine.py      vLLM / NInfer → 统一指标字典
├── tap.py         逐请求记录 + 三种协议的 SSE 解析
├── gateway.py     零缓冲透传代理（含 /v1/completions 兼容）
├── compat.py      legacy completions ⇄ chat 翻译
├── store.py       ring buffer + SQLite（保留最近 200 条原文）
├── controller.py  run_llm.sh 编排 + 状态机 + 契约探针
├── api.py         REST / WS / 静态页 / Prometheus 导出
├── app.py         两个 uvicorn + 采集循环
├── docker_scope.py  docker 作用域（白名单 verb + 逐对象校验）
├── launcher.py    engines/*.yaml -> LaunchPlan（run_llm.sh 的替代品）
├── engines/       vllm.yaml · ninfer.yaml（profiles/mounts/env/preflight）
├── mcp_server.py  MCP（stdio / http）
└── tools/shot.py  本地截图（CDP，无 npm 依赖）
└── web/           index.html · style.css · app.js · vendor/uPlot
```

## FAQ

**启动报 "gateway port 8000 is already in use"？**
引擎还在 8000。`../run_llm.sh --stop` 后由 monitor 接管，或临时 `MONITOR_GATEWAY_PORT=8010`。

**切换引擎要多久？**
vLLM 冷启动要重新加载权重/编译 kernel，几分钟；NInfer 大约 10 秒。UI 顶部会显示状态与进度，
起停过程会中断在飞请求。

**为什么 `/v1/completions` 在 NInfer 下也能用？**
网关内联了原来 `compat_openai.py` 的翻译逻辑（legacy completions ⇄ chat），
并且是在 tap 之后翻译，所以指标照样准确。

**能不能只监控不起停？**
可以：不起停就用 `/api/state`、图表、历史；控制按钮会提示先停掉旧引擎。
（"只监控、不占端口"的旁路模式没有实现——那样 vLLM 就没有逐请求数据。）

**能删别的项目的镜像吗？**
不能。写操作只认 scope 内的 repo tag / 容器名，而且删除前会确认没有任何容器（包括
其他项目的）在用。`docker image prune` / `system prune` 这类全局命令在代码层就被拒绝。

**数据存在哪？**
`state/monitor.db`（SQLite，WAL）。原文只留最近 200 条，30 天后自动清元数据；
`DELETE /api/requests?keep=0` 可清空。
