# MCP 服务（llm-monitor）

把 llm-monitor 的控制面打包成 **MCP（Model Context Protocol）** 服务，让 Claude Desktop /
Cursor / Claude Code / 自研 agent 直接"用嘴"起停引擎、切 profile、看吞吐、查每个请求、拉镜像。

一句话：**25 个工具，全部是对 `:9000` HTTP 控制面的薄封装**——MCP 进程自己不碰 docker、
不碰引擎、不读数据库，只发 REST 请求。所以网页上能做的，MCP 都能做；网页上没有的，MCP 也没有。

```
┌────────────┐  MCP (http /mcp · stdio) ┌───────────────────────────┐  HTTP  ┌────────────────────┐
│ Claude /   │ ────────────────────────▶ │ monitor 主进程 :9000       │ ─────▶ │ Controller / Store │
│ Cursor /   │ ◀──────────────────────── │  UI · REST · WS · MCP     │ ◀───── │                    │
│ 自研 agent │        JSON 结果           └───────────────────────────┘        └─────────┬──────────┘
└────────────┘                                                                      │
                                                                    docker / pid / :8001 engine
```

---

## 1. 启动

MCP 端点由 monitor 主进程直接提供，**和 UI 同一个端口**，不用另起服务、不用另占端口：

```bash
cd /home/eedy/workspace/llm/monitor
./run.sh
#   endpoint: http://127.0.0.1:9000/mcp    （streamable HTTP，stateless）
```

客户端注册（Claude Desktop / Cursor / pi …）：

```json
{"mcpServers": {"llm-monitor": {"url": "http://127.0.0.1:9000/mcp"}}}
```

配了 `server.auth_token` 时带同一个 token（UI 的中间件同样管 `/mcp`）：

```json
{
  "mcpServers": {
    "llm-monitor": {
      "url": "http://127.0.0.1:9000/mcp",
      "headers": {"x-monitor-token": "<token>"}
    }
  }
}
```

两种可选的独立模式（客户端只会拉子进程，或 MCP 必须单独跑在别的机器/端口时）：

```bash
# stdio（也可以 ./run.sh mcp）
../.venv/bin/python -m monitor.mcp_server

# 常驻 HTTP（注意：本机 9001 已被 hindsight 容器占用，换个空闲端口）
../.venv/bin/python -m monitor.mcp_server --transport http --host 127.0.0.1 --port 9010
#   endpoint: http://127.0.0.1:9010/mcp

# 旧版 SSE 传输
../.venv/bin/python -m monitor.mcp_server --transport sse --port 9010
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--url` | `http://127.0.0.1:9000` | 控制面地址（env `MONITOR_URL`） |
| `--token` | `""` | 控制面 `server.auth_token`（env `MONITOR_TOKEN`，作为 `x-monitor-token` 头发送） |
| `--transport` | `stdio` | `stdio` / `http`（streamable-http，路径 `/mcp`）/ `sse` |
| `--host` / `--port` | `127.0.0.1` / `9001` | 仅独立 http/sse 用（本机 9001 被 hindsight 占用，建议 `9010`） |

前提：**monitor 主进程在跑**（`./run.sh`）。MCP 只是客户端，API 不在就报连接错误。
挂载在 UI 上的 `/mcp` 是 **stateless** 的：工具都是纯 REST 调用，不存会话，主进程重启后
客户端直接重连即可（不需要重新握手拿 session id）。

### 客户端注册

Claude Desktop / Cursor 的 `mcpServers` 配置：

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

Claude Code：

```bash
claude mcp add llm-monitor -- \
  /home/eedy/workspace/llm/.venv/bin/python -m monitor.mcp_server \
  --url http://127.0.0.1:9000
```

> 改了工具列表（比如新增工具）要**重连/重启 MCP server**，客户端会缓存 `tools/list`。

---

## 2. 通用约定

* **返回**：JSON。HTTP 非 2xx 会变成 MCP 错误（`raise_for_status`），错误信息里带 API 的
  `detail`（例如 `engine.start: another lifecycle action is in progress`）。
* **写操作两段式**：所有 `start/stop/restart/switch/docker pull/remove/prune` 都要
  `confirm=true`；不带 confirm 只返回 **预览或提示**，不改状态。这是防手滑的硬约定。
* **时间**：快照里的时间戳是 epoch 秒（`first_at` / `last_at` / `started_at`）；工具参数里的
  日期是 `YYYY-MM-DD`，**本地时区整天、含首尾两天**。
* **单位**：tok/s、ms、MiB、W、°C、元。token 数量是整数。
* **两套 token 口径**（容易混）：
  * `engine.tokens`（`llm_metrics("tokens")`）= 引擎 `/metrics` 的**进程内计数器**，引擎重启归零；
  * `ledger`（`llm_status.ledger` / `llm_ledger`）= **请求表**聚合，跨重启、按 30 天保留，
    带金额。
* **超时**：MCP → API 的 httpx 超时 120 s。真正慢的是启动（vLLM 冷启动可几分钟），
  这类接口返回的是"已受理"，进度靠 `llm_status` / `llm_logs` 轮询。
* **docker 只在 vLLM scope 内**：读枚举全局（为了判断"有没有别的容器在用"），
  写只能碰 `docker.image_repos` + `docker.containers` + `llm-monitor.managed=1` 的资源；
  全局 `prune` / volume / network 在 API 层就被拒绝。

---

## 3. 工具清单

24 个，分五组。`R`=只读，`W`=写（需 `confirm=true`）。

| 工具 | | 一句话 |
|---|---|---|
| `llm_status` | R | 精简快照：状态/模型/请求/缓存/速度/投机 + GPU + **ledger/金额** |
| `llm_metrics` | R | 完整归一化指标，按 section 取 |
| `llm_ledger` | R | Token 累计账本（本次运行 / 库内 / 日期区间按天） |
| `llm_cost` | R | DeepSeek-flash 等价费用 + GPU 电费 |
| `llm_gpu` | R | GPU 遥测 + 每进程显存 |
| `llm_live_requests` | R | 在飞请求（含 `cancellable`：能否中止） |
| `llm_requests` | R | 请求历史（元数据） |
| `llm_request` | R | 单条请求全文 |
| `llm_series` | R | 1 Hz 时间序列 |
| `llm_logs` | R | 引擎日志尾部 |
| `llm_probe` | R | 契约探针（3 项，可选 5 项） |
| `llm_dry_run` | R | 只算不改：启动计划/argv |
| `llm_start` | W | 起引擎（vllm/ninfer + profile） |
| `llm_stop` | W | 停引擎 |
| `llm_cancel_request` | W | 中止**一个**在飞请求（关闭上游流，引擎 abort） |
| `llm_restart` | W | 重建（换 profile/ctx） |
| `llm_switch` | W | 原子切换引擎（停→等显存→起→健康检查） |
| `llm_bench` | R\* | 真打一条流式请求，量 TTFT/decode（\*会产生真实推理开销） |
| `llm_docker_status` | R | scope 内容器/镜像/磁盘 |
| `llm_docker_pull` | W | 拉/更新镜像（异步任务） |
| `llm_docker_job` | R | 拉取任务进度 |
| `llm_docker_remove_image` | W | 删镜像（多重校验） |
| `llm_docker_remove_container` | W | 删已停止的容器 |
| `llm_docker_prune` | W | 只删账本判定可删的镜像 |
| `llm_docker_restart_with` | W | 用指定本地镜像重建 vLLM |

---

### 3.1 观测

#### `llm_status() -> dict`

看一眼就够用。返回 `state / engine / detail / model / requests / tokens / cache / speed /
spec / gpu / live`，外加：

```jsonc
"ledger": {
  "requests": 730, "total_tokens": 30837344,
  "prompt": 30302742, "prompt_cached": 26702409, "prompt_miss": 3600333,
  "output": 534602, "thinking": 321033, "content": 213569,
  "cache_hit_pct": 88.1, "cache_estimated": true,   // vLLM 不报逐请求缓存 → 引擎比例估算
  "first_at": 1792249876.2, "last_at": 1792341047.1,
  "session_total_tokens": 0,                        // 本进程，重启归零
  "deepseek_cny": 6.2728, "energy_kwh": 0.0217, "electricity_cny": 0.013
}
```

`speed` 里同时有瞬时值和 1 分钟均值：`decode_tps` / `decode_tps_60s`、`prefill_tps` /
`prefill_tps_60s`。

#### `llm_metrics(section="all") -> dict`

`section` 取值：

| section | 内容 |
|---|---|
| `all` | `engine` + `gpu` + `gateway` + `summary` + `ledger` + `billing` |
| `requests` `cache` `speed` `spec` `config` | 引擎侧对应块（进程内计数器） |
| `tokens` | 引擎侧 token 计数器（≠ 账本） |
| `ledger` | 请求表账本 `{session, history}` |
| `billing` | 金额 `{price, session, history}` |

未知 section 会返回 `{"error": …, "available": [...]}`，不会抛异常——方便 agent 试探。

#### `llm_ledger(start=None, end=None) -> dict`

* 无参：`{session, history, billing, billing_history}`，其中每个 scope 都是
  `requests/total/prompt/prompt_cached/prompt_miss/output/thinking/content/cache_hit_pct/
  cache_estimated/first_at/last_at`，并带 `peak`（高峰时段子集，用于峰谷计价）。
* 带 `start`/`end`（`YYYY-MM-DD`）：`{start, end, days, scope, billing, daily[]}`，
  `daily` 一天一行、没流量的日子补 0，行内附 `deepseek_cny` / `energy_kwh` / `electricity_cny`。

```python
llm_ledger()                                    # 本次运行 vs 库内累计
llm_ledger(start="2026-10-01", end="2026-10-10")  # 双十区间，按天
```

#### `llm_cost(start=None, end=None) -> dict`

* 无参：`{price, session, history}`；`price` 是价目表（`deepseek-flash`、命中/未命中/输出
  单价、`offpeak_factor`、`peak_hours`、`cny_per_kwh`）。
* 每个 scope：`{deepseek: {cost, peak_price, offpeak_price, peak_surcharge, peak_token_pct},
  energy: {kwh, cost, seconds, since}, total}`。
* 带日期：`{price, cost, per_day[]}`。

口径：价目是**高峰价**（元/百万 token），北京时间周一至周五 9–12、14–18 为高峰、其余半价；
电价默认 0.60 元/kWh（成都居民一档 0.5224 / 工商业平段 ~0.6125 之间），可在 `config.yaml`
的 `billing.*` 改。**电费只算显卡**（NVML 功耗积分），不含 CPU/电源损耗。

#### `llm_gpu() -> dict`

`{mode, gpus: [{index, name, util_pct, mem_used_mib, mem_total_mib, mem_used_pct, power_w,
power_limit_w, temp_c, temp_limit_c, fan_pct, sm_clock_mhz, mem_clock_mhz, pcie_gen,
pcie_width, throttle[], procs[]}]}`。`throttle` 是降频原因列表（空 = 无降频）。

#### `llm_live_requests() -> dict`

`{items: [{id, phase, protocol, stream, prompt_tokens, thinking_tokens, content_tokens,
content_tail, thinking_tail, elapsed_ms, live_decode_tps, …}]}`。`phase` ∈
`queued|prefill|decode|done|error`。

#### `llm_requests(limit=20, engine=None, status=None, search=None) -> dict`

`{total, items[]}`。`status` = `ok|error`；`search` 模糊匹配 id/model/endpoint；
`engine` = `vllm|ninfer`。只有元数据，正文用 `llm_request`。

#### `llm_request(request_id) -> dict`

单条全量：`timing`（ttft/ttfc/decode/total）、token 拆分（prompt / cached / thinking /
content）、`thinking_text` / `content_text`（只对最近 200 条保留）、`sampling`、`usage`、
`engine_record`（NInfer JSONL 原始 `request_done`）。

#### `llm_series(window_s=3600) -> dict`

`{ts[], keys[], decode_tps[], prefill_tps[], ttft_p50_ms[], itl_ms[], kv_pct[], gpu_util[],
vram_pct[], power_w[], temp_c[], running[], waiting[]}`，1 Hz，内存环形缓冲保留 2 h
（超出窗口会返回较短数组）。

#### `llm_logs(tail=200) -> dict`

`{engine, source, lines[]}`。vLLM 走 `docker logs --tail`，NInfer 读 `run/<name>.log`
（只读尾部，文件会被 maintenance 截断，不会一次性读入内存）。

---

### 3.2 验证与计划

#### `llm_probe(extended=False) -> dict`

真发请求验证"契约"：`/health`、`/v1/models`、`/v1/chat/completions`，各返回状态码/耗时。
`extended=True` 追加 `/v1/responses`、`/v1/messages`（Anthropic 形状）——**会真的打引擎**，
在 NInfer 上曾触发引擎侧 bug，所以默认关闭。

#### `llm_dry_run(engine="vllm", profile="stage1", ctx=None, kv_capacity=None, spec_tokens=None) -> dict`

返回**扁平的**计划对象：`{engine, kind, profile, ok, command[], argv[], engine_argv[],
command_text, host, port, ctx, served_model_name, container_name, image, models_dir, mounts[],
env, labels, pidfile, logfile, warnings[], errors[]}`。start 前先看这个最稳：预检不过会直接给
`errors`，`ok=false`。

---

### 3.3 生命周期（写，`confirm=true`）

```python
llm_start(engine="vllm", profile="mtp", ctx="128k", confirm=True)
llm_stop(confirm=True)
llm_restart(engine="vllm", profile="dflash", ctx="128k", confirm=True)
llm_switch(engine="ninfer", profile="stage1", ctx="256k", confirm=True)
```

* `engine`: `vllm|ninfer`；`profile`: `stage1|mtp|dflash|dflash2|kv2|kv3`
  （最终由 `engines/*.yaml` 决定，`llm_metrics("config")` / `llm_status` 能看到当前值）。
* 不带 `confirm`：`llm_start`/`llm_switch` 返回 `preview`（dry-run 结果）+ `hint`；
  `llm_stop`/`llm_restart` 只返回 `hint`。
* 并发保护：同时只允许一个生命周期动作，第二个会返回
  `another lifecycle action is in progress`。
* 起停会**中断在飞请求**；vLLM 冷启动可能几分钟，`llm_status.state` 依次是
  `starting → healthy`。启动阶段的进度/日志尾巴在控制台快照的 `controller.startup`
  （`/api/state`，也可从网页看到），MCP 侧用 `llm_logs` 跟 `llm_status.state` 轮询即可。

---

### 3.4 压测

#### `llm_bench(prompt=…, max_tokens=256, model=None) -> dict`

经**网关**（:8000，走 tap）发一条真流式请求，返回
`{ok, ttft_ms, total_s, completion_tokens, decode_tps, usage}`。它会走完整的旁路记账，
所以跑完 `llm_ledger` 的数字会涨——这是特性，用来验证链路。

---

### 3.5 Docker（仅 vLLM scope）

```python
llm_docker_status()
llm_docker_pull(ref="vllm/vllm-openai:cu130-sm120", confirm=True)  # → {job id}
llm_docker_job(job_id, lines=80)                                   # 轮询进度/日志
llm_docker_remove_image(image_id, confirm=True)
llm_docker_remove_container(name="vllm-qwen38", confirm=True)
llm_docker_prune(confirm=True)
llm_docker_restart_with(ref, profile="stage1", ctx="128k", confirm=True)
```

行为要点：

* `pull` 返回 `{ok, job{id, kind, ref, state, progress, old_id, new_id, changed, …}}`；
  轮询时把 `job.id` 传给 `llm_docker_job`。它兼做**更新检查**：tag 指向的 image id
  变了就 `changed=true`，旧 id 记为 superseded（之后才允许清理）。
* 删除镜像前会检查：有容器在用 → 拒绝；只有 scope 外 tag → 拒绝；不在账本里 → 拒绝。
* `prune` 只删 `llm_docker_status` 里标为 removable 的那些，逐个再校验；**从不**调
  `docker image prune` / `system prune` / volume / network。

---

## 4. 典型工作流

**"现在这套跑下来，如果买 DeepSeek 要多少钱？"**

```
llm_cost()                                        # 全量：6.27 元 API + 0.01 元电费
llm_cost(start="2026-10-01", end="2026-10-10")    # 某个区间，按天
llm_ledger(start="2026-10-18", end="2026-10-18")  # 那天的 token 拆分
```

**"把 vLLM 换成 mtp profile，先别动手"**

```
llm_status()                        # 确认现在是什么
llm_dry_run(engine="vllm", profile="mtp", ctx="128k")   # 看 argv / mounts / 预检
llm_restart(engine="vllm", profile="mtp", ctx="128k", confirm=True)
llm_status()                        # starting → healthy，看 startup 阶段
```

**"引擎起不来了，怎么回事？"**

```
llm_status()         # state / detail / startup.tail
llm_logs(tail=200)   # 引擎日志尾部
llm_dry_run(...)     # 预检错误（镜像不在、模型目录缺失、显存参数冲突…）
llm_gpu()            # 显存是否被僵尸容器占着
```

**"镜像更新了，切过去"**

```
llm_docker_status()                                   # 当前 image id / 账本
llm_docker_pull(ref="vllm/vllm-openai:cu130-sm120", confirm=True)  # → job_id
llm_docker_job(job_id)                                # 轮询到 finished，看 changed
llm_docker_restart_with(ref="...", profile="mtp", confirm=True)
```

**"这条请求为什么慢？"**

```
llm_requests(limit=20, status="error")   # 先找可疑的
llm_request("<id>")                     # ttft / prefill / decode / thinking 拆分 + 全文
llm_series(window_s=600)                # 同期的吞吐/延迟曲线
```

---

## 5. 已知边界

* **单机、单实例**：一个 MCP server 对一个 monitor（`--url`）。
* **无鉴权体系**：默认只监听本机；`server.auth_token` 非空时 `--token` 必须对上。
* **写操作不能撤销**：`confirm=true` 的请求会真的停掉在飞请求。
* **`llm_bench` / `llm_probe(extended)` 会真的推理**：占显存、算进 token 账本与金额。
* **金额是估算**：DeepSeek 价目按高峰/空闲拆分（法定节假日未建模），缓存命中在 vLLM 上
  是引擎比例估算（`cache_estimated=true`），电费只算显卡。
* **历史正文只留最近 200 条**，30 天后连元数据一起清（`storage.*`）；`llm_request` 对老记录
  的 `thinking_text` / `content_text` 会是 `null`。
* **中止的边界**：`llm_cancel_request` 靠关掉上游 SSE 让引擎 abort。网关默认把客户端的
  `stream:false` 改写成上游流式（`gateway.upstream_stream`），所以普通 chat 请求都能中止；
  但 `n>1` / `logprobs` 的请求、以及旧式 `/v1/completions` 非流式请求走缓冲路径，
  **引擎已自己吞完整条生成，任何手段都停不下来**——API 会返回 409 并写明原因。
* 工具列表变了需要**重启 MCP server**（客户端缓存 `tools/list`）。

---

## 6. 自检

```bash
# 工具数量 / 名字
../.venv/bin/python - <<'PY'
import asyncio, json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    p = StdioServerParameters(command="../.venv/bin/python",
                              args=["-m", "monitor.mcp_server"],
                              cwd=".")
    async with stdio_client(p) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            print([t.name for t in (await s.list_tools()).tools])
asyncio.run(main())
PY
```

单测：`tests/test_mcp.py`（形状/区间/section 回落，不依赖运行中的 monitor）。
