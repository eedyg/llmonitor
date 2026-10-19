# 设计：统一的 vLLM / NInfer 监控与控制面

状态：**已确认并实现（P0 + P1）**
最后更新：2026-10-17

## 1. 结论先行

* 两个引擎**都没有** Web 监控页（vLLM 0.31 只有 `/metrics`+`/health`；NInfer 只有
  `/metrics` + 逐请求 JSONL + stderr 吞吐）。
* 因此统一监控**不需要改引擎**，只需要一个旁路数据面 + 一套归一化字典。
* vLLM **没有逐请求数据**，所以 monitor 必须站在数据路径上：网关独占公网端口，
  引擎退到 loopback。

## 2. 已拍板的决策

| # | 决策 | 结果 |
|---|---|---|
| 1 | 网关前置（monitor 占公网端口，引擎退到内部端口） | ✅ 接受 |
| 2 | MoE 专家激活面板 | ❌ 暂时不做（当前模型是 dense，没有专家） |
| 3 | prompt / 输出原文保留 | 最近 **200 条**，元数据 30 天 |
| 4 | 前端 | 原生 JS + vendored uPlot（无构建）+ **neumorphism** |
| 5 | 部署 | 单进程（两个 uvicorn server 共用事件循环） |

## 3. 架构

三层，单进程：

1. **Gateway Tap** — `:8000`，所有客户端入口。请求/响应原样透传（零缓冲、零改写），
   同时把一份字节喂给 `tap.py`。支持三种协议：OpenAI chat / legacy completions /
   Responses / Anthropic messages。NInfer 缺 `/v1/completions`，在 tap 之后翻译。
2. **Collectors** — 引擎 `/metrics`（2s，直方图→分位）、NVML（1s）、
   NInfer JSONL tail（1s，按 `x-request-id` 回填）、docker/pid 探测（2s）、日志 tail。
3. **Controller** — 唯一写状态的地方。状态机
   `stopped → starting → healthy → stopping → failed`，全局锁，起停都通过
   `engines/*.yaml`（run_llm.sh 已淘汰，此处为历史记录）：

   ```bash
   HOST_PORT=8001 COMPAT=0 RUN_DIR=... ../run_llm.sh \
       --engine {vllm|ninfer} --profile ... --host 127.0.0.1 --ctx ... \
       [--kv-capacity ...] [--spec-tokens ...] \
       [-- --request-log-jsonl ... --log-stats-interval-ms 1000]
   ```

   切换 = `stop` → 等 NVML 显存释放（<2 GiB）→ `start` → 轮询 `/health` →
   （手动）契约探针。

## 3.1 Launcher（P2b，已接管启动）

`run_llm.sh` 的**参数构建**已经声明式化：`engines/{vllm,ninfer}.yaml` + `launcher.py`
产出 `LaunchPlan{argv, engine_argv, mounts, env, labels, warnings, errors}`。

* 对拍方法：假 `docker` / 假 `ninfer-serve` 放进 PATH，让 run_llm.sh 走**真实启动路径**
  （不是解析 `--dry-run` 文本——kv3 的多行 JSON 会让文本解析产生歧义），抓 NUL 分隔的 argv，
  与 plan 逐参数比较。
* 结果：**31/31**（含"双方都必须拒绝"的组合）。
* 唯一有意差异：容器加 `--label llm-monitor.managed=1`（docker scope 用）。
* P2b 已完成：controller 默认执行 `LaunchPlan`；`MONITOR_ENGINE_LAUNCHER=0` 可回退到
  `run_llm.sh`。启动阶段由 spec 里的 `startup:` 正则驱动（UI 显示真实进度），
  停止走 `docker stop -t 25` + `rm` / SIGTERM→SIGKILL，并等待端口释放。
* 实测：`vllm → ninfer → vllm` 两次真实切换，NInfer ≈10s、vLLM ≈90s。
* 已完成：观察期结束后直接淘汰 `run_llm.sh`（2026-10-19）。argv 契约改由
  `tests/golden/launch_argv.json` + `tests/test_launch_golden.py` 冻结，
  对拍工具 `tools/parity.py` 随之删除。

## 4. 统一指标字典

```jsonc
{
  "engine":   { "kind", "model", "served", "ok", "scraped_at" },
  "requests": { "running", "waiting", "paused", "prefilling", "decode_ready",
                "replaying", "preemptions" },
  "tokens":   { "prompt", "prompt_cached", "prefill", "generation", "decode", "replayed" },
  "cache":    { "kv_pct", "kv_used_pages", "kv_capacity_pages", "prefix_queries",
                "prefix_hits", "hit_rate", "host_used_bytes", "pools" },
  "speed":    { "decode_tps", "decode_tps_60s", "prefill_tps", "ttft_p50/p90/p99_ms", "itl_p50_ms",
                "queue_p50_ms", "e2e_p50/p90_ms" },
  "spec":     { "backend", "draft_window", "rounds", "drafted", "accepted",
                "accept_rate", "fallback" },
  "config":   { /* 来自 cache_config_info / ninfer gauges */ }
}
```

映射（节选）：

| 统一字段 | vLLM | NInfer |
|---|---|---|
| `requests.running` | `vllm:num_requests_running` | `ninfer_requests_running` |
| `requests.prefilling` | —（tap 推断） | `ninfer_requests_prefilling` |
| `cache.kv_pct` | `vllm:kv_cache_usage_perc`×100 | `used/capacity` page groups |
| `cache.hit_rate` | `prefix_cache_hits/queries` | `prompt_cached/prompt` |
| `speed.ttft_p50_ms` | `vllm:time_to_first_token_seconds` | `ninfer_time_to_first_token_seconds` |
| `speed.decode_tps_60s` / `prefill_tps_60s` | 对应计数器 60 s 窗口 | 对应计数器 60 s 窗口 |
| `spec.accept_rate` | `vllm:spec_decode_num_accepted/draft_tokens` | `ninfer_spec_decode_accepted/draft_tokens` |

`/metrics` 再导出一遍为 `llm_*`（`llm_engine_up`、`llm_requests_running`、
`llm_decode_tokens_per_second`、`llm_gpu_power_watts` …），方便以后接 Grafana。

## 5. 逐请求记录

```jsonc
{ "id":"mreq_…", "engine":"ninfer", "endpoint":"/v1/chat/completions", "stream":true,
  "prompt":{"tokens":18422,"cached":16384,"bytes":74213,"messages":12,"has_media":false},
  "output":{"thinking_tokens":210,"content_tokens":354,"finish":"stop"},
  "timing":{"ttft_ms":610,"decode_ms":8203,"total_ms":8840},
  "speed":{"decode_tps":43.1}, "spec":{...}, "sampling":{...},
  "text":{"reasoning":"…","content":"…"},   // 只留最近 200 条
  "engine_record":{ /* NInfer JSONL request_done */ } }
```

精度策略：
* 非流式、或流式带 `include_usage` → 精确 usage。
* 流式無 usage → 增量块计数（两引擎均 1 token/块），prompt 用字符估算（UI 标 `~`）。
* NInfer → JSONL 回填，覆盖为精确值。

## 5.1 引擎参数表单（按引擎声明）

`engines/*.yaml` 新增 `options:`，每个字段就是 launcher 变量名：

```yaml
options:
  - key: kv_cache_dtype
    label: "KV 类型"
    type: enum
    choices: [fp8, auto, fp8_e4m3, fp8_e5m2]
    default: fp8
    hint: "fp8 让 256k 放得下；auto(bf16) 在 256k 会 OOM"
```

* `GET /api/engines` 输出每个引擎的 profiles + options；UI 按引擎重建表单
  （vLLM 是 `kv_cache_dtype`，NInfer 是 `kv_dtype`，默认都是 fp8）。
* `POST /api/engine/plan` 接收任意 opts；`spec_tokens` 用 `show_if.profiles` 控制显隐。
* 值保存在 localStorage（按引擎分开），启动/预览/计划三条路径共用同一份 opts。
* 新增字段零前端改动；`tests/test_launcher.py::test_schema_keys_are_known_variables`
  防止 spec 里写错 key。

## 5.2 Token 累计（只读账本）

总览的 Token 卡是**纯 SQL 聚合**，不写库、不维护累加器，所以永远不会和请求表对不上：

```
total = prompt + output
prompt = COALESCE(NULLIF(prompt_tokens,0), prompt_tokens_est, 0)
output = thinking_tokens + content_tokens
prompt_cached / prompt_miss = 命中 / 没命中（miss = prompt - cached）
```

三种口径共用 `Store._token_scope()` + `Store._finish_scope()`：

| 口径 | 范围 | 用途 |
|---|---|---|
| `session` | `started_at >= 进程启动` | 重启归零；放副标题 |
| `history` | 库内保留期全部行（默认显示） | 重启不丢 |
| `range` | `[本地 00:00, 次日 00:00)`，按天 GROUP BY | `/api/tokens?start&end` + 明细表 |

`strftime('%Y-%m-%d', started_at, 'unixepoch', 'localtime')` 做分日，边界用 `time.mktime`
推出本地零点（不用 `+86400` 硬加，DST 会错一天）。静默的日子补 0 行。

**缓存命中是估算的**：vLLM 的 response usage 没有 `prompt_tokens_details.cached_tokens`，
没有任何一行带 `prompt_cached` 时，用引擎侧 `prompt_cached/prompt` 比例乘请求侧 prompt，
并返回 `cache_estimated=true`（UI 前缀 `≈` + tooltip）。NInfer 的
`prefix_cache_hit_tokens` 是精确值。

## 5.3 磁盘边界

会自己长大的东西都有上限，`maintenance.py` 每 `logs.check_interval_s`（30 s）检查：

| 文件 | 上限 | 做法 |
|---|---|---|
| `run/<engine>.log` | `logs.max_log_mb` 64 | 留尾部 `keep_tail_mb` 8，**原地** truncate |
| `run/ninfer-requests.jsonl` | `logs.jsonl_max_mb` 64 | 同上（NInfer `ios::out\|app`） |
| `state/lifecycle.log` | `logs.lifecycle_max_mb` 8 | 同上 |
| `state/monitor.db` | `retention_days` 30 + `max_db_mb` 512 | 先按时间删 → 超硬上限删最老 10%/轮 → `incremental_vacuum` |

**为什么 truncate 而不是 rename 轮转**：引擎进程继承的是 O_APPEND 的 fd，rename 之后它会
继续写一个已 unlink 的 inode —— 磁盘不释放、日志也读不到。原地截断后 O_APPEND 会重新定位到
新 EOF，进程无感。

DB 启动时把 `auto_vacuum` 切到 `INCREMENTAL`（设了就只在下次 VACUUM 生效，所以小库立即
VACUUM 一次），这样以后删行能真还页，而不需要 VACUUM 的 2 倍磁盘。

Docker 容器日志不在我们控制面内（`/var/lib/docker`），文档里给 `--log-opt max-size/max-file`。

## 5.4 金额（等价 API 费 + 电费）

* **口径分离**：token 账本只回答“多少 token”，钱在两处算：`billing.token_cost()` 用
  `billing.deepseek.*` 价目表；`store.record_energy()` 负责电量积分。UI 的三格分别是
  DeepSeek 等价 / GPU 电费 / 合计。
* **峰谷**：价目表是“高峰价 + 空闲半价”，所以 SQL 只多算一个 **peak 子集**
  （`PEAK_SQL`：北京时间 `%w` 1–5 且 `%H IN (9,10,11,14,15,16,17)`），
  混算价 = `k·P·total + (1-k)·P·peak`。法定节假日不建模（误差 ≤ 几个百分点）。
* **电量**：`series_loop` 每秒把 `point.power_w` 交给 `record_energy()`，梯形积分成 Wh，
  按本地日 UPSERT 进 `gpu_energy`；内存里另外存一份 session 计数。写库按 5 Wh / 5 s 批量，
  避免 1 Hz 写盘；`dt` 上限 10 s，防止休眠/重启后凭空造电量。
* **只算显卡**：NVML 读的是板卡功耗，不含 CPU / PSU 损耗 / 风扇 —— 文档里写明，不然会和电表对不上。

## 6. Docker 管理（仅 vLLM）

* **scope**：`docker.image_repos` + `docker.containers`（默认取 `engine.container_names.vllm`）
  + `llm-monitor.managed=1` label。
* **读全局、写 scope**：删除镜像前必须知道"是否有任何容器在用"，所以枚举是全局只读的；
  写操作逐个对象校验。
* **verb 白名单**：`ps/inspect/image ls/image inspect/pull/rmi/rm/stop/start/logs/version/system df`；
  全局 prune / volume / network 操作在 `_assert_command` 里直接抛 `ScopeViolation`。
* **tag 粒度**：删镜像只删 scope 内 tag；scope 外 tag（镜像站别名）保留，并明确提示。
* **ledger**：`docker_images` 表记录 monitor 拉取过的 image id 与 supersede 关系，只有
  ledger 里的悬空镜像才允许删除。
* **pull 任务**：异步 + 流式日志 + 超时 + 取消；post-pull 若 image id 变化，旧 id 标记
  superseded（立刻变成可清理）。
* **不动的东西**：其它容器、其它镜像、volume、network、builder cache。

## 7. UI

单页、无构建、neumorphism（同一底色 + 双向投影；inset 用于凹槽/输入框/图表容器）。
按逻辑分 4 个 tab（总览 / 请求 / 引擎 / Docker），tab 状态进 URL hash + localStorage；
uPlot 在面板隐藏时宽度为 0，所以**每次切到总览都会重建图表**，其余 tab 的数据加载
（历史 / 日志 / docker）也改成按需触发。
宽度自适应：≤1900px 时 KPI 4/行、图表 2/行；≥1900px 时 KPI 8/行、图表 4/行、
控制表单一行放下（minmax 168px 自动列）。
浅色/深色两套 CSS 变量，跟随 `localStorage`，支持 `?theme=dark`。
图表 = uPlot（vendored，离线可用），1 Hz 追加，保留 2h。
`prefers-reduced-motion` 下关闭动画。

## 8. 明确不做（本期）

* MoE per-expert 激活热力图（需要改 NInfer router；当前模型 dense）。
* 多机 / 多实例聚合。
* 旁路模式（只监控不占端口）—— 会失去 vLLM 的逐请求数据。
* 鉴权体系（只有一个可选共享 token；默认只应在可信 LAN）。

## 9. 验收

* `pytest` 62 项通过（Prometheus 解析、SSE 三协议、completions 翻译、docker scope 守卫、
  launcher、token 账本（区间/分日/估算）、日志截断与 DB 硬上限）。
* `tests/test_launch_golden.py` 32/32 冻结 argv（等价于原 parity 的契约）。
* 起停命令经 `--dry-run` 校验；契约探针 5 项 200。
* 现有 vLLM 在 `:8000` 上被 adopt 后，`/api/state`、图表、历史、GPU 均正常。
* 真实切换（vLLM ⇄ NInfer）需要在可以中断服务时执行，见 README FAQ。
* Docker 面板对真实 docker 只读验证通过（1 个容器 / 1 个镜像 / 17 个 scope 外容器）；
  删除、prune、pull-timeout、cancel 用注入的 fake runner 单测覆盖。
