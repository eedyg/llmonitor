/* llm-monitor UI — vanilla JS, no build step. */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const MAX_POINTS = 3600;

  const TAB_KEY = "llm-monitor-tab";
  const TOK_RANGE_KEY = "llm-monitor-token-range";
  const TABS = ["overview", "requests", "engine", "docker"];

  const state = {
    snap: null,
    tab: "overview",
    engine: "vllm",
    page: 0,
    pageSize: 50,
    total: 0,
    connected: false,
    pollTimer: null,
    profileTouched: false,   // user picked a profile explicitly
    charts: [],
    schemas: {},
    userPickedEngine: false,
    tokenRange: null,     // /api/tokens response when a date range is applied
    renderedRange: null,
    ts: [],
    series: {},          // key -> array aligned with ts
    historyTimer: null,
  };

  const SERIES_KEYS = [
    "decode_tps", "prefill_tps", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms",
    "itl_ms", "kv_pct", "vram_pct", "gpu_util", "power_w", "temp_c", "fan_pct",
    "running", "waiting", "vram_used_mib",
    // KV offload tiers (sampled every kvcache.collect_interval_s)
    "kv_cpu_used_pct", "kv_cpu_active_pct", "kv_disk_pct", "kv_disk_used_gb",
    "kv_disk_read_mbps", "kv_disk_write_mbps",
  ];

  /* ── tiny helpers ───────────────────────────────────────────────────── */
  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function fmt(value, digits = 1, dash = "—") {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return dash;
    const n = Number(value);
    if (!Number.isFinite(n)) return dash;
    if (Math.abs(n) >= 10000) return (n / 1000).toFixed(1) + "k";
    return n.toFixed(digits);
  }

  function fmtInt(value, dash = "—") {
    if (value === null || value === undefined) return dash;
    return Number(value).toLocaleString("en-US");
  }

  function fmtTok(value, dash = "—") {
    if (value === null || value === undefined) return dash;
    const n = Number(value);
    if (!Number.isFinite(n)) return dash;
    if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
    return n.toLocaleString("en-US");
  }

  function fmtMoney(value) {
    if (value === null || value === undefined) return "—";
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    if (n !== 0 && Math.abs(n) < 0.01) return n.toFixed(4);
    return n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function fmtKwh(value) {
    const n = Number(value);
    if (!Number.isFinite(n) || n === 0) return "0";
    if (n >= 100) return n.toFixed(0);
    if (n >= 1) return n.toFixed(2);
    if (n >= 0.01) return n.toFixed(3);
    return String(Number(n.toPrecision(2)));
  }

  function fmtUptime(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h) return `${h}h${String(m).padStart(2, "0")}m`;
    if (m) return `${m}m${String(s).padStart(2, "0")}s`;
    return `${s}s`;
  }

  function todayIso() {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  }

  function fmtDaySpan(firstAt, lastAt) {
    if (!firstAt || !lastAt) return "—";
    const fmt = (ts) => {
      const d = new Date(Number(ts) * 1000);
      return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    };
    const a = fmt(firstAt);
    const b = fmt(lastAt);
    return a === b ? a : `${a} → ${b}`;
  }

  function fmtBytes(n) {
    if (!n) return "—";
    const units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let i = 0, v = Number(n);
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
    return `${v.toFixed(i ? 1 : 0)} ${units[i]}`;
  }

  function stripAnsi(text) {
    return String(text || "").replace(/\x1b\[[0-9;]*m/g, "");
  }

  function hhmmss(ts) {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString("zh-CN", { hour12: false });
  }

  function shortProto(p) {
    return { openai_chat: "chat", openai_text: "text", openai_responses: "responses", anthropic: "messages" }[p] || p;
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { "content-type": "application/json" },
      ...opts,
    });
    if (!res.ok) {
      let detail = "";
      try { detail = (await res.json()).detail || (await res.text()); } catch { /* ignore */ }
      throw new Error(`${res.status} ${detail}`);
    }
    return res.json();
  }

  function toast(title, body, kind = "") {
    const box = el("div", `toast ${kind}`);
    box.append(el("div", "t-title", title));
    if (body) box.append(el("div", "t-body", body));
    $("#toasts").append(box);
    setTimeout(() => { box.style.opacity = "0"; box.style.transition = "opacity .4s"; }, 4200);
    setTimeout(() => box.remove(), 4800);
  }

  /* ── theme ──────────────────────────────────────────────────────────── */
  const THEME_KEY = "llm-monitor-theme";

  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
    $("#theme-toggle").textContent = theme === "dark" ? "◑" : "◐";
    buildCharts();
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  /* ── charts ─────────────────────────────────────────────────────────── */
  const CHART_DEFS = [
    {
      id: "chart-speed",
      defs: [
        { key: "decode_tps", label: "decode", color: "accent", width: 2 },
        { key: "prefill_tps", label: "prefill", color: "good", width: 1.6, scale: "y2" },
      ],
      left: "tok/s", right: "prefill",
    },
    {
      id: "chart-latency",
      defs: [
        { key: "ttft_p50_ms", label: "TTFT p50", color: "accent", width: 2 },
        { key: "ttft_p90_ms", label: "TTFT p90", color: "warn", width: 1.4 },
        { key: "itl_ms", label: "ITL", color: "violet", width: 1.2 },
      ],
      left: "ms",
    },
    {
      id: "chart-cache",
      defs: [
        { key: "kv_pct", label: "KV", color: "good", width: 2 },
        { key: "vram_pct", label: "VRAM", color: "accent", width: 1.6 },
        { key: "gpu_util", label: "SM util", color: "warn", width: 1.2 },
      ],
      left: "%", range: [0, 100],
    },
    {
      id: "chart-kvoffload",
      defs: [
        { key: "kv_disk_pct", label: "磁盘占用", color: "violet", width: 2 },
        { key: "kv_cpu_used_pct", label: "CPU 占用", color: "accent", width: 1.4 },
        { key: "kv_disk_read_mbps", label: "读", color: "good", width: 1.2, scale: "y2" },
        { key: "kv_disk_write_mbps", label: "写", color: "warn", width: 1.2, scale: "y2" },
      ],
      left: "%", right: "MiB/s", range: [0, 100],
    },
    {
      id: "chart-power",
      defs: [
        { key: "power_w", label: "power", color: "bad", width: 2 },
        { key: "temp_c", label: "temp", color: "warn", width: 1.4, scale: "y2" },
      ],
      left: "W", right: "°C",
    },
  ];

  function dataFor(defs) {
    return [state.ts, ...defs.map((def) => state.series[def.key] || [])];
  }

  function ensureSeries() {
    for (const key of SERIES_KEYS) {
      if (!Array.isArray(state.series[key])) state.series[key] = [];
    }
  }

  function buildCharts() {
    for (const item of state.charts) {
      try { item.plot.destroy(); } catch { /* ignore */ }
    }
    state.charts = [];

    const stroke = cssVar("--muted");
    const grid = cssVar("--line");
    const font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';

    for (const spec of CHART_DEFS) {
      const host = document.getElementById(spec.id);
      if (!host) continue;
      host.replaceChildren();
      const width = Math.max(320, host.clientWidth - 16);
      const series = [{ label: "time", show: false }];
      for (const def of spec.defs) {
        series.push({
          label: def.label,
          stroke: cssVar(`--${def.color}`) || cssVar("--accent"),
          width: def.width,
          scale: def.scale || "y",
          points: { show: false },
          spanGaps: false,
          value: (u, v) => (v == null ? "—" : v.toFixed(1)),
        });
      }
      const timeFmt = (u, splits) => {
        const span = (u.scales.x.max || 0) - (u.scales.x.min || 0);
        return splits.map((t) => {
          const d = new Date(t * 1000);
          const p2 = (n) => String(n).padStart(2, "0");
          if (span > 6 * 3600) return `${p2(d.getHours())}:${p2(d.getMinutes())}`;
          if (span > 600) return `${p2(d.getHours())}:${p2(d.getMinutes())}`;
          return `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`;
        });
      };
      const axes = [
        { stroke, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid }, font, size: 62,
          values: timeFmt },
        {
          stroke, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid }, font, size: 46,
          scale: "y", label: spec.left, labelSize: 14, labelFont: font,
          ...(spec.range ? { range: () => spec.range } : {}),
        },
      ];
      if (spec.right) {
        axes.push({
          stroke, grid: { show: false }, ticks: { stroke: grid }, font, size: 46,
          scale: "y2", side: 1, label: spec.right, labelSize: 14, labelFont: font,
        });
      }
      const scales = { x: { time: true }, y: { auto: true } };
      if (spec.right) scales.y2 = { auto: true };
      const plot = new uPlot(
        {
          width,
          height: 196,
          padding: [8, 10, 0, 0],
          cursor: { sync: { key: "llm-monitor" }, points: { size: 7 } },
          legend: { show: true, live: true },
          scales,
          series,
          axes,
        },
        dataFor(spec.defs),
        host
      );
      state.charts.push({ spec, plot, keys: spec.defs.map((d) => d.key) });
    }
    pushChartData();
  }

  function pushChartData() {
    for (const item of state.charts) {
      try { item.plot.setData(dataFor(item.spec.defs)); } catch { /* ignore */ }
    }
  }

  function appendPoint(point) {
    if (!point || !point.ts) return;
    ensureSeries();
    const last = state.ts[state.ts.length - 1];
    if (last && point.ts <= last + 0.35) {
      // same second: overwrite the last sample
      state.ts[state.ts.length - 1] = point.ts;
      for (const key of SERIES_KEYS) state.series[key][state.series[key].length - 1] = point[key];
      return;
    }
    state.ts.push(point.ts);
    for (const key of SERIES_KEYS) state.series[key].push(point[key]);
    if (state.ts.length > MAX_POINTS) {
      state.ts.shift();
      for (const key of SERIES_KEYS) state.series[key].shift();
    }
  }

  async function loadSeries(windowSec = 3600) {
    ensureSeries();
    const data = await api(`/api/series?window=${windowSec}`);
    state.ts = data.ts || [];
    for (const key of SERIES_KEYS) state.series[key] = data[key] || [];
    pushChartData();
    drawSparklines();
  }

  /* ── sparklines ─────────────────────────────────────────────────────── */
  function drawSpark(key) {
    const canvas = document.querySelector(`canvas.spark[data-key="${key}"]`);
    if (!canvas) return;
    const values = (state.series[key] || []).filter((v) => v != null).slice(-80);
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 74;
    const h = canvas.clientHeight || 26;
    if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (values.length < 2) return;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = max - min || 1;
    const accent = cssVar("--accent");
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - 2 - ((v - min) / span) * (h - 4);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.4;
    ctx.globalAlpha = 0.75;
    ctx.stroke();
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.globalAlpha = 0.08;
    ctx.fillStyle = accent;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  function drawSparklines() {
    for (const key of ["decode_tps", "prefill_tps", "ttft_p50_ms", "kv_pct", "running", "gpu_util", "vram_pct", "power_w"]) {
      drawSpark(key);
    }
  }

  /* ── tabs ───────────────────────────────────────────────────────────── */
  function activateTab(name, opts = {}) {
    if (!TABS.includes(name)) name = "overview";
    state.tab = name;
    for (const btn of document.querySelectorAll(".tab")) {
      btn.classList.toggle("active", btn.dataset.tab === name);
    }
    for (const panel of document.querySelectorAll(".tab-panel")) {
      panel.hidden = panel.dataset.panel !== name;
    }
    if (opts.persist !== false) localStorage.setItem(TAB_KEY, name);
    if (opts.hash !== false) history.replaceState(null, "", `#${name}`);
    // charts need a laid-out container: rebuild whenever the tab becomes visible
    if (name === "overview") {
      buildCharts();
      drawSparklines();
    }
    if (name === "requests") {
      if (state.snap) renderLive(state.snap.live || []);
      loadHistory();
    }
    if (name === "engine") loadLogs();
    if (name === "docker") loadDocker();
  }

  function updateTabStatus(ctrl, live) {
    const dot = $("#tab-engine-dot");
    if (dot) dot.dataset.state = ctrl.state || "unknown";
    const badge = $("#tab-live-count");
    if (badge) {
      const count = live.length;
      badge.textContent = String(count);
      badge.hidden = count === 0;
    }
  }

  /* ── snapshot rendering ─────────────────────────────────────────────── */
  const STATE_LABEL = {
    unknown: "未知", stopped: "已停止", starting: "启动中", healthy: "运行中",
    stopping: "停止中", unhealthy: "不健康", failed: "失败",
  };

  function renderSnapshot(snap) {
    state.snap = snap;
    const ctrl = snap.controller || {};
    const engine = snap.engine || {};
    const model = engine.model || {};
    const gpu = (snap.gpu && snap.gpu.gpus && snap.gpu.gpus[0]) || {};

    $("#state-pill").dataset.state = ctrl.state || "unknown";
    $("#state-label").textContent = STATE_LABEL[ctrl.state] || ctrl.state || "未知";

    $("#chip-engine").textContent = `${ctrl.engine || "—"} · ${model.served || "no model"}`;
    $("#chip-ports").textContent = `:${ctrl.gateway_port || "?"} → :${ctrl.engine_port || "?"}`;
    const since = ctrl.since ? Math.max(0, Date.now() / 1000 - ctrl.since) : 0;
    $("#chip-uptime").textContent = ctrl.state === "healthy" ? `up ${Math.floor(since / 60)}m${String(Math.floor(since % 60)).padStart(2, "0")}s` : `state ${Math.floor(since)}s`;
    $("#ctrl-hint").textContent = ctrl.detail || "—";

    if (!state.userPickedEngine && ctrl.engine && ctrl.engine !== state.engine) {
      state.engine = ctrl.engine;
      selectEngineButton(state.engine);
      syncEngineDefaults();
    }

    renderStartup(ctrl);
    renderKpis(snap);
    renderEngineInfo(snap);
    renderGpu(snap);
    renderKvCache(snap);
    renderSummary(snap);
    renderTokens(snap);
    updateTabStatus(ctrl, snap.live || []);
    if (state.tab === "requests") renderLive(snap.live || []);

    appendPoint(snap.point);
    if (state.tab === "overview") {
      pushChartData();
      drawSparklines();
    }
  }

  function renderStartup(ctrl) {
    const box = $("#startup");
    const s = ctrl.startup;
    const show = s && (ctrl.state === "starting" || (s.error && ctrl.state === "failed"));
    if (!show) { box.hidden = true; return; }
    box.hidden = false;
    const phases = s.phases || [];
    const row = $("#startup-phases");
    row.replaceChildren();
    phases.forEach((name, i) => {
      const chip = el("span", "phase-chip", name);
      if (i < s.index) chip.classList.add("done");
      else if (i === s.index && !s.error) chip.classList.add("active");
      row.append(chip);
    });
    if (s.error) row.append(el("span", "phase-chip error", "错误"));
    const lines = (s.tail || []).map(stripAnsi).slice(-6).join("\n");
    const head = s.error ? (s.detail || "启动失败") : (s.detail || "等待引擎日志…");
    $("#startup-tail").textContent = head + (lines ? "\n" + lines : "");
  }

  function renderKpis(snap) {
    const speed = (snap.engine && snap.engine.speed) || {};
    const req = (snap.engine && snap.engine.requests) || {};
    const cache = (snap.engine && snap.engine.cache) || {};
    const tokens = (snap.engine && snap.engine.tokens) || {};
    const spec = (snap.engine && snap.engine.spec) || {};
    const gpu = (snap.gpu && snap.gpu.gpus && snap.gpu.gpus[0]) || {};

    $("#k-decode").textContent = fmt(speed.decode_tps, 1);
    $("#k-decode-1m").textContent = fmt(speed.decode_tps_60s, 1);
    $("#k-decode-sub").textContent = `ITL ${fmt(speed.itl_p50_ms, 1)}ms · 累计 ${fmtInt(tokens.generation)} tok`;
    $("#k-prefill").textContent = fmt(speed.prefill_tps, 0);
    $("#k-prefill-1m").textContent = fmt(speed.prefill_tps_60s, 0);
    // vLLM only books prompt tokens when a request completes, so a counter-rate
    // reads 0 % during a long prefill. The live figure is derived from KV growth
    // and the sub-line says which source is being shown.
    const prefillSrc = speed.prefill_source === "live" ? "实时（KV 增长推算）" : "结算值";
    $("#k-prefill-sub").textContent =
      `prompt 均值 ${fmt(speed.prompt_tokens_avg, 0)} tok · ${prefillSrc}`;
    $("#k-ttft").textContent = fmt(speed.ttft_p50_ms, 0);
    $("#k-ttft-sub").textContent = `p90 ${fmt(speed.ttft_p90_ms, 0)} · p99 ${fmt(speed.ttft_p99_ms, 0)} ms`;
    $("#k-kv").textContent = fmt(cache.kv_pct, 1);
    $("#k-kv-sub").textContent = `${fmtInt(cache.kv_used_pages)}/${fmtInt(cache.kv_capacity_pages)} pages · 命中 ${fmt((cache.hit_rate || 0) * 100, 1)}%`;
    $("#k-req").textContent = `${fmtInt(req.running)}`;
    $("#k-req-sub").textContent = `waiting ${fmtInt(req.waiting)} · prefill ${fmtInt(req.prefilling)} · 抢占 ${fmtInt(req.preemptions)}`;
    $("#k-gpu").textContent = fmt(gpu.util_pct, 0);
    $("#k-gpu-sub").textContent = `SM ${fmt(gpu.sm_clock_mhz, 0)}MHz · ${gpu.throttle && gpu.throttle.length ? "throttle: " + gpu.throttle.join(",") : "无降频"}`;
    $("#k-vram").textContent = fmtInt(gpu.mem_used_mib);
    $("#k-vram-sub").textContent = `/ ${fmtInt(gpu.mem_total_mib)} MiB · ${fmt(gpu.mem_used_pct, 1)}%`;
    $("#k-power").textContent = fmt(gpu.power_w, 0);
    $("#k-power-sub").textContent = `${fmt(gpu.temp_c, 0)}°C · 风扇 ${fmt(gpu.fan_pct, 0)}% · 限 ${fmt(gpu.power_limit_w, 0)}W`;

    const specTxt = spec.backend ? `${spec.backend} ${fmt((spec.accept_rate || 0) * 100, 1)}%` : "off";
    $("#k-decode-sub").textContent += ` · spec ${specTxt}`;
  }

  function phaseLabel(phase) {
    return { queued: "QUEUED", prefill: "PREFILL", decode: "DECODE", done: "DONE", error: "ERROR" }[phase] || (phase || "?").toUpperCase();
  }

  function renderLive(items) {
    const host = $("#live-list");
    $("#live-hint").textContent = items.length ? `${items.length} 个在飞` : "0 个在飞";
    host.replaceChildren();
    if (!items.length) {
      host.append(el("div", "empty", "当前没有在飞请求"));
      return;
    }
    for (const r of items) {
      const card = el("div", "req");
      card.dataset.phase = r.phase || "queued";

      const head = el("div", "req-head");
      head.append(el("span", `badge ${r.phase || ""}`, phaseLabel(r.phase)));
      head.append(el("span", "ep", shortProto(r.protocol) + (r.stream ? " · stream" : "")));
      head.append(el("span", "time", `${fmt((r.elapsed_ms || 0) / 1000, 2)}s`));
      card.append(head);

      const meta = el("div", "req-meta");
      const add = (label, value) => {
        const span = el("span");
        span.append(document.createTextNode(label + " "));
        span.append(el("b", null, value));
        meta.append(span);
      };
      add("prompt", `${fmtInt(r.prompt_tokens ?? r.prompt_tokens_est)} tok${r.prompt_cached ? ` (cache ${fmtInt(r.prompt_cached)})` : ""}`);
      add("out", `${fmtInt(r.thinking_tokens)} think + ${fmtInt(r.content_tokens)}`);
      add("decode", `${fmt(r.decode_tps ?? r.live_decode_tps, 1)} tok/s`);
      if (r.max_tokens) add("max", fmtInt(r.max_tokens));
      if (r.client_ip) add("from", r.client_ip);
      card.append(meta);

      if (r.thinking_tail) {
        const tail = el("div", "req-tail thinking");
        tail.append(el("span", "lbl", "思考"));
        tail.append(document.createTextNode(r.thinking_tail.slice(-320)));
        card.append(tail);
      }
      if (r.content_tail) {
        const tail = el("div", "req-tail");
        tail.append(el("span", "lbl", "输出"));
        tail.append(document.createTextNode(r.content_tail.slice(-320)));
        card.append(tail);
      }
      host.append(card);
    }
  }

  /* Cumulative token ledger: total (in+out) -> prompt / output -> hit|miss,
     thinking|content. Sourced from the SQLite request rows (read-only).
     Falls back to a date-range aggregate when the picker is applied. */
  function renderTokens(snap) {
    const totals = snap.tokens || {};
    const range = state.tokenRange;
    const session = totals.session || {};
    const history = totals.history || {};
    // Default view is the DB-backed total: it survives restarts, which is the
    // whole point of the ledger. "This run" lives in the sub-line.
    const s = (range ? range.scope : history) || {};
    const set = (sel, text, title) => {
      const node = $(sel);
      if (!node) return;
      node.textContent = text;
      node.title = title || "";
    };
    const pct = (part, whole) => (whole > 0 ? (100 * Number(part || 0)) / whole : null);
    const pctText = (value) => (value == null ? "" : `${fmt(value, 1)}%`);
    const approx = s.cache_estimated ? "≈" : "";

    if (range) {
      set("#tok-hint", `${range.start} → ${range.end} · ${range.days} 天`,
          `按本地时区整天聚合，含首尾两天；共 ${fmtInt(s.requests)} 个请求`);
    } else if (s.requests) {
      set("#tok-hint", `库内累计 · ${fmtDaySpan(s.first_at, s.last_at)}`,
          `state/monitor.db 里保留的全部请求（30 天保留期）· ${fmtInt(s.total)} tok`);
    } else {
      set("#tok-hint", "库内暂无记录", "还没有任何已完成的请求");
    }
    set("#tok-total", fmtTok(s.total), `${fmtInt(s.total)} tok（in + out）`);
    if (range) {
      const perDay = range.days > 0 ? Number(s.total || 0) / range.days : null;
      set("#tok-total-sub", `${fmtInt(s.requests)} 个请求 · 日均 ${fmtTok(perDay)} tok`,
          `按天汇总，共 ${range.days} 天`);
    } else {
      set("#tok-total-sub", `${fmtInt(s.requests)} 个请求 · 本次运行 ${fmtTok(session.total)} tok`,
          `本次运行 ${fmtUptime(session.uptime_s)} · ${fmtInt(session.requests)} 个请求 · ${fmtInt(session.total)} tok`);
    }

    set("#tok-prompt", fmtTok(s.prompt), `${fmtInt(s.prompt)} tok`);
    set("#tok-output", fmtTok(s.output), `${fmtInt(s.output)} tok`);

    const cacheUnknown = s.cache_unknown === true;
    const inPct = cacheUnknown ? null : pct(s.prompt_cached, s.prompt);
    const hitBar = $("#tok-in-hit-bar");
    const missBar = $("#tok-in-miss-bar");
    if (hitBar && missBar) {
      hitBar.style.width = inPct == null ? "0%" : `${inPct}%`;
      missBar.style.width = inPct == null ? "0%" : `${100 - inPct}%`;
    }
    // Say where the estimated ratio comes from: the engine counters reset on
    // every engine restart, so a small sample can look like a cache regression.
    const engCache = (snap.engine || {}).cache || {};
    const sample = Number(engCache.prefix_queries || 0);
    const estNote = sample > 0
      ? `按引擎侧命中率估算（本引擎进程 ${fmtInt(sample)} tok，命中 ${fmt(100 * (engCache.hit_rate || 0), 1)}%）`
      : "vLLM 响应里没有 cached_tokens，按引擎侧命中率估算";
    set("#tok-in-hit", cacheUnknown ? "未知" : `${approx}${fmtInt(s.prompt_cached)}`,
        s.cache_estimated
          ? estNote
          : cacheUnknown
            ? "引擎没报告 cached_tokens，也没有留存命中率 → 金额按全部未命中估算（上限）"
            : `${fmtInt(s.prompt_cached)} tok`);
    set("#tok-in-miss", cacheUnknown ? "未知" : `${approx}${fmtInt(s.prompt_miss)}`);
    set("#tok-in-hit-pct", pctText(inPct));
    set("#tok-in-miss-pct", inPct == null ? "" : pctText(100 - inPct));

    const thPct = pct(s.thinking, s.output);
    const thBar = $("#tok-out-think-bar");
    const ctBar = $("#tok-out-content-bar");
    if (thBar && ctBar) {
      thBar.style.width = thPct == null ? "0%" : `${thPct}%`;
      ctBar.style.width = thPct == null ? "0%" : `${100 - thPct}%`;
    }
    set("#tok-thinking", fmtInt(s.thinking), `${fmtInt(s.thinking)} tok`);
    set("#tok-content", fmtInt(s.content), `${fmtInt(s.content)} tok`);
    set("#tok-thinking-pct", pctText(thPct));
    set("#tok-content-pct", thPct == null ? "" : pctText(100 - thPct));

    renderTokenMoney(range, snap);

    if (range !== state.renderedRange) {
      renderTokenDays(range);
      state.renderedRange = range;
    }
  }

  /* Money: what these tokens would cost at the deepseek-flash rate card, plus
     the GPU's electricity. Both are approximations, both are read-only. */
  function renderTokenMoney(range, snap) {
    const block = range ? (range.billing || {}) : (snap.billing || {});
    const bill = range ? block.scope : block.history;
    const price = block.price || (snap.billing || {}).price || {};
    const set = (sel, text, title) => {
      const node = $(sel);
      if (!node) return;
      node.textContent = text;
      node.title = title || "";
    };
    const label = $("#tok-cost-ds-label");
    if (label && price.model) label.textContent = `${price.model} 等价费用`;
    if (!bill) {
      // No billing block in the snapshot: the server process predates it (the
      // static files are read from disk, so the UI can be newer than the API).
      set("#tok-cost-ds", "—");
      set("#tok-cost-ds-sub", "服务端未返回金额字段 · 重启 monitor 后生效",
          "快照里没有 billing，通常是 monitor 进程还是旧代码");
      set("#tok-cost-power", "—");
      set("#tok-cost-power-sub", "同上");
      return;
    }
    const ds = bill.deepseek || {};
    const energy = bill.energy || {};
    const peakPct = ds.peak_token_pct;
    set("#tok-cost-ds", fmtMoney(ds.cost),
        `${price.model || "deepseek"} 价目：命中 ${price.per_million?.cache_hit} / 未命中 ${price.per_million?.cache_miss} / 输出 ${price.per_million?.output} 元每百万 token（高峰价，空闲半价）`);
    const peakText = peakPct == null
      ? "无 token"
      : `高峰时段 ${fmt(peakPct, 1)}% · 全高峰价 ${fmtMoney(ds.peak_price)}`;
    set("#tok-cost-ds-sub",
        ds.cache_unknown ? `按全部未命中估算（上限）· ${peakText}` : peakText,
        `高峰时段多花 ${fmtMoney(ds.peak_surcharge)}；全在空闲时段 ${fmtMoney(ds.offpeak_price)}`);
    set("#tok-cost-power", fmtMoney(energy.cost),
        `GPU 功耗积分得到的电量 × ${price.cny_per_kwh ?? "?"} 元/kWh`);
    set("#tok-cost-power-sub",
        energy.kwh ? `${fmtKwh(energy.kwh)} kWh × ¥${fmt(price.cny_per_kwh, 2)}`
                   : "电量从本次升级后开始累计",
        energy.since ? `从 ${energy.since} 开始记录` : "");
  }

  /* Per-day rows for the selected range (oldest first). */
  function renderTokenDays(range) {
    const wrap = $("#tok-days-wrap");
    const table = $("#tok-days");
    if (!wrap || !table) return;
    if (!range || !(range.daily || []).length) {
      wrap.hidden = true;
      table.replaceChildren();
      return;
    }
    wrap.hidden = false;
    const head = el("tr");
    for (const label of ["日期", "请求", "总 token", "prompt in", "命中", "未命中",
                         "output", "thinking", "正文", "DS ¥", "电费 ¥"]) {
      head.append(el("th", null, label));
    }
    const thead = el("thead");
    thead.append(head);
    const body = el("tbody");
    for (const day of range.daily) {
      const mark = day.cache_estimated ? "≈" : (day.cache_unknown ? "?" : "");
      const tr = el("tr");
      tr.append(el("td", "mono", day.date));
      tr.append(el("td", null, fmtInt(day.requests)));
      tr.append(el("td", "strong", fmtInt(day.total)));
      tr.append(el("td", null, fmtInt(day.prompt)));
      tr.append(el("td", "good", `${mark}${fmtInt(day.prompt_cached)}`));
      tr.append(el("td", "dim", `${mark}${fmtInt(day.prompt_miss)}`));
      tr.append(el("td", null, fmtInt(day.output)));
      tr.append(el("td", "dim", fmtInt(day.thinking)));
      tr.append(el("td", null, fmtInt(day.content)));
      const money = day.billing || {};
      tr.append(el("td", "money", fmtMoney((money.deepseek || {}).cost)));
      tr.append(el("td", "money dim", fmtMoney((money.energy || {}).cost)));
      body.append(tr);
    }
    table.replaceChildren(thead, body);
  }

  async function applyTokenRange() {
    const from = $("#tok-from").value || todayIso();
    const to = $("#tok-to").value || from;
    try {
      const data = await api(`/api/tokens?start=${encodeURIComponent(from)}&end=${encodeURIComponent(to)}`);
      state.tokenRange = data;
      $("#tok-from").value = data.start;
      $("#tok-to").value = data.end;
      localStorage.setItem(TOK_RANGE_KEY, JSON.stringify({ start: data.start, end: data.end }));
      renderTokens(state.snap || {});
    } catch (err) {
      toast("区间汇总失败", String(err), "bad");
    }
  }

  function resetTokenRange() {
    state.tokenRange = null;
    localStorage.removeItem(TOK_RANGE_KEY);
    renderTokens(state.snap || {});
  }

  function renderSummary(snap) {    const summary = snap.summary || {};
    const hour = summary.hour || {};
    const set = (sel, value) => { const node = $(sel); if (node) node.textContent = value; };
    set("#sum-total", fmtInt(summary.total));
    set("#sum-errors", fmtInt(summary.errors));
    set("#sum-decode", hour.decode_tps_avg == null ? "—" : `${fmt(hour.decode_tps_avg, 1)} tok/s`);
    set("#sum-ttft", hour.ttft_ms_avg == null ? "—" : `${fmt(hour.ttft_ms_avg, 0)} ms`);
    set("#sum-prompt", hour.prompt_tokens_avg == null ? "—" : `${fmt(hour.prompt_tokens_avg, 0)} tok`);
    set("#sum-out", hour.content_tokens_avg == null ? "—" : `${fmt(hour.content_tokens_avg, 0)} tok`);
  }

  function renderEngineInfo(snap) {
    const engine = snap.engine || {};
    const model = engine.model || {};
    const cfg = engine.config || {};
    const cache = engine.cache || {};
    const req = engine.requests || {};
    const tokens = engine.tokens || {};
    const speed = engine.speed || {};
    const spec = engine.spec || {};
    const kv = snap.kvcache || {};
    const disk = kv.disk || {};
    const cpu = kv.cpu || {};

    $("#info-engine").textContent = engine.ok ? `${model.engine} · ${model.served}` : (snap.engine_error || "engine not reachable");

    const rows = [
      ["状态", engine.ok ? "serving" : "down"],
      ["启动方式", snap.controller ? (snap.controller.plan_source || "run_llm.sh") : "—"],
      ["模型", model.served || "—"],
      ["上下文上限", model.max_model_len ? fmtInt(model.max_model_len) + " tok" : (cfg.max_context ? fmtInt(cfg.max_context) + " tok" : "—")],
      ["KV 容量", cfg.kv_tokens ? fmtInt(cfg.kv_tokens) + " tok" : (cfg.max_concurrency ? "—" : "—")],
      ["KV dtype", cfg.kv_dtype || "—"],
      ["KV 页", `${fmtInt(cache.kv_used_pages)} / ${fmtInt(cache.kv_capacity_pages)}`],
      ["并发", cfg.max_concurrency ? fmt(cfg.max_concurrency, 2) : "—"],
      ["Prefix 查询/命中", `${fmtInt(cache.prefix_queries)} / ${fmtInt(cache.prefix_hits)}`],
      ["MTP 投机", spec.backend
        ? `${spec.backend} · drafted ${fmtInt(spec.drafted)} · 接受 ${fmt((spec.accept_rate || 0) * 100, 1)}%`
        : "off"],
      ["Prompt 总量", `${fmtInt(tokens.prompt)} tok`],
      ["生成总量", `${fmtInt(tokens.generation)} tok`],
      ["E2E p50/p90", `${fmt(speed.e2e_p50_ms, 0)} / ${fmt(speed.e2e_p90_ms, 0)} ms`],
      ["队列 p50", `${fmt(speed.queue_p50_ms, 1)} ms`],
      ["抢占 / replay", `${fmtInt(req.preemptions)} / ${fmtInt(req.replaying)}`],
      ["磁盘 cache 目录", kv.enabled ? (kv.dir || "—") : "offload 已关闭"],
      ["磁盘 cache 用量", kv.enabled
        ? `${fmt((disk.used_bytes || 0) / 1024 ** 3, 2)} / ${fmt((disk.capacity_bytes || 0) / 1024 ** 3, 1)} GiB`
          + (disk.limit_bytes ? `（上限 ${fmt(disk.limit_bytes / 1024 ** 3, 1)} GiB）` : "（不限）")
        : "—"],
      ["内存 cache", kv.enabled
        ? `${fmt((cpu.budget_bytes || 0) / 1024 ** 3, 1)} GiB 预算 · 实占 `
          + `${fmt((cpu.region_used_bytes || 0) / 1024 ** 3, 2)} GiB · 在飞 ${fmt(cpu.active_pct, 0)}%`
        : "—"],
    ];
    const dl = $("#engine-info");
    dl.replaceChildren();
    for (const [k, v] of rows) {
      dl.append(el("dt", null, k));
      dl.append(el("dd", null, v));
    }
  }

  function renderGpu(snap) {
    const gpu = (snap.gpu && snap.gpu.gpus && snap.gpu.gpus[0]) || null;
    const dl = $("#gpu-info");
    dl.replaceChildren();
    $("#info-gpu-mode").textContent = (snap.gpu && snap.gpu.mode) || "off";
    if (!gpu) {
      dl.append(el("dt", null, "状态"));
      dl.append(el("dd", null, "NVML unavailable"));
      $("#gpu-procs").replaceChildren();
      return;
    }
    const rows = [
      ["型号", gpu.name],
      ["显存", `${fmtInt(gpu.mem_used_mib)} / ${fmtInt(gpu.mem_total_mib)} MiB`],
      ["利用率", `${fmt(gpu.util_pct, 0)} % (mem ${fmt(gpu.mem_util_pct, 0)}%)`],
      ["功耗", `${fmt(gpu.power_w, 1)} / ${fmt(gpu.power_limit_w, 0)} W`],
      ["温度", `${fmt(gpu.temp_c, 0)} °C (limit ${fmt(gpu.temp_limit_c, 0)})`],
      ["风扇", `${fmt(gpu.fan_pct, 0)} %`],
      ["时钟", `SM ${fmt(gpu.sm_clock_mhz, 0)} · MEM ${fmt(gpu.mem_clock_mhz, 0)} MHz`],
      ["PCIe", `Gen${fmt(gpu.pcie_gen, 0)} x${fmt(gpu.pcie_width, 0)}`],
      ["降频", gpu.throttle && gpu.throttle.length ? gpu.throttle.join(", ") : "无"],
    ];
    for (const [k, v] of rows) {
      dl.append(el("dt", null, k));
      dl.append(el("dd", null, v));
    }
    const procs = $("#gpu-procs");
    procs.replaceChildren();
    for (const proc of gpu.procs || []) {
      const row = el("div", "proc");
      row.append(el("span", null, `${proc.name} (${proc.pid})`));
      row.append(el("span", null, `${fmtInt(proc.mem_mib)} MiB`));
      procs.append(row);
    }
  }

  /* ── history ────────────────────────────────────────────────────────── */
  async function loadHistory() {
    const params = new URLSearchParams({
      limit: String(state.pageSize),
      offset: String(state.page * state.pageSize),
    });
    const search = $("#filter-search").value.trim();
    const engine = $("#filter-engine").value;
    const status = $("#filter-status").value;
    if (search) params.set("search", search);
    if (engine) params.set("engine", engine);
    if (status) params.set("status", status);
    try {
      const data = await api(`/api/requests?${params}`);
      state.total = data.total;
      renderHistory(data.items || []);
    } catch (err) {
      toast("加载历史失败", String(err), "bad");
    }
  }

  function scheduleHistory() {
    if (state.tab !== "requests") return;
    if (state.historyTimer) return;
    state.historyTimer = setTimeout(() => {
      state.historyTimer = null;
      loadHistory();
    }, 1200);
  }

  function renderKvCache(snap) {
    const kv = snap.kvcache || {};
    const box = $("#kv-tiers");
    const dl = $("#kv-tier-info");
    if (!box || !dl) return;
    box.replaceChildren();
    dl.replaceChildren();
    const enabled = !!kv.enabled;
    $("#kv-tier-hint").textContent = enabled
      ? `${kv.dir || "—"} · 低频采样`
      : "未启用（config.yaml → kvcache.enabled）";
    const gpu = kv.gpu || {};
    const cpu = kv.cpu || {};
    const disk = kv.disk || {};
    const gib = (bytes) => (bytes ? `${fmt(bytes / 1024 ** 3, 1)} GiB` : "—");
    const mbps = (bps) => {
      if (bps === null || bps === undefined) return "—";
      if (bps >= 1024 ** 3) return `${fmt(bps / 1024 ** 3, 2)} GiB/s`;
      if (bps >= 1024 ** 2) return `${fmt(bps / 1024 ** 2, 1)} MiB/s`;
      return `${fmt(bps / 1024, 0)} KiB/s`;
    };
    // 用「生效上限」（= capacity − 保留区）而不是整个分区：limit=0 时 vLLM 会一直
    // 写到 ENOSPC（实测 2102 次写失败、prefill 慢 4 倍），见 kvcache.py 的保留区注释。
    const diskLimit = Number(disk.effective_limit_bytes || disk.capacity_bytes || 0);
    const diskPct = diskLimit
      ? (Number(disk.used_bytes || 0) / diskLimit) * 100
      : null;
    const rows = [
      {
        name: "GPU 显存",
        pct: gpu.pct,
        text: gpu.blocks
          ? `${fmt(gpu.pct, 1)}% · ${fmtInt(gpu.blocks)} blocks`
          : `${fmt(gpu.pct, 1)}%`,
        color: "var(--good)",
      },
      {
        name: "CPU 内存",
        // vLLM 没有 CPU 层占用率指标，只在飞占用率。真实占用取自 tmpfs mmap
        // 的实占块数（宿主机侧），所以这里显示「实占」，「在飞」只作参考。
        pct: cpu.budget_bytes ? (100.0 * (cpu.region_used_bytes || 0)) / cpu.budget_bytes : cpu.active_pct,
        text: `${gib(cpu.budget_bytes)} 预算 · 实占 ${gib(cpu.region_used_bytes)}`
          + ` · 在飞 ${fmt(cpu.active_pct, 0)}%`,
        color: "var(--accent)",
        title: "实占 = /dev/shm 区域真正落地的内存；在飞 = 被进行中传输临时钉住的占比（vLLM 唯一暴露的指标）",
      },
      {
        name: "磁盘",
        pct: diskPct,
        text: `${gib(disk.used_bytes)} / ${gib(diskLimit)}`
          + ` · 剩 ${gib(disk.free_bytes)}`
          + (disk.limit_bytes ? `（上限 ${gib(disk.limit_bytes)}）`
                              : `（自动保留 ${gib(disk.reserve_bytes)}）`),
        color: disk.near_full ? "var(--bad)" : "var(--violet)",
        title: disk.limit_bytes
          ? "受 kvcache.disk_limit_gb 限制"
          : "limit=0 → 用满分区但留出保留区，避免 vLLM 写到 ENOSPC",
      },
    ];
    for (const row of rows) {
      const wrap = el("div", enabled ? "kv-tier" : "kv-tier off");
      if (row.title) wrap.title = row.title;
      wrap.append(el("span", "kv-tier-name", row.name));
      const bar = el("div", "kv-bar");
      const fill = el("i");
      fill.style.width = `${Math.max(0, Math.min(100, Number(row.pct) || 0))}%`;
      fill.style.background = row.color;
      bar.append(fill);
      wrap.append(bar);
      wrap.append(el("span", "kv-tier-val", row.text));
      box.append(wrap);
    }
    const info = [
      ["磁盘 IO", `读 ${mbps(disk.read_bps)} · 写 ${mbps(disk.write_bps)}`],
      ["磁盘命中", (disk.hit_rate === null || disk.hit_rate === undefined)
        ? "—"
        : `${fmt(disk.hit_rate * 100, 1)} %（${fmtInt(disk.hits)}/${fmtInt(disk.queries)} 块）`],
      ["磁盘文件", `${fmtInt(disk.files)} 个 · 生效上限 ${gib(diskLimit)}`
        + (disk.limit_bytes ? "" : `（自动，保留 ${gib(disk.reserve_bytes)}）`)],
      ["失败/丢弃", `升级失败 ${fmtInt(disk.promo_fail)} · 背压丢弃 ${fmtInt(disk.dropped_stores)}`],
    ];
    if (disk.near_full) {
      info.unshift(["⚠️ 磁盘将满", `剩余 ${gib(disk.free_bytes)} < 保留区 `
        + `${gib(disk.reserve_bytes)}：offload 写入即将失败（ENOSPC）`]);
    }
    const health = kv.health || {};
    info.push(["磁盘健康", health.error
      ? `不可用：${health.error}`
      : `已用 ${fmt(health.pct_used, 0)}% · ${fmt(health.temp_c, 1)}°C · 余量 ${fmt(health.spare_pct, 0)}%`
        + ` · 媒体错误 ${fmtInt(health.media_errors)} · 累计写 ${fmt(health.written_gb, 0)} GiB`]);
    if (kv.reclaim && kv.reclaim.last_at) {
      const rec = kv.reclaim;
      info.push(["上次回收", rec.error
        ? `失败：${rec.error}`
        : `${gib(rec.last_bytes)} / ${fmtInt(rec.last_files)} 个文件`]);
    }
    for (const [k, v] of info) {
      dl.append(el("dt", null, k));
      dl.append(el("dd", null, v));
    }
  }

  function renderHistory(items) {
    const body = $("#history-body");
    body.replaceChildren();
    if (!items.length) {
      const tr = el("tr");
      const td = el("td", null, "暂无记录");
      td.colSpan = 12;
      td.style.textAlign = "center";
      td.style.color = "var(--muted)";
      td.style.padding = "26px";
      tr.append(td);
      body.append(tr);
    }
    for (const r of items) {
      const tr = el("tr");
      tr.addEventListener("click", () => openDetail(r.id));

      tr.append(el("td", "mono", hhmmss(r.started_at)));
      const engineTd = el("td");
      const tag = el("span", `engine-tag ${r.engine || ""}`, r.engine || "—");
      engineTd.append(tag);
      tr.append(engineTd);
      tr.append(el("td", null, shortProto(r.protocol) + (r.stream ? " ⚡" : "")));

      const prompt = r.prompt_tokens ?? r.prompt_tokens_est;
      tr.append(el("td", "num", fmtInt(prompt) + (r.prompt_tokens == null && prompt != null ? "~" : "")));
      const cachePct = r.prompt_tokens ? Math.round((100 * (r.prompt_cached || 0)) / r.prompt_tokens) : null;
      tr.append(el("td", "num", cachePct == null ? "—" : cachePct + "%"));
      tr.append(el("td", "num", fmtInt(r.thinking_tokens)));
      tr.append(el("td", "num", fmtInt(r.content_tokens)));
      tr.append(el("td", "num", r.ttft_ms == null ? "—" : fmt(r.ttft_ms, 0)));
      tr.append(el("td", "num", fmt(r.decode_tps, 1)));
      tr.append(el("td", "num", r.elapsed_ms == null ? "—" : (r.elapsed_ms / 1000).toFixed(2) + "s"));

      const statusTd = el("td");
      const ok = (r.status || 0) < 400 && !r.error;
      statusTd.append(el("span", ok ? "ok" : "bad", ok ? `${r.status || "ok"}` : "ERR"));
      tr.append(statusTd);
      tr.append(el("td", "mono", r.client_ip || "—"));

      body.append(tr);
    }
    const pages = Math.max(1, Math.ceil(state.total / state.pageSize));
    $("#page-info").textContent = `第 ${state.page + 1} / ${pages} 页 · 共 ${state.total} 条`;
    $("#page-prev").disabled = state.page === 0;
    $("#page-next").disabled = state.page + 1 >= pages;
  }

  async function openDetail(id) {
    const drawer = $("#drawer");
    const body = $("#drawer-body");
    body.replaceChildren(el("div", "empty", "加载中…"));
    drawer.classList.add("open");
    drawer.setAttribute("aria-hidden", "false");
    $("#scrim").classList.add("show");
    try {
      const r = await api(`/api/requests/${id}`);
      body.replaceChildren();

      const cell = (label, value) => {
        const c = el("div", "cell");
        c.append(el("span", null, label));
        c.append(el("b", null, value));
        return c;
      };
      const grid = el("div", "detail-grid");
      grid.append(cell("prompt", fmtInt(r.prompt_tokens ?? r.prompt_tokens_est)));
      grid.append(cell("cached", fmtInt(r.prompt_cached)));
      grid.append(cell("thinking", fmtInt(r.thinking_tokens)));
      grid.append(cell("content", fmtInt(r.content_tokens)));
      grid.append(cell("ttft", r.ttft_ms == null ? "—" : fmt(r.ttft_ms, 0) + "ms"));
      grid.append(cell("decode", fmt(r.decode_tps, 1) + " t/s"));
      grid.append(cell("total", r.elapsed_ms == null ? "—" : (r.elapsed_ms / 1000).toFixed(2) + "s"));
      grid.append(cell("request", fmtBytes(r.request_bytes)));
      body.append(grid);

      const meta = el("div", "detail-section");
      meta.append(el("h3", null, "元数据"));
      const pre = el("pre", "json");
      pre.textContent = JSON.stringify(
        {
          id: r.id,
          engine: r.engine,
          endpoint: r.endpoint,
          protocol: r.protocol,
          stream: r.stream,
          model: r.model,
          finish_reason: r.finish_reason,
          status: r.status,
          error: r.error,
          client: r.client_ip,
          user_agent: r.client_ua,
          sampling: r.sampling,
          engine_request_id: r.engine_request_id,
          usage: r.usage,
        },
        null,
        2
      );
      meta.append(pre);
      body.append(meta);

      if (r.thinking_text) {
        const section = el("div", "detail-section");
        section.append(el("h3", null, "思考输出 reasoning"));
        const block = el("pre", "thinking");
        block.textContent = r.thinking_text;
        section.append(block);
        body.append(section);
      }
      if (r.content_text) {
        const section = el("div", "detail-section");
        section.append(el("h3", null, "正式输出 content"));
        const block = el("pre");
        block.textContent = r.content_text;
        section.append(block);
        body.append(section);
      }
      if (r.engine_record) {
        const section = el("div", "detail-section");
        section.append(el("h3", null, "引擎原始记录 (NInfer JSONL)"));
        const block = el("pre", "json");
        block.textContent = JSON.stringify(r.engine_record, null, 2);
        section.append(block);
        body.append(section);
      }
    } catch (err) {
      body.replaceChildren(el("div", "empty", String(err)));
    }
  }

  function closeDrawer() {
    $("#drawer").classList.remove("open");
    $("#drawer").setAttribute("aria-hidden", "true");
    $("#scrim").classList.remove("show");
  }

  /* ── engine schema form ─────────────────────────────────────────────── */
  function savedOptions(engine) {
    try {
      return JSON.parse(localStorage.getItem(`llm-monitor-opts-${engine}`) || "{}");
    } catch {
      return {};
    }
  }

  function saveOption(engine, key, value) {
    const values = savedOptions(engine);
    values[key] = value;
    localStorage.setItem(`llm-monitor-opts-${engine}`, JSON.stringify(values));
  }

  function selectEngineButton(engine) {
    for (const btn of document.querySelectorAll("#engine-seg button")) {
      btn.classList.toggle("active", btn.dataset.engine === engine);
    }
  }

  async function loadEngines() {
    try {
      const data = await api("/api/engines");
      state.schemas = data.engines || {};
    } catch (err) {
      state.schemas = {};
    }
    if (!state.schemas[state.engine]) {
      state.engine = Object.keys(state.schemas)[0] || "vllm";
    }
    selectEngineButton(state.engine);
    renderProfiles();
    renderEngineOptions();
  }

  function schema() {
    return state.schemas[state.engine] || { profiles: [], options: [] };
  }

  function renderProfiles() {
    const sel = $("#profile");
    const current = sel.value;
    const sch = schema();
    const profiles = sch.profiles || [];
    sel.replaceChildren();
    for (const name of profiles) {
      const option = el("option", null, name);
      option.value = name;
      sel.append(option);
    }
    // The static <option> used to win here, which silently pinned the default to
    // stage1. Only honour the current selection when the user actually changed it.
    if (state.profileTouched && profiles.includes(current)) sel.value = current;
    else if (sch.default_profile && profiles.includes(sch.default_profile)) {
      sel.value = sch.default_profile;
    } else if (profiles.includes(current)) sel.value = current;
    else if (profiles.length) sel.value = profiles[0];
  }

  function renderEngineOptions() {
    const host = $("#engine-options");
    if (!host) return;
    host.replaceChildren();
    const s = schema();
    const profile = $("#profile").value;
    const saved = savedOptions(state.engine);
    const values = {};
    for (const opt of s.options || []) {
      values[opt.key] = saved[opt.key] !== undefined ? saved[opt.key] : opt.default;
    }
    const norm = (v) => (v === true ? "1" : v === false ? "0" : String(v ?? ""));
    for (const opt of s.options || []) {
      if (opt.show_if) {
        const cond = opt.show_if;
        let visible = true;
        if (Array.isArray(cond.profiles) && !cond.profiles.includes(profile)) visible = false;
        for (const [key, allowed] of Object.entries(cond)) {
          if (key === "profiles") continue;
          const list = (Array.isArray(allowed) ? allowed : [allowed]).map(norm);
          if (!list.includes(norm(values[key]))) { visible = false; break; }
        }
        if (!visible) continue;
      }
      const field = el("label", "field");
      field.dataset.key = opt.key;
      field.append(el("span", null, opt.label || opt.key));
      let input;
      if (opt.type === "enum" || opt.type === "bool") {
        input = el("select");
        const choices = opt.type === "bool" ? ["1", "0"] : (opt.choices || []);
        for (const choice of choices) {
          const label = opt.type === "bool"
            ? (choice === "1" ? "开" : "关")
            : ((opt.choice_labels && opt.choice_labels[choice]) || choice || "默认");
          const option = el("option", null, label);
          option.value = choice;
          input.append(option);
        }
      } else {
        input = el("input");
        input.type = opt.type === "password" ? "password"
          : (opt.type === "int" || opt.type === "float") ? "number" : "text";
        if (opt.step) input.step = String(opt.step);
        if (opt.placeholder) input.placeholder = opt.placeholder;
      }
      input.dataset.key = opt.key;
      input.classList.add("mono");
      const raw = saved[opt.key] !== undefined ? saved[opt.key] : (opt.default ?? "");
      input.value = raw === null || raw === undefined ? "" : String(raw);
      input.addEventListener("change", () => {
        saveOption(state.engine, opt.key, input.value);
        previewCommand();
      });
      field.append(input);
      if (opt.hint) {
        const hint = el("small", "field-hint", opt.hint);
        hint.title = opt.hint;
        field.append(hint);
      }
      host.append(field);
    }
  }

  function syncEngineDefaults() {
    const s = schema();
    if (s.default_ctx) $("#ctx").value = s.default_ctx;
    renderProfiles();
    renderEngineOptions();
  }

  /* ── actions ────────────────────────────────────────────────────────── */
  function collectOpts() {
    const extra = $("#extra").value.trim();
    const opts = {
      engine: state.engine,
      profile: $("#profile").value,
      ctx: $("#ctx").value,
      extra_args: extra ? extra.split(/\s+/).filter(Boolean) : [],
    };
    for (const input of document.querySelectorAll("#engine-options [data-key]")) {
      opts[input.dataset.key] = input.value;
    }
    return opts;
  }

  async function post(path, body) {
    try {
      return await api(path, { method: "POST", body: JSON.stringify(body || {}) });
    } catch (err) {
      toast("操作失败", String(err), "bad");
      return null;
    }
  }

  async function previewCommand() {
    const opts = collectOpts();
    try {
      const data = await api("/api/engine/plan", { method: "POST", body: JSON.stringify(opts) });
      const parts = [];
      if (Array.isArray(data.command)) {
        parts.push(data.command.map((c) => (c.includes(" ") ? JSON.stringify(c) : c)).join(" "));
      } else {
        parts.push(data.command);
      }
      for (const warning of data.warnings || []) parts.push(`# WARN ${warning}`);
      for (const error of data.errors || []) parts.push(`# ERROR ${error}`);
      $("#cmd-preview").textContent = parts.join("\n");
    } catch (err) {
      $("#cmd-preview").textContent = String(err);
    }
  }

  async function applyEngine() {
    const ctrl = (state.snap && state.snap.controller) || {};
    const opts = collectOpts();
    if (ctrl.state === "starting" || ctrl.state === "stopping") {
      toast("请稍候", "引擎正在切换中", "warn");
      return;
    }
    if (ctrl.state === "healthy" || ctrl.state === "unhealthy") {
      if (ctrl.engine !== opts.engine) {
        if (!confirm(`当前运行 ${ctrl.engine}，切换到 ${opts.engine}？\n会先停止当前引擎再启动新的。`)) return;
        const res = await post("/api/engine/switch", opts);
        if (res && res.ok) toast("切换已开始", `${ctrl.engine} → ${opts.engine}`, "ok");
      } else {
        if (!confirm(`重新创建 ${opts.engine}？当前在飞请求会被中断。`)) return;
        const res = await post("/api/engine/restart", opts);
        if (res && res.ok) toast("重启已开始", opts.engine, "ok");
      }
      return;
    }
    const res = await post("/api/engine/start", opts);
    if (res && res.ok) toast("启动已开始", `${opts.engine} · ${opts.profile}`, "ok");
    else if (res && res.error) toast("启动失败", res.error, "bad");
  }

  async function stopEngine() {
    if (!confirm("停止当前引擎？在飞请求会被中断。")) return;
    const res = await post("/api/engine/stop", {});
    if (res && res.ok) toast("已停止", "", "ok");
    else if (res && res.error) toast("停止失败", res.error, "bad");
  }

  async function restartEngine() {
    if (!confirm("重启当前引擎？")) return;
    const res = await post("/api/engine/restart", collectOpts());
    if (res && res.ok) toast("重启已开始", "", "ok");
    else if (res && res.error) toast("重启失败", res.error, "bad");
  }

  async function runProbe() {
    const box = $("#probe-results");
    box.replaceChildren(el("span", "hint", "探测中…"));
    try {
      const extended = $("#probe-extended").checked;
      const res = await api("/api/engine/probe", {
        method: "POST", body: JSON.stringify({ extended }),
      });
      box.replaceChildren();
      for (const t of res.tests) {
        const chip = el("div", `probe ${t.ok ? "ok" : "bad"}`);
        chip.append(el("span", "dot"));
        chip.append(el("span", null, t.name));
        chip.append(el("span", "ms", `${t.ms}ms`));
        if (!t.ok && t.detail) chip.title = t.detail;
        box.append(chip);
      }
      if (res.engine_alive_after === false) {
        toast("探针后引擎不可用", res.warning || "请重启引擎", "bad");
      } else {
        toast(res.ok ? "契约探针全部通过" : "契约探针有失败项", res.ok ? "" : "查看详情", res.ok ? "ok" : "bad");
      }
    } catch (err) {
      box.replaceChildren();
      toast("探针失败", String(err), "bad");
    }
  }

  async function loadLogs() {
    try {
      const data = await api("/api/logs?tail=80");
      const pre = $("#log-tail");
      pre.textContent = data.error
        ? `[${data.error}]`
        : (data.lines || []).map(stripAnsi).join("\n");
      pre.scrollTop = pre.scrollHeight;
    } catch (err) {
      $("#log-tail").textContent = String(err);
    }
  }

  /* ── docker (vLLM scope) ────────────────────────────────────────────── */
  function fmtMb(mb) {
    if (mb == null) return "—";
    if (mb >= 1024) return (mb / 1024).toFixed(1) + " GiB";
    return Math.round(mb) + " MiB";
  }

  function fmtDate(iso) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "—";
    return d.toLocaleDateString("zh-CN", { year: "2-digit", month: "2-digit", day: "2-digit" });
  }

  async function loadDocker() {
    try {
      const data = await api("/api/docker/summary");
      state.docker = data;
      renderDocker(data);
    } catch (err) {
      $("#docker-scope").textContent = "docker 不可用";
      $("#docker-disk").textContent = "";
      $("#docker-images").replaceChildren();
      $("#docker-containers").replaceChildren(el("div", "empty", String(err)));
    }
  }

  function renderDocker(data) {
    const scope = data.scope || {};
    if (!data.available) {
      $("#docker-scope").textContent = data.enabled === false ? "已禁用" : "不可用";
      $("#docker-disk").textContent = "";
      $("#docker-images").replaceChildren();
      $("#docker-containers").replaceChildren(el("div", "empty", data.error || data.reason || "—"));
      return;
    }
    $("#docker-scope").textContent = "scope: " + (scope.image_repos || []).join(", ");
    $("#docker-disk").textContent = `镜像 ${fmtMb(data.disk.images_mb)} · 可回收 ${fmtMb(data.disk.reclaimable_mb)}`;
    $("#docker-note").textContent =
      `只读枚举了 ${data.other_containers || 0} 个 scope 外容器用于安全检查；写操作仅限 ${(scope.containers || []).join(", ")} 与上述仓库。`;

    // -- containers ------------------------------------------------------
    const box = $("#docker-containers");
    box.replaceChildren();
    if (!(data.containers || []).length) {
      box.append(el("div", "empty", "没有 vLLM 容器"));
    }
    for (const c of data.containers || []) {
      const card = el("div", "docker-item");
      const head = el("div", "di-head");
      head.append(el("span", "di-name", c.name));
      head.append(el("span", `img-state ${c.state === "running" ? "current" : "old"}`, c.state));
      card.append(head);
      for (const [k, v] of [
        ["image", c.image_ref],
        ["size", c.size || "—"],
        ["created", c.created],
        ["status", c.status],
      ]) {
        const row = el("div", "di-row");
        row.append(el("span", null, k));
        row.append(el("b", null, v));
        card.append(row);
      }
      const actions = el("div", "di-actions");
      if (c.state === "running") {
        const stop = el("button", "btn micro danger", "停止引擎");
        stop.addEventListener("click", stopEngine);
        actions.append(stop);
      }
      const rm = el("button", "btn micro", "删除容器");
      if (c.state === "running") {
        rm.disabled = true;
        rm.title = "先停止引擎";
      } else {
        rm.addEventListener("click", () => dockerRemoveContainer(c.name));
      }
      actions.append(rm);
      card.append(actions);
      box.append(card);
    }

    // -- images ----------------------------------------------------------
    const body = $("#docker-images");
    body.replaceChildren();
    if (!(data.images || []).length) {
      const tr = el("tr");
      const td = el("td", null, "scope 内没有镜像");
      td.colSpan = 7;
      td.style.textAlign = "center";
      td.style.color = "var(--muted)";
      td.style.padding = "22px";
      tr.append(td);
      body.append(tr);
    }
    for (const img of data.images || []) {
      const tr = el("tr");

      const tagTd = el("td");
      for (const tag of img.in_scope_tags) tagTd.append(el("span", "img-tag", tag));
      for (const tag of img.out_of_scope_tags) {
        const span = el("span", "img-tag out", tag);
        span.title = "不在 scope 内，不会被操作";
        tagTd.append(span);
      }
      if (img.dangling) tagTd.append(el("span", "img-tag out", "<none>"));
      tr.append(tagTd);

      tr.append(el("td", "mono", img.short));
      tr.append(el("td", "num", fmtMb(img.size_mb)));
      tr.append(el("td", "mono", fmtDate(img.created)));

      const usedTd = el("td", "mono", (img.in_use_by || []).join(", ") || "—");
      tr.append(usedTd);

      const stateTd = el("td");
      let stateLabel = "未使用", stateClass = "";
      if (img.in_use_here) { stateLabel = "当前使用"; stateClass = "current"; }
      else if (img.in_use_by && img.in_use_by.length) { stateLabel = "其他容器"; stateClass = "used"; }
      else if (img.dangling) { stateLabel = "悬空"; stateClass = "old"; }
      else if (img.tracked) { stateLabel = "旧版本"; stateClass = "old"; }
      else if (!img.removable) { stateLabel = "锁定"; stateClass = "blocked"; }
      const chip = el("span", `img-state ${stateClass}`, stateLabel);
      if (img.blocked_reason) chip.title = img.blocked_reason;
      stateTd.append(chip);
      tr.append(stateTd);

      const actions = el("td");
      if (img.in_scope_tags.length) {
        const use = el("button", "btn micro", "用它重启");
        use.addEventListener("click", (e) => { e.stopPropagation(); dockerRestartWith(img.in_scope_tags[0]); });
        actions.append(use);
        actions.append(document.createTextNode(" "));
      }
      const rm = el("button", "btn micro danger", "删除");
      if (!img.removable) {
        rm.disabled = true;
        rm.title = img.blocked_reason || "";
      } else {
        rm.addEventListener("click", (e) => { e.stopPropagation(); dockerRemoveImage(img.id); });
      }
      actions.append(rm);
      tr.append(actions);
      body.append(tr);
    }

    // ledger: which images monitor pulled / superseded / removed
    const ledger = data.ledger || [];
    const ledgerBox = $("#docker-ledger");
    if (ledgerBox) {
      ledgerBox.hidden = ledger.length === 0;
      const body = $("#ledger-body");
      body.replaceChildren();
      for (const row of ledger) {
        const tr = el("tr");
        tr.append(el("td", null, String(row.id || "").replace("sha256:", "").slice(0, 12)));
        let refs = [];
        try { refs = JSON.parse(row.refs || "[]"); } catch { refs = []; }
        tr.append(el("td", null, refs.join(", ") || "—"));
        tr.append(el("td", null, row.first_seen ? new Date(row.first_seen * 1000).toLocaleString("zh-CN", { hour12: false }) : "—"));
        tr.append(el("td", null, row.superseded_at ? new Date(row.superseded_at * 1000).toLocaleString("zh-CN", { hour12: false }) : "—"));
        tr.append(el("td", null, row.removed_at ? new Date(row.removed_at * 1000).toLocaleString("zh-CN", { hour12: false }) : "—"));
        tr.append(el("td", null, row.source || "—"));
        body.append(tr);
      }
    }

    if (!state.dockerRefSet) {
      const firstTag = (data.containers || [])[0]?.image_ref
        || (data.images || []).flatMap((i) => i.in_scope_tags)[0];
      if (firstTag) { $("#docker-ref").value = firstTag; state.dockerRefSet = true; }
    }
  }

  async function dockerPull() {
    const ref = $("#docker-ref").value.trim();
    if (!ref) return;
    const res = await post("/api/docker/pull", { ref });
    if (!res || !res.ok) return;
    $("#docker-log").textContent = `pulling ${ref} …`;
    state.dockerJob = res.job.id;
    $("#btn-docker-cancel").hidden = false;
    pollDockerJob(res.job.id, ref);
  }

  async function dockerCancel() {
    if (!state.dockerJob) return;
    try {
      await api(`/api/docker/jobs/${state.dockerJob}/cancel`, { method: "POST", body: "{}" });
      toast("已请求取消", "docker pull 正在终止", "warn");
    } catch (err) {
      toast("取消失败", String(err), "bad");
    }
  }

  function pollDockerJob(id, ref) {
    const timer = setInterval(async () => {
      try {
        const job = await api(`/api/docker/jobs/${id}?lines=400`);
        const pre = $("#docker-log");
        pre.textContent = (job.lines && job.lines.length ? job.lines.join("\n") : job.progress || "…");
        pre.scrollTop = pre.scrollHeight;
        if (["done", "failed", "cancelled"].includes(job.state)) {
          clearInterval(timer);
          state.dockerJob = null;
          $("#btn-docker-cancel").hidden = true;
          if (job.state === "cancelled") {
            toast("拉取已取消", job.timeout_hit ? "超时终止" : "", "warn");
          } else if (job.state === "failed") {
            toast("拉取失败", job.error || "", "bad");
          } else if (job.changed) {
            toast("镜像已更新", `${(job.old_id || "").slice(7, 19)} → ${(job.new_id || "").slice(7, 19)}，旧镜像可清理`, "ok");
          } else {
            toast("镜像已是最新", ref, "");
          }
          loadDocker();
        }
      } catch (err) {
        clearInterval(timer);
        toast("拉取状态丢失", String(err), "bad");
      }
    }, 1000);
  }

  async function dockerPrune() {
    try {
      const preview = await api("/api/docker/prune", { method: "POST", body: "{}" });
      if (!preview.count) { toast("没有可清理的镜像", "当前 scope 内都可删除项为 0", ""); return; }
      const lines = preview.preview.map((p) => `· ${(p.tags || []).join(", ")}  ${fmtMb(p.size_mb)}`).join("\n");
      if (!confirm(`将删除 ${preview.count} 个镜像，回收约 ${fmtMb(preview.reclaimable_mb)}：\n\n${lines}`)) return;
      const res = await api("/api/docker/prune", { method: "POST", body: JSON.stringify({ confirm: true }) });
      const bad = (res.removed || []).filter((r) => !r.ok);
      toast(bad.length ? "部分镜像未删除" : "清理完成",
            `${(res.removed || []).length - bad.length} 个镜像，约 ${fmtMb(res.reclaimable_mb)}` +
            (bad.length ? `；${bad.length} 个失败` : ""),
            bad.length ? "warn" : "ok");
      loadDocker();
    } catch (err) {
      toast("清理失败", String(err), "bad");
    }
  }

  async function dockerRemoveImage(id) {
    if (!confirm("删除该镜像？（只影响 vLLM scope）")) return;
    try {
      const res = await api("/api/docker/images/remove", {
        method: "POST", body: JSON.stringify({ id, confirm: true }),
      });
      toast("已删除", (res.command || []).join(" "), "ok");
    } catch (err) {
      toast("删除失败", String(err), "bad");
    }
    loadDocker();
  }

  async function dockerRemoveContainer(name) {
    if (!confirm(`删除容器 ${name}？`)) return;
    try {
      const res = await api("/api/docker/containers/remove", {
        method: "POST", body: JSON.stringify({ name, confirm: true }),
      });
      toast("容器已删除", (res.command || []).join(" "), "ok");
    } catch (err) {
      toast("删除容器失败", String(err), "bad");
    }
    loadDocker();
  }

  async function dockerRestartWith(ref) {
    if (!confirm(`用镜像 ${ref} 重启 vLLM？\n会重新创建容器，冷启动需要几分钟。`)) return;
    const opts = collectOpts();
    const res = await post("/api/docker/restart-with", { ref, profile: opts.profile, ctx: opts.ctx });
    if (res && res.ok) toast("用新镜像重启中", ref, "ok");
    else if (res && res.error) toast("重启失败", res.error, "bad");
    loadDocker();
  }

  /* ── live link ──────────────────────────────────────────────────────── */
  let ws = null;
  let wsRetries = 0;

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const token = new URLSearchParams(location.search).get("token");
    ws = new WebSocket(`${proto}://${location.host}/ws${token ? `?token=${encodeURIComponent(token)}` : ""}`);
    ws.onopen = () => {
      state.connected = true;
      wsRetries = 0;
      if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
    };
    ws.onmessage = (event) => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
      if (data.type === "snapshot") {
        renderSnapshot(data);
      } else if (data.type === "request_done" || data.type === "request_enriched") {
        scheduleHistory();
      } else if (data.type === "docker_changed") {
        loadDocker();
      } else if (data.type === "docker_job" && data.job && ["done", "failed"].includes(data.job.state)) {
        loadDocker();
      } else if (data.type === "engine_state") {
        if (data.state === "healthy") toast("引擎就绪", data.engine || "", "ok");
        if (data.state === "failed") toast("引擎启动失败", data.detail || "", "bad");
      }
    };
    ws.onclose = () => {
      state.connected = false;
      wsRetries += 1;
      if (!state.pollTimer) {
        state.pollTimer = setInterval(async () => {
          try { renderSnapshot(await api("/api/state")); } catch { /* ignore */ }
        }, 2000);
      }
      setTimeout(connect, Math.min(8000, 500 * wsRetries));
    };
    ws.onerror = () => { try { ws.close(); } catch { /* ignore */ } };
  }

  /* ── boot ───────────────────────────────────────────────────────────── */
  async function boot() {
    const urlTheme = new URLSearchParams(location.search).get("theme");
    if (urlTheme === "dark" || urlTheme === "light") localStorage.setItem(THEME_KEY, urlTheme);
    const theme = localStorage.getItem(THEME_KEY) || "light";
    document.documentElement.dataset.theme = theme;
    $("#theme-toggle").textContent = theme === "dark" ? "◑" : "◐";

    $("#theme-toggle").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(next);
      drawSparklines();
    });

    for (const btn of document.querySelectorAll(".tab")) {
      btn.addEventListener("click", () => activateTab(btn.dataset.tab));
    }

    // token ledger date range (read-only aggregation, remembered across reloads)
    $("#tok-from").value = todayIso();
    $("#tok-to").value = todayIso();
    $("#tok-apply").addEventListener("click", applyTokenRange);
    $("#tok-reset").addEventListener("click", resetTokenRange);
    for (const id of ["#tok-from", "#tok-to"]) {
      $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") applyTokenRange(); });
    }
    const savedRange = localStorage.getItem(TOK_RANGE_KEY);
    if (savedRange) {
      try {
        const r = JSON.parse(savedRange);
        if (r && r.start && r.end) {
          $("#tok-from").value = r.start;
          $("#tok-to").value = r.end;
          applyTokenRange().catch(() => {});
        }
      } catch (err) { /* ignore a corrupt entry */ }
    }

    for (const btn of document.querySelectorAll("#engine-seg button")) {
      btn.addEventListener("click", () => {
        state.engine = btn.dataset.engine;
        state.userPickedEngine = true;
        selectEngineButton(state.engine);
        syncEngineDefaults();
        previewCommand();
      });
    }
    $("#profile").addEventListener("change", () => {
      state.profileTouched = true;
      renderEngineOptions();
      previewCommand();
    });
    $("#ctx").addEventListener("change", previewCommand);

    $("#btn-apply").addEventListener("click", applyEngine);
    $("#btn-stop").addEventListener("click", stopEngine);
    $("#btn-restart").addEventListener("click", restartEngine);
    $("#btn-probe").addEventListener("click", runProbe);
    $("#btn-dry").addEventListener("click", previewCommand);
    $("#btn-refresh").addEventListener("click", () => { state.page = 0; loadHistory(); });
    $("#btn-logs").addEventListener("click", loadLogs);
    $("#btn-docker-refresh").addEventListener("click", loadDocker);
    $("#btn-docker-pull").addEventListener("click", dockerPull);
    $("#btn-docker-prune").addEventListener("click", dockerPrune);
    $("#btn-docker-cancel").addEventListener("click", dockerCancel);
    $("#page-prev").addEventListener("click", () => { if (state.page > 0) { state.page -= 1; loadHistory(); } });
    $("#page-next").addEventListener("click", () => { state.page += 1; loadHistory(); });
    $("#filter-engine").addEventListener("change", () => { state.page = 0; loadHistory(); });
    $("#filter-status").addEventListener("change", () => { state.page = 0; loadHistory(); });
    $("#filter-search").addEventListener("input", () => {
      clearTimeout(window.__searchTimer);
      window.__searchTimer = setTimeout(() => { state.page = 0; loadHistory(); }, 350);
    });
    $("#drawer-close").addEventListener("click", closeDrawer);
    $("#scrim").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

    let resizeTimer = null;
    window.addEventListener("resize", () => {
      if (state.tab !== "overview") return;
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => { buildCharts(); drawSparklines(); }, 200);
    });

    ensureSeries();
    await loadEngines();

    const queryTab = new URLSearchParams(location.search).get("tab");
    const hashTab = location.hash.replace("#", "");
    const savedTab = localStorage.getItem(TAB_KEY);
    const initialTab = TABS.includes(queryTab) ? queryTab
      : TABS.includes(hashTab) ? hashTab
      : TABS.includes(savedTab) ? savedTab : "overview";
    activateTab(initialTab, { persist: false, hash: false });

    try { await loadSeries(3600); } catch { /* ignore */ }
    try { renderSnapshot(await api("/api/state")); } catch { /* ignore */ }
    previewCommand();
    connect();
    setInterval(() => { if (state.tab === "engine") loadLogs(); }, 15000);
    setInterval(() => { if (state.tab === "docker") loadDocker(); }, 20000);

    const deepLink = new URLSearchParams(location.search).get("open");
    if (deepLink) {
      activateTab("requests");
      openDetail(deepLink);
    }
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
