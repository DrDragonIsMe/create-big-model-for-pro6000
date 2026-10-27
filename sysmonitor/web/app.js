/* sysmonitor Web 前端：数据/语义/快捷键与 CLI 对齐 */
"use strict";

const SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏";
const LOAD_SPIN = "◐◓◑◒";
const ARR_IN = "↓⇣", ARR_OUT = "↑⇡";

const $ = (s, el = document) => el.querySelector(s);
const $id = (s) => document.getElementById(s);

/* ── 格式化（与 sysmon.py 同口径）── */
function fmtBytes(n) {
  const u = ["B", "K", "M", "G", "T"];
  n = Math.max(0, Number(n) || 0);
  for (const x of u) {
    if (n < 1024 || x === "T") return x === "B" ? `${n.toFixed(0)}${x}` : `${n.toFixed(1)}${x}`;
    n /= 1024;
  }
}
const fmtMB = (mb) => fmtBytes((Number(mb) || 0) * 1024 * 1024);
function fmtTokens(n) {
  if (n == null) return "--";
  n = Number(n);
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e4) return (n / 1e3).toFixed(1) + "K";
  return Math.round(n).toLocaleString();
}
function fmtDur(s) {
  if (s == null || s < 0) return "-";
  s = Math.floor(s);
  if (s < 60) return s + "s";
  if (s < 3600) return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
  if (s < 86400) return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}`;
  return `${Math.floor(s / 86400)}d${String(Math.floor((s % 86400) / 3600)).padStart(2, "0")}h`;
}
const sevClass = (p) => (p >= 85 ? "crit" : p >= 60 ? "warn" : "ok");
const TEMP_WARN = 70, TEMP_CRIT = 85, PRESSURE_WARN = 85, PRESSURE_CRIT = 95;
const LEVEL_RANK = {critical:0, warning:1, info:2, ok:3};
const tempClass = (t) => t >= TEMP_CRIT ? "temp-crit" : t >= TEMP_WARN ? "temp-warn" : "normal";
const pctOf = (a,b) => b ? 100 * (Number(a)||0) / Number(b) : 0;
const escHTML = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function healthAlerts(snap){
  const out=[],mem=snap.mem||{},disk=snap.disk||{};
  if(snap.error)out.push({label:"critical",title:"采集异常",detail:snap.error});
  if(!snap.gateway_connected)out.push({label:"critical",title:"网关不可达",detail:snap.gateway||"监控服务无法读取模型指标"});
  for(const g of snap.gpus||[]){
    if(Number(g.temp||0)>=TEMP_CRIT)out.push({label:"critical",title:`GPU ${g.index} 温度过高`,detail:`${g.temp}°C，告警阈值 ${TEMP_CRIT}°C`});
    else if(Number(g.temp||0)>=TEMP_WARN)out.push({label:"warning",title:`GPU ${g.index} 温度偏高`,detail:`${g.temp}°C，注意阈值 ${TEMP_WARN}°C`});
    if(pctOf(g.power,g.power_limit)>=PRESSURE_CRIT)out.push({label:"warning",title:`GPU ${g.index} 接近功率上限`,detail:`${(g.power||0).toFixed(0)} / ${(g.power_limit||0).toFixed(0)} W`});
    if(Number(g.mem_util||0)>=95)out.push({label:"warning",title:`GPU ${g.index} 显存带宽饱和`,detail:`${(g.mem_util||0).toFixed(0)}%`});
  }
  if((snap.agg?.queued||0)>0)out.push({label:"warning",title:"推理请求排队",detail:`${snap.agg.queued} 个请求等待槽位`});
  for(const m of snap.models||[])if(m.status==="loading")out.push({label:"info",title:`${m.name} 正在加载`,detail:"模型冷启动通常需要 1–5 分钟"});
  if((mem.swap_pct||0)>=PRESSURE_WARN)out.push({label:"warning",title:"交换分区偏高",detail:`${mem.swap_pct}% · ${fmtBytes(mem.swap_used)} / ${fmtBytes(mem.swap_total)}`});
  if((disk.pct||0)>=PRESSURE_WARN)out.push({label:"warning",title:"系统盘空间不足",detail:`${disk.pct}% · 剩余 ${fmtBytes(disk.free)}`});
  if((snap.cpu?.total||0)>=PRESSURE_WARN)out.push({label:"warning",title:"CPU 压力偏高",detail:`${snap.cpu.total}% · load ${snap.cpu.load?.[0]??"-"}`});
  return out.sort((a,b)=>LEVEL_RANK[a.label]-LEVEL_RANK[b.label]);
}
function renderAlerts(snap){
  const alerts=healthAlerts(snap),body=$id("alerts");
  const count=(l)=>alerts.filter(a=>a.label===l).length;
  const c=count("critical"),w=count("warning"),i=count("info"),status=c?"critical":w?"warning":"ok";
  const cards=alerts.length?alerts.map(a=>`<article class="alert ${escHTML(a.label)}"><span class="alert-icon">${a.label==="critical"?"▲":a.label==="warning"?"△":"●"}</span><div class="alert-copy"><b>${escHTML(a.title)}</b><p>${escHTML(a.detail)}</p></div><span class="alert-level">${a.label==="critical"?"严重":a.label==="warning"?"预警":"信息"}</span></article>`).join(""):`<article class="alert ok"><span class="alert-icon">✓</span><div class="alert-copy"><b>系统健康</b><p>温度、网关、队列、交换分区与磁盘压力均低于预警阈值。</p></div><span class="alert-level">正常</span></article>`;
  body.className=`alert-panel ${status}`;
  body.innerHTML=`<div class="alert-head"><div><h2>${c?"严重告警":w?"运行预警":"系统健康"}</h2><p>阈值：温度 ${TEMP_WARN}°C / ${TEMP_CRIT}°C · 系统压力 ${PRESSURE_WARN}%</p></div><div class="alert-counts"><b class="critical">${c}</b><span>严重</span><b class="warning">${w}</b><span>预警</span><b class="info">${i}</b><span>信息</span></div></div><div class="alert-list">${cards}</div>`;
}

/* ── 精选主题底色（官方调色板的真值；未知主题用服务端换算色）── */
const CURATED = {
  "catppuccin-latte": {
    "--bg": "#f4f5f8", "--bg2": "#e9ebf1", "--card": "#ffffff", "--surface": "#e2e6ed",
    "--border": "#ccd3df", "--text": "#26364c", "--subtext": "#526177", "--dim": "#718096",
    "--accent": "#276fbf", "--ok": "#147a52", "--warn": "#9a5a08", "--crit": "#c43855",
    "--mem": "#087f8c", "--model": "#7652b5", "--disk": "#a85420",
    "--header-fg": "#ffffff", "--header-bg": "#276fbf",
    "--gpu0": "#7652b5", "--gpu1": "#276fbf", "--gpu2": "#087f8c", "--gpu3": "#147a52",
  },
  "catppuccin-mocha": {
    "--bg": "#1e1e2e", "--bg2": "#181825", "--card": "#24243a", "--surface": "#313244",
    "--border": "#45475a", "--text": "#cdd6f4", "--subtext": "#a6adc8", "--dim": "#6c7086",
    "--accent": "#89b4fa", "--ok": "#a6e3a1", "--warn": "#f9e2af", "--crit": "#f38ba8",
    "--mem": "#94e2d5", "--model": "#cba6f7", "--disk": "#fab387",
    "--header-fg": "#11111b", "--header-bg": "#cba6f7",
    "--gpu0": "#cba6f7", "--gpu1": "#89b4fa", "--gpu2": "#94e2d5", "--gpu3": "#a6e3a1",
  },
  "tokyo-night": {
    "--bg": "#1a1b26", "--bg2": "#16161e", "--card": "#1f2335", "--surface": "#292e42",
    "--border": "#414868", "--text": "#a9b1d6", "--subtext": "#9aa5ce", "--dim": "#565f89",
    "--accent": "#7dcfff", "--ok": "#9ece6a", "--warn": "#e0af68", "--crit": "#f7768e",
    "--mem": "#73daca", "--model": "#bb9af7", "--disk": "#ff9e64",
    "--header-fg": "#1a1b26", "--header-bg": "#7aa2f7",
    "--gpu0": "#7aa2f7", "--gpu1": "#bb9af7", "--gpu2": "#7dcfff", "--gpu3": "#73daca",
  },
  dracula: {
    "--bg": "#282a36", "--bg2": "#21222c", "--card": "#343746", "--surface": "#44475a",
    "--border": "#6272a4", "--text": "#f8f8f2", "--subtext": "#d6d6d6", "--dim": "#7c83a4",
    "--accent": "#8be9fd", "--ok": "#50fa7b", "--warn": "#f1fa8c", "--crit": "#ff5555",
    "--mem": "#8be9fd", "--model": "#bd93f9", "--disk": "#ffb86c",
    "--header-fg": "#282a36", "--header-bg": "#bd93f9",
    "--gpu0": "#bd93f9", "--gpu1": "#8be9fd", "--gpu2": "#50fa7b", "--gpu3": "#ff79c6",
  },
  nord: {
    "--bg": "#2e3440", "--bg2": "#242933", "--card": "#3b4252", "--surface": "#434c5e",
    "--border": "#4c566a", "--text": "#d8dee9", "--subtext": "#b9c2cf", "--dim": "#7b88a1",
    "--accent": "#88c0d0", "--ok": "#a3be8c", "--warn": "#ebcb8b", "--crit": "#bf616a",
    "--mem": "#8fbcbb", "--model": "#b48ead", "--disk": "#d08770",
    "--header-fg": "#2e3440", "--header-bg": "#88c0d0",
    "--gpu0": "#88c0d0", "--gpu1": "#b48ead", "--gpu2": "#a3be8c", "--gpu3": "#8fbcbb",
  },
  "gruvbox-dark": {
    "--bg": "#282828", "--bg2": "#1d2021", "--card": "#32302f", "--surface": "#3c3836",
    "--border": "#504945", "--text": "#ebdbb2", "--subtext": "#d5c4a1", "--dim": "#928374",
    "--accent": "#83a598", "--ok": "#b8bb26", "--warn": "#fabd2f", "--crit": "#fb4934",
    "--mem": "#8ec07c", "--model": "#d3869b", "--disk": "#fe8019",
    "--header-fg": "#282828", "--header-bg": "#83a598",
    "--gpu0": "#83a598", "--gpu1": "#d3869b", "--gpu2": "#b8bb26", "--gpu3": "#8ec07c",
  },
  "one-dark": {
    "--bg": "#282c34", "--bg2": "#21252b", "--card": "#2c313a", "--surface": "#3e4451",
    "--border": "#4b5263", "--text": "#abb2bf", "--subtext": "#9da5b4", "--dim": "#6b717e",
    "--accent": "#61afef", "--ok": "#98c379", "--warn": "#e5c07b", "--crit": "#e06c75",
    "--mem": "#56b6c2", "--model": "#c678dd", "--disk": "#d19a66",
    "--header-fg": "#282c34", "--header-bg": "#61afef",
    "--gpu0": "#61afef", "--gpu1": "#c678dd", "--gpu2": "#98c379", "--gpu3": "#56b6c2",
  },
  "solarized-dark": {
    "--bg": "#002b36", "--bg2": "#00212b", "--card": "#073642", "--surface": "#094352",
    "--border": "#0f4d5c", "--text": "#93a1a1", "--subtext": "#839496", "--dim": "#586e75",
    "--accent": "#268bd2", "--ok": "#859900", "--warn": "#b58900", "--crit": "#dc322f",
    "--mem": "#2aa198", "--model": "#6c71c4", "--disk": "#cb4b16",
    "--header-fg": "#002b36", "--header-bg": "#268bd2",
    "--gpu0": "#268bd2", "--gpu1": "#6c71c4", "--gpu2": "#859900", "--gpu3": "#2aa198",
  },
};
const ROLE2VAR = { accent: "--accent", ok: "--ok", warn: "--warn", crit: "--crit",
                   mem: "--mem", model: "--model", disk: "--disk", dim: "--dim", border: "--border" };

const THEME_VARS = new Set([
  ...Object.values(CURATED).flatMap(vars => Object.keys(vars)),
  ...Object.values(ROLE2VAR),
]);
async function applyTheme(name) {
  let vars = CURATED[name];
  if (!vars) {
    try {
      const t = await fetch(`/api/theme?name=${encodeURIComponent(name)}`).then((r) => r.json());
      vars = {};
      for (const [role, hex] of Object.entries(t.colors || {})) {
        const v = ROLE2VAR[role];
        if (!v) continue;
        if (Array.isArray(hex)) { vars["--header-fg"] = hex[0]; vars["--header-bg"] = hex[1]; }
        else vars[v] = hex;
      }
      if (Array.isArray(t.gpu)) t.gpu.slice(0, 4).forEach((h, i) => (vars[`--gpu${i}`] = h));
    } catch (e) { return; }
  }
  for (const key of THEME_VARS) document.documentElement.style.removeProperty(key);
  for (const [k, v] of Object.entries(vars)) document.documentElement.style.setProperty(k, v);
  document.documentElement.dataset.theme = name === "catppuccin-latte" ? "light" : "dark";
}

/* ── 小组件工厂 ── */
function makeGauge(label, wide) {
  const el = document.createElement("div");
  el.className = "gauge" + (wide ? " wide-label" : "");
  el.innerHTML = `<span class="g-label">${label}</span><span class="g-val mono">--</span>
    <div class="g-track"><div class="g-fill"></div></div><span class="g-detail"></span>`;
  el._val = $(".g-val", el); el._fill = $(".g-fill", el); el._detail = $(".g-detail", el);
  return el;
}
function setGauge(g, pct, kind, detail) {
  pct = Math.max(0, Math.min(100, Number(pct) || 0));
  const cls = kind === "mem" ? "mem" : sevClass(pct);
  g._val.textContent = pct.toFixed(1) + "%";
  g._val.className = "g-val mono " + cls;
  g._fill.style.width = pct.toFixed(1) + "%";
  g._fill.className = "g-fill " + cls + (pct > 0.5 ? " live" : "");
  g._detail.textContent = detail || "";
  g._detail.style.display = detail ? "" : "none";
}

/* ── GPU 卡 ── */
const gpuCache = new Map();
function renderGpus(snap) {
  const grid = $id("gpuGrid");
  const seen = new Set();
  for (const g of snap.gpus) {
    seen.add(g.index);
    let card = gpuCache.get(g.index);
    if (!card) {
      card = { el: document.createElement("section"), gauges: {} };
      card.el.className = "card gpu-card";
      card.el.innerHTML = `<div><span class="gpu-name"></span></div><div class="badges"></div>`;
      card.name = $(".gpu-name", card.el);
      for (const lab of ["算力", "显存", "带宽"]) {
        const gg = makeGauge(lab);
        card.gauges[lab] = gg;
        card.el.appendChild(gg);
      }
      gpuCache.set(g.index, card);
      grid.appendChild(card.el);
    }
    const tcls = tempClass(g.temp), powerPct = pctOf(g.power, g.power_limit);
    card.el.className = `card gpu-card ${tcls}`;
    card.el.style.setProperty("--gc", `var(--gpu${g.index % 4})`);
    card.name.textContent = `GPU ${g.index} · ${g.name.replace(/^NVIDIA /, "")}`;
    const memPct = g.mem_total ? g.mem_used / g.mem_total * 100 : 0;
    const free = (g.mem_total || 0) - g.mem_used;
    setGauge(card.gauges["算力"], g.util, "sev");
    setGauge(card.gauges["显存"], memPct, "mem",
      `${fmtMB(g.mem_used)}/${fmtMB(g.mem_total)} 空闲${fmtMB(free)}`);
    setGauge(card.gauges["带宽"], g.mem_util || 0, "sev");
    $(".badges", card.el).innerHTML =
      `<span class="badge temp ${tcls}">${g.temp ?? "-"}°C<small>${tcls === "temp-crit" ? " 严重" : tcls === "temp-warn" ? " 偏高" : " 正常"}</small></span>` +
      (g.power_limit ? `<span class="badge ${powerPct >= PRESSURE_CRIT ? "temp-warn" : ""}">${g.power.toFixed(0)} / ${g.power_limit.toFixed(0)} W · ${powerPct.toFixed(0)}%</span>` : "") +
      (g.fan != null ? `<span class="badge">风扇 ${g.fan.toFixed(0)}%</span>` : "");
  }
  for (const [idx, c] of gpuCache) if (!seen.has(idx)) { c.el.remove(); gpuCache.delete(idx); }
}

/* ── LLM ── */
const modelCache = new Map();
let animFrame = 0;
function renderLlm(snap) {
  const body = $id("llmBody");
  if (!snap.gateway_connected) {
    body.innerHTML = `<div class="banner">⛔ 网关离线/不可达（${snap.gateway}）——token 指标暂不可用，其余面板照常</div>`;
    for (const [, c] of modelCache) c.remove();
    modelCache.clear();
    return;
  }
  if (!snap.models.length) {
    body.innerHTML = `<div class="dim" style="padding:8px 2px">已连接，暂无已加载模型 —— 用客户端发起请求或经网关 /v1/gateway/load 加载</div>`;
    for (const [, c] of modelCache) c.remove();
    modelCache.clear();
    return;
  }
  const seen = new Set();
  for (const m of snap.models) {
    seen.add(m.name);
    let row = modelCache.get(m.name);
    if (!row) {
      row = document.createElement("div");
      row.className = "model";
      row.innerHTML = `
        <div class="m-l1">
          <span class="m-name"></span><span class="m-meta"></span>
          <span class="phase"></span>
        </div>
        <div class="m-l2">
          <span class="rate t"><b>--</b><small>t/s 生成速率</small></span>
          <span class="rate p"><b>--</b><small>p/s Prompt速率</small></span>
          <span class="slots"></span>
          <span class="m-cum dim"></span>
        </div>`;
      modelCache.set(m.name, row);
      body.appendChild(row);
    }
    $(".m-name", row).textContent = m.name;
    $(".m-meta", row).textContent =
      `${m.engine === "vllm" ? "vLLM" : "llama"} · GPU ${m.gpus} · ${m.status}`;
    const t = Number(m.rate) || 0, p = Number(m.prompt_rate) || 0;
    const tEl = $(".rate.t b", row), pEl = $(".rate.p b", row);
    tEl.textContent = t > 0 ? t.toFixed(1) : "--";
    $(".rate.t", row).classList.toggle("zero", !(t > 0));
    pEl.textContent = p > 0 ? Math.round(p).toLocaleString() : "--";
    $(".rate.p", row).classList.toggle("zero", !(p > 0));

    const ph = $(".phase", row);
    if (m.prefill_active) {
      ph.className = "phase prompt";
      // vLLM 只报速率、没有单请求进度：进度条留空，免得恒显 0%
      const prog = m.prefill_progress == null ? null : Number(m.prefill_progress);
      const bar = prog == null ? "" :
        `<span class="g-track m-pbar"><span class="g-fill warn live" style="width:${(prog * 100).toFixed(1)}%"></span></span>
         <b>${(prog * 100).toFixed(0)}%</b>`;
      const el = m.prefill_elapsed ? ` · ${Number(m.prefill_elapsed).toFixed(0)}s` : "";
      const tok = m.prefill_tokens == null ? "" :
        `<small>${fmtTokens(m.prefill_tokens)} tok${el}</small>`;
      ph.innerHTML = `<span class="spin" data-spin="braille">⠋</span> Prompt ${bar}${tok}`;
    } else if (m.status === "loading") {
      ph.className = "phase load";
      ph.innerHTML = `<span class="spin" data-spin="load">◐</span> 模型加载中…`;
    } else if (t > 0 || (m.active > 0)) {
      ph.className = "phase gen";
      ph.innerHTML = `<span class="dot busy" style="width:10px;height:10px"></span> 生成中`;
    } else {
      ph.className = "phase idle";
      ph.innerHTML = `空闲 <b>${fmtDur(m.idle)}</b>`;
    }

    const sl = $(".slots", row);
    const slots = m.slots, act = m.active;
    if (slots) {
      const q = Math.max(0, act - slots);
      const lit = Math.min(act, slots, 8);
      let bars = "";
      for (let i = 0; i < Math.min(slots, 8); i++)
        bars += `<i class="${i < lit ? "on " + (q > 0 ? "q" : "act") : ""}"></i>`;
      sl.innerHTML = `ACT ${bars} <b>${act}/${slots}</b>` +
        (q > 0 ? ` <span class="qbadge">⚠ +${q} 排队</span>` : "");
    } else {
      sl.innerHTML = `ACT <b>${act}</b>${act > 0 ? ` <span class="ok-txt">进行中</span>` : ""}`;
    }
    $(".m-cum", row).textContent =
      `Σprompt ${fmtTokens(m.prompt_total)} · Σgen ${fmtTokens(m.generation_total)} · 空闲 ${fmtDur(m.idle)}`;
  }
  for (const [k, el] of modelCache) if (!seen.has(k)) { el.remove(); modelCache.delete(k); }
}

/* ── CPU / 系统 / 网络 / 磁盘IO ── */
let cpuGauge = null, coreEls = [];
function renderCpu(snap) {
  if (!cpuGauge) { cpuGauge = makeGauge("总体"); $id("cpuTotal").replaceWith(cpuGauge); cpuGauge.id = "cpuTotal"; }
  const c = snap.cpu;
  setGauge(cpuGauge, c.total, "sev");
  $id("cpuMeta").textContent =
    `${c.count} 核 · ${c.freq ? (c.freq / 1000).toFixed(2) + "GHz · " : ""}load ${c.load[0]} ${c.load[1]} ${c.load[2]}`;
  const box = $id("cores");
  if (coreEls.length !== c.per_core.length) {
    box.innerHTML = ""; coreEls = [];
    for (let i = 0; i < c.per_core.length; i++) {
      const b = document.createElement("i");
      b.title = `#${i}`;
      box.appendChild(b); coreEls.push(b);
    }
  }
  c.per_core.forEach((v, i) => {
    const b = coreEls[i];
    b.style.height = Math.max(4, v) + "%";
    b.className = v >= 85 ? "crit" : v >= 60 ? "warn" : "";
    b.title = `#${i}: ${v.toFixed(1)}%`;
  });
  $id("coresLabel").textContent = `每核 ${c.per_core.length} 根竖条 = 当前占用（▰▱ 仪表的纵版对应）`;
}

let sysGauges = null;
function renderSys(snap) {
  if (!sysGauges) {
    const body = $id("sysBody");
    body.innerHTML = "";
    sysGauges = {};
    for (const lab of ["内存", "交换", "磁盘", "盘读写"]) {
      const g = makeGauge(lab); body.appendChild(g); sysGauges[lab] = g;
    }
  }
  const m = snap.mem, d = snap.disk;
  setGauge(sysGauges["内存"], m.pct, "mem",
    `${fmtBytes(m.used)}/${fmtBytes(m.total)} 可用 ${fmtBytes(m.avail)}`);
  setGauge(sysGauges["交换"], m.swap_pct, "sev", `${fmtBytes(m.swap_used)}/${fmtBytes(m.swap_total)}`);
  setGauge(sysGauges["磁盘"], d.pct, "sev",
    `${fmtBytes(d.used)}/${fmtBytes(d.total)} 剩余 ${fmtBytes(d.free)}`);
  setGauge(sysGauges["盘读写"], Math.min(100, (d.read_bps + d.write_bps) / (500e6) * 100), "sev",
    `读 ${fmtBytes(d.read_bps)}/s · 写 ${fmtBytes(d.write_bps)}/s（满格≈500MB/s）`);
}

const nicCache = new Map();
function renderNet(snap) {
  const body = $id("netBody");
  if (!snap.net.length) { body.innerHTML = `<div class="dim small" style="padding:6px 0">无活动网卡（已排除 lo/veth/docker）</div>`; return; }
  const rel = Math.max(1e5, ...snap.net.map((n) => Math.max(n.rx, n.tx)));
  const seen = new Set();
  for (const n of snap.net.slice(0, 4)) {
    seen.add(n.name);
    let row = nicCache.get(n.name);
    if (!row) {
      row = document.createElement("div"); row.className = "nic";
      row.innerHTML =
        `<div class="nic-name"></div>` +
        `<div class="nic-dir"><span class="arr down">↓</span><span class="g-track"><span class="g-fill"></span></span><span class="nic-rate"></span></div>` +
        `<div class="nic-dir"><span class="arr up">↑</span><span class="g-track"><span class="g-fill"></span></span><span class="nic-rate"></span></div>`;
      nicCache.set(n.name, row); body.appendChild(row);
    }
    const scale = n.link_mbps ? (n.link_mbps * 1e6) / 8 : rel;
    $(".nic-name", row).innerHTML = `${n.name}<small>${n.ip || ""}${n.link_mbps ? ` · ${n.link_mbps >= 1000 ? n.link_mbps / 1000 + "G" : n.link_mbps + "M"}` : ""}</small>`;
    const dirs = row.querySelectorAll(".nic-dir");
    [[n.rx, 0], [n.tx, 1]].forEach(([rate, i]) => {
      const pct = Math.min(100, rate / scale * 100);
      const d = dirs[i];
      $(".arr", d).classList.toggle("flow", rate > 0);
      const fill = $(".g-fill", d);
      fill.style.width = pct.toFixed(2) + "%";
      fill.className = "g-fill " + sevClass(pct) + (pct > 0.5 ? " live" : "");
      $(".nic-rate", d).textContent = fmtBytes(rate) + "/s";
    });
  }
  for (const [k, el] of nicCache) if (!seen.has(k)) { el.remove(); nicCache.delete(k); }
  let tot = $(".net-total", body);
  if (!tot) { tot = document.createElement("div"); tot.className = "net-total dim"; body.appendChild(tot); }
  tot.textContent = `Σ累计（开机以来）↓ ${fmtBytes(snap.net.reduce((s, n) => s + n.rx_total, 0))} · ↑ ${fmtBytes(snap.net.reduce((s, n) => s + n.tx_total, 0))}`;
}

const dioCache = new Map();
function renderDio(snap) {
  const body = $id("dioBody");
  const seen = new Set();
  for (const d of snap.disks.slice(0, 4)) {
    seen.add(d.name);
    let g = dioCache.get(d.name);
    if (!g) { g = makeGauge(d.name, true); dioCache.set(d.name, g); body.appendChild(g); }
    setGauge(g, d.util, "sev",
      `读 ${fmtBytes(d.r_bps)}/s · 写 ${fmtBytes(d.w_bps)}/s · ${Math.round(d.r_iops + d.w_iops).toLocaleString()} io/s`);
  }
  for (const [k, el] of dioCache) if (!seen.has(k)) { el.remove(); dioCache.delete(k); }
  if (!snap.disks.length) body.innerHTML = `<div class="dim small">无物理盘数据</div>`;
}

/* ── 进程表 ── */
let procSort = "vram";
const procCache = new Map();
function renderProcs(snap) {
  const rows = [...snap.procs];
  const key = procSort === "vram" ? (r) => r.vram || -1 : procSort === "cpu" ? (r) => r.cpu : (r) => r.pmem;
  rows.sort((a, b) => key(b) - key(a));
  const shown = rows.slice(0, 24);          // 控制页面长度；全量仍在服务端三键并集里
  const gpuTotal = {};
  for (const g of snap.gpus) gpuTotal[g.index] = g.mem_total;
  const body = $id("procBody");
  const seen = new Set();
  shown.forEach((r, idx) => {
    seen.add(r.pid);
    let tr = procCache.get(r.pid);
    if (!tr) {
      tr = document.createElement("tr");
      tr.innerHTML = "<td></td>".repeat(9);
      procCache.set(r.pid, tr);
    }
    const td = tr.children;
    const vpct = r.vram && gpuTotal[r.gpu] ? r.vram / gpuTotal[r.gpu] * 100 : 0;
    td[0].textContent = r.gpu != null ? "GPU" + r.gpu : "·";
    td[0].className = r.gpu != null ? "mem-txt" : "dim";
    td[1].textContent = r.pid; td[1].className = "num dim";
    td[2].textContent = r.user; td[2].className = "dim";
    td[3].textContent = r.cpu.toFixed(1);
    td[3].className = "num " + (r.cpu >= 85 ? "crit" : r.cpu >= 60 ? "warn-txt" : "dim");
    td[4].textContent = r.pmem.toFixed(1); td[4].className = "num " + (r.pmem >= 10 ? "warn-txt" : "dim");
    td[5].textContent = fmtBytes(r.rss); td[5].className = "num dim";
    td[6].textContent = r.vram ? fmtMB(r.vram) : "·";
    td[6].className = "num " + (r.vram ? sevClass(vpct) + "-txt" : "dim");
    td[7].innerHTML = `<span class="pbar"><i class="${sevClass(vpct)}"></i></span>`;
    const bar = $("i", td[7]);
    bar.style.width = vpct.toFixed(1) + "%";
    bar.style.background = `var(--${vpct >= 85 ? "crit" : vpct >= 60 ? "warn" : "ok"})`;
    td[8].innerHTML = r.model ? `<span class="mtag">[${r.model}]</span> ` : "";
    td[8].appendChild(document.createTextNode(r.cmd));
    td[8].className = "cmd";
    td[8].title = r.cmd;
    body.appendChild(tr);           // 重排顺序（复用节点保留过渡）
  });
  for (const [pid, tr] of procCache) if (!seen.has(pid)) { tr.remove(); procCache.delete(pid); }
  $id("procMeta").textContent =
    `共 ${snap.procs_total} 进程（服务端取三排序键并集 ${rows.length} 条，显示前 ${shown.length}）· ${snap.procs.filter((r) => r.vram).length} 个占显存`;
  document.querySelectorAll(".sortbtns button")
    .forEach((b) => b.classList.toggle("on", b.dataset.sort === procSort));
}

/* ── 顶栏聚合/时钟 ── */
function renderTop(snap) {
  $id("host").textContent = snap.host;
  const a = snap.agg;
  $id("agg").innerHTML =
    `<span class="big ok-txt">Σ ${a.t.toFixed(1)} t/s</span>` +
    (a.p > 0 ? `<span class="big warn-txt">Σ ${Math.round(a.p).toLocaleString()} p/s</span>` : "") +
    `<span>请求 <b>${a.req}</b></span>` +
    (a.queued ? `<span class="qbadge">▲${a.queued} 排队</span>` : "");
  const dot = $("#gw .dot");
  dot.classList.toggle("off", !snap.gateway_connected);
  dot.classList.toggle("busy", snap.gateway_connected && (a.t > 0 || a.p > 0));
  if (snap.error) $id("errInfo").textContent = "采集错误: " + snap.error;
  else $id("errInfo").textContent = "";
  const age = Math.max(0, Date.now() / 1000 - (snap.ts || 0));
  $id("refreshInfo").textContent =
    paused ? "⏸ 已暂停（画面冻结，后台仍在采样）" : `数据 ${age.toFixed(1)}s 前 · ${(intervalMs / 1000).toFixed(1)}s 自动刷新`;
}

/* ── 主循环 ── */
let paused = false, timer = null, intervalMs = 1000;
async function tick() {
  try {
    const r = await fetch("/api/snapshot");
    const snap = await r.json();
    if (!snap || !snap.gpus) throw new Error("empty");
    lastSnap = snap;
    if (!paused) paint(snap);
    document.body.classList.remove("conn-err");
  } catch (e) {
    $id("errInfo").textContent = "与采样服务连接中断：" + e.message;
  }
}
let lastSnap = null;
function paint(snap) {
  renderTop(snap); renderAlerts(snap); renderGpus(snap); renderLlm(snap);
  renderCpu(snap); renderSys(snap); renderNet(snap); renderDio(snap); renderProcs(snap);
}
function setPaused(v) {
  paused = v;
  document.body.classList.toggle("paused", v);
  $id("pauseBtn").classList.toggle("paused", v);
  $id("pauseBtn").textContent = v ? "▶ 继续" : "⏸ 暂停";
  if (v && lastSnap) renderTop(lastSnap);
}
function restartTimer() { clearInterval(timer); timer = setInterval(tick, intervalMs); tick(); }

/* 动画帧驱动器（spinner/箭头流动，对应 CLI _FRAME；暂停时冻结）*/
setInterval(() => {
  if (paused) return;
  animFrame++;
  document.querySelectorAll("[data-spin]").forEach((el) => {
    const seq = el.dataset.spin === "load" ? LOAD_SPIN : SPINNER;
    el.textContent = seq[animFrame % seq.length];
  });
  document.querySelectorAll(".arr.flow").forEach((el) => {
    const seq = el.classList.contains("down") ? ARR_IN : ARR_OUT;
    el.textContent = seq[Math.floor(animFrame / 3) % 2];
  });
}, 150);

/* 墙钟 */
setInterval(() => {
  const d = new Date();
  const up = lastSnap ? fmtDur(lastSnap.uptime + (Date.now() / 1000 - lastSnap.ts)) : "";
  $id("clock").textContent = `${d.toLocaleTimeString("zh-CN", { hour12: false })} · 运行 ${up}`;
}, 1000);

/* ── 启动 ── */
(async function boot() {
  try {
    const t = await fetch("/api/themes").then((r) => r.json());
    const sel = $id("themeSel"), appearanceSel = $id("appearanceSel");
    const media = matchMedia("(prefers-color-scheme: dark)");
    let appearance = localStorage.getItem("sysmonAppearance") || "auto";
    let darkTheme = localStorage.getItem("sysmonTheme") || t.default;
    if (!t.themes.includes(darkTheme)) darkTheme = t.default;
    if (!["auto", "light", "dark"].includes(appearance)) appearance = "auto";
    const urlTheme = new URLSearchParams(location.search).get("theme");
    if (urlTheme === "light" || urlTheme === "catppuccin-latte") appearance = "light";
    else if (urlTheme === "dark") appearance = "dark";
    else if (t.themes.includes(urlTheme)) { darkTheme = urlTheme; appearance = "dark"; }
    sel.innerHTML = t.themes.map(n => `<option>${n}</option>`).join("");
    sel.value = darkTheme;
    const resolvedTheme = () => appearance === "light" || (appearance === "auto" && !media.matches)
      ? "catppuccin-latte" : darkTheme;
    const paintTheme = () => {
      appearanceSel.value = appearance;
      sel.value = darkTheme;
      sel.disabled = resolvedTheme() === "catppuccin-latte";
      sel.title = sel.disabled ? "浅色外观使用 Catppuccin Latte；切换到深色后可选择配色" : "深色配色主题";
      applyTheme(resolvedTheme());
    };
    appearanceSel.onchange = () => {
      appearance = appearanceSel.value;
      localStorage.setItem("sysmonAppearance", appearance);
      paintTheme();
    };
    sel.onchange = () => {
      darkTheme = sel.value;
      localStorage.setItem("sysmonTheme", darkTheme);
      paintTheme();
    };
    media.addEventListener("change", () => { if (appearance === "auto") paintTheme(); });
    document.addEventListener("keydown", (e) => {
      if ((e.key !== "t" && e.key !== "T") || e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
      const sequence = ["catppuccin-latte", ...t.themes];
      const next = sequence[(sequence.indexOf(resolvedTheme()) + 1) % sequence.length];
      appearance = next === "catppuccin-latte" ? "light" : "dark";
      if (next !== "catppuccin-latte") darkTheme = next;
      localStorage.setItem("sysmonAppearance", appearance);
      localStorage.setItem("sysmonTheme", darkTheme);
      paintTheme();
    });
    paintTheme();
  } catch (e) { /* 主题列表失败不阻塞数据 */ }

  $id("pauseBtn").onclick = () => setPaused(!paused);
  document.querySelectorAll(".sortbtns button").forEach((b) =>
    (b.onclick = () => { procSort = b.dataset.sort; if (lastSnap) renderProcs(lastSnap); }));
  $id("intSel").onchange = (e) => { intervalMs = +e.target.value; restartTimer(); };

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); setPaused(!paused); }
    else if (e.key === "p" || e.key === "P") {
      procSort = { vram: "cpu", cpu: "pmem", pmem: "vram" }[procSort];
      if (lastSnap) renderProcs(lastSnap);
    }
  });

  restartTimer();
})();
