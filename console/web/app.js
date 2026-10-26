/* AI 集群控制台前端：浏览器直连每台同构节点的 agent:9001；DGX 异构机借最近一个
   agent 的 /api/relay 只读代理。无构建、无外部依赖——和这套蓝本一样的“拿到就能跑”。 */
"use strict";
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

const DEFAULT_NODES = [
  {id:"home-pro6000-dual", label:"家庭双 Pro6000", kind:"stack", url:"http://100.95.107.118:9001"},
  {id:"workstation-4090-dual", label:"4090 工作站", kind:"stack", url:"http://192.168.100.111:9001"},
  {id:"tju-pro6000-single", label:"天大 Pro6000", kind:"stack", url:"http://100.119.153.75:9001"},
  {id:"dgx-spark-pair", label:"DGX Spark 双机", kind:"spark", relay:"dgx-spark-pair"},
];
let nodes = JSON.parse(localStorage.getItem("ai-nodes") || "null") || DEFAULT_NODES;
let token = localStorage.getItem("ai-token") || "";
let cur = null;         // 当前查看节点 {node, snap, catalog, modes}
let tab = "ov";
let timers = [];
const st = {};          // id -> 最新 snapshot / 状态

const fmtK = n => n >= 1048576 ? (n/1048576).toFixed(2)+"M" : n >= 1000 ? Math.round(n/1000)+"K" : String(n);

async function api(node, path, opts={}) {
  const headers = Object.assign({"Content-Type":"application/json"}, opts.headers||{});
  if (token) headers["Authorization"] = "Bearer " + token;
  const r = await fetch(node.url + path, Object.assign({headers, cache:"no-store"}, opts));
  const j = await r.json().catch(()=>({__text:"非 JSON 响应"}));
  if (!r.ok) throw Object.assign(new Error(j?.error?.message || j?.error || ("HTTP "+r.status)), {body:j, status:r.status});
  return j;
}
function relayNode(){ return nodes.find(n => n.kind==="stack" && st[n.id] && !st[n.id].__err) || nodes.find(n=>n.kind==="stack"); }
async function sparkApi(path){
  const via = relayNode();
  const n = nodes.find(x => x.id==="dgx-spark-pair");
  return api(via, `/api/relay?node=${encodeURIComponent(n.relay)}&path=${encodeURIComponent(path)}`);
}

function toast(msg, ms=2600){ const t=$("#toast"); t.textContent=msg; t.classList.add("show"); clearTimeout(t._h); t._h=setTimeout(()=>t.classList.remove("show"), ms); }
function modal(title, body, note, onOk, okLabel="执行"){
  $("#mTitle").textContent = title; $("#mBody").textContent = body; $("#mNote").textContent = note||"";
  $("#mOk").textContent = okLabel || "执行"; $("#modal").classList.remove("hidden");
  $("#mOk").onclick = async () => { $("#modal").classList.add("hidden"); try { await onOk(); } catch(e){ toast("✗ "+e.message); } };
}
$("#mClose").onclick = () => $("#modal").classList.add("hidden");

/* ---------------- 总览卡片 ---------------- */
function gpuBars(gs){
  // 两行条：上=显存占用（蓝），下=计算利用率（青）——sysmon 的 mem_used/util 是两个维度，叠一条会互相掩盖
  return (gs||[]).map(g=>{
    const memPct = g.mem_total ? (100*g.mem_used/g.mem_total) : 0;
    return `<div class="gpubar"><div class="glabel"><span>GPU${g.index} ${esc((g.name||"").replace(/^NVIDIA /,""))}</span>
      <span>${Math.round(g.util)}% · ${(g.mem_used/1024).toFixed(1)}/${(g.mem_total/1024).toFixed(0)}GB · ${g.temp??'-'}°C ${Math.round(g.power??0)}W</span></div>
      <div class="bar"><div class="m" style="width:${Math.min(100,memPct)}%"></div></div>
      <div class="bar" style="height:5px;margin-top:2px"><div class="u" style="width:${Math.min(100,g.util)}%"></div></div></div>`;
  }).join("");
}
function modelChips(snap){
  const ms = snap?.sysmon?.models || [];
  if (!ms.length) return `<span class="chip">无加载模型</span>`;
  return ms.map(m=>{
    const cls = m.status==="ready" ? "ready" : "load";
    const rate = (m.rate?` ${Math.round(m.rate)}t/s`:"");
    return `<span class="chip ${cls}">${esc(m.name)}·GPU${esc(m.gpus)}${rate}</span>`;
  }).join("");
}
function card(n){
  const s = st[n.id];
  const offline = !s || s.__err;
  const sm = s?.sysmon, env = s?.env || (s?.snapshot && s.snapshot.env) || {};
  const snap = n.kind==="spark" ? s?.snapshot : s;
  const smx = snap?.sysmon;
  const gpuCount = n.kind==="spark" ? (smx?.gpus||[]).length : (sm?.gpus||[]).length;
  return `<div class="card ${offline?"off":""}" data-id="${n.id}">
    <div class="chead"><span class="cname"><span class="dot ${offline?"err":"on"}"></span>${esc(n.label)}
      <span class="ctag">${esc(n.id)}</span><span class="ctag">${n.kind==="spark"?"只读":"gw v"+esc(snap?.gateway_version||"?")}</span></span>
      <span class="mono hint">${offline?"离线/不可达":esc(snap?.node||"")}</span></div>
    <div class="kv">
      <span>LLM <b>${esc(snap?.env?.LLM_NAME || smx?.models?.[0]?.name || "-")}</b></span>
      <span>模式 <b>${esc(snap?.env?.AI_MODE || "-")}</b></span>
      <span>ctx <b>${snap?.env?.LLM_MAX_LEN?fmtK(+snap.env.LLM_MAX_LEN):"-"}</b></span>
      <span>CPU <b>${smx?Math.round(smx.cpu?.pct||0):"-"}%</b></span>
      <span>内存 <b>${smx?(smx.mem?.used_gb?.toFixed(0)||"-")+"G":"-"}</b></span>
      <span>磁盘 <b>${snap?.disk?snap.disk.free_gb+"G余":"-"}</b></span>
    </div>
    ${gpuBars(smx?.gpus)}
    <div class="chips">${modelChips(snap)}</div>
  </div>`;
}
async function refreshGrid(){
  $("#clock").textContent = new Date().toLocaleTimeString();
  await Promise.allSettled(nodes.map(async n=>{
    try {
      if (n.kind==="spark") {
        const [snap, models] = await Promise.allSettled([sparkApi("/api/snapshot"), sparkApi("/v1/models")]);
        st[n.id] = {snapshot:{sysmon: snap.status==="fulfilled"?snap.value:null,
                              gateway:{backends:{}}, env:{},
                              models_list: models.status==="fulfilled"?(models.value.data||[]).map(m=>m.id):[]}};
      } else {
        st[n.id] = await api(n, "/api/snapshot");
      }
    } catch(e){ st[n.id] = {__err:String(e.message||e)}; }
  }));
  $("#grid").innerHTML = nodes.map(card).join("");
  document.querySelectorAll(".card").forEach(el=>el.onclick=()=>openNode(el.dataset.id));
  if (cur && !$("#detail").classList.contains("hidden")) refreshDetail();
}
function sparkSnapshot(n){ // 统一渲染入口
  return st[n.id]?.snapshot || null;
}

/* ---------------- 详情 ---------------- */
function openNode(id){
  cur = nodes.find(x=>x.id===id);
  $("#grid").classList.add("hidden"); $("#detail").classList.remove("hidden");
  $("#dTitle").textContent = `${cur.label} · ${cur.id}`;
  $("#btnDanger").style.display = cur.kind==="stack" ? "" : "none";
  setTab("ov");
}
$("#btnBack").onclick = ()=>{ $("#detail").classList.add("hidden"); $("#grid").classList.remove("hidden"); cur=null; clearInterval(timers.detail); };
document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>setTab(b.dataset.tab));
function setTab(t){
  tab = t;
  document.querySelectorAll(".tab").forEach(b=>b.classList.toggle("on", b.dataset.tab===t));
  ["ov","cat","lay","combo","jobs"].forEach(k=>$("#tab-"+k).classList.toggle("hidden", k!==t));
  clearInterval(timers.detail);
  timers.detail = setInterval(refreshDetail, t==="lay"||t==="ov" ? 3000 : 4000);
  refreshDetail();
}
async function refreshDetail(){
  if (!cur) return;
  if (cur.kind==="spark") return renderSpark();
  try {
    // need 是普通对象没有 .map——必须显式 Object.values 再回挂键名，
    // 直接 Promise.all(对象) 会 TypeError: not iterable（首轮实测踩过）
    const proms = {snap: api(cur,"/api/snapshot")};
    if (tab==="cat") proms.catalog = api(cur,"/api/catalog");
    if (tab==="lay") proms.modes = api(cur,"/api/modes");
    if (tab==="combo") proms.combos = api(cur,"/api/combos");
    if (tab==="jobs") proms.jobs = api(cur,"/api/jobs");
    const got = Object.fromEntries(await Promise.all(
      Object.keys(proms).map(async k => [k, await proms[k]])));
    st[cur.id] = got.snap;
    if (tab==="ov") renderOv(got.snap);
    if (tab==="cat") renderCat(got.snap, got.catalog);
    if (tab==="lay") renderLay(got.snap, got.modes);
    if (tab==="combo") renderCombo(got.combos);
    if (tab==="jobs") renderJobs(got.jobs);
  } catch(e){ toast("✗ "+e.message); }
}

function renderOv(snap){
  const sm = snap.sysmon||{}, gw = snap.gateway||{}, b = gw.backends||{};
  const rows = Object.entries(b).map(([n,x])=>`<tr><td>${esc(n)}</td><td class="mono">${esc(x.status)}</td>
    <td class="mono">[${(x.gpus||[]).join(",")}] ×${x.replicas||1}</td><td class="mono">${x.in_flight||0}</td></tr>`).join("")
    || `<tr><td colspan="4" class="hint">无运行中后端（懒加载：请求到达自动起）</td></tr>`;
  $("#tab-ov").innerHTML = `
    <div class="kv" style="grid-template-columns:repeat(4,1fr)">
      <span>网关 <b>v${esc(snap.gateway_version||"?")}${snap.gateway_layout_modes?" · layout✓":""}</b></span>
      <span>布局 <b>${esc(snap.env?.AI_MODE||"-")}</b></span>
      <span>视频 <b>${esc(snap.env?.VIDEO_NAME||"none")}</b></span>
      <span>sysmon <b>${snap.sysmon?"在线":"离线"+(snap.sysmon_error?"("+esc(snap.sysmon_error)+")":"")}</b></span>
    </div>
    <h4>后端</h4><table><tr><th>模型</th><th>状态</th><th>GPU×副本</th><th>在途</th></tr>${rows}</table>
    <h4>吞吐（sysmonitor）</h4>
    <table><tr><th>模型</th><th>生成 t/s</th><th>预填 p/s</th><th>状态</th></tr>
      ${(sm.models||[]).map(m=>`<tr><td>${esc(m.name)}</td><td class="mono">${(m.rate||0).toFixed(1)}</td>
        <td class="mono">${m.prefill_active?("⚡"+Math.round(m.prompt_rate||0)):"-"}</td><td class="mono">${esc(m.status||"")}</td></tr>`).join("")
        || `<tr><td colspan=4 class="hint">—</td></tr>`}</table>
    <h4>机器</h4>${gpuBars(sm.gpus)}
    <div class="kv" style="grid-template-columns:repeat(4,1fr)">
      <span>CPU <b>${Math.round(sm.cpu?.pct||0)}%</b></span>
      <span>内存 <b>${(sm.mem?.used_gb||0).toFixed(0)}/${(sm.mem?.total_gb||0).toFixed(0)}G</b></span>
      <span>网络 <b>↓${((sm.net?.rx_kbps||0)/1024).toFixed(1)} ↑${((sm.net?.tx_kbps||0)/1024).toFixed(1)}MB/s</b></span>
      <span>磁盘 <b>${snap.disk?.free_gb??'-'}G 余</b></span>
    </div>
    <div class="frow">
      <button class="small" onclick="actSwitch('video')">切到视频优先（--switch video）</button>
      <button class="small" onclick="actSwitch('llm')">切回文本优先</button>
      <button class="small" onclick="actUnloadAll()">卸载全部空闲文本</button>
    </div>`;
}

async function renderCat(snap, cat){
  const gwb = snap.gateway?.backends||{};
  const llmRows = cat.llm.map(d=>`<tr>
    <td>${d.downloaded?"★":"　"} <b>${esc(d.name)}</b>${d.deprecated?' <span class="badge amber">弃用</span>':""}<br>
      <span class="hint">${esc(d.desc)}</span></td>
    <td class="mono">${esc(d.engine)}<br>${Object.entries(d.variants).map(([v,g])=>`${v}:${g}G`).join(" ")}</td>
    <td class="mono">${d.min_gpus}卡<br>ctx ${d.ctx?fmtK(d.ctx):"-"}</td>
    <td><div class="frow" style="margin:0">
      ${d.downloaded?`<button class="small" onclick="actLoad('${esc(d.name)}')">加载</button>`:
        `<button class="small" onclick="actDownload('${esc(d.name)}')">下载</button>
         ${d.variants&&d.variants.fp8!==undefined&&d.name?`<button class="small" onclick="actDownload('${esc(d.name)}','fp8')">下载fp8</button>`:""}`}
      ${gwb[d.name]?`<button class="small" onclick="actUnload('${esc(d.name)}')">卸载</button>
        <button class="small" onclick="actVerify('${esc(d.name)}')">实测</button>`:""}
    </div></td></tr>`).join("");
  const vidRows = cat.video.map(v=>`<tr><td>${v.downloaded?"★":"　"} ${esc(v.name)}</td>
      <td class="mono">${v.vram_480p_peak? v.vram_480p_peak+"G实测":(v.vram_480p??"-")+"G声明"}</td>
      <td class="mono">${v.vram_720p??"-"}</td>
      <td><span class="hint">${esc(v.desc)}</span></td>
      <td>${v.downloaded?"":`<button class="small" onclick="actDownload('${esc(v.name)}')">下载</button>`}</td></tr>`).join("");
  const imgRows = cat.image.map(i=>`<tr><td>${i.downloaded?"★":"　"} ${esc(i.name)}</td><td class="mono">~${i.size_gb}G</td>
      <td><span class="hint">${esc(i.desc)}</span></td>
      <td>${i.downloaded?"":`<button class="small" onclick="actDownload('${esc(i.name)}')">下载</button>`}</td></tr>`).join("");
  $("#tab-cat").innerHTML = `
    <h4>文本模型（下载状态=本机文件实测；“能跑什么”以布局页目录为准）</h4>
    <table><tr><th>LLM</th><th>引擎/变体</th><th>摆放</th><th>操作</th></tr>${llmRows}</table>
    <h4>视频模型（ComfyUI，整卡给视频按<b>实测峰值</b>判档）</h4>
    <table><tr><th>模型</th><th>480p</th><th>720p</th><th>说明</th><th></th></tr>${vidRows}</table>
    ${imgRows?`<h4>生图（与视频同卡轮流加载）</h4><table><tr><th>模型</th><th>权重</th><th>说明</th><th></th></tr>${imgRows}</table>`:""}`;
}

async function renderLay(snap, modes){
  const llms = Object.entries(modes.models||{});
  $("#tab-lay").innerHTML = `<div class="hint">运行态切换＝网关当场改摆放（models.json 只读不落盘）；
    “持久化”＝宿主 menu.py 按本机显存重写 gw/<code>.env</code>（下个请求生效）。
    <b>enforced</b>=现在切过去的实际 ctx；<b>ceiling</b>=持久化后该布局能吃满的 ctx。</div>` +
    llms.map(([name,m])=>{
      const act = m.active||{};
      return `<h4>${esc(name)} <span class="badge blue">${esc(m.engine)}</span>
        <span class="hint">现行: GPU[${(act.gpus||[]).join(",")}] ×${act.replicas||1} · enforced ctx ${fmtK(m.enforced_max_len||0)}</span></h4>` +
        (m.modes||[]).map(x=>`<div class="modecard ${x.active?"active":""}">
          <h4>${esc(x.id)} ${x.active?'<span class="badge green">现行</span>':""}
            ${x.video_coexist?'<span class="badge blue">可共存视频</span>':""}
            ${x.ctx_gain_requires_host?'<span class="badge amber">持久化后 ctx→'+fmtK(x.est_ctx_ceiling||0)+'</span>':""}</h4>
          <p>${esc(x.label||"")} · GPU[${(x.gpus||[]).join(",")}] ×${x.replicas??"-"}${x.tp?` TP=${x.tp}`:""} · enforced ${fmtK(x.enforced_max_len||0)}${x.est_ctx_ceiling&&x.est_ctx_ceiling!==x.enforced_max_len?` / ceiling ${fmtK(x.est_ctx_ceiling)}`:""}</p>
          <p>${esc(x.note||"")}</p>
          <div class="frow">
            <button class="small" ${x.runnable_via_load?"":"disabled"} onclick='actMode(${JSON.stringify(name)},${JSON.stringify(x.id)})'>运行态切换</button>
            ${x.host_command?`<button class="small" onclick='actPersist(${JSON.stringify(name)},${JSON.stringify(x.id)},${JSON.stringify(x.host_command)})'>持久化</button>`:""}
          </div></div>`).join("");
    }).join("");
}

async function renderJobs(jobList){
  const rows = (jobList||[]).slice().reverse().map(j=>
    `<div class="jrow" onclick="showLog('${esc(j.id)}')">
      <span class="jstat ${esc(j.status)}">${esc(j.status)}</span>
      <span style="color:var(--acc)">${esc(j.kind)}</span><span>${esc(j.note||"")}</span>
      <span class="hint" style="margin-left:auto">${new Date((j.started)*1000).toLocaleTimeString()} · ${esc(j.id)}</span></div>`).join("")
    || `<div class="hint">暂无任务</div>`;
  $("#tab-jobs").innerHTML = `<h4>任务（下载 / 持久布局 / 实测 / 初始化）</h4>${rows}
    ${showLog._cur?`<h4 id="logTitle">日志 ${esc(showLog._cur)}</h4><pre class="log" id="logBox">…</pre>`:""}`;
  if (showLog._cur) fetchLog(showLog._cur);
}
async function showLog(id){ showLog._cur=id; setTab("jobs"); }
async function fetchLog(id){
  try { const j = await api(cur, `/api/jobs/${id}/log?tail=12000`);
    const box = $("#logBox"); if (box && showLog._cur===id){ box.textContent = j.log; box.scrollTop = box.scrollHeight; }
  } catch(e){ /* 节点切换竞态时静默 */ }
}

function renderCombo(cb){
  if (cb.error) { $("#tab-combo").innerHTML = `<div class="hint">${esc(cb.error)}</div>`; return; }
  const cur0 = cb.current||{}; const c = cb.cards||{};
  const head = `<div class="hint">本机 ${c.count}×${c.gmem_gb}GB · 另一卡给视频的预算 <b>${c.video_card_gb}GB</b>
    · 单卡 LLM 位半池上限 ${c.half_budget_gb}GB · 当前：<b>${esc(cur0.llm||"?")}</b>（${esc(cur0.mode||"?")}）
    视频 <b>${esc(cur0.video||"none")}</b>${cur0.comfy==="1"?" · ComfyUI 在岗":""}。<br>${esc(cb.note||"")}</div>`;
  $("#tab-combo").innerHTML = head + (cb.nodes||[]).filter(x=>x.downloaded).map(x=>{
    return `<h4>${esc(x.llm)} <span class="badge blue">${esc(x.variant)} ${x.weights_gb}G</span>
      <span class="hint">${esc(x.engine)} · ctx ${x.ctx_single?fmtK(x.ctx_single):"-"}</span></h4>` +
      (x.combos||[]).map(slot=>{
        if (slot.kind==="exclusive")
          return `<div class="modecard"><h4>${esc(slot.layout)}</h4><p class="hint">独占布局没有"另一张卡"，视频/生图让位。</p></div>`;
        const vids = (slot.video||[]).map(v=>{
          const ok = !!v.tier;
          return `<span class="chip ${ok?"ready":""}" title="${esc(v.desc)}">${ok?"✓":"✗"} ${esc(v.model)}
            ${ok?`·${v.tier}${v.measured?"(实)":"(宣)"}`:""}${v.downloaded?"":" ⇓"}</span>`;
        }).join(" ");
        const imgs = (slot.image||[]).map(i=>
          `<span class="chip ${i.fits?"ready":""}">${i.fits?"✓":"✗"} ${esc(i.model)} ~${i.need_gb}G${i.downloaded?"":" ⇓"}</span>`).join(" ");
        const best = (slot.video||[]).filter(v=>v.tier&&v.downloaded)[0];
        return `<div class="modecard"><h4>${esc(slot.layout)} <span class="hint">视频预算 ${slot.video_budget_gb}G</span></h4>
          <div class="chips">${vids||'<span class="hint">无视频模型</span>'}${imgs?"　生图："+imgs:""}</div>
          <div class="frow">${best?`<button class="small" onclick='actCombo(${JSON.stringify(x.llm)},${JSON.stringify(best.model)})'>采纳：${esc(x.llm)} + ${esc(best.model)}</button>`:""}
            ${(slot.video||[]).filter(v=>v.tier).map(v=>`<button class="small ghost" onclick='actCombo(${JSON.stringify(x.llm)},${JSON.stringify(v.model)})'>${esc(v.model)}</button>`).join("")}
            <button class="small ghost" onclick='actCombo(${JSON.stringify(x.llm)},"none")'>纯文本</button></div></div>`;
      }).join("");
  }).join("") + `<div class="hint" style="margin-top:10px">⇓=未下载（采纳时自动先下）；(实)=按实测单卡峰值判档，(宣)=按声明值；
    独占位候选（iq3s 等）与互斥关系见「布局切换」页。</div>`;
}
window.actCombo = async (llm, video) => {
  const body = {llm, video: video||"none"};
  let dry;
  try { dry = await api(cur,"/api/apply",{method:"POST",body:JSON.stringify(Object.assign({dry:1},body))}); }
  catch(e){ return toast("✗ "+e.message); }
  modal("采纳组合（宿主 menu.py 持久化）", dry.command,
    "缺的模型会先自动下载；已加载后端不动，下个请求按新布局拉起", async()=>{
    await api(cur,"/api/apply",{method:"POST",body:JSON.stringify(body)});
    toast("已提交 → 任务页看日志"); setTab("jobs"); });
};

function renderSpark(){
  const s = sparkSnapshot(cur)||{}; const sm = s.sysmon||{};
  $("#tab-ov").innerHTML = `<div class="hint">异构节点（DGX Spark 独立栈，不在 ~/AI 蓝本内）：GUI 仅经 agent 白名单代理做<b>只读</b>监控；
    模型/服务操作用其自身 <code>~/gateway/vllm_service.sh</code>。</div>
    <div class="kv" style="grid-template-columns:repeat(4,1fr)"><span>主机 <b>${esc(sm.host||"-")}</b></span>
    <span>网关 <b>${sm.gateway_connected?"连通":"未连通"}</b></span>
    <span>模型 <b>${(s.models_list||[]).length}</b></span>
    <span>GPU <b>${(sm.gpus||[]).length}</b></span></div>
    ${gpuBars(sm.gpus)}
    <h4>模型列表（/v1/models）</h4><table><tr><th>id</th></tr>${(s.models_list||[]).map(m=>`<tr><td class="mono">${esc(m)}</td></tr>`).join("")}</table>`;
  ["cat","lay","combo","jobs"].forEach(k=>$("#tab-"+k).innerHTML = `<div class="hint">异构节点不支持此视图（只读降级）。</div>`);
}

/* ---------------- 操作 ---------------- */
window.actDownload = async (name, variant) => {
  modal("下载模型", `${cur.label} ← ${name}${variant?" ("+variant+")":""}`,
        "manage.sh download：走 hf-mirror；日志进「任务」页", async()=>{
    await api(cur, "/api/download", {method:"POST", body:JSON.stringify({name, variant:variant||""})});
    toast("已开始下载 → 任务页看进度"); setTab("jobs");
  });
};
window.actVerify = async name => { await api(cur,"/api/verify",{method:"POST",body:JSON.stringify({model:name})}); toast("实测任务已启动"); setTab("jobs"); };
window.actLoad = async name => { toast("加载中…（冷启动 1-5 分钟）");
  try { const r = await api(cur,"/api/gateway/load",{method:"POST",body:JSON.stringify({model:name,wait:false})});
        pollOp(r.op?.request_id, name); } catch(e){ toast("✗ "+e.message); } };
window.actUnload = async name => { await api(cur,"/api/gateway/unload",{method:"POST",body:JSON.stringify({model:name})}); toast("已卸载 "+name); refreshDetail(); };
window.actUnloadAll = async () => { const r = await api(cur,"/api/gateway/unload_all",{method:"POST",body:"{}"}); toast("卸载 "+(r.unloaded||[]).join(",")+(r.busy?.length?"（正忙未动: "+r.busy.join(",")+"）":"")); };
window.actSwitch = async t => { modal("切换显存归属", `target=${t}`, "等价 start.sh --switch "+t, async()=>{
  const r = await api(cur,"/api/gateway/switch",{method:"POST",body:JSON.stringify({target:t,wait:false})});
  toast("切换中…"); pollOp(r.op?.request_id, "switch"); }); };
window.actMode = async (name, mode) => { modal("运行态切布局（不落盘）", `${name} → mode=${mode}`,
  "网关先卸旧后端再按新摆放拉起；在途请求会被拒(409)", async()=>{
  try { const r = await api(cur,"/api/gateway/load",{method:"POST",body:JSON.stringify({model:name,mode,wait:false})});
        toast("切换中 → 任务/概览看状态"); pollOp(r.op?.request_id, name);} catch(e){ toast("✗ "+e.message);} }); };
window.actPersist = async (name, mode, hostCmd) => {
  let body = {llm:name, video: st[cur.id]?.env?.VIDEO_NAME || "none"};
  if (mode==="tp") body.tp = (st[cur.id]?.sysmon?.gpus||[]).length || 2;
  if (mode==="replica") body.replicas = (st[cur.id]?.sysmon?.gpus||[]).length || 2;
  let dry;
  try { dry = await api(cur,"/api/apply",{method:"POST",body:JSON.stringify(Object.assign({dry:1},body))}); }
  catch(e){ return toast("✗ "+e.message); }
  modal("持久化布局（宿主 menu.py 重写 .env + gw）", dry.command,
    "dry-run 预览如上；确认后作为任务执行，已加载后端不动、下个请求按新配置拉起", async()=>{
    const r = await api(cur,"/api/apply",{method:"POST",body:JSON.stringify(body)});
    toast("已提交 → 任务页看日志"); setTab("jobs"); });
};
$("#btnDanger").onclick = () => {
  if (!cur || cur.kind!=="stack") return;
  modal("初始化本机（重操作）", "init.sh --minimal / --full / --no-start",
    "可能触发大批模型下载与镜像构建；确认后再选模式", async()=>{}, "下一步…");
  $("#mOk").onclick = async()=>{ $("#modal").classList.add("hidden");
    const m = prompt("初始化模式: minimal | full | start_only（仅起服务不下载）", "start_only");
    if (!m) return;
    await api(cur,"/api/init",{method:"POST",body:JSON.stringify({mode:m})});
    toast("初始化任务已启动"); setTab("jobs"); };
};
async function pollOp(rid, name){
  if (!rid) return;
  for (let i=0;i<180;i++){
    await new Promise(r=>setTimeout(r,3000));
    try { const o = await api(cur, "/api/gateway/ops/"+rid);
      if (o.status==="success"){ toast("✓ "+name+" 就绪"); refreshDetail(); return; }
      if (o.status==="failed"){ toast("✗ "+(o.error||"失败").slice(0,90)); return; } }
    catch(e){ return toast("✗ "+e.message); }
  }
}

/* ---------------- 顶栏 ---------------- */
$("#btnNodes").onclick = () => {
  const txt = prompt("节点列表（每行 id|标签|kind|agentURL）", nodes.map(n=>[n.id,n.label,n.kind,n.url||""].join("|")).join("\n"));
  if (!txt) return;
  nodes = txt.split("\n").filter(x=>x.trim()).map(x=>{const [id,label,kind,url]=x.split("|"); return {id,label,kind,url};});
  localStorage.setItem("ai-nodes", JSON.stringify(nodes)); toast("已保存"); refreshGrid();
};
$("#btnToken").onclick = () => { const t = prompt("CONSOLE_TOKEN（留空=不设）", token)||""; token=t; localStorage.setItem("ai-token",t); toast("已保存"); refreshGrid(); };

refreshGrid();
setInterval(refreshGrid, 4000);
