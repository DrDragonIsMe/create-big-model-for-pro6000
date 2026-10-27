/* AI 集群控制台：原生无依赖，浏览器直连同构节点，DGX 经白名单只读代理。 */
"use strict";
const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

const DEFAULT_NODES = [
  {id:"home-pro6000-dual", label:"家庭双 Pro6000", kind:"stack", url:"http://100.95.107.118:9001"},
  {id:"workstation-4090-dual", label:"4090 工作站", kind:"stack", url:"http://100.78.215.20:9001"},
  {id:"tju-pro6000-single", label:"天大 Pro6000", kind:"stack", url:"http://100.119.153.75:9001"},
  {id:"dgx-spark-pair", label:"DGX Spark 双机", kind:"spark", relay:"dgx-spark-pair"},
];
let nodes = JSON.parse(localStorage.getItem("ai-nodes") || "null") || DEFAULT_NODES;
let token = localStorage.getItem("ai-token") || "";
let cur = null;
let tab = "ov";
const timers = {};
const st = {};

const TEMP_WARN = 83, TEMP_CRIT = 88, PRESSURE_WARN = 85, PRESSURE_CRIT = 95;
const LEVEL_RANK = {critical:0,warning:1,info:2,ok:3};
const fmtK = n => n >= 1048576 ? (n/1048576).toFixed(2)+"M" : n >= 1000 ? Math.round(n/1000)+"K" : String(Math.round(n||0));
const clamp = n => Math.max(0,Math.min(100,Number(n)||0));
function fmtBytes(n){const u=["B","K","M","G","T"];n=Math.max(0,Number(n)||0);for(const x of u){if(n<1024||x==="T")return x==="B"?`${n.toFixed(0)}${x}`:`${n.toFixed(1)}${x}`;n/=1024;}}
const fmtMB = mb => fmtBytes((Number(mb)||0)*1048576);
function fmtDur(s){if(s==null||s<0)return "-";s=Math.floor(s);if(s<60)return s+"s";if(s<3600)return `${Math.floor(s/60)}m`;if(s<86400)return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;return `${Math.floor(s/86400)}d ${Math.floor((s%86400)/3600)}h`;}
const sum = (xs,key) => (xs||[]).reduce((a,x)=>a+(Number(x[key])||0),0);
const netSum = sm => ({rx:sum(sm?.net,"rx"),tx:sum(sm?.net,"tx")});
const pct = (a,b) => b ? 100*a/b : 0;
const sev = v => v>=PRESSURE_CRIT?"crit":v>=PRESSURE_WARN?"warn":v>=60?"elevated":"";
const tempClass = t => t>=TEMP_CRIT?"temp-crit":t>=TEMP_WARN?"temp-warn":"";

async function api(node,path,opts={}){
  const headers=Object.assign({"Content-Type":"application/json"},opts.headers||{});
  if(token)headers.Authorization="Bearer "+token;
  const r=await fetch(node.url+path,Object.assign({headers,cache:"no-store"},opts));
  const j=await r.json().catch(()=>({__text:"非 JSON 响应"}));
  if(!r.ok)throw Object.assign(new Error(j?.error?.message||j?.error||("HTTP "+r.status)),{body:j,status:r.status});
  return j;
}
function relayNode(){return nodes.find(n=>n.kind==="stack"&&st[n.id]&&!st[n.id].__err)||nodes.find(n=>n.kind==="stack");}
async function sparkApi(path){const via=relayNode(),n=nodes.find(x=>x.id==="dgx-spark-pair");return api(via,`/api/relay?node=${encodeURIComponent(n.relay)}&path=${encodeURIComponent(path)}`);}
function nodeSnapshot(n){return n.kind==="spark"?st[n.id]?.snapshot:st[n.id];}
function toast(msg,ms=2600){const t=$("#toast");t.textContent=msg;t.classList.add("show");clearTimeout(t._h);t._h=setTimeout(()=>t.classList.remove("show"),ms);}
function modal(title,body,note,onOk,okLabel="执行"){
  $("#mTitle").textContent=title;$("#mBody").textContent=body;$("#mNote").textContent=note||"";$("#mOk").textContent=okLabel||"执行";$("#modal").classList.remove("hidden");
  $("#mOk").onclick=async()=>{$("#modal").classList.add("hidden");try{await onOk();}catch(e){toast("✗ "+e.message);}};
}
$("#mClose").onclick=()=>$("#modal").classList.add("hidden");
$("#modal").onclick=e=>{if(e.target===$("#modal"))$("#modal").classList.add("hidden");};
document.addEventListener("keydown",e=>{if(e.key==="Escape")$("#modal").classList.add("hidden");});

function alert(label,title,detail){return{label,title,detail:detail||""};}
function snapshotAlerts(node,snap){
  const sm=snap?.sysmon||{},gw=snap?.gateway||{},out=[];
  if(node.kind==="spark"&&sm.gateway_connected===false)out.push(alert("critical","DGX 网关不可达",String(sm.gateway||"监控无法连接模型服务；管理状态接口与 /v1/models 均不可用")));
  if(node.kind==="spark"&&sm.gateway_connected===true&&sm.gateway_protocol==="openai-compatible")out.push(alert("info","DGX 以 OpenAI 兼容模式在线","/v1/gateway/status 不支持；模型可用性已由 /v1/models 确认，token 指标不可用"));
  for(const cn of sm.cluster_nodes||[])if(!cn.online)out.push(alert("critical",`DGX Spark ${cn.role==="head"?"head":"worker"} 不可达`,cn.error||cn.address||cn.host||"SSH/辅助脚本探测失败"));
  if(!snap?.sysmon)out.push(alert("critical",`${node.label} · sysmonitor 不可达`,snap?.sysmon_error||"无法读取 GPU、CPU、内存与网络指标"));
  if(snap?.gateway_error)out.push(alert("warning",`${node.label} · 网关状态读取失败`,String(snap.gateway_error)));
  for(const g of sm.gpus||[]){
    const name=`${node.label} · GPU ${g.index}`;
    if(Number(g.temp||0)>=TEMP_CRIT)out.push(alert("critical",`${name} 温度过高`,`${g.temp}°C，告警阈值 ${TEMP_CRIT}°C`));
    else if(Number(g.temp||0)>=TEMP_WARN)out.push(alert("warning",`${name} 温度偏高`,`${g.temp}°C，注意阈值 ${TEMP_WARN}°C`));
    const pr=pct(g.power,g.power_limit);
    if(pr>=PRESSURE_CRIT)out.push(alert("warning",`${name} 接近功率上限`,`${(g.power||0).toFixed(0)} / ${(g.power_limit||0).toFixed(0)} W`));
    if(Number(g.mem_util||0)>=95)out.push(alert("warning",`${name} 显存带宽饱和`,`${(g.mem_util||0).toFixed(0)}%`));
  }
  if(sm.agg?.queued>0)out.push(alert("warning",`${node.label} · 推理请求排队`,`${sm.agg.queued} 个请求等待槽位`));
  for(const m of sm.models||[])if(m.status==="loading")out.push(alert("info",`${node.label} · ${m.name} 正在加载`,m.detail||"冷启动通常需要 1–5 分钟"));
  if((sm.mem?.swap_pct||0)>=PRESSURE_WARN)out.push(alert("warning",`${node.label} · 交换分区偏高`,`${sm.mem.swap_pct}% · ${fmtBytes(sm.mem.swap_used)} / ${fmtBytes(sm.mem.swap_total)}`));
  if((sm.disk?.pct||0)>=PRESSURE_WARN)out.push(alert("warning",`${node.label} · 系统盘空间不足`,`${sm.disk.pct}% · 剩余 ${fmtBytes(sm.disk.free)}`));
  if((sm.cpu?.total||0)>=PRESSURE_WARN)out.push(alert("warning",`${node.label} · CPU 压力偏高`,`${sm.cpu.total}% · load ${sm.cpu.load?.[0]??"-"}`));
  return out.sort((a,b)=>LEVEL_RANK[a.label]-LEVEL_RANK[b.label]);
}
function alertSummary(list){return{critical:list.filter(a=>a.label==="critical").length,warning:list.filter(a=>a.label==="warning").length,info:list.filter(a=>a.label==="info").length};}
function alertCard(a){return `<article class="alert ${esc(a.label)}" role="status"><span class="alert-icon" aria-hidden="true">${a.label==="critical"?"▲":a.label==="warning"?"△":"●"}</span><div class="alert-copy"><b>${esc(a.title)}</b><p>${esc(a.detail||"需要关注当前运行状态")}</p></div><span class="alert-level">${a.label==="critical"?"严重":a.label==="warning"?"预警":"信息"}</span></article>`;}
function alertPanel(alerts,title,note="温度预警 83°C / 严重 88°C · 系统压力预警 85%"){
  const s=alertSummary(alerts),statusClass=s.critical?"critical":s.warning?"warning":s.info?"info":"ok";
  const statusText=s.critical?"需要处理":s.warning?"运行预警":s.info?"信息提示":"全部正常";
  const top=alerts.length?`${alerts[0].title}${alerts[0].detail?` · ${alerts[0].detail}`:""}`:"温度、网关与队列均在正常范围";
  const body=alerts.length?alerts.map(alertCard).join(""):`<article class="alert ok"><span class="alert-icon" aria-hidden="true">✓</span><div class="alert-copy"><b>未发现告警</b><p>温度、系统压力、网关与队列均低于预警阈值。</p></div><span class="alert-level">正常</span></article>`;
  return `<details class="alert-strip ${statusClass}"><summary aria-label="${esc(title)}：${esc(statusText)}"><span class="alert-strip-label">${esc(title)}</span><span class="alert-state ${statusClass}">${esc(statusText)}</span><span class="alert-counts"><b class="critical">${s.critical}</b><span>严重</span><b class="warning">${s.warning}</b><span>预警</span><b class="info">${s.info}</b><span>信息</span></span><span class="alert-top" title="${esc(top)}">${esc(top)}</span><span class="alert-disclosure">详情</span></summary><div class="alert-detail-panel"><p>${esc(note)}</p><div class="alert-list">${body}</div></div></details>`;
}
function nodeStatusBadge(list){const s=alertSummary(list);if(s.critical)return `<span class="node-status critical">${s.critical} 严重</span>`;if(s.warning)return `<span class="node-status warning">${s.warning} 预警</span>`;if(s.info)return `<span class="node-status info">${s.info} 信息</span>`;return `<span class="node-status ok">正常</span>`;}
function compactAlerts(list){if(!list.length)return "";const s=alertSummary(list);const label=s.critical?`${s.critical} 项严重`:s.warning?`${s.warning} 项预警`:`${s.info} 项信息`;return `<div class="compact-alerts ${list[0].label}" title="${esc(list[0].title)} ${esc(list[0].detail||"")}"><b>${esc(label)}</b><span>${esc(list[0].title.replace(/^[^·]+· /,""))}</span></div>`;}

function bar(value,kind="u"){const p=clamp(value),cls=p>=PRESSURE_CRIT?"crit":p>=PRESSURE_WARN?"warn":p>=60?"elevated":"";return `<div class="bar"><div class="${kind} ${cls}" style="width:${p}%"></div></div>`;}
function gpuBars(gs){return (gs||[]).map(g=>{const mem=pct(g.mem_used,g.mem_total);return `<div class="gpubar"><div class="glabel"><span>GPU${g.index} ${esc((g.name||"").replace(/^NVIDIA /,""))}</span><span class="gpu-inline"><b class="${tempClass(g.temp)}">${g.temp??"-"}°C</b> · ${Math.round(g.util||0)}% 算力 · ${fmtMB(g.mem_used)} / ${fmtMB(g.mem_total)}</span></div>${bar(mem,"m")}<div class="bar util"><div class="u" style="width:${clamp(g.util)}%"></div></div></div>`;}).join("");}
function gpuDetails(gs){return `<div class="detail-gpus">${(gs||[]).map(g=>{const kind=g.memory_kind||"dedicated",memoryLabel=kind==="unified"?"统一内存":"显存",mem=pct(g.mem_used,g.mem_total),free=(g.mem_total||0)-(g.mem_used||0),powerPct=pct(g.power,g.power_limit);return `<div class="gpu-detail ${tempClass(g.temp)}"><div class="glabel"><span>GPU ${g.index} · ${esc((g.name||"").replace(/^NVIDIA /,""))}</span><span>${mem.toFixed(1)}% ${memoryLabel}</span></div><div class="gpubar"><div class="glabel"><span>计算利用率</span><span>${(g.util||0).toFixed(0)}%</span></div>${bar(g.util)}<div class="glabel"><span>${kind==="unified"?`vLLM 驻留 ${fmtMB(g.mem_used)} / ${memoryLabel} ${fmtMB(g.mem_total)}`:`${memoryLabel} ${fmtMB(g.mem_used)} / ${fmtMB(g.mem_total)}`}</span><span>${kind==="unified"?`OS 已用 ${fmtMB(g.os_used_mb||0)}`:`空闲 ${fmtMB(free)}`}</span></div>${bar(mem,"m")}${kind==="unified"?"":`<div class="glabel"><span>显存带宽利用率</span><span>${(g.mem_util||0).toFixed(0)}%</span></div>${bar(g.mem_util)}`}</div><div class="gpu-meta"><span class="chip temp ${tempClass(g.temp)||"normal"}"><b>${g.temp??"-"}°C</b><small>${g.temp>=TEMP_CRIT?"严重":g.temp>=TEMP_WARN?"偏高":"正常"}</small></span><span class="chip ${powerPct>=PRESSURE_CRIT?"temp-warn":""}"><b>${Math.round(g.power||0)}W</b><small>${g.power_limit?`功率 ${powerPct.toFixed(0)}%`:"功率"}</small></span>${g.fan!=null?`<span class="chip"><b>${Math.round(g.fan)}%</b><small>风扇</small></span>`:`<span class="chip"><b>${esc(g.host||"-")}</b><small>${esc(g.role||"node")}</small></span>`}</div></div>`;}).join("")||`<div class="empty">暂无 GPU 数据</div>`}</div>`;}
function modelChips(ms){if(!ms?.length)return `<span class="chip">暂无已加载模型 · 请求到达后按需启动</span>`;return ms.map(m=>`<span class="chip ${m.status==="ready"?"ready":m.status==="loading"?"load":"warn"}">${esc(m.name)} · GPU ${esc(m.gpus)}${m.rate?` · ${Math.round(m.rate)} t/s`:""}</span>`).join("");}

function card(n){
  const s=st[n.id],offline=!s||s.__err,snap=nodeSnapshot(n)||{},sm=snap.sysmon||{},env=snap.env||{},models=sm.models||[];
  const active=models[0],net=netSum(sm),diskPct=sm.disk?.pct??pct(snap.disk?.used_gb,snap.disk?.total_gb),cpuPct=sm.cpu?.total||0,memPct=sm.mem?.pct||0;
  const alerts=offline?[{label:"critical",title:`${n.label} · 节点不可达`,detail:s?.__err||"管理代理无响应"}]:snapshotAlerts(n,snap);
  const highestGpuTemp=Math.max(0,...(sm.gpus||[]).map(g=>Number(g.temp)||0));
  return `<article class="card ${offline?"off":""} ${alertSummary(alerts).critical?"is-critical":alertSummary(alerts).warning?"is-warning":""}" data-id="${esc(n.id)}" tabindex="0" role="button" aria-label="查看 ${esc(n.label)}">
    <div class="chead"><div class="cname"><span class="dot ${offline?"err":alertSummary(alerts).critical?"err":"on"}"></span>${esc(n.label)}<span class="ctag">${esc(n.id)}</span><span class="ctag">${n.kind==="spark"?"只读":"GW v"+esc(snap.gateway_version||"?")}</span></div><div class="chead-right"><span class="node-host hint">${offline?"离线 / 不可达":esc(snap.node||sm.host||"")}</span>${nodeStatusBadge(alerts)}</div></div>
    ${compactAlerts(alerts)}
    <div class="model-focus"><div><span class="label">${active?"运行模型":"默认文本模型"}</span><strong>${esc(active?.name||env.LLM_NAME||"未配置")}</strong><small>${active?esc(active.status):"请求到达后按需启动"}</small></div><div class="rate-hero"><span class="label">生成吞吐</span><b>${(sm.agg?.t||0).toFixed(1)}</b><small>t/s</small></div></div>
    <div class="metrics">
      <div class="metric ${sev(cpuPct)}"><span class="label">CPU</span><b>${sm.cpu?cpuPct.toFixed(1):"-"}%</b><small>${sm.cpu?.count||"-"} 核 · load ${sm.cpu?.load?.[0]??"-"}</small></div>
      <div class="metric mem"><span class="label">内存</span><b>${sm.mem?memPct.toFixed(1):"-"}%</b><small>${sm.mem?fmtBytes(sm.mem.used):"-"} · swap ${sm.mem?.swap_pct??"-"}%</small></div>
      <div class="metric ${sev(diskPct)}"><span class="label">磁盘</span><b>${Number(diskPct||0).toFixed(1)}%</b><small>${sm.disk?fmtBytes(sm.disk.free)+" 可用":snap.disk?snap.disk.free_gb+"G 可用":"-"}</small></div>
      <div class="metric ${highestGpuTemp?tempClass(highestGpuTemp):""}"><span class="label">GPU 温度</span><b>${highestGpuTemp?highestGpuTemp+"°C":"-"}</b><small>预警 ${TEMP_WARN}°C · 严重 ${TEMP_CRIT}°C</small></div>
    </div>
    <div class="gpu-list">${gpuBars(sm.gpus)}</div>
    <div class="chips">${modelChips(models)}</div>
    <div class="card-foot"><span>${(sm.gpus||[]).length} GPU · 运行 ${fmtDur(sm.uptime)}</span><span>请求 ${sm.agg?.req||0}${sm.agg?.queued?` · <b class="q-badge">${sm.agg.queued} 排队</b>`:""} · ${n.kind==="spark"?"只读监控":"进入节点 →"}</span></div>
  </article>`;
}
function renderClusterSummary(alerts){
  const snaps=nodes.map(nodeSnapshot).filter(Boolean),online=nodes.filter(n=>st[n.id]&&!st[n.id].__err).length;
  const gpuCount=snaps.reduce((a,s)=>a+(s.sysmon?.gpus||[]).length,0),rate=snaps.reduce((a,s)=>a+(s.sysmon?.agg?.t||0),0),req=snaps.reduce((a,s)=>a+(s.sysmon?.agg?.req||0),0),s=alertSummary(alerts);
  $("#clusterSummary").innerHTML=`<div class="summary-item"><b>${online}/${nodes.length}</b><span>节点在线</span></div><div class="summary-item"><b>${gpuCount}</b><span>GPU 总数</span></div><div class="summary-item"><b>${rate.toFixed(1)}</b><span>聚合 t/s</span></div><div class="summary-item"><b>${req}</b><span>活动请求</span></div><div class="summary-item ${s.critical?"critical":s.warning?"warning":""}"><b>${s.critical+s.warning}</b><span>实时告警</span></div>`;
}
function renderClusterAlerts(alerts){$("#clusterAlerts").innerHTML=alertPanel(alerts,"集群健康告警");}
async function refreshGrid(){
  $("#clock").textContent=new Date().toLocaleTimeString("zh-CN",{hour12:false});
  await Promise.allSettled(nodes.map(async n=>{try{if(n.kind==="spark"){const [snap,models]=await Promise.allSettled([sparkApi("/api/snapshot"),sparkApi("/v1/models")]);st[n.id]={snapshot:{node:n.label,sysmon:snap.status==="fulfilled"?snap.value:null,sysmon_error:snap.status==="fulfilled"?null:String(snap.reason?.message||"不可达"),gateway:{backends:{}},env:{},models_list:models.status==="fulfilled"?(models.value.data||[]).map(m=>m.id):[]}};}else st[n.id]=await api(n,"/api/snapshot");}catch(e){st[n.id]={__err:String(e.message||e)};}}));
  const alerts=nodes.flatMap(n=>snapshotAlerts(n,nodeSnapshot(n)||{}));
  $("#grid").innerHTML=nodes.map(card).join("");renderClusterSummary(alerts);renderClusterAlerts(alerts);
  document.querySelectorAll(".card").forEach(el=>{el.onclick=()=>openNode(el.dataset.id);el.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();openNode(el.dataset.id);}};});
}

function openNode(id){detailReq++;cur=nodes.find(x=>x.id===id);$("#overview").classList.add("hidden");$("#detail").classList.remove("hidden");$("#dTitle").textContent=`${cur.label} · ${cur.id}`;$("#btnDanger").style.display=cur.kind==="stack"?"":"none";setTab("ov");}
$("#btnBack").onclick=()=>{detailReq++;$("#detail").classList.add("hidden");$("#overview").classList.remove("hidden");cur=null;clearInterval(timers.detail);refreshGrid();};
document.querySelectorAll(".tab").forEach(b=>b.onclick=()=>setTab(b.dataset.tab));
function setTab(t){detailReq++;tab=t;document.querySelectorAll(".tab").forEach(b=>b.classList.toggle("on",b.dataset.tab===t));["ov","cat","lay","combo","jobs"].forEach(k=>$("#tab-"+k).classList.toggle("hidden",k!==t));clearInterval(timers.detail);timers.detail=setInterval(refreshDetail,t==="lay"||t==="ov"?3000:4000);refreshDetail();}
let detailReq=0;
async function refreshDetail(){
  if(!cur)return;if(cur.kind==="spark")return renderSpark();
  const req=++detailReq,node=cur,reqTab=tab;
  try{
    const proms={snap:api(node,"/api/snapshot")};
    if(reqTab==="cat")proms.catalog=api(node,"/api/catalog");
    if(reqTab==="lay")proms.modes=api(node,"/api/modes");
    if(reqTab==="combo")proms.combos=api(node,"/api/combos");
    if(reqTab==="jobs")proms.jobs=api(node,"/api/jobs");
    const got=Object.fromEntries(await Promise.all(Object.keys(proms).map(async k=>[k,await proms[k]])));
    if(req!==detailReq||node!==cur||reqTab!==tab)return;
    st[node.id]=got.snap;
    if(reqTab==="ov")renderOv(got.snap);
    if(reqTab==="cat")renderCat(got.snap,got.catalog);
    if(reqTab==="lay")renderLay(got.snap,got.modes);
    if(reqTab==="combo")renderCombo(got.combos);
    if(reqTab==="jobs")renderJobs(got.jobs);
  }catch(e){if(req===detailReq&&node===cur&&reqTab===tab)toast("✗ "+e.message);}
}
function stat(label,value,small="",cls=""){return `<div class="stat ${cls}"><span>${label}</span><b>${value}</b>${small?`<small>${small}</small>`:""}</div>`;}
function resource(label,value,kind,detail){const p=clamp(value),cls=kind==="mem"?"":sev(p);return `<div class="resource-row ${cls}"><span>${label}</span><span>${p.toFixed(1)}%</span>${bar(value,kind)}<div class="resource-detail">${detail||""}</div></div>`;}
function modelTable(sm,gw){
  const backends=gw?.backends||{};
  const rows=(sm.models||[]).map(m=>{const b=backends[m.name]||{},active=m.active??b.in_flight??0,limited=m.metrics_ok===false,status=limited?"info":m.status==="ready"?"ready":m.status==="loading"?"loading":"warn";return `<tr><td><b>${esc(m.name)}</b><br><span class="hint">${esc(m.engine)} · ${esc(m.status)}${m.max_model_len?` · ctx ${fmtK(m.max_model_len)}`:""}</span></td><td class="mono">GPU [${esc(m.gpus)}] ×${b.replicas||1}</td><td class="num">${active}${m.slots?` / ${m.slots}`:""}${active>m.slots&&m.slots?` <b class="q-badge">+${active-m.slots}</b>`:""}</td><td class="num ok-text">${(m.rate||0).toFixed(1)}</td><td class="num warn-text">${m.prompt_rate?Math.round(m.prompt_rate).toLocaleString():"--"}</td><td class="num">${limited?"状态接口不支持":`${fmtK(m.prompt_total||0)} / ${fmtK(m.generation_total||0)}`}</td><td class="status-pill ${status}">${limited?"ready*":esc(m.status||"-")}</td></tr>`;}).join("");
  return `<div class="table-wrap"><table><thead><tr><th>模型 / 引擎</th><th>摆放</th><th>活动 / 槽位</th><th>生成 t/s</th><th>Prompt p/s</th><th>累计 Prompt / Gen</th><th>状态</th></tr></thead><tbody>${rows||`<tr><td colspan="7" class="empty">暂无运行中后端，请求到达后按需加载</td></tr>`}</tbody></table></div>`;
}
function processTable(sm){
  const rows=[...(sm.procs||[])].sort((a,b)=>(b.vram||-1)-(a.vram||-1)||b.cpu-a.cpu).slice(0,10);
  return `<div class="table-wrap"><table><thead><tr><th>GPU</th><th>PID</th><th>用户</th><th>%CPU</th><th>%MEM</th><th>RSS</th><th>VRAM</th><th>模型 / 命令</th></tr></thead><tbody>${rows.map(p=>`<tr><td class="mono">${p.gpu!=null?"GPU"+p.gpu:"·"}</td><td class="mono">${p.pid}</td><td>${esc(p.user)}</td><td class="num ${sev(p.cpu)}">${(p.cpu||0).toFixed(1)}</td><td class="num">${(p.pmem||0).toFixed(1)}</td><td class="num">${fmtBytes(p.rss)}</td><td class="num">${p.vram?fmtMB(p.vram):"·"}</td><td class="cmd-cell" title="${esc(p.cmd)}">${p.model?`<span class="ok-text">[${esc(p.model)}]</span> `:""}${esc(p.cmd)}</td></tr>`).join("")||`<tr><td colspan="8" class="empty">暂无进程数据</td></tr>`}</tbody></table></div><div class="hint" style="margin-top:10px">按显存优先展示前 10 个进程 · 共 ${sm.procs_total||sm.procs?.length||0} 个进程</div>`;
}
function dgxClusterSection(sm){
  const cluster=sm.cluster||{},nodes=cluster.nodes||[];
  if(nodes.length<=1)return "";
  const cards=nodes.map(n=>{const mem=n.memory||{},gpu=n.gpu||{},c=n.container||{},resident=(gpu.resident_model_mb||0)*1048576;return `<article class="cluster-node ${n.online?"":"offline"}"><div class="cluster-node-head"><span class="status-dot ${n.online?"online":"offline"}"></span><b>${esc(n.host||n.address)}</b><span class="role-chip">${n.role==="head"?"HEAD":"WORKER"}</span><span class="cluster-ip">${esc(n.address||"-")}</span><span class="cluster-role">${esc(c.status||(n.online?"状态未知":"离线"))}</span></div><div class="cluster-gauge"><div class="cluster-gauge-label"><span>统一内存</span><b>${fmtBytes(mem.used||0)} / ${fmtBytes(mem.total||0)}</b></div>${bar(pct(mem.used,mem.total),"m")}<div class="cluster-detail">可用 ${fmtBytes(mem.available||0)} · OS 占用 ${pct(mem.used,mem.total).toFixed(1)}%</div></div><div class="cluster-gauge"><div class="cluster-gauge-label"><span>vLLM 模型驻留</span><b>${fmtBytes(resident)}</b></div>${bar(pct(resident,mem.total),"u")}<div class="cluster-detail">按各 Spark 节点计算进程 GPU 内存分配汇总</div></div><div class="cluster-facts"><span class="chip ${tempClass(gpu.temp)||"ready"}"><b>${gpu.temp??"-"}°C</b><small>GPU</small></span><span class="chip"><b>${Math.round(gpu.util||0)}%</b><small>算力</small></span><span class="chip"><b>${Math.round(gpu.power||0)}W</b><small>功耗</small></span></div></article>`;}).join("");
  return `<section class="panel cluster-panel"><div class="cluster-head"><div><h3 class="panel-title">DGX Spark 双机集群 <small>TP=2 · GB10 统一内存</small></h3><div class="hint">head 与 worker 分别显示，不把统一内存伪装成独立显存。</div></div><div class="cluster-summary"><span><b>${cluster.online_count}/${cluster.node_count}</b><small>节点在线</small></span><span><b>${fmtBytes(cluster.used_bytes||0)}</b><small>合计内存</small></span><span><b>${fmtBytes(cluster.resident_model_bytes||0)}</b><small>模型驻留</small></span><span><b>${cluster.used_pct||0}%</b><small>合计占用</small></span></div></div><div class="cluster-grid">${cards}</div></section>`;
}
function hardwarePanels(sm){
  const c=sm.cpu||{},m=sm.mem||{},d=sm.disk||{},net=netSum(sm);
  const cores=(c.per_core||[]).map((v,i)=>`<i class="${sev(v)}" style="height:${Math.max(4,clamp(v))}%" title="核心 ${i}: ${v.toFixed(1)}%"></i>`).join("");
  const nics=(sm.net||[]).map(n=>`<tr><td><b>${esc(n.name)}</b><br><span class="hint">${esc(n.ip||"-")}</span></td><td class="mono">${n.link_mbps?(n.link_mbps>=1000?n.link_mbps/1000+" Gbps":n.link_mbps+" Mbps"):"虚拟 / 未知"}</td><td class="num ok-text">↓ ${fmtBytes(n.rx)}/s</td><td class="num">↑ ${fmtBytes(n.tx)}/s</td><td class="num">${fmtBytes(n.rx_total)} / ${fmtBytes(n.tx_total)}</td></tr>`).join("");
  const disks=(sm.disks||[]).map(x=>resource(x.name,x.util,"u",`读 ${fmtBytes(x.r_bps)}/s · 写 ${fmtBytes(x.w_bps)}/s · ${Math.round((x.r_iops||0)+(x.w_iops||0)).toLocaleString()} io/s`)).join("");
  return `${dgxClusterSection(sm)}<section class="panel"><h3 class="panel-title">GPU 实时状态 <small>容量、压力与温度分层显示</small></h3>${gpuDetails(sm.gpus)}</section>
    <div class="detail-grid"><section class="panel"><h3 class="panel-title">处理器 <small>${c.count||"-"} 核 · ${c.freq?(c.freq/1000).toFixed(2)+" GHz":"频率未知"}</small></h3><div class="resource-list">${resource("总体",c.total,"u",`load ${(c.load||[]).map(x=>Number(x).toFixed(2)).join(" · ")||"-"}`)}</div><div class="core-grid">${cores}</div><div class="hint" style="margin-top:8px">每核实时占用 · 60% / 85% / 95% 分层</div></section>
    <section class="panel"><h3 class="panel-title">内存与交换分区</h3><div class="resource-list">${resource("内存",m.pct,"m",`${fmtBytes(m.used)} / ${fmtBytes(m.total)} · 可用 ${fmtBytes(m.avail)}`)}${resource("Swap",m.swap_pct,"u",`${fmtBytes(m.swap_used)} / ${fmtBytes(m.swap_total)}`)}</div></section>
    <section class="panel"><h3 class="panel-title">存储与 I/O</h3><div class="resource-list">${resource("系统盘",d.pct,"u",`${fmtBytes(d.used)} / ${fmtBytes(d.total)} · 剩余 ${fmtBytes(d.free)}`)}${disks||resource("盘读写",0,"u","暂无物理盘数据")}</div></section>
    <section class="panel"><h3 class="panel-title">网络 <small>合计 ↓${fmtBytes(net.rx)}/s · ↑${fmtBytes(net.tx)}/s</small></h3><div class="table-wrap"><table><thead><tr><th>接口</th><th>链路</th><th>接收</th><th>发送</th><th>累计 RX / TX</th></tr></thead><tbody>${nics||`<tr><td colspan="5" class="empty">暂无活动网卡</td></tr>`}</tbody></table></div></section></div>
    <section class="panel" style="margin-top:18px"><h3 class="panel-title">高占用进程 <small>GPU / CPU / 内存</small></h3>${processTable(sm)}</section>`;
}
function renderOv(snap){
  const sm=snap.sysmon||{},gw=snap.gateway||{},env=snap.env||{},net=netSum(sm),alerts=snapshotAlerts(cur,snap);
  $("#tab-ov").innerHTML=`<div class="stat-grid">
    ${stat("网关",`v${esc(snap.gateway_version||"?")}`,snap.gateway_layout_modes?"layout modes 可用":snap.gateway_error||"基础模式","accent")}
    ${stat("配置布局",esc(env.AI_MODE||"-"),`${esc(env.LLM_NAME||"-")} · ctx ${env.LLM_MAX_LEN?fmtK(+env.LLM_MAX_LEN):"-"}`)}
    ${stat("视频服务",esc(env.VIDEO_NAME||"none"),env.COMFY_ENABLED==="1"?"ComfyUI 已启用":"未启用")}
    ${stat("主机运行",fmtDur(sm.uptime),esc(sm.host||snap.node||"-"))}
    ${stat("活动 / 排队",`${sm.agg?.req||0} / ${sm.agg?.queued||0}`,"当前推理请求",sm.agg?.queued?"warning":"")}
    ${stat("网络吞吐",`↓${fmtBytes(net.rx)}/s`,`↑${fmtBytes(net.tx)}/s`)}
    </div>
    ${alertPanel(alerts,"节点实时告警")}
    <section class="panel"><h3 class="panel-title">模型服务 <small>sysmonitor 与网关实时数据</small></h3>${modelTable(sm,gw)}<div class="frow"><button class="small" onclick="actSwitch('video')">切到视频优先</button><button class="small" onclick="actSwitch('llm')">切回文本优先</button><button class="small ghost" onclick="actUnloadAll()">卸载全部空闲文本</button></div></section>
    ${snap.sysmon?hardwarePanels(sm):`<section class="panel"><div class="empty">sysmonitor 离线：${esc(snap.sysmon_error||"未知错误")}</div></section>`}`;
}

async function renderCat(snap,cat){
  const gwb=snap.gateway?.backends||{};
  const llmRows=cat.llm.map(d=>`<tr><td>${d.downloaded?"★":"　"} <b>${esc(d.name)}</b>${d.deprecated?' <span class="badge amber">弃用</span>':""}<br><span class="hint">${esc(d.desc)}</span></td><td class="mono">${esc(d.engine)}<br>${Object.entries(d.variants).map(([v,g])=>`${v}:${g}G`).join(" ")}</td><td class="mono">${d.min_gpus}卡<br>ctx ${d.ctx?fmtK(d.ctx):"-"}</td><td><div class="frow" style="margin:0">${d.downloaded?`<button class="small" onclick="actLoad('${esc(d.name)}')">加载</button>`:`<button class="small" onclick="actDownload('${esc(d.name)}')">下载</button>${d.variants&&d.variants.fp8!==undefined?`<button class="small" onclick="actDownload('${esc(d.name)}','fp8')">下载 fp8</button>`:""}`}${gwb[d.name]?`<button class="small" onclick="actUnload('${esc(d.name)}')">卸载</button><button class="small" onclick="actVerify('${esc(d.name)}')">实测</button>`:""}</div></td></tr>`).join("");
  const vidRows=cat.video.map(v=>`<tr><td>${v.downloaded?"★":"　"} ${esc(v.name)}</td><td class="mono">${v.vram_480p_peak?v.vram_480p_peak+"G 实测":(v.vram_480p??"-")+"G 声明"}</td><td class="mono">${v.vram_720p??"-"}</td><td><span class="hint">${esc(v.desc)}</span></td><td>${v.downloaded?"":`<button class="small" onclick="actDownload('${esc(v.name)}')">下载</button>`}</td></tr>`).join("");
  const imgRows=cat.image.map(i=>`<tr><td>${i.downloaded?"★":"　"} ${esc(i.name)}</td><td class="mono">~${i.size_gb}G</td><td><span class="hint">${esc(i.desc)}</span></td><td>${i.downloaded?"":`<button class="small" onclick="actDownload('${esc(i.name)}')">下载</button>`}</td></tr>`).join("");
  $("#tab-cat").innerHTML=`<section class="panel"><h3 class="panel-title">文本模型 <small>下载状态为本机文件实测</small></h3><div class="table-wrap"><table><thead><tr><th>LLM</th><th>引擎 / 变体</th><th>摆放</th><th>操作</th></tr></thead><tbody>${llmRows}</tbody></table></div></section><section class="panel"><h3 class="panel-title">视频模型 <small>按实测峰值判档</small></h3><div class="table-wrap"><table><thead><tr><th>模型</th><th>480p</th><th>720p</th><th>说明</th><th></th></tr></thead><tbody>${vidRows}</tbody></table></div></section>${imgRows?`<section class="panel"><h3 class="panel-title">生图模型 <small>与视频同卡轮流加载</small></h3><div class="table-wrap"><table><thead><tr><th>模型</th><th>权重</th><th>说明</th><th></th></tr></thead><tbody>${imgRows}</tbody></table></div></section>`:""}`;
}
async function renderLay(snap,modes){
  const llms=Object.entries(modes.models||{});
  $("#tab-lay").innerHTML=`<section class="panel"><div class="hint">运行态切换由网关当场改摆放，models.json 只读；持久化由宿主 menu.py 重写配置。<b>enforced</b> 为当前实际上下文，<b>ceiling</b> 为持久化后的上限。</div>${llms.map(([name,m])=>{const act=m.active||{};return `<h4>${esc(name)} <span class="badge blue">${esc(m.engine)}</span><span class="hint">现行 GPU[${(act.gpus||[]).join(",")}] ×${act.replicas||1} · ctx ${fmtK(m.enforced_max_len||0)}</span></h4>`+(m.modes||[]).map(x=>`<div class="modecard ${x.active?"active":""}"><h4>${esc(x.id)} ${x.active?'<span class="badge green">现行</span>':""}${x.video_coexist?'<span class="badge blue">可共存视频</span>':""}${x.ctx_gain_requires_host?'<span class="badge amber">持久化后 ctx→'+fmtK(x.est_ctx_ceiling||0)+'</span>':""}</h4><p>${esc(x.label||"")} · GPU[${(x.gpus||[]).join(",")}] ×${x.replicas??"-"}${x.tp?` TP=${x.tp}`:""} · enforced ${fmtK(x.enforced_max_len||0)}${x.est_ctx_ceiling&&x.est_ctx_ceiling!==x.enforced_max_len?` / ceiling ${fmtK(x.est_ctx_ceiling)}`:""}</p><p>${esc(x.note||"")}</p><div class="frow"><button class="small" ${x.runnable_via_load?"":"disabled"} onclick='actMode(${JSON.stringify(name)},${JSON.stringify(x.id)})'>运行态切换</button>${x.host_command?`<button class="small" onclick='actPersist(${JSON.stringify(name)},${JSON.stringify(x.id)},${JSON.stringify(x.host_command)})'>持久化</button>`:""}</div></div>`).join("");}).join("")}</section>`;
}
async function renderJobs(jobList){
  const rows=(jobList||[]).slice().reverse().map(j=>`<div class="jrow" onclick="showLog('${esc(j.id)}')"><span class="jstat ${esc(j.status)}">${esc(j.status)}</span><span style="color:var(--acc)">${esc(j.kind)}</span><span>${esc(j.note||"")}</span><span class="hint" style="margin-left:auto">${new Date(j.started*1000).toLocaleTimeString()} · ${esc(j.id)}</span></div>`).join("")||`<div class="empty">暂无任务</div>`;
  $("#tab-jobs").innerHTML=`<section class="panel"><h3 class="panel-title">任务日志 <small>下载 / 持久布局 / 实测 / 初始化</small></h3>${rows}${showLog._cur?`<h4 id="logTitle">日志 ${esc(showLog._cur)}</h4><pre class="log" id="logBox">…</pre>`:""}</section>`;if(showLog._cur)fetchLog(showLog._cur);
}
async function showLog(id){showLog._cur=id;setTab("jobs");}
async function fetchLog(id){try{const j=await api(cur,`/api/jobs/${id}/log?tail=12000`),box=$("#logBox");if(box&&showLog._cur===id){box.textContent=j.log;box.scrollTop=box.scrollHeight;}}catch(e){}}
function renderCombo(cb){
  if(cb.error){$("#tab-combo").innerHTML=`<section class="panel"><div class="empty">${esc(cb.error)}</div></section>`;return;}
  const cur0=cb.current||{},c=cb.cards||{};
  const head=`<section class="panel"><div class="hint">本机 ${c.count}×${c.gmem_gb}GB · 视频预算 <b>${c.video_card_gb}GB</b> · 单卡 LLM 半池上限 ${c.half_budget_gb}GB<br>当前 <b>${esc(cur0.llm||"?")}</b>（${esc(cur0.mode||"?")}） + ${esc(cur0.video||"none")}。${esc(cb.note||"")}</div>`;
  $("#tab-combo").innerHTML=head+(cb.nodes||[]).filter(x=>x.downloaded).map(x=>`<h4>${esc(x.llm)} <span class="badge blue">${esc(x.variant)} ${x.weights_gb}G</span><span class="hint">${esc(x.engine)} · ctx ${x.ctx_single?fmtK(x.ctx_single):"-"}</span></h4>`+(x.combos||[]).map(slot=>{if(slot.kind==="exclusive")return `<div class="modecard"><h4>${esc(slot.layout)}</h4><p>独占布局没有另一张卡，视频与生图让位。</p></div>`;const vids=(slot.video||[]).map(v=>{const ok=!!v.tier;return `<span class="chip ${ok?"ready":""}" title="${esc(v.desc)}">${ok?"✓":"✗"} ${esc(v.model)} ${ok?`·${v.tier}${v.measured?"(实)":"(宣)"}`:""}${v.downloaded?"":" ⇓"}</span>`;}).join(" ");const imgs=(slot.image||[]).map(i=>`<span class="chip ${i.fits?"ready":""}">${i.fits?"✓":"✗"} ${esc(i.model)} ~${i.need_gb}G${i.downloaded?"":" ⇓"}</span>`).join(" ");const best=(slot.video||[]).filter(v=>v.tier&&v.downloaded)[0];return `<div class="modecard"><h4>${esc(slot.layout)} <span class="hint">视频预算 ${slot.video_budget_gb}G</span></h4><div class="chips">${vids||'<span class="hint">无视频模型</span>'}${imgs?" 生图："+imgs:""}</div><div class="frow">${best?`<button class="small" onclick='actCombo(${JSON.stringify(x.llm)},${JSON.stringify(best.model)})'>采纳：${esc(x.llm)} + ${esc(best.model)}</button>`:""}${(slot.video||[]).filter(v=>v.tier).map(v=>`<button class="small ghost" onclick='actCombo(${JSON.stringify(x.llm)},${JSON.stringify(v.model)})'>${esc(v.model)}</button>`).join("")}<button class="small ghost" onclick='actCombo(${JSON.stringify(x.llm)},"none")'>纯文本</button></div></div>`;}).join("")).join("")+`<div class="hint" style="margin:12px 0">⇓ = 未下载；(实) = 实测单卡峰值，(宣) = 声明值。独占候选与互斥关系见布局切换页。</div></section>`;
}
window.actCombo=async(llm,video)=>{const body={llm,video:video||"none"};let dry;try{dry=await api(cur,"/api/apply",{method:"POST",body:JSON.stringify(Object.assign({dry:1},body))});}catch(e){return toast("✗ "+e.message);}modal("采纳组合（宿主持久化）",dry.command,"缺的模型会先自动下载；已加载后端不动，下个请求按新布局拉起",async()=>{await api(cur,"/api/apply",{method:"POST",body:JSON.stringify(body)});toast("已提交 → 任务页看日志");setTab("jobs");});};
function renderSpark(){
  const s=nodeSnapshot(cur)||{},sm=s.sysmon||{},alerts=snapshotAlerts(cur,s);
  $("#tab-ov").innerHTML=`<div class="stat-grid">${stat("节点类型","DGX Spark","独立软件栈 · 只读")}${stat("主机",esc(sm.host||"-"),`运行 ${fmtDur(sm.uptime)}`)}${stat("网关",sm.gateway_connected?"连通":"未连通","模型列表 "+(s.models_list||[]).length+" 个",sm.gateway_connected?"":"warning")}${stat("GPU",(sm.gpus||[]).length,"统一内存架构")}</div>${alertPanel(alerts,"DGX Spark 实时告警")}<section class="panel"><div class="hint">异构节点不开放写操作；模型与服务仍由 ~/gateway/vllm_service.sh 管理。</div><div class="chips">${(s.models_list||[]).map(m=>`<span class="chip ready">${esc(m)}</span>`).join("")}</div></section>${hardwarePanels(sm)}`;
  ["cat","lay","combo","jobs"].forEach(k=>$("#tab-"+k).innerHTML=`<section class="panel"><div class="empty">异构节点不支持此视图（只读监控）。</div></section>`);
}

window.actDownload=async(name,variant)=>{modal("下载模型",`${cur.label} ← ${name}${variant?" ("+variant+")":""}`,"manage.sh download：日志进入任务页",async()=>{await api(cur,"/api/download",{method:"POST",body:JSON.stringify({name,variant:variant||""})});toast("已开始下载 → 任务页看进度");setTab("jobs");});};
window.actVerify=async name=>{await api(cur,"/api/verify",{method:"POST",body:JSON.stringify({model:name})});toast("实测任务已启动");setTab("jobs");};
window.actLoad=async name=>{toast("加载中…（冷启动 1-5 分钟）");try{const r=await api(cur,"/api/gateway/load",{method:"POST",body:JSON.stringify({model:name,wait:false})});pollOp(r.op?.request_id,name);}catch(e){toast("✗ "+e.message);}};
window.actUnload=async name=>{await api(cur,"/api/gateway/unload",{method:"POST",body:JSON.stringify({model:name})});toast("已卸载 "+name);refreshDetail();};
window.actUnloadAll=async()=>{const r=await api(cur,"/api/gateway/unload_all",{method:"POST",body:"{}"});toast("卸载 "+(r.unloaded||[]).join(",")+(r.busy?.length?"（正忙未动: "+r.busy.join(",")+"）":""));};
window.actSwitch=async t=>{modal("切换显存归属",`target=${t}`,"等价 start.sh --switch "+t,async()=>{const r=await api(cur,"/api/gateway/switch",{method:"POST",body:JSON.stringify({target:t,wait:false})});toast("切换中…");pollOp(r.op?.request_id,"switch");});};
window.actMode=async(name,mode)=>{modal("运行态切布局（不落盘）",`${name} → mode=${mode}`,"网关先卸旧后端再按新摆放拉起；在途请求会被拒 (409)",async()=>{try{const r=await api(cur,"/api/gateway/load",{method:"POST",body:JSON.stringify({model:name,mode,wait:false})});toast("切换中 → 概览看状态");pollOp(r.op?.request_id,name);}catch(e){toast("✗ "+e.message);}});};
window.actPersist=async(name,mode,hostCmd)=>{const body={llm:name,video:st[cur.id]?.env?.VIDEO_NAME||"none"};if(mode==="tp")body.tp=(st[cur.id]?.sysmon?.gpus||[]).length||2;if(mode==="replica")body.replicas=(st[cur.id]?.sysmon?.gpus||[]).length||2;let dry;try{dry=await api(cur,"/api/apply",{method:"POST",body:JSON.stringify(Object.assign({dry:1},body))});}catch(e){return toast("✗ "+e.message);}modal("持久化布局",dry.command,"确认后作为任务执行；已加载后端不动，下个请求按新配置拉起",async()=>{await api(cur,"/api/apply",{method:"POST",body:JSON.stringify(body)});toast("已提交 → 任务页看日志");setTab("jobs");});};
$("#btnDanger").onclick=()=>{if(!cur||cur.kind!=="stack")return;modal("初始化本机（重操作）","init.sh --minimal / --full / --no-start","可能触发大批模型下载与镜像构建；确认后再选模式",async()=>{},"下一步…");$("#mOk").onclick=async()=>{$("#modal").classList.add("hidden");const m=prompt("初始化模式: minimal | full | start_only（仅起服务不下载）","start_only");if(!m)return;await api(cur,"/api/init",{method:"POST",body:JSON.stringify({mode:m})});toast("初始化任务已启动");setTab("jobs");};};
async function pollOp(rid,name){if(!rid)return;for(let i=0;i<180;i++){await new Promise(r=>setTimeout(r,3000));try{const o=await api(cur,"/api/gateway/ops/"+rid);if(o.status==="success"){toast("✓ "+name+" 就绪");refreshDetail();return;}if(o.status==="failed"){toast("✗ "+(o.error||"失败").slice(0,90));return;}}catch(e){return toast("✗ "+e.message);}}}

const themeMedia=matchMedia("(prefers-color-scheme: dark)");
let themePref=localStorage.getItem("ai-theme")||"auto";
function applyThemePref(pref,persist=true){themePref=pref;if(persist)localStorage.setItem("ai-theme",pref);document.documentElement.dataset.theme=pref==="auto"?(themeMedia.matches?"dark":"light"):pref;$("#themeSel").value=pref;}
$("#themeSel").onchange=e=>applyThemePref(e.target.value);
themeMedia.addEventListener("change",()=>{if(themePref==="auto")applyThemePref("auto",false);});
applyThemePref(themePref,false);
$("#btnNodes").onclick=()=>{const txt=prompt("节点列表（每行 id|标签|kind|agentURL）",nodes.map(n=>[n.id,n.label,n.kind,n.url||""].join("|")).join("\n"));if(!txt)return;nodes=txt.split("\n").filter(x=>x.trim()).map(x=>{const [id,label,kind,url]=x.split("|");return{id,label,kind,url};});localStorage.setItem("ai-nodes",JSON.stringify(nodes));toast("已保存");refreshGrid();};
$("#btnToken").onclick=()=>{const t=prompt("CONSOLE_TOKEN（留空=不设）",token)||"";token=t;localStorage.setItem("ai-token",t);toast("已保存");refreshGrid();};
refreshGrid();setInterval(refreshGrid,4000);
