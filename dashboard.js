/* ============================================================
   DHARA Analytics Dashboard — dashboard.js  v2.0
   Dual-pretrained Hindi Image Captioning with Cross-Attention
   and Transformer Representations
   Author: Dr. Himanshu Sharma
   https://www.linkedin.com/in/dr-himanshusharma/
   Theme and layout shared with the SAGE dashboard.
   ============================================================ */
'use strict';

/* ── Configuration ─────────────────────────────────────────── */
const GH = {
  owner  : 'himanshusharma-csit',
  repo   : 'dhara',             // ← change if the repository has another name
  branch : 'main',
  data   : 'Data'               // folder written by export_results.py
};
const BATCHES = [32, 64, 128, 256];
const PAPER_WT = 5;             // word threshold used for every result in the paper
const DS_INFO = {
  pascal1k: {label:'PASCAL 1K-Hindi', ico:'🖼️', desc:'1,000 images · 5 manual Hindi captions each'},
  flickr8k: {label:'Flickr 8K-Hindi', ico:'📷', desc:'8,091 images · translated Hindi captions'}
};

/* Palette (same tokens as dashboard.css) */
const C = {blue:'#2563eb',purple:'#7c3aed',green:'#059669',orange:'#d97706',red:'#dc2626',cyan:'#0891b2',pink:'#db2777',slate:'#4b5580'};
const BCOL = {32:C.blue,64:C.purple,128:C.green,256:C.orange};
const WCOL = {1:C.cyan,2:C.pink,3:C.purple,4:C.orange,5:C.blue};
const MET = [['B1','BLEU-1'],['B2','BLEU-2'],['B3','BLEU-3'],['B4','BLEU-4'],['M','METEOR'],['C','CIDEr']];
const MLAB = Object.fromEntries([...MET,['R','ROUGE-L']]);
const MCOL = {B1:C.blue,B2:C.purple,B3:C.cyan,B4:C.green,M:C.orange,C:C.pink,R:C.red};

/* ── Global state ──────────────────────────────────────────── */
let RUNS = [];          // runs.json
let D = null;           // active run
let BENCH = null;       // benchmarks.json
let HUMAN = null;       // human_eval.json
let CHS = {};
const CACHE = {};
let PICK = {ds:null, wt:PAPER_WT, bs:null};
let CAP_SHOWN = 12, CAP_SORT = 'file', SAME_SEED = 0;

/* ══════════════════════════════════════════════════════════════
   1.  DATA ACCESS
   ══════════════════════════════════════════════════════════════ */
function rawUrl(path){ return `https://raw.githubusercontent.com/${GH.owner}/${GH.repo}/${GH.branch}/${path}`; }
async function fetchJSON(path){
  const tries=[path+'?v='+Date.now(), rawUrl(path)+'?v='+Date.now()];
  let last=null;
  for(const u of tries){
    try{ const r=await fetch(u,{cache:'no-store'}); if(r.ok) return await r.json(); last=new Error(`HTTP ${r.status} for ${path}`); }
    catch(e){ last=e; }
  }
  throw last||new Error('Cannot fetch '+path);
}
async function getRun(id){
  if(!CACHE[id]) CACHE[id]=prepare(await fetchJSON(`${GH.data}/runs/${id}.json`));
  return CACHE[id];
}
async function getBench(){ if(!BENCH) BENCH=await fetchJSON(`${GH.data}/benchmarks.json`); return BENCH; }
async function getHuman(){ if(!HUMAN) HUMAN=await fetchJSON(`${GH.data}/human_eval.json`); return HUMAN; }
const runId=(ds,wt,bs)=>`${ds}-wt${wt}-b${bs}`;
const findRun=(ds,wt,bs)=>RUNS.find(r=>r.dataset===ds&&r.word_threshold===wt&&r.batch_size===bs);

/* ── stats helpers ─────────────────────────────────────────── */
const isNum=v=>typeof v==='number'&&isFinite(v);
const mean=a=>{a=a.filter(isNum);return a.length?a.reduce((s,v)=>s+v,0)/a.length:null;};
const fx=(v,d=2)=>isNum(v)?v.toFixed(d):'–';
const sgn=(v,d=2)=>isNum(v)?(v>=0?'+':'')+v.toFixed(d):'–';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function pearson(a,b){
  const p=a.map((x,i)=>[x,b[i]]).filter(([x,y])=>isNum(x)&&isNum(y)); if(p.length<3) return null;
  const mx=mean(p.map(q=>q[0])),my=mean(p.map(q=>q[1]));
  let sxy=0,sx=0,sy=0; p.forEach(([x,y])=>{sxy+=(x-mx)*(y-my);sx+=(x-mx)**2;sy+=(y-my)**2;});
  return sx&&sy?sxy/Math.sqrt(sx*sy):null;
}
/* ROUGE-L is not in the logs (the pipeline wrote 0); the paper's value is used for paper runs. */
function paperRouge(ds,wt,bs){
  if(!BENCH||wt!==PAPER_WT) return null;
  const p=BENCH.datasets[ds]&&BENCH.datasets[ds].paper_dhara[String(bs)]; return p?p.R:null;
}
function testOf(run){
  if(!run||!run.test) return null;
  const t={...run.test}; if(!isNum(t.R)) { t.R=paperRouge(run.dataset,run.word_threshold,run.batch_size); t.Rpaper=isNum(t.R); }
  return t;
}

function prepare(raw){
  const d=raw;
  d.tr=(raw.train&&raw.train.epochs)||[];
  d.val=raw.val||[];
  d.testEpoch=raw.test?raw.test.epoch:null;
  d.bestVal=d.val.length?d.val.reduce((b,v)=>(v.B4>b.B4?v:b)):null;
  d.valAtTest=d.val.find(v=>v.epoch===d.testEpoch)||null;
  d.caps=(raw.test_captions&&raw.test_captions.items)||[];
  d.hasRefs=d.caps.some(c=>c.refs&&c.refs.length);
  return d;
}

/* ══════════════════════════════════════════════════════════════
   2.  RUN SELECTOR
   ══════════════════════════════════════════════════════════════ */
function setStatus(msg,type='loading'){
  const bar=document.getElementById('statusBar'),txt=document.getElementById('statusText'),spn=document.getElementById('statusSpinner');
  bar.className='status-bar'+(type==='ok'?' ok':type==='err'?' err':''); txt.textContent=msg;
  if(spn) spn.className='status-spinner'+(type==='loading'?'':' hide');
}
function openModal(){ document.getElementById('overlay').style.display='flex'; }
window.openModal=openModal;

const AV_KEYS=[['train','Training logs'],['val','Validation'],['test','Test results'],['captions','Test captions'],['evolution','Caption evolution']];

async function scanRuns(){
  document.getElementById('repoName').textContent=`${GH.owner} / ${GH.repo}`;
  setStatus('Loading runs…','loading');
  try{
    RUNS=await fetchJSON(`${GH.data}/runs.json`);
    await getBench().catch(()=>null);
    if(!Array.isArray(RUNS)||!RUNS.length){ setStatus('⚠ runs.json is empty — run export_results.py.','err'); return; }
    const nPaper=RUNS.filter(r=>r.paper_config).length;
    setStatus(`✓ ${RUNS.length} runs · ${nPaper} used in the paper`,'ok');
    const want=new URLSearchParams(location.search).get('run');
    const w=want&&RUNS.find(r=>r.id===want);
    if(w){ PICK={ds:w.dataset,wt:w.word_threshold,bs:w.batch_size}; drawPicker(); loadSelectedRun(); return; }
    PICK.ds=PICK.ds||RUNS[0].dataset; drawPicker();
  }catch(err){
    console.error(err);
    setStatus('❌ data/runs.json not found.','err');
    document.getElementById('extraInfo').innerHTML=`
      <div class="src-note"><b>Run the exporter first:</b><br>
      <code>python export_results.py --src "path/to/DHARA Saved Resources" --out data</code><br>
      then commit the <code>data/</code> folder. Opening index.html straight from disk needs a local server
      (<code>python -m http.server</code>).</div>
      <button class="retry-btn" onclick="scanRuns()">↺ Retry</button>`;
  }
}
window.scanRuns=scanRuns;

function drawPicker(){
  const dss=[...new Set(RUNS.map(r=>r.dataset))];
  document.getElementById('dsRow').innerHTML=dss.map(ds=>{
    const n=RUNS.filter(r=>r.dataset===ds).length, i=DS_INFO[ds]||{label:ds,ico:'📁',desc:''};
    return `<div class="ds-card${PICK.ds===ds?' sel':''}" onclick="pickDs('${ds}')"><div class="ds-name">${i.ico} ${esc(i.label)}</div><div class="ds-desc">${esc(i.desc)} · ${n} run${n>1?'s':''}</div></div>`;
  }).join('');
  const wts=[...new Set(RUNS.filter(r=>r.dataset===PICK.ds).map(r=>r.word_threshold))].sort((a,b)=>a-b);
  if(!wts.includes(PICK.wt)) PICK.wt=wts.includes(PAPER_WT)?PAPER_WT:wts[0];
  document.getElementById('wtRow').innerHTML=`<span class="wt-lbl">Word threshold</span>`+wts.map(w=>
    `<button class="wt-chip${PICK.wt===w?' sel':''}" onclick="pickWt(${w})">≥ ${w}${w===PAPER_WT?'<span class="paper">PAPER</span>':''}</button>`).join('')+
    `<span class="cap-note" style="margin-left:6px">minimum word frequency kept in the vocabulary</span>`;
  const list=document.getElementById('batchList');
  list.innerHTML='';
  BATCHES.forEach(bs=>{
    const info=findRun(PICK.ds,PICK.wt,bs);
    const div=document.createElement('div');
    if(!info){
      div.className='batch-opt pending';
      div.innerHTML=`<div class="batch-opt-ico">⏳</div><div style="flex:1;min-width:0"><div class="batch-opt-name">Batch-${bs} <span style="font-size:10px;font-weight:700;color:var(--orange)">● Not exported</span></div>
        <div class="batch-opt-desc">No folder for this configuration in the exported data</div></div>`;
      list.appendChild(div); return;
    }
    const av=info.availability||{};
    const dots=AV_KEYS.map(([k,l])=>`<span class="bdot ${av[k]?'ok':'no'}" title="${l}: ${av[k]?'available':'missing'}"></span>`).join('');
    const t=info.test;
    const dup=info.duplicate_logs_of?` · <span style="color:var(--orange)">logs duplicate ${esc(info.duplicate_logs_of)}</span>`:'';
    div.className='batch-opt'+(PICK.bs===bs?' sel':'');
    div.innerHTML=`<div class="batch-opt-ico">📁</div><div style="flex:1;min-width:0">
        <div class="batch-opt-name">Batch-${bs} ${info.paper_config?'<span class="tag-paper">PAPER</span>':''}</div>
        <div class="batch-opt-desc">${t?`Test BLEU-1 ${fx(t.B1)} · BLEU-4 ${fx(t.B4)} · CIDEr ${fx(t.C)} (epoch ${t.epoch})`:'No test result'} · vocab ${info.vocab_size??'–'}${dup}</div>
        <div class="batch-dots">${dots}</div></div><div class="chk">✓</div>`;
    div.addEventListener('click',()=>{ PICK.bs=bs; list.querySelectorAll('.batch-opt').forEach(o=>o.classList.remove('sel')); div.classList.add('sel'); document.getElementById('btnLoad').disabled=false; });
    list.appendChild(div);
  });
  document.getElementById('btnLoad').disabled=!(PICK.bs&&findRun(PICK.ds,PICK.wt,PICK.bs));
}
function pickDs(ds){ PICK.ds=ds; drawPicker(); }
function pickWt(w){ PICK.wt=w; drawPicker(); }
window.pickDs=pickDs; window.pickWt=pickWt;

async function loadSelectedRun(){
  const info=findRun(PICK.ds,PICK.wt,PICK.bs); if(!info) return;
  const btn=document.getElementById('btnLoad'),prog=document.getElementById('fetchProgress'),bar=document.getElementById('fetchProgressBar');
  btn.disabled=true; prog.style.display='block'; bar.style.width='30%';
  setStatus(`Loading ${info.id}…`,'loading');
  try{
    D=await getRun(info.id); bar.style.width='100%';
    setTimeout(()=>{
      document.getElementById('overlay').style.display='none';
      prog.style.display='none'; bar.style.width='0%'; btn.disabled=false;
      document.getElementById('runLabel').textContent=`${D.dataset_label} · word threshold ≥ ${D.word_threshold} · Batch-${D.batch_size} · ${(D.tr.length||D.val.length)?(D.tr.length||D.val.length)+' epochs':'test result only'}${D.paper_config?' · paper configuration':''}`;
      document.getElementById('cpBadge').textContent=D.test?`Epoch ${D.test.epoch} — Tested Checkpoint`:'No test result';
      const u=new URL(location.href); u.searchParams.set('run',D.id); history.replaceState(null,'',u);
      setStatus(`✓ ${D.id} loaded`,'ok');
      renderAll();
    },200);
  }catch(err){ console.error(err); setStatus('❌ '+err.message,'err'); btn.disabled=false; prog.style.display='none'; }
}
window.loadSelectedRun=loadSelectedRun;

/* ══════════════════════════════════════════════════════════════
   3.  CHART HELPERS
   ══════════════════════════════════════════════════════════════ */
Chart.defaults.color='#8892b0';
Chart.defaults.borderColor='#e2e6f0';
Chart.defaults.font.family="'Outfit',sans-serif";
Chart.defaults.font.size=11;
const GC='#e8edf5';
const LEG={position:'top',labels:{usePointStyle:true,boxWidth:7,color:'#4b5580',padding:10}};
function sc(xl,yl){
  return{x:{title:{display:!!xl,text:xl,color:'#8892b0'},grid:{color:GC},ticks:{color:'#8892b0'}},
         y:{title:{display:!!yl,text:yl,color:'#8892b0'},grid:{color:GC},ticks:{color:'#8892b0'}}};
}
function grd(ctx,c1,c2,h=300){ const g=ctx.createLinearGradient(0,0,0,h); g.addColorStop(0,c1); g.addColorStop(1,c2); return g; }
function rgba(hex,a){ const n=parseInt(hex.slice(1),16); return `rgba(${n>>16&255},${n>>8&255},${n&255},${a})`; }
function mkChart(id,type,data,opts,plugins){
  if(CHS[id]) CHS[id].destroy();
  const el=document.getElementById(id); if(!el) return null;
  const fixed=el.parentElement&&el.parentElement.hasAttribute('data-h');
  CHS[id]=new Chart(el,{type,data,options:{responsive:true,maintainAspectRatio:!fixed,...opts},plugins:plugins||[]});
  return CHS[id];
}
function mkChartNoAR(id,type,data,opts,plugins){
  if(CHS[id]) CHS[id].destroy();
  const el=document.getElementById(id); if(!el) return null;
  CHS[id]=new Chart(el,{type,data,options:{responsive:true,maintainAspectRatio:false,...opts},plugins:plugins||[]});
  return CHS[id];
}
function emptyCanvas(id,msg){
  if(CHS[id]){CHS[id].destroy(); delete CHS[id];}
  const el=document.getElementById(id); if(!el) return;
  const ctx=el.getContext('2d'); ctx.clearRect(0,0,el.width,el.height);
  ctx.fillStyle='#8892b0'; ctx.font="13px 'Outfit',sans-serif"; ctx.textAlign='center';
  ctx.fillText(msg,el.width/2,el.height/2);
}
function heatColor(t){
  const stops=[[255,255,255],[219,234,254],[96,165,250],[29,78,216]];
  t=Math.max(0,Math.min(1,t||0)); const n=stops.length-1,si=Math.min(Math.floor(t*n),n-1),f=t*n-si;
  const[r1,g1,b1]=stops[si],[r2,g2,b2]=stops[si+1];
  return`rgb(${Math.round(r1+(r2-r1)*f)},${Math.round(g1+(g2-g1)*f)},${Math.round(b1+(b2-b1)*f)})`;
}
function divColor(t){
  t=Math.max(-1,Math.min(1,t||0));
  if(t>=0) return `rgb(${Math.round(255-t*(255-37))},${Math.round(255-t*(255-99))},${Math.round(255-t*(255-235))})`;
  const u=-t; return `rgb(${Math.round(255-u*(255-220))},${Math.round(255-u*(255-38))},${Math.round(255-u*(255-38))})`;
}
const ink=t=>t>.55?'#fff':'#1e2340';
/* vertical marker for the tested epoch */
const epochMarker={
  id:'epochMarker',
  afterDatasetsDraw(chart,args,o){
    if(!o||o.epoch==null) return; const x=chart.scales.x; if(!x) return;
    const px=x.getPixelForValue(chart.data.labels?chart.data.labels.indexOf(o.epoch):o.epoch);
    if(!isNum(px)||px<x.left||px>x.right) return;
    const {ctx,chartArea:a}=chart; ctx.save(); ctx.strokeStyle=rgba(C.red,.55); ctx.setLineDash([4,4]); ctx.lineWidth=1.2;
    ctx.beginPath(); ctx.moveTo(px,a.top); ctx.lineTo(px,a.bottom); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle=C.red; ctx.font="700 10px 'Outfit',sans-serif"; ctx.textAlign=px>a.right-60?'right':'left';
    ctx.fillText(o.label||'tested',px+(px>a.right-60?-4:4),a.top+10); ctx.restore();
  }
};

/* ══════════════════════════════════════════════════════════════
   4.  NAVIGATION & TABS
   ══════════════════════════════════════════════════════════════ */
const TITLES={dashboard:'Dashboard — Overview',training:'Training Logs',evaluation:'Evaluation Metrics',captions:'Hindi Captions',
  threshold:'Word-Threshold Study',bench:'Benchmark Comparison',human:'Human Evaluation',compare:'Batch-Size Comparison',insights:'Key Insights'};
function goPage(name,btn){
  if(window.innerWidth<=900) closeSb();
  const prev=document.querySelector('.page.active');
  if(prev){ prev.classList.remove('active'); void prev.offsetWidth; }
  document.getElementById('page-'+name).classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  if(btn){ btn.classList.add('active'); const ico=btn.querySelector('.nav-ico'); if(ico){ico.style.transform='scale(1.35)';setTimeout(()=>ico.style.transform='',220);} }
  else { const b=[...document.querySelectorAll('.nav-btn')].find(x=>(x.getAttribute('onclick')||'').includes(`'${name}'`)); if(b) b.classList.add('active'); }
  document.getElementById('pageTitle').textContent=TITLES[name]||name;
  if(D){ if(name==='compare') renderCompare(); if(name==='bench') renderBench(); if(name==='threshold') renderThreshold(); if(name==='human') renderHuman(); if(name==='captions') renderSame(); }
  [30,300,700].forEach(ms=>setTimeout(()=>Object.values(CHS).forEach(c=>{ if(c.canvas&&c.canvas.offsetParent) c.resize(); }),ms));
}
window.goPage=goPage;
function toggleSb(){
  const sb=document.getElementById('sidebar'),ov=document.getElementById('sbOverlay'),hb=document.getElementById('hamburger');
  if(sb.classList.contains('open')) closeSb(); else { sb.classList.add('open'); ov.classList.add('show'); hb.classList.add('open'); document.body.style.overflow='hidden'; }
}
function closeSb(){
  document.getElementById('sidebar').classList.remove('open'); document.getElementById('sbOverlay').classList.remove('show');
  document.getElementById('hamburger').classList.remove('open'); document.body.style.overflow='';
}
window.toggleSb=toggleSb; window.closeSb=closeSb;
window.addEventListener('resize',()=>{ if(window.innerWidth>900) closeSb(); });
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeSb(); });
function initTabs(){
  document.querySelectorAll('.tabs').forEach(bar=>{
    bar.addEventListener('click',e=>{
      const tab=e.target.closest('.tab'); if(!tab||!tab.dataset.pane) return;
      bar.querySelectorAll('.tab').forEach(t=>t.classList.remove('active')); tab.classList.add('active');
      [...bar.querySelectorAll('[data-pane]')].map(t=>t.dataset.pane).forEach(p=>{ const el=document.getElementById(p); if(el) el.style.display=p===tab.dataset.pane?'block':'none'; });
      setTimeout(()=>Object.values(CHS).forEach(c=>{ if(c.canvas&&c.canvas.offsetParent) c.resize(); }),20);
    });
  });
}
function buildTabs(id,items,onPick,activeKey){
  const bar=document.getElementById(id); if(!bar) return;
  bar.innerHTML=items.map(([k,l])=>`<button class="tab${String(k)===String(activeKey)?' active':''}" data-k="${esc(k)}">${l}</button>`).join('');
  bar.onclick=e=>{ const t=e.target.closest('.tab'); if(!t) return; bar.querySelectorAll('.tab').forEach(x=>x.classList.remove('active')); t.classList.add('active'); onPick(t.dataset.k); };
}

/* ══════════════════════════════════════════════════════════════
   5.  KPI GRID
   ══════════════════════════════════════════════════════════════ */
function kpiHTML(kpis){
  return kpis.map(k=>`
    <div class="kpi">
      <div class="kpi-ico" style="background:${k.bg}">${k.ico}</div>
      <div>
        <div class="kpi-lbl">${k.lbl}</div>
        <div class="kpi-val" style="color:${k.col}" data-target="${k.val}">${k.val}</div>
        ${k.delta?`<div class="kpi-delta ${k.dn?'dn':'up'}">${k.delta}</div>`:''}
        ${k.sub?`<div class="kpi-sub">${k.sub}</div>`:''}
      </div>
    </div>`).join('');
}
function countUp(root){
  requestAnimationFrame(()=>{
    root.querySelectorAll('.kpi-val').forEach((el,i)=>{
      const raw=parseFloat(el.dataset.target); if(isNaN(raw)||!/^-?[\d.]+$/.test(el.dataset.target)) return;
      const dec=(el.dataset.target.split('.')[1]||'').length; let start=null; const dur=800+i*60;
      (function step(ts){ if(!start) start=ts; const p=Math.min((ts-start)/dur,1),e=1-Math.pow(1-p,3); el.textContent=(raw*e).toFixed(dec); if(p<1) requestAnimationFrame(step); })(performance.now());
    });
  });
}
/* best published baseline for a metric */
function bestPublished(ds,k){
  if(!BENCH||!BENCH.datasets[ds]) return null;
  let best=null; BENCH.datasets[ds].methods.forEach(m=>{ const v=m.scores[k]; if(isNum(v)&&(!best||v>best.v)) best={v,name:m.name,ref:m.ref}; });
  return best;
}
function renderKPI(){
  const T=testOf(D), el=document.getElementById('kpiGrid');
  const vs=(k)=>{ const b=bestPublished(D.dataset,k); if(!b||!T||!isNum(T[k])) return {}; const d=T[k]-b.v; return {delta:`${d>=0?'▲ +':'▼ '}${fx(d)} vs ${esc(b.name)} (${fx(b.v)})`,dn:d<0}; };
  const fl=D.tr.length?D.tr[D.tr.length-1]:null, f0=D.tr.length?D.tr[0]:null;
  const kp=[
    {ico:'🅱️',bg:'#dbeafe',col:C.blue,lbl:'Test BLEU-1',val:fx(T&&T.B1),...vs('B1')},
    {ico:'4️⃣',bg:'#d1fae5',col:C.green,lbl:'Test BLEU-4',val:fx(T&&T.B4),...vs('B4')},
    {ico:'🌠',bg:'#fef3c7',col:C.orange,lbl:'Test METEOR',val:fx(T&&T.M),sub:'baselines do not report it'},
    {ico:'🎯',bg:'#fce7f3',col:C.pink,lbl:'Test CIDEr',val:fx(T&&T.C),sub:'×100, as in the paper'},
    {ico:'📏',bg:'#fee2e2',col:C.red,lbl:'Test ROUGE-L',val:fx(T&&T.R),sub:T&&T.Rpaper?'from the paper (not logged)':T&&isNum(T.R)?'logged':'not logged for this run'},
    {ico:'⭐',bg:'#ede9fe',col:C.purple,lbl:'Tested Epoch',val:T?String(T.epoch):'–',sub:D.bestVal?`best validation BLEU-4: epoch ${D.bestVal.epoch} (${fx(D.bestVal.B4)})`:''},
    {ico:'📉',bg:'#cffafe',col:C.cyan,lbl:'Final Training Loss',val:fl?fx(fl.mean,3):'–',sub:fl&&f0?`from ${fx(f0.mean,3)} at epoch ${f0.epoch} (−${fx((1-fl.mean/f0.mean)*100,1)}%)`:'no training log'},
    {ico:'📚',bg:'#d1fae5',col:C.green,lbl:'Vocabulary',val:D.vocab_size?String(D.vocab_size):'–',sub:`words seen ≥ ${D.word_threshold} times`}
  ];
  el.innerHTML=kpiHTML(kp); countUp(el);
}
function srcText(){
  const T=D.test;
  let s=`<b>${esc(D.dataset_label)}</b>, word threshold ≥ <b>${D.word_threshold}</b>, batch size <b>${D.batch_size}</b>. `;
  s+=T?`Test figures come from the checkpoint the pipeline tested (epoch <b>${T.epoch}</b>); curves are per-epoch training loss and validation scores.`:'No test result was exported for this run.';
  if(D.paper_config) s+=' <span class="tag-paper">PAPER CONFIGURATION</span>';
  else s+=` <span class="tag-extra">not in the paper</span> The manuscript reports threshold ${PAPER_WT} only.`;
  if(D.duplicate_logs_of) s+=`<br><b>Note:</b> the training log in this folder is byte-identical to <b>${esc(D.duplicate_logs_of)}</b>, so it was not used; only the test result is shown.`;
  return s;
}

/* ══════════════════════════════════════════════════════════════
   6.  DASHBOARD
   ══════════════════════════════════════════════════════════════ */
function renderDashboard(){
  document.getElementById('srcNote').innerHTML=srcText();
  const te=D.testEpoch, T=testOf(D);
  if(D.tr.length){
    const ctx=document.getElementById('c-loss').getContext('2d');
    mkChart('c-loss','line',{labels:D.tr.map(r=>r.epoch),datasets:[
      {label:'Mean batch loss',data:D.tr.map(r=>r.mean),borderColor:C.blue,backgroundColor:grd(ctx,'rgba(37,99,235,.22)','rgba(37,99,235,0)'),borderWidth:2.5,fill:true,tension:.4,
       pointRadius:D.tr.map(r=>r.epoch===te?7:0),pointBackgroundColor:C.red,pointBorderColor:'#fff',pointBorderWidth:2}
    ]},{plugins:{legend:LEG,tooltip:{mode:'index',intersect:false,callbacks:{title:c=>`Epoch ${c[0].label}${+c[0].label===te?' · tested':''}`}}},scales:sc('Epoch','Cross-entropy loss')});
  } else emptyCanvas('c-loss','No training log for this run');

  if(T){
    const ks=['B1','B2','B3','B4','M','C','R'];
    const val=D.valAtTest;
    mkChart('c-testBar','bar',{labels:ks.map(k=>MLAB[k]),datasets:[
      {label:'Test',data:ks.map(k=>T[k]),backgroundColor:ks.map(k=>rgba(MCOL[k],.8)),borderRadius:6,borderSkipped:false},
      ...(val?[{label:`Validation (epoch ${val.epoch})`,data:ks.map(k=>val[k]),backgroundColor:ks.map(k=>rgba(MCOL[k],.25)),borderColor:ks.map(k=>MCOL[k]),borderWidth:1.2,borderRadius:6,borderSkipped:false}]:[])
    ]},{plugins:{legend:LEG,tooltip:{mode:'index',callbacks:{label:c=>`${c.dataset.label}: ${fx(c.parsed.y)}${c.label==='ROUGE-L'&&T.Rpaper&&c.datasetIndex===0?' (paper)':''}`}}},scales:sc('','Score')});
  } else emptyCanvas('c-testBar','No test result');

  if(D.val.length){
    const L=D.val.map(v=>v.epoch);
    mkChartNoAR('c-valBleu','line',{labels:L,datasets:['B1','B2','B3','B4'].map(k=>({label:MLAB[k],data:D.val.map(v=>v[k]),borderColor:MCOL[k],backgroundColor:'transparent',borderWidth:2,tension:.35,pointRadius:0}))},
      {plugins:{legend:LEG,tooltip:{mode:'index',intersect:false},epochMarker:{epoch:te}},scales:sc('Epoch','Score')},[epochMarker]);
    mkChartNoAR('c-valMC','line',{labels:L,datasets:[
      {label:'METEOR',data:D.val.map(v=>v.M),borderColor:C.orange,backgroundColor:'transparent',borderWidth:2.2,tension:.35,pointRadius:0},
      {label:'CIDEr',data:D.val.map(v=>v.C),borderColor:C.pink,backgroundColor:rgba(C.pink,.08),fill:true,borderWidth:2.2,tension:.35,pointRadius:0}
    ]},{plugins:{legend:LEG,tooltip:{mode:'index',intersect:false},epochMarker:{epoch:te}},scales:sc('Epoch','Score')},[epochMarker]);
  } else { emptyCanvas('c-valBleu','No validation log'); emptyCanvas('c-valMC','No validation log'); }

  if(T){
    const ks=['B1','B2','B3','B4','M','C'];
    const sets=[{label:'Test',data:ks.map(k=>T[k]),borderColor:C.green,backgroundColor:rgba(C.green,.15),borderWidth:2,pointRadius:3,pointBackgroundColor:C.green}];
    if(D.valAtTest) sets.push({label:`Validation · epoch ${D.valAtTest.epoch}`,data:ks.map(k=>D.valAtTest[k]),borderColor:C.blue,backgroundColor:rgba(C.blue,.1),borderWidth:2,pointRadius:3,pointBackgroundColor:C.blue});
    const mx=Math.ceil(Math.max(...sets.flatMap(s=>s.data).filter(isNum))/10)*10;
    mkChartNoAR('c-radar','radar',{labels:ks.map(k=>MLAB[k]),datasets:sets},{plugins:{legend:LEG,tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${fx(c.parsed.r)}`}}},
      scales:{r:{min:0,max:mx,angleLines:{color:'#e2e6f0'},grid:{color:'#e2e6f0'},pointLabels:{color:'#4b5580',font:{size:11}},ticks:{color:'#8892b0',stepSize:mx/5,backdropColor:'transparent'}}}});
  } else emptyCanvas('c-radar','No test result');
}

/* ══════════════════════════════════════════════════════════════
   7.  TRAINING
   ══════════════════════════════════════════════════════════════ */
function renderTraining(){
  const tr=D.tr, te=D.testEpoch;
  if(!tr.length){
    ['c-lossBand','c-batchLoss','c-lossDelta','c-lossB4'].forEach(id=>emptyCanvas(id,'No training log for this run'));
    document.getElementById('trainStats').innerHTML='<div class="muted">No training log was exported for this run.</div>'; return;
  }
  const L=tr.map(r=>r.epoch);
  mkChart('c-lossBand','line',{labels:L,datasets:[
    {label:'Max batch loss',data:tr.map(r=>r.max),borderColor:rgba(C.blue,.25),backgroundColor:rgba(C.blue,.1),borderWidth:1,pointRadius:0,fill:'+1',tension:.3},
    {label:'Min batch loss',data:tr.map(r=>r.min),borderColor:rgba(C.blue,.25),backgroundColor:'transparent',borderWidth:1,pointRadius:0,tension:.3},
    {label:'Mean ± σ',data:tr.map(r=>r.mean),borderColor:C.blue,borderWidth:2.5,pointRadius:tr.map(r=>r.epoch===te?6:0),pointBackgroundColor:C.red,tension:.3,backgroundColor:'transparent'}
  ]},{plugins:{legend:LEG,tooltip:{mode:'index',intersect:false,callbacks:{afterBody:c=>{const r=tr[c[0].dataIndex];return `σ ${fx(r.std,3)} · sum ${fx(r.loss_sum,1)}`;}}}},scales:sc('Epoch','Loss')});

  const nb=D.train.n_batches, f0=tr[0], fl=tr[tr.length-1];
  const minE=tr.reduce((b,r)=>r.mean<b.mean?r:b);
  const steps=nb*tr.length;
  document.getElementById('trainStats').innerHTML=`
    <div class="stat-rows">
      <table class="data-table">
        <tr><td>Batch size</td><td class="best">${D.batch_size}</td></tr>
        <tr><td>Optimisation steps per epoch</td><td>${nb.toLocaleString()}</td></tr>
        <tr><td>Total steps (${tr.length} epochs)</td><td class="best">${steps.toLocaleString()}</td></tr>
        <tr><td>Initial mean loss (epoch ${f0.epoch})</td><td>${fx(f0.mean,3)}</td></tr>
        <tr><td>Final mean loss (epoch ${fl.epoch})</td><td>${fx(fl.mean,3)}</td></tr>
        <tr><td>Lowest mean loss</td><td>${fx(minE.mean,3)} (epoch ${minE.epoch})</td></tr>
        <tr><td>Reduction</td><td class="best">${fx((1-fl.mean/f0.mean)*100,1)}%</td></tr>
        <tr><td>Tested epoch · loss</td><td>${te??'–'}${te!=null&&tr[te]?` · ${fx(tr[te].mean,3)}`:''}</td></tr>
        ${D.test&&isNum(D.test.tf_loss)?`<tr><td>Teacher-forced test loss</td><td>${fx(D.test.tf_loss,3)}</td></tr>`:''}
      </table>
    </div>`;

  const sl=document.getElementById('batchSlider'); sl.max=tr.length-1; sl.value=Math.min(te??0,tr.length-1);
  const drawB=()=>{
    const i=+sl.value, ep=tr[i], cur=D.train.batch_curves[i]||[];
    document.getElementById('batchSliderVal').textContent=`Epoch ${ep.epoch}${ep.epoch===te?' ★':''}`;
    const prev=i>0?D.train.batch_curves[i-1]:null;
    mkChart('c-batchLoss','line',{labels:cur.map((_,j)=>j+1),datasets:[
      {label:`Epoch ${ep.epoch}`,data:cur,borderColor:C.purple,backgroundColor:rgba(C.purple,.08),fill:true,borderWidth:1.6,pointRadius:0,tension:.25},
      ...(prev?[{label:`Epoch ${tr[i-1].epoch}`,data:prev,borderColor:rgba(C.slate,.35),borderWidth:1,pointRadius:0,borderDash:[4,3],tension:.25}]:[])
    ]},{animation:false,plugins:{legend:LEG,tooltip:{mode:'index',intersect:false}},scales:sc(D.train.batch_curve_bucketed?'Step (bucket-averaged)':'Step','Loss')});
    document.getElementById('batchNote').textContent=`Mean ${fx(ep.mean,3)} · σ ${fx(ep.std,3)} · first step ${fx(ep.first,3)} · last step ${fx(ep.last,3)} · ${nb} steps`+(D.train.batch_curve_bucketed?` (shown as ${cur.length} bucket means)`:'');
  };
  sl.oninput=drawB; drawB();

  const drop=tr.map((r,i)=>i?((tr[i-1].mean-r.mean)/tr[i-1].mean)*100:null);
  mkChart('c-lossDelta','bar',{labels:L,datasets:[{label:'Loss reduction vs previous epoch (%)',data:drop,backgroundColor:drop.map(v=>v>=0?rgba(C.orange,.7):rgba(C.red,.7)),borderRadius:4}]},
    {plugins:{legend:LEG,tooltip:{callbacks:{label:c=>`${fx(c.parsed.y,2)}%`}}},scales:sc('Epoch','%')});

  if(D.val.length){
    const V=Object.fromEntries(D.val.map(v=>[v.epoch,v]));
    const pts=tr.filter(r=>V[r.epoch]).map(r=>({x:r.mean,y:V[r.epoch].B4,e:r.epoch}));
    const r=pearson(pts.map(p=>p.x),pts.map(p=>p.y));
    mkChart('c-lossB4','scatter',{datasets:[{label:`Epochs (r = ${fx(r)})`,data:pts,backgroundColor:pts.map(p=>p.e===te?C.red:rgba(C.cyan,.6)),pointRadius:pts.map(p=>p.e===te?7:4)}]},
      {plugins:{legend:LEG,tooltip:{callbacks:{label:c=>`Epoch ${c.raw.e}: loss ${fx(c.raw.x,3)}, BLEU-4 ${fx(c.raw.y)}`}}},scales:sc('Mean training loss','Validation BLEU-4')});
  } else emptyCanvas('c-lossB4','No validation log');
}

/* ══════════════════════════════════════════════════════════════
   8.  EVALUATION
   ══════════════════════════════════════════════════════════════ */
function renderEvaluation(){
  const V=D.val, te=D.testEpoch, T=testOf(D);
  if(V.length){
    const L=V.map(v=>v.epoch), mk=(k,lbl)=>({label:lbl||MLAB[k],data:V.map(v=>v[k]),borderColor:MCOL[k],backgroundColor:'transparent',borderWidth:2,tension:.35,pointRadius:V.map(v=>v.epoch===te?5:0),pointBackgroundColor:MCOL[k]});
    const op=yl=>({plugins:{legend:LEG,tooltip:{mode:'index',intersect:false},epochMarker:{epoch:te}},scales:sc('Epoch',yl)});
    mkChart('c-evBleu','line',{labels:L,datasets:['B1','B2','B3','B4'].map(k=>mk(k))},op('BLEU'),[epochMarker]);
    mkChart('c-evMeteor','line',{labels:L,datasets:[{...mk('M'),fill:true,backgroundColor:rgba(C.orange,.1)}]},op('METEOR'),[epochMarker]);
    mkChart('c-evCider','line',{labels:L,datasets:[{...mk('C'),fill:true,backgroundColor:rgba(C.pink,.1)}]},op('CIDEr (×100)'),[epochMarker]);
    const sl=document.getElementById('evalSlider'); sl.max=V.length-1; sl.value=Math.max(0,V.findIndex(v=>v.epoch===te));
    sl.oninput=()=>renderEvalChips(+sl.value); renderEvalChips(+sl.value);
  } else ['c-evBleu','c-evMeteor','c-evCider'].forEach(id=>emptyCanvas(id,'No validation log'));

  /* test table */
  const tt=document.getElementById('testTable');
  if(T){
    const all=D.test_all||[D.test];
    tt.innerHTML=`<thead><tr><th>Checkpoint</th>${MET.map(m=>`<th>${m[1]}</th>`).join('')}<th>ROUGE-L</th><th>TF loss</th></tr></thead><tbody>${
      all.map(t=>{const tR=t.epoch===T.epoch?T:t;return `<tr class="${t.epoch===T.epoch?'hl':''}"><td>Epoch ${t.epoch}${t.epoch===T.epoch?' ⭐':''}</td>${MET.map(([k])=>`<td class="num">${fx(t[k])}</td>`).join('')}<td class="num">${fx(tR.R)}${tR.Rpaper?'<span class="dag">†</span>':''}</td><td class="num">${fx(t.tf_loss,3)}</td></tr>`;}).join('')
    }</tbody>`;
    document.getElementById('testNote').innerHTML=(T.Rpaper?'† ROUGE-L from the manuscript (the pipeline logged ROUGE as 0). ':'')+'CIDEr ×100. TF loss: teacher-forced cross-entropy on the test set.';
  } else { tt.innerHTML='<tr><td class="muted">No test result</td></tr>'; document.getElementById('testNote').textContent=''; }

  if(T&&D.valAtTest){
    const ks=['B1','B2','B3','B4','M','C'];
    mkChart('c-valTest','bar',{labels:ks.map(k=>MLAB[k]),datasets:[
      {label:`Validation · epoch ${te}`,data:ks.map(k=>D.valAtTest[k]),backgroundColor:rgba(C.blue,.7),borderRadius:6,borderSkipped:false},
      {label:'Test',data:ks.map(k=>T[k]),backgroundColor:rgba(C.green,.75),borderRadius:6,borderSkipped:false}
    ]},{plugins:{legend:LEG,tooltip:{mode:'index',callbacks:{afterBody:c=>{const k=ks[c[0].dataIndex];return `Gap (test − val): ${sgn(T[k]-D.valAtTest[k])}`;}}}},scales:sc('','Score')});
  } else emptyCanvas('c-valTest','Needs both validation and test results');

  /* top-10 */
  const t10=document.getElementById('top10Table');
  if(V.length){
    const top=[...V].sort((a,b)=>b.B4-a.B4).slice(0,10);
    const ptop=paperTop3();
    t10.innerHTML=`<thead><tr><th>#</th><th>Epoch</th>${MET.map(m=>`<th>${m[1]}</th>`).join('')}</tr></thead><tbody>${
      top.map((v,i)=>`<tr class="${v.epoch===te?'hl':''}"><td>${i+1}</td><td>${v.epoch}${v.epoch===te?' ⭐':''}${ptop&&ptop.includes(v.epoch)?` <span class="tag-paper">${esc(BENCH.datasets[D.dataset].val_table)}</span>`:''}</td>${MET.map(([k])=>`<td class="num${k==='B4'?' best':''}">${fx(v[k])}</td>`).join('')}</tr>`).join('')}</tbody>`;
  } else t10.innerHTML='<tr><td class="muted">No validation log</td></tr>';

  /* correlation heat */
  const ch=document.getElementById('corrHeat');
  if(V.length>3){
    const ks=['B1','B2','B3','B4','M','C'];
    const lossBy=Object.fromEntries(D.tr.map(r=>[r.epoch,r.mean]));
    const cols=[...ks.map(k=>[k,MLAB[k],V.map(v=>v[k])]),...(D.tr.length?[['L','Train loss',V.map(v=>lossBy[v.epoch])]]:[])];
    ch.innerHTML=`<table class="data-table compact heatmap-table"><thead><tr><th></th>${cols.map(c=>`<th>${c[1]}</th>`).join('')}</tr></thead><tbody>${
      cols.map(a=>`<tr><td><b>${a[1]}</b></td>${cols.map(b=>{const r=pearson(a[2],b[2]);return `<td class="heat-cell" style="background:${divColor(r)};color:${Math.abs(r||0)>.6?'#fff':'#1e2340'}">${fx(r)}</td>`;}).join('')}</tr>`).join('')}</tbody></table>`;
  } else ch.innerHTML='<div class="muted">Not enough epochs.</div>';
}
function paperTop3(){
  if(!BENCH||!D.paper_config) return null;
  const p=BENCH.datasets[D.dataset]&&BENCH.datasets[D.dataset].paper_val_top3[String(D.batch_size)];
  return p?p.map(x=>x.epoch):null;
}
function renderEvalChips(i){
  const v=D.val[i]; if(!v) return; const te=D.testEpoch;
  document.getElementById('evalSliderVal').textContent=`Epoch ${v.epoch}${v.epoch===te?' ★':''}`;
  const rank=[...D.val].sort((a,b)=>b.B4-a.B4).findIndex(x=>x.epoch===v.epoch)+1;
  const loss=D.tr.find(r=>r.epoch===v.epoch);
  document.getElementById('evalChips').innerHTML=[...MET.map(([k,l])=>[l,fx(v[k]),MCOL[k]]),['Rank by BLEU-4',`${rank} / ${D.val.length}`,C.slate],['Train loss',loss?fx(loss.mean,3):'–',C.cyan]]
    .map(([l,val,c])=>`<div class="chip"><div class="chip-l">${l}</div><div class="chip-v" style="color:${c}">${val}</div></div>`).join('');
}

/* ══════════════════════════════════════════════════════════════
   9.  CAPTIONS
   ══════════════════════════════════════════════════════════════ */
function hl(text){
  const t=String(text||'').split(/(\s+)/); let prev=null;
  return t.map(tok=>{ if(/^\s+$/.test(tok)) return tok; const r=prev!==null&&tok===prev; prev=tok; return r?`<mark class="rep">${esc(tok)}</mark>`:esc(tok); }).join('');
}
const glyph=img=>{ const h=[...String(img)].reduce((a,c)=>(a*31+c.charCodeAt(0))>>>0,7); const g=['🖼️','🏞️','🌆','🏙️','🌄','🌇','🎑','🗾']; return `<div class="cap-glyph" title="${esc(img)}">${g[h%g.length]}</div>`; };
const shortImg=img=>String(img).replace(/\.jpg$/,'');

function renderCaptions(){
  CAP_SHOWN=12;
  const n=D.caps.length;
  document.getElementById('capCount').textContent=n?`${n} images · epoch ${D.testEpoch}`:'';
  const sorts=[['file','By image'],['short','Shortest'],['long','Longest'],['rep','Repeats first']];
  if(D.hasRefs) sorts.splice(1,0,['best','Best BLEU-4'],['worst','Worst BLEU-4']);
  if(!sorts.find(s=>s[0]===CAP_SORT)) CAP_SORT='file';
  buildTabs('capSort',sorts,k=>{CAP_SORT=k;drawCaptionGrid(true);},CAP_SORT);
  drawCaptionGrid(true);

  /* evolution */
  const ev=D.evolution, tabs=document.getElementById('evoTabs');
  if(ev&&ev.images.length){
    tabs.innerHTML=ev.images.map((img,i)=>`<button class="evo-tab${i===0?' active':''}" onclick="setEvoImg('${esc(img)}',this)">${esc(shortImg(img).slice(-8))}</button>`).join('');
    setEvoImg(ev.images[0],tabs.querySelector('.evo-tab'));
  } else { tabs.innerHTML=''; document.getElementById('evoTimeline').innerHTML='<div class="muted">No validation caption dump (GeneratedPredictions.txt) for this run.</div>'; }

  /* length distribution */
  const st=D.test_captions&&D.test_captions.stats, rs=D.reference_stats;
  if(st){
    const bins=Array.from({length:24},(_,i)=>i+2);
    const pct=(s,b)=>{const h=s.len_hist||{};const n=Object.values(h).reduce((a,v)=>a+v,0)||1;return (h[String(b)]||0)/n*100;};
    mkChart('c-capDist','bar',{labels:bins,datasets:[
      {label:`Generated (mean ${fx(st.mean_len,1)})`,data:bins.map(b=>pct(st,b)),backgroundColor:rgba(C.purple,.55),borderColor:C.purple,borderWidth:1.2,borderRadius:3},
      ...(rs?[{label:`References (mean ${fx(rs.mean_len,1)})`,data:bins.map(b=>pct(rs,b)),backgroundColor:rgba(C.slate,.25),borderColor:C.slate,borderWidth:1.2,borderRadius:3}]:[])
    ]},{plugins:{legend:LEG,tooltip:{mode:'index',intersect:false}},scales:sc('Length (words)','Share of captions %')});
    const tw=(st.top_words||[]).slice(0,15);
    mkChart('c-topWords','bar',{labels:tw.map(w=>w[0]),datasets:[{label:'Occurrences in test captions',data:tw.map(w=>w[1]),backgroundColor:rgba(C.orange,.7),borderRadius:4}]},
      {indexAxis:'y',plugins:{legend:LEG},scales:{x:{grid:{color:GC}},y:{grid:{display:false},ticks:{font:{family:"'Noto Sans Devanagari','Outfit',sans-serif",size:12}}}}});
  } else { emptyCanvas('c-capDist','No test captions for this run'); emptyCanvas('c-topWords','No test captions for this run'); }

  /* quality per epoch */
  const q=D.val_caption_stats;
  const QK=[['repeat_pct','Word repeat %'],['loop_pct','Phrase repeat %'],['distinct1','Distinct-1 %'],['distinct2','Distinct-2 %'],['unique_pct','Unique captions %'],['mean_len','Mean length'],['vocab_used','Words used']];
  const drawQ=k=>{
    if(!q||!q.length){ emptyCanvas('c-capQual','No validation caption dump for this run'); return; }
    const lbl=QK.find(x=>x[0]===k)[1];
    mkChart('c-capQual','line',{labels:q.map(r=>r.epoch),datasets:[{label:lbl,data:q.map(r=>r[k]),borderColor:C.red,backgroundColor:rgba(C.red,.08),fill:true,borderWidth:2.2,tension:.3,
      pointRadius:q.map(r=>r.epoch===D.testEpoch?6:0),pointBackgroundColor:C.red}]},
      {plugins:{legend:LEG,tooltip:{mode:'index',intersect:false},epochMarker:{epoch:D.testEpoch}},scales:sc('Epoch',lbl)},[epochMarker]);
  };
  buildTabs('qualTabs',QK,drawQ,'repeat_pct'); drawQ('repeat_pct');
}
function capFilter(){
  const qs=(document.getElementById('capSearch').value||'').trim();
  let a=D.caps.filter(c=>!qs||c.cap.includes(qs)||c.img.includes(qs)||(c.refs||[]).some(r=>r.includes(qs)));
  const len=c=>c.cap.split(/\s+/).length, rep=c=>{const t=c.cap.split(/\s+/);return t.some((w,i)=>i&&w===t[i-1])?1:0;};
  if(CAP_SORT==='best') a=[...a].sort((x,y)=>(y.bleu4??-1)-(x.bleu4??-1));
  if(CAP_SORT==='worst') a=[...a].sort((x,y)=>(x.bleu4??999)-(y.bleu4??999));
  if(CAP_SORT==='short') a=[...a].sort((x,y)=>len(x)-len(y));
  if(CAP_SORT==='long') a=[...a].sort((x,y)=>len(y)-len(x));
  if(CAP_SORT==='rep') a=[...a].sort((x,y)=>rep(y)-rep(x)||len(y)-len(x));
  return a;
}
function drawCaptionGrid(reset){
  if(reset) CAP_SHOWN=12;
  const g=document.getElementById('capGrid');
  if(!D.caps.length){ g.innerHTML='<div class="muted">No test captions were exported for this run (only threshold-5 runs keep them).</div>'; document.getElementById('moreCaps').style.display='none'; return; }
  const a=capFilter();
  g.innerHTML=a.length?a.slice(0,CAP_SHOWN).map(c=>`
    <div class="cap-item">${glyph(c.img)}
      <div class="cap-body">
        <div class="cap-file">${esc(c.img)}${isNum(c.bleu4)?` <span class="cap-score">· BLEU-1 ${fx(c.bleu1,1)} · BLEU-4 ${fx(c.bleu4,1)}</span>`:''}</div>
        <div class="cap-line"><span class="cap-tag gen">gen</span><div class="cap-text">${hl(c.cap)}</div></div>
        ${(c.refs||[]).slice(0,3).map(r=>`<div class="cap-line"><span class="cap-tag gt">ref</span><div class="cap-gt">${esc(r)}</div></div>`).join('')}
      </div>
    </div>`).join(''):'<div class="muted">No caption matches the search.</div>';
  document.getElementById('moreCaps').style.display=CAP_SHOWN<a.length?'inline-flex':'none';
}
function moreCaptions(){ CAP_SHOWN+=12; drawCaptionGrid(false); }
window.moreCaptions=moreCaptions; window.drawCaptionGrid=drawCaptionGrid;
function setEvoImg(img,btn){
  document.querySelectorAll('.evo-tab').forEach(b=>b.classList.remove('active')); if(btn) btn.classList.add('active');
  const ev=D.evolution, caps=ev.captions[img]||[], te=D.testEpoch, refs=(ev.refs||{})[img]||[];
  let prev=null;
  document.getElementById('evoTimeline').innerHTML=`<div class="cap-item" style="margin-bottom:10px">${glyph(img)}<div class="cap-body"><div class="cap-file">${esc(img)}</div><div class="cap-note">Validation image · caption of every epoch · tested epoch highlighted · rows identical to the previous epoch are faded</div>
      ${refs.slice(0,2).map(r=>`<div class="cap-line"><span class="cap-tag gt">ref</span><div class="cap-gt">${esc(r)}</div></div>`).join('')}</div></div>`+
    ev.epochs.map((ep,i)=>{ const c=caps[i]; const same=c===prev; prev=c;
      return `<div class="evo-row${ep===te?' best-ep':''}" style="${same&&ep!==te?'opacity:.45':''}"><div class="evo-ep">E${ep}${ep===te?'★':''}</div><div class="evo-pair"><div class="evo-cap">${c!=null?hl(c):'–'}</div></div></div>`; }).join('');
}
window.setEvoImg=setEvoImg;

/* one test image, every batch size (paper Fig. 5 style) */
async function renderSame(){
  const g=document.getElementById('sameGrid');
  const ids=BATCHES.map(b=>findRun(D.dataset,D.word_threshold,b)).filter(r=>r&&r.availability&&r.availability.captions);
  if(ids.length<2){ g.innerHTML='<div class="muted">Test captions are exported for fewer than two batch sizes at this threshold.</div>'; document.getElementById('sameSrc').textContent=''; return; }
  const runs=await Promise.all(ids.map(r=>getRun(r.id)));
  const maps=runs.map(r=>Object.fromEntries(r.caps.map(c=>[c.img,c])));
  const common=Object.keys(maps[0]).filter(k=>maps.every(m=>m[k])).sort();
  if(!common.length){ g.innerHTML='<div class="muted">No common test images.</div>'; return; }
  document.getElementById('sameSrc').textContent=`${common.length} common images`;
  const pick=[]; for(let i=0;i<4&&i<common.length;i++) pick.push(common[(SAME_SEED*7+i*Math.floor(common.length/4)+SAME_SEED)%common.length]);
  g.innerHTML=pick.map(img=>{
    const refs=(maps.find(m=>m[img]&&m[img].refs)||{})[img];
    return `<div class="cap-item">${glyph(img)}<div class="cap-body"><div class="cap-file">${esc(img)}</div>
      ${runs.map((r,i)=>`<div class="cap-line"><span class="cap-tag b${r.batch_size}">B${r.batch_size}</span><div class="cap-text">${hl(maps[i][img].cap)}</div></div>`).join('')}
      ${refs&&refs.refs?refs.refs.slice(0,2).map(t=>`<div class="cap-line"><span class="cap-tag gt">ref</span><div class="cap-gt">${esc(t)}</div></div>`).join(''):''}
    </div></div>`;}).join('');
}
function shuffleSame(){ SAME_SEED++; renderSame(); }
window.shuffleSame=shuffleSame;

/* ══════════════════════════════════════════════════════════════
   10.  WORD THRESHOLD
   ══════════════════════════════════════════════════════════════ */
let WT_KEY='B4';
function renderThreshold(){
  const ds='flickr8k';
  const R=RUNS.filter(r=>r.dataset===ds);
  const wts=[...new Set(R.map(r=>r.word_threshold))].sort((a,b)=>a-b);
  const note=document.getElementById('wtNote'), body=document.getElementById('wtBody');
  if(wts.length<2){ note.innerHTML='Only one word threshold has been exported, so there is nothing to compare.'; body.style.display='none'; return; }
  body.style.display='';
  const bs=D.dataset===ds?D.batch_size:256;
  note.innerHTML=`The vocabulary keeps only Hindi words seen at least <i>threshold</i> times in the training captions. The paper uses <b>≥ ${PAPER_WT}</b> on both datasets; the Flickr 8K-Hindi runs below were also trained at thresholds ${wts.filter(w=>w!==PAPER_WT).join(', ')}, which are <b>not reported in the manuscript</b>. ${D.dataset!==ds?'This study exists for Flickr 8K-Hindi only; batch size 256 is shown in the per-batch charts.':''}`;
  const T=(w,b)=>{const r=R.find(x=>x.word_threshold===w&&x.batch_size===b);return r?testOf(r):null;};

  /* KPIs: best configuration per metric */
  const all=R.filter(r=>r.test);
  const best=k=>all.reduce((a,r)=>r.test[k]>a.test[k]?r:a);
  const kp=[['B1','🅱️','#dbeafe',C.blue],['B4','4️⃣','#d1fae5',C.green],['M','🌠','#fef3c7',C.orange],['C','🎯','#fce7f3',C.pink]].map(([k,ico,bg,col])=>{
    const b=best(k), p=all.filter(r=>r.word_threshold===PAPER_WT).reduce((a,r)=>r.test[k]>a.test[k]?r:a);
    return {ico,bg,col,lbl:`Best test ${MLAB[k]}`,val:fx(b.test[k]),sub:`threshold ${b.word_threshold} · Batch-${b.batch_size}`,delta:`${sgn(b.test[k]-p.test[k])} vs best paper run (threshold ${PAPER_WT}, B${p.batch_size}: ${fx(p.test[k])})`,dn:false};
  });
  const kg=document.getElementById('wtKpi'); kg.innerHTML=kpiHTML(kp); countUp(kg);

  const draw=k=>{
    WT_KEY=k;
    mkChart('c-wtLine','line',{labels:wts.map(w=>`≥ ${w}`),datasets:BATCHES.map(b=>({label:`Batch-${b}`,data:wts.map(w=>{const t=T(w,b);return t?t[k]:null;}),borderColor:BCOL[b],backgroundColor:BCOL[b],borderWidth:2.2,tension:.25,pointRadius:5,spanGaps:true}))},
      {plugins:{legend:LEG,tooltip:{mode:'index',intersect:false}},scales:sc('Word threshold',`Test ${MLAB[k]}`)});
    document.getElementById('wtHeatLbl').textContent=`Test ${MLAB[k]}`;
    const vals=wts.flatMap(w=>BATCHES.map(b=>{const t=T(w,b);return t?t[k]:null;})).filter(isNum);
    const lo=Math.min(...vals),hi=Math.max(...vals);
    document.getElementById('wtHeat').innerHTML=`<table class="data-table compact heatmap-table"><thead><tr><th>Threshold</th><th>Vocab</th>${BATCHES.map(b=>`<th>B${b}</th>`).join('')}</tr></thead><tbody>${
      wts.map(w=>{const vv=(R.find(r=>r.word_threshold===w)||{}).vocab_size;return `<tr><td><b>≥ ${w}</b>${w===PAPER_WT?' <span class="tag-paper">PAPER</span>':''}</td><td class="num">${vv??'–'}</td>${BATCHES.map(b=>{const t=T(w,b),v=t?t[k]:null,s=isNum(v)?(v-lo)/((hi-lo)||1):0;
        return `<td class="heat-cell" style="background:${isNum(v)?heatColor(s):'#fff'};color:${ink(s)}">${fx(v)}</td>`;}).join('')}</tr>`;}).join('')}</tbody></table>`;
  };
  buildTabs('wtTabs',MET,draw,WT_KEY); draw(WT_KEY);

  const vocab=wts.map(w=>(R.find(r=>r.word_threshold===w)||{}).vocab_size);
  mkChart('c-vocab','bar',{labels:wts.map(w=>`≥ ${w}`),datasets:[{label:'Vocabulary size',data:vocab,backgroundColor:wts.map(w=>rgba(WCOL[w]||C.blue,.75)),borderRadius:6}]},
    {plugins:{legend:LEG,tooltip:{callbacks:{label:c=>`${c.parsed.y} words`}}},scales:sc('Word threshold','Words')});

  document.getElementById('wtValLbl').textContent=`Batch-${bs}`; document.getElementById('wtLossLbl').textContent=`Batch-${bs}`;
  const rs=wts.map(w=>R.find(r=>r.word_threshold===w&&r.batch_size===bs)).filter(Boolean);
  const maxE=Math.max(...rs.map(r=>(r.val||[]).length),0);
  mkChart('c-wtVal','line',{labels:Array.from({length:maxE},(_,i)=>i),datasets:rs.filter(r=>r.val&&r.val.length).map(r=>({label:`≥ ${r.word_threshold}`,data:r.val.map(v=>v.B4),borderColor:WCOL[r.word_threshold],backgroundColor:'transparent',borderWidth:r.word_threshold===PAPER_WT?2.6:1.6,pointRadius:0,tension:.3}))},
    {plugins:{legend:LEG,tooltip:{mode:'index',intersect:false}},scales:sc('Epoch','Validation BLEU-4')});
  const maxL=Math.max(...rs.map(r=>(r.loss||[]).length),0);
  mkChart('c-wtLoss','line',{labels:Array.from({length:maxL},(_,i)=>i),datasets:rs.filter(r=>r.loss&&r.loss.length).map(r=>({label:`≥ ${r.word_threshold}`,data:r.loss,borderColor:WCOL[r.word_threshold],backgroundColor:'transparent',borderWidth:r.word_threshold===PAPER_WT?2.6:1.6,pointRadius:0,tension:.3}))},
    {plugins:{legend:LEG,tooltip:{mode:'index',intersect:false}},scales:sc('Epoch','Mean training loss')});

  document.getElementById('wtTable').innerHTML=`<thead><tr><th>Threshold</th><th>Batch</th><th>Vocab</th><th>Epoch</th>${MET.map(m=>`<th>${m[1]}</th>`).join('')}<th>TF loss</th><th>Notes</th></tr></thead><tbody>${
    [...R].sort((a,b)=>a.word_threshold-b.word_threshold||a.batch_size-b.batch_size).map(r=>{const t=r.test||{};const cur=D&&r.id===D.id;
      return `<tr class="${cur?'hl':''}"><td>≥ ${r.word_threshold}${r.paper_config?' <span class="tag-paper">PAPER</span>':''}</td><td>B${r.batch_size}</td><td class="num">${r.vocab_size??'–'}</td><td class="num">${t.epoch??'–'}</td>${MET.map(([k])=>{const isB=all.length&&best(k).id===r.id;return `<td class="num${isB?' best':''}">${fx(t[k])}</td>`;}).join('')}<td class="num">${fx(t.tf_loss,3)}</td><td class="muted">${r.duplicate_logs_of?'logs duplicate '+esc(r.duplicate_logs_of):''}</td></tr>`;}).join('')}</tbody>`;
}

/* ══════════════════════════════════════════════════════════════
   11.  BENCHMARKS
   ══════════════════════════════════════════════════════════════ */
let B_DS=null, B_BS=null, B_MET='B1';
async function renderBench(){
  let bm; try{ bm=await getBench(); }catch(e){ document.getElementById('benchNote').innerHTML='<b>data/benchmarks.json</b> was not found.'; return; }
  if(!B_DS) B_DS=D.dataset;
  buildTabs('benchDsTabs',Object.keys(bm.datasets).map(k=>[k,bm.datasets[k].label]),k=>{B_DS=k;B_BS=null;renderBench();},B_DS);
  const dsb=bm.datasets[B_DS];
  const runs=BATCHES.map(b=>findRun(B_DS,PAPER_WT,b)).filter(r=>r&&r.test);
  if(!B_BS||!runs.find(r=>r.batch_size===+B_BS)){
    const best=runs.reduce((a,r)=>!a||r.test.B4>a.test.B4?r:a,null);
    B_BS=(D.dataset===B_DS&&D.word_threshold===PAPER_WT)?D.batch_size:best?best.batch_size:BATCHES[0];
  }
  buildTabs('benchBatchTabs',runs.map(r=>[r.batch_size,`DHARA B = ${r.batch_size}`]),k=>{B_BS=+k;renderBench();},B_BS);
  const run=runs.find(r=>r.batch_size===+B_BS); const T=testOf(run);
  document.getElementById('benchNote').innerHTML=`Published Hindi captioning results on <b>${esc(dsb.label)}</b> (${esc(dsb.table)} of the paper) against DHARA at word threshold ${PAPER_WT}, <b>batch size ${B_BS}</b>, tested checkpoint epoch ${T?T.epoch:'–'}. DHARA values are read from the exported test results; the baselines are copied from the paper.`;
  document.getElementById('benchTblLbl').textContent=`${dsb.label} · ${dsb.table}`;

  const cols=bm.metrics;
  const pub=dsb.methods.map(m=>({name:m.name,sub:`${m.authors} [${m.ref}]`,kind:'pub',vals:m.scores}));
  const ours=runs.map(r=>({name:`DHARA · Batch-${r.batch_size}`,kind:+r.batch_size===+B_BS?'sel':'dh',vals:testOf(r)}));
  const rows=[...pub,...ours];
  const top={}; cols.forEach(c=>top[c.key]=Math.max(...rows.map(r=>r.vals[c.key]).filter(isNum)));

  /* KPIs */
  const kp=[]; ['B1','B2','B3','B4'].forEach((k,i)=>{ const b=bestPublished(B_DS,k); if(!b||!T) return; const d=T[k]-b.v;
    kp.push({ico:['🅱️','2️⃣','3️⃣','4️⃣'][i],bg:d>=0?'#d1fae5':'#fee2e2',col:d>=0?C.green:C.red,lbl:`${MLAB[k]} · DHARA B${B_BS}`,val:fx(T[k]),delta:`${d>=0?'▲ +':'▼ '}${fx(d)} vs ${esc(b.name)} [${b.ref}] (${fx(b.v)})`,dn:d<0}); });
  const kg=document.getElementById('benchKpi'); kg.innerHTML=kpiHTML(kp); countUp(kg);

  document.getElementById('benchTable').innerHTML=`<thead><tr><th>Author(s)</th><th>Model</th>${cols.map(c=>`<th>${esc(c.label)}</th>`).join('')}</tr></thead><tbody>${
    rows.map(r=>`<tr class="${r.kind==='sel'?'hl':r.kind==='dh'?'sub-row':''}"><td>${r.kind==='pub'?esc(r.sub):'<b>DHARA</b> (ours)'}</td><td>${r.kind==='pub'?esc(r.name):`<b>${esc(r.name.replace('DHARA · ',''))}</b>`}</td>${
      cols.map(c=>{const v=r.vals[c.key];const isTop=isNum(v)&&Math.abs(v-top[c.key])<1e-9;return `<td class="num${isTop?' best':''}">${isNum(v)?fx(v,c.key==='R'?3:2):'<span class="muted">–</span>'}${c.key==='R'&&r.kind!=='pub'&&r.vals.Rpaper?'<span class="dag">†</span>':''}</td>`;}).join('')}</tr>`).join('')}</tbody>`;
  document.getElementById('benchNotes').innerHTML=(bm.notes||[]).map(n=>`<li>${esc(n)}</li>`).join('')+'<li>Bold: best value in the column. Highlighted row: the DHARA batch size selected above. † ROUGE-L from the paper.</li>';

  /* bar by method */
  const drawBar=k=>{
    B_MET=k; const R=rows.filter(r=>isNum(r.vals[k])).sort((a,b)=>b.vals[k]-a.vals[k]);
    mkChartNoAR('c-benchBar','bar',{labels:R.map(r=>r.kind==='pub'?r.name:`${r.name.replace('DHARA · ','DHARA ')}${r.kind==='sel'?' ★':''}`),datasets:[{label:MLAB[k],data:R.map(r=>r.vals[k]),
      backgroundColor:R.map(r=>r.kind==='pub'?rgba(C.slate,.45):r.kind==='sel'?rgba(C.blue,.95):rgba(C.blue,.45)),borderRadius:4,borderSkipped:false}]},
      {indexAxis:'y',plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${MLAB[k]}: ${fx(c.parsed.x)}`}}},scales:{x:{grid:{color:GC},min:0},y:{grid:{display:false},ticks:{autoSkip:false}}}});
  };
  buildTabs('benchMetricTabs',[['B1','BLEU-1'],['B2','BLEU-2'],['B3','BLEU-3'],['B4','BLEU-4']],drawBar,B_MET); drawBar(B_MET);

  /* margin */
  const G=['B1','B2','B3','B4'];
  mkChart('c-benchGain','bar',{labels:G.map(k=>MLAB[k]),datasets:runs.map(r=>{const t=testOf(r);return {label:`Batch-${r.batch_size}`,data:G.map(k=>{const b=bestPublished(B_DS,k);return b?t[k]-b.v:null;}),
      backgroundColor:rgba(BCOL[r.batch_size],+r.batch_size===+B_BS?.9:.4),borderRadius:5,borderSkipped:false};})},
    {plugins:{legend:LEG,tooltip:{callbacks:{label:c=>{const k=G[c.dataIndex],b=bestPublished(B_DS,k);return `${c.dataset.label}: ${sgn(c.parsed.y)} vs ${b.name} (${fx(b.v)})`;}}}},
     scales:{x:{grid:{display:false}},y:{title:{display:true,text:'DHARA − best published'},grid:{color:GC}}}});

  document.getElementById('benchMethods').innerHTML=`<thead><tr><th>Model</th><th>Authors</th><th>Ref.</th><th>Approach</th><th>BLEU-1 / BLEU-4</th></tr></thead><tbody>${
    dsb.methods.map(m=>`<tr><td><b>${esc(m.name)}</b></td><td>${esc(m.authors)}</td><td>[${esc(m.ref)}]${m.ref_note?` <span class="tag-extra" title="${esc(m.ref_note)}">[? ] in tables</span>`:''}</td><td>${esc(m.approach)}</td><td class="num">${fx(m.scores.B1)} / ${fx(m.scores.B4)}</td></tr>`).join('')}
    <tr class="hl"><td><b>DHARA (ours)</b></td><td>Sharma &amp; Jasrotia</td><td>–</td><td>Frozen CLIP ViT-B/32 patch tokens + frozen FastText-Hindi with an MLP projection, cross-attention fusion and a transformer decoder; beam search (width 5)</td><td class="num"><b>${T?`${fx(T.B1)} / ${fx(T.B4)}`:'–'}</b></td></tr></tbody>`;

  renderChecks(bm);
  [60,350].forEach(ms=>setTimeout(()=>['c-benchBar','c-benchGain'].forEach(id=>CHS[id]&&CHS[id].resize()),ms));
}
window.renderBench=renderBench;

/* compares the paper's tables with the exported data */
async function renderChecks(bm){
  const out=[];
  const close=(a,b,tol)=>isNum(a)&&isNum(b)&&Math.abs(a-b)<=tol;
  for(const [ds,info] of Object.entries(bm.datasets)){
    /* test tables */
    const bad=[], missing=[];
    for(const [bs,p] of Object.entries(info.paper_dhara)){
      const r=findRun(ds,PAPER_WT,+bs); if(!r||!r.test){ missing.push(bs); continue; }
      MET.forEach(([k,l])=>{ if(!close(Math.round(r.test[k]*100)/100,p[k],0.011)) bad.push(`B${bs} ${l}: paper ${fx(p[k])}, data ${fx(r.test[k])}`); });
    }
    if(missing.length) out.push(['warn',`${info.table} (${info.label}): no exported test result for batch ${missing.join(', ')}.`]);
    else out.push(bad.length?['bad',`${info.table} (${info.label}) differs from the data: ${bad.join('; ')}.`]:['ok',`${info.table} (${info.label}): every DHARA test value (BLEU-1…4, METEOR, CIDEr) matches the exported test results.`]);

    /* validation top-3 tables */
    for(const [bs,rows] of Object.entries(info.paper_val_top3)){
      const r=findRun(ds,PAPER_WT,+bs); if(!r||!r.val||!r.val.length) continue;
      const V=Object.fromEntries(r.val.map(v=>[v.epoch,v]));
      const vb=[]; rows.forEach(p=>{ const v=V[p.epoch]; if(!v){vb.push(`epoch ${p.epoch} missing`);return;} MET.forEach(([k,l])=>{ if(!close(Math.round(v[k]*100)/100,p[k],0.011)) vb.push(`epoch ${p.epoch} ${l}: paper ${fx(p[k])}, data ${fx(v[k])}`); }); });
      const top=[...r.val].sort((a,b)=>b.B4-a.B4).slice(0,3).map(v=>v.epoch);
      const pe=rows.map(p=>p.epoch);
      const sameSet=top.every(e=>pe.includes(e));
      if(vb.length) out.push(['bad',`${info.val_table} B${bs}: ${vb.join('; ')}.`]);
      else if(!sameSet){
        const miss=top.filter(e=>!pe.includes(e)).map(e=>`epoch ${e} (BLEU-4 ${fx(V[e].B4)})`);
        out.push(['warn',`${info.val_table} B${bs}: the listed epochs ${pe.join(', ')} have the right values, but ranked by validation BLEU-4 the top three are ${top.join(', ')}; the table leaves out ${miss.join(' and ')}. The tested checkpoint is epoch ${r.test?r.test.epoch:'–'}.`]);
      }
      else out.push(['ok',`${info.val_table} B${bs}: top-3 validation epochs (${pe.join(', ')}) and their values match the data.`]);
    }
    /* tested checkpoint = best validation BLEU-4? */
    BATCHES.forEach(bs=>{ const r=findRun(ds,PAPER_WT,bs); if(!r||!r.test||!r.best_val) return;
      if(r.test.epoch!==r.best_val.epoch) out.push(['warn',`${info.label} B${bs}: the tested checkpoint (epoch ${r.test.epoch}, validation BLEU-4 ${fx((r.val.find(v=>v.epoch===r.test.epoch)||{}).B4)}) is not the epoch with the highest validation BLEU-4 (epoch ${r.best_val.epoch}, ${fx(r.best_val.B4)}). Section 5.1 says checkpoints are chosen by validation performance.`]); });
    if(info.methods.some(m=>m.ref_note)) out.push(['warn',`${info.table}: ChitraVivran is cited as “Sharma et al. [? ]” — a broken reference in the LaTeX source (it is [37] in the introduction).`]);
  }
  /* human */
  try{
    const h=await getHuman(); const hp=bm.human_paper;
    Object.entries(h).forEach(([ds,v])=>{ const p=hp[ds]; if(!p) return;
      const ok=close(Math.round(v.fluency.mean*100)/100,p.fluency,0.011)&&close(Math.round(v.adequacy.mean*100)/100,p.adequacy,0.011);
      out.push(ok?['ok',`${hp.table} (${bm.datasets[ds].label}): fluency ${fx(p.fluency)} and adequacy ${fx(p.adequacy)} match the rating sheet (${v.n_responses} ratings of ${v.n_images} images).`]
                 :['bad',`${hp.table} (${bm.datasets[ds].label}): paper ${fx(p.fluency)} / ${fx(p.adequacy)}, sheet ${fx(v.fluency.mean)} / ${fx(v.adequacy.mean)}.`]);
      const claimed=ds==='pascal1k'?100:100;
      if(v.n_images<claimed) out.push(['warn',`Section 5.5 (${bm.datasets[ds].label}): the text says ${claimed} images were rated; the sheet holds ratings for ${v.n_images} distinct images (${v.n_responses} responses).`]);
    });
  }catch(e){}
  /* ROUGE */
  out.push(['warn','ROUGE-L in Tables 1–4 cannot be checked: the training and test logs store ROUGE as 0.']);
  document.getElementById('checkList').innerHTML=out.map(([c,t])=>`<li class="${c}"><span class="ck">${c==='ok'?'✓':c==='warn'?'!':'✗'}</span><span>${t}</span></li>`).join('');
}

/* ══════════════════════════════════════════════════════════════
   12.  HUMAN EVALUATION
   ══════════════════════════════════════════════════════════════ */
let H_DS='flickr8k', H_EX='best';
async function renderHuman(){
  let h; try{ h=await getHuman(); }catch(e){ document.getElementById('humanNote').innerHTML='<b>data/human_eval.json</b> was not found. Run the exporter with <code>--human</code>.'; return; }
  const ds=Object.keys(h); const lab=k=>(BENCH&&BENCH.datasets[k]?BENCH.datasets[k].label:k);
  if(D&&h[D.dataset]) H_DS=D.dataset;
  document.getElementById('humanNote').innerHTML=`Hindi-proficient raters scored captions from the best batch configuration of each dataset on a 5-point Likert scale: <b>fluency</b> (does the Hindi read naturally?) and <b>adequacy</b> (does it describe the image?). ${ds.map(k=>`${lab(k)}: ${h[k].n_responses} ratings of ${h[k].n_images} images`).join(' · ')}. Only aggregates are published; rater names stay in the spreadsheet.`;
  const kp=[]; ds.forEach(k=>{ const v=h[k];
    kp.push({ico:'🗣️',bg:'#dbeafe',col:C.blue,lbl:`${lab(k)} · Fluency`,val:fx(v.fluency.mean),sub:`σ ${fx(v.fluency.std)} · ${v.n_responses} ratings`});
    kp.push({ico:'🎯',bg:'#d1fae5',col:C.green,lbl:`${lab(k)} · Adequacy`,val:fx(v.adequacy.mean),sub:`σ ${fx(v.adequacy.std)} · r(F,A) = ${fx(v.corr)}`}); });
  const kg=document.getElementById('humanKpi'); kg.innerHTML=kpiHTML(kp); countUp(kg);
  const L=['1','2','3','4','5'], DC={pascal1k:C.purple,flickr8k:C.orange};
  const dist=(k,f)=>{const hh=h[k][f].hist,n=Object.values(hh).reduce((a,b)=>a+b,0)||1;return L.map(x=>(hh[x]||0)/n*100);};
  ['fluency','adequacy'].forEach((f,i)=>mkChart(i?'c-hAde':'c-hFlu','bar',{labels:['1 · very poor','2','3','4','5 · excellent'].map((x,j)=>f==='adequacy'?['1 · unrelated','2','3','4','5 · fully accurate'][j]:x),
    datasets:ds.map(k=>({label:`${lab(k)} (mean ${fx(h[k][f].mean)})`,data:dist(k,f),backgroundColor:rgba(DC[k]||C.blue,.7),borderRadius:5}))},
    {plugins:{legend:LEG,tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${fx(c.parsed.y,1)}%`}}},scales:sc('Score','Share of ratings %')}));
  const drawJ=k=>{ H_DS=k; const j=h[k].joint, mx=Math.max(...j.map(x=>x[2]));
    mkChart('c-hJoint','bubble',{datasets:[{label:lab(k),data:j.map(([f,a,c])=>({x:f,y:a,r:4+16*Math.sqrt(c/mx),c})),backgroundColor:rgba(DC[k]||C.blue,.5),borderColor:DC[k]||C.blue}]},
      {plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`Fluency ${c.raw.x}, adequacy ${c.raw.y}: ${c.raw.c} rating${c.raw.c>1?'s':''}`}}},
       scales:{x:{min:.5,max:5.5,ticks:{stepSize:1},title:{display:true,text:'Fluency'},grid:{color:GC}},y:{min:.5,max:5.5,ticks:{stepSize:1},title:{display:true,text:'Adequacy'},grid:{color:GC}}}}); };
  buildTabs('hJointTabs',ds.map(k=>[k,lab(k)]),drawJ,H_DS); drawJ(H_DS);
  mkChart('c-hImg','scatter',{datasets:ds.map(k=>({label:lab(k),data:h[k].per_image.map(([f,a])=>({x:f,y:a})),backgroundColor:rgba(DC[k]||C.blue,.55),pointRadius:5}))},
    {plugins:{legend:LEG},scales:{x:{min:.8,max:5.2,title:{display:true,text:'Mean fluency'},grid:{color:GC}},y:{min:.8,max:5.2,title:{display:true,text:'Mean adequacy'},grid:{color:GC}}}});
  const drawEx=key=>{ const [k,which]=key.split('|'); H_EX=key;
    const ex=[...h[k].examples].sort((a,b)=>which==='best'?(b.F+b.A)-(a.F+a.A):(a.F+a.A)-(b.F+b.A)).slice(0,6);
    document.getElementById('hExGrid').innerHTML=ex.map(x=>`<div class="cap-item">${glyph(x.img)}<div class="cap-body"><div class="cap-file">${esc(x.img)} <span class="cap-score">· fluency ${fx(x.F,1)} · adequacy ${fx(x.A,1)}</span></div><div class="cap-line"><span class="cap-tag gen">gen</span><div class="cap-text">${hl(x.cap)}</div></div></div></div>`).join(''); };
  const exKeys=ds.flatMap(k=>[[`${k}|best`,`${lab(k)} · highest`],[`${k}|worst`,`${lab(k)} · lowest`]]);
  const ek=exKeys.find(e=>e[0]===H_EX)?H_EX:exKeys[0][0];
  buildTabs('hExTabs',exKeys,drawEx,ek); drawEx(ek);
}

/* ══════════════════════════════════════════════════════════════
   13.  BATCH COMPARISON
   ══════════════════════════════════════════════════════════════ */
let CMP_KEY='B1';
function renderCompare(){
  const ds=D.dataset, wt=D.word_threshold;
  const rs=BATCHES.map(b=>findRun(ds,wt,b)).filter(Boolean);
  document.getElementById('cmpNote').innerHTML=`All batch sizes of <b>${esc(D.dataset_label)}</b> at word threshold ≥ <b>${wt}</b>${wt===PAPER_WT?' (paper configuration)':''}. Each run trains for 40 epochs, so smaller batches take more optimisation steps.`;
  document.getElementById('cmpTblLbl').textContent=wt===PAPER_WT&&BENCH?`${BENCH.datasets[ds].val_table} & ${BENCH.datasets[ds].table}`:`threshold ≥ ${wt}`;
  const bestBy=k=>rs.filter(r=>r.test).reduce((a,r)=>!a||r.test[k]>a.test[k]?r:a,null);
  document.getElementById('cmpTable').innerHTML=`<thead><tr><th>Batch</th><th>Steps / epoch</th><th>Total steps</th><th>Final loss</th><th>Best val. BLEU-4 (epoch)</th><th>Tested epoch</th>${MET.map(m=>`<th>Test ${m[1]}</th>`).join('')}<th>ROUGE-L</th></tr></thead><tbody>${
    rs.map(r=>{const t=testOf(r)||{},cur=r.id===D.id;return `<tr class="${cur?'hl':''}"><td><b>B${r.batch_size}</b></td><td class="num">${r.n_batches?r.n_batches.toLocaleString():'–'}</td><td class="num">${r.n_batches?(r.n_batches*r.n_epochs).toLocaleString():'–'}</td><td class="num">${r.loss&&r.loss.length?fx(r.loss[r.loss.length-1],3):'–'}</td><td class="num">${r.best_val?`${fx(r.best_val.B4)} (${r.best_val.epoch})`:'–'}</td><td class="num">${t.epoch??'–'}</td>${
      MET.map(([k])=>{const b=bestBy(k);return `<td class="num${b&&b.id===r.id?' best':''}">${fx(t[k])}</td>`;}).join('')}<td class="num">${fx(t.R,3)}${t.Rpaper?'<span class="dag">†</span>':''}</td></tr>`;}).join('')}</tbody>`;

  const lr=rs.filter(r=>r.loss&&r.loss.length), mL=Math.max(...lr.map(r=>r.loss.length),0);
  if(lr.length) mkChart('c-cmpLoss','line',{labels:Array.from({length:mL},(_,i)=>i),datasets:lr.map(r=>({label:`Batch ${r.batch_size}`,data:r.loss,borderColor:BCOL[r.batch_size],backgroundColor:'transparent',borderWidth:r.id===D.id?2.8:1.8,pointRadius:0,tension:.3}))},
    {plugins:{legend:LEG,tooltip:{mode:'index',intersect:false}},scales:sc('Epoch','Average loss')});
  else emptyCanvas('c-cmpLoss','No training logs');

  const sr=rs.filter(r=>r.n_batches&&r.test);
  if(sr.length) mkChart('c-cmpSteps','bar',{labels:sr.map(r=>`B${r.batch_size}`),datasets:[
      {label:'Total optimisation steps',data:sr.map(r=>r.n_batches*r.n_epochs),backgroundColor:sr.map(r=>rgba(BCOL[r.batch_size],.55)),borderRadius:6,yAxisID:'y'},
      {label:'Test BLEU-4',type:'line',data:sr.map(r=>r.test.B4),borderColor:C.red,backgroundColor:C.red,pointRadius:6,borderWidth:2,yAxisID:'y1'}
    ]},{plugins:{legend:LEG,tooltip:{mode:'index'}},scales:{x:{grid:{display:false}},y:{title:{display:true,text:'Steps'},grid:{color:GC}},y1:{position:'right',title:{display:true,text:'BLEU-4'},grid:{display:false}}}});
  else emptyCanvas('c-cmpSteps','No data');

  const figs={pascal1k:{B1:'6a',B2:'6b',B3:'7a',B4:'7b',M:'8a',C:'8b'},flickr8k:{B1:'9a',B2:'9b',B3:'10a',B4:'10b',M:'11a',C:'11b'}};
  const drawV=k=>{ CMP_KEY=k; const vr=rs.filter(r=>r.val&&r.val.length), mE=Math.max(...vr.map(r=>r.val.length),0);
    document.getElementById('cmpFigLbl').textContent=wt===PAPER_WT&&figs[ds]?`Paper Fig. ${figs[ds][k]}`:`threshold ≥ ${wt}`;
    mkChart('c-cmpVal','line',{labels:Array.from({length:mE},(_,i)=>i),datasets:vr.map(r=>({label:`Batch ${r.batch_size}`,data:r.val.map(v=>v[k]),borderColor:BCOL[r.batch_size],backgroundColor:'transparent',borderWidth:r.id===D.id?2.8:1.8,pointRadius:0,tension:.3}))},
      {plugins:{legend:LEG,tooltip:{mode:'index',intersect:false}},scales:sc('Epoch',`Validation ${MLAB[k]}`)}); };
  buildTabs('cmpTabs',MET,drawV,CMP_KEY); drawV(CMP_KEY);

  const tr=rs.filter(r=>r.test); const ks=['B1','B2','B3','B4','M','C','R'];
  mkChart('c-cmpTest','bar',{labels:ks.map(k=>MLAB[k]),datasets:tr.map(r=>{const t=testOf(r);return {label:`Batch ${r.batch_size}`,data:ks.map(k=>t[k]),backgroundColor:rgba(BCOL[r.batch_size],r.id===D.id?.9:.55),borderRadius:5,borderSkipped:false};})},
    {plugins:{legend:LEG,tooltip:{mode:'index'}},scales:sc('','Test score')});
}

/* ══════════════════════════════════════════════════════════════
   14.  INSIGHTS
   ══════════════════════════════════════════════════════════════ */
function renderInsights(){
  const ins=[], T=testOf(D);
  if(T&&D.bestVal) ins.push({ico:'⭐',bg:'#ede9fe',title:`Tested Checkpoint: Epoch ${T.epoch}`,desc:T.epoch===D.bestVal.epoch
    ?`The tested checkpoint is also the epoch with the highest validation BLEU-4 (${fx(D.bestVal.B4)}). On the test set it reaches BLEU-4 ${fx(T.B4)}, a gap of ${sgn(T.B4-D.bestVal.B4)}.`
    :`The tested checkpoint has validation BLEU-4 ${fx(D.valAtTest&&D.valAtTest.B4)}, while the best validation BLEU-4 is ${fx(D.bestVal.B4)} at epoch ${D.bestVal.epoch}. Test BLEU-4 is ${fx(T.B4)}.`});
  if(T&&BENCH){
    const ks=['B1','B2','B3','B4']; const w=ks.filter(k=>{const b=bestPublished(D.dataset,k);return b&&T[k]>b.v;});
    const l=ks.filter(k=>!w.includes(k)).map(k=>{const b=bestPublished(D.dataset,k);return `${MLAB[k]} (${fx(T[k])} vs ${b.name} ${fx(b.v)})`;});
    ins.push({ico:'🏆',bg:'#fef3c7',title:'Against Published Baselines',desc:`This run beats the strongest published result on ${w.length} of 4 BLEU scores${w.length?` (${w.map(k=>MLAB[k]).join(', ')})`:''}.${l.length?` It trails on ${l.join(', ')}.`:''} Baselines report no METEOR or CIDEr.`});
  }
  if(D.tr.length&&D.val.length){
    const f0=D.tr[0],fl=D.tr[D.tr.length-1]; const pk=Math.max(...D.val.map(v=>v.B1)); const e90=D.val.find(v=>v.B1>=.9*pk);
    ins.push({ico:'📉',bg:'#cffafe',title:'Convergence',desc:`Mean loss falls from ${fx(f0.mean,2)} to ${fx(fl.mean,2)} (−${fx((1-fl.mean/f0.mean)*100,0)}%) over ${D.tr.length} epochs of ${D.train.n_batches} steps. Validation BLEU-1 reaches 90% of its peak (${fx(pk)}) by epoch ${e90?e90.epoch:'–'}.`});
    const last=D.val[D.val.length-1];
    ins.push({ico:'🧭',bg:'#fee2e2',title:'Late-Training Behaviour',desc:`Validation BLEU-4 peaks at ${fx(D.bestVal.B4)} (epoch ${D.bestVal.epoch}) and ends at ${fx(last.B4)} (epoch ${last.epoch}), ${sgn(last.B4-D.bestVal.B4)} from the peak, while training loss keeps falling — ${last.B4<D.bestVal.B4-0.5?'a sign of mild over-fitting to the training captions.':'no clear sign of over-fitting.'}`});
  }
  const rs=BATCHES.map(b=>findRun(D.dataset,D.word_threshold,b)).filter(r=>r&&r.test);
  if(rs.length>1){ const b=rs.reduce((a,r)=>r.test.B4>a.test.B4?r:a), c=rs.reduce((a,r)=>r.test.C>a.test.C?r:a);
    ins.push({ico:'⚖️',bg:'#dbeafe',title:'Batch Size',desc:`At this threshold the best test BLEU-4 comes from Batch-${b.batch_size} (${fx(b.test.B4)}) and the best CIDEr from Batch-${c.batch_size} (${fx(c.test.C)}). ${D.dataset==='pascal1k'?'With only 800 training images, small batches give many more updates in 40 epochs.':'With ~6.5k training images, larger batches get enough updates and benefit from steadier gradients.'}`}); }
  const st=D.test_captions&&D.test_captions.stats;
  if(st) ins.push({ico:'🔤',bg:'#fce7f3',title:'Caption Quality',desc:`Test captions average ${fx(st.mean_len,1)} words${D.reference_stats?` (references: ${fx(D.reference_stats.mean_len,1)})`:''}, use ${st.vocab_used} distinct words${D.vocab_size?` of a ${D.vocab_size}-word vocabulary`:''}, and ${fx(st.unique_pct,0)}% are unique. ${fx(st.repeat_pct,1)}% contain a word repeated back-to-back and ${fx(st.loop_pct,1)}% repeat a phrase.`});
  if(D.dataset==='flickr8k'){
    const fr=RUNS.filter(r=>r.dataset==='flickr8k'&&r.batch_size===D.batch_size&&r.test);
    if(fr.length>1){ const b=fr.reduce((a,r)=>r.test.B4>a.test.B4?r:a), p=fr.find(r=>r.word_threshold===PAPER_WT);
      ins.push({ico:'🧮',bg:'#d1fae5',title:'Word Threshold',desc:`For Batch-${D.batch_size}, the best test BLEU-4 is at threshold ${b.word_threshold} (${fx(b.test.B4)}, vocabulary ${b.vocab_size})${p?` against ${fx(p.test.B4)} at the paper's threshold ${PAPER_WT} (vocabulary ${p.vocab_size})`:''}. These extra runs are not in the manuscript.`}); }
  }
  const draw=()=>{ document.getElementById('insightGrid').innerHTML=ins.map(x=>`<div class="insight"><div class="ins-ico" style="background:${x.bg}">${x.ico}</div><div class="ins-title">${x.title}</div><div class="ins-desc">${x.desc}</div></div>`).join(''); };
  draw();
  const forRun=D.id;
  getHuman().then(h=>{ if(D.id!==forRun||!h[D.dataset]) return; const v=h[D.dataset];
    ins.push({ico:'🧑‍⚖️',bg:'#ede9fe',title:'Human Judgement',desc:`On ${esc(D.dataset_label)}, raters gave mean fluency ${fx(v.fluency.mean)} and adequacy ${fx(v.adequacy.mean)} out of 5 (${v.n_responses} ratings). Fluency is higher than adequacy: the Hindi usually reads well even when the content is wrong.`}); draw(); }).catch(()=>{});

  document.getElementById('archSvg').innerHTML=archSVG();
  const dm=D.dataset_meta||{};
  document.getElementById('archTable').innerHTML=`
    <table class="data-table">
      <tr><td>Task</td><td class="best">Hindi image captioning (low-resource)</td></tr>
      <tr><td>Dataset</td><td>${esc(D.dataset_label)}${dm.n_train?` · ${dm.n_train} train / ${dm.n_val} val / ${dm.n_test} test images`:''}${dm.avg_len?` · average reference length ${dm.avg_len} words`:''}</td></tr>
      <tr><td>Visual encoder</td><td>CLIP ViT-B/32, frozen · 224×224 input · 49 patches + CLS → 50×768 → linear → 50×512</td></tr>
      <tr><td>Text encoder</td><td>FastText-Hindi 300-d, frozen · MLP 300→512 (GELU, LayerNorm) · sinusoidal positions</td></tr>
      <tr><td>Fusion</td><td>Multi-head cross-attention: caption tokens query the 50 visual tokens · residual + LayerNorm</td></tr>
      <tr><td>Decoder</td><td>Transformer: masked self-attention, cross-attention over V, FFN 512→2048→512 · softmax over the vocabulary</td></tr>
      <tr><td>Training</td><td>Sparse categorical cross-entropy · Adam · per-epoch learning-rate decay 0.3 · dropout 0.2 · max caption length 32 · ${D.tr.length||D.val.length} epochs</td></tr>
      <tr><td>Inference</td><td>Beam search, width 5</td></tr>
      <tr><td>Word threshold · vocabulary</td><td class="best">≥ ${D.word_threshold} · ${D.vocab_size??'–'} words</td></tr>
      <tr><td>Batch size · steps per epoch</td><td class="best">${D.batch_size} · ${D.train?D.train.n_batches:'–'}</td></tr>
      <tr><td>Tested checkpoint</td><td class="best">${T?`epoch ${T.epoch} · BLEU-4 ${fx(T.B4)} · CIDEr ${fx(T.C)}`:'–'}</td></tr>
      <tr><td>Exported</td><td class="muted">${esc(D.exported||'')}</td></tr>
    </table>`;
}
function archSVG(){
  const bx=(x,y,w,h,cls,title,sub,frz)=>`<rect class="bx ${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="10"/>
    <text class="k" x="${x+w/2}" y="${y+(sub?h/2-4:h/2+4)}" text-anchor="middle">${title}</text>${sub?`<text class="s" x="${x+w/2}" y="${y+h/2+12}" text-anchor="middle">${sub}</text>`:''}${frz?`<text class="frz" x="${x+w-8}" y="${y+12}" text-anchor="end">FROZEN</text>`:''}`;
  const ar=(x1,y1,x2,y2)=>`<path class="ar" d="M${x1},${y1} L${x2},${y2}"/>`;
  const arp=d=>`<path class="ar" d="${d}"/>`;
  return `<svg viewBox="0 0 1000 300" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="DHARA architecture">
    <defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#8892b0"/></marker></defs>
    ${bx(10,30,120,56,'v','Image','224 × 224 × 3')}
    ${bx(160,30,150,56,'v','CLIP ViT-B/32','12 layers · 50 × 768',1)}
    ${bx(340,30,130,56,'v','Linear 768→512','V ∈ ℝ⁵⁰ˣ⁵¹²')}
    ${bx(10,200,120,56,'t','Hindi caption','tokens w₁…w_t')}
    ${bx(160,200,150,56,'t','FastText-Hindi','300-d subword',1)}
    ${bx(340,200,130,56,'t','MLP + PE','T ∈ ℝᵗˣ⁵¹²')}
    ${bx(510,110,150,64,'f','Cross-attention','Q = T · K, V = V')}
    ${bx(700,110,160,64,'d','Transformer decoder','self-attn · cross-attn · FFN')}
    ${bx(890,110,100,64,'d','Softmax','beam = 5')}
    ${ar(130,58,158,58)}${ar(310,58,338,58)}${ar(130,228,158,228)}${ar(310,228,338,228)}
    ${arp('M470,58 C490,58 495,125 508,128')}${arp('M470,228 C490,228 495,160 508,156')}
    ${ar(660,142,698,142)}${ar(860,142,888,142)}
    ${arp('M470,48 C620,20 760,40 780,108')}
    <text class="s" x="640" y="34" text-anchor="middle">visual tokens reused by every decoder layer</text>
    <text class="s" x="585" y="200" text-anchor="middle">F_d ∈ ℝᵗˣ⁵¹²</text>
  </svg>`;
}

/* ══════════════════════════════════════════════════════════════
   15.  RENDER ALL & INIT
   ══════════════════════════════════════════════════════════════ */
function renderAll(){
  B_DS=null; B_BS=null;
  [renderKPI,renderDashboard,renderTraining,renderEvaluation,renderCaptions,renderInsights].forEach(f=>{ try{ f(); }catch(e){ console.error(f.name,e); } });
  const act=document.querySelector('.page.active'); const name=act?act.id.replace('page-',''):'';
  if(name==='compare') renderCompare(); if(name==='bench') renderBench(); if(name==='threshold') renderThreshold(); if(name==='human') renderHuman(); if(name==='captions') renderSame();
}
document.addEventListener('DOMContentLoaded',()=>{ initTabs(); scanRuns(); });
