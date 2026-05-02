/* ============================================================
   DHARA Analytics Dashboard — dashboard.js  v3.0
   Author: Dr. Himanshu Sharma
   https://www.linkedin.com/in/dr-himanshusharma/
   ============================================================ */
'use strict';

/* ── GitHub config ─────────────────────────────────────────── */
const GH = {
  owner  : 'himanshusharma-csit',
  repo   : 'dhara',
  branch : 'main',
  root   : 'SavedResources'
};
function rawUrl(path) {
  return `https://raw.githubusercontent.com/${GH.owner}/${GH.repo}/${GH.branch}/${path}`;
}

/* ── Required files per batch ──────────────────────────────── */
const REQ = [
  'TrainingLogs.xlsx',
  'PredictionSplit.xlsx',
  'TestResults.xlsx',
  'TestCaptions.txt',
  'GeneratedPredictions.txt'
];

/* ── Global state ──────────────────────────────────────────── */
let D   = null;   // active dataset
let CHS = {};     // chart instances
let SEL = null;   // selected batch folder name

/* ══════════════════════════════════════════════════════════════
   1.  BATCH MODAL
   ══════════════════════════════════════════════════════════════ */
function setStatus(msg, type='loading') {
  const bar = document.getElementById('statusBar');
  const txt = document.getElementById('statusText');
  const spn = document.getElementById('statusSpinner');
  bar.className = 'status-bar' + (type==='ok'?' ok':type==='err'?' err':'');
  txt.textContent = msg;
  if (spn) spn.className = 'status-spinner' + (type==='loading'?'':' hide');
}

async function scanBatches() {
  const list = document.getElementById('batchList');
  list.innerHTML = '';
  document.getElementById('extraInfo').innerHTML = '';
  document.getElementById('btnLoad').disabled = true;
  SEL = null;

  setStatus('Loading batches from GitHub…', 'loading');

  try {
    /* Fetch batches.json — plain raw URL, no API, no rate limit */
    const url = rawUrl('batches.json') + '?nocache=' + Date.now();
    const res = await fetch(url);

    if (!res.ok) {
      setStatus('❌ batches.json not found in repo root.', 'err');
      document.getElementById('extraInfo').innerHTML = `
        <div style="background:var(--bl);border:1px solid rgba(37,99,235,.2);border-radius:10px;padding:14px;font-size:12px;color:var(--t2);line-height:1.7;margin-bottom:12px">
          <strong style="color:var(--blue)">📋 Create batches.json in your repo root:</strong><br>
          <code style="background:var(--s3);padding:6px 10px;border-radius:6px;display:block;font-family:monospace;font-size:11px;margin-top:6px">["Batch-128","Batch-256"]</code>
          <button class="retry-btn" onclick="scanBatches()" style="margin-top:10px">↺ Retry</button>
        </div>`;
      return;
    }

    const names = await res.json();
    if (!Array.isArray(names) || names.length === 0) {
      setStatus('⚠ batches.json is empty.', 'err');
      return;
    }

    setStatus(`Found ${names.length} batch${names.length>1?'es':''} — checking files…`, 'loading');

    /* Check each batch folder via HEAD requests (no API) */
    const results = await Promise.all(names.map(async name => {
      const checks = await Promise.all(REQ.map(async file => {
        try {
          const r = await fetch(rawUrl(`${GH.root}/${name}/${file}`) + '?nocache=' + Date.now(), { method:'HEAD' });
          return { file, ok: r.ok };
        } catch(_) { return { file, ok: false }; }
      }));
      return { name, checks };
    }));

    results.forEach(({ name, checks }) => addBatchCard(list, name, checks));

    const ready = results.filter(r => r.checks.every(c=>c.ok)).length;
    setStatus(`✓ ${names.length} batch${names.length>1?'es':''} found · ${ready} fully ready`, 'ok');

  } catch(err) {
    console.error(err);
    setStatus('⚠ Network error — check your connection.', 'err');
    document.getElementById('extraInfo').innerHTML =
      `<button class="retry-btn" onclick="scanBatches()">↺ Retry</button>`;
  }
}

function addBatchCard(list, name, checks) {
  const allOk = checks.every(c => c.ok);
  const div = document.createElement('div');
  div.className = 'batch-opt' + (!allOk ? ' bad' : '');
  div.dataset.name = name;
  const dots = checks.map(c =>
    `<span class="bdot ${c.ok?'ok':'no'}" title="${c.file}"></span>`
  ).join('');
  const ready = checks.filter(c=>c.ok).length;
  div.innerHTML = `
    <div class="batch-opt-ico">📁</div>
    <div style="flex:1;min-width:0">
      <div class="batch-opt-name">${name}
        <span style="font-size:10px;font-weight:700;color:${allOk?'var(--blue)':'var(--red)'}">
          ${allOk?'⬤ Ready':'⚠ '+ready+'/'+REQ.length+' files'}
        </span>
      </div>
      <div class="batch-opt-desc">SavedResources/${name} · ${GH.owner}/${GH.repo}</div>
      <div class="batch-dots">${dots}</div>
    </div>
    <div class="chk">✓</div>`;
  if (allOk) {
    div.addEventListener('click', () => {
      document.querySelectorAll('.batch-opt').forEach(o=>o.classList.remove('sel'));
      div.classList.add('sel');
      SEL = name;
      document.getElementById('btnLoad').disabled = false;
    });
  }
  list.appendChild(div);
}

async function loadSelectedBatch() {
  if (!SEL) return;
  const btn   = document.getElementById('btnLoad');
  const prog  = document.getElementById('fetchProgress') || document.createElement('div');
  const bar   = document.getElementById('fetchProgressBar') || document.createElement('div');
  const loadSt= document.getElementById('loadState') || document.createElement('div');
  btn.disabled = true;
  loadSt.style.display = 'flex';
  setStatus(`Fetching files for ${SEL}…`, 'loading');

  try {
    const step = 100 / REQ.length;

    /* Fetch all 5 files */
    async function getFile(filename, asText) {
      const url = rawUrl(`${GH.root}/${SEL}/${filename}`);
      const r = await fetch(url);
      if (!r.ok) throw new Error(`Cannot fetch ${filename} (HTTP ${r.status})`);
      if (asText) return r.text();
      return r.arrayBuffer();
    }

    bar.style.width = '10%';
    const tlBuf  = await getFile('TrainingLogs.xlsx', false);   bar.style.width='30%';
    const psBuf  = await getFile('PredictionSplit.xlsx', false); bar.style.width='50%';
    const trBuf  = await getFile('TestResults.xlsx', false);     bar.style.width='65%';
    const tcText = await getFile('TestCaptions.txt', true);      bar.style.width='80%';
    const gpText = await getFile('GeneratedPredictions.txt', true); bar.style.width='95%';

    /* Parse */
    D = parseAllFiles(tlBuf, psBuf, trBuf, tcText, gpText, SEL);
    bar.style.width = '100%';

    /* Close modal and render */
    setTimeout(() => {
      document.getElementById('overlay').style.display = 'none';
      document.getElementById('batchLabel').textContent =
        `${D.batchName} · Flickr 8K-Hindi · ${D.training.length} Epochs`;
      document.getElementById('cpBadge').textContent =
        `Epoch ${D.testResult.epoch} — Best Checkpoint`;
      renderAll();
    }, 300);

  } catch(err) {
    console.error(err);
    setStatus('❌ ' + err.message, 'err');
    btn.disabled = false;
    loadSt.style.display = 'none';
  }
}
window.loadSelectedBatch = loadSelectedBatch;
window.scanBatches = scanBatches;

/* ══════════════════════════════════════════════════════════════
   2.  FILE PARSERS
   ══════════════════════════════════════════════════════════════ */
function xlsxRows(buf) {
  const wb = XLSX.read(buf, { type:'array' });
  const ws = wb.Sheets[wb.SheetNames[0]];
  return XLSX.utils.sheet_to_json(ws, { header:1, defval:null });
}

function parseAllFiles(tlBuf, psBuf, trBuf, tcText, gpText, batchName) {
  /* TrainingLogs */
  const tlRows = xlsxRows(tlBuf);
  const training = [];
  for (let i=1;i<tlRows.length;i++) {
    const r = tlRows[i];
    if (r[0]==null) continue;
    training.push({
      epoch: parseInt(r[0]),
      epochLoss: parseFloat(r[1]),
      batchLosses: r.slice(2).filter(v=>v!=null).map(Number)
    });
  }

  /* PredictionSplit */
  const psRows = xlsxRows(psBuf);
  const validation = [];
  for (let i=1;i<psRows.length;i++) {
    const r = psRows[i];
    if (r[0]==null) continue;
    validation.push({
      epoch:parseInt(r[0]), bleu1:r[1], bleu2:r[2], bleu3:r[3], bleu4:r[4],
      meteor:r[5], rouge:r[6]||0, cider:r[7]
    });
  }

  /* TestResults */
  const trRows = xlsxRows(trBuf);
  const tr = trRows[1];
  const testResult = {
    epoch:parseInt(tr[0]), batch:parseInt(tr[1]), tfLoss:tr[2],
    bleu1:tr[3], bleu2:tr[4], bleu3:tr[5], bleu4:tr[6],
    meteor:tr[7], rouge:tr[8]||0, cider:tr[9]
  };

  /* TestCaptions */
  const tcRaw = JSON.parse(tcText);
  const testCaptions = Object.entries(tcRaw).slice(0,20).map(([img,v])=>({img,cap:v[0]}));

  /* GeneratedPredictions */
  const sep = /[-]{80,}\s*\n\s*Epoch Number (\d+)\s*\n\s*[-]{80,}\s*\n/g;
  const parts = gpText.split(sep).slice(1);
  const epData = {};
  for (let i=0;i<parts.length;i+=2) {
    const ep = parseInt(parts[i]);
    const block = parts[i+1].replace(/\n[-]{80,}[\s\S]*$/,'').trim();
    try { epData[ep] = JSON.parse(block); } catch(_) {}
  }
  const epochs = Object.keys(epData).map(Number).sort((a,b)=>a-b);
  const fixImgs = Object.keys(epData[epochs[0]]||{}).slice(0,5);

  const capEvo = {};
  fixImgs.forEach(img => {
    capEvo[img] = {};
    epochs.forEach(ep => { if(epData[ep]&&epData[ep][img]) capEvo[img][String(ep)]=epData[ep][img][0]; });
  });

  const capLens = {};
  epochs.forEach(ep => {
    if(epData[ep]) capLens[ep]=Object.values(epData[ep]).map(v=>v[0].split(' ').length);
  });

  const keyEps=[0,5,10,15,20,25,30,35,39];
  const sampleLengths={};
  keyEps.forEach(ep=>{ sampleLengths[String(ep)]=capLens[ep]||[]; });

  const lengthStats=epochs.map(ep=>{
    const ls=capLens[ep]||[];
    if(!ls.length) return null;
    const mean=ls.reduce((a,b)=>a+b)/ls.length;
    const std=Math.sqrt(ls.reduce((s,v)=>s+(v-mean)**2,0)/ls.length);
    return{epoch:ep,mean:parseFloat(mean.toFixed(3)),std:parseFloat(std.toFixed(3)),min:Math.min(...ls),max:Math.max(...ls)};
  }).filter(Boolean);

  return { batchName, training, validation, testResult, testCaptions, captionEvolution:capEvo, sampleLengths, lengthStats };
}

/* ══════════════════════════════════════════════════════════════
   3.  CHART HELPERS
   ══════════════════════════════════════════════════════════════ */
Chart.defaults.color='#8892b0';
Chart.defaults.borderColor='#e2e6f0';
Chart.defaults.font.family="'Outfit',sans-serif";
Chart.defaults.font.size=11;

const GC='#e8edf5';
function sc(xl,yl){
  return{
    x:{title:{display:!!xl,text:xl,color:'#8892b0'},grid:{color:GC},ticks:{color:'#8892b0'}},
    y:{title:{display:!!yl,text:yl,color:'#8892b0'},grid:{color:GC},ticks:{color:'#8892b0'}}
  };
}
function grd(ctx,c1,c2,h=300){
  const g=ctx.createLinearGradient(0,0,0,h);g.addColorStop(0,c1);g.addColorStop(1,c2);return g;
}
function mkChart(id,type,data,opts){
  if(CHS[id]) CHS[id].destroy();
  const el=document.getElementById(id);
  if(!el) return null;
  CHS[id]=new Chart(el,{type,data,options:{responsive:true,maintainAspectRatio:true,...opts}});
  return CHS[id];
}
function mkChartNoAR(id,type,data,opts){
  if(CHS[id]) CHS[id].destroy();
  const el=document.getElementById(id);
  if(!el) return null;
  CHS[id]=new Chart(el,{type,data,options:{responsive:true,maintainAspectRatio:false,...opts}});
  return CHS[id];
}
function pearson(x,y){
  const n=x.length,mx=x.reduce((a,b)=>a+b)/n,my=y.reduce((a,b)=>a+b)/n;
  const num=x.reduce((s,xi,i)=>s+(xi-mx)*(y[i]-my),0);
  const dx=Math.sqrt(x.reduce((s,xi)=>s+(xi-mx)**2,0));
  const dy=Math.sqrt(y.reduce((s,yi)=>s+(yi-my)**2,0));
  return(dx&&dy)?num/(dx*dy):0;
}
function kde(data,bw,pts){
  return pts.map(x=>data.reduce((s,xi)=>s+Math.exp(-.5*((x-xi)/bw)**2),0)/(data.length*bw*Math.sqrt(2*Math.PI)));
}
function heatColor(t){
  const stops=[[255,255,255],[255,237,213],[249,115,22],[185,28,28]];
  const n=stops.length-1,si=Math.min(Math.floor(t*n),n-1),f=t*n-si;
  const[r1,g1,b1]=stops[si],[r2,g2,b2]=stops[si+1]||stops[si];
  return`rgb(${Math.round(r1+(r2-r1)*f)},${Math.round(g1+(g2-g1)*f)},${Math.round(b1+(b2-b1)*f)})`;
}
function corrColor(r){
  if(r>=0) return`rgb(${Math.round(255-r*120)},${Math.round(255-r*170)},${Math.round(255-r*120)})`;
  return`rgb(${Math.round(255+r*120)},${Math.round(255+r*120)},255)`;
}

/* ══════════════════════════════════════════════════════════════
   4.  NAVIGATION
   ══════════════════════════════════════════════════════════════ */
const TITLES={dashboard:'Dashboard — Overview',training:'Training Logs',evaluation:'Evaluation Metrics',captions:'Hindi Captions',advanced:'Advanced Analytics',insights:'Key Insights'};

function goPage(name,btn){
  if(window.innerWidth<=900) closeSb();
  const prev=document.querySelector('.page.active');
  if(prev){ prev.classList.remove('active'); void prev.offsetWidth; }
  const next=document.getElementById('page-'+name);
  next.classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  if(btn){
    btn.classList.add('active');
    const ico=btn.querySelector('.nav-ico');
    if(ico){ico.style.transform='scale(1.35)';setTimeout(()=>ico.style.transform='',220);}
  }
  document.getElementById('pageTitle').textContent=TITLES[name]||name;
}
window.goPage=goPage;

function toggleSb(){
  const sb=document.getElementById('sidebar');
  const ov=document.getElementById('sbOverlay');
  const hb=document.getElementById('hamburger');
  const open=sb.classList.contains('open');
  if(open){closeSb();}else{sb.classList.add('open');ov.classList.add('show');hb.classList.add('open');document.body.style.overflow='hidden';}
}
function closeSb(){
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sbOverlay').classList.remove('show');
  document.getElementById('hamburger').classList.remove('open');
  document.body.style.overflow='';
}
window.toggleSb=toggleSb; window.closeSb=closeSb;
window.addEventListener('resize',()=>{ if(window.innerWidth>900) closeSb(); });
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeSb(); });

/* ══════════════════════════════════════════════════════════════
   5.  TABS
   ══════════════════════════════════════════════════════════════ */
function initTabs(){
  document.querySelectorAll('.tabs').forEach(bar=>{
    bar.querySelectorAll('.tab').forEach(tab=>{
      tab.addEventListener('click',function(){
        bar.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
        this.classList.add('active');
        const pane=this.dataset.pane;
        const card=bar.closest('.card,.page');
        if(!card) return;
        card.querySelectorAll('[id]').forEach(el=>{
          if([...bar.querySelectorAll('[data-pane]')].some(t=>t.dataset.pane===el.id))
            el.style.display=el.id===pane?'block':'none';
        });
      });
    });
  });
}

/* ══════════════════════════════════════════════════════════════
   6.  KPI GRID
   ══════════════════════════════════════════════════════════════ */
function renderKPI(){
  const T=D.testResult,V=D.validation,V0=V[0];
  const pB1=Math.max(...V.map(v=>v.bleu1)),pMt=Math.max(...V.map(v=>v.meteor));
  const kpis=[
    {ico:'📝',bg:'#dbeafe',col:'#2563eb',lbl:'Test BLEU-1',   val:T.bleu1.toFixed(2),delta:`▲ +${(T.bleu1-V0.bleu1).toFixed(2)} vs Epoch 0`},
    {ico:'🎯',bg:'#ede9fe',col:'#7c3aed',lbl:'Test BLEU-4',   val:T.bleu4.toFixed(2),delta:`▲ +${(T.bleu4-V0.bleu4).toFixed(2)} vs Epoch 0`},
    {ico:'🌟',bg:'#d1fae5',col:'#059669',lbl:'Test METEOR',   val:T.meteor.toFixed(2),delta:`▲ +${(T.meteor-V0.meteor).toFixed(2)} vs Epoch 0`},
    {ico:'💡',bg:'#fef3c7',col:'#d97706',lbl:'Test CIDEr',    val:T.cider.toFixed(4),delta:`▲ +${(T.cider-V0.cider).toFixed(4)} vs Epoch 0`},
    {ico:'📉',bg:'#fee2e2',col:'#dc2626',lbl:'Final Epoch Loss',val:D.training[D.training.length-1].epochLoss.toFixed(1),delta:`▼ ${(100-D.training[D.training.length-1].epochLoss/D.training[0].epochLoss*100).toFixed(1)}% reduction`,dn:true},
    {ico:'🔥',bg:'#cffafe',col:'#0891b2',lbl:'Peak Val BLEU-1',val:pB1.toFixed(2),sub:'Validation set'},
    {ico:'⚡',bg:'#fce7f3',col:'#db2777',lbl:'Peak Val METEOR',val:pMt.toFixed(2),sub:'Validation set'},
    {ico:'🧪',bg:'#dbeafe',col:'#2563eb',lbl:'TF Loss · Best Epoch',val:T.tfLoss.toFixed(3),sub:'Teacher-forced loss'},
  ];
  document.getElementById('kpiGrid').innerHTML=kpis.map(k=>`
    <div class="kpi">
      <div class="kpi-ico" style="background:${k.bg}">${k.ico}</div>
      <div>
        <div class="kpi-lbl">${k.lbl}</div>
        <div class="kpi-val" style="color:${k.col}" data-target="${k.val}">${k.val}</div>
        ${k.delta?`<div class="kpi-delta ${k.dn?'dn':'up'}">${k.delta}</div>`:''}
        ${k.sub?`<div class="kpi-sub">${k.sub}</div>`:''}
      </div>
    </div>`).join('');
  /* Count-up animation */
  requestAnimationFrame(()=>{
    document.querySelectorAll('.kpi-val').forEach((el,i)=>{
      const raw=parseFloat(el.dataset.target);
      if(isNaN(raw)) return;
      const dec=(el.dataset.target.split('.')[1]||'').length;
      let start=null;
      const dur=800+i*60;
      (function step(ts){
        if(!start) start=ts;
        const p=Math.min((ts-start)/dur,1);
        const e=1-Math.pow(1-p,3);
        el.textContent=(raw*e).toFixed(dec);
        if(p<1) requestAnimationFrame(step);
      })(performance.now());
    });
  });
}

/* ══════════════════════════════════════════════════════════════
   7.  DASHBOARD PAGE
   ══════════════════════════════════════════════════════════════ */
function renderDashboard(){
  const T=D.testResult, Vt=D.validation[D.testResult.epoch];
  /* Epoch loss */
  const elCtx=document.getElementById('c-epochLoss').getContext('2d');
  mkChart('c-epochLoss','line',{
    labels:D.training.map(d=>d.epoch),
    datasets:[{label:'Epoch Loss',data:D.training.map(d=>d.epochLoss),
      borderColor:'#2563eb',backgroundColor:grd(elCtx,'rgba(37,99,235,.22)','rgba(37,99,235,0)'),
      borderWidth:2.5,fill:true,tension:0.42,
      pointRadius:D.training.map((_,i)=>i===T.epoch?7:0),
      pointBackgroundColor:'#dc2626',pointBorderColor:'#fff',pointBorderWidth:2}]
  },{plugins:{legend:{display:false},tooltip:{callbacks:{title:c=>`Epoch ${c[0].label}`,label:c=>`Loss: ${c.parsed.y.toFixed(2)}`}}},scales:sc('Epoch','Cumulative Loss')});

  /* Bar comparison */
  mkChart('c-barComp','bar',{
    labels:['BLEU-1','BLEU-2','BLEU-3','BLEU-4','METEOR','CIDEr×100'],
    datasets:[
      {label:'Test',data:[T.bleu1,T.bleu2,T.bleu3,T.bleu4,T.meteor,T.cider*100],backgroundColor:'rgba(37,99,235,.8)',borderRadius:6,borderSkipped:false},
      {label:'Validation',data:[Vt.bleu1,Vt.bleu2,Vt.bleu3,Vt.bleu4,Vt.meteor,Vt.cider*100],backgroundColor:'rgba(124,58,237,.6)',borderRadius:6,borderSkipped:false}
    ]
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:8,color:'#4b5580'}},tooltip:{mode:'index'}},scales:sc('','Score')});

  /* BLEU 1&2 */
  const b12ctx=document.getElementById('c-bleu12').getContext('2d');
  mkChartNoAR('c-bleu12','line',{
    labels:D.validation.map(d=>d.epoch),
    datasets:[
      {label:'BLEU-1',data:D.validation.map(d=>d.bleu1),borderColor:'#2563eb',backgroundColor:grd(b12ctx,'rgba(37,99,235,.12)','rgba(37,99,235,0)'),fill:true,borderWidth:2.5,tension:0.35,pointRadius:0},
      {label:'BLEU-2',data:D.validation.map(d=>d.bleu2),borderColor:'#7c3aed',backgroundColor:grd(b12ctx,'rgba(124,58,237,.1)','rgba(124,58,237,0)'),fill:true,borderWidth:2,tension:0.35,pointRadius:0}
    ]
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:7,color:'#4b5580',padding:8}},tooltip:{mode:'index',intersect:false}},scales:sc('Epoch','Score')});

  /* BLEU 3&4 */
  const b34ctx=document.getElementById('c-bleu34').getContext('2d');
  const bpEp=T.epoch;
  mkChartNoAR('c-bleu34','line',{
    labels:D.validation.map(d=>d.epoch),
    datasets:[
      {label:'BLEU-3',data:D.validation.map(d=>d.bleu3),borderColor:'#059669',backgroundColor:grd(b34ctx,'rgba(5,150,105,.12)','rgba(5,150,105,0)'),fill:true,borderWidth:2,tension:0.35,pointRadius:0},
      {label:'BLEU-4',data:D.validation.map(d=>d.bleu4),borderColor:'#d97706',backgroundColor:grd(b34ctx,'rgba(217,119,6,.12)','rgba(217,119,6,0)'),fill:true,borderWidth:2.5,tension:0.35,
        pointRadius:D.validation.map((_,i)=>i===bpEp?7:0),pointBackgroundColor:'#d97706',pointBorderColor:'#fff',pointBorderWidth:2}
    ]
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:7,color:'#4b5580',padding:8}},tooltip:{mode:'index',intersect:false}},scales:sc('Epoch','Score')});

  /* Radar */
  const MX={b1:Math.max(...D.validation.map(v=>v.bleu1)),b2:Math.max(...D.validation.map(v=>v.bleu2)),b3:Math.max(...D.validation.map(v=>v.bleu3)),b4:Math.max(...D.validation.map(v=>v.bleu4)),mt:Math.max(...D.validation.map(v=>v.meteor)),cd:Math.max(...D.validation.map(v=>v.cider))};
  const n=(v,m)=>v/m*100;
  mkChartNoAR('c-radar','radar',{
    labels:['BLEU-1','BLEU-2','BLEU-3','BLEU-4','METEOR','CIDEr'],
    datasets:[
      {label:'Test',data:[n(T.bleu1,MX.b1),n(T.bleu2,MX.b2),n(T.bleu3,MX.b3),n(T.bleu4,MX.b4),n(T.meteor,MX.mt),n(T.cider,MX.cd)],borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,.14)',borderWidth:2,pointRadius:3,pointBackgroundColor:'#2563eb'},
      {label:`Val E${T.epoch}`,data:[n(Vt.bleu1,MX.b1),n(Vt.bleu2,MX.b2),n(Vt.bleu3,MX.b3),n(Vt.bleu4,MX.b4),n(Vt.meteor,MX.mt),n(Vt.cider,MX.cd)],borderColor:'#7c3aed',backgroundColor:'rgba(124,58,237,.11)',borderWidth:2,pointRadius:3,pointBackgroundColor:'#7c3aed'}
    ]
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:7,color:'#4b5580',padding:8}}},scales:{r:{angleLines:{color:'#e2e6f0'},grid:{color:'#e2e6f0'},pointLabels:{color:'#4b5580',font:{size:11}},ticks:{color:'#8892b0',stepSize:25,backdropColor:'transparent'}}}});
}

/* ══════════════════════════════════════════════════════════════
   8.  TRAINING PAGE
   ══════════════════════════════════════════════════════════════ */
function renderTraining(){
  /* Full epoch loss */
  const ctx=document.getElementById('c-epochLossFull').getContext('2d');
  mkChart('c-epochLossFull','line',{
    labels:D.training.map(d=>d.epoch),
    datasets:[{label:'Epoch Loss',data:D.training.map(d=>d.epochLoss),borderColor:'#2563eb',backgroundColor:grd(ctx,'rgba(37,99,235,.2)','rgba(37,99,235,0)'),borderWidth:2,fill:true,tension:0.42,
      pointRadius:D.training.map((_,i)=>i===D.testResult.epoch?7:0),pointBackgroundColor:'#dc2626',pointBorderColor:'#fff',pointBorderWidth:2}]
  },{plugins:{legend:{display:false},tooltip:{callbacks:{title:c=>`Epoch ${c[0].label}`,label:c=>`Loss: ${c.parsed.y.toFixed(3)}`}}},scales:sc('Epoch','Cumulative Loss')});

  /* Progress bars */
  const maxL=D.training[0].epochLoss;
  const steps=[0,5,10,D.testResult.epoch,25,30,D.training.length-1];
  const cols=['#dc2626','#ea580c','#d97706','#16a34a','#2563eb','#7c3aed','#059669'];
  const prog_html=steps.map((ep,i)=>{
    const row=D.training[ep]; if(!row) return '';
    return`<div><div class="prog-h"><span class="prog-n">Epoch ${ep}</span><span class="prog-v">${row.epochLoss.toFixed(1)}</span></div>
    <div class="prog-track"><div class="prog-fill" data-w="${(row.epochLoss/maxL*100).toFixed(1)}" style="width:0%;background:${cols[i]}"></div></div></div>`;
  }).join('');
  document.getElementById('lossBars').innerHTML=prog_html;
  requestAnimationFrame(()=>{
    document.querySelectorAll('#lossBars .prog-fill').forEach((el,i)=>{
      setTimeout(()=>{ el.style.transition='width .7s cubic-bezier(.34,1.2,.64,1)'; el.style.width=el.dataset.w+'%'; },i*80);
    });
  });

  /* Stats table */
  const last=D.training[D.training.length-1];
  document.getElementById('trainStats').innerHTML=`
    <table class="data-table">
      <tr><td>Total Epochs</td><td class="best">${D.training.length}</td></tr>
      <tr><td>Batches / Epoch</td><td>${D.training[0].batchLosses.length}</td></tr>
      <tr><td>Initial Loss</td><td>${D.training[0].epochLoss.toFixed(2)}</td></tr>
      <tr><td>Final Loss</td><td class="best">${last.epochLoss.toFixed(2)}</td></tr>
      <tr><td>Reduction</td><td class="best">${(100-last.epochLoss/D.training[0].epochLoss*100).toFixed(1)}%</td></tr>
      <tr><td>Best Checkpoint</td><td class="best">Epoch ${D.testResult.epoch}</td></tr>
    </table>`;

  /* Batch loss */
  renderBatchLoss(0);
  const sl=document.getElementById('batchSlider');
  sl.max=D.training.length-1;
  sl.oninput=function(){
    document.getElementById('batchSliderVal').textContent=`Epoch ${this.value}`;
    renderBatchLoss(parseInt(this.value));
  };

  /* Caption length */
  const ls=D.lengthStats;
  mkChart('c-capLen','line',{
    labels:ls.map(d=>d.epoch),
    datasets:[
      {label:'Mean',data:ls.map(d=>d.mean),borderColor:'#7c3aed',backgroundColor:'rgba(124,58,237,.1)',fill:true,borderWidth:2,tension:0.3,pointRadius:0},
      {label:'Mean+σ',data:ls.map(d=>d.mean+d.std),borderColor:'rgba(124,58,237,.3)',backgroundColor:'rgba(124,58,237,.06)',fill:'-1',borderWidth:1,tension:0.3,pointRadius:0,borderDash:[4,3]},
      {label:'Mean-σ',data:ls.map(d=>Math.max(0,d.mean-d.std)),borderColor:'rgba(124,58,237,.3)',backgroundColor:'transparent',fill:false,borderWidth:1,tension:0.3,pointRadius:0,borderDash:[4,3]},
    ]
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:7,color:'#4b5580',padding:10}},tooltip:{mode:'index',intersect:false}},scales:sc('Epoch','Words')});
}

function renderBatchLoss(epoch){
  const row=D.training[epoch]; if(!row) return;
  const ctx=document.getElementById('c-batchLoss').getContext('2d');
  mkChart('c-batchLoss','line',{
    labels:row.batchLosses.map((_,i)=>i+1),
    datasets:[{label:'Batch Loss',data:row.batchLosses,borderColor:'#7c3aed',backgroundColor:grd(ctx,'rgba(124,58,237,.2)','rgba(124,58,237,0)'),borderWidth:1.5,fill:true,tension:0.25,pointRadius:0}]
  },{plugins:{legend:{display:false},tooltip:{callbacks:{title:c=>`Batch ${c[0].label}`,label:c=>`Loss: ${c.parsed.y.toFixed(4)}`}}},scales:sc('Batch #','Loss')});
}

/* ══════════════════════════════════════════════════════════════
   9.  EVALUATION PAGE
   ══════════════════════════════════════════════════════════════ */
function renderEvaluation(){
  const T=D.testResult, ep=T.epoch;

  /* BLEU 1&2 full */
  const f12ctx=document.getElementById('c-bleuFull12').getContext('2d');
  mkChart('c-bleuFull12','line',{
    labels:D.validation.map(d=>d.epoch),
    datasets:[
      {label:'BLEU-1',data:D.validation.map(d=>d.bleu1),borderColor:'#2563eb',backgroundColor:grd(f12ctx,'rgba(37,99,235,.14)','rgba(37,99,235,0)'),fill:true,borderWidth:2.5,tension:0.35,pointRadius:D.validation.map((_,i)=>i===ep?6:0),pointBackgroundColor:'#2563eb',pointBorderColor:'#fff',pointBorderWidth:2},
      {label:'BLEU-2',data:D.validation.map(d=>d.bleu2),borderColor:'#7c3aed',backgroundColor:grd(f12ctx,'rgba(124,58,237,.1)','rgba(124,58,237,0)'),fill:true,borderWidth:2,tension:0.35,pointRadius:0}
    ]
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:7,color:'#4b5580',padding:10}},tooltip:{mode:'index',intersect:false}},scales:sc('Epoch','Score')});

  /* BLEU 3&4 full */
  const f34ctx=document.getElementById('c-bleuFull34').getContext('2d');
  mkChart('c-bleuFull34','line',{
    labels:D.validation.map(d=>d.epoch),
    datasets:[
      {label:'BLEU-3',data:D.validation.map(d=>d.bleu3),borderColor:'#059669',backgroundColor:grd(f34ctx,'rgba(5,150,105,.12)','rgba(5,150,105,0)'),fill:true,borderWidth:2,tension:0.35,pointRadius:0},
      {label:'BLEU-4',data:D.validation.map(d=>d.bleu4),borderColor:'#d97706',backgroundColor:grd(f34ctx,'rgba(217,119,6,.12)','rgba(217,119,6,0)'),fill:true,borderWidth:2.5,tension:0.35,pointRadius:D.validation.map((_,i)=>i===ep?6:0),pointBackgroundColor:'#d97706',pointBorderColor:'#fff',pointBorderWidth:2}
    ]
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:7,color:'#4b5580',padding:10}},tooltip:{mode:'index',intersect:false}},scales:sc('Epoch','Score')});

  /* METEOR & CIDEr */
  const mcCtx=document.getElementById('c-meteorCider').getContext('2d');
  mkChart('c-meteorCider','line',{
    labels:D.validation.map(d=>d.epoch),
    datasets:[
      {label:'METEOR',data:D.validation.map(d=>d.meteor),borderColor:'#db2777',backgroundColor:grd(mcCtx,'rgba(219,39,119,.15)','rgba(219,39,119,0)'),fill:true,borderWidth:2,tension:0.4,pointRadius:0,yAxisID:'y'},
      {label:'CIDEr×100',data:D.validation.map(d=>d.cider*100),borderColor:'#d97706',backgroundColor:'transparent',fill:false,borderWidth:2,tension:0.4,pointRadius:0,yAxisID:'y2'}
    ]
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:7,color:'#4b5580',padding:10}},tooltip:{mode:'index',intersect:false}},
   scales:{x:{grid:{color:GC},ticks:{color:'#8892b0'}},y:{position:'left',grid:{color:GC},ticks:{color:'#db2777'}},y2:{position:'right',grid:{drawOnChartArea:false},ticks:{color:'#d97706'}}}});

  /* Epoch inspector */
  const esl=document.getElementById('evalSlider');
  esl.max=D.validation.length-1; esl.value=T.epoch;
  document.getElementById('evalSliderVal').textContent=`Epoch ${T.epoch}`;
  renderEvalChips(T.epoch);
  esl.oninput=function(){
    document.getElementById('evalSliderVal').textContent=`Epoch ${this.value}`;
    renderEvalChips(parseInt(this.value));
  };

  /* Scatter */
  mkChart('c-scatter','scatter',{
    datasets:[{data:D.training.map((d,i)=>({x:d.epochLoss,y:D.validation[i].bleu4,e:i})),
      backgroundColor:D.training.map((_,i)=>{const t=i/(D.training.length-1);return`rgba(${Math.round(37+100*t)},${Math.round(99-40*t)},${Math.round(235-100*t)},.8)`;
      }),pointRadius:5,borderWidth:0}]
  },{plugins:{legend:{display:false},tooltip:{callbacks:{title:c=>`Epoch ${c[0].raw.e}`,label:c=>[`Loss: ${c.raw.x.toFixed(1)}`,`BLEU-4: ${c.raw.y.toFixed(3)}`]}}},scales:sc('Epoch Loss','BLEU-4')});

  /* Test bar */
  mkChart('c-testBar','bar',{
    labels:['BLEU-1','BLEU-2','BLEU-3','BLEU-4','METEOR','CIDEr×100'],
    datasets:[{label:'Test Score',data:[T.bleu1,T.bleu2,T.bleu3,T.bleu4,T.meteor,T.cider*100],
      backgroundColor:['#2563eb','#7c3aed','#059669','#d97706','#db2777','#0891b2'].map(c=>c+'cc'),borderRadius:8,borderSkipped:false}]
  },{plugins:{legend:{display:false}},scales:sc('','Score')});

  /* Top 10 table */
  const sorted=[...D.validation].sort((a,b)=>b.bleu4-a.bleu4).slice(0,10);
  document.getElementById('top10Table').innerHTML=`
    <thead><tr><th>#</th><th>Epoch</th><th>BLEU-1</th><th>BLEU-2</th><th>BLEU-3</th><th>BLEU-4</th><th>METEOR</th><th>CIDEr</th></tr></thead>
    <tbody>${sorted.map((v,i)=>`<tr>
      <td>${i+1}</td>
      <td class="${v.epoch===T.epoch?'best':''}">${v.epoch}${v.epoch===T.epoch?' ⭐':''}</td>
      <td>${v.bleu1.toFixed(3)}</td><td>${v.bleu2.toFixed(3)}</td><td>${v.bleu3.toFixed(3)}</td>
      <td class="best">${v.bleu4.toFixed(3)}</td>
      <td>${v.meteor.toFixed(3)}</td><td>${v.cider.toFixed(5)}</td>
    </tr>`).join('')}</tbody>`;
}

function renderEvalChips(epoch){
  const d=D.validation[epoch]; if(!d) return;
  const items=[
    {l:'BLEU-1',v:d.bleu1.toFixed(3),c:'#2563eb'},{l:'BLEU-2',v:d.bleu2.toFixed(3),c:'#7c3aed'},
    {l:'BLEU-3',v:d.bleu3.toFixed(3),c:'#059669'},{l:'BLEU-4',v:d.bleu4.toFixed(3),c:'#d97706'},
    {l:'METEOR',v:d.meteor.toFixed(3),c:'#db2777'},{l:'CIDEr',v:d.cider.toFixed(5),c:'#0891b2'},
    {l:'Ep Loss',v:(D.training[epoch]?.epochLoss||0).toFixed(1),c:'#4b5580'},
  ];
  document.getElementById('evalChips').innerHTML=items.map(x=>
    `<div class="chip"><div class="chip-l">${x.l}</div><div class="chip-v" style="color:${x.c}">${x.v}</div></div>`
  ).join('');
}

/* ══════════════════════════════════════════════════════════════
   10.  CAPTIONS PAGE
   ══════════════════════════════════════════════════════════════ */
function renderCaptions(){
  const imgs=Object.keys(D.captionEvolution);
  document.getElementById('evoTabs').innerHTML=imgs.map((img,i)=>
    `<button class="evo-tab${i===0?' active':''}" onclick="setEvoImg('${img}',this)">${img.split('.')[0].slice(-8)}</button>`
  ).join('');
  setEvoImg(imgs[0], document.querySelector('.evo-tab'));

  document.getElementById('capGrid').innerHTML=D.testCaptions.map(c=>`
    <div class="cap-item">
      <div class="cap-badge">🔤 Hindi · Best Epoch</div>
      <div class="cap-file">${c.img}</div>
      <div class="cap-text">${c.cap}</div>
    </div>`).join('');

  const keyEps=Object.keys(D.sampleLengths).filter(k=>D.sampleLengths[k].length);
  const cols=['#2563eb','#7c3aed','#059669','#d97706','#db2777','#0891b2','#059669','#dc2626','#ea580c'];
  const maxLen=38;
  const bins=Array.from({length:Math.floor(maxLen/2)},(_,i)=>i*2+2);
  mkChart('c-capDist','bar',{
    labels:bins,
    datasets:keyEps.map((ep,i)=>{
      const data=D.sampleLengths[ep];
      const hist=bins.map(b=>data.filter(l=>l>=b&&l<b+2).length/data.length*100);
      return{label:`E${ep}`,data:hist,backgroundColor:cols[i]+'55',borderColor:cols[i],borderWidth:1.5,borderRadius:3};
    })
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:8,color:'#4b5580',padding:10}},tooltip:{mode:'index',intersect:false}},scales:sc('Length (words)','Frequency %')});
}

function setEvoImg(img,btn){
  document.querySelectorAll('.evo-tab').forEach(b=>b.classList.remove('active'));
  if(btn) btn.classList.add('active');
  const evo=D.captionEvolution[img]||{};
  const show=[0,2,5,8,10,13,D.testResult.epoch,20,25,30,35,D.training.length-1];
  document.getElementById('evoTimeline').innerHTML=show.map(ep=>`
    <div class="evo-row">
      <div class="evo-ep">E${ep}</div>
      <div class="evo-cap">${evo[String(ep)]||'–'}</div>
    </div>`).join('');
}
window.setEvoImg=setEvoImg;

/* ══════════════════════════════════════════════════════════════
   11.  ADVANCED PAGE
   ══════════════════════════════════════════════════════════════ */
function renderAdvanced(){
  const vd=D.validation;
  const keys=['bleu1','bleu2','bleu3','bleu4','meteor','cider'];
  const lbls=['B-1','B-2','B-3','B-4','MTR','CDR'];
  const series=keys.map(k=>vd.map(v=>v[k]));

  /* Correlation heatmap */
  const pairs=[];
  for(let i=0;i<keys.length;i++) for(let j=0;j<keys.length;j++)
    pairs.push({x:i,y:j,v:parseFloat(pearson(series[i],series[j]).toFixed(3))});
  mkChart('c-corr','scatter',{
    datasets:[{data:pairs,backgroundColor:pairs.map(p=>corrColor(p.v)),pointRadius:26,pointStyle:'rect',borderWidth:0}]
  },{plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`r = ${c.raw.v.toFixed(3)}`}}},
   scales:{
     x:{type:'linear',min:-.5,max:5.5,ticks:{stepSize:1,callback:v=>lbls[Math.round(v)]||''},grid:{display:false}},
     y:{type:'linear',min:-.5,max:5.5,ticks:{stepSize:1,callback:v=>lbls[Math.round(v)]||''},grid:{display:false}}
   }});

  /* KDE */
  const kEps=Object.keys(D.sampleLengths).filter(k=>D.sampleLengths[k].length).slice(0,5);
  const kCols=['#2563eb','#7c3aed','#059669','#d97706','#db2777'];
  const kPts=Array.from({length:40},(_,i)=>i+1);
  mkChart('c-kde','line',{
    labels:kPts,
    datasets:kEps.map((ep,i)=>{
      const data=D.sampleLengths[ep];
      const bw=Math.max(1.06*Math.sqrt(data.reduce((a,b)=>a+b)/data.length)*Math.pow(data.length,-.2),.8);
      return{label:`E${ep}`,data:kde(data,bw,kPts).map(v=>parseFloat((v*100).toFixed(4))),borderColor:kCols[i],backgroundColor:'transparent',borderWidth:2,tension:.4,pointRadius:0};
    })
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:7,color:'#4b5580',padding:10}},tooltip:{mode:'index',intersect:false}},scales:sc('Length','Density×100')});

  /* BLEU-4 distribution */
  const b4=vd.map(v=>v.bleu4);
  const sb4=[...b4].sort((a,b)=>a-b);
  const q1=sb4[Math.floor(sb4.length*.25)],q3=sb4[Math.floor(sb4.length*.75)];
  mkChart('c-bleu4dist','line',{
    labels:vd.map(d=>d.epoch),
    datasets:[
      {label:'BLEU-4',data:b4,borderColor:'#d97706',backgroundColor:'rgba(217,119,6,.14)',fill:true,borderWidth:2.5,tension:.35,pointRadius:0},
      {label:'Q3',data:Array(b4.length).fill(q3),borderColor:'rgba(217,119,6,.35)',backgroundColor:'transparent',borderWidth:1,tension:0,pointRadius:0,borderDash:[5,4]},
      {label:'Q1',data:Array(b4.length).fill(q1),borderColor:'rgba(217,119,6,.35)',backgroundColor:'transparent',borderWidth:1,tension:0,pointRadius:0,borderDash:[5,4]},
    ]
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:7,color:'#4b5580',padding:10}},tooltip:{mode:'index',intersect:false}},scales:sc('Epoch','BLEU-4')});

  /* Multi-scatter */
  const mDef=[{k:'meteor',c:'#db2777',l:'METEOR'},{k:'bleu1',c:'#2563eb',l:'BLEU-1'},{k:'cider',c:'#d97706',l:'CIDEr×100'}];
  mkChart('c-multiScatter','scatter',{
    datasets:mDef.map(m=>({label:m.l,data:D.training.map((d,i)=>({x:d.epochLoss,y:m.k==='cider'?vd[i].cider*100:vd[i][m.k]})),backgroundColor:m.c+'88',pointRadius:4,borderWidth:0}))
  },{plugins:{legend:{position:'top',labels:{usePointStyle:true,boxWidth:7,color:'#4b5580',padding:10}}},scales:sc('Epoch Loss','Score')});

  /* Metric heatmap table */
  const hKeys=['bleu1','bleu2','bleu3','bleu4','meteor','cider'];
  const hLbls=['BLEU-1','BLEU-2','BLEU-3','BLEU-4','METEOR','CIDEr'];
  const mima=hKeys.map(k=>{const vs=vd.map(v=>v[k]);return{min:Math.min(...vs),max:Math.max(...vs)};});
  const showEps=[0,2,4,6,8,10,12,14,D.testResult.epoch,20,22,25,27,30,32,35,37,D.training.length-1];
  const rows=showEps.map(ep=>{
    const v=vd[ep]; if(!v) return '';
    const cells=hKeys.map((k,i)=>{
      const{min,max}=mima[i],t=(v[k]-min)/(max-min||1);
      const bg=heatColor(t),fg=t>.6?'white':'#1e2340';
      return`<td style="background:${bg};color:${fg}">${v[k].toFixed(2)}</td>`;
    });
    return`<tr><th style="text-align:right;padding-right:10px;color:#2563eb;font-weight:700;white-space:nowrap">E${ep}</th>${cells.join('')}</tr>`;
  });
  document.getElementById('heatmapWrap').innerHTML=`
    <table class="heatmap-table">
      <thead><tr><th></th>${hLbls.map(l=>`<th>${l}</th>`).join('')}</tr></thead>
      <tbody>${rows.join('')}</tbody>
    </table>`;
}

/* ══════════════════════════════════════════════════════════════
   12.  INSIGHTS PAGE
   ══════════════════════════════════════════════════════════════ */
function renderInsights(){
  const T=D.testResult,V=D.validation;
  const pB4=V.reduce((b,v)=>v.bleu4>b.bleu4?v:b);
  const pMt=V.reduce((b,v)=>v.meteor>b.meteor?v:b);
  const last=D.training[D.training.length-1];
  const ins=[
    {ico:'📉',bg:'#dbeafe',title:'Rapid Early Convergence',desc:`Loss drops ${D.training[0].epochLoss.toFixed(0)} → ${D.training[2].epochLoss.toFixed(0)} in just 2 epochs (${(100-D.training[2].epochLoss/D.training[0].epochLoss*100).toFixed(0)}% reduction), showing strong initial CLIP–Hindi feature alignment.`},
    {ico:'📈',bg:'#d1fae5',title:'Stable Late-Stage Learning',desc:`Epochs 25–${last.epoch} plateau at ${D.training[25]?.epochLoss.toFixed(0)||'~'}–${last.epochLoss.toFixed(0)} loss, yet METEOR keeps improving — semantic refinement beyond loss convergence.`},
    {ico:'🎯',bg:'#fef3c7',title:`Best Checkpoint: Epoch ${T.epoch}`,desc:`Epoch ${T.epoch} yields peak validation BLEU-4 (${pB4.bleu4.toFixed(3)}). Test confirms generalisation: BLEU-4 ${T.bleu4.toFixed(3)} — a +${(T.bleu4-pB4.bleu4).toFixed(3)} autoregressive lift.`},
    {ico:'🔤',bg:'#ede9fe',title:'Hindi Caption Coherence',desc:`Generated Devanagari captions show well-formed grammatical structures with rich visual descriptors, confirming successful cross-modal alignment between CLIP and FastText-Hindi.`},
    {ico:'⚡',bg:'#cffafe',title:'BLEU-1 vs BLEU-4 Gap',desc:`BLEU-1 (${V[T.epoch].bleu1.toFixed(1)}) to BLEU-4 (${V[T.epoch].bleu4.toFixed(1)}) ratio ~${(V[T.epoch].bleu1/V[T.epoch].bleu4).toFixed(1)}× reflects natural n-gram degradation — expected for Hindi captioning.`},
    {ico:'⚠️',bg:'#fee2e2',title:'ROUGE-L Anomaly',desc:'ROUGE-L records 0.0 across all epochs — likely a Devanagari tokenisation mismatch between generated tokens and reference strings. Requires investigation before submission.'},
    {ico:'📊',bg:'#fce7f3',title:'High Inter-Metric Correlation',desc:'Strong Pearson r > 0.95 between BLEU-1, METEOR, and CIDEr confirms all metrics are consistently capturing the same underlying improvements in caption quality.'},
    {ico:'🔬',bg:'#d1fae5',title:'Caption Length Stability',desc:`Mean caption length stabilises near ${D.lengthStats[5]?.mean.toFixed(1)||'9'} words after Epoch 5. Early epochs show higher variance as the model explores vocabulary.`},
    {ico:'🏆',bg:'#fef3c7',title:`Peak METEOR: Epoch ${pMt.epoch}`,desc:`Best validation METEOR is ${pMt.meteor.toFixed(3)} at Epoch ${pMt.epoch} — occurring after the BLEU-4 peak, suggesting semantic quality continues improving after n-gram precision plateaus.`},
  ];
  document.getElementById('insightGrid').innerHTML=ins.map(x=>`
    <div class="insight">
      <div class="ins-ico" style="background:${x.bg}">${x.ico}</div>
      <div class="ins-title">${x.title}</div>
      <div class="ins-desc">${x.desc}</div>
    </div>`).join('');

  document.getElementById('archTable').innerHTML=`
    <table class="data-table">
      <tr><td>Visual Encoder</td><td class="best">CLIP ViT-B/32 (Frozen)</td></tr>
      <tr><td>Text Encoder</td><td class="best">FastText-Hindi 300-d → 512-d (Frozen)</td></tr>
      <tr><td>Fusion Block</td><td>Asymmetric Cross-Modal Attention + Residual + LayerNorm</td></tr>
      <tr><td>Decoder</td><td>Transformer — Masked Self-Attention, Autoregressive</td></tr>
      <tr><td>Dataset</td><td>Flickr 8K-Hindi</td></tr>
      <tr><td>Training</td><td>2-Phase: Full LR → Fine-tune (Fusion frozen) + Early Stop</td></tr>
      <tr><td>Batch Size</td><td class="best">${D.batchName.replace('Batch-','')}</td></tr>
      <tr><td>Best Epoch</td><td class="best">${T.epoch}</td></tr>
      <tr><td>Test BLEU-4</td><td class="best">${T.bleu4.toFixed(4)}</td></tr>
      <tr><td>Test METEOR</td><td class="best">${T.meteor.toFixed(4)}</td></tr>
    </table>`;
}

/* ══════════════════════════════════════════════════════════════
   13.  RENDER ALL
   ══════════════════════════════════════════════════════════════ */
function renderAll(){
  renderKPI();
  renderDashboard();
  renderTraining();
  renderEvaluation();
  renderCaptions();
  renderAdvanced();
  renderInsights();
}

/* ══════════════════════════════════════════════════════════════
   14.  INIT
   ══════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  scanBatches();
});
