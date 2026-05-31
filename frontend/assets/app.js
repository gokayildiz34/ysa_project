/* ── API ─────────────────────────────────── */
var API_BASE = window.location.protocol === 'file:' || window.location.port === '5500' ? 'http://localhost:8000' : '';
var API = {
  health:           API_BASE + '/api/health',
  modelInfo:        API_BASE + '/api/model-info',
  upload:           API_BASE + '/api/analyze-upload',
  realisticFiles:   API_BASE + '/api/realistic-files',
  analyzeRealistic: function(f) { return API_BASE + '/api/analyze-realistic/' + encodeURIComponent(f); },
  analyses:         API_BASE + '/api/analyses',
  analysisDetail:   function(id) { return API_BASE + '/api/analyses/' + id; },
};

/* ── MODEL REGISTRY ──────────────────────── */
const MODELS = {
  autoencoder:     { label:'Autoencoder',       icon:'🧠', color:'#00d4aa', desc:'Deep learning reconstruction' },
  isolation_forest:{ label:'Isolation Forest',  icon:'🌲', color:'#a78bfa', desc:'Ensemble tree-based isolation' },
  ocsvm:           { label:'One-Class SVM',     icon:'🔵', color:'#fb923c', desc:'Support vector boundary' },
  pca:             { label:'PCA Reconstruction',icon:'📊', color:'#38bdf8', desc:'Linear dimensionality reduction' },
};

/* ── STATE ────────────────────────────────── */
const state = { selectedModel: null, lastResult: null, currentSection: 'analyze' };

/* ── UTILS ────────────────────────────────── */
const fmt  = (v, d=4) => v == null ? '-' : typeof v === 'number' ? v.toFixed(d) : v;
const fmtP = (v) => v == null ? '-' : (v*100).toFixed(1)+'%';

async function fetchJson(url, opts) {
  opts = opts || {};
  var res = await fetch(url, opts);
  if (!res.ok) {
    var err = await res.json().catch(function(){ return {detail: res.statusText}; });
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

/* ── TOAST ────────────────────────────────── */
function toast(msg, type) {
  type = type || 'info';
  var c = document.getElementById('toastContainer');
  if (!c) return;
  var el = document.createElement('div');
  el.className = 'toast toast-' + type;
  var icon = type === 'error'
    ? '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>'
    : '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>';
  el.innerHTML = icon + '<span>' + msg + '</span>';
  c.appendChild(el);
  setTimeout(function(){ if (el.parentNode) el.parentNode.removeChild(el); }, 4500);
}

/* ── LOADING ─────────────────────────────── */
function showLoading(text, sub) {
  text = text || 'Analyzing traffic...';
  sub = sub || '';
  var lt = document.getElementById('loadingText');
  var lm = document.getElementById('loadingModel');
  var lo = document.getElementById('loadingOverlay');
  if (lt) lt.textContent = text;
  if (lm) lm.textContent = sub;
  if (lo) lo.classList.remove('hidden');
}
function hideLoading() {
  var lo = document.getElementById('loadingOverlay');
  if (lo) lo.classList.add('hidden');
}

/* ── CLOCK ────────────────────────────────── */
function updateClock() {
  var el = document.getElementById('currentTime');
  if (el) el.textContent = new Date().toLocaleTimeString('tr-TR', {hour12: false});
}
setInterval(updateClock, 1000);
updateClock();

/* ── NAVIGATION ──────────────────────────── */
var SECTION_TITLES = {
  analyze:            'Trafik Analiz Konsolu',
  realistic:          'Gerçekçi Test Dosyaları',
  history:            'Analiz Geçmişi',
  model:              'Autoencoder - Model Bilgisi',
  compare:            'Model Karşılaştırma',
  'isolation-forest': 'Isolation Forest - Rapor',
  ocsvm:              'One-Class SVM - Rapor',
  pca:                'PCA Yeniden Yapılandırma - Rapor',
};

function switchSection(section) {
  state.currentSection = section;
  var sections = document.querySelectorAll('.page-section');
  for (var i = 0; i < sections.length; i++) {
    sections[i].classList.add('hidden');
  }
  var target = document.getElementById('section-' + section);
  if (target) target.classList.remove('hidden');

  var navItems = document.querySelectorAll('.nav-item');
  for (var j = 0; j < navItems.length; j++) {
    if (navItems[j].dataset.section === section) {
      navItems[j].classList.add('active');
    } else {
      navItems[j].classList.remove('active');
    }
  }
  var titleEl = document.getElementById('topbarTitle');
  if (titleEl) titleEl.textContent = SECTION_TITLES[section] || 'NetAnomAI';
}

/* NAV CLICK EVENTS */
(function() {
  var navItems = document.querySelectorAll('.nav-item');
  for (var i = 0; i < navItems.length; i++) {
    (function(a) {
      a.addEventListener('click', function(e) {
        e.preventDefault();
        var s = a.dataset.section;
        if (!s) return;
        switchSection(s);
        if (s === 'history')           loadHistory();
        if (s === 'realistic')         loadRealisticFiles();
        if (s === 'model')             loadModelInfo();
        if (s === 'compare')           loadCompare();
        if (s === 'isolation-forest')  loadModelReport('isolation_forest', 'ifReport');
        if (s === 'ocsvm')             loadModelReport('ocsvm', 'ocsvmReport');
        if (s === 'pca')               loadModelReport('pca', 'pcaReport');
      });
    })(navItems[i]);
  }
})();

/* ── API STATUS ───────────────────────────── */
function checkApi() {
  var el  = document.getElementById('apiStatus');
  if (!el) return;
  var lbl = el.querySelector('.api-label');
  fetchJson(API.health).then(function() {
    el.className = 'api-status online';
    if (lbl) lbl.textContent = 'API Çevrimiçi';
  }).catch(function() {
    el.className = 'api-status offline';
    if (lbl) lbl.textContent = 'API Çevrimdışı';
  });
}

/* ── MODEL SELECTION ─────────────────────── */
function selectModel(modelKey) {
  state.selectedModel = modelKey;
  var m = MODELS[modelKey];
  if (!m) return;

  var cards = document.querySelectorAll('.model-option');
  for (var i = 0; i < cards.length; i++) {
    if (cards[i].dataset.model === modelKey) {
      cards[i].classList.add('selected');
    } else {
      cards[i].classList.remove('selected');
    }
  }

  var banner = document.getElementById('selectedModelBanner');
  var smbIcon = document.getElementById('smbIcon');
  var smbName = document.getElementById('smbName');
  var smbDesc = document.getElementById('smbDesc');
  if (smbIcon) smbIcon.textContent = m.icon;
  if (smbName) smbName.textContent = m.label;
  if (smbDesc) smbDesc.textContent = m.desc;
  if (banner)  banner.classList.remove('hidden');

  var step2 = document.getElementById('step2');
  if (step2) step2.classList.add('unlocked');

  updateSubmitBtn();
}

/* MODEL CARD EVENTS */
(function() {
  var cards = document.querySelectorAll('.model-option');
  for (var i = 0; i < cards.length; i++) {
    (function(card) {
      card.addEventListener('click', function() {
        selectModel(card.dataset.model);
      });
      card.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          selectModel(card.dataset.model);
        }
      });
    })(cards[i]);
  }
})();

/* CHANGE MODEL BUTTON */
var changeModelBtn = document.getElementById('changeModelBtn');
if (changeModelBtn) {
  changeModelBtn.addEventListener('click', function() {
    state.selectedModel = null;
    var cards = document.querySelectorAll('.model-option');
    for (var i = 0; i < cards.length; i++) {
      cards[i].classList.remove('selected');
    }
    var banner = document.getElementById('selectedModelBanner');
    var step2 = document.getElementById('step2');
    var uploadBtn = document.getElementById('uploadBtn');
    if (banner) banner.classList.add('hidden');
    if (step2)  step2.classList.remove('unlocked');
    if (uploadBtn) uploadBtn.disabled = true;
  });
}

function updateSubmitBtn() {
  var btn    = document.getElementById('uploadBtn');
  var fileEl = document.getElementById('pcapFile');
  if (!btn || !fileEl) return;
  var hasFile  = fileEl.files && fileEl.files.length > 0;
  var hasModel = !!state.selectedModel;
  btn.disabled = !(hasFile && hasModel);
}

/* ── UPLOAD DROP ZONE ────────────────────── */
var dropZone   = document.getElementById('uploadDropZone');
var fileInput  = document.getElementById('pcapFile');
var fileNameEl = document.getElementById('uploadFileName');

function setFile(f) {
  if (!f) return;
  if (fileNameEl) {
    fileNameEl.textContent = 'Dosya: ' + f.name;
    fileNameEl.classList.remove('hidden');
  }
  if (dropZone) dropZone.classList.add('has-file');
  if (fileInput) {
    try {
      var dt = new DataTransfer();
      dt.items.add(f);
      fileInput.files = dt.files;
    } catch(e) {}
  }
  updateSubmitBtn();
}

if (dropZone) {
  dropZone.addEventListener('dragover', function(e) {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', function() {
    dropZone.classList.remove('drag-over');
  });
  dropZone.addEventListener('drop', function(e) {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    var f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });
}

if (fileInput) {
  fileInput.addEventListener('change', function() {
    if (fileInput.files && fileInput.files[0]) setFile(fileInput.files[0]);
  });
}

/* ── UPLOAD SUBMIT ───────────────────────── */
var uploadForm = document.getElementById('uploadForm');
if (uploadForm) {
  uploadForm.addEventListener('submit', function(e) {
    e.preventDefault();
    var fi = document.getElementById('pcapFile');
    if (!fi || !fi.files || !fi.files.length) { toast('Lutfen bir .pcap/.pcapng dosyasi secin.', 'error'); return; }
    if (!state.selectedModel) { toast('Lutfen once bir model secin.', 'error'); return; }

    var ratio = (document.getElementById('ratioInput') || {}).value || '0.10';
    var wsize = (document.getElementById('windowSizeInput') || {}).value || '1.0';
    var model = state.selectedModel;
    var fd = new FormData();
    fd.append('file', fi.files[0]);

    var m = MODELS[model];
    showLoading('Analiz ediliyor: ' + fi.files[0].name, 'Model: ' + m.label);
    fetchJson(API.upload + '?anomaly_ratio_threshold=' + ratio + '&window_size=' + wsize + '&model_type=' + model, {method: 'POST', body: fd})
      .then(function(result) {
        hideLoading();
        renderResult(result, model);
        toast(m.icon + ' ' + result.result + ' - ' + m.label, result.result === 'ANOMALY' ? 'error' : 'success');
      })
      .catch(function(err) {
        hideLoading();
        toast(err.message, 'error');
      });
  });
}

/* ── RENDER RESULT ───────────────────────── */
function renderResult(r, modelKey) {
  state.lastResult = Object.assign({}, r, {_model: modelKey});
  var panel = document.getElementById('resultPanel');
  if (!panel) return;
  panel.classList.remove('hidden');

  var m = MODELS[modelKey] || {label: modelKey || 'Unknown', icon: '?', color: '#00d4aa'};

  var badge = document.getElementById('resultBadge');
  if (badge) {
    badge.textContent = r.result;
    badge.className = 'result-badge ' + (r.result === 'ANOMALY' ? 'result-anomaly' : 'result-normal');
  }

  var rt = document.getElementById('resultTitle');
  var rs = document.getElementById('resultSubtitle');
  var mt = document.getElementById('resultModelTag');
  if (rt) rt.textContent = r.file_name;
  if (rs) rs.textContent = 'Analiz #' + r.analysis_id + ' - ' + (r.created_at || '');
  if (mt) {
    mt.textContent = m.icon + ' ' + m.label;
    mt.style.borderColor = m.color + '55';
    mt.style.color = m.color;
  }

  function setVal(id, v) { var el = document.getElementById(id); if (el) el.textContent = v; }
  setVal('totalWindows',   r.total_windows);
  setVal('anomalyWindows', r.anomaly_windows);
  setVal('anomalyRatio',   fmtP(r.anomaly_ratio));
  setVal('avgError',       fmt(r.avg_error, 6));
  setVal('maxError',       fmt(r.max_error, 6));
  setVal('thresholdValue', fmt(r.threshold, 8));

  renderRanges(r.anomaly_ranges || []);
  renderTopWindows(r.top_windows || []);
  drawChart(r.windows || [], r.threshold, m.color);

  setTimeout(function() { panel.scrollIntoView({behavior: 'smooth', block: 'start'}); }, 100);
}

function renderRanges(ranges) {
  var el = document.getElementById('rangeList');
  if (!el) return;
  if (!ranges || !ranges.length) {
    el.innerHTML = '<div class="list-item empty">Anormal aralık tespit edilmedi.</div>';
    return;
  }
  var html = '';
  for (var i = 0; i < ranges.length; i++) {
    html += '<div class="list-item anomaly">Aralık ' + (i+1) + ': pencere ' + ranges[i].start_window + ' - ' + ranges[i].end_window + '</div>';
  }
  el.innerHTML = html;
}

function renderTopWindows(windows) {
  var el = document.getElementById('topWindowList');
  if (!el) return;
  if (!windows || !windows.length) { el.innerHTML = '<div class="list-item empty">Veri yok.</div>'; return; }
  var html = '';
  for (var i = 0; i < windows.length; i++) {
    var w = windows[i];
    html += '<div class="list-item ' + (w.is_anomaly ? 'anomaly' : '') + '">';
    html += 'Pencere ' + w.window_id + ' | skor ' + fmt(w.reconstruction_error, 6) + ' | ' + (w.is_anomaly ? 'ANOMALİ' : 'NORMAL');
    html += '</div>';
  }
  el.innerHTML = html;
}

/* ── CHART ────────────────────────────────── */
function drawChart(windows, threshold, lineColor) {
  lineColor = lineColor || '#00d4aa';
  var canvas = document.getElementById('errorChart');
  if (!canvas) return;
  var dpr   = window.devicePixelRatio || 1;
  var W_css = canvas.parentElement.getBoundingClientRect().width;
  var H_css = 260;
  canvas.width  = W_css * dpr;
  canvas.height = H_css * dpr;
  canvas.style.width  = W_css + 'px';
  canvas.style.height = H_css + 'px';

  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W_css, H_css);
  if (!windows || !windows.length) return;

  var pad = {top:20, right:20, bottom:36, left:56};
  var pw = W_css - pad.left - pad.right;
  var ph = H_css - pad.top  - pad.bottom;

  var errors = windows.map(function(w){ return w.reconstruction_error; });
  var maxE   = Math.max.apply(null, errors.concat([threshold || 0])) * 1.1;
  var minId  = Math.min.apply(null, windows.map(function(w){ return w.window_id; }));
  var maxId  = Math.max.apply(null, windows.map(function(w){ return w.window_id; }));
  var span   = Math.max(1, maxId - minId);

  function xS(id) { return pad.left + ((id - minId) / span) * pw; }
  function yS(v)  { return pad.top  + ph - (v / maxE) * ph; }

  ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
  for (var i = 0; i <= 4; i++) {
    var y = pad.top + (ph / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pw, y); ctx.stroke();
    ctx.fillStyle = 'rgba(155,171,194,0.55)';
    ctx.font = "10px 'JetBrains Mono',monospace"; ctx.textAlign = 'right';
    ctx.fillText((maxE * (1 - i/4)).toExponential(1), pad.left - 6, y + 4);
  }

  ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(pad.left, pad.top + ph); ctx.lineTo(pad.left + pw, pad.top + ph); ctx.stroke();

  if (threshold != null) {
    var ty = yS(threshold);
    ctx.strokeStyle = '#ff4757'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(pad.left, ty); ctx.lineTo(pad.left + pw, ty); ctx.stroke();
    ctx.setLineDash([]);
  }

  for (var k = 0; k < windows.length - 1; k++) {
    var w = windows[k], wn = windows[k+1];
    var x1=xS(w.window_id), y1=yS(w.reconstruction_error);
    var x2=xS(wn.window_id), y2=yS(wn.reconstruction_error);
    var by = pad.top + ph;
    ctx.beginPath(); ctx.moveTo(x1,by); ctx.lineTo(x1,y1); ctx.lineTo(x2,y2); ctx.lineTo(x2,by); ctx.closePath();
    ctx.fillStyle = w.is_anomaly ? 'rgba(255,71,87,0.1)' : 'rgba(0,212,170,0.06)';
    ctx.fill();
  }

  ctx.strokeStyle = lineColor; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  ctx.beginPath();
  for (var n = 0; n < windows.length; n++) {
    var wx = xS(windows[n].window_id), wy = yS(windows[n].reconstruction_error);
    if (n === 0) ctx.moveTo(wx, wy); else ctx.lineTo(wx, wy);
  }
  ctx.stroke();

  for (var p = 0; p < windows.length; p++) {
    if (!windows[p].is_anomaly) continue;
    ctx.fillStyle = '#ff4757';
    ctx.beginPath();
    ctx.arc(xS(windows[p].window_id), yS(windows[p].reconstruction_error), 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = 'rgba(155,171,194,0.55)'; ctx.font = "10px 'JetBrains Mono',monospace"; ctx.textAlign = 'center';
  var step = Math.floor(windows.length / Math.min(windows.length, 8)) || 1;
  for (var q = 0; q < windows.length; q += step) {
    ctx.fillText('w' + windows[q].window_id, xS(windows[q].window_id), pad.top + ph + 18);
  }
}

/* ── REALISTIC FILES ─────────────────────── */
function loadRealisticFiles() {
  var c = document.getElementById('realisticFiles');
  if (!c) return;
  c.innerHTML = '<div class="loading-placeholder"><div class="loading-spinner" style="width:20px;height:20px"></div> Yukleniyor...</div>';
  fetchJson(API.realisticFiles).then(function(data) {
    c.innerHTML = '';
    if (!data.files || !data.files.length) {
      c.innerHTML = '<p style="color:var(--text3);padding:16px 0">data/raw/realistic_test klasorunde pcap dosyasi bulunamadi.</p>';
      return;
    }
    for (var i = 0; i < data.files.length; i++) {
      (function(f) {
        var card = document.createElement('div');
        card.className = 'file-card';
        card.innerHTML =
          '<div class="file-card-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg></div>' +
          '<div class="file-card-name">' + f.file_name + '</div>' +
          '<div class="file-card-size">' + (f.size_bytes/1024/1024).toFixed(2) + ' MB</div>' +
          '<button class="btn btn-primary btn-sm" style="width:100%">Analiz Et</button>';
        card.querySelector('button').addEventListener('click', function() { analyzeRealistic(f.file_name); });
        c.appendChild(card);
      })(data.files[i]);
    }
  }).catch(function(e) {
    c.innerHTML = '<p style="color:var(--danger)">' + e.message + '</p>';
    toast(e.message, 'error');
  });
}

function analyzeRealistic(fileName) {
  var ratio  = (document.getElementById('ratioInput') || {}).value || '0.10';
  var wsize  = (document.getElementById('windowSizeInput') || {}).value || '1.0';
  var modelEl = document.getElementById('realisticModelSelect');
  var model  = modelEl ? modelEl.value : 'autoencoder';
  var m = MODELS[model] || MODELS['autoencoder'];
  showLoading('Analiz ediliyor: ' + fileName, 'Model: ' + m.label);
  switchSection('analyze');
  var url = API.analyzeRealistic(fileName) + '?anomaly_ratio_threshold=' + ratio + '&window_size=' + wsize + '&model_type=' + model;
  fetchJson(url, {method: 'POST'}).then(function(result) {
    hideLoading();
    renderResult(result, model);
    toast(m.icon + ' ' + fileName + ': ' + result.result, 'success');
  }).catch(function(err) {
    hideLoading();
    toast(err.message, 'error');
    switchSection('realistic');
  });
}

/* ── HISTORY ─────────────────────────────── */
function loadHistory() {
  var body = document.getElementById('historyBody');
  if (!body) return;
  body.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text3);padding:20px">Yükleniyor...</td></tr>';
  fetchJson(API.analyses).then(function(data) {
    body.innerHTML = '';
    if (!data.items || !data.items.length) {
      body.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text3);padding:20px">Henüz analiz yok.</td></tr>';
      return;
    }
    var statsId = 'historyStatsBanner';
    var statsBanner = document.getElementById(statsId);
    if (!statsBanner) {
      statsBanner = document.createElement('div');
      statsBanner.id = statsId;
      statsBanner.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:16px';
      var tableWrap = document.querySelector('.table-wrap');
      if (tableWrap && tableWrap.parentElement) tableWrap.parentElement.insertBefore(statsBanner, tableWrap);
    }
    var anomalyCount = data.items.filter(function(i){ return i.result === 'ANOMALY'; }).length;
    var normalCount  = data.items.filter(function(i){ return i.result === 'NORMAL'; }).length;
    var modelSet = [];
    data.items.forEach(function(i) { var mk = i.model_type || 'autoencoder'; if (modelSet.indexOf(mk) < 0) modelSet.push(mk); });
    var stats = [
      {label:'Toplam',  val: data.items.length,  color:'var(--text)'},
      {label:'Anomali', val: anomalyCount,        color:'var(--danger)'},
      {label:'Normal',  val: normalCount,         color:'var(--accent)'},
      {label:'Model',   val: modelSet.length,     color:'var(--purple)'},
      {label:'Son',     val: (data.items[0] && data.items[0].created_at) ? data.items[0].created_at.split('T')[0] : '-', color:'var(--text2)'},
    ];
    statsBanner.innerHTML = stats.map(function(s) {
      return '<div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px">' +
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--text3)">' + s.label + '</div>' +
        '<div style="font-size:20px;font-weight:800;font-family:\'JetBrains Mono\',monospace;color:' + s.color + ';margin-top:4px">' + s.val + '</div>' +
        '</div>';
    }).join('');

    for (var idx = 0; idx < data.items.length; idx++) {
      (function(item) {
        var tr = document.createElement('tr');
        var bc = item.result === 'ANOMALY' ? 'badge-anomaly' : 'badge-normal';
        var mk = item.model_type || 'autoencoder';
        var mi = MODELS[mk] || {icon: '?', label: mk};
        tr.innerHTML =
          '<td style="font-family:\'JetBrains Mono\',monospace;color:var(--text3)">#' + item.id + '</td>' +
          '<td style="font-weight:500;color:var(--text)">' + item.file_name + '</td>' +
          '<td style="font-size:11px">' + mi.icon + ' ' + mi.label + '</td>' +
          '<td><span style="font-size:11px;color:var(--text3)">' + (item.source || 'upload') + '</span></td>' +
          '<td><span class="badge ' + bc + '">' + item.result + '</span></td>' +
          '<td style="font-family:\'JetBrains Mono\',monospace">' + fmtP(item.anomaly_ratio) + '</td>' +
          '<td style="font-family:\'JetBrains Mono\',monospace">' + item.anomaly_windows + '/' + item.total_windows + '</td>' +
          '<td style="color:var(--text3);font-size:12px">' + item.created_at + '</td>';
        tr.addEventListener('click', function() { openHistoryDetail(item.id, mk); });
        body.appendChild(tr);
      })(data.items[idx]);
    }
  }).catch(function(e) {
    body.innerHTML = '<tr><td colspan="8" style="color:var(--danger);padding:16px">' + e.message + '</td></tr>';
    toast(e.message, 'error');
  });
}

function openHistoryDetail(id, modelKey) {
  modelKey = modelKey || 'autoencoder';
  showLoading('Analiz detayı yükleniyor...');
  fetchJson(API.analysisDetail(id)).then(function(data) {
    hideLoading();
    renderResult(Object.assign({}, data, {analysis_id: id}), data.model_type || modelKey);
    switchSection('analyze');
  }).catch(function(e) {
    hideLoading();
    toast(e.message, 'error');
  });
}

/* ── MODEL INFO ──────────────────────────── */
function loadModelInfo() {
  var c = document.getElementById('modelInfo');
  if (!c) return;
  c.innerHTML = '<div class="loading-placeholder"><div class="loading-spinner" style="width:20px;height:20px"></div> Yükleniyor...</div>';
  fetchJson(API.modelInfo).then(function(d) {
    c.innerHTML = '';
    var topRow = document.createElement('div');
    topRow.style.cssText = 'grid-column:1/-1;display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:4px';
    var topMetrics = [
      {label:'Doğruluk',   val: d.metrics ? fmtP(d.metrics.accuracy)  : '-', color:'var(--accent)'},
      {label:'Kesinlik',   val: d.metrics ? fmtP(d.metrics.precision) : '-', color:'var(--accent)'},
      {label:'Duyarlılık', val: d.metrics ? fmtP(d.metrics.recall)    : '-', color:'var(--accent)'},
      {label:'F1-Skoru',   val: d.metrics ? fmtP(d.metrics.f1_score)  : '-', color:'var(--accent)'},
      {label:'ROC-AUC',    val: d.metrics ? fmtP(d.metrics.roc_auc)   : '-', color:'var(--accent)'},
      {label:'Eşik',       val: fmt(d.threshold, 8), color:'var(--orange)'},
      {label:'Özellik',    val: d.feature_count, color:'var(--blue)'},
    ];
    topRow.innerHTML = topMetrics.map(function(m) {
      return '<div class="report-metric-card">' +
        '<div class="report-metric-label">' + m.label + '</div>' +
        '<div class="report-metric-val" style="color:' + m.color + '">' + m.val + '</div>' +
        '</div>';
    }).join('');
    c.appendChild(topRow);

    c.appendChild(buildModelCard('Temel İstatistikler', [
      {k:'Eşik Değeri',  v: fmt(d.threshold, 8)},
      {k:'Özellik Sayısı', v: d.feature_count},
      {k:'Skor Türü',    v: 'Yeniden Yapılandırma Hatası (MSE)'},
    ]));

    if (d.best_config && d.best_config.config) {
      var cfg = d.best_config.config;
      c.appendChild(buildModelCard('Mimari', [
        {k:'Gizli Katmanlar',  v: (cfg.hidden_layers || []).join(' > ')},
        {k:'Darboğaz Boyutu',  v: cfg.bottleneck_dim},
        {k:'Dropout Oranı',    v: cfg.dropout_rate},
        {k:'Öğrenme Oranı',    v: cfg.learning_rate},
        {k:'Batch Boyutu',     v: cfg.batch_size},
        {k:'Ölçekleyici',      v: cfg.scaler},
      ]));
      c.appendChild(buildModelCard('Eğitim Detayları', [
        {k:'En İyi Deneme',       v: d.best_config.trial},
        {k:'En İyi Val Kaybı',    v: fmt(d.best_config.best_val_loss, 8)},
        {k:'Son Eğitim Kaybı',    v: fmt(d.best_config.final_train_loss, 8)},
        {k:'Eğitim Satırları',    v: d.best_config.train_rows},
        {k:'Doğrulama Satırları', v: d.best_config.validation_rows},
        {k:'Eşik Yüzdeliği',      v: d.best_config.threshold_percentile + '.'},
      ]));
    }

    if (d.metrics && d.metrics.confusion_matrix) {
      var cm = d.metrics.confusion_matrix;
      var cmCard = document.createElement('div'); cmCard.className = 'model-card';
      cmCard.innerHTML = '<div class="model-card-title">Karmaşıklık Matrisi</div>' +
        '<div class="confusion-matrix">' +
        '<div class="cm-label"></div><div class="cm-label">Tahmin: Normal</div><div class="cm-label">Tahmin: Anomali</div>' +
        '<div class="cm-label">Gerçek: Normal</div><div class="cm-cell tp">' + cm[0][0] + '</div><div class="cm-cell fp">' + cm[0][1] + '</div>' +
        '<div class="cm-label">Gerçek: Anomali</div><div class="cm-cell fp">' + cm[1][0] + '</div><div class="cm-cell tp">' + cm[1][1] + '</div>' +
        '</div>' +
        '<div style="margin-top:12px;font-size:11px;color:var(--text3)">DP=' + cm[1][1] + ' | TN=' + cm[0][0] + ' | YP=' + cm[0][1] + ' | YN=' + cm[1][0] + '</div>';
      c.appendChild(cmCard);
    }

    /* ── Autoencoder Görselleri ── */
    var ts = Date.now();
    var aeImages = [
      {file:'loss_curve.png',                   title:'Eğitim Kayıp Eğrisi'},
      {file:'reconstruction_error_histogram.png',title:'Yeniden Yapılandırma Hatası Dağılımı'},
      {file:'confusion_matrix.png',             title:'Karmaşıklık Matrisi Grafiği'},
    ];
    var imagesRow = document.createElement('div'); imagesRow.className = 'report-images-row';
    imagesRow.style.cssText = 'grid-column:1/-1';
    for (var ii = 0; ii < aeImages.length; ii++) {
      var img = aeImages[ii];
      var imgCard = document.createElement('div'); imgCard.className = 'report-image-card';
      imgCard.innerHTML = '<div class="report-image-title">' + img.title + '</div>' +
        '<img src="' + API_BASE + '/outputs/' + img.file + '?t=' + ts + '" alt="' + img.title + '" loading="lazy" onerror="this.parentElement.style.display=\'none\'" />';
      imagesRow.appendChild(imgCard);
    }
    c.appendChild(imagesRow);

    if (d.feature_columns && d.feature_columns.length) {
      var fcCard = document.createElement('div'); fcCard.className = 'model-card'; fcCard.style.gridColumn = '1/-1';
      fcCard.innerHTML = '<div class="model-card-title">Özellik Sütunları (' + d.feature_columns.length + ')</div>' +
        '<div class="model-features">' + d.feature_columns.map(function(f){ return '<span class="feature-tag">' + f + '</span>'; }).join('') + '</div>';
      c.appendChild(fcCard);
    }
  }).catch(function(e) {
    c.innerHTML = '<div style="color:var(--danger);padding:16px">' + e.message + '</div>';
  });
}

function buildModelCard(title, rows) {
  var card = document.createElement('div'); card.className = 'model-card';
  var html = '<div class="model-card-title">' + title + '</div><div class="model-kv">';
  for (var i = 0; i < rows.length; i++) {
    html += '<div class="model-kv-row"><span class="model-kv-key">' + rows[i].k + '</span><span class="model-kv-val">' + (rows[i].v != null ? rows[i].v : '-') + '</span></div>';
  }
  html += '</div>';
  card.innerHTML = html;
  return card;
}

/* ── COMPARE ─────────────────────────────── */
function loadCompare() {
  var c = document.getElementById('compareGrid');
  if (!c) return;
  c.innerHTML = '<div class="loading-placeholder"><div class="loading-spinner" style="width:20px;height:20px"></div> Yükleniyor...</div>';
  fetchJson(API_BASE + '/api/model-comparison').then(function(data) {
    c.innerHTML = '';
    var keys = Object.keys(data);
    var winner = keys[0];
    for (var i = 1; i < keys.length; i++) {
      if ((data[keys[i]].f1_score || 0) > (data[winner].f1_score || 0)) winner = keys[i];
    }
    var wm = MODELS[winner] || {label: winner, icon: '?', color: 'var(--accent)'};
    var banner = document.createElement('div');
    banner.style.cssText = 'grid-column:1/-1;background:var(--accent-glow);border:1px solid rgba(0,212,170,0.25);border-radius:var(--radius);padding:16px 22px;display:flex;align-items:center;gap:14px;margin-bottom:4px';
    banner.innerHTML = '<span style="font-size:28px">' + wm.icon + '</span><div><div style="font-size:12px;color:var(--text3);text-transform:uppercase;letter-spacing:.6px;font-weight:700">En İyi F1-Skoru</div><div style="font-size:18px;font-weight:800;color:var(--accent)">' + wm.label + ' <span style="font-family:\'JetBrains Mono\',monospace;font-size:16px">' + fmtP(data[winner].f1_score) + '</span></div></div>';
    c.appendChild(banner);

    for (var idx = 0; idx < keys.length; idx++) {
      var key = keys[idx];
      var stats = data[key];
      var m = MODELS[key] || {label: key, icon: '?', color: '#888'};
      var cm = stats.confusion_matrix;
      var card = document.createElement('div'); card.className = 'compare-card';
      var metricKeys = ['accuracy','precision','recall','f1_score','roc_auc'];
      var mhtml = '<div class="compare-card-header"><span style="font-size:24px">' + m.icon + '</span><div class="compare-card-name">' + m.label + '</div></div>';
      for (var mk = 0; mk < metricKeys.length; mk++) {
        var kk = metricKeys[mk];
        var klabel = kk.replace('_',' ').toUpperCase();
        mhtml += '<div class="compare-metric-row"><span class="compare-metric-key">' + klabel + '</span><span class="compare-metric-val" style="color:' + m.color + '">' + fmtP(stats[kk]) + '</span></div>';
        mhtml += '<div class="compare-bar-wrap"><div class="compare-bar" style="width:' + ((stats[kk]||0)*100).toFixed(1) + '%;background:' + m.color + '"></div></div>';
      }
      if (cm) {
        mhtml += '<div style="margin-top:8px;padding-top:12px;border-top:1px solid var(--border)"><div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--text3);margin-bottom:8px">Confusion Matrix</div>';
        mhtml += '<div class="confusion-matrix"><div class="cm-label"></div><div class="cm-label" style="font-size:9px">Pred N</div><div class="cm-label" style="font-size:9px">Pred A</div>';
        mhtml += '<div class="cm-label" style="font-size:9px">Act N</div><div class="cm-cell tp" style="font-size:12px">' + cm[0][0] + '</div><div class="cm-cell fp" style="font-size:12px">' + cm[0][1] + '</div>';
        mhtml += '<div class="cm-label" style="font-size:9px">Act A</div><div class="cm-cell fp" style="font-size:12px">' + cm[1][0] + '</div><div class="cm-cell tp" style="font-size:12px">' + cm[1][1] + '</div></div>';
        mhtml += '<div style="font-size:10px;color:var(--text3);margin-top:6px">Threshold: <span style="font-family:\'JetBrains Mono\',monospace;color:' + m.color + '">' + fmt(stats.threshold, 6) + '</span></div></div>';
      }
      card.innerHTML = mhtml;
      c.appendChild(card);
    }
  }).catch(function(e) {
    c.innerHTML = '<div class="compare-placeholder"><p>Karşılaştırma verisi yok. 04b_evaluate_all_models.py scriptini çalıştırın.</p></div>';
  });
}

/* ── MODEL REPORT ─────────────────────────── */
var MODEL_REPORT_CONFIG = {
  isolation_forest: {
    color: '#a78bfa', colorClass: 'val-purple',
    images: [{file:'confusion_matrix.png',title:'Confusion Matrix'},{file:'score_histogram.png',title:'Score Distribution'},{file:'roc_curve.png',title:'ROC Curve'}],
    configKeys: [['model_type','Model Type'],['n_estimators','Estimators'],['contamination','Contamination'],['threshold_percentile','Threshold %'],['threshold','Threshold'],['train_rows','Train Rows'],['val_rows','Val Rows'],['val_anomaly_rate','Val Anomaly Rate']],
  },
  ocsvm: {
    color: '#fb923c', colorClass: 'val-orange',
    images: [{file:'confusion_matrix.png',title:'Confusion Matrix'},{file:'score_histogram.png',title:'Score Distribution'},{file:'roc_curve.png',title:'ROC Curve'}],
    configKeys: [['model_type','Model Type'],['kernel','Kernel'],['nu','Nu'],['gamma','Gamma'],['threshold_percentile','Threshold %'],['threshold','Threshold'],['train_rows_used','Train Rows Used'],['train_rows_total','Train Rows Total'],['val_rows','Val Rows'],['val_anomaly_rate','Val Anomaly Rate']],
  },
  pca: {
    color: '#38bdf8', colorClass: 'val-blue',
    images: [{file:'confusion_matrix.png',title:'Confusion Matrix'},{file:'reconstruction_error_histogram.png',title:'Reconstruction Error'},{file:'roc_curve.png',title:'ROC Curve'},{file:'explained_variance.png',title:'Explained Variance'}],
    configKeys: [['model_type','Model Type'],['n_components_requested','Components Req.'],['n_components_fitted','Components Fit.'],['explained_variance','Explained Var.'],['threshold_percentile','Threshold %'],['threshold','Threshold'],['train_rows','Train Rows'],['val_rows','Val Rows'],['val_anomaly_rate','Val Anomaly Rate']],
  },
};

function loadModelReport(modelKey, containerId) {
  var c = document.getElementById(containerId);
  if (!c) return;
  var cfg = MODEL_REPORT_CONFIG[modelKey];
  if (!cfg) return;
  c.innerHTML = '<div class="loading-placeholder"><div class="loading-spinner" style="width:20px;height:20px"></div> Rapor yükleniyor...</div>';
  fetchJson(API_BASE + '/api/model-metrics/' + modelKey).then(function(d) {
    c.innerHTML = '';
    var metrics = [
      {label:'Doğruluk',   val: fmtP(d.accuracy),  sub: fmt(d.accuracy, 6)},
      {label:'Kesinlik',   val: fmtP(d.precision), sub: fmt(d.precision, 6)},
      {label:'Duyarlılık', val: fmtP(d.recall),    sub: fmt(d.recall, 6)},
      {label:'F1-Skoru',   val: fmtP(d.f1_score),  sub: fmt(d.f1_score, 6)},
      {label:'ROC-AUC',    val: d.roc_auc != null ? fmtP(d.roc_auc) : '-', sub: d.roc_auc != null ? fmt(d.roc_auc, 6) : ''},
      {label:'Eşik',       val: fmt(d.threshold, 8), sub: d.score_type || ''},
      {label:'Test Satırı',val: d.test_rows, sub: 'Normal ' + d.normal_rows + ' / Anomali ' + d.anomaly_rows},
    ];
    if (d.n_components) metrics.push({label:'PCA Comp.', val: d.n_components, sub: 'Var: ' + (d.explained_variance*100).toFixed(1) + '%'});

    var metricsRow = document.createElement('div'); metricsRow.className = 'report-metrics-row';
    metricsRow.innerHTML = metrics.map(function(m) {
      return '<div class="report-metric-card">' +
        '<div class="report-metric-label">' + m.label + '</div>' +
        '<div class="report-metric-val ' + cfg.colorClass + '">' + m.val + '</div>' +
        (m.sub ? '<div class="report-metric-sub">' + m.sub + '</div>' : '') +
        '</div>';
    }).join('');
    c.appendChild(metricsRow);

    var bottomRow = document.createElement('div');
    bottomRow.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;gap:14px';

    if (d.confusion_matrix) {
      var cm = d.confusion_matrix;
      var cmCard = document.createElement('div'); cmCard.className = 'report-cm-card';
      cmCard.innerHTML = '<div class="report-cm-title">Karmaşıklık Matrisi</div>' +
        '<div class="confusion-matrix">' +
        '<div class="cm-label"></div><div class="cm-label">Tahmin: Normal</div><div class="cm-label">Tahmin: Anomali</div>' +
        '<div class="cm-label">Gerçek: Normal</div><div class="cm-cell tp">' + cm[0][0] + '</div><div class="cm-cell fp">' + cm[0][1] + '</div>' +
        '<div class="cm-label">Gerçek: Anomali</div><div class="cm-cell fp">' + cm[1][0] + '</div><div class="cm-cell tp">' + cm[1][1] + '</div>' +
        '</div>';
      bottomRow.appendChild(cmCard);
    }

    if (d.config && Object.keys(d.config).length) {
      var cfgCard = document.createElement('div'); cfgCard.className = 'report-config-card';
      var cfgHtml = '<div class="report-config-title">Model Yapılandırması</div><div class="report-config-kv">';
      for (var ci = 0; ci < cfg.configKeys.length; ci++) {
        var ck = cfg.configKeys[ci][0], cl = cfg.configKeys[ci][1];
        if (d.config[ck] != null) {
          cfgHtml += '<div class="report-config-row"><span class="report-config-key">' + cl + '</span><span class="report-config-val">' + d.config[ck] + '</span></div>';
        }
      }
      cfgHtml += '</div>';
      cfgCard.innerHTML = cfgHtml;
      bottomRow.appendChild(cfgCard);
    }
    if (bottomRow.children.length) c.appendChild(bottomRow);

    var ts = Date.now();
    var imagesRow = document.createElement('div'); imagesRow.className = 'report-images-row';
    for (var ii = 0; ii < cfg.images.length; ii++) {
      var img = cfg.images[ii];
      var imgCard = document.createElement('div'); imgCard.className = 'report-image-card';
      imgCard.innerHTML = '<div class="report-image-title">' + img.title + '</div>' +
        '<img src="' + API_BASE + '/outputs/' + modelKey + '/' + img.file + '?t=' + ts + '" alt="' + img.title + '" loading="lazy" onerror="this.parentElement.style.display=\'none\'" />';
      imagesRow.appendChild(imgCard);
    }
    c.appendChild(imagesRow);

    if (d.classification_report) {
      var clsDiv = document.createElement('div'); clsDiv.className = 'report-classification';
      clsDiv.innerHTML = '<div class="report-classification-title">Sınıflandırma Raporu</div><pre>' + d.classification_report + '</pre>';
      c.appendChild(clsDiv);
    }
  }).catch(function(e) {
    c.innerHTML = '<div class="report-placeholder"><p>' + e.message + '</p><p style="font-size:11px;color:var(--text3)">Önce değerlendirme scriptini çalıştırın.</p></div>';
    toast(e.message, 'error');
  });
}

/* ── EVENTS ──────────────────────────────── */
var closeResult = document.getElementById('closeResult');
if (closeResult) {
  closeResult.addEventListener('click', function() {
    var panel = document.getElementById('resultPanel');
    if (panel) panel.classList.add('hidden');
  });
}

var refreshRealistic = document.getElementById('refreshRealistic');
if (refreshRealistic) refreshRealistic.addEventListener('click', loadRealisticFiles);

var refreshHistory = document.getElementById('refreshHistory');
if (refreshHistory) refreshHistory.addEventListener('click', loadHistory);

var refreshModel = document.getElementById('refreshModel');
if (refreshModel) refreshModel.addEventListener('click', loadModelInfo);

var refreshIF = document.getElementById('refreshIF');
if (refreshIF) refreshIF.addEventListener('click', function() { loadModelReport('isolation_forest', 'ifReport'); });

var refreshOCSVM = document.getElementById('refreshOCSVM');
if (refreshOCSVM) refreshOCSVM.addEventListener('click', function() { loadModelReport('ocsvm', 'ocsvmReport'); });

var refreshPCA = document.getElementById('refreshPCA');
if (refreshPCA) refreshPCA.addEventListener('click', function() { loadModelReport('pca', 'pcaReport'); });

window.addEventListener('resize', function() {
  if (state.lastResult) {
    drawChart(
      state.lastResult.windows || [],
      state.lastResult.threshold,
      (MODELS[state.lastResult._model] || {color: '#00d4aa'}).color
    );
  }
});

var themeBtn = document.getElementById('themeToggleBtn');
if (themeBtn) {
  var t = localStorage.getItem('theme') || 'dark';
  if (t === 'light') document.documentElement.setAttribute('data-theme', 'light');
  themeBtn.addEventListener('click', function() {
    var nt = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
    if (nt === 'light') document.documentElement.setAttribute('data-theme', 'light');
    else document.documentElement.removeAttribute('data-theme');
    localStorage.setItem('theme', nt);
  });
}

checkApi();
loadRealisticFiles();
loadHistory();
loadModelInfo();
