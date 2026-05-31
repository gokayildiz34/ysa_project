function renderRanges(ranges) {
  var el = document.getElementById('rangeList');
  if (!el) return;
  if (!ranges || !ranges.length) {
    el.innerHTML = '<div class="list-item empty">Anormal aralık tespit edilmedi.</div>';
    return;
  }
  var html = '';
  for (var i = 0; i < ranges.length; i++) {
    html += '<div class="list-item anomaly">Aralık ' + (i + 1) + ': pencere ' + ranges[i].start_window + ' - ' + ranges[i].end_window + '</div>';
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

function drawChart(windows, threshold, lineColor) {
  lineColor = lineColor || '#00d4aa';
  var canvas = document.getElementById('errorChart');
  if (!canvas) return;
  var dpr = window.devicePixelRatio || 1;
  var W_css = canvas.parentElement.getBoundingClientRect().width;
  var H_css = 260;
  canvas.width = W_css * dpr;
  canvas.height = H_css * dpr;
  canvas.style.width = W_css + 'px';
  canvas.style.height = H_css + 'px';

  var ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W_css, H_css);
  if (!windows || !windows.length) return;

  var pad = { top: 20, right: 20, bottom: 36, left: 56 };
  var pw = W_css - pad.left - pad.right;
  var ph = H_css - pad.top - pad.bottom;

  var errors = windows.map(function (w) { return w.reconstruction_error; });
  var maxE = Math.max.apply(null, errors.concat([threshold || 0])) * 1.1;
  var minId = Math.min.apply(null, windows.map(function (w) { return w.window_id; }));
  var maxId = Math.max.apply(null, windows.map(function (w) { return w.window_id; }));
  var span = Math.max(1, maxId - minId);

  function xS(id) { return pad.left + ((id - minId) / span) * pw; }
  function yS(v) { return pad.top + ph - (v / maxE) * ph; }

  ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
  for (var i = 0; i <= 4; i++) {
    var y = pad.top + (ph / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pw, y); ctx.stroke();
    ctx.fillStyle = 'rgba(155,171,194,0.55)';
    ctx.font = "10px 'JetBrains Mono',monospace"; ctx.textAlign = 'right';
    ctx.fillText((maxE * (1 - i / 4)).toExponential(1), pad.left - 6, y + 4);
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
    var w = windows[k], wn = windows[k + 1];
    var x1 = xS(w.window_id), y1 = yS(w.reconstruction_error);
    var x2 = xS(wn.window_id), y2 = yS(wn.reconstruction_error);
    var by = pad.top + ph;
    ctx.beginPath(); ctx.moveTo(x1, by); ctx.lineTo(x1, y1); ctx.lineTo(x2, y2); ctx.lineTo(x2, by); ctx.closePath();
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

function loadHistory() {
  var body = document.getElementById('historyBody');
  if (!body) return;
  body.innerHTML = '<tr><td colspan="8" style="text-align:center;color:var(--text3);padding:20px">Yükleniyor...</td></tr>';
  fetchJson(API.analyses).then(function (data) {
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
    var anomalyCount = data.items.filter(function (i) { return i.result === 'ANOMALY'; }).length;
    var normalCount = data.items.filter(function (i) { return i.result === 'NORMAL'; }).length;
    var modelSet = [];
    data.items.forEach(function (i) { var mk = i.model_type || 'autoencoder'; if (modelSet.indexOf(mk) < 0) modelSet.push(mk); });
    var stats = [
      { label: 'Toplam', val: data.items.length, color: 'var(--text)' },
      { label: 'Anomali', val: anomalyCount, color: 'var(--danger)' },
      { label: 'Normal', val: normalCount, color: 'var(--accent)' },
      { label: 'Model', val: modelSet.length, color: 'var(--purple)' },
      { label: 'Son', val: (data.items[0] && data.items[0].created_at) ? data.items[0].created_at.split('T')[0] : '-', color: 'var(--text2)' },
    ];
    statsBanner.innerHTML = stats.map(function (s) {
      return '<div style="background:var(--surface2);border:1px solid var(--border);border-radius:var(--radius);padding:14px 16px">' +
        '<div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--text3)">' + s.label + '</div>' +
        '<div style="font-size:20px;font-weight:800;font-family:\'JetBrains Mono\',monospace;color:' + s.color + ';margin-top:4px">' + s.val + '</div>' +
        '</div>';
    }).join('');

    for (var idx = 0; idx < data.items.length; idx++) {
      (function (item) {
        var tr = document.createElement('tr');
        var bc = item.result === 'ANOMALY' ? 'badge-anomaly' : 'badge-normal';
        var mk = item.model_type || 'autoencoder';
        var mi = MODELS[mk] || { icon: '?', label: mk };
        tr.innerHTML =
          '<td style="font-family:\'JetBrains Mono\',monospace;color:var(--text3)">#' + item.id + '</td>' +
          '<td style="font-weight:500;color:var(--text)">' + item.file_name + '</td>' +
          '<td style="font-size:11px">' + mi.icon + ' ' + mi.label + '</td>' +
          '<td><span style="font-size:11px;color:var(--text3)">' + (item.source || 'upload') + '</span></td>' +
          '<td><span class="badge ' + bc + '">' + item.result + '</span></td>' +
          '<td style="font-family:\'JetBrains Mono\',monospace">' + fmtP(item.anomaly_ratio) + '</td>' +
          '<td style="font-family:\'JetBrains Mono\',monospace">' + item.anomaly_windows + '/' + item.total_windows + '</td>' +
          '<td style="color:var(--text3);font-size:12px">' + item.created_at + '</td>';
        tr.addEventListener('click', function () { openHistoryDetail(item.id, mk); });
        body.appendChild(tr);
      })(data.items[idx]);
    }
  }).catch(function (e) {
    body.innerHTML = '<tr><td colspan="8" style="color:var(--danger);padding:16px">' + e.message + '</td></tr>';
    toast(e.message, 'error');
  });
}