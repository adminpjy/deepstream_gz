"use strict";

(() => {
  const API = Object.freeze({
    start: "/api/v1/recognition/sessions/start",
    sessions: "/api/v1/recognition/sessions",
    capabilities: "/api/v1/recognition/capabilities",
    service: "/api/v1/recognition/service",
  });
  const state = {
    initialized: false,
    submitting: false,
    sessions: [],
    capabilities: null,
    pollTimer: null,
  };

  document.addEventListener("DOMContentLoaded", initializeProductionUi);

  function initializeProductionUi() {
    if (state.initialized) return;
    const form = document.getElementById("taskForm");
    const rtspPanel = document.getElementById("rtspSourcePanel");
    if (!form || !rtspPanel) return;
    state.initialized = true;
    installStyles();
    injectFeatureControls(form);
    injectSessionPanel();
    form.addEventListener("submit", handleProductionSubmit, true);
    void refreshCapabilities();
    void refreshProductionSessions();
    state.pollTimer = window.setInterval(() => {
      if (!document.hidden) void refreshProductionSessions();
    }, 3000);
    window.addEventListener("pagehide", () => {
      if (state.pollTimer !== null) window.clearInterval(state.pollTimer);
    });
  }

  function installStyles() {
    const style = document.createElement("style");
    style.textContent = `
      .production-feature-panel{margin:16px 0;padding:16px;border:1px solid rgba(116,229,255,.18);border-radius:12px;background:rgba(7,24,30,.56)}
      .production-feature-panel h3{margin:0 0 12px;font-size:14px;letter-spacing:.04em}
      .production-core-grid,.production-optional-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:8px 0 14px}
      .production-check{display:flex;align-items:center;gap:8px;font-size:13px;min-height:30px}
      .production-check input{accent-color:#54d9ff}
      .production-check.unavailable{opacity:.48}
      .production-baseline{display:grid;gap:7px;margin-top:10px;padding-top:12px;border-top:1px solid rgba(116,229,255,.12)}
      .production-baseline input[type=file]{width:100%;font-size:12px}
      .production-hint{margin:0;color:var(--muted,#8aa4ad);font-size:12px;line-height:1.5}
      .production-session-panel{margin-top:20px}
      .production-session-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:12px;padding:16px}
      .production-session-card{border:1px solid rgba(116,229,255,.16);border-radius:12px;overflow:hidden;background:rgba(5,18,24,.72)}
      .production-session-card img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;background:#02080b}
      .production-session-meta{padding:12px;display:grid;gap:7px;font-size:12px}
      .production-session-title{display:flex;justify-content:space-between;gap:8px;font-size:13px;font-weight:700}
      .production-session-tags{display:flex;flex-wrap:wrap;gap:5px}
      .production-session-tags span{border:1px solid rgba(116,229,255,.15);border-radius:999px;padding:3px 7px;color:var(--muted,#9db2ba)}
      .production-session-actions{display:flex;justify-content:flex-end;margin-top:4px}
      .production-empty{padding:20px;color:var(--muted,#9db2ba);text-align:center}
      @media(max-width:700px){.production-core-grid,.production-optional-grid{grid-template-columns:1fr}}
    `;
    document.head.appendChild(style);
  }

  function injectFeatureControls(form) {
    const submitButton = document.getElementById("startTaskButton");
    if (!submitButton) return;
    const panel = document.createElement("section");
    panel.id = "productionFeaturePanel";
    panel.className = "production-feature-panel";
    panel.innerHTML = `
      <h3>生产识别能力（RTSP）</h3>
      <p class="production-hint">基础识别固定启用，不能关闭；场景识别按本次启动参数独立启用。</p>
      <div class="production-core-grid" aria-label="固定基础识别">
        ${fixedCheck("人形检测", "prodCorePerson")}
        ${fixedCheck("人员跟踪", "prodCoreTrack")}
        ${fixedCheck("人脸检测/跟踪", "prodCoreFace")}
        ${fixedCheck("人脸识别", "prodCoreRecognition")}
      </div>
      <strong>场景识别</strong>
      <div class="production-optional-grid">
        ${optionalCheck("吸烟", "prodSmoking")}
        ${optionalCheck("吃东西", "prodEating")}
        ${optionalCheck("喝水", "prodDrinking")}
        ${optionalCheck("物品遗留", "prodLeftObject")}
        <label class="production-check unavailable" title="已预留，当前版本尚未实现"><input id="prodLargeObject" type="checkbox" disabled />大件物品搬运（预留）</label>
      </div>
      <div class="production-baseline" id="prodBaselineArea" hidden>
        <label for="prodBaselineFile"><strong>人员进入前正常场景图片</strong></label>
        <input id="prodBaselineFile" type="file" accept="image/jpeg,image/png,image/webp" />
        <p class="production-hint">测试页面可上传；正式 CPU 静态分析服务也可调用摄像头 baseline REST 接口。若该摄像头已有基准图，可不重复上传。</p>
        <div class="form-row">
          <label class="field"><span>差异像素阈值</span><input id="prodPixelThreshold" type="number" min="1" max="255" step="1" value="28" /></label>
          <label class="field"><span>最小变化比例</span><input id="prodMinAreaRatio" type="number" min="0.0001" max="0.5" step="0.0001" value="0.0015" /></label>
        </div>
      </div>
    `;
    submitButton.parentNode.insertBefore(panel, submitButton);
    document.getElementById("prodLeftObject").addEventListener("change", (event) => {
      document.getElementById("prodBaselineArea").hidden = !event.target.checked;
    });
    const idle = document.getElementById("idleTimeout");
    const label = idle && idle.closest("label");
    const labelText = label && label.querySelector(":scope > span:first-child");
    if (labelText) labelText.textContent = "人员消失自动退出";
  }

  function fixedCheck(label, id) {
    return `<label class="production-check"><input id="${id}" type="checkbox" checked disabled />${escapeHtml(label)}</label>`;
  }

  function optionalCheck(label, id) {
    return `<label class="production-check" id="${id}Label"><input id="${id}" type="checkbox" />${escapeHtml(label)}</label>`;
  }

  function injectSessionPanel() {
    const main = document.querySelector("main");
    if (!main) return;
    const panel = document.createElement("section");
    panel.className = "panel production-session-panel";
    panel.setAttribute("aria-labelledby", "productionSessionsTitle");
    panel.innerHTML = `
      <div class="panel-heading tasks-heading">
        <div><p class="section-index">04 / PRODUCTION</p><h2 id="productionSessionsTitle">生产 GPU Session</h2></div>
        <div class="tasks-tools"><span id="productionGpuSummary">GPU 初始化中</span><button class="button button-ghost button-compact" id="refreshProductionSessions" type="button"><span class="button-icon" aria-hidden="true">↻</span>刷新</button></div>
      </div>
      <div id="productionSessionGrid" class="production-session-grid"></div>
    `;
    main.appendChild(panel);
    document.getElementById("refreshProductionSessions").addEventListener("click", () => void refreshProductionSessions(true));
  }

  async function refreshCapabilities() {
    try {
      const capabilities = await requestJson(API.capabilities);
      state.capabilities = capabilities;
      for (const [feature, id] of [["smoking","prodSmoking"],["eating","prodEating"],["drinking","prodDrinking"]]) {
        const info = capabilities.optional && capabilities.optional[feature];
        const input = document.getElementById(id);
        const label = document.getElementById(`${id}Label`);
        if (!input || !label || !info) continue;
        if (!info.available) {
          input.checked = false;
          input.disabled = true;
          label.classList.add("unavailable");
          label.title = `模型不可用：${info.reason || "未部署"}`;
        } else {
          input.disabled = false;
          label.classList.remove("unavailable");
          label.title = "";
        }
      }
    } catch (error) {
      showProductionError(`无法读取生产能力：${errorMessage(error)}`);
    }
  }

  async function handleProductionSubmit(event) {
    const rtspPanel = document.getElementById("rtspSourcePanel");
    if (!rtspPanel || rtspPanel.hidden) return; // local-file tests remain on legacy API
    event.preventDefault();
    event.stopImmediatePropagation();
    if (state.submitting) return;
    state.submitting = true;
    const button = document.getElementById("startTaskButton");
    const buttonText = document.getElementById("startTaskButtonText");
    const status = document.getElementById("submissionStatus");
    if (button) button.disabled = true;
    if (buttonText) buttonText.textContent = "正在启动生产识别…";
    if (status) status.textContent = "正在分配 GPU 并动态接入视频流…";
    try {
      const payload = buildProductionPayload();
      if (payload.features.leftObject) {
        const fileInput = document.getElementById("prodBaselineFile");
        const file = fileInput && fileInput.files && fileInput.files[0];
        if (file) await uploadBaseline(payload.cameraId, file);
      }
      const created = await requestJson(API.start, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      if (status) status.textContent = `已启动：GPU ${created.gpuId} · Session ${created.sessionId}`;
      showProductionToast(`生产识别已启动 · GPU ${created.gpuId}`);
      incrementCameraId();
      await refreshProductionSessions(true);
    } catch (error) {
      const message = errorMessage(error);
      if (status) status.textContent = message;
      showProductionError(message);
    } finally {
      state.submitting = false;
      if (button) button.disabled = false;
      if (buttonText) buttonText.textContent = "开始分析";
    }
  }

  function buildProductionPayload() {
    const cameraId = document.getElementById("cameraId").value.trim();
    const streamUrl = document.getElementById("rtspUrl").value.trim();
    const nominalFps = boundedNumber(document.getElementById("nominalFps").value, "视频帧率", 0.1, 240);
    const personAbsentSeconds = boundedNumber(document.getElementById("idleTimeout").value, "人员消失时间", 1, 3600);
    if (!/^[A-Za-z0-9_.-]{1,64}$/.test(cameraId)) throw new Error("摄像头编号格式无效。");
    if (!/^rtsps?:\/\//i.test(streamUrl)) throw new Error("请输入有效 RTSP 地址。");
    return {
      cameraId,
      streamUrl,
      nominalFps,
      features: {
        smoking: Boolean(document.getElementById("prodSmoking").checked),
        eating: Boolean(document.getElementById("prodEating").checked),
        drinking: Boolean(document.getElementById("prodDrinking").checked),
        leftObject: Boolean(document.getElementById("prodLeftObject").checked),
        largeObjectMoving: false,
      },
      exitPolicy: {personAbsentSeconds},
      leftObject: {
        pixelThreshold: boundedNumber(document.getElementById("prodPixelThreshold").value, "差异像素阈值", 1, 255),
        minAreaRatio: boundedNumber(document.getElementById("prodMinAreaRatio").value, "最小变化比例", 0.0001, 0.5),
        minComponentAreaRatio: 0.00035,
        confirmFrames: 3,
        maxRecentFrames: 8,
      },
      context: {source: "test_web"},
    };
  }

  async function uploadBaseline(cameraId, file) {
    if (!(file instanceof File) || file.size <= 0) throw new Error("正常场景图片为空。");
    const contentType = file.type || "application/octet-stream";
    await requestJson(`/api/v1/recognition/cameras/${encodeURIComponent(cameraId)}/baseline`, {
      method: "POST",
      headers: {"Content-Type": contentType},
      body: file,
    });
  }

  async function refreshProductionSessions(surfaceErrors = false) {
    try {
      const [sessions, service] = await Promise.all([
        requestJson(API.sessions),
        requestJson(API.service),
      ]);
      state.sessions = Array.isArray(sessions.sessions) ? sessions.sessions : [];
      const summary = document.getElementById("productionGpuSummary");
      if (summary) summary.textContent = `${service.readyGpuCount || 0}/${service.gpuCount || 0} GPU READY · ${service.activeSessions || 0} 路活动`;
      renderProductionSessions();
    } catch (error) {
      if (surfaceErrors) showProductionError(errorMessage(error));
    }
  }

  function renderProductionSessions() {
    const grid = document.getElementById("productionSessionGrid");
    if (!grid) return;
    const active = state.sessions.filter((item) => ["starting","active","stopping"].includes(String(item.state || "")));
    if (!active.length) {
      grid.innerHTML = `<div class="production-empty">暂无生产 RTSP Session。切换到 RTSP，选择识别场景并点击“开始分析”。</div>`;
      return;
    }
    grid.innerHTML = active.map((item) => {
      const features = Object.entries(item.features || {}).filter(([, enabled]) => enabled).map(([name]) => featureLabel(name));
      const preview = item.previewUrl || `/api/v1/recognition/sessions/${encodeURIComponent(item.sessionId)}/preview.jpg`;
      return `<article class="production-session-card" data-session-id="${escapeHtml(item.sessionId)}">
        <img src="${escapeHtml(preview)}?t=${Date.now()}" alt="${escapeHtml(item.cameraId || "摄像头")} 实时预览" loading="lazy" />
        <div class="production-session-meta">
          <div class="production-session-title"><span>${escapeHtml(item.cameraId || "-")}</span><span>${escapeHtml(String(item.state || "-"))}</span></div>
          <div>GPU ${escapeHtml(String(item.gpuId ?? "-"))} · Session ${escapeHtml(item.sessionId || "-")}</div>
          <div>无人计时 ${Number(item.idleSeconds || 0).toFixed(1)}s / ${Number((item.exitPolicy || {}).personAbsentSeconds || 0).toFixed(0)}s</div>
          <div class="production-session-tags">${features.length ? features.map((name) => `<span>${escapeHtml(name)}</span>`).join("") : "<span>仅基础识别</span>"}</div>
          <div class="production-session-actions"><button type="button" class="button button-ghost button-compact production-stop" data-session-id="${escapeHtml(item.sessionId)}">停止识别</button></div>
        </div>
      </article>`;
    }).join("");
    grid.querySelectorAll(".production-stop").forEach((button) => {
      button.addEventListener("click", () => void stopProductionSession(button.dataset.sessionId, button));
    });
  }

  async function stopProductionSession(sessionId, button) {
    if (!sessionId || button.disabled) return;
    button.disabled = true;
    try {
      await requestJson(`/api/v1/recognition/sessions/${encodeURIComponent(sessionId)}/stop`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({reason: "test_web_requested"}),
      });
      showProductionToast("生产识别已停止");
      await refreshProductionSessions(true);
    } catch (error) {
      showProductionError(errorMessage(error));
      button.disabled = false;
    }
  }

  async function requestJson(url, options = {}) {
    const response = await fetch(url, {cache: "no-store", ...options});
    let data = null;
    try { data = await response.json(); } catch (_) { data = null; }
    if (!response.ok) {
      const error = new Error((data && data.error) || `HTTP ${response.status}`);
      error.code = data && data.code;
      error.detail = data && data.detail;
      throw error;
    }
    return data || {};
  }

  function boundedNumber(raw, label, min, max) {
    const value = Number(raw);
    if (!Number.isFinite(value) || value < min || value > max) throw new Error(`${label}必须在 ${min} 到 ${max} 之间。`);
    return value;
  }

  function featureLabel(value) {
    return ({smoking:"吸烟",eating:"吃东西",drinking:"喝水",leftObject:"物品遗留",largeObjectMoving:"大件搬运"})[value] || value;
  }

  function incrementCameraId() {
    const input = document.getElementById("cameraId");
    const match = /^camera-(\d+)$/i.exec(input.value.trim());
    if (!match) return;
    input.value = `camera-${String(Number(match[1]) + 1).padStart(Math.max(2, match[1].length), "0")}`;
  }

  function showProductionError(message) {
    const banner = document.getElementById("errorBanner");
    const target = document.getElementById("errorMessage");
    if (target) target.textContent = message;
    if (banner) banner.hidden = false;
    showProductionToast(message, true);
  }

  function showProductionToast(message, error = false) {
    const region = document.getElementById("toastRegion");
    if (!region) return;
    const toast = document.createElement("div");
    toast.className = `toast${error ? " is-error" : ""}`;
    toast.textContent = message;
    region.appendChild(toast);
    window.setTimeout(() => toast.remove(), 4500);
  }

  function errorMessage(error) {
    if (!(error instanceof Error)) return String(error || "未知错误");
    if (error.code === "SESSION_ALREADY_ACTIVE" && error.detail && error.detail.sessionId) return `${error.message}（Session ${error.detail.sessionId}）`;
    return error.message;
  }

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"})[char]);
  }
})();
