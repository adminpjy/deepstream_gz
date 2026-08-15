"use strict";

(() => {
  const API = Object.freeze({
    start: "/api/v1/recognition/sessions/start",
    sessions: "/api/v1/recognition/sessions",
    capabilities: "/api/v1/recognition/capabilities",
    service: "/api/v1/recognition/service",
  });
  const MAX_LIVE_TILES = 4;
  const state = {
    initialized: false,
    submitting: false,
    sessions: [],
    capabilities: null,
    pollTimer: null,
    previewNodes: new Map(),
    legacyPreviewObserver: null,
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
    installLiveWallBridge();
    form.addEventListener("submit", handleProductionSubmit, true);
    void refreshCapabilities();
    void refreshProductionSessions();
    state.pollTimer = window.setInterval(() => {
      if (!document.hidden) void refreshProductionSessions();
    }, 3000);
    window.addEventListener("pagehide", () => {
      if (state.pollTimer !== null) window.clearInterval(state.pollTimer);
      if (state.legacyPreviewObserver) state.legacyPreviewObserver.disconnect();
      for (const entry of state.previewNodes.values()) disconnectProductionPreview(entry);
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
      .production-live-meta{display:flex;flex-wrap:wrap;gap:6px;padding:8px 2px 2px;color:var(--muted,#9db2ba);font-size:11px}
      .production-live-meta span{border:1px solid rgba(116,229,255,.15);border-radius:999px;padding:3px 7px}
      #productionPreviewGridBody:empty{display:none}
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
        ${optionalCheck("打电话", "prodPhone")}
        ${optionalCheck("物品遗留", "prodLeftObject")}
        <label class="production-check unavailable" title="火焰模型为全画面分类模型，当前生产 Session 暂未接入"><input id="prodFire" type="checkbox" disabled />火焰（待接入）</label>
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

  function installLiveWallBridge() {
    const legacyBody = document.getElementById("previewGridBody");
    const productionBody = document.getElementById("productionPreviewGridBody");
    if (!legacyBody || !productionBody) return;
    state.legacyPreviewObserver = new MutationObserver(() => {
      renderProductionLiveWall();
      syncLiveWallIndicator();
    });
    state.legacyPreviewObserver.observe(legacyBody, {childList: true, subtree: true});
    syncLiveWallIndicator();
  }

  async function refreshCapabilities() {
    try {
      const capabilities = await requestJson(API.capabilities);
      state.capabilities = capabilities;
      for (const [feature, id] of [["smoking","prodSmoking"],["eating","prodEating"],["drinking","prodDrinking"],["phone","prodPhone"]]) {
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
        phone: Boolean(document.getElementById("prodPhone").checked),
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
      const [sessions] = await Promise.all([
        requestJson(API.sessions),
        requestJson(API.service),
      ]);
      state.sessions = Array.isArray(sessions.sessions) ? sessions.sessions : [];
      renderProductionLiveWall();
      syncLiveWallIndicator();
    } catch (error) {
      if (surfaceErrors) showProductionError(errorMessage(error));
    }
  }

  function renderProductionLiveWall() {
    const body = document.getElementById("productionPreviewGridBody");
    if (!body) return;
    const active = state.sessions.filter((item) => ["starting","active","stopping"].includes(String(item.state || "")));
    const legacyCount = legacyPreviewCount();
    const visible = active.slice(0, Math.max(0, MAX_LIVE_TILES - legacyCount));
    const visibleIds = new Set(visible.map((item) => String(item.sessionId || "")));

    for (const [sessionId, entry] of state.previewNodes.entries()) {
      if (!visibleIds.has(sessionId)) {
        disconnectProductionPreview(entry);
        entry.cell.remove();
        state.previewNodes.delete(sessionId);
      }
    }

    for (const item of visible) {
      const sessionId = String(item.sessionId || "");
      if (!sessionId) continue;
      let entry = state.previewNodes.get(sessionId);
      if (!entry) {
        entry = createProductionPreviewEntry(item);
        state.previewNodes.set(sessionId, entry);
      }
      updateProductionPreviewEntry(entry, item);
    }

    body.replaceChildren();
    const entries = visible.map((item) => state.previewNodes.get(String(item.sessionId || ""))).filter(Boolean);
    for (let index = 0; index < entries.length; index += 2) {
      const row = document.createElement("tr");
      row.append(entries[index].cell);
      if (entries[index + 1]) row.append(entries[index + 1].cell);
      else row.append(document.createElement("td"));
      body.append(row);
    }
    body.dataset.count = String(entries.length);
  }

  function createProductionPreviewEntry(item) {
    const sessionId = String(item.sessionId || "");
    const cell = document.createElement("td");
    cell.dataset.productionSessionId = sessionId;

    const heading = document.createElement("div");
    heading.className = "selected-task-summary";
    const camera = document.createElement("div");
    const cameraLabel = document.createElement("span");
    cameraLabel.textContent = "摄像头";
    const cameraValue = document.createElement("strong");
    camera.append(cameraLabel, cameraValue);
    const gpu = document.createElement("div");
    const gpuLabel = document.createElement("span");
    gpuLabel.textContent = "生产 GPU";
    const gpuValue = document.createElement("strong");
    gpu.append(gpuLabel, gpuValue);
    const stop = document.createElement("button");
    stop.type = "button";
    stop.className = "button button-danger button-compact";
    stop.textContent = "■ 停止";
    stop.addEventListener("click", () => void stopProductionSession(sessionId, stop));
    heading.append(camera, gpu, stop);

    const stage = document.createElement("div");
    stage.className = "preview-stage";
    const image = document.createElement("img");
    image.alt = `${item.cameraId || sessionId} 实时生产分析画面`;
    const placeholder = document.createElement("div");
    placeholder.className = "preview-placeholder";
    placeholder.innerHTML = "<strong>连接中</strong><span>正在等待生产 MJPEG 画面…</span>";
    const error = document.createElement("div");
    error.className = "preview-error";
    error.hidden = true;
    error.innerHTML = "<strong>预览暂时不可用</strong><span>将在 Session 状态刷新后重新连接</span>";
    stage.append(image, placeholder, error);

    const meta = document.createElement("div");
    meta.className = "production-live-meta";
    cell.append(heading, stage, meta);

    const entry = {sessionId, cell, cameraValue, gpuValue, stop, image, placeholder, error, meta, url: null};
    image.addEventListener("load", () => {
      entry.placeholder.hidden = true;
      entry.error.hidden = true;
      entry.image.hidden = false;
    });
    image.addEventListener("error", () => {
      entry.image.hidden = true;
      entry.placeholder.hidden = true;
      entry.error.hidden = false;
      entry.url = null;
    });
    return entry;
  }

  function updateProductionPreviewEntry(entry, item) {
    entry.cameraValue.textContent = item.cameraId || "—";
    entry.gpuValue.textContent = `GPU ${item.gpuId ?? "—"}`;
    entry.stop.disabled = String(item.state || "") === "stopping";
    const features = Object.entries(item.features || {}).filter(([, enabled]) => enabled).map(([name]) => featureLabel(name));
    entry.meta.replaceChildren();
    for (const text of [`Session ${shortId(item.sessionId)}`, ...features]) {
      const badge = document.createElement("span");
      badge.textContent = text;
      entry.meta.append(badge);
    }
    if (!features.length) {
      const badge = document.createElement("span");
      badge.textContent = "仅基础识别";
      entry.meta.append(badge);
    }
    const url = `/api/v1/recognition/sessions/${encodeURIComponent(item.sessionId)}/stream.mjpg`;
    if (entry.url !== url || !entry.image.getAttribute("src")) {
      entry.url = url;
      entry.error.hidden = true;
      entry.placeholder.hidden = false;
      entry.image.hidden = false;
      entry.image.src = `${url}?t=${Date.now()}`;
    }
  }

  function disconnectProductionPreview(entry) {
    if (!entry) return;
    entry.image.removeAttribute("src");
    entry.url = null;
  }

  function legacyPreviewCount() {
    const body = document.getElementById("previewGridBody");
    if (!body) return 0;
    return body.querySelectorAll("td .preview-stage").length;
  }

  function syncLiveWallIndicator() {
    const productionBody = document.getElementById("productionPreviewGridBody");
    const productionCount = Number(productionBody && productionBody.dataset.count || 0);
    const total = legacyPreviewCount() + productionCount;
    const empty = document.getElementById("previewEmpty");
    const indicator = document.getElementById("liveIndicator");
    if (empty) empty.hidden = total > 0;
    if (indicator) {
      indicator.dataset.active = total > 0 ? "true" : "false";
      const text = indicator.querySelector("span");
      if (text) text.textContent = `${total} 路预览`;
    }
  }

  async function stopProductionSession(sessionId, button) {
    if (!sessionId || (button && button.disabled)) return;
    if (button) button.disabled = true;
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
      if (button) button.disabled = false;
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
    return ({smoking:"吸烟",eating:"吃东西",drinking:"喝水",phone:"打电话",leftObject:"物品遗留",largeObjectMoving:"大件搬运"})[value] || value;
  }

  function shortId(value) {
    const text = String(value || "");
    return text.length <= 8 ? text : `${text.slice(0, 8)}…`;
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