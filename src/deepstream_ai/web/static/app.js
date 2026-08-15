"use strict";

const API = Object.freeze({ service: "/api/service", tasks: "/api/tasks", uploads: "/api/uploads", restart: "/api/service/restart" });
const POLL_INTERVAL_MS = 3000;
// The service defaults to four concurrent pipelines. Keep the browser preview
// wall at the same bound so persistent MJPEG connections never starve normal
// API requests on browsers with conservative per-origin connection limits.
const MAX_PREVIEW_TILES = 4;

const state = {
  sourceMode: "file",
  selectedFile: null,
  tasks: [],
  submitting: false,
  refreshing: false,
  stoppingTaskIds: new Set(),
  startingTaskIds: new Set(),
  pollingTimer: null,
  previewNodes: new Map(),
};

const elements = {};
document.addEventListener("DOMContentLoaded", initialize);

function initialize() {
  collectElements();
  bindEvents();
  // Page load is read-only. No historical task is started from initialize().
  void refreshDashboard({ surfaceErrors: true });
  schedulePolling();
}

function collectElements() {
  const ids = [
    "servicePill", "serviceStatusText", "serviceDetail", "restartServiceButton",
    "errorBanner", "errorMessage", "dismissErrorButton", "runningCount", "pendingCount", "totalCount",
    "fileTab", "rtspTab", "fileSourcePanel", "rtspSourcePanel", "taskForm", "dropZone", "videoFile",
    "fileSelection", "selectedFileName", "selectedFileMeta", "clearFileButton", "rtspUrl", "nominalFps",
    "cameraId", "idleTimeout", "startTaskButton", "startTaskButtonText", "submissionStatus", "liveIndicator",
    "previewGridBody", "previewEmpty", "lastUpdatedText", "refreshTasksButton", "tasksTableBody", "tasksEmpty", "toastRegion",
  ];
  for (const id of ids) {
    const node = document.getElementById(id);
    if (!node) throw new Error(`页面缺少必要元素: #${id}`);
    elements[id] = node;
  }
}

function bindEvents() {
  elements.fileTab.addEventListener("click", () => setSourceMode("file"));
  elements.rtspTab.addEventListener("click", () => setSourceMode("rtsp"));
  elements.videoFile.addEventListener("change", () => setSelectedFile(elements.videoFile.files && elements.videoFile.files[0]));
  elements.clearFileButton.addEventListener("click", clearSelectedFile);
  elements.taskForm.addEventListener("submit", handleTaskSubmit);
  elements.dismissErrorButton.addEventListener("click", clearError);
  elements.refreshTasksButton.addEventListener("click", () => void refreshDashboard({ surfaceErrors: true, force: true }));
  elements.restartServiceButton.addEventListener("click", () => void restartService());

  for (const eventName of ["dragenter", "dragover"]) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      elements.dropZone.classList.add("is-dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.remove("is-dragging");
    });
  }
  elements.dropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) setSelectedFile(file);
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopPolling();
    else {
      void refreshDashboard({ surfaceErrors: false, force: true });
      schedulePolling();
    }
  });
  window.addEventListener("pagehide", () => {
    stopPolling();
    for (const entry of state.previewNodes.values()) disconnectPreview(entry);
  });
}

function setSourceMode(mode) {
  if (!["file", "rtsp"].includes(mode)) return;
  state.sourceMode = mode;
  const fileActive = mode === "file";
  elements.fileTab.classList.toggle("is-active", fileActive);
  elements.fileTab.setAttribute("aria-selected", String(fileActive));
  elements.rtspTab.classList.toggle("is-active", !fileActive);
  elements.rtspTab.setAttribute("aria-selected", String(!fileActive));
  elements.fileSourcePanel.hidden = !fileActive;
  elements.rtspSourcePanel.hidden = fileActive;
  clearSubmissionStatus();
}

function setSelectedFile(file) {
  if (!(file instanceof File)) return clearSelectedFile();
  state.selectedFile = file;
  elements.dropZone.hidden = true;
  elements.fileSelection.hidden = false;
  elements.selectedFileName.textContent = file.name || "未命名视频";
  elements.selectedFileMeta.textContent = `${formatBytes(file.size)} · ${file.type || "类型未知"}`;
  clearSubmissionStatus();
}

function clearSelectedFile() {
  state.selectedFile = null;
  elements.videoFile.value = "";
  elements.dropZone.hidden = false;
  elements.fileSelection.hidden = true;
  elements.selectedFileName.textContent = "";
  elements.selectedFileMeta.textContent = "";
}

async function handleTaskSubmit(event) {
  event.preventDefault();
  if (state.submitting) return;
  clearError();
  clearSubmissionStatus();
  setSubmitting(true, state.sourceMode === "file" && state.selectedFile ? `正在上传 ${shortMessage(state.selectedFile.name, 24)}…` : "正在准备启动…");
  try {
    const payload = await buildTaskPayload();
    setSubmitting(true, "正在启动分析…");
    await requestJson(API.tasks, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setSubmissionStatus("已手动启动，可继续添加下一路视频。", "success");
    showToast("分析任务已启动");
    if (state.sourceMode === "file") clearSelectedFile();
    incrementCameraId();
    await refreshDashboard({ surfaceErrors: false, force: true });
  } catch (error) {
    const message = errorMessage(error);
    showError(message);
    setSubmissionStatus(message, "error");
  } finally {
    setSubmitting(false);
  }
}

function buildStableBehaviorFeatures() {
  const selected = (id) => {
    const input = document.getElementById(id);
    return Boolean(input && input.checked && !input.disabled);
  };
  return {
    smoking: selected("prodSmoking"),
    eating: selected("prodEating"),
    drinking: selected("prodDrinking"),
  };
}

async function buildTaskPayload() {
  const cameraId = elements.cameraId.value.trim();
  const idleTimeout = parseBoundedNumber(elements.idleTimeout.value, "空闲超时", 1, 86400);
  if (!cameraId) throw new Error("请输入摄像头编号。");
  if (!/^[\w.-]{1,64}$/u.test(cameraId)) throw new Error("摄像头编号只能包含字母、数字、下划线、点和连字符。");

  if (state.sourceMode === "file") {
    if (!(state.selectedFile instanceof File)) throw new Error("请先选择一个本地视频文件。");
    if (state.selectedFile.size <= 0) throw new Error("不能上传空文件。");
    const upload = await uploadFile(state.selectedFile);
    if (!upload || !upload.upload_id) throw new Error("上传服务未返回 upload_id。");
    return { source_type: "file", upload_id: String(upload.upload_id), camera_id: cameraId, idle_timeout_sec: idleTimeout };
  }

  const url = elements.rtspUrl.value.trim();
  if (!/^rtsps?:\/\//i.test(url)) throw new Error("RTSP 地址必须以 rtsp:// 或 rtsps:// 开头。");
  return {
    source_type: "rtsp",
    url,
    camera_id: cameraId,
    nominal_fps: parseBoundedNumber(elements.nominalFps.value, "标称帧率", 0.1, 240),
    idle_timeout_sec: idleTimeout,
    features: buildStableBehaviorFeatures(),
  };
}

function incrementCameraId() {
  const value = elements.cameraId.value.trim();
  const match = /^camera-(\d+)$/i.exec(value);
  if (!match) return;
  const width = Math.max(2, match[1].length);
  const next = Number(match[1]) + 1;
  elements.cameraId.value = `camera-${String(next).padStart(width, "0")}`;
}

async function uploadFile(file) {
  const query = new URLSearchParams({ filename: file.name || "video" });
  return requestJson(`${API.uploads}?${query.toString()}`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: file,
  });
}

async function startExistingTask(taskId) {
  const id = String(taskId || "");
  if (!id || state.startingTaskIds.has(id)) return;
  clearError();
  state.startingTaskIds.add(id);
  renderTasks();
  try {
    const created = await requestJson(`${API.tasks}/${encodeURIComponent(id)}/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    showToast(`已启动 ${created.camera_id || "历史视频源"}`);
    await refreshDashboard({ surfaceErrors: false, force: true });
  } catch (error) {
    const message = errorMessage(error);
    showError(message);
    showToast(message, "error");
  } finally {
    state.startingTaskIds.delete(id);
    renderTasks();
  }
}

async function stopTask(taskId) {
  const id = String(taskId || "");
  if (!id || state.stoppingTaskIds.has(id)) return;
  state.stoppingTaskIds.add(id);
  renderTasks();
  renderPreviewWall();
  try {
    await requestJson(`${API.tasks}/${encodeURIComponent(id)}/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    showToast("停止请求已发送");
    await refreshDashboard({ surfaceErrors: false, force: true });
  } catch (error) {
    const message = errorMessage(error);
    showError(message);
    showToast(message, "error");
  } finally {
    state.stoppingTaskIds.delete(id);
    renderTasks();
    renderPreviewWall();
  }
}

async function restartService() {
  if (!window.confirm("确定要重启 DeepStream 分析服务吗？运行中的任务可能会被中断。历史任务不会自动重新启动。")) return;
  elements.restartServiceButton.disabled = true;
  elements.servicePill.dataset.state = "warning";
  elements.serviceStatusText.textContent = "正在重启服务";
  try {
    await requestJson(API.restart, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    showToast("服务已重启；历史任务保持停止状态");
    window.setTimeout(() => void refreshDashboard({ surfaceErrors: false, force: true }), 1200);
  } catch (error) {
    showError(errorMessage(error));
  } finally {
    elements.restartServiceButton.disabled = false;
  }
}

async function refreshDashboard({ surfaceErrors = false, force = false } = {}) {
  if (state.refreshing && !force) return;
  state.refreshing = true;
  elements.refreshTasksButton.disabled = true;
  const [serviceResult, tasksResult] = await Promise.allSettled([
    requestJson(API.service, { cache: "no-store" }),
    requestJson(API.tasks, { cache: "no-store" }),
  ]);
  if (serviceResult.status === "fulfilled") renderService(serviceResult.value);
  else renderServiceOffline(serviceResult.reason);

  if (tasksResult.status === "fulfilled") {
    const rawTasks = Array.isArray(tasksResult.value) ? tasksResult.value : Array.isArray(tasksResult.value && tasksResult.value.tasks) ? tasksResult.value.tasks : [];
    state.tasks = rawTasks.map(normalizeTask).filter((task) => task.id);
    renderTasks();
    renderPreviewWall();
    updateOverview();
    elements.lastUpdatedText.textContent = `更新于 ${formatClock(new Date())}`;
  } else if (surfaceErrors) showError(errorMessage(tasksResult.reason));
  if (surfaceErrors && serviceResult.status === "rejected") showError(errorMessage(serviceResult.reason));
  state.refreshing = false;
  elements.refreshTasksButton.disabled = false;
}

function renderService(service) {
  const rawStatus = String(service && (service.status ?? service.state ?? "online")).toLowerCase();
  const healthy = service && typeof service.healthy === "boolean" ? service.healthy : !["offline", "failed", "error", "unhealthy", "stopped"].includes(rawStatus);
  const warning = ["starting", "restarting", "degraded", "busy"].includes(rawStatus);
  elements.servicePill.dataset.state = healthy ? (warning ? "warning" : "online") : "offline";
  elements.serviceStatusText.textContent = healthy ? (warning ? "服务准备中" : "服务在线") : "服务异常";
  const details = [];
  if (service && service.version) details.push(`v${service.version}`);
  if (service && service.detector) details.push(String(service.detector));
  if (service && service.tracker) details.push(String(service.tracker));
  if (service && Number.isFinite(Number(service.max_active_tasks))) details.push(`活动任务 ${Number(service.active_tasks) || 0}/${Number(service.max_active_tasks)}`);
  if (service && service.message) details.push(String(service.message));
  elements.serviceDetail.textContent = details.join(" · ") || statusLabel(rawStatus);
}

function renderServiceOffline(error) {
  elements.servicePill.dataset.state = "offline";
  elements.serviceStatusText.textContent = "服务不可达";
  elements.serviceDetail.textContent = shortMessage(errorMessage(error), 90);
}

function renderPreviewWall() {
  // Only a pipeline that has reached RUNNING gets an MJPEG connection. A stale
  // STARTING history row must never open a long-lived stream on page load.
  const activeTasks = state.tasks.filter((task) => isPreviewStatus(task.status)).slice(0, MAX_PREVIEW_TILES);
  const activeIds = new Set(activeTasks.map((task) => task.id));

  for (const [taskId, entry] of state.previewNodes.entries()) {
    if (!activeIds.has(taskId)) {
      disconnectPreview(entry);
      entry.cell.remove();
      state.previewNodes.delete(taskId);
    }
  }

  for (const task of activeTasks) {
    let entry = state.previewNodes.get(task.id);
    if (!entry) {
      entry = createPreviewEntry(task);
      state.previewNodes.set(task.id, entry);
    }
    updatePreviewEntry(entry, task);
  }

  rebuildPreviewRows(activeTasks.map((task) => state.previewNodes.get(task.id)).filter(Boolean));
  elements.previewEmpty.hidden = activeTasks.length > 0;
  elements.liveIndicator.dataset.active = activeTasks.length > 0 ? "true" : "false";
  elements.liveIndicator.querySelector("span").textContent = `${activeTasks.length} 路预览`;
}

function createPreviewEntry(task) {
  const cell = document.createElement("td");
  const heading = document.createElement("div");
  heading.className = "selected-task-summary";
  const camera = document.createElement("div");
  const cameraLabel = document.createElement("span");
  cameraLabel.textContent = "摄像头";
  const cameraValue = document.createElement("strong");
  camera.append(cameraLabel, cameraValue);
  const taskBox = document.createElement("div");
  const taskLabel = document.createElement("span");
  taskLabel.textContent = "任务";
  const taskValue = document.createElement("strong");
  taskBox.append(taskLabel, taskValue);
  const stop = document.createElement("button");
  stop.type = "button";
  stop.className = "button button-danger button-compact";
  stop.textContent = "■ 停止";
  stop.addEventListener("click", () => void stopTask(task.id));
  heading.append(camera, taskBox, stop);

  const stage = document.createElement("div");
  stage.className = "preview-stage";
  const image = document.createElement("img");
  image.alt = `${task.cameraId || task.id} 实时分析画面`;
  const placeholder = document.createElement("div");
  placeholder.className = "preview-placeholder";
  placeholder.innerHTML = "<strong>连接中</strong><span>正在等待 MJPEG 画面…</span>";
  const error = document.createElement("div");
  error.className = "preview-error";
  error.hidden = true;
  error.innerHTML = "<strong>预览暂时不可用</strong><span>将在任务状态刷新后重新连接</span>";
  stage.append(image, placeholder, error);
  cell.append(heading, stage);

  const entry = { taskId: task.id, cell, cameraValue, taskValue, stop, image, placeholder, error, url: null };
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

function updatePreviewEntry(entry, task) {
  entry.cameraValue.textContent = task.cameraId || "—";
  entry.taskValue.textContent = shortId(task.id);
  entry.taskValue.title = task.id;
  entry.stop.disabled = state.stoppingTaskIds.has(task.id) || !isStoppableStatus(task.status);
  const url = safePreviewUrl(task);
  if (entry.url !== url || !entry.image.getAttribute("src")) {
    entry.url = url;
    entry.error.hidden = true;
    entry.placeholder.hidden = false;
    entry.image.hidden = false;
    entry.image.src = withCacheBuster(url);
  }
}

function rebuildPreviewRows(entries) {
  elements.previewGridBody.replaceChildren();
  for (let index = 0; index < entries.length; index += 2) {
    const row = document.createElement("tr");
    row.append(entries[index].cell);
    if (entries[index + 1]) row.append(entries[index + 1].cell);
    else row.append(document.createElement("td"));
    elements.previewGridBody.append(row);
  }
}

function disconnectPreview(entry) {
  if (!entry) return;
  entry.image.removeAttribute("src");
  entry.url = null;
}

function renderTasks() {
  elements.tasksTableBody.replaceChildren();
  elements.tasksEmpty.hidden = state.tasks.length > 0;
  for (const task of state.tasks) {
    const row = document.createElement("tr");
    row.append(taskCell(task), sourceCell(task), textCell(task.cameraId || "—", "camera-code"), statusCell(task.status), textCell(formatDateTime(task.updatedAt), "task-time"), actionCell(task));
    elements.tasksTableBody.append(row);
  }
}

function taskCell(task) {
  const cell = document.createElement("td");
  const wrap = document.createElement("div");
  wrap.className = "task-primary";
  const title = document.createElement("strong");
  title.textContent = shortId(task.id);
  title.title = task.id;
  const meta = document.createElement("span");
  meta.textContent = task.createdAt ? `创建于 ${formatDateTime(task.createdAt)}` : "分析任务";
  wrap.append(title, meta);
  cell.append(wrap);
  return cell;
}

function sourceCell(task) {
  const cell = document.createElement("td");
  const wrap = document.createElement("div");
  wrap.className = "task-source";
  const title = document.createElement("strong");
  title.textContent = task.sourceType === "rtsp" ? "RTSP 视频流" : "本地文件";
  const detail = document.createElement("span");
  detail.textContent = shortMessage(task.sourceLabel || "—", 54);
  detail.title = task.sourceLabel || "";
  wrap.append(title, detail);
  cell.append(wrap);
  return cell;
}

function textCell(value, className) {
  const cell = document.createElement("td");
  const span = document.createElement("span");
  if (className) span.className = className;
  span.textContent = String(value);
  cell.append(span);
  return cell;
}

function statusCell(status) {
  const cell = document.createElement("td");
  const badge = document.createElement("span");
  const canonical = canonicalStatus(status);
  badge.className = `status-badge status-${canonical}`;
  badge.textContent = statusLabel(canonical);
  cell.append(badge);
  return cell;
}

function actionCell(task) {
  const cell = document.createElement("td");
  if (isStoppableStatus(task.status)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button button-danger button-compact row-action";
    button.disabled = state.stoppingTaskIds.has(task.id) || canonicalStatus(task.status) === "stopping";
    button.textContent = button.disabled ? "停止中" : "停止";
    button.addEventListener("click", () => void stopTask(task.id));
    cell.append(button);
    return cell;
  }
  if (isStartableStatus(task.status) && task.restartable !== false) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button button-primary button-compact row-action";
    button.disabled = state.startingTaskIds.has(task.id);
    button.textContent = button.disabled ? "启动中" : "启动";
    button.addEventListener("click", () => void startExistingTask(task.id));
    cell.append(button);
    return cell;
  }
  cell.textContent = "—";
  return cell;
}

function updateOverview() {
  const running = state.tasks.filter((task) => canonicalStatus(task.status) === "running").length;
  const pending = state.tasks.filter((task) => ["pending", "starting", "stopping"].includes(canonicalStatus(task.status))).length;
  elements.runningCount.textContent = String(running);
  elements.pendingCount.textContent = String(pending);
  elements.totalCount.textContent = String(state.tasks.length);
}

function normalizeTask(raw) {
  const source = raw && typeof raw.source === "object" ? raw.source : {};
  const id = getTaskId(raw);
  const sourceType = String(raw?.source_type ?? source.type ?? "file").toLowerCase();
  const sourceLabel = sourceType === "rtsp"
    ? String(raw?.source_label ?? raw?.url ?? source.url ?? "RTSP")
    : String(raw?.source_label ?? raw?.filename ?? raw?.upload?.filename ?? source.filename ?? raw?.upload_id ?? "本地视频");
  return {
    raw,
    id,
    status: canonicalStatus(raw?.status ?? raw?.state ?? "pending"),
    sourceType: sourceType === "rtsp" ? "rtsp" : "file",
    sourceLabel,
    cameraId: String(raw?.camera_id ?? raw?.camera ?? source.camera_id ?? ""),
    createdAt: raw?.created_at ?? raw?.createdAt ?? null,
    updatedAt: raw?.updated_at ?? raw?.updatedAt ?? raw?.finished_at ?? raw?.started_at ?? null,
    previewUrl: raw?.preview_url ?? raw?.stream_url ?? null,
    historical: Boolean(raw?.historical),
    restartable: raw?.restartable !== false,
  };
}

function getTaskId(task) {
  const value = task && (task.id ?? task.task_id);
  return value === undefined || value === null ? "" : String(value);
}

function canonicalStatus(value) {
  const status = String(value || "unknown").trim().toLowerCase();
  const aliases = { created: "pending", queued: "pending", waiting: "pending", initializing: "starting", active: "running", processing: "running", cancelling: "stopping", canceled: "stopped", cancelled: "stopped", complete: "succeeded", completed: "succeeded", success: "succeeded", finished: "succeeded", error: "failed" };
  return aliases[status] || status;
}

function statusLabel(value) {
  const labels = { online: "在线", ready: "就绪", healthy: "正常", busy: "繁忙", degraded: "降级", restarting: "重启中", pending: "等待手动启动", starting: "启动中", running: "运行中", stopping: "停止中", stopped: "已停止", succeeded: "已完成", failed: "失败", offline: "离线", unknown: "未知" };
  const canonical = canonicalStatus(value);
  return labels[canonical] || String(value || "未知");
}

function isPreviewStatus(value) { return canonicalStatus(value) === "running"; }
function isStoppableStatus(value) { return ["starting", "running", "stopping"].includes(canonicalStatus(value)); }
function isStartableStatus(value) { return ["pending", "stopped", "succeeded", "failed"].includes(canonicalStatus(value)); }

function safePreviewUrl(task) {
  const fallback = `${API.tasks}/${encodeURIComponent(task.id)}/stream.mjpg`;
  if (!task.previewUrl) return fallback;
  try {
    const candidate = new URL(String(task.previewUrl), window.location.origin);
    if (candidate.origin !== window.location.origin || !candidate.pathname.startsWith("/api/")) return fallback;
    return `${candidate.pathname}${candidate.search}`;
  } catch {
    return fallback;
  }
}

function withCacheBuster(url) {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}_=${Date.now()}`;
}

async function requestJson(url, options = {}) {
  let response;
  try {
    response = await fetch(url, {
      credentials: "same-origin",
      ...options,
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
  } catch (error) {
    throw new Error(`无法连接分析服务：${errorMessage(error)}`);
  }
  const text = await response.text();
  let payload = null;
  if (text) {
    try { payload = JSON.parse(text); } catch { payload = text; }
  }
  if (!response.ok) {
    const detail = typeof payload === "object" && payload ? payload.detail ?? payload.error ?? payload.message : payload;
    throw new Error(detail ? String(detail) : `服务返回 HTTP ${response.status}`);
  }
  return payload ?? {};
}

function setSubmitting(active, message) {
  state.submitting = active;
  elements.startTaskButton.disabled = active;
  elements.fileTab.disabled = active;
  elements.rtspTab.disabled = active;
  elements.startTaskButtonText.textContent = active ? (message || "处理中…") : "开始分析";
}

function setSubmissionStatus(message, tone = "") {
  elements.submissionStatus.textContent = message;
  if (tone) elements.submissionStatus.dataset.tone = tone;
  else delete elements.submissionStatus.dataset.tone;
}
function clearSubmissionStatus() { setSubmissionStatus(""); }
function showError(message) { elements.errorMessage.textContent = message; elements.errorBanner.hidden = false; }
function clearError() { elements.errorBanner.hidden = true; elements.errorMessage.textContent = ""; }

function showToast(message, type = "success") {
  const toast = document.createElement("div");
  toast.className = `toast${type === "error" ? " toast-error" : ""}`;
  toast.textContent = message;
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 3600);
}

function schedulePolling() {
  stopPolling();
  if (!document.hidden) state.pollingTimer = window.setInterval(() => void refreshDashboard({ surfaceErrors: false }), POLL_INTERVAL_MS);
}
function stopPolling() { if (state.pollingTimer !== null) { window.clearInterval(state.pollingTimer); state.pollingTimer = null; } }

function parseBoundedNumber(value, label, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < minimum || parsed > maximum) throw new Error(`${label}必须在 ${minimum} 到 ${maximum} 之间。`);
  return parsed;
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "大小未知";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = bytes / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) { size /= 1024; index += 1; }
  return `${size >= 100 ? size.toFixed(0) : size.toFixed(1)} ${units[index]}`;
}

function formatClock(value) {
  return new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(value);
}

function formatDateTime(value) {
  if (!value) return "—";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(date);
}

function shortId(value) {
  const text = String(value || "");
  return text.length > 18 ? `${text.slice(0, 8)}…${text.slice(-6)}` : text;
}
function shortMessage(value, maximum) {
  const text = String(value || "");
  return text.length > maximum ? `${text.slice(0, Math.max(1, maximum - 1))}…` : text;
}
function errorMessage(error) { return error instanceof Error && error.message ? error.message : String(error || "发生未知错误"); }
