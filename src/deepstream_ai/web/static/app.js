"use strict";

const API = Object.freeze({
  service: "/api/service",
  tasks: "/api/tasks",
  uploads: "/api/uploads",
  restart: "/api/service/restart",
});

const POLL_INTERVAL_MS = 3000;
const PREVIEW_RETRY_MS = 2500;

const state = {
  sourceMode: "file",
  selectedFile: null,
  tasks: [],
  selectedTaskId: null,
  submitting: false,
  refreshing: false,
  stoppingTaskIds: new Set(),
  pollingTimer: null,
  previewRetryTimer: null,
  previewUrl: null,
};

const elements = {};

document.addEventListener("DOMContentLoaded", initialize);

function initialize() {
  collectElements();
  bindEvents();
  updateClock();
  window.setInterval(updateClock, 1000);
  void refreshDashboard({ surfaceErrors: true });
  schedulePolling();
}

function collectElements() {
  const ids = [
    "servicePill",
    "serviceStatusText",
    "serviceDetail",
    "restartServiceButton",
    "errorBanner",
    "errorMessage",
    "dismissErrorButton",
    "runningCount",
    "pendingCount",
    "totalCount",
    "fileTab",
    "rtspTab",
    "fileSourcePanel",
    "rtspSourcePanel",
    "taskForm",
    "dropZone",
    "videoFile",
    "fileSelection",
    "selectedFileName",
    "selectedFileMeta",
    "clearFileButton",
    "rtspUrl",
    "nominalFps",
    "cameraId",
    "idleTimeout",
    "startTaskButton",
    "startTaskButtonText",
    "submissionStatus",
    "liveIndicator",
    "previewStage",
    "previewImage",
    "previewPlaceholder",
    "previewError",
    "previewClock",
    "selectedTaskId",
    "selectedCamera",
    "selectedSource",
    "stopSelectedButton",
    "lastUpdatedText",
    "refreshTasksButton",
    "tasksTableBody",
    "tasksEmpty",
    "toastRegion",
  ];

  for (const id of ids) {
    const node = document.getElementById(id);
    if (!node) {
      throw new Error(`页面缺少必要元素: #${id}`);
    }
    elements[id] = node;
  }
}

function bindEvents() {
  elements.fileTab.addEventListener("click", () => setSourceMode("file"));
  elements.rtspTab.addEventListener("click", () => setSourceMode("rtsp"));
  elements.videoFile.addEventListener("change", () => {
    setSelectedFile(elements.videoFile.files && elements.videoFile.files[0]);
  });
  elements.clearFileButton.addEventListener("click", clearSelectedFile);
  elements.taskForm.addEventListener("submit", handleTaskSubmit);
  elements.dismissErrorButton.addEventListener("click", clearError);
  elements.refreshTasksButton.addEventListener("click", () => {
    void refreshDashboard({ surfaceErrors: true, force: true });
  });
  elements.stopSelectedButton.addEventListener("click", () => {
    if (state.selectedTaskId) {
      void stopTask(state.selectedTaskId);
    }
  });
  elements.restartServiceButton.addEventListener("click", () => {
    void restartService();
  });
  elements.previewImage.addEventListener("load", handlePreviewLoad);
  elements.previewImage.addEventListener("error", handlePreviewError);

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
    if (file) {
      setSelectedFile(file);
    }
  });

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopPolling();
    } else {
      void refreshDashboard({ surfaceErrors: false, force: true });
      schedulePolling();
    }
  });
  window.addEventListener("pagehide", () => {
    stopPolling();
    clearPreviewRetry();
  });
}

function setSourceMode(mode) {
  if (!['file', 'rtsp'].includes(mode)) {
    return;
  }
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
  if (!(file instanceof File)) {
    clearSelectedFile();
    return;
  }
  state.selectedFile = file;
  elements.dropZone.hidden = true;
  elements.fileSelection.hidden = false;
  elements.selectedFileName.textContent = file.name || "未命名视频";
  const type = file.type || "类型未知";
  elements.selectedFileMeta.textContent = `${formatBytes(file.size)} · ${type}`;
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
  if (state.submitting) {
    return;
  }

  clearError();
  clearSubmissionStatus();

  setSubmitting(
    true,
    state.sourceMode === "file" && state.selectedFile
      ? `正在上传 ${shortMessage(state.selectedFile.name, 24)}…`
      : "正在创建任务…",
  );
  try {
    const payload = await buildTaskPayload();
    setSubmitting(true, "正在创建任务…");
    const task = await requestJson(API.tasks, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const normalized = normalizeTask(task);
    state.selectedTaskId = normalized.id || getTaskId(task) || state.selectedTaskId;
    setSubmissionStatus("任务已创建，正在等待分析服务启动。", "success");
    showToast("任务创建成功");
    if (state.sourceMode === "file") {
      clearSelectedFile();
    }
    await refreshDashboard({ surfaceErrors: false, force: true });
  } catch (error) {
    const message = errorMessage(error);
    showError(message);
    setSubmissionStatus(message, "error");
  } finally {
    setSubmitting(false);
  }
}

async function buildTaskPayload() {
  const cameraId = elements.cameraId.value.trim();
  const idleTimeout = parseBoundedNumber(elements.idleTimeout.value, "空闲超时", 1, 86400);
  if (!cameraId) {
    throw new Error("请输入摄像头编号。");
  }
  if (!/^[\w.-]{1,64}$/u.test(cameraId)) {
    throw new Error("摄像头编号只能包含字母、数字、下划线、点和连字符。");
  }

  if (state.sourceMode === "file") {
    if (!(state.selectedFile instanceof File)) {
      throw new Error("请先选择一个本地视频文件。");
    }
    if (state.selectedFile.size <= 0) {
      throw new Error("不能上传空文件。");
    }
    setSubmitting(true, `正在上传 ${state.selectedFile.name}…`);
    const upload = await uploadFile(state.selectedFile);
    if (!upload || !upload.upload_id) {
      throw new Error("上传服务未返回 upload_id。");
    }
    const details = [upload.codec, upload.fps ? `${formatNumber(upload.fps)} FPS` : null]
      .filter(Boolean)
      .join(" · ");
    setSubmissionStatus(details ? `上传完成：${details}` : "视频上传完成。", "success");
    return {
      source_type: "file",
      upload_id: String(upload.upload_id),
      camera_id: cameraId,
      idle_timeout_sec: idleTimeout,
    };
  }

  const url = elements.rtspUrl.value.trim();
  if (!/^rtsps?:\/\//i.test(url)) {
    throw new Error("RTSP 地址必须以 rtsp:// 或 rtsps:// 开头。");
  }
  const nominalFps = parseBoundedNumber(elements.nominalFps.value, "标称帧率", 0.1, 240);
  return {
    source_type: "rtsp",
    url,
    camera_id: cameraId,
    nominal_fps: nominalFps,
    idle_timeout_sec: idleTimeout,
  };
}

async function uploadFile(file) {
  const query = new URLSearchParams({ filename: file.name || "video" });
  return requestJson(`${API.uploads}?${query.toString()}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
    },
    body: file,
  });
}

async function stopTask(taskId) {
  const id = String(taskId || "");
  if (!id || state.stoppingTaskIds.has(id)) {
    return;
  }
  clearError();
  state.stoppingTaskIds.add(id);
  renderTasks();
  updateSelectedTask();
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
    updateSelectedTask();
  }
}

async function restartService() {
  if (!window.confirm("确定要重启 DeepStream 分析服务吗？运行中的任务可能会被中断。")) {
    return;
  }
  clearError();
  elements.restartServiceButton.disabled = true;
  elements.servicePill.dataset.state = "warning";
  elements.serviceStatusText.textContent = "正在重启服务";
  try {
    await requestJson(API.restart, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    showToast("服务重启请求已发送");
    window.setTimeout(() => {
      void refreshDashboard({ surfaceErrors: false, force: true });
    }, 1200);
  } catch (error) {
    const message = errorMessage(error);
    showError(message);
    showToast(message, "error");
  } finally {
    elements.restartServiceButton.disabled = false;
  }
}

async function refreshDashboard({ surfaceErrors = false, force = false } = {}) {
  if (state.refreshing && !force) {
    return;
  }
  state.refreshing = true;
  elements.refreshTasksButton.disabled = true;
  elements.refreshTasksButton.setAttribute("aria-busy", "true");

  const [serviceResult, tasksResult] = await Promise.allSettled([
    requestJson(API.service, { cache: "no-store" }),
    requestJson(API.tasks, { cache: "no-store" }),
  ]);

  if (serviceResult.status === "fulfilled") {
    renderService(serviceResult.value);
  } else {
    renderServiceOffline(serviceResult.reason);
  }

  if (tasksResult.status === "fulfilled") {
    const rawTasks = Array.isArray(tasksResult.value)
      ? tasksResult.value
      : Array.isArray(tasksResult.value && tasksResult.value.tasks)
        ? tasksResult.value.tasks
        : [];
    state.tasks = rawTasks.map(normalizeTask).filter((task) => task.id);
    reconcileSelection();
    renderTasks();
    updateOverview();
    updateSelectedTask();
    elements.lastUpdatedText.textContent = `更新于 ${formatClock(new Date())}`;
  } else if (surfaceErrors) {
    showError(errorMessage(tasksResult.reason));
  }

  if (surfaceErrors && serviceResult.status === "rejected") {
    showError(errorMessage(serviceResult.reason));
  }

  state.refreshing = false;
  elements.refreshTasksButton.disabled = false;
  elements.refreshTasksButton.removeAttribute("aria-busy");
}

function renderService(service) {
  const rawStatus = String(service && (service.status ?? service.state ?? "online")).toLowerCase();
  const healthy = service && typeof service.healthy === "boolean"
    ? service.healthy
    : !["offline", "failed", "error", "unhealthy", "stopped"].includes(rawStatus);
  const warning = ["starting", "restarting", "degraded", "busy"].includes(rawStatus);
  elements.servicePill.dataset.state = healthy ? (warning ? "warning" : "online") : "offline";
  elements.serviceStatusText.textContent = healthy
    ? (warning ? "服务准备中" : "服务在线")
    : "服务异常";

  const details = [];
  if (service && service.version) details.push(`v${service.version}`);
  if (service && service.detector) details.push(String(service.detector));
  if (service && service.tracker) details.push(String(service.tracker));
  if (service && Number.isFinite(Number(service.max_active_tasks))) {
    details.push(`活动任务 ${Number(service.active_tasks) || 0}/${Number(service.max_active_tasks)}`);
  }
  if (service && service.message) details.push(String(service.message));
  elements.serviceDetail.textContent = details.join(" · ") || statusLabel(rawStatus);
}

function renderServiceOffline(error) {
  elements.servicePill.dataset.state = "offline";
  elements.serviceStatusText.textContent = "服务不可达";
  elements.serviceDetail.textContent = shortMessage(errorMessage(error), 90);
}

function reconcileSelection() {
  if (state.selectedTaskId && state.tasks.some((task) => task.id === state.selectedTaskId)) {
    return;
  }
  const active = state.tasks.find((task) => isActiveStatus(task.status));
  state.selectedTaskId = active ? active.id : (state.tasks[0] ? state.tasks[0].id : null);
}

function renderTasks() {
  elements.tasksTableBody.replaceChildren();
  elements.tasksEmpty.hidden = state.tasks.length > 0;

  for (const task of state.tasks) {
    const row = document.createElement("tr");
    row.className = "task-row";
    row.tabIndex = 0;
    row.classList.toggle("is-selected", task.id === state.selectedTaskId);
    row.setAttribute("aria-selected", String(task.id === state.selectedTaskId));
    row.addEventListener("click", () => selectTask(task.id));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectTask(task.id);
      }
    });

    row.append(
      taskCell(task),
      sourceCell(task),
      textCell(task.cameraId || "—", "camera-code"),
      statusCell(task.status),
      textCell(formatDateTime(task.updatedAt), "task-time"),
      actionCell(task),
    );
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
    const stopping = state.stoppingTaskIds.has(task.id);
    button.disabled = stopping;
    button.textContent = stopping ? "停止中" : "停止";
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      void stopTask(task.id);
    });
    cell.append(button);
  } else {
    const view = document.createElement("button");
    view.type = "button";
    view.className = "button button-ghost button-compact row-action";
    view.textContent = "查看";
    view.addEventListener("click", (event) => {
      event.stopPropagation();
      selectTask(task.id);
    });
    cell.append(view);
  }
  return cell;
}

function selectTask(taskId) {
  state.selectedTaskId = String(taskId);
  renderTasks();
  updateSelectedTask();
}

function updateSelectedTask() {
  const task = state.tasks.find((item) => item.id === state.selectedTaskId) || null;
  if (!task) {
    elements.selectedTaskId.textContent = "未选择";
    elements.selectedTaskId.title = "";
    elements.selectedCamera.textContent = "—";
    elements.selectedSource.textContent = "—";
    elements.stopSelectedButton.disabled = true;
    showPreviewPlaceholder("等待选择运行中的任务", "实时 MJPEG 画面将在这里显示");
    return;
  }

  elements.selectedTaskId.textContent = shortId(task.id);
  elements.selectedTaskId.title = task.id;
  elements.selectedCamera.textContent = task.cameraId || "—";
  elements.selectedSource.textContent = task.sourceType === "rtsp" ? "RTSP" : "本地文件";
  elements.stopSelectedButton.disabled =
    !isStoppableStatus(task.status) || state.stoppingTaskIds.has(task.id);

  if (isActiveStatus(task.status)) {
    connectPreview(task);
  } else {
    showPreviewPlaceholder(
      `任务${statusLabel(task.status)}`,
      isTerminalStatus(task.status) ? "请选择其他运行中的任务查看实时画面" : "视频流将在任务启动后自动连接",
    );
  }
}

function connectPreview(task) {
  const url = safePreviewUrl(task);
  if (state.previewUrl === url && !elements.previewImage.hidden) {
    return;
  }
  clearPreviewRetry();
  state.previewUrl = url;
  elements.previewPlaceholder.hidden = true;
  elements.previewError.hidden = true;
  elements.previewImage.hidden = false;
  elements.liveIndicator.dataset.active = "true";
  elements.liveIndicator.querySelector("span").textContent = "连接中";
  elements.previewImage.src = withCacheBuster(url);
}

function handlePreviewLoad() {
  elements.previewError.hidden = true;
  elements.previewPlaceholder.hidden = true;
  elements.previewImage.hidden = false;
  elements.liveIndicator.dataset.active = "true";
  elements.liveIndicator.querySelector("span").textContent = "实时";
}

function handlePreviewError() {
  const task = state.tasks.find((item) => item.id === state.selectedTaskId);
  if (!task || !isActiveStatus(task.status)) {
    return;
  }
  elements.previewImage.hidden = true;
  elements.previewError.hidden = false;
  elements.liveIndicator.dataset.active = "false";
  elements.liveIndicator.querySelector("span").textContent = "重连中";
  clearPreviewRetry();
  state.previewRetryTimer = window.setTimeout(() => {
    state.previewRetryTimer = null;
    const current = state.tasks.find((item) => item.id === state.selectedTaskId);
    if (current && isActiveStatus(current.status)) {
      state.previewUrl = null;
      connectPreview(current);
    }
  }, PREVIEW_RETRY_MS);
}

function showPreviewPlaceholder(title, detail) {
  clearPreviewRetry();
  state.previewUrl = null;
  elements.previewImage.removeAttribute("src");
  elements.previewImage.hidden = true;
  elements.previewError.hidden = true;
  elements.previewPlaceholder.hidden = false;
  const strong = elements.previewPlaceholder.querySelector("strong");
  const span = elements.previewPlaceholder.querySelector(":scope > span:last-child");
  if (strong) strong.textContent = title;
  if (span) span.textContent = detail;
  elements.liveIndicator.dataset.active = "false";
  elements.liveIndicator.querySelector("span").textContent = "未连接";
}

function clearPreviewRetry() {
  if (state.previewRetryTimer !== null) {
    window.clearTimeout(state.previewRetryTimer);
    state.previewRetryTimer = null;
  }
}

function updateOverview() {
  const running = state.tasks.filter((task) => canonicalStatus(task.status) === "running").length;
  const pending = state.tasks.filter((task) =>
    ["pending", "starting", "stopping"].includes(canonicalStatus(task.status)),
  ).length;
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
    : String(
      raw?.source_label
      ?? raw?.filename
      ?? raw?.upload?.filename
      ?? source.filename
      ?? raw?.upload_id
      ?? "本地视频",
    );
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
  };
}

function getTaskId(task) {
  const value = task && (task.id ?? task.task_id);
  return value === undefined || value === null ? "" : String(value);
}

function canonicalStatus(value) {
  const status = String(value || "unknown").trim().toLowerCase();
  const aliases = {
    created: "pending",
    queued: "pending",
    waiting: "pending",
    initializing: "starting",
    active: "running",
    processing: "running",
    cancelling: "stopping",
    canceled: "stopped",
    cancelled: "stopped",
    complete: "succeeded",
    completed: "succeeded",
    success: "succeeded",
    finished: "succeeded",
    error: "failed",
  };
  return aliases[status] || status;
}

function statusLabel(value) {
  const labels = {
    online: "在线",
    ready: "就绪",
    healthy: "正常",
    busy: "繁忙",
    degraded: "降级",
    restarting: "重启中",
    pending: "等待中",
    starting: "启动中",
    running: "运行中",
    stopping: "停止中",
    stopped: "已停止",
    succeeded: "已完成",
    failed: "失败",
    offline: "离线",
    unknown: "未知",
  };
  const canonical = canonicalStatus(value);
  return labels[canonical] || String(value || "未知");
}

function isActiveStatus(value) {
  return ["starting", "running"].includes(canonicalStatus(value));
}

function isStoppableStatus(value) {
  return ["pending", "starting", "running"].includes(canonicalStatus(value));
}

function isTerminalStatus(value) {
  return ["stopped", "succeeded", "failed"].includes(canonicalStatus(value));
}

function safePreviewUrl(task) {
  const fallback = `${API.tasks}/${encodeURIComponent(task.id)}/stream.mjpg`;
  if (!task.previewUrl) {
    return fallback;
  }
  try {
    const candidate = new URL(String(task.previewUrl), window.location.origin);
    if (candidate.origin !== window.location.origin || !candidate.pathname.startsWith("/api/")) {
      return fallback;
    }
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
      headers: {
        Accept: "application/json",
        ...(options.headers || {}),
      },
    });
  } catch (error) {
    throw new Error(`无法连接分析服务：${errorMessage(error)}`);
  }

  const text = await response.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }
  if (!response.ok) {
    const detail = typeof payload === "object" && payload
      ? payload.detail ?? payload.error ?? payload.message
      : payload;
    throw new Error(detail ? String(detail) : `服务返回 HTTP ${response.status}`);
  }
  return payload ?? {};
}

function setSubmitting(active, message) {
  state.submitting = active;
  elements.startTaskButton.disabled = active;
  elements.fileTab.disabled = active;
  elements.rtspTab.disabled = active;
  elements.startTaskButton.setAttribute("aria-busy", String(active));
  elements.startTaskButtonText.textContent = active ? (message || "处理中…") : "开始分析";
}

function setSubmissionStatus(message, tone = "") {
  elements.submissionStatus.textContent = message;
  if (tone) {
    elements.submissionStatus.dataset.tone = tone;
  } else {
    delete elements.submissionStatus.dataset.tone;
  }
}

function clearSubmissionStatus() {
  setSubmissionStatus("");
}

function showError(message) {
  elements.errorMessage.textContent = message;
  elements.errorBanner.hidden = false;
}

function clearError() {
  elements.errorBanner.hidden = true;
  elements.errorMessage.textContent = "";
}

function showToast(message, type = "success") {
  const toast = document.createElement("div");
  toast.className = `toast${type === "error" ? " toast-error" : ""}`;
  toast.textContent = message;
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), 3600);
}

function schedulePolling() {
  stopPolling();
  if (!document.hidden) {
    state.pollingTimer = window.setInterval(() => {
      void refreshDashboard({ surfaceErrors: false });
    }, POLL_INTERVAL_MS);
  }
}

function stopPolling() {
  if (state.pollingTimer !== null) {
    window.clearInterval(state.pollingTimer);
    state.pollingTimer = null;
  }
}

function updateClock() {
  if (elements.previewClock) {
    elements.previewClock.textContent = formatClock(new Date());
  }
}

function parseBoundedNumber(value, label, minimum, maximum) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < minimum || parsed > maximum) {
    throw new Error(`${label}必须在 ${minimum} 到 ${maximum} 之间。`);
  }
  return parsed;
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "大小未知";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let size = bytes / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size >= 100 ? size.toFixed(0) : size.toFixed(1)} ${units[index]}`;
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(number % 1 === 0 ? 0 : 2) : "—";
}

function formatClock(value) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(value);
}

function formatDateTime(value) {
  if (!value) return "—";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
}

function shortId(value) {
  const text = String(value || "");
  return text.length > 18 ? `${text.slice(0, 8)}…${text.slice(-6)}` : text;
}

function shortMessage(value, maximum) {
  const text = String(value || "");
  return text.length > maximum ? `${text.slice(0, Math.max(1, maximum - 1))}…` : text;
}

function errorMessage(error) {
  if (error instanceof Error && error.message) return error.message;
  return String(error || "发生未知错误");
}
