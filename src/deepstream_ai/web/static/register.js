(() => {
  const form = document.getElementById('faceRegisterForm');
  const workerId = document.getElementById('workerId');
  const mode = document.getElementById('registerMode');
  const fileInput = document.getElementById('faceFile');
  const dropZone = document.getElementById('faceDropZone');
  const previewCard = document.getElementById('facePreviewCard');
  const preview = document.getElementById('facePreview');
  const previewName = document.getElementById('facePreviewName');
  const previewMeta = document.getElementById('facePreviewMeta');
  const clearButton = document.getElementById('clearFaceFile');
  const registerButton = document.getElementById('registerButton');
  const registerButtonText = document.getElementById('registerButtonText');
  const registerStatus = document.getElementById('registerStatus');
  const workerHint = document.getElementById('workerHint');
  const modeHint = document.getElementById('modeHint');
  const resultPanel = document.getElementById('registrationResultPanel');
  const qualityPill = document.getElementById('qualityPill');
  const qualityPillText = document.getElementById('qualityPillText');
  const qualityMetrics = document.getElementById('qualityMetrics');
  const issuesBlock = document.getElementById('qualityIssuesBlock');
  const warningsBlock = document.getElementById('qualityWarningsBlock');
  const issuesList = document.getElementById('qualityIssues');
  const warningsList = document.getElementById('qualityWarnings');
  const success = document.getElementById('qualitySuccess');
  const errorBanner = document.getElementById('registerError');
  const errorText = document.getElementById('registerErrorText');
  const dismissError = document.getElementById('dismissRegisterError');

  let selectedFile = null;
  let previewUrl = null;
  let workerLookupTimer = null;

  const formatBytes = (value) => {
    if (!Number.isFinite(value)) return '';
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / 1024 / 1024).toFixed(2)} MB`;
  };

  const showError = (message) => {
    errorText.textContent = message || '操作失败';
    errorBanner.hidden = false;
  };

  const clearError = () => {
    errorBanner.hidden = true;
    errorText.textContent = '';
  };

  const resetResult = () => {
    resultPanel.hidden = true;
    qualityMetrics.replaceChildren();
    issuesList.replaceChildren();
    warningsList.replaceChildren();
    issuesBlock.hidden = true;
    warningsBlock.hidden = true;
    success.hidden = true;
    success.textContent = '';
  };

  const setFile = (file) => {
    if (!file) {
      selectedFile = null;
      fileInput.value = '';
      previewCard.hidden = true;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = null;
      preview.removeAttribute('src');
      resetResult();
      return;
    }
    const allowed = ['image/jpeg', 'image/png', 'image/webp'];
    if (!allowed.includes(file.type)) {
      showError('仅支持 JPEG、PNG 或 WebP 图片。');
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      showError('图片不能超过 12MB。');
      return;
    }
    clearError();
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    previewName.textContent = file.name;
    previewMeta.textContent = `${formatBytes(file.size)} · ${file.type.replace('image/', '').toUpperCase()}`;
    previewCard.hidden = false;
    resetResult();
  };

  const renderMetrics = (metrics = {}) => {
    const items = [
      ['图片', metrics.image_width && metrics.image_height ? `${metrics.image_width}×${metrics.image_height}` : '—'],
      ['人脸', metrics.face_width && metrics.face_height ? `${metrics.face_width}×${metrics.face_height}` : '—'],
      ['检测', metrics.detector != null ? Number(metrics.detector).toFixed(3) : '—'],
      ['清晰度', metrics.blur != null ? Number(metrics.blur).toFixed(3) : '—'],
      ['正脸', metrics.frontal != null ? Number(metrics.frontal).toFixed(3) : '—'],
      ['综合质量', metrics.quality != null ? Number(metrics.quality).toFixed(3) : '—'],
      ['亮度', metrics.brightness != null ? `${Number(metrics.brightness).toFixed(0)}/255` : '—'],
      ['姿态', metrics.pose || '—'],
    ];
    qualityMetrics.replaceChildren(...items.map(([label, value]) => {
      const article = document.createElement('article');
      const span = document.createElement('span');
      const strong = document.createElement('strong');
      span.textContent = label;
      strong.textContent = value;
      article.append(span, strong);
      return article;
    }));
  };

  const renderList = (block, list, items) => {
    list.replaceChildren();
    const values = Array.isArray(items) ? items : [];
    block.hidden = values.length === 0;
    for (const item of values) {
      const li = document.createElement('li');
      li.textContent = item;
      list.appendChild(li);
    }
  };

  const renderResult = (data, httpOk) => {
    resultPanel.hidden = false;
    renderMetrics(data.metrics || {});
    renderList(issuesBlock, issuesList, data.issues);
    renderList(warningsBlock, warningsList, data.warnings);
    const accepted = Boolean(data.accepted);
    const stored = Boolean(data.stored);
    qualityPill.dataset.state = accepted ? 'ready' : 'error';
    qualityPillText.textContent = accepted ? (stored ? '已注册' : '质量通过') : '不合格';
    if (accepted) {
      success.hidden = false;
      if (stored) {
        const count = data.template_count ?? '—';
        success.textContent = `Work ID ${data.worker_id} 注册成功。当前共 ${count} 个有效人脸模板。`;
      } else {
        success.textContent = `照片质量通过，但未新增模板：${(data.warnings || []).slice(-1)[0] || '与已有模板重复。'}`;
      }
    } else {
      success.hidden = true;
    }
    if (!httpOk && !data.issues?.length && data.error) showError(data.error);
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const lookupWorker = async () => {
    const value = workerId.value.trim();
    if (!value) {
      workerHint.textContent = '输入后会显示该人员已有模板数量。';
      return;
    }
    try {
      const response = await fetch(`/api/faces/worker?worker_id=${encodeURIComponent(value)}`, { cache: 'no-store' });
      if (!response.ok) return;
      const data = await response.json();
      workerHint.textContent = data.template_count > 0
        ? `该人员已有 ${data.template_count} 个模板。可选择“补充角度”继续增强侧脸覆盖。`
        : '该人员还没有模板，请先使用清晰正脸进行首次注册。';
    } catch (_) {
      // Worker lookup is advisory; registration itself will return the real error.
    }
  };

  workerId.addEventListener('input', () => {
    if (workerLookupTimer) clearTimeout(workerLookupTimer);
    workerLookupTimer = setTimeout(lookupWorker, 350);
  });

  mode.addEventListener('change', () => {
    modeHint.textContent = mode.value === 'primary'
      ? '首次注册请使用无遮挡正脸。主照片质量门槛最高，用作身份基准。'
      : '补充角度不会覆盖已有正脸，而是增加一个独立模板；建议左右各 15°~35°，逐级补充。';
    resetResult();
  });

  fileInput.addEventListener('change', () => setFile(fileInput.files?.[0] || null));
  clearButton.addEventListener('click', () => setFile(null));
  dismissError.addEventListener('click', clearError);

  for (const eventName of ['dragenter', 'dragover']) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.add('is-dragging');
    });
  }
  for (const eventName of ['dragleave', 'drop']) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropZone.classList.remove('is-dragging');
    });
  }
  dropZone.addEventListener('drop', (event) => setFile(event.dataTransfer?.files?.[0] || null));

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    clearError();
    resetResult();
    const id = workerId.value.trim();
    if (!id) {
      showError('请输入 Work ID。');
      workerId.focus();
      return;
    }
    if (!selectedFile) {
      showError('请选择要注册的人脸照片。');
      return;
    }
    registerButton.disabled = true;
    registerButtonText.textContent = '正在检测与注册…';
    registerStatus.textContent = '正在执行 SCRFD 质量检查；合格后才会生成 AdaFace 特征。';
    try {
      const query = new URLSearchParams({
        worker_id: id,
        mode: mode.value,
        filename: selectedFile.name,
      });
      const response = await fetch(`/api/faces/register?${query.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': selectedFile.type || 'application/octet-stream' },
        body: selectedFile,
      });
      let data;
      try {
        data = await response.json();
      } catch (_) {
        data = { error: `服务返回 ${response.status}` };
      }
      renderResult(data, response.ok);
      registerStatus.textContent = data.stored
        ? '注册完成。运行时识别会自动在该 Work ID 的全部模板中寻找最相似角度。'
        : data.accepted
          ? '质量检查通过，但本次没有新增模板。'
          : '照片未入库，请按质量提示重新提交。';
      await lookupWorker();
    } catch (error) {
      showError(error instanceof Error ? error.message : String(error));
      registerStatus.textContent = '注册请求失败。';
    } finally {
      registerButton.disabled = false;
      registerButtonText.textContent = '检测质量并注册';
    }
  });
})();
