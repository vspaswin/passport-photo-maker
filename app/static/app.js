(() => {
  const $ = (id) => document.getElementById(id);
  const docType = $("docType");
  const docDescription = $("docDescription");
  const childMode = $("childMode");
  const dropzone = $("dropzone");
  const fileInput = $("fileInput");
  const fileName = $("fileName");
  const checkBtn = $("checkBtn");
  const convertBtn = $("convertBtn");
  const reframeBtn = $("reframeBtn");
  const status = $("status");
  const emptyState = $("emptyState");
  const result = $("result");
  const preview = $("preview");
  const guidePreview = $("guidePreview");
  const originalPreview = $("originalPreview");
  const showGuides = $("showGuides");
  const metricsEl = $("metrics");
  const warningsEl = $("warnings");
  const fileList = $("fileList");
  const downloads = $("downloads");
  const validationBox = $("validationBox");
  const rejectBadge = $("rejectBadge");
  const passBadge = $("passBadge");
  const infoBadge = $("infoBadge");
  const usageBox = $("usageBox");
  const progress = $("progress");
  const disclaimer = $("disclaimer");
  const printGuide = $("printGuide");
  const finetunePanel = $("finetunePanel");
  const scaleSlider = $("scaleSlider");
  const oxSlider = $("oxSlider");
  const oySlider = $("oySlider");
  const scaleVal = $("scaleVal");
  const oxVal = $("oxVal");
  const oyVal = $("oyVal");

  let selectedFile = null;
  let currentJobId = null;

  const DESCRIPTIONS = {
    "indian-passport":
      "India 2×2″ (VFS / abroad / US-style) — square, white BG, head 1–1⅜″.",
    "passport-seva-35x45":
      "Passport Seva India — 35×45 mm print + 630×810 upload (<250 KB), face ~80–85%.",
    "us-passport":
      "US Passport / Visa — 2×2″, white background, ICAO-style geometry.",
  };

  function setStatus(msg, kind) {
    status.textContent = msg || "";
    status.className = "status" + (kind ? " " + kind : "");
  }

  function setBusy(busy) {
    checkBtn.disabled = busy || !selectedFile;
    convertBtn.disabled = busy || !selectedFile;
    if (reframeBtn) reframeBtn.disabled = busy || !currentJobId;
  }

  function setProgress(active) {
    if (!active) {
      progress.classList.add("hidden");
      progress.querySelectorAll("[data-step]").forEach((el) =>
        el.classList.remove("on", "done")
      );
      return;
    }
    progress.classList.remove("hidden");
    const order = ["check", "bg", "frame", "qc"];
    const idx = order.indexOf(active);
    progress.querySelectorAll("[data-step]").forEach((el) => {
      const si = order.indexOf(el.getAttribute("data-step"));
      el.classList.toggle("on", si === idx);
      el.classList.toggle("done", si < idx);
    });
  }

  function updateDocDescription() {
    docDescription.textContent = DESCRIPTIONS[docType.value] || "";
  }

  function setFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      setStatus("Please choose an image file.", "error");
      return;
    }
    selectedFile = file;
    currentJobId = null;
    fileName.textContent = file.name + " · " + Math.round(file.size / 1024) + " KB";
    setBusy(false);
    setStatus("Ready — Check or Convert.");
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files?.[0]) setFile(fileInput.files[0]);
  });
  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
  });

  function pill(label, ok) {
    const span = document.createElement("span");
    span.className =
      "pill " + (ok === true ? "ok" : ok === false ? "bad" : "neutral");
    span.textContent = label;
    return span;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function showResultShell() {
    emptyState.classList.add("hidden");
    result.classList.remove("hidden");
  }

  function renderIssueList(title, report, titleClass) {
    const wrap = document.createElement("div");
    wrap.className = "val-section";
    const h = document.createElement("h3");
    h.className = "val-title " + (titleClass || "");
    h.textContent = title;
    wrap.appendChild(h);
    const issues = (report && report.issues) || [];
    if (!issues.length) {
      const p = document.createElement("p");
      p.className = "hint";
      p.textContent = "No issues.";
      wrap.appendChild(p);
      return wrap;
    }
    const list = document.createElement("ul");
    list.className = "issue-list";
    issues.forEach((issue) => {
      const li = document.createElement("li");
      li.innerHTML =
        `<strong>${escapeHtml(issue.message)}</strong>` +
        `<div class="fix">${escapeHtml(issue.how_to_fix || "")}</div>` +
        `<div class="code">${escapeHtml(issue.code || "")}</div>`;
      list.appendChild(li);
    });
    wrap.appendChild(list);
    return wrap;
  }

  function updateUsage(usage) {
    if (!usage || !usageBox) return;
    usageBox.innerHTML =
      `Today: <strong>${usage.checks}</strong> checks · ` +
      `<strong>${usage.converts}</strong> converts · ` +
      `<strong>${usage.credit_balance}</strong> credits`;
  }

  async function refreshStatus() {
    try {
      const res = await fetch("/api/status", { credentials: "same-origin" });
      const data = await res.json();
      updateUsage(data.usage);
    } catch (_) {}
  }

  function syncSliders(ft) {
    if (!ft) return;
    scaleSlider.value = ft.scale_factor ?? 1;
    oxSlider.value = ft.offset_x_frac ?? 0;
    oySlider.value = ft.offset_y_frac ?? 0;
    scaleVal.textContent = Number(scaleSlider.value).toFixed(2);
    oxVal.textContent = Number(oxSlider.value).toFixed(3);
    oyVal.textContent = Number(oySlider.value).toFixed(3);
  }

  [scaleSlider, oxSlider, oySlider].forEach((el) => {
    el.addEventListener("input", () => {
      scaleVal.textContent = Number(scaleSlider.value).toFixed(2);
      oxVal.textContent = Number(oxSlider.value).toFixed(3);
      oyVal.textContent = Number(oySlider.value).toFixed(3);
    });
  });

  function toggleGuides() {
    const on = showGuides.checked;
    if (on && guidePreview.src) {
      preview.classList.add("hidden");
      guidePreview.classList.remove("hidden");
    } else {
      guidePreview.classList.add("hidden");
      if (preview.src) preview.classList.remove("hidden");
    }
  }
  showGuides.addEventListener("change", toggleGuides);

  function renderCheckResult(data) {
    showResultShell();
    setProgress(null);
    currentJobId = null;
    finetunePanel.classList.add("hidden");
    preview.classList.add("hidden");
    guidePreview.classList.add("hidden");
    originalPreview.classList.add("hidden");
    downloads.classList.add("hidden");
    printGuide.classList.add("hidden");
    passBadge.classList.add("hidden");
    rejectBadge.classList.add("hidden");
    infoBadge.classList.remove("hidden");
    disclaimer.classList.remove("hidden");
    disclaimer.textContent = data.disclaimer || "";

    const rec = data.recommendation;
    infoBadge.textContent =
      rec === "already_ok"
        ? "Already OK as-is"
        : rec === "convertible"
          ? "Convertible"
          : "Retake required";
    infoBadge.className =
      "info-badge " +
      (rec === "already_ok" ? "ok" : rec === "convertible" ? "warn" : "bad");

    metricsEl.innerHTML = "";
    metricsEl.appendChild(pill("As-is", !!data.as_is?.passed));
    metricsEl.appendChild(pill("Convertible", !!data.convertible?.passed));

    validationBox.innerHTML = "";
    const summary = document.createElement("p");
    summary.className = "summary-text";
    summary.textContent = data.summary || "";
    validationBox.appendChild(summary);
    validationBox.appendChild(
      renderIssueList(
        data.as_is?.passed ? "As-is — passed" : "As-is — failed",
        data.as_is,
        data.as_is?.passed ? "ok" : "bad"
      )
    );
    validationBox.appendChild(
      renderIssueList(
        data.convertible?.passed ? "Convertible — passed" : "Convertible — failed",
        data.convertible,
        data.convertible?.passed ? "ok" : "bad"
      )
    );
    updateUsage(data.usage);
  }

  function renderConvertSuccess(data) {
    showResultShell();
    setProgress(null);
    currentJobId = data.job_id || null;
    passBadge.classList.remove("hidden");
    rejectBadge.classList.add("hidden");
    infoBadge.classList.add("hidden");
    downloads.classList.remove("hidden");
    finetunePanel.classList.toggle("hidden", !currentJobId);
    disclaimer.classList.remove("hidden");
    disclaimer.textContent = data.disclaimer || "";
    syncSliders(data.finetune);

    preview.classList.remove("hidden");
    preview.src = data.preview_data_url || data.preview_url || "";
    if (data.guide_url) {
      guidePreview.src = data.guide_url + "?t=" + Date.now();
    }
    if (data.original_url) {
      originalPreview.src = data.original_url + "?t=" + Date.now();
      originalPreview.classList.remove("hidden");
    }
    showGuides.checked = false;
    toggleGuides();

    metricsEl.innerHTML = "";
    const m = data.metrics || {};
    metricsEl.appendChild(pill("QC passed", true));
    if (data.job_id)
      metricsEl.appendChild(pill("Job " + data.job_id.slice(0, 8), true));
    metricsEl.appendChild(pill(`Head: ${m.head_height_in}"`, true));
    if (m.upload_600_kb != null)
      metricsEl.appendChild(pill(`Upload: ${m.upload_600_kb} KB`, true));

    validationBox.innerHTML = "";
    const p = document.createElement("p");
    p.className = "summary-text";
    p.textContent =
      data.mode === "reframe"
        ? "Reframed and re-validated. Downloads updated."
        : "Converted and validated. Fine-tune framing if needed, then download.";
    validationBox.appendChild(p);

    if (data.print_tip) {
      printGuide.classList.remove("hidden");
      printGuide.innerHTML =
        `<h3>Print (Letter / GP-701)</h3><p>Use <code>${escapeHtml(
          data.print_tip.letter_file || "*_sheet_letter.jpg"
        )}</code></p><ul>` +
        (data.print_tip.settings || [])
          .map((s) => `<li>${escapeHtml(s)}</li>`)
          .join("") +
        "</ul>";
    }

    warningsEl.innerHTML = "";
    if (data.warnings?.length) {
      warningsEl.innerHTML = data.warnings.map((w) => `<div>⚠ ${w}</div>`).join("");
    }

    fileList.innerHTML = "";
    (data.files || []).forEach((f) => {
      const li = document.createElement("li");
      const left = document.createElement("div");
      left.className = "meta";
      left.textContent = `${f.name}${f.size_kb != null ? " · " + f.size_kb + " KB" : ""}`;
      const a = document.createElement("a");
      a.href = f.download_url;
      a.textContent = "Download";
      a.setAttribute("download", f.name);
      li.appendChild(left);
      li.appendChild(a);
      fileList.appendChild(li);
    });
    updateUsage(data.usage);
    setBusy(false);
  }

  function renderConvertFailure(data) {
    showResultShell();
    setProgress(null);
    currentJobId = null;
    finetunePanel.classList.add("hidden");
    preview.classList.add("hidden");
    guidePreview.classList.add("hidden");
    originalPreview.classList.add("hidden");
    passBadge.classList.add("hidden");
    rejectBadge.classList.remove("hidden");
    infoBadge.classList.add("hidden");
    downloads.classList.add("hidden");
    printGuide.classList.add("hidden");
    metricsEl.innerHTML = "";
    metricsEl.appendChild(pill(data.error || "Failed", false));
    disclaimer.classList.add("hidden");
    validationBox.innerHTML = "";
    const p = document.createElement("p");
    p.className = "summary-text";
    p.textContent = data.message || "Rejected.";
    validationBox.appendChild(p);
    if (data.validation) {
      validationBox.appendChild(renderIssueList("Issues", data.validation, "bad"));
    }
    updateUsage(data.usage);
  }

  function formBase() {
    const form = new FormData();
    form.append("doc_type", docType.value);
    form.append("child_mode", childMode.checked ? "true" : "false");
    form.append("scale_factor", scaleSlider.value);
    form.append("offset_x_frac", oxSlider.value);
    form.append("offset_y_frac", oySlider.value);
    return form;
  }

  checkBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    setBusy(true);
    setProgress("check");
    setStatus("Checking…", "");
    try {
      const form = formBase();
      form.append("file", selectedFile);
      const res = await fetch("/api/validate", {
        method: "POST",
        body: form,
        credentials: "same-origin",
      });
      const data = await res.json();
      if (res.status === 429) {
        renderConvertFailure(data);
        setStatus(data.message, "error");
        return;
      }
      if (!res.ok) throw new Error(data.detail || data.message || "Check failed");
      renderCheckResult(data);
      setStatus(data.summary || "Done", data.recommendation === "retake" ? "error" : "ok");
    } catch (e) {
      setStatus(e.message, "error");
    } finally {
      setBusy(false);
      setProgress(null);
    }
  });

  convertBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    setBusy(true);
    setProgress("check");
    setStatus("Converting…", "");
    const timers = [
      setTimeout(() => setProgress("bg"), 300),
      setTimeout(() => setProgress("frame"), 1500),
      setTimeout(() => setProgress("qc"), 3500),
    ];
    try {
      const form = formBase();
      form.append("file", selectedFile);
      const res = await fetch("/api/convert", {
        method: "POST",
        body: form,
        credentials: "same-origin",
      });
      const data = await res.json();
      timers.forEach(clearTimeout);
      if (!res.ok || data.ok === false) {
        renderConvertFailure(data);
        setStatus(data.message || "Rejected", "error");
        return;
      }
      renderConvertSuccess(data);
      setStatus("QC passed. Fine-tune if needed, then download.", "ok");
    } catch (e) {
      timers.forEach(clearTimeout);
      setStatus(e.message, "error");
    } finally {
      setBusy(false);
      setProgress(null);
    }
  });

  reframeBtn.addEventListener("click", async () => {
    if (!currentJobId) return;
    setBusy(true);
    setStatus("Applying fine-tune…", "");
    try {
      const form = new FormData();
      form.append("scale_factor", scaleSlider.value);
      form.append("offset_x_frac", oxSlider.value);
      form.append("offset_y_frac", oySlider.value);
      const res = await fetch(`/api/jobs/${currentJobId}/reframe`, {
        method: "POST",
        body: form,
        credentials: "same-origin",
      });
      const data = await res.json();
      if (!res.ok || data.ok === false) {
        setStatus(data.message || "Reframe failed QC", "error");
        if (data.validation) {
          validationBox.innerHTML = "";
          validationBox.appendChild(
            renderIssueList("Fine-tune QC issues", data.validation, "bad")
          );
        }
        return;
      }
      renderConvertSuccess(data);
      setStatus("Fine-tune applied.", "ok");
    } catch (e) {
      setStatus(e.message, "error");
    } finally {
      setBusy(false);
    }
  });

  document.querySelectorAll(".buy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const form = new FormData();
      form.append("pack_id", btn.getAttribute("data-pack"));
      try {
        const res = await fetch("/api/billing/checkout", {
          method: "POST",
          body: form,
          credentials: "same-origin",
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Checkout failed");
        if (data.checkout_url) location.href = data.checkout_url;
      } catch (e) {
        setStatus(e.message, "error");
      }
    });
  });

  docType.addEventListener("change", updateDocDescription);
  updateDocDescription();
  refreshStatus();
})();
