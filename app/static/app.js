(() => {
  const docType = document.getElementById("docType");
  const docDescription = document.getElementById("docDescription");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const fileName = document.getElementById("fileName");
  const checkBtn = document.getElementById("checkBtn");
  const convertBtn = document.getElementById("convertBtn");
  const status = document.getElementById("status");
  const emptyState = document.getElementById("emptyState");
  const result = document.getElementById("result");
  const preview = document.getElementById("preview");
  const metricsEl = document.getElementById("metrics");
  const warningsEl = document.getElementById("warnings");
  const fileList = document.getElementById("fileList");
  const downloads = document.getElementById("downloads");
  const validationBox = document.getElementById("validationBox");
  const rejectBadge = document.getElementById("rejectBadge");
  const passBadge = document.getElementById("passBadge");
  const infoBadge = document.getElementById("infoBadge");
  const usageBox = document.getElementById("usageBox");
  const progress = document.getElementById("progress");
  const disclaimer = document.getElementById("disclaimer");
  const printGuide = document.getElementById("printGuide");

  let selectedFile = null;

  const DESCRIPTIONS = {
    "indian-passport":
      "Indian Passport / Visa / OCI — 2×2\", white background, VFS/MEA geometry.",
  };

  function setStatus(msg, kind) {
    status.textContent = msg || "";
    status.className = "status" + (kind ? " " + kind : "");
  }

  function setBusy(busy) {
    checkBtn.disabled = busy || !selectedFile;
    convertBtn.disabled = busy || !selectedFile;
  }

  function setProgress(active) {
    if (!active) {
      progress.classList.add("hidden");
      progress.querySelectorAll("[data-step]").forEach((el) => el.classList.remove("on", "done"));
      return;
    }
    progress.classList.remove("hidden");
    const order = ["check", "bg", "frame", "qc"];
    const idx = order.indexOf(active);
    progress.querySelectorAll("[data-step]").forEach((el) => {
      const step = el.getAttribute("data-step");
      const si = order.indexOf(step);
      el.classList.toggle("on", si === idx);
      el.classList.toggle("done", si < idx);
    });
  }

  function updateDocDescription() {
    docDescription.textContent = DESCRIPTIONS[docType.value] || "";
  }

  function setFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setStatus("Please choose an image file.", "error");
      return;
    }
    selectedFile = file;
    fileName.textContent = file.name + " · " + Math.round(file.size / 1024) + " KB";
    setBusy(false);
    setStatus("Ready — Check only, or Convert to passport.");
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) setFile(fileInput.files[0]);
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
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
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
      const res = await fetch("/api/status");
      const data = await res.json();
      updateUsage(data.usage);
    } catch (_) {}
  }

  function renderCheckResult(data) {
    showResultShell();
    setProgress(null);
    preview.classList.add("hidden");
    preview.removeAttribute("src");
    downloads.classList.add("hidden");
    fileList.innerHTML = "";
    warningsEl.innerHTML = "";
    printGuide.classList.add("hidden");
    passBadge.classList.add("hidden");
    rejectBadge.classList.add("hidden");
    infoBadge.classList.remove("hidden");
    disclaimer.classList.remove("hidden");
    disclaimer.textContent = data.disclaimer || "";

    const rec = data.recommendation;
    const labels = {
      already_ok: "Already OK as-is",
      convertible: "Not OK as-is · Convertible",
      retake: "Retake required",
    };
    infoBadge.textContent = labels[rec] || rec;
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
        data.convertible?.passed
          ? "Convertible — passed"
          : "Convertible — failed",
        data.convertible,
        data.convertible?.passed ? "ok" : "bad"
      )
    );
    updateUsage(data.usage);
  }

  function renderConvertSuccess(data) {
    showResultShell();
    setProgress(null);
    preview.classList.remove("hidden");
    preview.src = data.preview_data_url;
    passBadge.classList.remove("hidden");
    rejectBadge.classList.add("hidden");
    infoBadge.classList.add("hidden");
    downloads.classList.remove("hidden");
    disclaimer.classList.remove("hidden");
    disclaimer.textContent = data.disclaimer || "";

    metricsEl.innerHTML = "";
    const m = data.metrics || {};
    metricsEl.appendChild(pill("Final QC passed", true));
    if (data.job_id) metricsEl.appendChild(pill("Job " + data.job_id.slice(0, 8), true));
    metricsEl.appendChild(pill(`Head: ${m.head_height_in}"`, m.head_height_ok === 1));
    if (m.upload_600_kb != null) {
      metricsEl.appendChild(pill(`Upload 600: ${m.upload_600_kb} KB`, true));
    }

    validationBox.innerHTML = "";
    const p = document.createElement("p");
    p.className = "summary-text";
    p.textContent =
      "Converted and re-validated. Ready for portal upload and physical print.";
    validationBox.appendChild(p);

    if (data.print_tip) {
      printGuide.classList.remove("hidden");
      printGuide.innerHTML =
        `<h3>Print guide (Letter / GP-701)</h3><p>Use <code>${escapeHtml(
          data.print_tip.letter_file || "*_sheet_letter.jpg"
        )}</code></p><ul>` +
        (data.print_tip.settings || [])
          .map((s) => `<li>${escapeHtml(s)}</li>`)
          .join("") +
        "</ul>";
    } else {
      printGuide.classList.add("hidden");
    }

    warningsEl.innerHTML = "";
    if (data.warnings && data.warnings.length) {
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
  }

  function renderConvertFailure(data) {
    showResultShell();
    setProgress(null);
    preview.classList.add("hidden");
    preview.removeAttribute("src");
    passBadge.classList.add("hidden");
    rejectBadge.classList.remove("hidden");
    infoBadge.classList.add("hidden");
    downloads.classList.add("hidden");
    printGuide.classList.add("hidden");
    fileList.innerHTML = "";
    metricsEl.innerHTML = "";
    metricsEl.appendChild(
      pill(
        data.error === "payment_required"
          ? "Credits required"
          : data.error === "output_validation_failed"
            ? "Final QC failed"
            : "Not convertible",
        false
      )
    );
    disclaimer.classList.add("hidden");
    warningsEl.innerHTML = "";
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

  async function postForm(url) {
    const form = new FormData();
    form.append("file", selectedFile);
    form.append("doc_type", docType.value);
    const res = await fetch(url, { method: "POST", body: form, credentials: "same-origin" });
    const data = await res.json().catch(() => ({}));
    return { res, data };
  }

  checkBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    setBusy(true);
    setProgress("check");
    setStatus("Checking photo…", "");
    try {
      const { res, data } = await postForm("/api/validate");
      if (res.status === 429) {
        renderConvertFailure(data);
        setStatus(data.message || "Check quota exceeded.", "error");
        return;
      }
      if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
      renderCheckResult(data);
      setStatus(data.summary || "Check complete.", data.recommendation === "retake" ? "error" : "ok");
    } catch (err) {
      console.error(err);
      setStatus(err.message || "Check failed.", "error");
    } finally {
      setBusy(false);
      setProgress(null);
    }
  });

  convertBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    setBusy(true);
    setProgress("check");
    setStatus("Validating → removing background → framing → final QC…", "");
    // Visual progress simulation while server works
    const timers = [
      setTimeout(() => setProgress("bg"), 400),
      setTimeout(() => setProgress("frame"), 2000),
      setTimeout(() => setProgress("qc"), 4000),
    ];
    try {
      const { res, data } = await postForm("/api/convert");
      timers.forEach(clearTimeout);
      if (res.status === 402 || res.status === 429) {
        renderConvertFailure(data);
        setStatus(data.message || "Payment or quota required.", "error");
        return;
      }
      if (res.status === 422 || data.ok === false) {
        renderConvertFailure(data);
        setStatus(data.message || "Conversion rejected.", "error");
        return;
      }
      if (!res.ok) throw new Error(data.detail || data.message || `HTTP ${res.status}`);
      renderConvertSuccess(data);
      setStatus("Passed automated QC. Download files below.", "ok");
    } catch (err) {
      timers.forEach(clearTimeout);
      console.error(err);
      setStatus(err.message || "Conversion failed.", "error");
    } finally {
      setBusy(false);
      setProgress(null);
    }
  });

  document.querySelectorAll(".buy-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const pack = btn.getAttribute("data-pack");
      const form = new FormData();
      form.append("pack_id", pack);
      try {
        const res = await fetch("/api/billing/checkout", {
          method: "POST",
          body: form,
          credentials: "same-origin",
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || data.message || "Checkout failed");
        if (data.checkout_url) window.location.href = data.checkout_url;
      } catch (err) {
        setStatus(err.message || "Checkout failed", "error");
      }
    });
  });

  docType.addEventListener("change", updateDocDescription);
  updateDocDescription();
  refreshStatus();

  const params = new URLSearchParams(location.search);
  if (params.get("billing") === "success") {
    setStatus("Payment received — credits will appear shortly after webhook.", "ok");
    refreshStatus();
  }
})();
