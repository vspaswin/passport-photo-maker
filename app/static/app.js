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

  let selectedFile = null;
  let modelReady = true;

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
    span.className = "pill " + (ok === true ? "ok" : ok === false ? "bad" : "neutral");
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

  function renderCheckResult(data) {
    showResultShell();
    preview.classList.add("hidden");
    preview.removeAttribute("src");
    downloads.classList.add("hidden");
    fileList.innerHTML = "";
    warningsEl.innerHTML = "";
    passBadge.classList.add("hidden");
    rejectBadge.classList.add("hidden");
    infoBadge.classList.remove("hidden");

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
    metricsEl.appendChild(pill("Recommendation: " + (rec || "?"), rec !== "retake"));

    validationBox.innerHTML = "";
    const summary = document.createElement("p");
    summary.className = "summary-text";
    summary.textContent = data.summary || "";
    validationBox.appendChild(summary);

    validationBox.appendChild(
      renderIssueList(
        data.as_is?.passed
          ? "As-is check — passed"
          : "As-is check — failed (not passport-ready yet)",
        data.as_is,
        data.as_is?.passed ? "ok" : "bad"
      )
    );
    validationBox.appendChild(
      renderIssueList(
        data.convertible?.passed
          ? "Convertible check — passed (Convert can try)"
          : "Convertible check — failed (cannot auto-fix)",
        data.convertible,
        data.convertible?.passed ? "ok" : "bad"
      )
    );
  }

  function renderConvertSuccess(data) {
    showResultShell();
    preview.classList.remove("hidden");
    preview.src = data.preview_data_url;
    passBadge.classList.remove("hidden");
    rejectBadge.classList.add("hidden");
    infoBadge.classList.add("hidden");
    downloads.classList.remove("hidden");

    metricsEl.innerHTML = "";
    const m = data.metrics || {};
    metricsEl.appendChild(pill("Final QC passed", true));
    metricsEl.appendChild(
      pill(`Head height: ${m.head_height_in}"`, m.head_height_ok === 1)
    );
    metricsEl.appendChild(
      pill(`Eyes from bottom: ${m.eye_from_bottom_in}"`, m.eye_position_ok === 1)
    );
    if (m.upload_600_kb != null) {
      metricsEl.appendChild(pill(`Upload 600: ${m.upload_600_kb} KB`, true));
    }

    validationBox.innerHTML = "";
    const p = document.createElement("p");
    p.className = "summary-text";
    p.textContent =
      "Converted and re-validated. Final photo passed automated Indian passport QC.";
    validationBox.appendChild(p);

    warningsEl.innerHTML = "";
    if (data.warnings && data.warnings.length) {
      warningsEl.innerHTML = data.warnings.map((w) => `<div>⚠ ${w}</div>`).join("");
    }

    fileList.innerHTML = "";
    (data.files || []).forEach((f) => {
      const li = document.createElement("li");
      const left = document.createElement("div");
      left.className = "meta";
      left.textContent = `${f.name} · ${f.size_kb} KB`;
      const a = document.createElement("a");
      a.href = f.download_url;
      a.textContent = "Download";
      a.setAttribute("download", f.name);
      li.appendChild(left);
      li.appendChild(a);
      fileList.appendChild(li);
    });
  }

  function renderConvertFailure(data) {
    showResultShell();
    preview.classList.add("hidden");
    preview.removeAttribute("src");
    passBadge.classList.add("hidden");
    rejectBadge.classList.remove("hidden");
    infoBadge.classList.add("hidden");
    downloads.classList.add("hidden");
    fileList.innerHTML = "";
    metricsEl.innerHTML = "";
    metricsEl.appendChild(
      pill(
        data.error === "output_validation_failed"
          ? "Converted but final QC failed"
          : "Not convertible",
        false
      )
    );
    warningsEl.innerHTML = "";
    validationBox.innerHTML = "";
    const p = document.createElement("p");
    p.className = "summary-text";
    p.textContent = data.message || "Conversion rejected.";
    validationBox.appendChild(p);
    validationBox.appendChild(
      renderIssueList("Issues", data.validation, "bad")
    );
  }

  async function refreshModelStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      modelReady = !!data.model_ready;
    } catch (_) {
      modelReady = true;
    }
  }

  async function postForm(url) {
    const form = new FormData();
    form.append("file", selectedFile);
    form.append("doc_type", docType.value);
    form.append("remove_bg", "true");
    form.append("strict", "true");
    const res = await fetch(url, { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    return { res, data };
  }

  checkBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    setBusy(true);
    setStatus("Checking photo (as-is + convertible)…", "");
    try {
      const { res, data } = await postForm("/api/validate");
      if (!res.ok) {
        throw new Error(data.detail || data.message || `HTTP ${res.status}`);
      }
      renderCheckResult(data);
      const kind =
        data.recommendation === "retake"
          ? "error"
          : data.recommendation === "convertible"
            ? ""
            : "ok";
      setStatus(data.summary || "Check complete.", kind);
    } catch (err) {
      console.error(err);
      setStatus(err.message || "Check failed.", "error");
    } finally {
      setBusy(false);
    }
  });

  convertBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    setBusy(true);
    await refreshModelStatus();
    if (!modelReady) {
      setStatus("Converting… may download model once (~170 MB)…", "");
    } else {
      setStatus("Checking convertible → converting → final QC…", "");
    }
    try {
      const { res, data } = await postForm("/api/convert");
      if (res.status === 422 || data.ok === false) {
        renderConvertFailure(data);
        setStatus(data.message || "Conversion rejected.", "error");
        return;
      }
      if (!res.ok) {
        const detail = data.detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : data.message || `HTTP ${res.status}`
        );
      }
      renderConvertSuccess(data);
      modelReady = true;
      setStatus(
        "Passed final QC. Download print + digital files below.",
        "ok"
      );
    } catch (err) {
      console.error(err);
      setStatus(err.message || "Conversion failed.", "error");
    } finally {
      setBusy(false);
    }
  });

  docType.addEventListener("change", updateDocDescription);
  updateDocDescription();
  refreshModelStatus();
})();
