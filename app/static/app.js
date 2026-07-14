(() => {
  const docType = document.getElementById("docType");
  const docDescription = document.getElementById("docDescription");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const fileName = document.getElementById("fileName");
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

  let selectedFile = null;
  let modelReady = true;

  const DESCRIPTIONS = {
    "indian-passport":
      "2×2 inch colour photo, white background, ICAO/VFS geometry. Strict automated QC required before any download.",
  };

  function setStatus(msg, kind) {
    status.textContent = msg || "";
    status.className = "status" + (kind ? " " + kind : "");
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
    convertBtn.disabled = false;
    setStatus("Ready — will validate, then convert only if checks pass.");
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
    span.className = "pill " + (ok ? "ok" : "bad");
    span.textContent = label;
    return span;
  }

  function showResultShell() {
    emptyState.classList.add("hidden");
    result.classList.remove("hidden");
  }

  function renderValidation(validation, passed) {
    validationBox.innerHTML = "";
    if (!validation) return;

    const title = document.createElement("h3");
    title.textContent = passed
      ? "Automated checks — all passed"
      : "Photo rejected — fix these issues";
    title.className = passed ? "val-title ok" : "val-title bad";
    validationBox.appendChild(title);

    const issues = validation.issues || [];
    if (!passed && issues.length) {
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
      validationBox.appendChild(list);
    } else if (passed) {
      const p = document.createElement("p");
      p.className = "hint";
      p.textContent =
        "Source and final photo passed face, eyes, sharpness, lighting, clothing, background, and geometry checks.";
      validationBox.appendChild(p);
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderSuccess(data) {
    showResultShell();
    preview.classList.remove("hidden");
    preview.src = data.preview_data_url;
    passBadge.classList.remove("hidden");
    rejectBadge.classList.add("hidden");
    downloads.classList.remove("hidden");

    metricsEl.innerHTML = "";
    const m = data.metrics || {};
    metricsEl.appendChild(pill("QC passed", true));
    metricsEl.appendChild(
      pill(`Head height: ${m.head_height_in}"`, m.head_height_ok === 1)
    );
    metricsEl.appendChild(
      pill(`Eyes from bottom: ${m.eye_from_bottom_in}"`, m.eye_position_ok === 1)
    );
    if (m.upload_600_kb != null) {
      metricsEl.appendChild(pill(`Upload 600: ${m.upload_600_kb} KB`, true));
    }

    renderValidation(data.validation, true);

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

  function renderFailure(data) {
    showResultShell();
    preview.classList.add("hidden");
    preview.removeAttribute("src");
    passBadge.classList.add("hidden");
    rejectBadge.classList.remove("hidden");
    downloads.classList.add("hidden");
    fileList.innerHTML = "";
    metricsEl.innerHTML = "";
    metricsEl.appendChild(pill("QC failed — no downloads", false));
    warningsEl.innerHTML = "";
    renderValidation(data.validation, false);
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

  convertBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    convertBtn.disabled = true;
    await refreshModelStatus();
    if (!modelReady) {
      setStatus(
        "Validating… may download background model once (~170 MB)…",
        ""
      );
    } else {
      setStatus("Validating photo, then converting if it passes…", "");
    }

    const form = new FormData();
    form.append("file", selectedFile);
    form.append("doc_type", docType.value);
    form.append("remove_bg", "true");
    form.append("strict", "true");

    try {
      const res = await fetch("/api/convert", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));

      if (res.status === 422 || data.error === "validation_failed") {
        renderFailure(data);
        const n = (data.validation && data.validation.issues
          ? data.validation.issues.length
          : 0);
        setStatus(
          data.message ||
            `Photo rejected (${n} issue${n === 1 ? "" : "s"}). Fix and retake.`,
          "error"
        );
        return;
      }

      if (!res.ok) {
        const detail = data.detail;
        const msg =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d) => d.msg || d).join("; ")
              : data.message || `HTTP ${res.status}`;
        throw new Error(msg);
      }

      renderSuccess(data);
      modelReady = true;
      setStatus(
        "Passed automated QC. Safe to download print + digital files.",
        "ok"
      );
    } catch (err) {
      console.error(err);
      setStatus(err.message || "Conversion failed.", "error");
    } finally {
      convertBtn.disabled = !selectedFile;
    }
  });

  docType.addEventListener("change", updateDocDescription);
  updateDocDescription();
  refreshModelStatus();
})();
