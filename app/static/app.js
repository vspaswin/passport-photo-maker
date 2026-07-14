(() => {
  const docType = document.getElementById("docType");
  const docDescription = document.getElementById("docDescription");
  const removeBg = document.getElementById("removeBg");
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

  let selectedFile = null;

  const DESCRIPTIONS = {
    "indian-passport":
      "2×2 inch colour photo, white background, ICAO/VFS geometry for Passport / Visa / OCI.",
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
    setStatus("Ready to convert.");
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

  function renderResult(data) {
    emptyState.classList.add("hidden");
    result.classList.remove("hidden");
    preview.src = data.preview_data_url;

    metricsEl.innerHTML = "";
    const m = data.metrics || {};
    metricsEl.appendChild(
      pill(
        `Head height: ${m.head_height_in}"`,
        m.head_height_ok === 1
      )
    );
    metricsEl.appendChild(
      pill(
        `Eyes from bottom: ${m.eye_from_bottom_in}"`,
        m.eye_position_ok === 1
      )
    );
    if (m.upload_600_kb != null) {
      metricsEl.appendChild(pill(`Upload 600: ${m.upload_600_kb} KB`, true));
    }
    if (m.upload_350_kb != null) {
      metricsEl.appendChild(pill(`Upload 350: ${m.upload_350_kb} KB`, true));
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

  let modelReady = true;

  async function refreshModelStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      modelReady = !!data.model_ready;
    } catch (_) {
      modelReady = true; // don't scare the user if status fails
    }
  }

  convertBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    convertBtn.disabled = true;
    await refreshModelStatus();
    if (removeBg.checked && !modelReady) {
      setStatus(
        "Converting… downloading background-removal model (one-time, ~170 MB)…",
        ""
      );
    } else {
      setStatus("Converting…", "");
    }

    const form = new FormData();
    form.append("file", selectedFile);
    form.append("doc_type", docType.value);
    form.append("remove_bg", removeBg.checked ? "true" : "false");

    try {
      const res = await fetch("/api/convert", { method: "POST", body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(data.detail || data.message || `HTTP ${res.status}`);
      }
      renderResult(data);
      modelReady = true;
      setStatus("Done. Download print and digital files below.", "ok");
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
