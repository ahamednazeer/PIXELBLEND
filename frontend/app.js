const form = document.getElementById("blendForm");
const generateBtn = document.getElementById("generateBtn");
const statusText = document.getElementById("statusText");
const metaText = document.getElementById("metaText");
const resultImage = document.getElementById("resultImage");
const outputFrame = document.getElementById("outputFrame");
const placeholder = outputFrame.querySelector(".placeholder");
const downloadLink = document.getElementById("downloadLink");
const styleWeightInput = document.getElementById("styleWeight");
const weightValue = document.getElementById("weightValue");
const weightValueSecondary = document.getElementById("weightValueSecondary");
const detailStrengthInput = document.getElementById("detailStrength");
const detailValue = document.getElementById("detailValue");
const styleIntensityInput = document.getElementById("styleIntensity");
const styleIntensityValue = document.getElementById("styleIntensityValue");
const highQualityInput = document.getElementById("highQuality");
const filePickers = document.querySelectorAll(".file-picker");

function setStatus(message, tone = "neutral") {
  statusText.textContent = message;
  statusText.classList.toggle("error", tone === "error");
  statusText.classList.toggle("busy", tone === "busy");
  statusText.classList.toggle("success", tone === "success");
}

function setMetaEntries(entries) {
  metaText.innerHTML = "";
  entries.forEach((entry) => {
    if (!entry.value) return;
    const chip = document.createElement("span");
    chip.className = "meta-chip";

    const label = document.createElement("span");
    label.className = "meta-key";
    label.textContent = entry.label;

    const value = document.createElement("span");
    value.className = "meta-value";
    value.textContent = entry.value;

    chip.append(label, value);
    metaText.append(chip);
  });
}

function shortenFilename(fileName) {
  if (fileName.length <= 42) return fileName;
  const extensionIndex = fileName.lastIndexOf(".");
  const extension = extensionIndex > 0 ? fileName.slice(extensionIndex) : "";
  const trimmedBase = fileName.slice(0, Math.max(24, 42 - extension.length - 3));
  return `${trimmedBase}...${extension}`;
}

function setOutputState(hasImage) {
  resultImage.hidden = !hasImage;
  placeholder.hidden = hasImage;
  outputFrame.classList.toggle("has-image", hasImage);
}

function updateWeightLabel() {
  const percentage = Math.round(Number(styleWeightInput.value) * 100);
  const complement = 100 - percentage;
  weightValue.textContent = `${percentage}%`;
  styleWeightInput.style.setProperty("--range-progress", `${percentage}%`);
  if (weightValueSecondary) {
    weightValueSecondary.textContent = `Style 2: ${complement}%`;
  }
}

function updateDetailLabel() {
  const percentage = Math.round(Number(detailStrengthInput.value) * 100);
  detailValue.textContent = `${percentage}%`;
  detailStrengthInput.style.setProperty("--range-progress", `${percentage}%`);
}

function updateStyleIntensityLabel() {
  const percentage = Math.round(Number(styleIntensityInput.value) * 100);
  styleIntensityValue.textContent = `${percentage}%`;
  styleIntensityInput.style.setProperty("--range-progress", `${percentage}%`);
}

styleWeightInput.addEventListener("input", updateWeightLabel);
detailStrengthInput.addEventListener("input", updateDetailLabel);
styleIntensityInput.addEventListener("input", updateStyleIntensityLabel);

updateWeightLabel();
updateDetailLabel();
updateStyleIntensityLabel();

filePickers.forEach((picker) => {
  const inputId = picker.dataset.inputId;
  const input = document.getElementById(inputId);
  const trigger = picker.querySelector(".file-trigger");
  const fileNameText = picker.querySelector(".file-name");

  if (!input || !trigger || !fileNameText) return;

  const syncDisplayName = () => {
    const selected = input.files && input.files[0];
    const fullName = selected ? selected.name : "No file selected";
    fileNameText.textContent = shortenFilename(fullName);
    fileNameText.title = fullName;
    picker.classList.toggle("has-file", Boolean(selected));
  };

  trigger.addEventListener("click", () => {
    input.click();
  });

  picker.addEventListener("click", (event) => {
    if (event.target === picker || event.target === fileNameText) {
      input.click();
    }
  });

  input.addEventListener("change", syncDisplayName);
  syncDisplayName();
});

resultImage.addEventListener("load", () => {
  setOutputState(true);
});

resultImage.addEventListener("error", () => {
  setOutputState(false);
  setStatus("Generated image could not be loaded.", "error");
});

setOutputState(false);

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = new FormData(form);
  // Checkbox value 'true' is set in HTML, but if unchecked it won't be in FormData.
  // Backend expects 'true' or 'false' string or boolean.
  // We can manually Append if needed, but standard submit is usually fine.
  if (!payload.has("high_quality")) {
    payload.append("high_quality", "false");
  }
  generateBtn.disabled = true;
  generateBtn.classList.add("loading");
  setStatus("Generating stylized image...", "busy");
  downloadLink.hidden = true;
  placeholder.textContent = "Rendering preview...";
  setOutputState(false);

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      body: payload,
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const detail = body.detail || "Generation failed.";
      throw new Error(detail);
    }

    const data = await response.json();
    const outputUrl = `${data.output_path}?t=${Date.now()}`;

    resultImage.src = outputUrl;

    downloadLink.href = data.download_path;
    downloadLink.hidden = false;

    setMetaEntries([
      { label: "Job", value: data.job_id },
      { label: "Model mode", value: data.model_mode },
    ]);
    setStatus("Done. You can now download the output.", "success");
    placeholder.textContent = "Generated image appears here.";
  } catch (error) {
    setStatus(error.message || "Generation failed.", "error");
    placeholder.textContent = "Generated image appears here.";
  } finally {
    generateBtn.disabled = false;
    generateBtn.classList.remove("loading");
  }
});

(async () => {
  try {
    const modelResponse = await fetch("/api/model-status");
    if (!modelResponse.ok) return;

    const modelStatus = await modelResponse.json();
    const entries = [
      { label: "Startup mode", value: modelStatus.mode },
      { label: "Model status", value: modelStatus.message },
    ];

    setMetaEntries(entries);
  } catch {
    // Keep UI usable even if status endpoint is unavailable.
  }
})();
