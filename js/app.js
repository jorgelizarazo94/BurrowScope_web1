const MODEL_URL = "models/bank_swallow_burrow_yolo11s_seg_production.onnx";
const METADATA_URL = "models/bank_swallow_burrow_yolo11s_seg_production_metadata.json";
const INPUT_SIZE = 1280;
const MAX_DISPLAY_WIDTH = 1400;
const MAX_DISPLAY_HEIGHT = 900;
const THRESHOLD_DIAGNOSTIC_CONFIDENCES = [0.05, 0.1, 0.2, 0.25, 0.35, 0.5];

let session = null;
let metadata = null;
let currentImage = null;
let currentImageName = null;
let currentDetections = [];
let lastOutputs = null;
let lastLetterbox = null;
let sampleManifest = [];
let selectedSampleIds = new Set();
let batchResults = [];
let batchRunning = false;
let lastThresholdDiagnostics = [];
let activeLanguage = "en";
let termsAccepted = false;
let termsDeclined = false;
let sessionAnalyses = [];
let currentAnalysisIndex = -1;

const TRANSLATIONS = {
  en: {
    headerTagline: "Training models for easier wildlife monitoring",
    fundingLabel: "Funding and support",
    heroEyebrow: "Training models for easier wildlife monitoring",
    heroTitle: "Bank Swallow burrow inspection",
    heroCopy: "Run local ONNX inference on one photo or selected image batches, then review boxes, masks, centroids, and exports.",
    quickTitle: "Visual inspection workflow",
    quickStep1: "Choose one bank photo, selected examples, or a local image set to inspect.",
    quickStep2: "Run the ONNX model in your browser and keep every image local.",
    quickStep3: "Review post-NMS boxes, masks, centroids, and confidence behaviour.",
    quickStep4: "Export CSV, JSON, or an overlay PNG for local field records.",
    modelUpdateHint: "If there is an updated inference model, upload it here.",
    consoleEyebrow: "Processing log",
    consoleTitle: "Session",
    footerCopy: "Bank Swallow burrow segmentation and inspection workspace.",
    termsEyebrow: "Local ONNX workspace",
    termsTitle: "Use conditions",
    termsP1: "BurrowScope is provided for non-profit research, conservation, and field inspection work.",
    termsP2: "The inference engine runs locally in this browser through ONNX Runtime. Your images, masks, counts, and exported results are not sent to a cloud service by this app.",
    termsP3: "Review the outputs before using them in reports. The tool supports per-image burrow segmentation and visual inspection.",
    termsDecline: "Do not accept",
    termsAccept: "Accept and continue locally",
    pausedText: "Use is paused until the local-use conditions are accepted.",
    reviewTerms: "Review terms",
  },
  fr: {
    headerTagline: "Entraîner des modèles pour faciliter le suivi de la faune",
    fundingLabel: "Financement et soutien",
    heroEyebrow: "Entraîner des modèles pour faciliter le suivi de la faune",
    heroTitle: "Inspection des terriers d'hirondelles de rivage",
    heroCopy: "Lancez l'inférence ONNX locale sur une photo ou sur des lots d'images sélectionnées, puis révisez les boîtes, les masques, les centroïdes et les exports.",
    quickTitle: "Flux d'inspection visuelle",
    quickStep1: "Choisissez une photo de berge, des exemples sélectionnés ou un ensemble local d'images à inspecter.",
    quickStep2: "Exécutez le modèle ONNX dans votre navigateur et gardez chaque image en local.",
    quickStep3: "Révisez les boîtes post-NMS, les masques, les centroïdes et le comportement des seuils de confiance.",
    quickStep4: "Exportez un CSV, un JSON ou un PNG de superposition pour vos dossiers de terrain.",
    modelUpdateHint: "S'il existe une mise à jour du modèle d'inférence, téléversez-la ici.",
    consoleEyebrow: "Journal de traitement",
    consoleTitle: "Session",
    footerCopy: "Espace d'inspection et de segmentation des terriers d'hirondelles de rivage.",
    termsEyebrow: "Espace ONNX local",
    termsTitle: "Conditions d'utilisation",
    termsP1: "BurrowScope est fourni pour la recherche, la conservation et l'inspection de terrain sans but lucratif.",
    termsP2: "Le moteur d'inférence fonctionne localement dans ce navigateur avec ONNX Runtime. Vos images, masques, comptes et résultats exportés ne sont pas envoyés vers un service infonuagique par cette application.",
    termsP3: "Veuillez réviser les résultats avant de les utiliser dans des rapports. L'outil sert à la segmentation et à l'inspection visuelle par image.",
    termsDecline: "Ne pas accepter",
    termsAccept: "Accepter et continuer en local",
    pausedText: "L'utilisation est en pause jusqu'à l'acceptation des conditions locales.",
    reviewTerms: "Revoir les conditions",
  },
};

function qs(id) {
  return document.getElementById(id);
}

function setLanguage(lang) {
  activeLanguage = lang === "fr" ? "fr" : "en";
  document.documentElement.lang = activeLanguage === "fr" ? "fr-CA" : "en";
  const copy = TRANSLATIONS[activeLanguage];
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    if (copy[key]) el.innerText = copy[key];
  });
  document.querySelectorAll("[data-lang-toggle]").forEach((button) => {
    button.classList.toggle("active", button.dataset.langToggle === activeLanguage);
  });
}

function openTermsModal() {
  const modal = qs("termsModal");
  if (!modal) return;
  modal.classList.remove("hidden");
  modal.classList.add("flex");
}

function closeTermsModal() {
  const modal = qs("termsModal");
  if (!modal) return;
  modal.classList.add("hidden");
  modal.classList.remove("flex");
}

function refreshUsageGate() {
  const locked = !termsAccepted;
  const imageInput = qs("imageInput");
  const imageLabel = qs("imageInputLabel");
  const runButton = qs("runBtn");
  const manualButton = qs("manualModelUpload");
  if (imageInput) imageInput.disabled = locked || !session;
  if (manualButton) manualButton.disabled = locked;
  if (imageLabel) {
    imageLabel.classList.toggle("opacity-50", locked || !session);
    imageLabel.classList.toggle("cursor-not-allowed", locked || !session);
    imageLabel.classList.toggle("cursor-pointer", !locked && Boolean(session));
    imageLabel.classList.toggle("border-teal-400", !locked && Boolean(session));
    imageLabel.classList.toggle("bg-teal-50", !locked && Boolean(session));
  }
  if (runButton) runButton.disabled = locked || !session || !currentImage || batchRunning;
  setHiddenIfPresent("termsPausedBar", !(locked && termsDeclined));
  updateSampleSelectionUi();
}

function acceptTerms() {
  termsAccepted = true;
  termsDeclined = false;
  closeTermsModal();
  refreshUsageGate();
}

function declineTerms() {
  termsAccepted = false;
  termsDeclined = true;
  closeTermsModal();
  refreshUsageGate();
}

function setTextIfPresent(id, text) {
  const el = qs(id);
  if (el) el.innerText = text;
}

function setHiddenIfPresent(id, hidden) {
  const el = qs(id);
  if (el) el.classList.toggle("hidden", hidden);
}

function setStatus(text, mode = "neutral") {
  const colors = {
    neutral: "bg-slate-100 text-slate-700 border-slate-200",
    ok: "bg-emerald-50 text-emerald-700 border-emerald-200",
    warn: "bg-amber-50 text-amber-700 border-amber-200",
    error: "bg-red-50 text-red-700 border-red-200",
  };
  qs("modelStatus").className = `rounded-lg border px-3 py-2 text-xs font-bold ${colors[mode]}`;
  qs("modelStatus").innerText = text;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function logLine(text) {
  const box = qs("logBox");
  const time = new Date().toLocaleTimeString();
  box.innerHTML += `<div><span class="text-slate-400">${escapeHtml(time)}</span> ${escapeHtml(text)}</div>`;
  box.scrollTop = box.scrollHeight;
}

function updateThresholdLabels() {
  const confidence = Number(qs("confidenceSlider").value).toFixed(2);
  const iou = Number(qs("iouSlider").value).toFixed(2);
  const mask = Number(qs("maskSlider").value).toFixed(2);
  const maxDetections = qs("maxDetections")?.value || "0";
  qs("confValue").innerText = confidence;
  qs("iouValue").innerText = iou;
  qs("maskValue").innerText = mask;
  setTextIfPresent("diagnosticConfidenceThreshold", confidence);
  setTextIfPresent("diagnosticIouThreshold", iou);
  setTextIfPresent("diagnosticMaskThreshold", mask);
  setTextIfPresent("diagnosticMaxDetections", maxDetections);
}

function formatThreshold(value) {
  return Number(value).toFixed(2);
}

function formatNumber(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

function emitDecoderDebug(message) {
  if (message === undefined || message === null) return;
  if (Array.isArray(message)) {
    message.forEach(emitDecoderDebug);
    return;
  }
  if (typeof message === "object") {
    try {
      logLine(`Debug: ${JSON.stringify(message)}`);
    } catch (err) {
      logLine(`Debug: ${String(message)}`);
    }
    return;
  }
  logLine(`Debug: ${message}`);
}

function readDecodedDetections(decoded, emitDebug = true) {
  if (Array.isArray(decoded)) return decoded;
  if (decoded && Array.isArray(decoded.detections)) {
    if (emitDebug) emitDecoderDebug(decoded.debug);
    return decoded.detections;
  }
  if (emitDebug) emitDecoderDebug(decoded?.debug);
  return [];
}

function getOutputSummary(outputs) {
  return Object.entries(outputs)
    .map(([name, tensor]) => `${name}[${tensor?.dims?.join("x") || "unknown"}]`)
    .join(", ");
}

function pickYoloOutputNames(outputs) {
  const entries = Object.entries(outputs || {});
  const pred = entries.find(([, tensor]) => {
    const dims = tensor?.dims || [];
    return dims.length === 3 && Math.min(Number(dims[1]), Number(dims[2])) >= 5;
  });
  const proto = entries.find(([name, tensor]) => {
    const dims = tensor?.dims || [];
    return name !== pred?.[0] && dims.length === 4;
  });
  return {
    predName: pred?.[0] || entries[0]?.[0] || null,
    protoName: proto?.[0] || entries.find(([name]) => name !== pred?.[0])?.[0] || null,
  };
}

function getDetectionId(det, index) {
  return det?.id ?? det?.detection_id ?? index + 1;
}

function getDetectionConfidence(det) {
  const confidence = Number(det?.confidence);
  return Number.isFinite(confidence) ? confidence : 0;
}

function getDetectionBox(det) {
  const rawBox = det?.box;
  let x1;
  let y1;
  let x2;
  let y2;
  let width;
  let height;
  let status;

  if (Array.isArray(rawBox)) {
    [x1, y1, x2, y2] = rawBox;
    width = x2 - x1;
    height = y2 - y1;
    status = det?.box_status;
  } else if (rawBox && typeof rawBox === "object") {
    x1 = Number(rawBox.x1);
    y1 = Number(rawBox.y1);
    x2 = Number(rawBox.x2);
    y2 = Number(rawBox.y2);
    width = Number(rawBox.width ?? x2 - x1);
    height = Number(rawBox.height ?? y2 - y1);
    status = rawBox.status ?? det?.box_status;
  } else {
    x1 = Number(det?.x1);
    y1 = Number(det?.y1);
    x2 = Number(det?.x2);
    y2 = Number(det?.y2);
    width = x2 - x1;
    height = y2 - y1;
    status = det?.box_status;
  }

  const available = [x1, y1, x2, y2, width, height].every(Number.isFinite) && width > 0 && height > 0;
  return {
    available,
    status: status || (available ? "available" : "missing"),
    x1: available ? x1 : 0,
    y1: available ? y1 : 0,
    x2: available ? x2 : 0,
    y2: available ? y2 : 0,
    width: available ? width : 0,
    height: available ? height : 0,
  };
}

function getDetectionCentroid(det, box) {
  const rawCentroid = det?.centroid;
  if (Array.isArray(rawCentroid)) {
    return { x: Number(rawCentroid[0]), y: Number(rawCentroid[1]) };
  }
  if (rawCentroid && typeof rawCentroid === "object") {
    return { x: Number(rawCentroid.x), y: Number(rawCentroid.y) };
  }
  return {
    x: box.x1 + box.width / 2,
    y: box.y1 + box.height / 2,
  };
}

function getDetectionMask(det) {
  const rawMask = det?.mask;
  const legacyArea = Number(det?.mask_area_pixels ?? 0);
  if (rawMask && typeof rawMask === "object") {
    const area = Number(rawMask.area_px ?? rawMask.area_pixels ?? rawMask.area ?? legacyArea);
    const polygonPoints = Array.isArray(rawMask.polygon) ? rawMask.polygon.length : 0;
    const available = rawMask.available !== undefined ? Boolean(rawMask.available) : polygonPoints > 0 || area > 0;
    return {
      available,
      status: rawMask.status || (available ? "available" : "not_available"),
      area_px: Number.isFinite(area) ? Math.round(area) : 0,
      polygon_points: polygonPoints,
    };
  }
  const available = Number.isFinite(legacyArea) && legacyArea > 0;
  return {
    available,
    status: available ? "available" : "not_provided",
    area_px: available ? Math.round(legacyArea) : 0,
    polygon_points: 0,
  };
}

function getCountSource(det) {
  return det?.count_source || "box";
}

function serialiseDetection(det, index) {
  const box = getDetectionBox(det);
  const centroid = getDetectionCentroid(det, box);
  const mask = getDetectionMask(det);
  return {
    id: getDetectionId(det, index),
    class_id: det.class_id ?? 0,
    class_name: det.class_name || "burrow",
    confidence: getDetectionConfidence(det),
    box: {
      x1: box.x1,
      y1: box.y1,
      x2: box.x2,
      y2: box.y2,
      width: box.width,
      height: box.height,
    },
    centroid: {
      x: centroid.x,
      y: centroid.y,
    },
    mask: {
      available: mask.available,
      polygon: Array.isArray(det?.mask?.polygon) ? det.mask.polygon : [],
      area_px: mask.area_px,
      status: mask.status,
    },
    count_source: getCountSource(det),
  };
}

function buildSessionRowsFromAnalysis(analysis) {
  const rows = analysis?.rows || [];
  if (!rows.length) {
    return [{
      session_image_index: analysis?.session_index ?? "",
      image_name: analysis?.name || "",
      id: "",
      class_id: "",
      class_name: "",
      confidence: "",
      x1: "",
      y1: "",
      x2: "",
      y2: "",
      width: "",
      height: "",
      centroid_x: "",
      centroid_y: "",
      mask_available: "",
      mask_status: "no_detections",
      mask_area_px: "",
      count_source: "no_detections",
      box_status: "",
      mask_polygon_points: "",
      threshold_confidence: analysis?.thresholds?.confidence ?? "",
      threshold_iou: analysis?.thresholds?.iou ?? "",
      threshold_mask: analysis?.thresholds?.mask ?? "",
      model_name: analysis?.model_name || metadata?.model_name || MODEL_URL,
      notes: "Image analysed with zero detections.",
    }];
  }
  return rows.map((row) => ({
    session_image_index: analysis?.session_index ?? "",
    image_name: analysis?.name || row.image_name || "",
    ...row,
  }));
}

function updateAnalysisNavigation() {
  const total = sessionAnalyses.length;
  const prevBtn = qs("prevAnalysedBtn");
  const nextBtn = qs("nextAnalysedBtn");
  const status = qs("analysisNavStatus");
  if (prevBtn) prevBtn.disabled = total <= 1 || currentAnalysisIndex <= 0;
  if (nextBtn) nextBtn.disabled = total <= 1 || currentAnalysisIndex < 0 || currentAnalysisIndex >= total - 1;
  if (status) {
    status.innerText = total > 0
      ? `Analysed image ${currentAnalysisIndex + 1} of ${total}: ${sessionAnalyses[currentAnalysisIndex]?.name || "Unnamed image"}`
      : "No analysed images in this session.";
  }
}

function drawStoredDetections() {
  drawBaseImage();
  const canvas = qs("viewerCanvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const display = {
    width: canvas.width,
    height: canvas.height,
    originalWidth: currentImage.naturalWidth,
    originalHeight: currentImage.naturalHeight,
  };
  const drawOptions = {
    maskThreshold: Number(qs("maskSlider").value),
    maskOpacity: 0.42,
    masksOptional: true,
  };
  if (typeof window.BurrowYoloSeg.drawDetectionMasks === "function") {
    try {
      window.BurrowYoloSeg.drawDetectionMasks(ctx, currentDetections, display, drawOptions);
      return;
    } catch (err) {
      logLine(`Stored overlay used boxes only: ${err.message}`);
    }
  }
  drawBoxesFallback(ctx, currentDetections, display);
}

function renderLoadedAnalysis() {
  qs("countValue").innerText = String(currentDetections.length);
  qs("exportCsvBtn").disabled = sessionAnalyses.length === 0;
  qs("exportJsonBtn").disabled = currentDetections.length === 0;
  qs("exportPngBtn").disabled = currentDetections.length === 0;
  updateDetectionDiagnostics(currentDetections);
  renderTable();
  updateAnalysisNavigation();
  drawStoredDetections();
}

function loadImageFromSource(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Could not reload analysed image from ${src}`));
    img.src = src;
  });
}

function storeCurrentAnalysis() {
  if (!currentImage) return;
  const rows = predictionRows();
  const analysis = {
    name: currentImageName,
    src: currentImage.currentSrc || currentImage.src,
    width: currentImage.naturalWidth,
    height: currentImage.naturalHeight,
    detections: currentDetections.map(serialiseDetection),
    rows,
    thresholds: {
      confidence: Number(qs("confidenceSlider").value),
      iou: Number(qs("iouSlider").value),
      mask: Number(qs("maskSlider").value),
    },
    model_name: metadata?.model_name || MODEL_URL,
  };
  const existingIndex = sessionAnalyses.findIndex((item) => item.src === analysis.src && item.name === analysis.name);
  if (existingIndex >= 0) {
    sessionAnalyses[existingIndex] = analysis;
    currentAnalysisIndex = existingIndex;
  } else {
    sessionAnalyses.push(analysis);
    currentAnalysisIndex = sessionAnalyses.length - 1;
  }
  sessionAnalyses.forEach((item, index) => {
    item.session_index = index + 1;
  });
  updateAnalysisNavigation();
}

async function showAnalysisAt(index) {
  if (index < 0 || index >= sessionAnalyses.length) return;
  const analysis = sessionAnalyses[index];
  const img = await loadImageFromSource(analysis.src);
  currentImage = img;
  currentImageName = analysis.name;
  currentDetections = analysis.detections.map((det) => ({ ...det, box: { ...det.box }, centroid: { ...det.centroid }, mask: { ...det.mask } }));
  currentAnalysisIndex = index;
  qs("imageName").innerText = analysis.name;
  qs("imageMeta").innerText = `${analysis.width} x ${analysis.height}px`;
  renderLoadedAnalysis();
}

function statusCounts(detections, reader) {
  const counts = new Map();
  detections.forEach((det) => {
    const status = reader(det).status || "unknown";
    counts.set(status, (counts.get(status) || 0) + 1);
  });
  return [...counts.entries()].map(([status, count]) => `${status}:${count}`).join(", ") || "none";
}

function resetDetectionDiagnostics() {
  setTextIfPresent("diagnosticDecodedBoxes", "Awaiting inference");
  setTextIfPresent("diagnosticPostNmsBoxes", "0");
  setTextIfPresent("diagnosticMaskAvailability", "Awaiting inference");
  setTextIfPresent(
    "maskStatusText",
    "Counts report decoded post-NMS boxes; mask availability can be reported separately.",
  );
  setHiddenIfPresent("maskUnavailableWarning", true);
}

function updateDetectionDiagnostics(detections) {
  const count = detections.length;
  const masksAvailable = detections.filter((det) => getDetectionMask(det).available).length;
  setTextIfPresent("diagnosticDecodedBoxes", `${count} returned by decoder`);
  setTextIfPresent("diagnosticPostNmsBoxes", String(count));
  setTextIfPresent(
    "diagnosticMaskAvailability",
    count > 0 ? `${masksAvailable}/${count} available (${statusCounts(detections, getDetectionMask)})` : "No detections",
  );
  setTextIfPresent(
    "maskStatusText",
    count > 0
      ? `Counts use ${count} post-NMS boxes. Masks available for ${masksAvailable}/${count} detections.`
      : "Counts report decoded post-NMS boxes; mask availability can be reported separately.",
  );
  setHiddenIfPresent("maskUnavailableWarning", count === 0 || masksAvailable === count);
}

async function loadMetadata() {
  try {
    const response = await fetch(METADATA_URL);
    metadata = await response.json();
    qs("modelName").innerText = metadata.model_name;
    qs("validationBadge").innerText = metadata.model_badge || "Local ONNX";
    qs("confidenceSlider").value = metadata.default_confidence_threshold ?? 0.2;
    qs("iouSlider").value = metadata.default_iou_threshold ?? 0.45;
    qs("maskSlider").value = metadata.default_mask_threshold ?? 0.5;
    updateThresholdLabels();
  } catch (err) {
    metadata = {};
    logLine(`Metadata not loaded: ${err.message}`);
  }
}

async function initSessionFromBuffer(buffer, sourceLabel) {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";
  ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
  setStatus("Loading ONNX model...", "neutral");
  session = await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"] });
  setStatus("ONNX ready", "ok");
  refreshUsageGate();
  logLine(`Model loaded from ${sourceLabel}.`);
  logLine(`Input: ${session.inputNames.join(", ")} | Outputs: ${session.outputNames.join(", ")}`);
}

async function autoLoadModel() {
  try {
    const response = await fetch(MODEL_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const buffer = await response.arrayBuffer();
    await initSessionFromBuffer(buffer, MODEL_URL);
  } catch (err) {
    setStatus("Auto-load failed. Upload ONNX manually.", "warn");
    qs("manualModelUpload").classList.remove("hidden");
    refreshUsageGate();
    logLine(`Auto-load failed: ${err.message}`);
  }
}

async function handleManualModel(event) {
  if (!termsAccepted) {
    openTermsModal();
    return;
  }
  const file = event.target.files[0];
  if (!file) return;
  const buffer = await file.arrayBuffer();
  await initSessionFromBuffer(buffer, file.name);
}

async function handleImageUpload(event) {
  if (!termsAccepted) {
    openTermsModal();
    return;
  }
  const file = event.target.files[0];
  if (!file) return;
  const { img } = await window.BurrowImageUtils.readImageFile(file);
  setCurrentImage(img, file.name);
}

function setCurrentImage(img, name) {
  currentImage = img;
  currentImageName = name;
  qs("imageName").innerText = name;
  qs("imageMeta").innerText = `${img.naturalWidth} x ${img.naturalHeight}px`;
  drawBaseImage();
  qs("runBtn").disabled = !session || batchRunning;
  qs("exportCsvBtn").disabled = true;
  qs("exportJsonBtn").disabled = true;
  qs("exportPngBtn").disabled = true;
  currentDetections = [];
  qs("countValue").innerText = "0";
  refreshUsageGate();
  resetDetectionDiagnostics();
  updateAnalysisNavigation();
  renderTable();
  logLine(`Image loaded: ${name}`);
}

async function loadSampleImages() {
  const grid = qs("sampleGrid");
  if (!grid) return;

  try {
    const response = await fetch("data/sample-images.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    sampleManifest = await response.json();
    renderSampleGrid();
  } catch (err) {
    grid.innerHTML = `<p class="col-span-2 text-sm text-slate-500">Example images not available.</p>`;
    logLine(`Sample images not loaded: ${err.message}`);
  }
}

function renderSampleGrid() {
  const grid = qs("sampleGrid");
  if (!grid) return;
  grid.innerHTML = "";

  for (const sample of sampleManifest) {
    const selected = selectedSampleIds.has(sample.id);
    const button = document.createElement("button");
    button.type = "button";
    button.className = `sample-card ${selected ? "selected" : ""}`;
    button.innerHTML = `
      <span class="sample-check">${selected ? "Selected" : "Select"}</span>
      <img src="${sample.thumb || sample.file}" alt="${sample.id}" loading="lazy" />
      <span class="sample-title">${sample.id.replace("_", " ")}</span>
    `;
    button.addEventListener("click", () => {
      toggleSampleSelection(sample.id);
    });
    grid.appendChild(button);
  }

  updateSampleSelectionUi();
}

function toggleSampleSelection(sampleId) {
  if (selectedSampleIds.has(sampleId)) {
    selectedSampleIds.delete(sampleId);
  } else {
    selectedSampleIds.add(sampleId);
  }
  renderSampleGrid();
}

function updateSampleSelectionUi() {
  const count = selectedSampleIds.size;
  const total = sampleManifest.length || 10;
  if (qs("selectedSampleCount")) qs("selectedSampleCount").innerText = String(count);
  if (qs("selectedSampleCountModal")) qs("selectedSampleCountModal").innerText = String(count);
  const enabled = count > 0 && Boolean(session) && !batchRunning && termsAccepted;
  if (qs("runSelectedSamplesBtn")) qs("runSelectedSamplesBtn").disabled = !enabled;
  if (qs("runSelectedSamplesModalBtn")) qs("runSelectedSamplesModalBtn").disabled = !enabled;
  const subtitle = qs("openSamplesBtn")?.querySelector("span span:nth-child(2)");
  if (subtitle) {
    subtitle.innerHTML = `<span id="selectedSampleCount">${count}</span> selected from ${total} example photos.`;
  }
}

function selectAllSamples() {
  selectedSampleIds = new Set(sampleManifest.map((sample) => sample.id));
  renderSampleGrid();
}

function clearSampleSelection() {
  selectedSampleIds.clear();
  renderSampleGrid();
}

function openSampleModal() {
  if (!termsAccepted) {
    openTermsModal();
    return;
  }
  const modal = qs("sampleModal");
  if (!modal) return;
  modal.classList.remove("hidden");
  modal.classList.add("flex");
}

function closeSampleModal() {
  const modal = qs("sampleModal");
  if (!modal) return;
  modal.classList.add("hidden");
  modal.classList.remove("flex");
}

async function runModelOnCurrentImage() {
  const prepared = window.BurrowImageUtils.letterboxImageToTensor(currentImage, INPUT_SIZE);
  lastLetterbox = prepared.info;
  if (!batchRunning) {
    logLine(
      `Input image: ${currentImageName} ${currentImage.naturalWidth}x${currentImage.naturalHeight}px; ONNX tensor 1x3x${INPUT_SIZE}x${INPUT_SIZE}; letterbox scale ${formatNumber(lastLetterbox.scale, 4)}, pad ${lastLetterbox.padX},${lastLetterbox.padY}.`,
    );
  }
  const tensor = new ort.Tensor("float32", prepared.tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const feeds = {};
  feeds[session.inputNames[0]] = tensor;
  lastOutputs = await session.run(feeds);
  decodeAndRender({ diagnostics: !batchRunning });
  return currentDetections;
}

function renderBatchResults() {
  const panel = qs("batchResultsPanel");
  const body = qs("batchResultsBody");
  if (!panel || !body) return;

  panel.classList.toggle("hidden", batchResults.length === 0);
  const total = batchResults.reduce((sum, row) => sum + row.count, 0);
  qs("batchTotalCount").innerText = String(total);
  body.innerHTML = "";

  for (const row of batchResults) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="px-3 py-2">
        <span class="block font-bold text-slate-800">${row.id.replace("_", " ")}</span>
        <span class="block max-w-[220px] truncate text-[10px] text-slate-500">${row.name}</span>
      </td>
      <td class="px-3 py-2 text-right font-extrabold text-blue-700">${row.count}</td>
    `;
    body.appendChild(tr);
  }
}

function setBatchControlsDisabled(disabled) {
  batchRunning = disabled;
  qs("runBtn").disabled = disabled || !session || !currentImage || !termsAccepted;
  if (qs("runSelectedSamplesBtn")) qs("runSelectedSamplesBtn").disabled = disabled || selectedSampleIds.size === 0 || !session || !termsAccepted;
  if (qs("runSelectedSamplesModalBtn")) qs("runSelectedSamplesModalBtn").disabled = disabled || selectedSampleIds.size === 0 || !session || !termsAccepted;
  if (qs("selectAllSamplesBtn")) qs("selectAllSamplesBtn").disabled = disabled;
  if (qs("clearSamplesBtn")) qs("clearSamplesBtn").disabled = disabled;
}

async function runSelectedSamples() {
  if (!termsAccepted) {
    openTermsModal();
    return;
  }
  if (!session) {
    setStatus("Load ONNX first.", "warn");
    return;
  }

  const selected = sampleManifest.filter((sample) => selectedSampleIds.has(sample.id));
  if (!selected.length) return;

  closeSampleModal();
  batchResults = [];
  renderBatchResults();
  setBatchControlsDisabled(true);
  setStatus(`Running batch 1/${selected.length}...`, "neutral");
  logLine(`Batch started with ${selected.length} example images.`);

  try {
    for (let index = 0; index < selected.length; index += 1) {
      const sample = selected[index];
      setStatus(`Running batch ${index + 1}/${selected.length}...`, "neutral");
      const { img } = await window.BurrowImageUtils.readImageUrl(sample.file);
      setCurrentImage(img, sample.original_name || sample.id);
      await new Promise((resolve) => requestAnimationFrame(resolve));
      const detections = await runModelOnCurrentImage();
      batchResults.push({
        id: sample.id,
        name: sample.original_name || sample.id,
        count: detections.length,
      });
      renderBatchResults();
      await new Promise((resolve) => setTimeout(resolve, 80));
    }

    const total = batchResults.reduce((sum, row) => sum + row.count, 0);
    setStatus(`Batch complete: ${total} burrows in ${batchResults.length} images.`, "ok");
    logLine(`Batch complete: ${total} burrows in ${batchResults.length} images.`);
  } catch (err) {
    setStatus("Batch failed", "error");
    logLine(`Batch error: ${err.message}`);
    console.error(err);
  } finally {
    setBatchControlsDisabled(false);
    updateSampleSelectionUi();
  }
}

function getDisplaySize(img) {
  const scale = Math.min(MAX_DISPLAY_WIDTH / img.naturalWidth, MAX_DISPLAY_HEIGHT / img.naturalHeight, 1);
  return {
    width: Math.round(img.naturalWidth * scale),
    height: Math.round(img.naturalHeight * scale),
    scale,
    originalWidth: img.naturalWidth,
    originalHeight: img.naturalHeight,
  };
}

function drawBaseImage() {
  if (!currentImage) return;
  const canvas = qs("viewerCanvas");
  const display = getDisplaySize(currentImage);
  canvas.width = display.width;
  canvas.height = display.height;
  canvas.style.backgroundImage = `url("${String(currentImage.src || "").replaceAll('"', '\\"')}")`;
  canvas.style.backgroundPosition = "center";
  canvas.style.backgroundRepeat = "no-repeat";
  canvas.style.backgroundSize = "100% 100%";
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
}

function displayBox(box, display) {
  return {
    x1: (box.x1 / display.originalWidth) * display.width,
    y1: (box.y1 / display.originalHeight) * display.height,
    x2: (box.x2 / display.originalWidth) * display.width,
    y2: (box.y2 / display.originalHeight) * display.height,
  };
}

function drawBoxesFallback(ctx, detections, display) {
  const colors = ["#14b8a6", "#f59e0b", "#3b82f6", "#ec4899", "#84cc16"];
  ctx.save();
  ctx.lineWidth = 2;
  ctx.font = "12px sans-serif";

  detections.forEach((det, index) => {
    const box = getDetectionBox(det);
    if (!box.available) return;
    const scaled = displayBox(box, display);
    const width = Math.max(0, scaled.x2 - scaled.x1);
    const height = Math.max(0, scaled.y2 - scaled.y1);
    const color = colors[index % colors.length];
    ctx.strokeStyle = color;
    ctx.strokeRect(scaled.x1, scaled.y1, width, height);
    ctx.fillStyle = "rgba(15, 23, 42, 0.8)";
    ctx.fillRect(scaled.x1, Math.max(0, scaled.y1 - 20), 74, 20);
    ctx.fillStyle = "white";
    ctx.fillText(`${getDetectionId(det, index)}: ${getDetectionConfidence(det).toFixed(2)}`, scaled.x1 + 4, Math.max(13, scaled.y1 - 6));
  });

  ctx.restore();
}

async function runInference() {
  if (!termsAccepted) {
    openTermsModal();
    return;
  }
  if (!session || !currentImage) return;
  qs("runBtn").disabled = true;
  qs("runBtnText").innerText = "Running...";
  setStatus("Running inference locally...", "neutral");

  try {
    await runModelOnCurrentImage();
    setStatus("Inference complete", "ok");
  } catch (err) {
    setStatus("Inference failed", "error");
    logLine(`Inference error: ${err.message}`);
    console.error(err);
  } finally {
    qs("runBtn").disabled = !termsAccepted || !session || !currentImage;
    qs("runBtnText").innerText = "Run Burrow Detection";
  }
}

function buildDecodeOptions(confThreshold, { debug = false } = {}) {
  const outputNames = pickYoloOutputNames(lastOutputs || {});
  const iouThreshold = Number(qs("iouSlider").value);
  const maskThreshold = Number(qs("maskSlider").value);
  const options = {
    predName: outputNames.predName,
    protoName: outputNames.protoName,
    confThreshold,
    iouThreshold,
    maskThreshold,
    maskOpacity: 0.42,
    masksOptional: true,
    letterbox: lastLetterbox,
    maxDetections: Number(qs("maxDetections").value),
    imageSize: currentImage ? {
      width: currentImage.naturalWidth,
      height: currentImage.naturalHeight,
    } : null,
  };

  if (debug) {
    options.debug = true;
    options.onDebug = emitDecoderDebug;
    options.debugLogger = emitDecoderDebug;
  }

  return options;
}

function logPostProcessSummary(detections, options) {
  logLine(`Model outputs: ${getOutputSummary(lastOutputs)}.`);
  logLine(`Selected YOLO outputs: predictions=${options.predName || "auto"}, prototypes=${options.protoName || "none"}.`);
  logLine(
    `Decode thresholds: confidence ${formatThreshold(options.confThreshold)}, IoU ${formatThreshold(options.iouThreshold)}, mask ${formatThreshold(options.maskThreshold)}, max ${options.maxDetections}.`,
  );
  logLine(`Final post-NMS boxes: ${detections.length}; count source is detections.length.`);
  logLine(`Box status: ${statusCounts(detections, getDetectionBox)} | Mask status: ${statusCounts(detections, getDetectionMask)}.`);
  const debug = window.BurrowYoloSeg?.lastDebug;
  if (debug) {
    logLine(
      `Postprocess debug: raw above conf=${debug.raw_candidates_above_confidence ?? "n/a"}, after NMS=${debug.after_nms ?? detections.length}, valid masks=${debug.valid_masks ?? "n/a"}, failed masks=${debug.failed_masks ?? "n/a"}.`,
    );
    logLine(`First 5 boxes before reverse letterbox: ${JSON.stringify(debug.first_5_boxes_before_reverse_letterbox || [])}`);
    logLine(`First 5 boxes after reverse letterbox: ${JSON.stringify(debug.first_5_boxes_after_reverse_letterbox || [])}`);
    logLine(`First 5 confidences: ${JSON.stringify(debug.first_5_confidences || [])}`);
    logLine(`First 5 mask statuses: ${JSON.stringify(debug.first_5_mask_statuses || [])}`);
  }
}

function logThresholdDiagnostics() {
  if (!lastOutputs || !lastLetterbox) return;
  const baseOptions = buildDecodeOptions(Number(qs("confidenceSlider").value));
  lastThresholdDiagnostics = typeof window.BurrowYoloSeg.thresholdDiagnostics === "function"
    ? window.BurrowYoloSeg.thresholdDiagnostics(lastOutputs, baseOptions, THRESHOLD_DIAGNOSTIC_CONFIDENCES)
    : [];
  const parts = lastThresholdDiagnostics.map((row) => (
    row.error
      ? `${formatThreshold(row.confidence_threshold)}:error`
      : `${formatThreshold(row.confidence_threshold)}:${row.post_nms_boxes}`
  ));

  logLine(`Threshold diagnostics (confidence -> post-NMS boxes): ${parts.join(" | ")}`);
}

function decodeAndRender({ diagnostics = false } = {}) {
  if (!lastOutputs || !lastLetterbox) return;
  updateThresholdLabels();
  const conf = Number(qs("confidenceSlider").value);
  const maskThreshold = Number(qs("maskSlider").value);
  const decodeOptions = buildDecodeOptions(conf, { debug: diagnostics });

  currentDetections = readDecodedDetections(window.BurrowYoloSeg.decodeYoloSeg(lastOutputs, decodeOptions));

  drawBaseImage();
  const canvas = qs("viewerCanvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const display = {
    width: canvas.width,
    height: canvas.height,
    originalWidth: currentImage.naturalWidth,
    originalHeight: currentImage.naturalHeight,
  };
  const drawOptions = {
    maskThreshold,
    maskOpacity: 0.42,
    masksOptional: true,
  };

  if (typeof window.BurrowYoloSeg.drawDetectionMasks === "function") {
    try {
      window.BurrowYoloSeg.drawDetectionMasks(ctx, currentDetections, display, drawOptions);
    } catch (err) {
      logLine(`Overlay draw skipped masks and used boxes only: ${err.message}`);
      drawBoxesFallback(ctx, currentDetections, display);
    }
  } else {
    logLine("Overlay draw function not available; using boxes only.");
    drawBoxesFallback(ctx, currentDetections, display);
  }

  const count = currentDetections.length;
  qs("countValue").innerText = String(count);
  qs("exportCsvBtn").disabled = count === 0 && sessionAnalyses.length === 0;
  qs("exportJsonBtn").disabled = count === 0;
  qs("exportPngBtn").disabled = count === 0;
  updateDetectionDiagnostics(currentDetections);
  renderTable();
  storeCurrentAnalysis();
  updateAnalysisNavigation();
  if (diagnostics) {
    logPostProcessSummary(currentDetections, decodeOptions);
    logThresholdDiagnostics();
  } else {
    logLine(`${count} burrow predictions after thresholds.`);
  }
}

function renderTable() {
  const body = qs("detectionsBody");
  body.innerHTML = "";
  for (const [index, det] of currentDetections.entries()) {
    const box = getDetectionBox(det);
    const centroid = getDetectionCentroid(det, box);
    const mask = getDetectionMask(det);
    const maskChipClass = mask.available ? "ok" : (mask.status === "empty" ? "warn" : "missing");
    const tr = document.createElement("tr");
    tr.className = "border-b border-slate-100";
    tr.innerHTML = `
      <td class="px-3 py-2 font-bold">${getDetectionId(det, index)}</td>
      <td class="px-3 py-2">${getDetectionConfidence(det).toFixed(3)}</td>
      <td class="px-3 py-2">${formatNumber(centroid.x, 0)}, ${formatNumber(centroid.y, 0)}</td>
      <td class="px-3 py-2">${formatNumber(box.x1, 0)}, ${formatNumber(box.y1, 0)}, ${formatNumber(box.x2, 0)}, ${formatNumber(box.y2, 0)}</td>
      <td class="px-3 py-2" title="${escapeHtml(mask.status)}">
        <span class="mono">${mask.area_px}</span>
        <span class="mask-status-chip ${maskChipClass}">${escapeHtml(mask.status)}</span>
      </td>
    `;
    body.appendChild(tr);
  }
  if (!currentDetections.length) {
    body.innerHTML = `<tr><td colspan="5" class="px-3 py-8 text-center text-slate-400">No predictions yet.</td></tr>`;
  }
}

function predictionRows() {
  const conf = Number(qs("confidenceSlider").value);
  const iou = Number(qs("iouSlider").value);
  const maskThreshold = Number(qs("maskSlider").value);
  return currentDetections.map((det, index) => {
    const box = getDetectionBox(det);
    const centroid = getDetectionCentroid(det, box);
    const mask = getDetectionMask(det);
    return {
      image_name: currentImageName,
      id: getDetectionId(det, index),
      class_id: det.class_id,
      class_name: det.class_name,
      confidence: getDetectionConfidence(det),
      x1: box.x1,
      y1: box.y1,
      x2: box.x2,
      y2: box.y2,
      width: box.width,
      height: box.height,
      centroid_x: centroid.x,
      centroid_y: centroid.y,
      mask_available: mask.available,
      mask_status: mask.status,
      mask_area_px: mask.area_px,
      count_source: getCountSource(det),
      box_status: box.status,
      mask_polygon_points: mask.polygon_points,
      threshold_confidence: conf,
      threshold_iou: iou,
      threshold_mask: maskThreshold,
      model_name: metadata?.model_name || MODEL_URL,
      notes: "",
    };
  });
}

function computeMean(values) {
  if (!values.length) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function computeStandardDeviation(values) {
  if (values.length <= 1) return 0;
  const mean = computeMean(values);
  const variance = values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / (values.length - 1);
  return Math.sqrt(Math.max(variance, 0));
}

function buildSessionSummaryRows() {
  const confidenceThreshold = Number(qs("confidenceSlider").value);
  return sessionAnalyses.map((analysis) => {
    const detections = analysis?.detections || [];
    const confidences = detections
      .map((det) => Number(det?.confidence))
      .filter((value) => Number.isFinite(value));
    const burrowCount = detections.length;
    const meanConfidence = computeMean(confidences);
    const confidenceSd = computeStandardDeviation(confidences);
    const lowConfidenceFraction = burrowCount
      ? confidences.filter((value) => value < Math.max(0.35, confidenceThreshold + 0.1)).length / burrowCount
      : 0;

    // Heuristic uncertainty proxy, not a calibrated predictive interval.
    const relativeUncertainty = burrowCount
      ? Math.min(0.95, 0.08 + (0.55 * lowConfidenceFraction) + (0.35 * confidenceSd / Math.max(meanConfidence, 0.15)))
      : 0;
    const estimatedCountSd = burrowCount * relativeUncertainty;

    return {
      session_image_index: analysis?.session_index ?? "",
      image_name: analysis?.name || "",
      burrow_count: burrowCount,
      mean_confidence: Number(meanConfidence.toFixed(4)),
      confidence_sd: Number(confidenceSd.toFixed(4)),
      low_confidence_fraction: Number(lowConfidenceFraction.toFixed(4)),
      estimated_count_sd: Number(estimatedCountSd.toFixed(2)),
      uncertainty_note: burrowCount
        ? "Heuristic estimate from confidence spread and low-confidence detections; not calibrated ground-truth error."
        : "No detections in this analysed image.",
    };
  });
}

function exportCsv() {
  const rows = sessionAnalyses.length
    ? sessionAnalyses.flatMap(buildSessionRowsFromAnalysis)
    : predictionRows();
  if (!rows.length) return;
  const headers = Object.keys(rows[0]);
  const lines = [
    headers.join(","),
    ...rows.map((row) => headers.map((h) => window.BurrowImageUtils.csvEscape(row[h])).join(",")),
  ];
  window.BurrowImageUtils.downloadTextFile(
    `burrowscope_session_predictions.csv`,
    lines.join("\n"),
    "text/csv",
  );

  const summaryRows = sessionAnalyses.length
    ? buildSessionSummaryRows()
    : [{
      session_image_index: 1,
      image_name: currentImageName || "",
      burrow_count: currentDetections.length,
      mean_confidence: Number(computeMean(currentDetections.map((det) => getDetectionConfidence(det))).toFixed(4)),
      confidence_sd: Number(computeStandardDeviation(currentDetections.map((det) => getDetectionConfidence(det))).toFixed(4)),
      low_confidence_fraction: currentDetections.length
        ? Number((currentDetections.filter((det) => getDetectionConfidence(det) < Math.max(0.35, Number(qs("confidenceSlider").value) + 0.1)).length / currentDetections.length).toFixed(4))
        : 0,
      estimated_count_sd: Number((currentDetections.length * 0.08).toFixed(2)),
      uncertainty_note: currentDetections.length
        ? "Heuristic estimate from confidence spread and low-confidence detections; not calibrated ground-truth error."
        : "No detections in this analysed image.",
    }];
  const summaryHeaders = Object.keys(summaryRows[0]);
  const summaryLines = [
    summaryHeaders.join(","),
    ...summaryRows.map((row) => summaryHeaders.map((h) => window.BurrowImageUtils.csvEscape(row[h])).join(",")),
  ];
  window.BurrowImageUtils.downloadTextFile(
    `burrowscope_session_image_summary.csv`,
    summaryLines.join("\n"),
    "text/csv",
  );

  logLine(`Detailed CSV and image-summary CSV exported for ${sessionAnalyses.length || 1} analysed image(s).`);
}

function exportJson() {
  const payload = {
    app_name: "BurrowScope",
    model_name: metadata?.model_name || MODEL_URL,
    created_at: new Date().toISOString(),
    confidence_threshold: Number(qs("confidenceSlider").value),
    iou_threshold: Number(qs("iouSlider").value),
    mask_threshold: Number(qs("maskSlider").value),
    threshold_diagnostics: lastThresholdDiagnostics,
    scope_note: "Per-image burrow segmentation only. Duplicate counting across overlapping photos is not solved here.",
    images: [
      {
        image_name: currentImageName,
        width: currentImage?.naturalWidth,
        height: currentImage?.naturalHeight,
        n_detections: currentDetections.length,
        count_source: "detections.length post-NMS boxes",
        detections: currentDetections.map(serialiseDetection),
      },
    ],
  };
  window.BurrowImageUtils.downloadTextFile(
    `${currentImageName || "burrows"}_burrowscope_predictions.json`,
    JSON.stringify(payload, null, 2),
    "application/json",
  );
}

function exportPng() {
  const canvas = qs("viewerCanvas");
  const a = document.createElement("a");
  a.href = canvas.toDataURL("image/png");
  a.download = `${currentImageName || "burrows"}_overlay.png`;
  a.click();
}

const PDF_LOGOS = [
  { src: "img/bankLogo1.png", width: 22, height: 22 },
  { src: "img/birds-canada-logo.svg", width: 30, height: 16 },
  { src: "img/Naturecounts-logo.svg", width: 36, height: 14 },
  { src: "img/animals-on-the-move-logo.svg", width: 34, height: 16 },
  { src: "img/western-logo.svg", width: 36, height: 14 },
  { src: "img/Mitacs_logo_blue.webp", width: 30, height: 10 },
];

function getInstructionSettings() {
  return {
    modelName: qs("modelName")?.innerText?.trim() || "bank_swallow_burrow_yolo11s_seg_production.onnx",
    modelStatus: qs("modelStatus")?.innerText?.trim() || "ONNX ready",
    confidence: Number(qs("confidenceSlider")?.value ?? 0.2).toFixed(2),
    iou: Number(qs("iouSlider")?.value ?? 0.5).toFixed(2),
    mask: Number(qs("maskSlider")?.value ?? 0.5).toFixed(2),
    maxDetections: qs("maxDetections")?.value || "500",
  };
}

function buildInstructionSections(settings) {
  return [
    {
      language: "English",
      title: "BurrowScope Instructions",
      subtitle: "Technical guide for local Bank Swallow burrow inspection",
      blocks: [
        {
          heading: "Purpose",
          text: [
            "BurrowScope is a local browser application for inspecting Bank Swallow burrows in bank or cliff photographs. It uses a YOLO segmentation model exported to ONNX. The practical output is per-image detection: post-NMS boxes, mask overlays, centroids, confidence values, and export files.",
            "The app is intended for non-profit research, conservation, field checking, and model-assisted review. It is not a replacement for biological judgement; it is a fast inspection layer that should be reviewed by a person before reporting.",
          ],
        },
        {
          heading: "Current inference configuration",
          text: [
            `Detection engine: Local ONNX. Model file: ${settings.modelName}. Current status shown by the app: ${settings.modelStatus}.`,
            "The model runs in the browser through ONNX Runtime Web. Images and inference results stay on the local machine unless the user exports and shares them manually.",
            `Confidence threshold: ${settings.confidence}. IoU NMS threshold: ${settings.iou}. Mask threshold: ${settings.mask}. Maximum detections: ${settings.maxDetections}.`,
            "If there is a newer inference model, use the Load Local .onnx Model File control in Model settings and then run the same image again to compare behaviour.",
          ],
        },
        {
          heading: "How to run an image or a small batch",
          text: [
            "1. Accept the local-use conditions when the application opens. This only unlocks the local controls.",
            "2. Use the example gallery for a quick test, or select/import a JPG, PNG, or WebP bank-wall image from your computer.",
            "3. Press Run Burrow Detection. The count badge reports final post-NMS boxes. NMS means non-maximum suppression: duplicate-like overlapping boxes are filtered inside the same image. Mask status is reported separately, so a failed mask should not erase a valid box detection.",
            "4. Inspect the overlay visually. Check whether boxes cover real burrow openings, whether centroids are inside those boxes, and whether obvious false positives or missed burrows appear.",
            "5. Export CSV for tabular review, JSON for full detection objects, or PNG for a visual record. The Instructions button exports this document.",
          ],
        },
        {
          heading: "Reading the results",
          text: [
            "Confidence controls how strict the detector is. A lower confidence threshold usually increases recall but may add false positives. A higher threshold usually reduces false positives but may miss faint or partly occluded burrows.",
            "IoU NMS controls duplicate suppression between overlapping boxes. If the same burrow is counted twice, a lower or moderate NMS setting can help. If nearby burrows are merged or suppressed, inspect the image carefully before changing the threshold.",
            "Mask threshold controls the binary mask extracted from the segmentation prototype. The object count is box-centred, because boxes are the most stable ONNX decoding signal. Masks are visual and spatial evidence, but they should not be allowed to delete detections.",
          ],
        },
        {
          heading: "Export schema",
          text: [
            "The CSV export is one row per detection and includes: id, confidence, x1, y1, x2, y2, width, height, centroid_x, centroid_y, mask_available, mask_status, mask_area_px, and count_source.",
            "The JSON export keeps the full detection objects, including box coordinates, centroid, confidence, class name, mask status, mask area, and the same count_source used by the visual count.",
          ],
        },
        {
          heading: "Important scope note",
          text: [
            "BurrowScope solves per-image burrow segmentation and inspection. It does not solve duplicate counting across overlapping photographs. The later spatial pipeline is: YOLO-seg predictions -> mask centroids and boxes -> image registration between overlapping photos -> shared coordinate system -> duplicate merging -> unique burrow count.",
          ],
        },
      ],
    },
    {
      language: "Français québécois",
      title: "Instructions BurrowScope",
      subtitle: "Guide technique pour l'inspection locale des terriers d'hirondelles de rivage",
      blocks: [
        {
          heading: "Objectif",
          text: [
            "BurrowScope est une application locale dans le navigateur pour inspecter les terriers d'hirondelles de rivage dans des photos de berges ou de falaises. Elle utilise un modèle YOLO de segmentation exporté en ONNX. Le résultat pratique est par image: boîtes post-NMS, masques superposés, centroïdes, valeurs de confiance et fichiers d'export.",
            "L'application est pensée pour la recherche sans but lucratif, la conservation, la vérification de terrain et la révision assistée par modèle. Ce n'est pas un remplacement du jugement biologique; c'est une couche rapide d'inspection qui doit être révisée par une personne avant un rapport.",
          ],
        },
        {
          heading: "Configuration actuelle d'inférence",
          text: [
            `Moteur de détection: ONNX local. Fichier modèle: ${settings.modelName}. État affiché par l'application: ${settings.modelStatus}.`,
            "Le modèle fonctionne dans le navigateur avec ONNX Runtime Web. Les images et les résultats d'inférence restent sur l'ordinateur local, sauf si l'utilisateur les exporte et les partage lui-même.",
            `Seuil de confiance: ${settings.confidence}. Seuil IoU NMS: ${settings.iou}. Seuil de masque: ${settings.mask}. Nombre maximal de détections: ${settings.maxDetections}.`,
            "S'il existe une version plus récente du modèle d'inférence, utilisez Load Local .onnx Model File dans Model settings, puis relancez la même image pour comparer le comportement.",
          ],
        },
        {
          heading: "Comment traiter une image ou un petit lot",
          text: [
            "1. Acceptez les conditions d'utilisation locale à l'ouverture de l'application. Cela déverrouille seulement les contrôles locaux.",
            "2. Utilisez la galerie d'exemples pour un test rapide, ou sélectionnez/importez une image JPG, PNG ou WebP d'une berge depuis votre ordinateur.",
            "3. Cliquez sur Run Burrow Detection. Le compteur affiche les boîtes finales post-NMS. NMS veut dire suppression non maximale: les boîtes qui se chevauchent comme des doublons sont filtrées dans la même image. L'état des masques est rapporté séparément; un masque qui échoue ne devrait pas effacer une boîte valide.",
            "4. Inspectez visuellement la superposition. Vérifiez si les boîtes couvrent de vrais trous de terriers, si les centroïdes sont dans ces boîtes et si des faux positifs ou des terriers manqués apparaissent.",
            "5. Exportez un CSV pour la révision tabulaire, un JSON pour les objets de détection complets, ou un PNG pour une trace visuelle. Le bouton Instructions exporte ce document.",
          ],
        },
        {
          heading: "Interpréter les résultats",
          text: [
            "Le seuil de confiance contrôle la sévérité du détecteur. Un seuil plus bas augmente souvent le rappel, mais peut ajouter des faux positifs. Un seuil plus élevé réduit souvent les faux positifs, mais peut manquer des terriers faibles ou partiellement cachés.",
            "Le seuil IoU NMS contrôle la suppression des doublons entre boîtes qui se chevauchent. Si le même terrier est compté deux fois, un réglage plus bas ou modéré peut aider. Si des terriers proches sont supprimés, il faut inspecter l'image avant de changer le seuil.",
            "Le seuil de masque contrôle le masque binaire extrait du prototype de segmentation. Le compte d'objets est centré sur les boîtes, parce que les boîtes sont le signal ONNX le plus stable. Les masques donnent une preuve visuelle et spatiale, mais ils ne doivent pas supprimer les détections.",
          ],
        },
        {
          heading: "Schéma d'export",
          text: [
            "L'export CSV contient une ligne par détection avec: id, confidence, x1, y1, x2, y2, width, height, centroid_x, centroid_y, mask_available, mask_status, mask_area_px et count_source.",
            "L'export JSON garde les objets de détection complets, incluant les coordonnées de boîte, le centroïde, la confiance, le nom de classe, l'état du masque, l'aire du masque et le même count_source utilisé par le compteur visuel.",
          ],
        },
        {
          heading: "Limite importante",
          text: [
            "BurrowScope règle la segmentation et l'inspection par image. Il ne règle pas encore le comptage des doublons dans des photos qui se chevauchent. Le pipeline spatial suivant est: prédictions YOLO-seg -> centroïdes et boîtes des masques -> recalage entre photos chevauchantes -> système de coordonnées commun -> fusion des doublons -> nombre unique de terriers.",
          ],
        },
      ],
    },
    {
      language: "Español latinoamericano",
      title: "Instrucciones de BurrowScope",
      subtitle: "Guía técnica para inspección local de madrigueras de golondrina ribereña",
      blocks: [
        {
          heading: "Para qué sirve",
          text: [
            "BurrowScope es una aplicación local en el navegador para revisar madrigueras de golondrina ribereña en fotos de barrancos, bancos o paredes de tierra. Usa un modelo YOLO de segmentación exportado a ONNX. El resultado práctico es por imagen: cajas post-NMS, máscaras, centroides, valores de confianza y archivos exportables.",
            "La app está pensada para investigación sin fines de lucro, conservación, revisión de campo y apoyo al etiquetado o inspección. No reemplaza el criterio biológico; ayuda a revisar rápido, pero los resultados deben ser vistos por una persona antes de usarlos en reportes.",
          ],
        },
        {
          heading: "Configuración actual de inferencia",
          text: [
            `Motor de detección: ONNX local. Archivo del modelo: ${settings.modelName}. Estado mostrado por la app: ${settings.modelStatus}.`,
            "El modelo corre en el navegador usando ONNX Runtime Web. Las imágenes y los resultados de inferencia se quedan en la máquina local, a menos que el usuario los exporte y los comparta manualmente.",
            `Umbral de confianza: ${settings.confidence}. Umbral IoU NMS: ${settings.iou}. Umbral de máscara: ${settings.mask}. Máximo de detecciones: ${settings.maxDetections}.`,
            "Si existe una versión más nueva del modelo de inferencia, use Load Local .onnx Model File en Model settings y vuelva a correr la misma imagen para comparar el comportamiento.",
          ],
        },
        {
          heading: "Cómo correr una imagen o un lote pequeño",
          text: [
            "1. Acepte las condiciones de uso local cuando se abre la aplicación. Esto solo habilita los controles locales.",
            "2. Use la galería de ejemplos para una prueba rápida, o seleccione/importe una imagen JPG, PNG o WebP desde su computador.",
            "3. Presione Run Burrow Detection. El contador muestra las cajas finales post-NMS. NMS significa supresión no máxima: cajas muy traslapadas, parecidas a duplicados, se filtran dentro de la misma imagen. El estado de las máscaras se reporta aparte, así que una máscara fallida no debe borrar una detección válida por caja.",
            "4. Revise visualmente el overlay. Confirme que las cajas estén sobre aberturas reales, que los centroides queden dentro de las cajas y que no haya patrones obvios de falsos positivos o burrows faltantes.",
            "5. Exporte CSV para revisar tablas, JSON para guardar los objetos completos de detección, o PNG para una evidencia visual. El botón Instructions exporta este documento.",
          ],
        },
        {
          heading: "Cómo interpretar los resultados",
          text: [
            "El umbral de confianza controla qué tan estricto es el detector. Un valor más bajo normalmente aumenta el recall, pero puede meter falsos positivos. Un valor más alto reduce ruido, pero puede perder madrigueras pequeñas, borrosas u ocultas.",
            "IoU NMS controla la supresión de cajas duplicadas que se traslapan. Si una misma madriguera aparece contada dos veces, un ajuste más bajo o moderado puede ayudar. Si madrigueras cercanas se eliminan, revise la imagen antes de cambiar el valor.",
            "El umbral de máscara controla la máscara binaria que sale del prototipo de segmentación. El conteo está centrado en cajas porque las cajas son la señal ONNX más estable. Las máscaras ayudan visual y espacialmente, pero no deben eliminar detecciones válidas.",
          ],
        },
        {
          heading: "Esquema de exportación",
          text: [
            "El CSV exportado tiene una fila por detección e incluye: id, confidence, x1, y1, x2, y2, width, height, centroid_x, centroid_y, mask_available, mask_status, mask_area_px y count_source.",
            "El JSON conserva los objetos completos de detección, incluyendo coordenadas de caja, centroide, confianza, nombre de clase, estado de máscara, área de máscara y el mismo count_source usado por el conteo visual.",
          ],
        },
        {
          heading: "Alcance importante",
          text: [
            "BurrowScope resuelve segmentación e inspección por imagen. Todavía no resuelve el conteo único entre fotos traslapadas. Ese pipeline espacial sería: predicciones YOLO-seg -> centroides y cajas de máscaras -> registro entre fotos traslapadas -> sistema de coordenadas compartido -> fusión de duplicados -> conteo único de madrigueras.",
          ],
        },
      ],
    },
  ];
}

function loadPdfLogo(asset) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        const scale = Math.min(4, Math.max(1, 360 / Math.max(img.naturalWidth || 1, img.naturalHeight || 1)));
        canvas.width = Math.max(1, Math.round((img.naturalWidth || 120) * scale));
        canvas.height = Math.max(1, Math.round((img.naturalHeight || 80) * scale));
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve({ ...asset, dataUrl: canvas.toDataURL("image/png") });
      } catch (err) {
        resolve(null);
      }
    };
    img.onerror = () => resolve(null);
    img.src = asset.src;
  });
}

async function loadPdfLogos() {
  const logos = await Promise.all(PDF_LOGOS.map(loadPdfLogo));
  return logos.filter(Boolean);
}

function addPdfHeader(doc, logos, language, pageWidth, margin) {
  let x = margin;
  const y = 10;
  logos.forEach((logo) => {
    try {
      doc.addImage(logo.dataUrl, "PNG", x, y, logo.width, logo.height, undefined, "FAST");
      x += logo.width + 4;
    } catch (err) {
      x += logo.width + 4;
    }
  });
  doc.setFont("helvetica", "bold");
  doc.setFontSize(8);
  doc.setTextColor(37, 99, 235);
  doc.text(language, pageWidth - margin, 14, { align: "right" });
  doc.setDrawColor(219, 227, 234);
  doc.line(margin, 35, pageWidth - margin, 35);
}

function addWrappedPdfText(doc, text, x, y, width, lineHeight) {
  const lines = doc.splitTextToSize(text, width);
  doc.text(lines, x, y);
  return y + lines.length * lineHeight;
}

function ensurePdfSpace(doc, y, needed, logos, language, pageWidth, pageHeight, margin) {
  if (y + needed <= pageHeight - margin) return y;
  doc.addPage();
  addPdfHeader(doc, logos, language, pageWidth, margin);
  return 45;
}

function addInstructionLanguage(doc, section, logos, isFirstPage) {
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 16;
  const textWidth = pageWidth - margin * 2;
  if (!isFirstPage) doc.addPage();
  addPdfHeader(doc, logos, section.language, pageWidth, margin);

  let y = 45;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(18);
  doc.setTextColor(15, 23, 42);
  y = addWrappedPdfText(doc, section.title, margin, y, textWidth, 7) + 2;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(10);
  doc.setTextColor(71, 85, 105);
  y = addWrappedPdfText(doc, section.subtitle, margin, y, textWidth, 5) + 5;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(9);
  doc.setTextColor(15, 118, 110);
  y = addWrappedPdfText(doc, "Creator / Créateur / Creador: Jorge Lizarazo", margin, y, textWidth, 5) + 6;

  section.blocks.forEach((block) => {
    y = ensurePdfSpace(doc, y, 18, logos, section.language, pageWidth, pageHeight, margin);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(12);
    doc.setTextColor(30, 64, 175);
    y = addWrappedPdfText(doc, block.heading, margin, y, textWidth, 6) + 2;

    doc.setFont("helvetica", "normal");
    doc.setFontSize(9.5);
    doc.setTextColor(30, 41, 59);
    block.text.forEach((paragraph) => {
      y = ensurePdfSpace(doc, y, 20, logos, section.language, pageWidth, pageHeight, margin);
      y = addWrappedPdfText(doc, paragraph, margin, y, textWidth, 4.8) + 3;
    });
    y += 2;
  });

  doc.setFont("helvetica", "italic");
  doc.setFontSize(8);
  doc.setTextColor(100, 116, 139);
  ensurePdfSpace(doc, y, 10, logos, section.language, pageWidth, pageHeight, margin);
  doc.text("BurrowScope - local ONNX inference guide", margin, pageHeight - 10);
}

async function exportInstructionsPdf() {
  const jsPDF = window.jspdf?.jsPDF;
  if (!jsPDF) {
    window.open("docs/BurrowScope_instructions.pdf", "_blank", "noopener,noreferrer");
    return;
  }

  const buttons = [qs("instructionsBtn"), qs("termsInstructionsBtn")].filter(Boolean);
  const originalLabels = buttons.map((button) => button.innerHTML);
  buttons.forEach((button) => {
    button.disabled = true;
    button.innerHTML = '<i data-lucide="loader-2" class="h-4 w-4"></i> Building PDF...';
  });
  if (buttons.length) {
    if (window.lucide) window.lucide.createIcons();
  }

  try {
    const settings = getInstructionSettings();
    const sections = buildInstructionSections(settings);
    const logos = await loadPdfLogos();
    const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "letter" });
    doc.setProperties({
      title: "BurrowScope Technical Instructions",
      subject: "Local ONNX Bank Swallow burrow segmentation workflow",
      author: "Jorge Lizarazo",
      creator: "BurrowScope",
    });

    sections.forEach((section, index) => addInstructionLanguage(doc, section, logos, index === 0));

    const pageCount = doc.getNumberOfPages();
    for (let page = 1; page <= pageCount; page += 1) {
      doc.setPage(page);
      doc.setFont("helvetica", "normal");
      doc.setFontSize(7);
      doc.setTextColor(100, 116, 139);
      doc.text(`Page ${page} / ${pageCount}`, doc.internal.pageSize.getWidth() - 16, doc.internal.pageSize.getHeight() - 10, { align: "right" });
    }

    doc.save("BurrowScope_instructions.pdf");
    logLine("Instructions PDF exported.");
  } catch (err) {
    console.error(err);
    logLine(`Instructions PDF export failed: ${err.message}`);
    alert(`Could not export the instructions PDF: ${err.message}`);
  } finally {
    buttons.forEach((button, index) => {
      button.disabled = false;
      button.innerHTML = originalLabels[index];
    });
    if (window.lucide) window.lucide.createIcons();
  }
}

async function initApp() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
  setLanguage("en");
  openTermsModal();
  document.querySelectorAll(".asset-img").forEach((img) => {
    img.addEventListener("error", () => {
      img.style.display = "none";
    });
  });
  document.querySelectorAll("[data-lang-toggle]").forEach((button) => {
    button.addEventListener("click", () => setLanguage(button.dataset.langToggle));
  });
  qs("acceptTermsBtn")?.addEventListener("click", acceptTerms);
  qs("declineTermsBtn")?.addEventListener("click", declineTerms);
  qs("termsInstructionsBtn")?.addEventListener("click", exportInstructionsPdf);
  qs("reviewTermsBtn")?.addEventListener("click", openTermsModal);
  updateThresholdLabels();
  refreshUsageGate();
  qs("onnxFileInput").addEventListener("change", handleManualModel);
  qs("imageInput").addEventListener("change", handleImageUpload);
  qs("openSamplesBtn")?.addEventListener("click", openSampleModal);
  qs("closeSamplesBtn")?.addEventListener("click", closeSampleModal);
  qs("selectAllSamplesBtn")?.addEventListener("click", selectAllSamples);
  qs("clearSamplesBtn")?.addEventListener("click", clearSampleSelection);
  qs("runSelectedSamplesBtn")?.addEventListener("click", runSelectedSamples);
  qs("runSelectedSamplesModalBtn")?.addEventListener("click", runSelectedSamples);
  qs("sampleModal")?.addEventListener("click", (event) => {
    if (event.target === qs("sampleModal")) closeSampleModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeSampleModal();
  });
  qs("runBtn").addEventListener("click", runInference);
  qs("prevAnalysedBtn")?.addEventListener("click", () => {
    if (currentAnalysisIndex > 0) showAnalysisAt(currentAnalysisIndex - 1).catch((err) => logLine(`Could not open previous analysed image: ${err.message}`));
  });
  qs("nextAnalysedBtn")?.addEventListener("click", () => {
    if (currentAnalysisIndex >= 0 && currentAnalysisIndex < sessionAnalyses.length - 1) showAnalysisAt(currentAnalysisIndex + 1).catch((err) => logLine(`Could not open next analysed image: ${err.message}`));
  });
  qs("instructionsBtn").addEventListener("click", exportInstructionsPdf);
  qs("exportCsvBtn").addEventListener("click", exportCsv);
  qs("exportJsonBtn").addEventListener("click", exportJson);
  qs("exportPngBtn").addEventListener("click", exportPng);
  ["confidenceSlider", "iouSlider", "maskSlider", "maxDetections"].forEach((id) => {
    qs(id).addEventListener("input", () => {
      updateThresholdLabels();
      if (lastOutputs) decodeAndRender();
    });
  });
  await loadSampleImages();
  await loadMetadata();
  await autoLoadModel();
  updateAnalysisNavigation();
  refreshUsageGate();
}

window.addEventListener("DOMContentLoaded", initApp);
