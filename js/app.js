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
let overlayHighlightsVisible = true;
let overlayRevealPercent = 100;
let localUploadQueue = [];
let localUploadSource = "files";
let zoomModeEnabled = false;
let zoomScale = 1;
let zoomPanX = 0;
let zoomPanY = 0;
let zoomPointerActive = false;
let zoomPointerStartX = 0;
let zoomPointerStartY = 0;
let zoomStartPanX = 0;
let zoomStartPanY = 0;

const TRANSLATIONS = {
  en: {
    headerTagline: "A model built with care to study Bank Swallows",
    fundingLabel: "Funding and support",
    quickEyebrow: "Quick start",
    heroEyebrow: "A model built with care to study Bank Swallows",
    heroTitle: "Bank Swallow burrow inspection",
    heroCopy: "Run local ONNX inference on one photo or selected image batches, then review boxes, masks, centroids, and exports.",
    quickTitle: "Visual inspection workflow",
    quickStep1: "Choose one bank photo, selected examples, or a local image set to inspect.",
    quickStep2: "Run the ONNX model in your browser and keep every image local.",
    quickStep3: "Review post-NMS boxes, masks, centroids, and confidence behaviour.",
    quickStep4: "Export CSV, JSON, or an overlay PNG for local field records.",
    modelSettingsEyebrow: "Model settings",
    detectionEngineTitle: "Detection engine",
    modelFileLabel: "Model file",
    manualModelUpload: "Load Local .onnx Model File",
    confidenceThresholdLabel: "Confidence Threshold",
    iouThresholdLabel: "IoU NMS Threshold",
    maskThresholdLabel: "Mask Threshold",
    maxDetectionsLabel: "Maximum detections",
    exampleImagesEyebrow: "Example images to learn how it works",
    exampleImagesTitle: "Try sample bank photos",
    openMiniGallery: "Open mini gallery",
    selectedExampleCountSuffix: "selected from 10 example photos.",
    runSelectedExamples: "Run selected examples",
    batchResultsTitle: "Batch results",
    imageColumn: "Image",
    burrowsColumn: "Burrows",
    imageInputEyebrow: "Image input",
    bankPhotoTitle: "Bank photo",
    imageInputHelp: "Place your own bank-wall images here to analyse them with the burrow model.",
    uploadBankPhoto: "Upload Bank Photo",
    imageFormatsHint: "JPG, PNG, or WebP. You can select several with Ctrl/Cmd.",
    selectFolderButton: "Select Folder",
    noImageLoaded: "No image loaded",
    chooseImageFirst: "Choose a bank-wall image first.",
    visualWorkspaceEyebrow: "Visual workspace",
    perImageDetectionTitle: "Per-image burrow detection",
    runBurrowDetection: "Run Burrow Detection",
    downloadCsvButton: "Download CSV Table",
    downloadCurrentCsvButton: "Download Current CSV",
    downloadSessionCsvButton: "Download Session CSV",
    downloadJsonButton: "Download Full JSON",
    downloadPngButton: "Download Marked-up PNG",
    downloadCurrentPngButton: "Download Current PNG",
    downloadSessionPngButton: "Download All Session PNGs",
    instructionsButton: "Instructions",
    glossaryButton: "Glossary",
    burrowHighlightsOn: "Burrow highlights on",
    burrowHighlightsOff: "Burrow highlights off",
    highlightReveal: "Highlight reveal",
    zoomModeOff: "Enable zoom",
    zoomModeOn: "Disable zoom",
    previousAnalysed: "Previous analysed",
    nextAnalysed: "Next analysed",
    noAnalysedImages: "No analysed images in this session.",
    viewerHelperText: "Masks, boxes, and centroids render here.",
    glossaryEyebrow: "Field glossary",
    glossaryTitle: "BurrowScope glossary",
    glossaryTermYolo: "YOLO",
    glossaryTermYoloSeg: "YOLO-seg",
    glossaryTermTransferLearning: "Transfer learning",
    glossaryTermOnnx: "ONNX",
    glossaryTermOnnxRuntime: "ONNX Runtime Web",
    glossaryTermNms: "NMS",
    glossaryTermIou: "IoU",
    glossaryTermConfidence: "Confidence threshold",
    glossaryTermMask: "Mask threshold",
    glossaryTermBoxCentroid: "Bounding box and centroid",
    glossaryYolo: "YOLO means “You Only Look Once”. Here it refers to the object-detection family used as the model architecture base.",
    glossaryYoloSeg: "YOLO-seg adds segmentation masks to the detector, so the app can show both burrow boxes and mask overlays for each image.",
    glossaryTransferLearning: "Transfer learning means the burrow model was fine-tuned from a pretrained YOLO segmentation model instead of being trained from zero.",
    glossaryOnnx: "ONNX is the portable inference format used by this app. The trained YOLO model was exported to ONNX for local browser inference.",
    glossaryOnnxRuntime: "ONNX Runtime Web is the engine that runs the exported model locally inside the browser, without sending images to a cloud service.",
    glossaryNms: "Non-maximum suppression removes duplicate-like overlapping detections so the final count is based on the most plausible boxes.",
    glossaryIou: "Intersection over Union measures how much two boxes overlap. It is used by NMS to decide when detections are too similar.",
    glossaryConfidence: "The confidence threshold controls how strong a prediction must be before it is kept for review.",
    glossaryMask: "The mask threshold controls how the predicted segmentation mask is binarised for visual overlay.",
    glossaryBoxCentroid: "A bounding box frames a predicted burrow opening. The centroid is the centre point of that box and helps with inspection and downstream spatial work.",
    detectionsDashboardEyebrow: "Detections dashboard",
    burrowsDetectedTitle: "Burrows Detected",
    maskStatusTitle: "Mask status",
    noPredictionsYet: "No predictions yet.",
    thresholdDiagnosticsTitle: "Threshold diagnostics",
    thresholdDiagnosticsSubtitle: "Decode, NMS, and mask checks are tracked independently.",
    confidenceCutoff: "Confidence cutoff",
    iouCutoff: "IoU NMS cutoff",
    maskCutoff: "Mask cutoff",
    maxDetectionsRow: "Max detections",
    decodedBoxes: "Decoded boxes",
    postNmsBoxes: "Post-NMS boxes",
    maskAvailability: "Mask availability",
    selectExampleImages: "Select example images",
    selectedModalSuffix: "selected. Choose several and run them as a batch.",
    selectAll: "Select all",
    clearButton: "Clear",
    runSelected: "Run selected",
    loadingExamples: "Loading examples...",
    modelUpdateHint: "If there is an updated inference model, upload it here.",
    consoleEyebrow: "Processing log",
    consoleTitle: "Session",
    footerCopy: "A Birds Canada and Western University initiative, supported by Mitacs Accelerate, to develop accessible and free AI tools for NGOs, students, and the Government of Canada to support insectivorous bird monitoring in Canada.",
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
    headerTagline: "Un modèle conçu avec soin pour étudier les hirondelles de rivage",
    fundingLabel: "Financement et soutien",
    quickEyebrow: "Démarrage rapide",
    heroEyebrow: "Un modèle conçu avec soin pour étudier les hirondelles de rivage",
    heroTitle: "Inspection des terriers d'hirondelles de rivage",
    heroCopy: "Lancez l'inférence ONNX locale sur une photo ou sur des lots d'images sélectionnées, puis révisez les boîtes, les masques, les centroïdes et les exports.",
    quickTitle: "Flux d'inspection visuelle",
    quickStep1: "Choisissez une photo de berge, des exemples sélectionnés ou un ensemble local d'images à inspecter.",
    quickStep2: "Exécutez le modèle ONNX dans votre navigateur et gardez chaque image en local.",
    quickStep3: "Révisez les boîtes post-NMS, les masques, les centroïdes et le comportement des seuils de confiance.",
    quickStep4: "Exportez un CSV, un JSON ou un PNG de superposition pour vos dossiers de terrain.",
    modelSettingsEyebrow: "Paramètres du modèle",
    detectionEngineTitle: "Moteur de détection",
    modelFileLabel: "Fichier du modèle",
    manualModelUpload: "Charger un fichier local .onnx",
    confidenceThresholdLabel: "Seuil de confiance",
    iouThresholdLabel: "Seuil IoU NMS",
    maskThresholdLabel: "Seuil du masque",
    maxDetectionsLabel: "Nombre maximal de détections",
    exampleImagesEyebrow: "Images d'exemple pour voir comment cela fonctionne",
    exampleImagesTitle: "Essayer des photos d'exemple",
    openMiniGallery: "Ouvrir la mini-galerie",
    selectedExampleCountSuffix: "sélectionnées parmi 10 photos d'exemple.",
    runSelectedExamples: "Lancer les exemples sélectionnés",
    batchResultsTitle: "Résultats du lot",
    imageColumn: "Image",
    burrowsColumn: "Terriers",
    imageInputEyebrow: "Entrée image",
    bankPhotoTitle: "Photo de berge",
    imageInputHelp: "Placez ici vos propres images de paroi de berge pour les analyser avec le modèle de terriers.",
    uploadBankPhoto: "Importer une photo de berge",
    imageFormatsHint: "JPG, PNG ou WebP. Vous pouvez en choisir plusieurs avec Ctrl/Cmd.",
    selectFolderButton: "Sélectionner un dossier",
    noImageLoaded: "Aucune image chargée",
    chooseImageFirst: "Choisissez d'abord une image de paroi de berge.",
    visualWorkspaceEyebrow: "Espace visuel",
    perImageDetectionTitle: "Détection de terriers par image",
    runBurrowDetection: "Lancer la détection des terriers",
    downloadCsvButton: "Télécharger le tableau CSV",
    downloadCurrentCsvButton: "Télécharger le CSV courant",
    downloadSessionCsvButton: "Télécharger le CSV de session",
    downloadJsonButton: "Télécharger le JSON complet",
    downloadPngButton: "Télécharger le PNG annoté",
    downloadCurrentPngButton: "Télécharger le PNG courant",
    downloadSessionPngButton: "Télécharger tous les PNG de session",
    instructionsButton: "Instructions",
    glossaryButton: "Glossaire",
    burrowHighlightsOn: "Surlignage des terriers activé",
    burrowHighlightsOff: "Surlignage des terriers désactivé",
    highlightReveal: "Révélation du surlignage",
    zoomModeOff: "Activer le zoom",
    zoomModeOn: "Désactiver le zoom",
    previousAnalysed: "Image analysée précédente",
    nextAnalysed: "Image analysée suivante",
    noAnalysedImages: "Aucune image analysée dans cette session.",
    viewerHelperText: "Les masques, boîtes et centroïdes s'affichent ici.",
    glossaryEyebrow: "Glossaire terrain",
    glossaryTitle: "Glossaire BurrowScope",
    glossaryTermYolo: "YOLO",
    glossaryTermYoloSeg: "YOLO-seg",
    glossaryTermTransferLearning: "Apprentissage par transfert",
    glossaryTermOnnx: "ONNX",
    glossaryTermOnnxRuntime: "ONNX Runtime Web",
    glossaryTermNms: "NMS",
    glossaryTermIou: "IoU",
    glossaryTermConfidence: "Seuil de confiance",
    glossaryTermMask: "Seuil du masque",
    glossaryTermBoxCentroid: "Boîte englobante et centroïde",
    glossaryYolo: "YOLO signifie « You Only Look Once ». Ici, cela désigne la famille de détection d'objets utilisée comme base d'architecture du modèle.",
    glossaryYoloSeg: "YOLO-seg ajoute des masques de segmentation au détecteur, donc l'application peut montrer les boîtes et les masques des terriers pour chaque image.",
    glossaryTransferLearning: "Le transfer learning veut dire que le modèle des terriers a été affiné à partir d'un modèle YOLO de segmentation préentraîné au lieu d'être entraîné depuis zéro.",
    glossaryOnnx: "ONNX est le format d'inférence portable utilisé par l'application. Le modèle YOLO entraîné a été exporté en ONNX pour l'inférence locale dans le navigateur.",
    glossaryOnnxRuntime: "ONNX Runtime Web est le moteur qui exécute le modèle exporté localement dans le navigateur, sans envoyer les images vers un service infonuagique.",
    glossaryNms: "La suppression non maximale enlève les détections qui se chevauchent comme des doublons, pour que le compte final repose sur les boîtes les plus plausibles.",
    glossaryIou: "L'Intersection over Union mesure le chevauchement entre deux boîtes. Elle sert à la NMS pour décider quand des détections sont trop semblables.",
    glossaryConfidence: "Le seuil de confiance contrôle la force minimale d'une prédiction avant qu'elle soit conservée pour la révision.",
    glossaryMask: "Le seuil du masque contrôle la binarisation du masque de segmentation prédit pour la superposition visuelle.",
    glossaryBoxCentroid: "Une boîte englobante encadre une ouverture de terrier prédite. Le centroïde est le point central de cette boîte et aide pour l'inspection et le travail spatial en aval.",
    detectionsDashboardEyebrow: "Tableau des détections",
    burrowsDetectedTitle: "Terriers détectés",
    maskStatusTitle: "État des masques",
    noPredictionsYet: "Aucune prédiction pour le moment.",
    thresholdDiagnosticsTitle: "Diagnostic des seuils",
    thresholdDiagnosticsSubtitle: "Le décodage, la NMS et les vérifications de masque sont suivis séparément.",
    confidenceCutoff: "Seuil de confiance",
    iouCutoff: "Seuil IoU NMS",
    maskCutoff: "Seuil du masque",
    maxDetectionsRow: "Détections max",
    decodedBoxes: "Boîtes décodées",
    postNmsBoxes: "Boîtes post-NMS",
    maskAvailability: "Disponibilité des masques",
    selectExampleImages: "Sélectionner des images d'exemple",
    selectedModalSuffix: "sélectionnées. Choisissez-en plusieurs et lancez-les en lot.",
    selectAll: "Tout sélectionner",
    clearButton: "Effacer",
    runSelected: "Lancer la sélection",
    loadingExamples: "Chargement des exemples...",
    modelUpdateHint: "S'il existe une mise à jour du modèle d'inférence, téléversez-la ici.",
    consoleEyebrow: "Journal de traitement",
    consoleTitle: "Session",
    footerCopy: "Une initiative de Birds Canada et de Western University, appuyée par Mitacs Accelerate, pour développer des outils d'IA accessibles et gratuits destinés aux ONG, aux étudiantes et étudiants, et au gouvernement du Canada afin de faciliter le suivi des oiseaux insectivores au Canada.",
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

function t(key) {
  return TRANSLATIONS[activeLanguage]?.[key] || TRANSLATIONS.en[key] || key;
}

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
  if (!currentImage && qs("imageName")?.dataset.i18n === "noImageLoaded") {
    qs("imageName").innerText = t("noImageLoaded");
  }
  if (!currentImage && qs("imageMeta")?.dataset.i18n === "chooseImageFirst") {
    qs("imageMeta").innerText = t("chooseImageFirst");
  }
  if (qs("runBtnText") && qs("runBtnText").innerText !== "Running..." && qs("runBtnText").innerText !== "Exécution...") {
    qs("runBtnText").innerText = t("runBurrowDetection");
  }
  updateOverlayControls();
  updateZoomControls();
  renderSampleGrid();
}

function syncModalState() {
  const openModal = [qs("termsModal"), qs("sampleModal"), qs("glossaryModal")].some((modal) => modal && !modal.classList.contains("hidden"));
  document.body.classList.toggle("modal-open", openModal);
}

function openTermsModal() {
  const modal = qs("termsModal");
  if (!modal) return;
  modal.classList.remove("hidden");
  modal.classList.add("flex");
  syncModalState();
}

function closeTermsModal() {
  const modal = qs("termsModal");
  if (!modal) return;
  modal.classList.add("hidden");
  modal.classList.remove("flex");
  syncModalState();
}

function openGlossaryModal() {
  const modal = qs("glossaryModal");
  if (!modal) return;
  modal.classList.remove("hidden");
  modal.classList.add("flex");
  syncModalState();
}

function closeGlossaryModal() {
  const modal = qs("glossaryModal");
  if (!modal) return;
  modal.classList.add("hidden");
  modal.classList.remove("flex");
  syncModalState();
}

function refreshUsageGate() {
  const locked = !termsAccepted;
  const imageInput = qs("imageInput");
  const folderInput = qs("folderInput");
  const imageLabel = qs("imageInputLabel");
  const runButton = qs("runBtn");
  const manualButton = qs("manualModelUpload");
  const folderButton = qs("folderSelectBtn");
  if (imageInput) imageInput.disabled = locked || !session;
  if (folderInput) folderInput.disabled = locked || !session;
  if (manualButton) manualButton.disabled = locked;
  if (folderButton) folderButton.disabled = locked || !session || batchRunning;
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

function getViewerCanvases() {
  return {
    base: qs("viewerCanvas"),
    overlay: qs("overlayCanvas"),
  };
}

function getViewerFrame() {
  return document.querySelector(".canvas-frame");
}

function getViewerStage() {
  return document.querySelector(".compare-stage");
}

function clampZoomScale(nextScale) {
  return Math.min(5, Math.max(1, nextScale));
}

function applyViewerTransform() {
  const frame = getViewerFrame();
  const stage = getViewerStage();
  if (!frame || !stage) return;
  stage.style.transform = `translate(${zoomPanX}px, ${zoomPanY}px) scale(${zoomScale})`;
  frame.classList.toggle("is-zoomed", zoomModeEnabled);
  frame.classList.toggle("is-panning", zoomPointerActive);
}

function resetViewerZoom() {
  zoomScale = 1;
  zoomPanX = 0;
  zoomPanY = 0;
  zoomPointerActive = false;
  applyViewerTransform();
  updateZoomControls();
}

function updateZoomControls() {
  const toggleBtn = qs("zoomToggleBtn");
  const resetBtn = qs("zoomResetBtn");
  const hasImage = Boolean(currentImage);
  if (toggleBtn) {
    toggleBtn.disabled = !hasImage;
    toggleBtn.innerHTML = zoomModeEnabled
      ? `<i data-lucide="scan-search" class="h-4 w-4"></i> ${escapeHtml(t("zoomModeOn"))}`
      : `<i data-lucide="zoom-in" class="h-4 w-4"></i> ${escapeHtml(t("zoomModeOff"))}`;
  }
  if (resetBtn) resetBtn.disabled = !hasImage || (!zoomModeEnabled && zoomScale === 1 && zoomPanX === 0 && zoomPanY === 0);
  if (window.lucide) window.lucide.createIcons();
}

function updateOverlayControls() {
  const slider = qs("overlayRevealSlider");
  const button = qs("toggleOverlayBtn");
  const divider = qs("overlayDivider");
  const hasImage = Boolean(currentImage);
  const hasDetections = currentDetections.length > 0;

  if (slider) {
    slider.value = String(overlayRevealPercent);
    slider.disabled = !hasImage || !hasDetections || !overlayHighlightsVisible;
  }

  if (button) {
    button.disabled = !hasImage || !hasDetections;
    button.innerHTML = overlayHighlightsVisible
      ? `<i data-lucide="layers-3" class="h-4 w-4"></i> ${escapeHtml(t("burrowHighlightsOn"))}`
      : `<i data-lucide="image" class="h-4 w-4"></i> ${escapeHtml(t("burrowHighlightsOff"))}`;
  }

  const overlayCanvas = qs("overlayCanvas");
  if (overlayCanvas) {
    overlayCanvas.style.opacity = overlayHighlightsVisible ? "1" : "0";
    overlayCanvas.style.clipPath = overlayHighlightsVisible
      ? `inset(0 ${Math.max(0, 100 - overlayRevealPercent)}% 0 0)`
      : "inset(0 100% 0 0)";
  }

  if (divider) {
    divider.classList.toggle("hidden", !overlayHighlightsVisible || !hasImage || !hasDetections);
    divider.style.left = `${overlayRevealPercent}%`;
  }

  if (window.lucide) window.lucide.createIcons();
  updateZoomControls();
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
      ? (activeLanguage === "fr"
        ? `Image analysée ${currentAnalysisIndex + 1} sur ${total} : ${sessionAnalyses[currentAnalysisIndex]?.name || "Image sans nom"}`
        : `Analysed image ${currentAnalysisIndex + 1} of ${total}: ${sessionAnalyses[currentAnalysisIndex]?.name || "Unnamed image"}`)
      : t("noAnalysedImages");
  }
}

function drawStoredDetections() {
  drawBaseImage();
  const canvas = qs("overlayCanvas");
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
  updateOverlayControls();
}

function renderLoadedAnalysis() {
  qs("countValue").innerText = String(currentDetections.length);
  if (qs("exportCurrentCsvBtn")) qs("exportCurrentCsvBtn").disabled = currentDetections.length === 0;
  if (qs("exportSessionCsvBtn")) qs("exportSessionCsvBtn").disabled = sessionAnalyses.length === 0;
  if (qs("exportJsonBtn")) qs("exportJsonBtn").disabled = currentDetections.length === 0;
  if (qs("exportCurrentPngBtn")) qs("exportCurrentPngBtn").disabled = currentDetections.length === 0;
  if (qs("exportSessionPngBtn")) qs("exportSessionPngBtn").disabled = sessionAnalyses.length === 0;
  updateDetectionDiagnostics(currentDetections);
  renderTable();
  updateAnalysisNavigation();
  drawStoredDetections();
  updateOverlayControls();
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
  resetViewerZoom();
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
  setTextIfPresent("diagnosticDecodedBoxes", activeLanguage === "fr" ? "En attente d'inférence" : "Awaiting inference");
  setTextIfPresent("diagnosticPostNmsBoxes", "0");
  setTextIfPresent("diagnosticMaskAvailability", activeLanguage === "fr" ? "En attente d'inférence" : "Awaiting inference");
  setTextIfPresent(
    "maskStatusText",
    activeLanguage === "fr"
      ? "Le comptage utilise les boîtes décodées post-NMS; la disponibilité des masques est indiquée séparément."
      : "Counts report decoded post-NMS boxes; mask availability can be reported separately.",
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
      ? (activeLanguage === "fr"
        ? `Le comptage utilise ${count} boîtes post-NMS. Masques disponibles pour ${masksAvailable}/${count} détections.`
        : `Counts use ${count} post-NMS boxes. Masks available for ${masksAvailable}/${count} detections.`)
      : (activeLanguage === "fr"
        ? "Le comptage utilise les boîtes décodées post-NMS; la disponibilité des masques est indiquée séparément."
        : "Counts report decoded post-NMS boxes; mask availability can be reported separately."),
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
  await prepareLocalUploadQueue(event.target.files, "files");
}

async function handleFolderUpload(event) {
  if (!termsAccepted) {
    openTermsModal();
    return;
  }
  await prepareLocalUploadQueue(event.target.files, "folder");
}

function isAcceptedImageFile(file) {
  if (!file) return false;
  if (file.type?.startsWith("image/")) return true;
  return /\.(jpg|jpeg|png|webp)$/i.test(file.name || "");
}

async function prepareLocalUploadQueue(fileList, source) {
  const files = Array.from(fileList || []).filter(isAcceptedImageFile);
  if (!files.length) return;
  localUploadQueue = files;
  localUploadSource = source;
  const { img } = await window.BurrowImageUtils.readImageFile(files[0]);
  setCurrentImage(img, files[0].name);
  if (files.length > 1) {
    qs("imageName").innerText = source === "folder"
      ? (activeLanguage === "fr" ? `${files.length} images du dossier prêtes` : `${files.length} folder images ready`)
      : (activeLanguage === "fr" ? `${files.length} images prêtes` : `${files.length} images ready`);
    qs("imageMeta").innerText = source === "folder"
      ? (activeLanguage === "fr"
        ? "Appuyez sur Run Burrow Detection pour analyser toutes les images du dossier."
        : "Press Run Burrow Detection to analyse all images in this folder.")
      : (activeLanguage === "fr"
        ? "Appuyez sur Run Burrow Detection pour analyser toutes les images sélectionnées."
        : "Press Run Burrow Detection to analyse all selected images.");
  }
  logLine(source === "folder"
    ? `Folder queue ready: ${files.length} image(s).`
    : `Local upload queue ready: ${files.length} image(s).`);
}

function setCurrentImage(img, name) {
  currentImage = img;
  currentImageName = name;
  qs("imageName").innerText = name;
  qs("imageMeta").innerText = `${img.naturalWidth} x ${img.naturalHeight}px`;
  overlayHighlightsVisible = true;
  overlayRevealPercent = 100;
  resetViewerZoom();
  drawBaseImage();
  qs("runBtn").disabled = !session || batchRunning;
  if (qs("exportCurrentCsvBtn")) qs("exportCurrentCsvBtn").disabled = true;
  if (qs("exportSessionCsvBtn")) qs("exportSessionCsvBtn").disabled = sessionAnalyses.length === 0;
  if (qs("exportJsonBtn")) qs("exportJsonBtn").disabled = true;
  if (qs("exportCurrentPngBtn")) qs("exportCurrentPngBtn").disabled = true;
  if (qs("exportSessionPngBtn")) qs("exportSessionPngBtn").disabled = sessionAnalyses.length === 0;
  currentDetections = [];
  qs("countValue").innerText = "0";
  refreshUsageGate();
  resetDetectionDiagnostics();
  updateAnalysisNavigation();
  renderTable();
  updateOverlayControls();
  logLine(`Image loaded: ${name}`);
}

async function runLocalUploadBatch() {
  if (!localUploadQueue.length) return;
  batchResults = [];
  renderBatchResults();
  setBatchControlsDisabled(true);
  const totalFiles = localUploadQueue.length;
  setStatus(
    activeLanguage === "fr"
      ? `Analyse du lot 1/${totalFiles}...`
      : `Running local batch 1/${totalFiles}...`,
    "neutral",
  );
  logLine(`Local batch started with ${totalFiles} image(s).`);

  try {
    for (let index = 0; index < localUploadQueue.length; index += 1) {
      const file = localUploadQueue[index];
      setStatus(
        activeLanguage === "fr"
          ? `Analyse du lot ${index + 1}/${totalFiles}...`
          : `Running local batch ${index + 1}/${totalFiles}...`,
        "neutral",
      );
      const { img } = await window.BurrowImageUtils.readImageFile(file);
      setCurrentImage(img, file.name);
      await new Promise((resolve) => requestAnimationFrame(resolve));
      const detections = await runModelOnCurrentImage();
      batchResults.push({
        id: file.name,
        name: file.webkitRelativePath || file.name,
        count: detections.length,
      });
      renderBatchResults();
      await new Promise((resolve) => setTimeout(resolve, 80));
    }
    const total = batchResults.reduce((sum, row) => sum + row.count, 0);
    setStatus(
      activeLanguage === "fr"
        ? `Lot terminé: ${total} terriers dans ${batchResults.length} images.`
        : `Batch complete: ${total} burrows in ${batchResults.length} images.`,
      "ok",
    );
    logLine(`Local batch complete: ${total} burrows in ${batchResults.length} images.`);
  } catch (err) {
    setStatus(activeLanguage === "fr" ? "Échec du lot" : "Batch failed", "error");
    logLine(`Local batch error: ${err.message}`);
    console.error(err);
  } finally {
    setBatchControlsDisabled(false);
  }
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
    grid.innerHTML = `<p class="col-span-2 text-sm text-slate-500">${activeLanguage === "fr" ? "Images d'exemple non disponibles." : "Example images not available."}</p>`;
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
      <span class="sample-check">${selected ? (activeLanguage === "fr" ? "Choisie" : "Selected") : (activeLanguage === "fr" ? "Choisir" : "Select")}</span>
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
  syncModalState();
}

function closeSampleModal() {
  const modal = qs("sampleModal");
  if (!modal) return;
  modal.classList.add("hidden");
  modal.classList.remove("flex");
  syncModalState();
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
  const { base, overlay } = getViewerCanvases();
  const display = getDisplaySize(currentImage);
  [base, overlay].forEach((canvas) => {
    if (!canvas) return;
    canvas.width = display.width;
    canvas.height = display.height;
    canvas.style.width = `${display.width}px`;
    canvas.style.height = `${display.height}px`;
  });
  base.style.backgroundImage = `url("${String(currentImage.src || "").replaceAll('"', '\\"')}")`;
  base.style.backgroundPosition = "center";
  base.style.backgroundRepeat = "no-repeat";
  base.style.backgroundSize = "100% 100%";
  const ctx = base.getContext("2d", { willReadFrequently: true });
  ctx.clearRect(0, 0, base.width, base.height);
  ctx.drawImage(currentImage, 0, 0, base.width, base.height);
  if (overlay) {
    const overlayCtx = overlay.getContext("2d", { willReadFrequently: true });
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  }
  applyViewerTransform();
  updateOverlayControls();
}

function setZoomMode(enabled) {
  zoomModeEnabled = Boolean(enabled);
  if (!zoomModeEnabled) {
    zoomScale = 1;
    zoomPanX = 0;
    zoomPanY = 0;
    zoomPointerActive = false;
  }
  applyViewerTransform();
  updateZoomControls();
}

function handleViewerWheel(event) {
  if (!zoomModeEnabled || !currentImage) return;
  event.preventDefault();
  const stage = getViewerStage();
  if (!stage) return;
  const previousScale = zoomScale;
  const nextScale = clampZoomScale(previousScale + (event.deltaY < 0 ? 0.2 : -0.2));
  if (nextScale === previousScale) return;
  const stageRect = stage.getBoundingClientRect();
  const anchorX = event.clientX - stageRect.left;
  const anchorY = event.clientY - stageRect.top;
  zoomPanX -= (anchorX / Math.max(previousScale, 0.001)) * (nextScale - previousScale);
  zoomPanY -= (anchorY / Math.max(previousScale, 0.001)) * (nextScale - previousScale);
  zoomScale = nextScale;
  applyViewerTransform();
  updateZoomControls();
}

function handleViewerPointerDown(event) {
  if (!zoomModeEnabled || zoomScale <= 1) return;
  zoomPointerActive = true;
  zoomPointerStartX = event.clientX;
  zoomPointerStartY = event.clientY;
  zoomStartPanX = zoomPanX;
  zoomStartPanY = zoomPanY;
  event.currentTarget?.setPointerCapture?.(event.pointerId);
  applyViewerTransform();
}

function handleViewerPointerMove(event) {
  if (!zoomModeEnabled || !zoomPointerActive) return;
  zoomPanX = zoomStartPanX + (event.clientX - zoomPointerStartX);
  zoomPanY = zoomStartPanY + (event.clientY - zoomPointerStartY);
  applyViewerTransform();
}

function stopViewerPointerInteraction(event) {
  if (!zoomPointerActive) return;
  zoomPointerActive = false;
  event?.currentTarget?.releasePointerCapture?.(event.pointerId);
  applyViewerTransform();
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
  if (localUploadQueue.length > 1) {
    await runLocalUploadBatch();
    return;
  }
  qs("runBtn").disabled = true;
  qs("runBtnText").innerText = activeLanguage === "fr" ? "Exécution..." : "Running...";
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
    qs("runBtnText").innerText = t("runBurrowDetection");
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
  const canvas = qs("overlayCanvas");
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
  if (qs("exportCurrentCsvBtn")) qs("exportCurrentCsvBtn").disabled = count === 0;
  if (qs("exportSessionCsvBtn")) qs("exportSessionCsvBtn").disabled = sessionAnalyses.length === 0;
  if (qs("exportJsonBtn")) qs("exportJsonBtn").disabled = count === 0;
  if (qs("exportCurrentPngBtn")) qs("exportCurrentPngBtn").disabled = count === 0;
  if (qs("exportSessionPngBtn")) qs("exportSessionPngBtn").disabled = sessionAnalyses.length === 0;
  updateDetectionDiagnostics(currentDetections);
  renderTable();
  storeCurrentAnalysis();
  updateAnalysisNavigation();
  updateOverlayControls();
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
    body.innerHTML = `<tr><td colspan="5" class="px-3 py-8 text-center text-slate-400">${escapeHtml(t("noPredictionsYet"))}</td></tr>`;
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

function downloadCsvRows(filename, rows) {
  if (!rows.length) return;
  const headers = Object.keys(rows[0]);
  const lines = [
    headers.join(","),
    ...rows.map((row) => headers.map((h) => window.BurrowImageUtils.csvEscape(row[h])).join(",")),
  ];
  window.BurrowImageUtils.downloadTextFile(filename, lines.join("\n"), "text/csv");
}

function buildCurrentImageSummaryRows() {
  return [{
    session_image_index: currentAnalysisIndex >= 0 ? currentAnalysisIndex + 1 : 1,
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
}

function exportCurrentCsv() {
  const rows = predictionRows();
  if (!rows.length) return;
  downloadCsvRows(`${currentImageName || "burrows"}_burrowscope_predictions.csv`, rows);
  downloadCsvRows(`${currentImageName || "burrows"}_burrow_summary.csv`, buildCurrentImageSummaryRows());
  logLine(`Current-image CSV exports created for ${currentImageName || "the current image"}.`);
}

function exportSessionCsv() {
  const rows = sessionAnalyses.length
    ? sessionAnalyses.flatMap(buildSessionRowsFromAnalysis)
    : predictionRows();
  if (!rows.length) return;
  downloadCsvRows(`burrowscope_session_predictions.csv`, rows);
  const summaryRows = sessionAnalyses.length ? buildSessionSummaryRows() : buildCurrentImageSummaryRows();
  downloadCsvRows(`burrowscope_session_image_summary.csv`, summaryRows);
  logLine(`Session CSV exports created for ${sessionAnalyses.length || 1} analysed image(s).`);
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

function renderAnalysisToPngDataUrl(analysis, img) {
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;
  const baseCanvas = document.createElement("canvas");
  baseCanvas.width = width;
  baseCanvas.height = height;
  const baseCtx = baseCanvas.getContext("2d", { willReadFrequently: true });
  baseCtx.drawImage(img, 0, 0, width, height);

  const overlayCanvas = document.createElement("canvas");
  overlayCanvas.width = width;
  overlayCanvas.height = height;
  const overlayCtx = overlayCanvas.getContext("2d", { willReadFrequently: true });
  const detections = (analysis?.detections || []).map((det) => ({ ...det, box: { ...det.box }, centroid: { ...det.centroid }, mask: { ...det.mask } }));
  const display = {
    width,
    height,
    originalWidth: width,
    originalHeight: height,
  };
  const drawOptions = {
    maskThreshold: Number(qs("maskSlider").value),
    maskOpacity: 0.42,
    masksOptional: true,
  };
  if (typeof window.BurrowYoloSeg.drawDetectionMasks === "function") {
    try {
      window.BurrowYoloSeg.drawDetectionMasks(overlayCtx, detections, display, drawOptions);
    } catch (err) {
      drawBoxesFallback(overlayCtx, detections, display);
    }
  } else {
    drawBoxesFallback(overlayCtx, detections, display);
  }

  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = width;
  exportCanvas.height = height;
  const exportCtx = exportCanvas.getContext("2d");
  exportCtx.drawImage(baseCanvas, 0, 0);
  if (detections.length) exportCtx.drawImage(overlayCanvas, 0, 0);
  return exportCanvas.toDataURL("image/png");
}

function triggerPngDownload(dataUrl, filename) {
  const a = document.createElement("a");
  a.href = dataUrl;
  a.download = filename;
  a.click();
}

function exportCurrentPng() {
  const baseCanvas = qs("viewerCanvas");
  const overlayCanvas = qs("overlayCanvas");
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = baseCanvas.width;
  exportCanvas.height = baseCanvas.height;
  const exportCtx = exportCanvas.getContext("2d");
  exportCtx.drawImage(baseCanvas, 0, 0);
  if (currentDetections.length && overlayCanvas) {
    exportCtx.drawImage(overlayCanvas, 0, 0);
  }
  triggerPngDownload(exportCanvas.toDataURL("image/png"), `${currentImageName || "burrows"}_overlay_current.png`);
  logLine(`Current-image PNG exported for ${currentImageName || "the current image"}.`);
}

async function exportSessionPngs() {
  if (!sessionAnalyses.length) return;
  for (let index = 0; index < sessionAnalyses.length; index += 1) {
    const analysis = sessionAnalyses[index];
    const img = await loadImageFromSource(analysis.src);
    const dataUrl = renderAnalysisToPngDataUrl(analysis, img);
    triggerPngDownload(dataUrl, `${analysis.name || `analysed_image_${index + 1}`}_overlay.png`);
    await new Promise((resolve) => setTimeout(resolve, 180));
  }
  logLine(`Session PNG export triggered for ${sessionAnalyses.length} analysed image(s).`);
}

const PDF_LOGOS = [
  { src: "img/bankLogo1.png", width: 22, height: 22, row: 0, maxHeight: 12 },
  { src: "img/birds-canada-logo.svg", width: 30, height: 16, row: 0, maxHeight: 10 },
  { src: "img/Naturecounts-logo.svg", width: 36, height: 14, row: 0, maxHeight: 9.5 },
  { src: "img/animals-on-the-move-logo.svg", width: 34, height: 16, row: 1, maxHeight: 9.5 },
  { src: "img/western-logo.svg", width: 36, height: 14, row: 1, maxHeight: 9.5 },
  { src: "img/Mitacs_logo_blue.webp", width: 30, height: 10, row: 1, maxHeight: 7.5 },
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
            "BurrowScope is a local browser application for inspecting Bank Swallow burrows in bank or cliff photographs. It uses a YOLO11 segmentation model fine-tuned with transfer learning and then exported to ONNX for local inference. The practical output is per-image detection: post-NMS boxes, mask overlays, centroids, confidence values, and export files.",
            "The app is intended for non-profit research, conservation, field checking, and model-assisted review. It is not a replacement for biological judgement; it is a fast inspection layer that should be reviewed by a person before reporting.",
          ],
        },
        {
          heading: "Current inference configuration",
          text: [
            `Detection engine: Local ONNX. Model file: ${settings.modelName}. Current status shown by the app: ${settings.modelStatus}.`,
            "The model runs in the browser through ONNX Runtime Web. The ONNX file is the exported inference form of the trained YOLO segmentation model. Images and inference results stay on the local machine unless the user exports and shares them manually.",
            `Confidence threshold: ${settings.confidence}. IoU NMS threshold: ${settings.iou}. Mask threshold: ${settings.mask}. Maximum detections: ${settings.maxDetections}.`,
            "The default values are meant to stay fixed in most cases, but they can be useful for troubleshooting if someone uses images that are very different from the training data. If there is a newer inference model, use the Load Local .onnx Model File control in Model settings and then run the same image again to compare behaviour.",
          ],
        },
        {
          heading: "How to run an image or a small batch",
          text: [
            "1. Accept the local-use conditions when the application opens. This only unlocks the local controls.",
            "2. Use the Image input / Bank photo panel to place your own JPG, PNG, or WebP bank-wall images into the app, or use the example gallery for a quick test. You can also select several images or a whole folder.",
            "3. Press Run Burrow Detection from the top of the Visual workspace panel. The count badge reports final post-NMS boxes. NMS means non-maximum suppression: duplicate-like overlapping boxes are filtered inside the same image. Mask status is reported separately, so a failed mask should not erase a valid box detection.",
            "4. Inspect the overlay visually. Check whether boxes cover real burrow openings, whether centroids are inside those boxes, and whether obvious false positives or missed burrows appear.",
            "5. Use Download Current CSV or Download Current PNG for the image on screen. Use Download Session CSV or Download All Session PNGs when you want files for every analysed image in the current session. The Instructions button exports this document.",
          ],
        },
        {
          heading: "Reading the results",
          text: [
            "Confidence controls how strict the detector is. A lower confidence threshold usually increases recall but may add false positives. A higher threshold usually reduces false positives but may miss faint or partly occluded burrows.",
            "IoU NMS controls duplicate suppression between overlapping boxes. If the same burrow is counted twice, a lower or moderate NMS setting can help. If nearby burrows are merged or suppressed, inspect the image carefully before changing the threshold.",
            "Mask threshold controls the binary mask extracted from the segmentation prototype. The object count is box-centred, because boxes are the most stable ONNX decoding signal. Masks are visual and spatial evidence, but they should not be allowed to delete detections.",
            "Changes to thresholds are mainly diagnostic and should not be treated as a substitute for retraining or re-validating the model on a new image domain. Behaviour still depends strongly on the original fine-tuning data and the transfer-learning stage used to build the model.",
          ],
        },
        {
          heading: "Export schema",
          text: [
            "Current CSV exports are tied to the image on screen. Session CSV exports include every analysed image in the current session and also generate an image-level summary table.",
            "Detection-level CSV rows include: id, confidence, x1, y1, x2, y2, width, height, centroid_x, centroid_y, mask_available, mask_status, mask_area_px, and count_source.",
            "The JSON export keeps the full detection objects for the current image, including box coordinates, centroid, confidence, class name, mask status, mask area, and the same count_source used by the visual count.",
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
            "BurrowScope est une application locale dans le navigateur pour inspecter les terriers d'hirondelles de rivage dans des photos de berges ou de falaises. Elle utilise un modèle YOLO11 de segmentation affiné par transfer learning, puis exporté en ONNX pour l'inférence locale. Le résultat pratique est par image: boîtes post-NMS, masques superposés, centroïdes, valeurs de confiance et fichiers d'export.",
            "L'application est pensée pour la recherche sans but lucratif, la conservation, la vérification de terrain et la révision assistée par modèle. Ce n'est pas un remplacement du jugement biologique; c'est une couche rapide d'inspection qui doit être révisée par une personne avant un rapport.",
          ],
        },
        {
          heading: "Configuration actuelle d'inférence",
          text: [
            `Moteur de détection: ONNX local. Fichier modèle: ${settings.modelName}. État affiché par l'application: ${settings.modelStatus}.`,
            "Le modèle fonctionne dans le navigateur avec ONNX Runtime Web. Le fichier ONNX est la forme d'inférence exportée du modèle YOLO de segmentation entraîné. Les images et les résultats d'inférence restent sur l'ordinateur local, sauf si l'utilisateur les exporte et les partage lui-même.",
            `Seuil de confiance: ${settings.confidence}. Seuil IoU NMS: ${settings.iou}. Seuil de masque: ${settings.mask}. Nombre maximal de détections: ${settings.maxDetections}.`,
            "Les valeurs par défaut sont faites pour rester fixes dans la plupart des cas, mais elles peuvent aider au dépannage si quelqu'un utilise des images très différentes de celles du jeu d'entraînement. S'il existe une version plus récente du modèle d'inférence, utilisez Load Local .onnx Model File dans Model settings, puis relancez la même image pour comparer le comportement.",
          ],
        },
        {
          heading: "Comment traiter une image ou un petit lot",
          text: [
            "1. Acceptez les conditions d'utilisation locale à l'ouverture de l'application. Cela déverrouille seulement les contrôles locaux.",
            "2. Utilisez le panneau Image input / Bank photo pour placer vos propres images JPG, PNG ou WebP dans l'application, ou utilisez la galerie d'exemples pour un test rapide. Vous pouvez aussi choisir plusieurs images ou un dossier complet.",
            "3. Cliquez sur Run Burrow Detection en haut du panneau Visual workspace. Le compteur affiche les boîtes finales post-NMS. NMS veut dire suppression non maximale: les boîtes qui se chevauchent comme des doublons sont filtrées dans la même image. L'état des masques est rapporté séparément; un masque qui échoue ne devrait pas effacer une boîte valide.",
            "4. Inspectez visuellement la superposition. Vérifiez si les boîtes couvrent de vrais trous de terriers, si les centroïdes sont dans ces boîtes et si des faux positifs ou des terriers manqués apparaissent.",
            "5. Utilisez Télécharger le CSV courant ou Télécharger le PNG courant pour l'image affichée. Utilisez Télécharger le CSV de session ou Télécharger tous les PNG de session si vous voulez des fichiers pour toutes les images analysées dans la session courante. Le bouton Instructions exporte ce document.",
          ],
        },
        {
          heading: "Interpréter les résultats",
          text: [
            "Le seuil de confiance contrôle la sévérité du détecteur. Un seuil plus bas augmente souvent le rappel, mais peut ajouter des faux positifs. Un seuil plus élevé réduit souvent les faux positifs, mais peut manquer des terriers faibles ou partiellement cachés.",
            "Le seuil IoU NMS contrôle la suppression des doublons entre boîtes qui se chevauchent. Si le même terrier est compté deux fois, un réglage plus bas ou modéré peut aider. Si des terriers proches sont supprimés, il faut inspecter l'image avant de changer le seuil.",
            "Le seuil de masque contrôle le masque binaire extrait du prototype de segmentation. Le compte d'objets est centré sur les boîtes, parce que les boîtes sont le signal ONNX le plus stable. Les masques donnent une preuve visuelle et spatiale, mais ils ne doivent pas supprimer les détections.",
            "Les changements de seuils sont surtout diagnostiques et ne remplacent pas un réentraînement ou une nouvelle validation du modèle pour un autre domaine d'images. Le comportement dépend encore fortement des données de fine-tuning et de l'étape de transfer learning utilisée pour construire le modèle.",
          ],
        },
        {
          heading: "Schéma d'export",
          text: [
            "Les exports CSV courants sont liés à l'image affichée. Les exports CSV de session incluent toutes les images analysées dans la session courante et produisent aussi un tableau résumé au niveau image.",
            "Les lignes CSV au niveau détection incluent: id, confidence, x1, y1, x2, y2, width, height, centroid_x, centroid_y, mask_available, mask_status, mask_area_px et count_source.",
            "L'export JSON garde les objets de détection complets pour l'image courante, incluant les coordonnées de boîte, le centroïde, la confiance, le nom de classe, l'état du masque, l'aire du masque et le même count_source utilisé par le compteur visuel.",
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
            "BurrowScope es una aplicación local en el navegador para revisar madrigueras de golondrina ribereña en fotos de barrancos, bancos o paredes de tierra. Usa un modelo YOLO11 de segmentación afinado con transfer learning y luego exportado a ONNX para inferencia local. El resultado práctico es por imagen: cajas post-NMS, máscaras, centroides, valores de confianza y archivos exportables.",
            "La app está pensada para investigación sin fines de lucro, conservación, revisión de campo y apoyo al etiquetado o inspección. No reemplaza el criterio biológico; ayuda a revisar rápido, pero los resultados deben ser vistos por una persona antes de usarlos en reportes.",
          ],
        },
        {
          heading: "Configuración actual de inferencia",
          text: [
            `Motor de detección: ONNX local. Archivo del modelo: ${settings.modelName}. Estado mostrado por la app: ${settings.modelStatus}.`,
            "El modelo corre en el navegador usando ONNX Runtime Web. El archivo ONNX es la forma de inferencia exportada del modelo YOLO de segmentación ya entrenado. Las imágenes y los resultados de inferencia se quedan en la máquina local, a menos que el usuario los exporte y los comparta manualmente.",
            `Umbral de confianza: ${settings.confidence}. Umbral IoU NMS: ${settings.iou}. Umbral de máscara: ${settings.mask}. Máximo de detecciones: ${settings.maxDetections}.`,
            "Los valores por defecto están pensados para mantenerse fijos en la mayoría de los casos, pero pueden servir para troubleshooting si alguien usa imágenes muy distintas a las del entrenamiento. Si existe una versión más nueva del modelo de inferencia, use Load Local .onnx Model File en Model settings y vuelva a correr la misma imagen para comparar el comportamiento.",
          ],
        },
        {
          heading: "Cómo correr una imagen o un lote pequeño",
          text: [
            "1. Acepte las condiciones de uso local cuando se abre la aplicación. Esto solo habilita los controles locales.",
            "2. Use el panel Image input / Bank photo para colocar sus propias imágenes JPG, PNG o WebP en la aplicación, o use la galería de ejemplos para una prueba rápida. También puede seleccionar varias imágenes o una carpeta completa.",
            "3. Presione Run Burrow Detection en la parte superior del panel Visual workspace. El contador muestra las cajas finales post-NMS. NMS significa supresión no máxima: cajas muy traslapadas, parecidas a duplicados, se filtran dentro de la misma imagen. El estado de las máscaras se reporta aparte, así que una máscara fallida no debe borrar una detección válida por caja.",
            "4. Revise visualmente el overlay. Confirme que las cajas estén sobre aberturas reales, que los centroides queden dentro de las cajas y que no haya patrones obvios de falsos positivos o burrows faltantes.",
            "5. Use Download Current CSV o Download Current PNG para la imagen que está en pantalla. Use Download Session CSV o Download All Session PNGs cuando quiera archivos para todas las imágenes analizadas en la sesión actual. El botón Instructions exporta este documento.",
          ],
        },
        {
          heading: "Cómo interpretar los resultados",
          text: [
            "El umbral de confianza controla qué tan estricto es el detector. Un valor más bajo normalmente aumenta el recall, pero puede meter falsos positivos. Un valor más alto reduce ruido, pero puede perder madrigueras pequeñas, borrosas u ocultas.",
            "IoU NMS controla la supresión de cajas duplicadas que se traslapan. Si una misma madriguera aparece contada dos veces, un ajuste más bajo o moderado puede ayudar. Si madrigueras cercanas se eliminan, revise la imagen antes de cambiar el valor.",
            "El umbral de máscara controla la máscara binaria que sale del prototipo de segmentación. El conteo está centrado en cajas porque las cajas son la señal ONNX más estable. Las máscaras ayudan visual y espacialmente, pero no deben eliminar detecciones válidas.",
            "Los cambios de umbral son sobre todo diagnósticos y no reemplazan reentrenar o volver a validar el modelo para un dominio nuevo de imágenes. El comportamiento sigue dependiendo bastante de los datos usados en el fine-tuning y de la etapa de transfer learning con la que se construyó el modelo.",
          ],
        },
        {
          heading: "Esquema de exportación",
          text: [
            "Los CSV de imagen actual están ligados a la imagen en pantalla. Los CSV de sesión incluyen todas las imágenes analizadas en la sesión actual y además generan una tabla resumen por imagen.",
            "Las filas CSV a nivel de detección incluyen: id, confidence, x1, y1, x2, y2, width, height, centroid_x, centroid_y, mask_available, mask_status, mask_area_px y count_source.",
            "El JSON conserva los objetos completos de detección para la imagen actual, incluyendo coordenadas de caja, centroide, confianza, nombre de clase, estado de máscara, área de máscara y el mismo count_source usado por el conteo visual.",
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

function getPdfLogoRenderSize(logo) {
  const maxHeight = logo.maxHeight || logo.height || 10;
  const aspectRatio = (logo.width || 1) / Math.max(logo.height || 1, 1);
  return {
    width: maxHeight * aspectRatio,
    height: maxHeight,
  };
}

function drawPdfLogoRow(doc, logos, pageWidth, margin, y) {
  if (!logos.length) return y;
  const gap = 5;
  const renderLogos = logos.map((logo) => ({
    ...logo,
    render: getPdfLogoRenderSize(logo),
  }));
  const totalWidth = renderLogos.reduce((sum, logo) => sum + logo.render.width, 0) + (gap * Math.max(renderLogos.length - 1, 0));
  let x = margin;
  const availableWidth = pageWidth - (margin * 2) - 28;
  if (totalWidth < availableWidth) {
    x += (availableWidth - totalWidth) / 2;
  }

  renderLogos.forEach((logo) => {
    try {
      doc.addImage(logo.dataUrl, "PNG", x, y, logo.render.width, logo.render.height, undefined, "FAST");
    } catch (err) {
      // Skip silently and keep layout moving.
    }
    x += logo.render.width + gap;
  });

  return y + Math.max(...renderLogos.map((logo) => logo.render.height));
}

function addPdfHeader(doc, logos, language, pageWidth, margin) {
  const topRow = logos.filter((logo) => (logo.row || 0) === 0);
  const bottomRow = logos.filter((logo) => (logo.row || 0) === 1);
  const topY = 9;
  const firstRowBottom = drawPdfLogoRow(doc, topRow, pageWidth, margin, topY);
  const secondRowBottom = drawPdfLogoRow(doc, bottomRow, pageWidth, margin, firstRowBottom + 3);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(8);
  doc.setTextColor(37, 99, 235);
  doc.text(language, pageWidth - margin, 12, { align: "right" });
  doc.setDrawColor(219, 227, 234);
  doc.line(margin, secondRowBottom + 5, pageWidth - margin, secondRowBottom + 5);
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
  return 55;
}

function addInstructionLanguage(doc, section, logos, isFirstPage) {
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin = 16;
  const textWidth = pageWidth - margin * 2;
  if (!isFirstPage) doc.addPage();
  addPdfHeader(doc, logos, section.language, pageWidth, margin);

  let y = 55;
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
  qs("folderInput")?.addEventListener("change", handleFolderUpload);
  qs("folderSelectBtn")?.addEventListener("click", () => qs("folderInput")?.click());
  qs("openSamplesBtn")?.addEventListener("click", openSampleModal);
  qs("closeSamplesBtn")?.addEventListener("click", closeSampleModal);
  qs("glossaryBtn")?.addEventListener("click", openGlossaryModal);
  qs("closeGlossaryBtn")?.addEventListener("click", closeGlossaryModal);
  qs("selectAllSamplesBtn")?.addEventListener("click", selectAllSamples);
  qs("clearSamplesBtn")?.addEventListener("click", clearSampleSelection);
  qs("runSelectedSamplesBtn")?.addEventListener("click", runSelectedSamples);
  qs("runSelectedSamplesModalBtn")?.addEventListener("click", runSelectedSamples);
  qs("sampleModal")?.addEventListener("click", (event) => {
    if (event.target === qs("sampleModal")) closeSampleModal();
  });
  qs("glossaryModal")?.addEventListener("click", (event) => {
    if (event.target === qs("glossaryModal")) closeGlossaryModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeSampleModal();
      closeGlossaryModal();
      if (zoomModeEnabled) setZoomMode(false);
    }
  });
  qs("runBtn").addEventListener("click", runInference);
  qs("toggleOverlayBtn")?.addEventListener("click", () => {
    overlayHighlightsVisible = !overlayHighlightsVisible;
    updateOverlayControls();
  });
  qs("overlayRevealSlider")?.addEventListener("input", (event) => {
    overlayRevealPercent = Number(event.target.value);
    updateOverlayControls();
  });
  qs("zoomToggleBtn")?.addEventListener("click", () => setZoomMode(!zoomModeEnabled));
  qs("zoomResetBtn")?.addEventListener("click", resetViewerZoom);
  qs("prevAnalysedBtn")?.addEventListener("click", () => {
    if (currentAnalysisIndex > 0) showAnalysisAt(currentAnalysisIndex - 1).catch((err) => logLine(`Could not open previous analysed image: ${err.message}`));
  });
  qs("nextAnalysedBtn")?.addEventListener("click", () => {
    if (currentAnalysisIndex >= 0 && currentAnalysisIndex < sessionAnalyses.length - 1) showAnalysisAt(currentAnalysisIndex + 1).catch((err) => logLine(`Could not open next analysed image: ${err.message}`));
  });
  qs("instructionsBtn")?.addEventListener("click", exportInstructionsPdf);
  qs("exportCurrentCsvBtn")?.addEventListener("click", exportCurrentCsv);
  qs("exportSessionCsvBtn")?.addEventListener("click", exportSessionCsv);
  qs("exportJsonBtn")?.addEventListener("click", exportJson);
  qs("exportCurrentPngBtn")?.addEventListener("click", exportCurrentPng);
  qs("exportSessionPngBtn")?.addEventListener("click", () => {
    exportSessionPngs().catch((err) => {
      logLine(`Session PNG export failed: ${err.message}`);
      console.error(err);
    });
  });
  const viewerFrame = getViewerFrame();
  viewerFrame?.addEventListener("wheel", handleViewerWheel, { passive: false });
  viewerFrame?.addEventListener("pointerdown", handleViewerPointerDown);
  viewerFrame?.addEventListener("pointermove", handleViewerPointerMove);
  viewerFrame?.addEventListener("pointerup", stopViewerPointerInteraction);
  viewerFrame?.addEventListener("pointercancel", stopViewerPointerInteraction);
  viewerFrame?.addEventListener("pointerleave", stopViewerPointerInteraction);
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
