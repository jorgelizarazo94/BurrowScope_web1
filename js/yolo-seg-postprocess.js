const DEFAULT_CLASS_NAMES = { 0: "burrow" };

let lastDebug = null;

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function isFiniteNumber(value) {
  return Number.isFinite(Number(value));
}

function confidenceValue(raw) {
  const value = Number(raw);
  if (!Number.isFinite(value)) return 0;
  if (value < 0 || value > 1) return sigmoid(value);
  return value;
}

function setLastDebug(debug) {
  lastDebug = debug;
  if (window.BurrowYoloSeg) {
    window.BurrowYoloSeg.lastDebug = debug;
  }
}

function tensorDims(tensor) {
  return Array.isArray(tensor?.dims) ? tensor.dims.map(Number) : [];
}

function outputShape(tensor) {
  const dims = tensorDims(tensor);
  return dims.length ? `[${dims.join(",")}]` : "[unknown]";
}

function isPredictionTensor(tensor) {
  const dims = tensorDims(tensor);
  if (dims.length !== 3) return false;
  return Math.min(dims[1], dims[2]) >= 5;
}

function isPrototypeTensor(tensor) {
  const dims = tensorDims(tensor);
  return dims.length === 4 && dims.slice(1).every((value) => Number.isFinite(value) && value > 0);
}

function resolveOutputs(outputs, options = {}) {
  const entries = Object.entries(outputs || {}).map(([name, tensor]) => ({ name, tensor }));
  let pred = options.predName && isPredictionTensor(outputs?.[options.predName])
    ? { name: options.predName, tensor: outputs[options.predName] }
    : null;
  let proto = options.protoName && isPrototypeTensor(outputs?.[options.protoName])
    ? { name: options.protoName, tensor: outputs[options.protoName] }
    : null;

  if (!pred) pred = entries.find((entry) => isPredictionTensor(entry.tensor)) || null;
  if (!proto) proto = entries.find((entry) => entry.name !== pred?.name && isPrototypeTensor(entry.tensor)) || null;

  if (!pred) {
    throw new Error("YOLO segmentation prediction output was not found. Expected a 3D tensor such as [1,37,33600].");
  }

  return { pred, proto, entries };
}

function getPredictionLayout(tensor) {
  const dims = tensorDims(tensor);
  const dim1 = dims[1];
  const dim2 = dims[2];
  const channelsFirst = dim1 <= dim2;
  return {
    channelsFirst,
    channels: channelsFirst ? dim1 : dim2,
    anchors: channelsFirst ? dim2 : dim1,
    layoutName: channelsFirst ? "channels-first" : "anchors-first",
  };
}

function makePredictionReader(tensor, layout) {
  const data = tensor.data;
  return (channel, anchor) => {
    if (layout.channelsFirst) return data[channel * layout.anchors + anchor];
    return data[anchor * layout.channels + channel];
  };
}

function getProtoInfo(tensor) {
  if (!tensor) return null;
  const dims = tensorDims(tensor);
  return {
    data: tensor.data,
    channels: dims[1],
    height: dims[2],
    width: dims[3],
    shape: outputShape(tensor),
  };
}

function inferClassAndMaskCounts(channelCount, protoChannels, options = {}) {
  const requestedClassCount = Number(options.numClasses ?? options.nClasses);
  const fallbackMaskChannels = Number(options.maskChannels ?? 32);
  const maskChannels = Number.isFinite(protoChannels) && protoChannels > 0 ? protoChannels : fallbackMaskChannels;

  if (Number.isFinite(requestedClassCount) && requestedClassCount > 0) {
    const nClasses = Math.floor(requestedClassCount);
    return { nClasses, nMasks: Math.max(0, channelCount - 4 - nClasses) };
  }

  const inferredClasses = channelCount - 4 - maskChannels;
  if (inferredClasses >= 1) return { nClasses: inferredClasses, nMasks: maskChannels };

  return { nClasses: 1, nMasks: Math.max(0, channelCount - 5) };
}

function detectCoordinateScale(readPred, anchors, inputSize) {
  let maxCoord = 0;
  const stride = Math.max(1, Math.floor(anchors / 600));
  for (let anchor = 0; anchor < anchors; anchor += stride) {
    for (let channel = 0; channel < 4; channel += 1) {
      const value = Math.abs(Number(readPred(channel, anchor)));
      if (Number.isFinite(value)) maxCoord = Math.max(maxCoord, value);
    }
  }
  return maxCoord <= 2 ? inputSize : 1;
}

function makeBox(x1, y1, x2, y2, maxWidth = Infinity, maxHeight = Infinity) {
  const left = clamp(Math.min(x1, x2), 0, maxWidth);
  const top = clamp(Math.min(y1, y2), 0, maxHeight);
  const right = clamp(Math.max(x1, x2), 0, maxWidth);
  const bottom = clamp(Math.max(y1, y2), 0, maxHeight);
  return {
    x1: left,
    y1: top,
    x2: right,
    y2: bottom,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  };
}

function reverseLetterboxBox(modelBox, letterbox) {
  const scale = Number(letterbox?.scale) || 1;
  const padX = Number(letterbox?.padX) || 0;
  const padY = Number(letterbox?.padY) || 0;
  const originalWidth = Number(letterbox?.originalWidth) || 0;
  const originalHeight = Number(letterbox?.originalHeight) || 0;
  const x1 = (modelBox.x1 - padX) / scale;
  const y1 = (modelBox.y1 - padY) / scale;
  const x2 = (modelBox.x2 - padX) / scale;
  const y2 = (modelBox.y2 - padY) / scale;
  return makeBox(x1, y1, x2, y2, originalWidth, originalHeight);
}

function originalToModelPoint(x, y, letterbox) {
  return {
    x: x * Number(letterbox.scale || 1) + Number(letterbox.padX || 0),
    y: y * Number(letterbox.scale || 1) + Number(letterbox.padY || 0),
  };
}

function modelToProtoPoint(x, y, letterbox, proto) {
  const inputSize = Number(letterbox.inputSize) || 1;
  return {
    x: clamp(Math.floor((x / inputSize) * proto.width), 0, proto.width - 1),
    y: clamp(Math.floor((y / inputSize) * proto.height), 0, proto.height - 1),
  };
}

function boxToDisplay(box, display) {
  const originalWidth = Number(display.originalWidth) || 1;
  const originalHeight = Number(display.originalHeight) || 1;
  const width = Number(display.width) || 0;
  const height = Number(display.height) || 0;
  return {
    x1: clamp(Math.floor((box.x1 / originalWidth) * width), 0, width),
    y1: clamp(Math.floor((box.y1 / originalHeight) * height), 0, height),
    x2: clamp(Math.ceil((box.x2 / originalWidth) * width), 0, width),
    y2: clamp(Math.ceil((box.y2 / originalHeight) * height), 0, height),
  };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return inter / Math.max(areaA + areaB - inter, 1e-9);
}

function nonMaxSuppression(candidates, iouThreshold, maxDetections) {
  const kept = [];
  const limit = Math.max(1, Number(maxDetections) || 600);
  const sorted = [...candidates].sort((a, b) => b.confidence - a.confidence);

  for (const candidate of sorted) {
    let keep = true;
    for (const chosen of kept) {
      if (candidate.class_id === chosen.class_id && iou(candidate.model_box, chosen.model_box) > iouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) kept.push(candidate);
    if (kept.length >= limit) break;
  }

  return kept;
}

function roundedBox(box) {
  return {
    x1: Math.round(box.x1 * 10) / 10,
    y1: Math.round(box.y1 * 10) / 10,
    x2: Math.round(box.x2 * 10) / 10,
    y2: Math.round(box.y2 * 10) / 10,
    width: Math.round(box.width * 10) / 10,
    height: Math.round(box.height * 10) / 10,
  };
}

function buildCandidates(outputs, options = {}) {
  const { pred, proto, entries } = resolveOutputs(outputs, options);
  const predTensor = pred.tensor;
  const protoInfo = getProtoInfo(proto?.tensor);
  const layout = getPredictionLayout(predTensor);
  const readPred = makePredictionReader(predTensor, layout);
  const inputSize = Number(options.letterbox?.inputSize) || Number(options.inputSize) || 1024;
  const coordinateScale = detectCoordinateScale(readPred, layout.anchors, inputSize);
  const { nClasses, nMasks } = inferClassAndMaskCounts(layout.channels, protoInfo?.channels, options);
  const confThreshold = Number(options.confThreshold ?? 0.25);
  const candidates = [];
  const debug = {
    outputs: entries.map((entry) => ({ name: entry.name, shape: outputShape(entry.tensor) })),
    prediction_output: pred.name,
    prototype_output: proto?.name || null,
    prediction_shape: outputShape(predTensor),
    prototype_shape: proto?.tensor ? outputShape(proto.tensor) : null,
    layout: layout.layoutName,
    channels: layout.channels,
    anchors: layout.anchors,
    classes: nClasses,
    mask_coefficients: nMasks,
    input_size: inputSize,
    coordinate_scale: coordinateScale,
    confidence_threshold: confThreshold,
    first_5_boxes_before_reverse_letterbox: [],
    first_5_boxes_after_reverse_letterbox: [],
    first_5_confidences: [],
  };

  for (let anchor = 0; anchor < layout.anchors; anchor += 1) {
    const cx = Number(readPred(0, anchor)) * coordinateScale;
    const cy = Number(readPred(1, anchor)) * coordinateScale;
    const w = Number(readPred(2, anchor)) * coordinateScale;
    const h = Number(readPred(3, anchor)) * coordinateScale;
    if (![cx, cy, w, h].every(isFiniteNumber) || w <= 0 || h <= 0) continue;

    let classId = 0;
    let confidence = 0;
    for (let c = 0; c < nClasses; c += 1) {
      const score = confidenceValue(readPred(4 + c, anchor));
      if (score > confidence) {
        confidence = score;
        classId = c;
      }
    }
    if (confidence < confThreshold) continue;

    const modelBox = makeBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, inputSize, inputSize);
    if (modelBox.width <= 0 || modelBox.height <= 0) continue;

    const originalBox = reverseLetterboxBox(modelBox, options.letterbox);
    if (originalBox.width <= 0 || originalBox.height <= 0) continue;

    const coeffs = [];
    for (let m = 0; m < nMasks; m += 1) {
      const value = Number(readPred(4 + nClasses + m, anchor));
      coeffs.push(Number.isFinite(value) ? value : 0);
    }

    if (debug.first_5_confidences.length < 5) {
      debug.first_5_boxes_before_reverse_letterbox.push(roundedBox(modelBox));
      debug.first_5_boxes_after_reverse_letterbox.push(roundedBox(originalBox));
      debug.first_5_confidences.push(Math.round(confidence * 10000) / 10000);
    }

    candidates.push({
      class_id: classId,
      confidence,
      model_box: modelBox,
      box: originalBox,
      coeffs,
      proto: protoInfo,
    });
  }

  debug.raw_candidates_above_confidence = candidates.length;
  return { candidates, debug };
}

function decodeYoloSeg(outputs, options = {}) {
  if (!options.letterbox) {
    throw new Error("Letterbox metadata is required to map YOLO boxes back to the original image.");
  }

  const { candidates, debug } = buildCandidates(outputs, options);
  const kept = nonMaxSuppression(candidates, Number(options.iouThreshold ?? 0.5), options.maxDetections);
  const classNames = options.classNames || DEFAULT_CLASS_NAMES;
  const maskThreshold = Number(options.maskThreshold ?? 0.5);
  const inputSize = Number(options.letterbox.inputSize) || Number(options.inputSize) || 1024;
  const letterbox = { ...options.letterbox, inputSize };

  const detections = kept.map((candidate, index) => {
    const className = classNames[candidate.class_id] ?? classNames[String(candidate.class_id)] ?? "burrow";
    const maskInputsAvailable = Boolean(
      candidate.proto
      && candidate.coeffs.length === candidate.proto.channels
      && candidate.proto.channels > 0,
    );
    const mask = {
      available: false,
      polygon: [],
      area_px: 0,
      status: maskInputsAvailable ? "missing" : "missing",
    };
    const detection = {
      id: index + 1,
      detection_id: index + 1,
      class_id: candidate.class_id,
      class_name: className,
      confidence: candidate.confidence,
      box: { ...candidate.box },
      model_box: { ...candidate.model_box },
      centroid: {
        x: candidate.box.x1 + candidate.box.width / 2,
        y: candidate.box.y1 + candidate.box.height / 2,
      },
      mask,
      count_source: "box",
    };

    Object.defineProperties(detection, {
      _coeffs: { value: candidate.coeffs, enumerable: false, writable: true },
      _proto: { value: candidate.proto, enumerable: false, writable: true },
      _letterbox: { value: letterbox, enumerable: false, writable: true },
      _maskThreshold: { value: maskThreshold, enumerable: false, writable: true },
    });

    return detection;
  });

  debug.after_nms = detections.length;
  debug.post_nms_boxes = detections.length;
  debug.valid_masks = 0;
  debug.failed_masks = detections.length;
  debug.first_5_mask_statuses = detections.slice(0, 5).map((det) => det.mask.status);
  setLastDebug(debug);

  if (typeof options.onDebug === "function") options.onDebug(debug);
  if (typeof options.debugLogger === "function" && options.debugLogger !== options.onDebug) options.debugLogger(debug);

  return detections;
}

function setMask(det, status, areaPx = 0, polygon = [], error = null) {
  const ok = status === "ok";
  det.mask = {
    available: ok,
    polygon: ok ? polygon : [],
    area_px: ok ? Math.max(0, Math.round(areaPx)) : 0,
    status,
    error,
  };
}

function drawLabel(ctx, det, color, sx, sy, canvasWidth) {
  const label = `${det.id}: ${Number(det.confidence).toFixed(2)}`;
  ctx.font = "12px sans-serif";
  const labelWidth = Math.ceil(ctx.measureText(label).width) + 8;
  const x = clamp(sx, 0, Math.max(0, canvasWidth - labelWidth));
  const y = Math.max(0, sy - 20);

  ctx.fillStyle = "rgba(15, 23, 42, 0.84)";
  ctx.fillRect(x, y, labelWidth, 20);
  ctx.fillStyle = "white";
  ctx.fillText(label, x + 4, y + 14);
}

function drawBoxCentroidAndLabel(ctx, det, display, color) {
  const displayBox = boxToDisplay(det.box, display);
  const width = displayBox.x2 - displayBox.x1;
  const height = displayBox.y2 - displayBox.y1;
  if (width <= 0 || height <= 0) return;

  const rgb = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
  ctx.save();
  ctx.strokeStyle = rgb;
  ctx.lineWidth = 2;
  ctx.strokeRect(displayBox.x1, displayBox.y1, width, height);

  const cx = (det.centroid.x / display.originalWidth) * display.width;
  const cy = (det.centroid.y / display.originalHeight) * display.height;
  ctx.fillStyle = rgb;
  ctx.beginPath();
  ctx.arc(cx, cy, 3.8, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 1.25;
  ctx.stroke();

  drawLabel(ctx, det, color, displayBox.x1, displayBox.y1, display.width);
  ctx.restore();
}

function drawMaskForDetection(overlayCtx, det, display, color, options = {}) {
  const proto = det._proto;
  const coeffs = det._coeffs || [];
  if (!proto || coeffs.length !== proto.channels) {
    setMask(det, "missing", 0, [], "Prototype tensor or mask coefficients unavailable.");
    return;
  }

  const displayBox = boxToDisplay(det.box, display);
  const sx1 = clamp(displayBox.x1, 0, display.width);
  const sy1 = clamp(displayBox.y1, 0, display.height);
  const sx2 = clamp(displayBox.x2, 0, display.width);
  const sy2 = clamp(displayBox.y2, 0, display.height);
  const width = Math.max(0, sx2 - sx1);
  const height = Math.max(0, sy2 - sy1);
  if (width <= 0 || height <= 0) {
    setMask(det, "out_of_bounds", 0, [], "Display box has no drawable area.");
    return;
  }

  const threshold = Number(options.maskThreshold ?? det._maskThreshold ?? 0.5);
  const alpha = clamp(Number(options.maskAlpha ?? options.maskOpacity ?? 0.42), 0.05, 0.9);
  const plane = proto.width * proto.height;
  const imageData = overlayCtx.getImageData(sx1, sy1, width, height);
  const out = imageData.data;
  let filled = 0;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (let y = 0; y < height; y += 1) {
    const displayY = sy1 + y;
    const originalY = (displayY / display.height) * display.originalHeight;
    const modelY = originalToModelPoint(0, originalY, det._letterbox).y;
    const protoY = modelToProtoPoint(0, modelY, det._letterbox, proto).y;

    for (let x = 0; x < width; x += 1) {
      const displayX = sx1 + x;
      const originalX = (displayX / display.width) * display.originalWidth;
      const modelX = originalToModelPoint(originalX, 0, det._letterbox).x;
      const protoX = modelToProtoPoint(modelX, 0, det._letterbox, proto).x;
      const protoIndex = protoY * proto.width + protoX;

      let value = 0;
      for (let channel = 0; channel < proto.channels; channel += 1) {
        value += proto.data[channel * plane + protoIndex] * coeffs[channel];
      }

      if (sigmoid(value) >= threshold) {
        const index = (y * width + x) * 4;
        out[index] = color[0];
        out[index + 1] = color[1];
        out[index + 2] = color[2];
        out[index + 3] = Math.round(255 * alpha);
        filled += 1;
        minX = Math.min(minX, originalX);
        minY = Math.min(minY, originalY);
        maxX = Math.max(maxX, originalX);
        maxY = Math.max(maxY, originalY);
      }
    }
  }

  if (filled <= 0) {
    setMask(det, "empty", 0, [], null);
    return;
  }

  overlayCtx.putImageData(imageData, sx1, sy1);
  const areaScale = (display.originalWidth / display.width) * (display.originalHeight / display.height);
  const areaPx = filled * areaScale;
  const polygon = [
    [minX, minY],
    [maxX, minY],
    [maxX, maxY],
    [minX, maxY],
  ];
  setMask(det, "ok", areaPx, polygon, null);
}

function drawDetectionMasks(ctx, detections, display, options = {}) {
  const colors = [
    [20, 184, 166],
    [245, 158, 11],
    [59, 130, 246],
    [236, 72, 153],
    [132, 204, 22],
    [244, 63, 94],
  ];

  const overlay = document.createElement("canvas");
  overlay.width = display.width;
  overlay.height = display.height;
  const overlayCtx = overlay.getContext("2d");
  const maskStatuses = [];

  detections.forEach((det, index) => {
    const color = colors[index % colors.length];
    try {
      drawMaskForDetection(overlayCtx, det, display, color, options);
    } catch (err) {
      setMask(det, "decode_failed", 0, [], err?.message || String(err));
    }
    maskStatuses.push(det.mask.status);
  });

  ctx.drawImage(overlay, 0, 0);

  detections.forEach((det, index) => {
    drawBoxCentroidAndLabel(ctx, det, display, colors[index % colors.length]);
  });

  const validMasks = detections.filter((det) => det.mask.status === "ok").length;
  if (lastDebug) {
    lastDebug.valid_masks = validMasks;
    lastDebug.failed_masks = detections.length - validMasks;
    lastDebug.first_5_mask_statuses = maskStatuses.slice(0, 5);
    setLastDebug(lastDebug);
  }
}

function thresholdDiagnostics(outputs, options = {}, thresholds = [0.05, 0.1, 0.2, 0.25, 0.35, 0.5]) {
  const previousDebug = lastDebug;
  const rows = thresholds.map((threshold) => {
    try {
      const detections = decodeYoloSeg(outputs, { ...options, confThreshold: threshold, onDebug: null, debugLogger: null });
      const debug = lastDebug || {};
      return {
        confidence_threshold: threshold,
        raw_candidates_above_confidence: debug.raw_candidates_above_confidence ?? detections.length,
        post_nms_boxes: detections.length,
        valid_masks: 0,
        failed_masks: detections.length,
      };
    } catch (err) {
      return {
        confidence_threshold: threshold,
        raw_candidates_above_confidence: null,
        post_nms_boxes: null,
        valid_masks: null,
        failed_masks: null,
        error: err?.message || String(err),
      };
    }
  });
  setLastDebug(previousDebug);
  return rows;
}

window.BurrowYoloSeg = {
  decodeYoloSeg,
  drawDetectionMasks,
  thresholdDiagnostics,
  lastDebug,
};
