function readImageFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => resolve({ img, url });
    img.onerror = () => reject(new Error(`Could not read image: ${file.name}`));
    img.src = url;
  });
}

function readImageUrl(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ img, url });
    img.onerror = () => reject(new Error(`Could not load image: ${url}`));
    img.src = url;
  });
}

function letterboxImageToTensor(img, size) {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, size, size);

  const scale = Math.min(size / img.naturalWidth, size / img.naturalHeight);
  const newW = Math.round(img.naturalWidth * scale);
  const newH = Math.round(img.naturalHeight * scale);
  const padX = Math.floor((size - newW) / 2);
  const padY = Math.floor((size - newH) / 2);

  ctx.drawImage(img, padX, padY, newW, newH);
  const imageData = ctx.getImageData(0, 0, size, size).data;
  const tensor = new Float32Array(1 * 3 * size * size);
  const plane = size * size;

  for (let i = 0; i < size * size; i += 1) {
    tensor[i] = imageData[i * 4] / 255.0;
    tensor[plane + i] = imageData[i * 4 + 1] / 255.0;
    tensor[plane * 2 + i] = imageData[i * 4 + 2] / 255.0;
  }

  return {
    tensor,
    canvas,
    info: {
      inputSize: size,
      scale,
      padX,
      padY,
      originalWidth: img.naturalWidth,
      originalHeight: img.naturalHeight,
      resizedWidth: newW,
      resizedHeight: newH,
    },
  };
}

function modelBoxToOriginal(box, info) {
  const [x1, y1, x2, y2] = box;
  const ox1 = (x1 - info.padX) / info.scale;
  const oy1 = (y1 - info.padY) / info.scale;
  const ox2 = (x2 - info.padX) / info.scale;
  const oy2 = (y2 - info.padY) / info.scale;
  return [
    Math.max(0, Math.min(info.originalWidth, ox1)),
    Math.max(0, Math.min(info.originalHeight, oy1)),
    Math.max(0, Math.min(info.originalWidth, ox2)),
    Math.max(0, Math.min(info.originalHeight, oy2)),
  ];
}

function downloadTextFile(filename, text, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function csvEscape(value) {
  const s = String(value ?? "");
  if (s.includes(",") || s.includes("\"") || s.includes("\n")) {
    return `"${s.replaceAll("\"", "\"\"")}"`;
  }
  return s;
}

window.BurrowImageUtils = {
  readImageFile,
  readImageUrl,
  letterboxImageToTensor,
  modelBoxToOriginal,
  downloadTextFile,
  csvEscape,
};
