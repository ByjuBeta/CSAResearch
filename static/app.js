const predictButton = document.querySelector("#predictButton");
const uploadInput = document.querySelector("#uploadInput");
const preview = document.querySelector("#preview");
const uploadModeButton = document.querySelector("#uploadModeButton");
const cameraModeButton = document.querySelector("#cameraModeButton");
const drawModeButton = document.querySelector("#drawModeButton");
const uploadPanel = document.querySelector("#uploadPanel");
const cameraPanel = document.querySelector("#cameraPanel");
const drawPanel = document.querySelector("#drawPanel");
const startCameraButton = document.querySelector("#startCameraButton");
const captureButton = document.querySelector("#captureButton");
const stopCameraButton = document.querySelector("#stopCameraButton");
const cameraVideo = document.querySelector("#cameraVideo");
const drawCanvas = document.querySelector("#drawCanvas");
const clearDrawingButton = document.querySelector("#clearDrawingButton");
const labelResult = document.querySelector("#labelResult");
const confidenceResult = document.querySelector("#confidenceResult");
const probabilityList = document.querySelector("#probabilityList");
const drawContext = drawCanvas.getContext("2d");

let selectedImage = "";
let cameraStream = null;
let isDrawing = false;
let lastPoint = null;

function renderProbabilities(probabilities) {
  probabilityList.innerHTML = "";
  probabilities.forEach((item, index) => {
    const percent = item.probability * 100;
    const row = document.createElement("div");
    row.className = `probability-row${index === 0 ? " is-top" : ""}`;
    row.innerHTML = `
      <strong>${item.label}</strong>
      <span class="bar-track"><span class="bar-fill" style="width: ${percent}%"></span></span>
      <span>${percent.toFixed(1)}%</span>
    `;
    probabilityList.append(row);
  });
}

function resetResults() {
  labelResult.textContent = "?";
  confidenceResult.textContent = "Confidence: -";
  probabilityList.innerHTML = "";
}

function setSelectedImage(dataUrl) {
  selectedImage = dataUrl;
  predictButton.disabled = false;
  resetResults();
}

function setMode(mode) {
  uploadPanel.hidden = mode !== "upload";
  cameraPanel.hidden = mode !== "camera";
  drawPanel.hidden = mode !== "draw";
  uploadModeButton.classList.toggle("is-active", mode === "upload");
  cameraModeButton.classList.toggle("is-active", mode === "camera");
  drawModeButton.classList.toggle("is-active", mode === "draw");
}

async function predict() {
  if (!selectedImage) return;

  predictButton.disabled = true;
  predictButton.textContent = "Classifying";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: selectedImage }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Classification failed.");

    labelResult.textContent = result.label;
    confidenceResult.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
    renderProbabilities(result.probabilities);
  } catch (error) {
    confidenceResult.textContent = error.message;
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Classify";
  }
}

function loadUploadedImage(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = () => {
    setSelectedImage(reader.result);
    preview.src = reader.result;
    preview.style.display = "block";
  };
  reader.readAsDataURL(file);
}

async function startCamera() {
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraVideo.srcObject = cameraStream;
    captureButton.disabled = false;
    stopCameraButton.disabled = false;
    startCameraButton.disabled = true;
    confidenceResult.textContent = "Camera started. Capture a frame to classify.";
  } catch (error) {
    confidenceResult.textContent = `Camera error: ${error.message}`;
  }
}

function stopCamera() {
  if (!cameraStream) return;
  cameraStream.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraVideo.srcObject = null;
  captureButton.disabled = true;
  stopCameraButton.disabled = true;
  startCameraButton.disabled = false;
}

function captureFrame() {
  const canvas = document.createElement("canvas");
  canvas.width = cameraVideo.videoWidth || 320;
  canvas.height = cameraVideo.videoHeight || 240;
  const context = canvas.getContext("2d");
  context.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
  setSelectedImage(canvas.toDataURL("image/png"));
}

function resetDrawing(selectBlank = true) {
  drawContext.fillStyle = "#ffffff";
  drawContext.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawContext.lineCap = "round";
  drawContext.lineJoin = "round";
  drawContext.lineWidth = 14;
  drawContext.strokeStyle = "#000000";
  if (selectBlank) {
    setSelectedImage(drawCanvas.toDataURL("image/png"));
  }
}

function drawPoint(event) {
  const rect = drawCanvas.getBoundingClientRect();
  const pointer = event.touches ? event.touches[0] : event;
  return {
    x: ((pointer.clientX - rect.left) / rect.width) * drawCanvas.width,
    y: ((pointer.clientY - rect.top) / rect.height) * drawCanvas.height,
  };
}

function startDrawing(event) {
  event.preventDefault();
  isDrawing = true;
  lastPoint = drawPoint(event);
}

function draw(event) {
  if (!isDrawing) return;
  event.preventDefault();
  const point = drawPoint(event);
  drawContext.beginPath();
  drawContext.moveTo(lastPoint.x, lastPoint.y);
  drawContext.lineTo(point.x, point.y);
  drawContext.stroke();
  lastPoint = point;
}

function finishDrawing() {
  if (!isDrawing) return;
  isDrawing = false;
  lastPoint = null;
  setSelectedImage(drawCanvas.toDataURL("image/png"));
}

uploadModeButton.addEventListener("click", () => setMode("upload"));
cameraModeButton.addEventListener("click", () => setMode("camera"));
drawModeButton.addEventListener("click", () => setMode("draw"));
predictButton.addEventListener("click", predict);
uploadInput.addEventListener("change", loadUploadedImage);
startCameraButton.addEventListener("click", startCamera);
captureButton.addEventListener("click", captureFrame);
stopCameraButton.addEventListener("click", stopCamera);
clearDrawingButton.addEventListener("click", resetDrawing);
drawCanvas.addEventListener("mousedown", startDrawing);
drawCanvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", finishDrawing);
drawCanvas.addEventListener("touchstart", startDrawing, { passive: false });
drawCanvas.addEventListener("touchmove", draw, { passive: false });
window.addEventListener("touchend", finishDrawing);

resetDrawing(false);
setMode("upload");
