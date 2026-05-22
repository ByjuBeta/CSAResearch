const canvas = document.querySelector("#digitCanvas");
const ctx = canvas.getContext("2d");
const clearButton = document.querySelector("#clearButton");
const predictButton = document.querySelector("#predictButton");
const uploadInput = document.querySelector("#uploadInput");
const digitResult = document.querySelector("#digitResult");
const confidenceResult = document.querySelector("#confidenceResult");
const probabilityList = document.querySelector("#probabilityList");

let isDrawing = false;
let lastPoint = null;

function resetCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = 18;
  ctx.strokeStyle = "#000000";
  digitResult.textContent = "?";
  confidenceResult.textContent = "Confidence: -";
  renderProbabilities(Array(10).fill(0), -1);
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const pointer = event.touches ? event.touches[0] : event;
  return {
    x: ((pointer.clientX - rect.left) / rect.width) * canvas.width,
    y: ((pointer.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function drawDot(point) {
  ctx.beginPath();
  ctx.arc(point.x, point.y, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#000000";
  ctx.fill();
}

function startDrawing(event) {
  event.preventDefault();
  isDrawing = true;
  lastPoint = canvasPoint(event);
  drawDot(lastPoint);
}

function draw(event) {
  if (!isDrawing) return;
  event.preventDefault();
  const point = canvasPoint(event);
  ctx.beginPath();
  ctx.moveTo(lastPoint.x, lastPoint.y);
  ctx.lineTo(point.x, point.y);
  ctx.stroke();
  drawDot(point);
  lastPoint = point;
}

function stopDrawing() {
  isDrawing = false;
  lastPoint = null;
}

function renderProbabilities(probabilities, winner) {
  probabilityList.innerHTML = "";
  probabilities.forEach((probability, digit) => {
    const percent = probability * 100;
    const row = document.createElement("div");
    row.className = `probability-row${digit === winner ? " is-winner" : ""}`;
    row.innerHTML = `
      <strong>${digit}</strong>
      <span class="bar-track"><span class="bar-fill" style="width: ${percent}%"></span></span>
      <span>${percent.toFixed(1)}%</span>
    `;
    probabilityList.append(row);
  });
}

async function predict() {
  predictButton.disabled = true;
  predictButton.textContent = "Predicting";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: canvas.toDataURL("image/png") }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed.");

    digitResult.textContent = result.digit;
    confidenceResult.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
    renderProbabilities(result.probabilities, result.digit);
  } catch (error) {
    confidenceResult.textContent = error.message;
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Predict";
  }
}

function loadUploadedImage(event) {
  const file = event.target.files[0];
  if (!file) return;

  const image = new Image();
  image.onload = () => {
    resetCanvas();
    const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
    const width = image.width * scale;
    const height = image.height * scale;
    const x = (canvas.width - width) / 2;
    const y = (canvas.height - height) / 2;
    ctx.drawImage(image, x, y, width, height);
  };
  image.src = URL.createObjectURL(file);
}

canvas.addEventListener("mousedown", startDrawing);
canvas.addEventListener("mousemove", draw);
window.addEventListener("mouseup", stopDrawing);
canvas.addEventListener("touchstart", startDrawing, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
window.addEventListener("touchend", stopDrawing);
clearButton.addEventListener("click", resetCanvas);
predictButton.addEventListener("click", predict);
uploadInput.addEventListener("change", loadUploadedImage);

resetCanvas();
