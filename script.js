// Handle file upload
document.getElementById("uploadForm").onsubmit = async function (e) {
  e.preventDefault();
  let formData = new FormData(document.getElementById("uploadForm"));
  let response = await fetch("/predict", {
    method: "POST",
    body: formData
  });
  let result = await response.json();
  document.getElementById("result").innerText = "Predicted Digit: " + result.digit;
};

// Camera setup
const video = document.getElementById("camera");
const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

async function capture() {
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0);

  canvas.toBlob(async function (blob) {
    let formData = new FormData();
    formData.append("file", blob, "capture.png");

    let response = await fetch("/predict", {
      method: "POST",
      body: formData
    });
    let result = await response.json();
    document.getElementById("cameraResult").innerText = "Predicted Digit: " + result.digit;
  }, "image/png");
}
