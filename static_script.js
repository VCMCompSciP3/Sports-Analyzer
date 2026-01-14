const videoInput = document.getElementById("videoInput");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const output = document.getElementById("output");

videoInput.onchange = () => {
    const file = videoInput.files[0];
    preview.src = URL.createObjectURL(file);
};

analyzeBtn.onclick = async () => {
    const file = videoInput.files[0];
    if (!file) return alert("Upload a video first");

    const form = new FormData();
    form.append("video", file);

    output.textContent = "Analyzing...";

    const res = await fetch("/analyze", {
        method: "POST",
        body: form
    });

    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
};
