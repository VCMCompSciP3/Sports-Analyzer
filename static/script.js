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
    if (!file) {
        alert("Please upload a video first.");
        return;
    }

    output.textContent = "Analyzing video... please wait.";

    const form = new FormData();
    form.append("video", file);

    const res = await fetch("/analyze", {
        method: "POST",
        body: form
    });

    const data = await res.json();
    output.textContent = JSON.stringify(data, null, 2);
};
