from flask import Flask, render_template, request, jsonify
from analyzer import analyze_video

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["video"]
    path = "uploaded_video.mp4"
    file.save(path)

    result = analyze_video(path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
