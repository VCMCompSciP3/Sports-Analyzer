import cv2
from google import genai
import base64
import json

client = genai.Client(api_key="YOUR_API_KEY")

def extract_frames(path, interval=30):
    vid = cv2.VideoCapture(path)
    frames = []
    i = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if i % interval == 0:
            _, buf = cv2.imencode(".jpg", frame)
            frames.append(buf.tobytes())
        i += 1
    return frames

def analyze_frame(image_bytes):
    prompt = """
    You are analyzing a sports broadcast frame.
    Extract:
    - sport
    - league
    - teams
    - event_type
    - game_number
    - approximate_date
    - reasoning
    Return JSON only.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, image_bytes]
    )

    try:
        return json.loads(response.text)
    except:
        return {"error": "Invalid JSON returned"}

def analyze_video(path):
    frames = extract_frames(path)
    results = []

    for f in frames[:10]:  # limit for speed
        results.append(analyze_frame(f))

    # Simple aggregation
    summary = {
        "sport": None,
        "league": None,
        "teams": None,
        "event_type": None,
        "game_number": None,
        "approximate_date": None,
        "frames_analyzed": len(frames)
    }

    # Pick first non-null values
    for r in results:
        for key in summary:
            if key in r and summary[key] is None:
                summary[key] = r[key]

    return summary
