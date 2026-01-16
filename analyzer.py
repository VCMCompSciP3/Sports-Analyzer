import cv2
import google.genai as genai
from google.genai.types import Content, Part
import base64
import json

client = genai.Client(api_key="AIzaSyBX1pbt2-NSAdfhl0n344hoFGtaqdy3o2o")

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
    Extract the following if visible:
    - sport
    - league
    - teams
    - players
    - scoreboard_text
    - quarter_or_period
    - time_remaining
    - event_type
    - game_number
    - arena
    - approximate_date
    Return JSON only.
    """

    response = client.models.generate_content(
        model="gemini-2.0-pro",
        contents=[
            Content(
                role="user",
                parts=[
                    Part.text(prompt),
                    Part.inline_data(
                        mime_type="image/jpeg",
                        data=image_bytes
                    )
                ]
            )
        ]
    )

    try:
        return json.loads(response.text)
    except Exception as e:
        return {
            "error": f"Invalid JSON: {e}",
            "raw": response.text
        }


def analyze_video(path):
    frames = extract_frames(path)
    results = []

    for f in frames[:30]:  # limit for speed
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
