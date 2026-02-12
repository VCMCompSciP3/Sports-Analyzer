import cv2
import base64
import json
import re
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# ---------------------------------------------------------
# INITIALIZE VERTEX AI
# ---------------------------------------------------------
vertexai.init(project="compscigeminiproject", location="us-east1")

MODEL_NAME = "gemini-2.5-pro"
model = GenerativeModel(MODEL_NAME)


# ---------------------------------------------------------
# FRAME EXTRACTION
# ---------------------------------------------------------
def extract_frames(path, max_frames=6):
    vid = cv2.VideoCapture(path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return []

    step = max(total_frames // max_frames, 1)
    frames = []

    for i in range(0, total_frames, step):
        vid.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = vid.read()
        if not ret:
            continue

        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            frames.append(buf.tobytes())

        if len(frames) >= max_frames:
            break

    vid.release()
    return frames


# ---------------------------------------------------------
# CLEAN JSON EXTRACTION
# ---------------------------------------------------------
def extract_json_from_response(text):
    """
    Gemini often returns JSON inside ```json ... ``` blocks.
    This extracts the inner JSON cleanly.
    """

    # 1. Try to extract JSON inside code fences
    fenced = re.findall(r"```json(.*?)```", text, re.DOTALL)
    if fenced:
        candidate = fenced[0].strip()
        try:
            return json.loads(candidate)
        except:
            pass

    # 2. Try to extract ANY JSON object in the text
    obj = re.findall(r"\{.*\}", text, re.DOTALL)
    if obj:
        try:
            return json.loads(obj[0])
        except:
            pass

    # 3. If all else fails, return fallback
    return {
        "sport": "unknown",
        "league": "unknown",
        "teams": [],
        "event_type": "unknown",
        "game_number": "unknown",
        "approximate_year": "unknown",
        "score": "unknown",
        "arena_or_location": "unknown",
        "broadcaster": "unknown",
        "series_status": "unknown",
        "home_team": "unknown",
        "away_team": "unknown",
        "star_players": [],
        "notable_players_with_numbers": [],
        "additional_context": text.strip()
    }


# ---------------------------------------------------------
# MULTI-FRAME ANALYSIS
# ---------------------------------------------------------
def analyze_frames_with_vertex(frames):
    if not frames:
        return {"error": "No frames extracted"}

    parts = []

    prompt = """
You are a world-class sports broadcast analysis AI.

You will receive multiple frames from the SAME sports broadcast.
Use ALL frames together to infer the most likely structured information.

Return ONLY valid JSON with the following fields:

{
  "sport": "...",
  "league": "...",
  "teams": ["...", "..."],
  "event_type": "...",
  "game_number": "...",
  "approximate_year": "...",
  "score": "...",
  "arena_or_location": "...",
  "broadcaster": "...",
  "series_status": "...",
  "home_team": "...",
  "away_team": "...",
  "star_players": ["..."],
  "notable_players_with_numbers": [
      {"name": "...", "jersey_number": "...", "team": "..."}
  ],
  "additional_context": "..."
}

Rules:
- NEVER return null.
- NEVER return empty objects.
- If unsure, infer the most likely value.
- Use jersey colors, logos, court/field design, scoreboard layout, and player silhouettes.
- ALWAYS return valid JSON.
"""

    parts.append({"text": prompt})

    # Add all frames
    for img in frames:
        b64 = base64.b64encode(img).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64
            }
        })

    # Call Gemini 2.5 Pro
    response = model.generate_content(
        [{"role": "user", "parts": parts}]
    )

    # Extract clean JSON
    return extract_json_from_response(response.text)


# ---------------------------------------------------------
# MAIN VIDEO ANALYSIS PIPELINE
# ---------------------------------------------------------
def analyze_video(path):
    frames = extract_frames(path, max_frames=6)
    result = analyze_frames_with_vertex(frames)
    result["frames_analyzed"] = len(frames)
    return result
