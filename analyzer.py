import cv2
import json
from google import genai

client = genai.Client(api_key="AIzaSyAyceuBbLUUxPMIm4q47OE5ZYWbH3pCf-c")


# ---------------------------------------------------------
# FRAME EXTRACTION
# ---------------------------------------------------------
def extract_frames(path, sample_rate=10):
    """
    Extract frames every `sample_rate` frames.
    Returns a list of JPEG byte arrays.
    """
    vid = cv2.VideoCapture(path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        if frame_count % sample_rate == 0:
            success, buf = cv2.imencode(".jpg", frame)
            if success:
                frames.append(buf.tobytes())

        frame_count += 1

    vid.release()
    return frames


# ---------------------------------------------------------
# GEMINI FRAME ANALYSIS
# ---------------------------------------------------------
def analyze_frame(image_bytes):
    print("Frame size:", len(image_bytes))

    prompt = """
    You are analyzing a sports broadcast frame.
    If you are not sure about something, make your best guess.
    Return JSON with these fields, even if uncertain:
    sport, league, teams, players, scoreboard_text,
    event_type, game_number, approximate_date.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {"text": prompt},
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_bytes
                }
            }
        ]
    )

    print("RAW RESPONSE:", response.text)

    try:
        return json.loads(response.text)
    except Exception as e:
        return {
            "error": f"Invalid JSON: {e}",
            "raw": response.text
        }


# ---------------------------------------------------------
# AGGREGATION HELPERS
# ---------------------------------------------------------
def safe_set(summary, key, value):
    """
    Only set a field if the value is meaningful.
    """
    if value and value not in ["unknown", "none", "null", [], {}, "N/A"]:
        summary[key] = value


# ---------------------------------------------------------
# AGGREGATE RESULTS ACROSS ALL FRAMES
# ---------------------------------------------------------
def aggregate_results(results):
    summary = {
        "sport": None,
        "league": None,
        "teams": set(),
        "players": set(),
        "event_type": None,
        "game_number": None,
        "approximate_date": None,
        "frames_analyzed": len(results)
    }

    for r in results:
        if not isinstance(r, dict):
            continue

        # Simple fields
        safe_set(summary, "sport", r.get("sport"))
        safe_set(summary, "league", r.get("league"))
        safe_set(summary, "event_type", r.get("event_type"))
        safe_set(summary, "game_number", r.get("game_number"))
        safe_set(summary, "approximate_date", r.get("approximate_date"))

        # Teams
        teams = r.get("teams")
        if teams and isinstance(teams, list):
            for t in teams:
                if t and t not in ["unknown", "none", "null"]:
                    summary["teams"].add(str(t))

        # Players
        players = r.get("players")
        if players and isinstance(players, list):
            for p in players:
                if p and p not in ["unknown", "none", "null"]:
                    summary["players"].add(str(p))

    # Convert sets to lists
    summary["teams"] = list(summary["teams"])
    summary["players"] = list(summary["players"])

    return summary


# ---------------------------------------------------------
# MAIN VIDEO ANALYSIS PIPELINE
# ---------------------------------------------------------
def analyze_video(path):
    frames = extract_frames(path, sample_rate=10)
    results = []

    for frame in frames:
        results.append(analyze_frame(frame))

    final_summary = aggregate_results(results)
    return final_summary
