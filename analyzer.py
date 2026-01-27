import cv2
import json
from google import genai

client = genai.Client(api_key="AIzaSyCpNsoSY5MPREYz5cIfi5snen0Q25Bnlew")


# ---------------------------------------------------------
# FRAME EXTRACTION (1 frame every ~2 seconds)
# ---------------------------------------------------------
def extract_frames(path, sample_rate=60):
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
# BATCHED GEMINI ANALYSIS (10 frames per request)
# ---------------------------------------------------------
def analyze_batch(frames):
    """
    Analyze up to 10 frames in a single Gemini request.
    """
    prompt = """
    You are analyzing sports broadcast frames.
    For EACH frame, return a JSON object with:
    sport, league, teams, players, scoreboard_text,
    event_type, game_number, approximate_date.
    If unsure, make your best guess.
    Return a JSON list with one object per frame.
    """

    contents = [{"text": prompt}]

    for frame in frames:
        contents.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": frame
            }
        })

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents
    )

    try:
        return json.loads(response.text)
    except Exception as e:
        return [{"error": f"Invalid JSON: {e}", "raw": response.text}]


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
        "frames_analyzed": 0
    }

    for r in results:
        summary["frames_analyzed"] += 1

        if not isinstance(r, dict):
            continue

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

    summary["teams"] = list(summary["teams"])
    summary["players"] = list(summary["players"])

    return summary


# ---------------------------------------------------------
# MAIN VIDEO ANALYSIS PIPELINE (FAST)
# ---------------------------------------------------------
def analyze_video(path):
    frames = extract_frames(path, sample_rate=60)  # 1 frame every ~2 seconds
    results = []

    # Process in batches of 10 frames
    for i in range(0, len(frames), 10):
        batch = frames[i:i+10]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)

        # EARLY STOPPING: once we know the sport, league, and both teams
        temp_summary = aggregate_results(results)
        if (
            temp_summary["sport"] and
            temp_summary["league"] and
            len(temp_summary["teams"]) >= 2
        ):
            break

    final_summary = aggregate_results(results)
    return final_summary
