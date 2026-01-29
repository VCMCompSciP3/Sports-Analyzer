import cv2
import json
import re
from google.cloud import vision

# Initialize Vision API client
client = vision.ImageAnnotatorClient()


# ---------------------------------------------------------
# FRAME EXTRACTION — FIRST 10 FRAMES
# ---------------------------------------------------------
def extract_frames(path, max_frames=10):
    vid = cv2.VideoCapture(path)
    frames = []
    count = 0

    while len(frames) < max_frames:
        ret, frame = vid.read()
        if not ret:
            break

        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            frames.append(buf.tobytes())

        count += 1

    vid.release()
    return frames


# ---------------------------------------------------------
# SEND FRAME TO VISION API (OCR + LOGO DETECTION)
# ---------------------------------------------------------
def analyze_frame(image_bytes):
    image = vision.Image(content=image_bytes)

    response = client.annotate_image({
        "image": image,
        "features": [
            {"type": vision.Feature.Type.TEXT_DETECTION},
            {"type": vision.Feature.Type.LOGO_DETECTION}
        ]
    })

    text = ""
    logos = []

    if response.text_annotations:
        text = response.text_annotations[0].description

    if response.logo_annotations:
        logos = [logo.description for logo in response.logo_annotations]

    return {
        "text": text,
        "logos": logos
    }


# ---------------------------------------------------------
# PARSE TEXT TO EXTRACT SPORTS INFO
# ---------------------------------------------------------
def parse_frame_data(text, logos):
    text_lower = text.lower()

    sport = None
    league = None
    event_type = None
    game_number = None
    approximate_date = None
    teams = set()

    # SPORT
    if "nba" in text_lower or "finals" in text_lower:
        sport = "Basketball"
        league = "NBA"

    # TEAMS (simple keyword matching)
    team_keywords = {
        "warriors": "Golden State Warriors",
        "cavaliers": "Cleveland Cavaliers",
        "cavs": "Cleveland Cavaliers",
        "gsw": "Golden State Warriors",
        "cle": "Cleveland Cavaliers",
        "lakers": "Los Angeles Lakers",
        "heat": "Miami Heat",
        "bulls": "Chicago Bulls"
    }

    for key, name in team_keywords.items():
        if key in text_lower:
            teams.add(name)
        for logo in logos:
            if key in logo.lower():
                teams.add(name)

    # EVENT TYPE
    if "finals" in text_lower:
        event_type = "NBA Finals"
    if "game 7" in text_lower:
        game_number = 7
    elif "game" in text_lower:
        match = re.search(r"game\s+(\d+)", text_lower)
        if match:
            game_number = int(match.group(1))

    # DATE (very rough)
    if "2016" in text_lower:
        approximate_date = "2016"
    if "2015" in text_lower:
        approximate_date = "2015"

    return {
        "sport": sport,
        "league": league,
        "teams": list(teams),
        "event_type": event_type,
        "game_number": game_number,
        "approximate_date": approximate_date
    }


# ---------------------------------------------------------
# AGGREGATE RESULTS ACROSS FRAMES
# ---------------------------------------------------------
def aggregate_results(results):
    summary = {
        "sport": None,
        "league": None,
        "teams": set(),
        "event_type": None,
        "game_number": None,
        "approximate_date": None,
        "frames_analyzed": len(results)
    }

    for r in results:
        if r["sport"] and not summary["sport"]:
            summary["sport"] = r["sport"]

        if r["league"] and not summary["league"]:
            summary["league"] = r["league"]

        for t in r["teams"]:
            summary["teams"].add(t)

        if r["event_type"] and not summary["event_type"]:
            summary["event_type"] = r["event_type"]

        if r["game_number"] and not summary["game_number"]:
            summary["game_number"] = r["game_number"]

        if r["approximate_date"] and not summary["approximate_date"]:
            summary["approximate_date"] = r["approximate_date"]

    summary["teams"] = list(summary["teams"])
    return summary


# ---------------------------------------------------------
# MAIN VIDEO ANALYSIS PIPELINE
# ---------------------------------------------------------
def analyze_video(path):
    frames = extract_frames(path, max_frames=10)

    frame_results = []
    for frame in frames:
        raw = analyze_frame(frame)
        parsed = parse_frame_data(raw["text"], raw["logos"])
        frame_results.append(parsed)

    return aggregate_results(frame_results)
