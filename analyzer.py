import cv2
import base64
import json
from collections import Counter, defaultdict
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# ---------------------------------------------------------
# INITIALIZE VERTEX AI
# ---------------------------------------------------------
# Make sure GOOGLE_APPLICATION_CREDENTIALS is set in your environment
# and that the project/location match your GCP setup.
vertexai.init(project="compscigeminiproject", location="us-central1")
model = GenerativeModel("gemini-1.5-flash")


# ---------------------------------------------------------
# FRAME EXTRACTION
# ---------------------------------------------------------
def extract_frames(path, max_frames=6):
    vid = cv2.VideoCapture(path)
    frames = []
    total = 0

    while len(frames) < max_frames:
        ret, frame = vid.read()
        if not ret:
            break

        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            frames.append(buf.tobytes())

        total += 1

    vid.release()
    return frames


# ---------------------------------------------------------
# SINGLE FRAME ANALYSIS WITH VERTEX AI
# ---------------------------------------------------------
def analyze_frame_with_vertex(image_bytes):
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = """
You are a world-class sports broadcast analysis AI.

You are given a single frame from a sports broadcast.
Analyze the image and extract as much structured information as possible.

Return a JSON object with the following fields:

- sport
- league
- teams
- event_type
- game_number
- approximate_year
- score
- arena_or_location
- broadcaster
- series_status
- home_team
- away_team
- star_players
- notable_players_with_numbers
- additional_context

Rules:
- Infer when needed.
- Never return null.
- Always return valid JSON.
"""

    response = model.generate_content(
        [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64
                        }
                    }
                ]
            }
        ]
    )

    try:
        data = json.loads(response.text)
    except Exception:
        data = {"additional_context": response.text}

    return data



# ---------------------------------------------------------
# AGGREGATE RESULTS ACROSS FRAMES
# ---------------------------------------------------------
def aggregate_frame_results(frame_results):
    """
    Combine multiple frame-level JSON results into a single summary.
    We use majority voting / most frequent values where possible.
    """
    if not frame_results:
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
            "additional_context": "",
            "frames_analyzed": 0
        }

    # Helper to pick most common non-"unknown" value
    def most_common(values):
        filtered = [v for v in values if v and v != "unknown"]
        if not filtered:
            return "unknown"
        return Counter(filtered).most_common(1)[0][0]

    # Collect fields
    sports = []
    leagues = []
    event_types = []
    game_numbers = []
    years = []
    scores = []
    arenas = []
    broadcasters = []
    series_statuses = []
    home_teams = []
    away_teams = []
    all_teams = []
    all_star_players = []
    all_notable_players = []
    all_context = []

    for r in frame_results:
        sports.append(r.get("sport", "unknown"))
        leagues.append(r.get("league", "unknown"))
        event_types.append(r.get("event_type", "unknown"))
        game_numbers.append(r.get("game_number", "unknown"))
        years.append(r.get("approximate_year", "unknown"))
        scores.append(r.get("score", "unknown"))
        arenas.append(r.get("arena_or_location", "unknown"))
        broadcasters.append(r.get("broadcaster", "unknown"))
        series_statuses.append(r.get("series_status", "unknown"))
        home_teams.append(r.get("home_team", "unknown"))
        away_teams.append(r.get("away_team", "unknown"))

        teams = r.get("teams", [])
        if isinstance(teams, list):
            all_teams.extend(teams)

        stars = r.get("star_players", [])
        if isinstance(stars, list):
            all_star_players.extend(stars)

        notable = r.get("notable_players_with_numbers", [])
        if isinstance(notable, list):
            all_notable_players.extend(notable)

        ctx = r.get("additional_context", "")
        if ctx:
            all_context.append(ctx)

    # Deduplicate teams and players
    unique_teams = sorted(set([t for t in all_teams if t]))
    unique_star_players = sorted(set([p for p in all_star_players if p]))

    # For notable players, dedupe by (name, jersey_number, team)
    seen_notable = set()
    dedup_notable = []
    for p in all_notable_players:
        name = p.get("name", "unknown")
        num = p.get("jersey_number", "unknown")
        team = p.get("team", "unknown")
        key = (name, num, team)
        if key not in seen_notable:
            seen_notable.add(key)
            dedup_notable.append(p)

    summary = {
        "sport": most_common(sports),
        "league": most_common(leagues),
        "teams": unique_teams,
        "event_type": most_common(event_types),
        "game_number": most_common(game_numbers),
        "approximate_year": most_common(years),
        "score": most_common(scores),
        "arena_or_location": most_common(arenas),
        "broadcaster": most_common(broadcasters),
        "series_status": most_common(series_statuses),
        "home_team": most_common(home_teams),
        "away_team": most_common(away_teams),
        "star_players": unique_star_players,
        "notable_players_with_numbers": dedup_notable,
        "additional_context": " ".join(all_context),
        "frames_analyzed": len(frame_results)
    }

    return summary


# ---------------------------------------------------------
# MAIN VIDEO ANALYSIS PIPELINE
# ---------------------------------------------------------
def analyze_video(path):
    frames = extract_frames(path, max_frames=6)

    frame_results = []
    for frame in frames:
        data = analyze_frame_with_vertex(frame)
        frame_results.append(data)

    summary = aggregate_frame_results(frame_results)
    return summary
