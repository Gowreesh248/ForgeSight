"""
zone_manager.py
----------------
Context-aware multi-camera zone layer for Securade.ai HUB / Forge Sight.

This module is additive: it does NOT touch or replace the existing YOLO
detection code in safety_app.py / securade.py. It only:

  1. Loads camera_config.json (per-camera zone + rule metadata).
  2. Draws the zone / PPE / status overlay banner on top of frames that
     have already been annotated by the existing detection pipeline.
  3. Maps each zone to a severity level.
  4. Writes/reads a structured event log (event_log.json) used by the
     dashboard for FEATURE 6 (dashboard cards) and FEATURE 7 (event log).

Nothing here changes the YOLO model, the existing policy_file format, or
the existing Telegram alert flow in securade.py -- it wraps around them.
"""

import json
import os
import time
from datetime import date, datetime

import cv2
import numpy as np

DEFAULT_CAMERA_CONFIG_PATH = "camera_config.json"
DEFAULT_EVENT_LOG_PATH = "event_log.json"

# ---------------------------------------------------------------------------
# FEATURE 2 + FEATURE 8: Zone types, display labels/colors and severity.
# Colors are in BGR (OpenCV convention) for use with cv2 drawing calls, plus
# a hex value for the HTML/Dash dashboard.
# ---------------------------------------------------------------------------
ZONE_INFO = {
    "Danger Zone": {
        "emoji": "\U0001F534",          # 🔴
        "label": "DANGER ZONE",
        "bgr": (0, 0, 255),
        "hex": "#e74c3c",
        "severity": "Critical",
    },
    "Restricted Zone": {
        "emoji": "\U0001F7E0",          # 🟠
        "label": "RESTRICTED AREA",
        "bgr": (0, 140, 255),
        "hex": "#e67e22",
        "severity": "High",
    },
    "Safe Zone": {
        "emoji": "\U0001F7E2",          # 🟢
        "label": "SAFE AREA",
        "bgr": (0, 200, 0),
        "hex": "#27ae60",
        "severity": "Medium",
    },
    "Privacy Zone": {
        "emoji": "\U0001F7E3",          # 🟣
        "label": "PRIVACY ZONE",
        "bgr": (200, 0, 160),
        "hex": "#8e44ad",
        "severity": "N/A",
    },
    "Office Zone": {
        "emoji": "\U0001F535",          # 🔵
        "label": "OFFICE",
        "bgr": (255, 140, 0),
        "hex": "#2980b9",
        "severity": "Low",
    },
}

UNKNOWN_ZONE = {
    "emoji": "\u26AA",
    "label": "UNKNOWN ZONE",
    "bgr": (200, 200, 200),
    "hex": "#95a5a6",
    "severity": "Low",
}


# ---------------------------------------------------------------------------
# FEATURE 4: Configuration loading
# ---------------------------------------------------------------------------
def load_camera_config(path=DEFAULT_CAMERA_CONFIG_PATH):
    """Load camera_config.json. Returns {} if the file is missing so the
    rest of the app can still run using only the legacy policy_file logic."""
    if not os.path.exists(path):
        print(f"[zone_manager] Warning: {path} not found, running without zone context.")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def resolve_camera_key(index, camera_config):
    """Map a camera's position in config['sources'] (0-based) to its
    camera_config.json key, e.g. index 0 -> 'Camera1'. No hardcoded per
    camera logic -- purely positional/dynamic."""
    key = f"Camera{index + 1}"
    return key if key in camera_config else None


def get_camera_profile(index, camera_config):
    """Return the profile dict for a camera, or a safe default if the
    camera has no entry in camera_config.json."""
    key = resolve_camera_key(index, camera_config)
    if key is None:
        return "CameraX", {
            "name": "Unconfigured Camera",
            "zone": "Safe Zone",
            "helmet": False, "vest": False, "shoes": False, "gloves": False,
            "restricted": False, "privacy": False, "fire_smoke": False,
        }
    return key, camera_config[key]


def get_zone_info(zone_name):
    return ZONE_INFO.get(zone_name, UNKNOWN_ZONE)


def get_required_ppe(profile):
    """FEATURE 3: turn the boolean flags in a camera profile into a
    human-readable list of required PPE for the overlay/dashboard."""
    ppe = []
    if profile.get("helmet"):
        ppe.append("Helmet")
    if profile.get("vest"):
        ppe.append("Vest")
    if profile.get("shoes"):
        ppe.append("Safety Shoes")
    if profile.get("gloves"):
        ppe.append("Gloves")
    return ppe


def is_privacy_zone(profile):
    return bool(profile.get("privacy")) or profile.get("zone") == "Privacy Zone"


def severity_for(profile):
    """FEATURE 8: severity is derived from the zone type."""
    return get_zone_info(profile.get("zone", "")).get("severity", "Low")


# ---------------------------------------------------------------------------
# FEATURE 5: Video overlay
# ---------------------------------------------------------------------------
def _put_banner(img, lines, origin, bg_color, text_color=(255, 255, 255), scale=0.55):
    """Draw a small filled banner with one or more lines of text starting
    at `origin` (top-left corner)."""
    x, y = origin
    line_h = 22
    max_w = 0
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        max_w = max(max_w, w)
    box_h = line_h * len(lines) + 10
    cv2.rectangle(img, (x, y), (x + max_w + 16, y + box_h), bg_color, -1, cv2.LINE_AA)
    for i, line in enumerate(lines):
        ty = y + 18 + i * line_h
        cv2.putText(img, line, (x + 8, ty), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    text_color, 1, cv2.LINE_AA)
    return box_h


def draw_zone_overlay(img, cam_key, profile, status="SAFE", violation_text=None):
    """FEATURE 5: draws camera name, zone, required PPE and status on the
    frame. This runs *after* the existing detection boxes have already
    been drawn by safety_app.py, so nothing there is disturbed."""
    zone_name = profile.get("zone", "")
    zi = get_zone_info(zone_name)
    cam_display_name = profile.get("name", cam_key)

    header_lines = [f"{cam_key} - {cam_display_name}", f"{zi['emoji']} {zi['label']}"]
    _put_banner(img, header_lines, (10, 10), zi["bgr"])

    if is_privacy_zone(profile):
        _put_banner(img, ["Privacy Monitoring Enabled", "(occupancy only, no recording)"],
                     (10, 70), (120, 0, 100))
        return

    ppe = get_required_ppe(profile)
    body_lines = []
    if ppe:
        body_lines.append("Required PPE: " + ", ".join(ppe))
    body_lines.append(f"Status: {status}")

    status_color = (0, 150, 0) if status == "SAFE" else (0, 0, 200)
    y_offset = 70
    y_offset += _put_banner(img, body_lines, (10, y_offset), (40, 40, 40))

    if violation_text:
        alert_prefix = "\U0001F6A8 UNAUTHORIZED ENTRY" if profile.get("restricted") else "\u26A0 PPE VIOLATION"
        _put_banner(img, [alert_prefix, violation_text, "Supervisor Alert"], (10, y_offset), status_color)


# ---------------------------------------------------------------------------
# FEATURE 7: Event log
# ---------------------------------------------------------------------------
def log_event(cam_key, profile, violation, log_path=DEFAULT_EVENT_LOG_PATH):
    """Append a structured violation event as one JSON line:
    {time, camera, camera_name, zone, violation, severity}"""
    t = time.localtime()
    event = {
        "time": time.strftime("%H:%M:%S", t),
        "date": date.today().isoformat(),
        "camera": cam_key,
        "camera_name": profile.get("name", cam_key),
        "zone": profile.get("zone", "Unknown"),
        "violation": violation,
        "severity": severity_for(profile),
    }
    try:
        with open(log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"[zone_manager] Failed to write event log: {e}")
    return event


def read_event_log(log_path=DEFAULT_EVENT_LOG_PATH, limit=None):
    """Read the event log back as a list of dicts (newest last).
    Used by the dashboard for FEATURE 6 / FEATURE 7."""
    if not os.path.exists(log_path):
        return []
    events = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit:
        events = events[-limit:]
    return events


def latest_status_by_camera(camera_config, log_path=DEFAULT_EVENT_LOG_PATH, stale_after_seconds=300):
    """FEATURE 6: build the per-camera dashboard card data -- the most
    recent violation for each camera (if any), else SAFE / Privacy Mode."""
    events = read_event_log(log_path)
    latest = {}
    for e in events:
        latest[e["camera"]] = e

    cards = []
    for cam_key, profile in camera_config.items():
        entry = latest.get(cam_key)
        if is_privacy_zone(profile):
            cards.append({
                "camera": cam_key,
                "name": profile.get("name", cam_key),
                "zone": profile.get("zone", ""),
                "status": "Privacy Mode",
                "severity": "N/A",
            })
        elif entry is not None:
            cards.append({
                "camera": cam_key,
                "name": profile.get("name", cam_key),
                "zone": profile.get("zone", ""),
                "status": entry["violation"],
                "severity": entry["severity"],
                "time": entry["time"],
            })
        else:
            cards.append({
                "camera": cam_key,
                "name": profile.get("name", cam_key),
                "zone": profile.get("zone", ""),
                "status": "SAFE",
                "severity": severity_for(profile),
            })
    return cards
