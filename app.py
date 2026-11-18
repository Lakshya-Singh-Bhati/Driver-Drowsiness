import os
import sys
import math
import time
import tempfile
import urllib.request
import requests
import cv2
import numpy as np
import streamlit as st
from scipy.spatial import distance as dist
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------
# PAGE CONFIG & GLASSMORPHISM DARK UI STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Dark Glassmorphism Theme */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a, #020617 75%);
        color: #f8fafc;
    }
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(14px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .status-normal {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #34d399;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.75); background: rgba(239, 68, 68, 0.25); }
        50% { box-shadow: 0 0 25px 8px rgba(239, 68, 68, 0.45); background: rgba(239, 68, 68, 0.55); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.75); background: rgba(239, 68, 68, 0.25); }
    }
    .status-alert {
        border: 2px solid #ef4444;
        color: #fecaca;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: 0.05em;
        animation: pulse-red 1s infinite;
    }
    .rest-stop-box {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.5);
        border-radius: 14px;
        padding: 16px;
        margin-top: 12px;
        color: #fde68a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# CONSTANTS & MEDIAPIPE MODEL AUTO-DOWNLOAD
# ---------------------------------------------------------
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Standard 6-point EAR landmark indices for MediaPipe 478-point Face Mesh
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading MediaPipe Face Landmarker model (~30 MB)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


@st.cache_resource
def load_face_landmarker():
    ensure_model_downloaded()
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ---------------------------------------------------------
# EAR CALCULATION, AUDIO ALERT & LIVE NEARBY REST STOPS
# ---------------------------------------------------------
def compute_ear(eye_points: np.ndarray) -> float:
    # EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    vertical_1 = dist.euclidean(eye_points[1], eye_points[5])
    vertical_2 = dist.euclidean(eye_points[2], eye_points[4])
    horizontal = dist.euclidean(eye_points[0], eye_points[3])
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def trigger_audio_beep():
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(1200, 350)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return r * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


@st.cache_data(ttl=600)
def get_live_rest_stops():
    try:
        # 1. Detect current approximate coordinates via IP geolocation
        geo = requests.get("https://ipinfo.io/json", timeout=3).json()
        lat, lon = map(float, geo["loc"].split(","))
        city = geo.get("city", "your area")

        # 2. Query OpenStreetMap Overpass API for named fuel stations, cafes, or rest areas within 6 km
        query = f"""
        [out:json][timeout:6];
        (
          node["amenity"~"fuel|cafe|fast_food"]["name"](around:6000,{lat},{lon});
          way["amenity"~"fuel|cafe|fast_food"]["name"](around:6000,{lat},{lon});
          node["highway"="rest_area"](around:6000,{lat},{lon});
        );
        out center 10;
        """
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            timeout=6,
        ).json()

        places = []
        seen_names = set()
        for el in resp.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name", "Highway Rest Area")
            if name in seen_names:
                continue
            seen_names.add(name)

            p_lat = el.get("lat") or el.get("center", {}).get("lat")
            p_lon = el.get("lon") or el.get("center", {}).get("lon")
            if p_lat is None or p_lon is None:
                continue

            dist_km = haversine_km(lat, lon, float(p_lat), float(p_lon))
            kind = (
                tags.get("amenity", tags.get("highway", "Stop"))
                .replace("_", " ")
                .title()
            )
            places.append((dist_km, name, kind))

        places.sort(key=lambda x: x[0])
        top_places = places[:3]

        if top_places:
            items_html = "".join(
                f"<li><b>{name}</b> ({kind}) — {d:.1f} km away</li>"
                for d, name, kind in top_places
            )
            maps_link = f"https://www.google.com/maps/search/rest+stop+fuel+cafe/@{lat},{lon},14z"
            return items_html, maps_link, city
    except Exception:
        pass

    return (
        "<li><b>Nearest Fuel & Coffee Stop</b> — Open Maps for live routing</li>"
        "<li><b>Highway Safe Parking Bay</b> — Open Maps for live routing</li>",
        "https://www.google.com/maps/search/rest+stop+near+me",
        "your location",
    )


# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("Configure detection sensitivity and input feed.")

ear_threshold = st.sidebar.slider(
    "EAR Threshold",
    min_value=0.15,
    max_value=0.35,
    value=0.25,
    step=0.01,
    help="Eye Aspect Ratio cutoff. Lower = stricter.",
)

warning_frames = st.sidebar.slider(
    "Warning Frames",
    min_value=5,
    max_value=60,
    value=20,
    step=1,
    help="Consecutive closed-eye frames before alert.",
)

video_source = st.sidebar.radio(
    "Video Source",
    options=["Webcam", "Upload Video"],
    index=0,
)

uploaded_file = None
if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Driver Video", type=["mp4", "avi", "mov", "mkv"]
    )

st.sidebar.markdown("---")
col_btn1, col_btn2 = st.sidebar.columns(2)
start_btn = col_btn1.button("▶️ Start", width="stretch", type="primary")
stop_btn = col_btn2.button("⏹️ Stop", width="stretch")

if "running" not in st.session_state:
    st.session_state.running = False

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False

# ---------------------------------------------------------
# MAIN DASHBOARD HEADER & PLACEHOLDERS
# ---------------------------------------------------------
st.title("🚗 Driver Drowsiness Detection System")
st.caption("Real-time facial landmark tracking & Eye Aspect Ratio (EAR) safety monitor")

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
fps_placeholder = m_col1.empty()
ear_placeholder = m_col2.empty()
blink_placeholder = m_col3.empty()
alert_placeholder = m_col4.empty()


def render_metric_card(placeholder, label: str, value: str, color: str = "#38bdf8"):
    placeholder.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color: {color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_metric_card(fps_placeholder, "Live FPS", "0.0")
render_metric_card(ear_placeholder, "Live EAR", "0.00")
render_metric_card(blink_placeholder, "Blink Count", "0")
render_metric_card(alert_placeholder, "Alert Count", "0", "#f87171")

status_placeholder = st.empty()
status_placeholder.markdown(
    '<div class="status-normal">🟢 SYSTEM READY — Click Start in the sidebar to begin monitoring</div>',
    unsafe_allow_html=True,
)

video_col, info_col = st.columns([2.2, 1])
frame_placeholder = video_col.empty()
rest_stop_placeholder = info_col.empty()

with info_col:
    st.markdown(
        """
        <div class="glass-card" style="text-align:left;">
            <h4 style="margin-top:0; color:#e2e8f0;">🧠 Live Telemetry Guide</h4>
            <ul style="color:#cbd5e1; font-size:0.9rem; padding-left:18px;">
                <li><b>EAR &ge; Threshold:</b> Eyes Open (Active)</li>
                <li><b>Brief Drop (2–5 frames):</b> Natural Blink</li>
                <li><b>Sustained Drop (&ge; Warning Frames):</b> Drowsiness Alert</li>
                <li><b>2+ Alerts:</b> Auto-fetches live nearby rest stops</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# VIDEO PROCESSING LOOP
# ---------------------------------------------------------
if st.session_state.running:
    detector = load_face_landmarker()

    cap = None
    temp_video_path = None

    if video_source == "Webcam":
        cap = cv2.VideoCapture(0)
    elif video_source == "Upload Video" and uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name
        tfile.close()
        cap = cv2.VideoCapture(temp_video_path)
    else:
        st.warning("⚠️ Please upload a video file in the sidebar before starting.")
        st.session_state.running = False

    if cap is not None and cap.isOpened():
        closed_frames = 0
        blink_count = 0
        alert_count = 0
        alert_active = False
        last_beep_time = 0.0
        prev_time = time.time()
        rest_stops_rendered = False

        try:
            while st.session_state.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if video_source == "Upload Video":
                        status_placeholder.markdown(
                            '<div class="status-normal">✅ Video processing complete.</div>',
                            unsafe_allow_html=True,
                        )
                    break

                if video_source == "Webcam":
                    frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = detector.detect(mp_image)

                avg_ear = 0.0
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    left_pts = np.array(
                        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE_IDX],
                        dtype=np.float32,
                    )
                    right_pts = np.array(
                        [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE_IDX],
                        dtype=np.float32,
                    )

                    left_ear = compute_ear(left_pts)
                    right_ear = compute_ear(right_pts)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Draw eye contours
                    eye_color = (0, 0, 255) if avg_ear < ear_threshold else (56, 189, 248)
                    cv2.polylines(frame, [left_pts.astype(np.int32)], True, eye_color, 2)
                    cv2.polylines(frame, [right_pts.astype(np.int32)], True, eye_color, 2)

                    # Blink & Drowsiness state logic
                    if avg_ear < ear_threshold:
                        closed_frames += 1
                        if closed_frames >= warning_frames:
                            if not alert_active:
                                alert_count += 1
                                alert_active = True
                            # Beep every 1.2 seconds while drowsy
                            if time.time() - last_beep_time > 1.2:
                                trigger_audio_beep()
                                last_beep_time = time.time()
                    else:
                        if 2 <= closed_frames < warning_frames:
                            blink_count += 1
                        closed_frames = 0
                        alert_active = False

                # FPS computation
                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-5)
                prev_time = curr_time

                # Overlay HUD on video frame
                cv2.putText(
                    frame,
                    f"EAR: {avg_ear:.2f} | Closed Frames: {closed_frames}",
                    (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255) if alert_active else (52, 211, 153),
                    2,
                )

                if alert_active:
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT! PULL OVER!",
                        (18, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        3,
                    )

                # Update live metrics
                render_metric_card(fps_placeholder, "Live FPS", f"{fps:.1f}")
                render_metric_card(
                    ear_placeholder,
                    "Live EAR",
                    f"{avg_ear:.2f}",
                    "#f87171" if avg_ear < ear_threshold and avg_ear > 0 else "#34d399",
                )
                render_metric_card(blink_placeholder, "Blink Count", str(blink_count))
                render_metric_card(alert_placeholder, "Alert Count", str(alert_count), "#f87171")

                # Update Status Banner
                if alert_active:
                    status_placeholder.markdown(
                        '<div class="status-alert">🚨 CRITICAL DROWSINESS DETECTED — TAKE A BREAK IMMEDIATELY! 🚨</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    status_placeholder.markdown(
                        '<div class="status-normal">🟢 DRIVER ALERT — Monitoring Active</div>',
                        unsafe_allow_html=True,
                    )

                # Fetch & display live nearby rest stops after sustained drowsiness
                if (alert_count >= 2 or closed_frames >= warning_frames * 2) and not rest_stops_rendered:
                    stops_html, maps_url, detected_city = get_live_rest_stops()
                    rest_stop_placeholder.markdown(
                        f"""
                        <div class="rest-stop-box">
                            <h4 style="margin-top:0;">🗺️ Nearest Safe Rest Stops ({detected_city})</h4>
                            <p style="font-size:0.88rem; margin-bottom:8px;">
                                Sustained fatigue detected. Pull over safely at one of these nearby stops:
                            </p>
                            <ul style="font-size:0.85rem; padding-left:18px; margin-bottom:10px;">
                                {stops_html}
                            </ul>
                            <a href="{maps_url}" target="_blank"
                               style="color:#fbbf24; font-weight:700; text-decoration:underline;">
                               📍 Open Live Rest Stops in Google Maps &rarr;
                            </a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    rest_stops_rendered = True

                frame_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width="stretch",
                )
        finally:
            cap.release()
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            st.session_state.running = False