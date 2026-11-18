"""
PathSnap evaluator interface  (runs on LAPTOP 2)

It only talks to the API over the network: it asks for the list of trips, sends the chosen
trip ID, and draws what comes back. It has no model and no ground-truth data.

Start (inside the folder that contains ui/, venv active):
  streamlit run ui/app.py
Then type Laptop 1's address in the sidebar, e.g. http://192.168.1.23:8000
"""
import os
import requests
import folium
import streamlit as st

st.set_page_config(page_title="PathSnap", page_icon="🛰️", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] code { background: #1e293b !important; color: #a5b4fc !important; }
section[data-testid="stSidebar"] input { background: #1e293b !important; color: #f8fafc !important;
    border: 1px solid #334155 !important; border-radius: 10px !important; }
section[data-testid="stSidebar"] .stButton button { width: 100%; border-radius: 12px; border: 0;
    padding: 0.7rem 1rem; font-weight: 700; font-size: 1rem; color: white !important;
    background: linear-gradient(90deg, #6366f1, #06b6d4); box-shadow: 0 6px 20px rgba(99,102,241,.35); }
section[data-testid="stSidebar"] .stButton button:hover { filter: brightness(1.08); }

.hero { padding: 1.6rem 2rem; border-radius: 20px; margin-bottom: 1.2rem;
    background: linear-gradient(120deg, #312e81 0%, #4f46e5 45%, #06b6d4 100%); color: white;
    box-shadow: 0 10px 30px rgba(49,46,129,.25); }
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; color: white; }
.hero p { margin: .35rem 0 0; opacity: .9; font-size: 1.02rem; }
.steps { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: 1rem; }
.step { background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25);
    padding: .3rem .8rem; border-radius: 999px; font-size: .82rem; font-weight: 600; }

.cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin: .4rem 0 1.2rem; }
.card { background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1rem 1.2rem;
    box-shadow: 0 2px 10px rgba(15,23,42,.05); }
.card .label { color: #64748b; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.card .value { color: #0f172a; font-size: 1.85rem; font-weight: 800; margin-top: .15rem; }
.card .sub { color: #64748b; font-size: .8rem; margin-top: .1rem; }
.card.main { background: linear-gradient(135deg, #eef2ff, #ecfeff); border-color: #c7d2fe; }
.card.main .value { color: #4338ca; }
.badge { display: inline-block; margin-top: .3rem; padding: .12rem .5rem; border-radius: 999px;
    font-size: .75rem; font-weight: 700; background: #dcfce7; color: #166534; }

.section-title { font-size: 1.15rem; font-weight: 700; color: #0f172a; margin: .2rem 0 .6rem; }
.tripinfo { color: #475569; font-size: .92rem; margin-bottom: .6rem; }
.legend { display: flex; gap: 1.2rem; font-size: .85rem; color: #334155; margin: .4rem 0 .6rem; }
.dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: .35rem; vertical-align: -1px; }
.line { display: inline-block; width: 22px; height: 4px; border-radius: 2px; margin-right: .35rem; vertical-align: 3px; }
.empty { text-align: center; padding: 4rem 1rem; color: #64748b; border: 2px dashed #cbd5e1; border-radius: 20px; }
.empty h3 { color: #334155; margin-bottom: .3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🛰️ PathSnap</h1>
  <p>Reconstructing real vehicle routes from noisy GPS by matching every point to the road network.</p>
  <div class="steps">
    <span class="step">① Preprocessing</span><span class="step">② Candidate roads (50 m)</span>
    <span class="step">③ HistGradientBoosting classifier</span><span class="step">④ HMM + Viterbi decoding</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar: connection + trip choice
st.sidebar.markdown("## ⚙️ Connection")
api_url = st.sidebar.text_input("API address (Laptop 1)",
                                os.environ.get("PATHSNAP_API", "http://localhost:8000")).rstrip("/")
try:
    health = requests.get(f"{api_url}/health", timeout=5).json()
    trips = requests.get(f"{api_url}/trips", timeout=10).json()["trips"]
    st.sidebar.markdown(f"🟢 **Connected**  \nModel: `{health['model']}`  \nTrips: `{health['partition']}`")
except Exception as e:
    st.sidebar.markdown("🔴 **Not connected**")
    st.markdown(f"""<div class="empty"><h3>Cannot reach the API</h3>
        Check that Laptop 1 is running the server and that the address on the left is correct.<br>
        <code>{api_url}</code></div>""", unsafe_allow_html=True)
    st.stop()

st.sidebar.markdown("## 🗺️ Evaluation trip")
trip_id = st.sidebar.selectbox(f"Choose one of {len(trips)} trips", trips)
run = st.sidebar.button("🚀  Reconstruct route", type="primary")

if run:
    with st.spinner(f"Laptop 1 is matching trip {trip_id} to the road network…"):
        try:
            r = requests.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=300)
            r.raise_for_status()
            res = r.json()
            st.session_state["result"] = res
            hist = st.session_state.setdefault("history", [])
            m = res.get("metrics") or {}
            hist.insert(0, {"Trip": res["trip_id"],
                            "Top-1 (HMM)": f"{m['top1_accuracy_hmm']:.2%}" if m else "n/a",
                            "Classifier only": f"{m['top1_accuracy_classifier_only']:.2%}" if m else "n/a",
                            "Route F1": f"{m['route_f1']:.2%}" if m else "n/a",
                            "Time (s)": res["processing_seconds"]})
        except Exception as e:
            st.error(f"API request failed: {e}")
            st.stop()

res = st.session_state.get("result")
if not res:
    st.markdown("""<div class="empty"><h3>Choose a trip and press “Reconstruct route”</h3>
        The selected trip ID is sent to the API on Laptop 1, which runs the full pipeline and
        returns the matched route with its accuracy.</div>""", unsafe_allow_html=True)
    st.stop()

# ------------------------------------------------------------------ metric cards
m = res.get("metrics")
if m:
    gain = (m["top1_accuracy_hmm"] - m["top1_accuracy_classifier_only"]) * 100
    cards = f"""
    <div class="cards">
      <div class="card main"><div class="label">Top-1 accuracy</div>
        <div class="value">{m['top1_accuracy_hmm']:.1%}</div>
        <span class="badge">▲ {gain:+.1f} pts from HMM</span></div>
      <div class="card"><div class="label">Classifier only</div>
        <div class="value">{m['top1_accuracy_classifier_only']:.1%}</div><div class="sub">before Viterbi</div></div>
      <div class="card"><div class="label">Route F1</div>
        <div class="value">{m['route_f1']:.1%}</div>
        <div class="sub">P {m['route_precision']:.1%} · R {m['route_recall']:.1%}</div></div>
      <div class="card"><div class="label">Route continuity</div>
        <div class="value">{res['route_continuity']:.1%}</div><div class="sub">connected steps</div></div>
      <div class="card"><div class="label">Processing time</div>
        <div class="value">{res['processing_seconds']:.2f}s</div><div class="sub">on Laptop 1</div></div>
    </div>"""
else:
    cards = f"""<div class="cards"><div class="card main"><div class="label">Top-1 accuracy</div>
        <div class="value">n/a</div><div class="sub">no reference route on server</div></div>
        <div class="card"><div class="label">Route continuity</div><div class="value">{res['route_continuity']:.1%}</div></div>
        <div class="card"><div class="label">Processing time</div><div class="value">{res['processing_seconds']:.2f}s</div></div></div>"""
st.markdown(f'<div class="section-title">Trip {res["trip_id"]}</div>', unsafe_allow_html=True)
st.markdown(cards, unsafe_allow_html=True)

# ------------------------------------------------------------------ map
st.markdown(f"""<div class="tripinfo">{res['n_gps_points']:,} GPS points → {res['n_route_arcs']} road segments ·
    model <code>{res['model']}</code></div>
    <div class="legend"><span><span class="line" style="background:#4f46e5"></span>Predicted route</span>
    <span><span class="dot" style="background:#f97316"></span>Raw GPS points</span>
    <span><span class="dot" style="background:#16a34a"></span>Start</span>
    <span><span class="dot" style="background:#dc2626"></span>End</span></div>""", unsafe_allow_html=True)

gps = res["gps_points"]
fmap = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles=None, control_scale=True)
folium.TileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                 attr="© OpenStreetMap contributors © CARTO", name="Light map").add_to(fmap)
folium.TileLayer("OpenStreetMap", name="Street map").add_to(fmap)
pred = folium.FeatureGroup(name="Predicted route")
for seg in res["predicted_route_segments"]:
    folium.PolyLine(seg, color="#4f46e5", weight=6, opacity=0.9).add_to(pred)
pred.add_to(fmap)
raw = folium.FeatureGroup(name="Raw GPS points")
for lat, lon in gps[::2]:
    folium.CircleMarker((lat, lon), radius=2.2, color="#f97316", weight=0,
                        fill=True, fill_opacity=0.75).add_to(raw)
raw.add_to(fmap)
folium.CircleMarker(gps[0], radius=8, color="white", weight=3, fill=True, fill_color="#16a34a",
                    fill_opacity=1, tooltip="Start").add_to(fmap)
folium.CircleMarker(gps[-1], radius=8, color="white", weight=3, fill=True, fill_color="#dc2626",
                    fill_opacity=1, tooltip="End").add_to(fmap)
fmap.fit_bounds([[min(p[0] for p in gps), min(p[1] for p in gps)],
                 [max(p[0] for p in gps), max(p[1] for p in gps)]], padding=(20, 20))
folium.LayerControl(collapsed=True).add_to(fmap)
html = fmap.get_root().render()
if hasattr(st, "iframe"):
    st.iframe(html, height=600)
else:
    import streamlit.components.v1 as components
    components.html(html, height=600)

# ------------------------------------------------------------------ history of requests
hist = st.session_state.get("history", [])
if len(hist) > 1:
    st.markdown('<div class="section-title">Trips evaluated this session</div>', unsafe_allow_html=True)
    st.dataframe(hist, hide_index=True, use_container_width=True)
