import threading
import signal
import sys
import time
from flask import Flask, Response, render_template_string, jsonify

app = Flask(__name__)

# Shared state
_latest_frames = {}
_running = True


def set_latest_frame(cam_name, frame):
    """Called by main loop to push latest frame to server"""
    _latest_frames[cam_name] = frame


@app.route("/")
def index():
    # Auto-refresh camera list via JavaScript
    return render_template_string("""
    <html>
    <head>
        <title>CCTV Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; background: #111; color: #eee; }
            .cam-block { margin: 15px 0; padding: 10px; border: 1px solid #444; border-radius: 8px; background: #222; }
            h1 { text-align: center; }
            h2 { margin: 5px 0; }
            img { border-radius: 6px; }
        </style>
        <script>
            async function loadCameras() {
                let res = await fetch("/cameras");
                let cams = await res.json();
                let container = document.getElementById("cams");
                container.innerHTML = "";
                cams.forEach(cam => {
                    let div = document.createElement("div");
                    div.className = "cam-block";
                    div.innerHTML = `<h2>${cam}</h2><img src="/video/${cam}" width="640">`;
                    container.appendChild(div);
                });
            }
            setInterval(loadCameras, 5000); // refresh every 5s
            window.onload = loadCameras;
        </script>
    </head>
    <body>
      <h1>CCTV Feeds</h1>
      <div id="cams"></div>
    </body>
    </html>
    """)


@app.route("/cameras")
def cameras():
    """Return list of current cameras"""
    return jsonify(list(_latest_frames.keys()))


@app.route("/video/<cam>")
def video_feed(cam):
    def gen():
        import cv2
        import numpy as np

        while _running:
            frame = _latest_frames.get(cam)
            if frame is None:
                time.sleep(0.1)
                continue

            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)

    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def start_http_server(host="0.0.0.0", port=8080):
    """Start Flask server in a background thread"""
    global _running

    def run():
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)  # silence Flask logs
        app.run(host=host, port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    print(f"[HTTP] Server running at http://{host}:{port}/")

    def handle_sigint(sig, frame):
        global _running
        _running = False
        print("\n[HTTP] Server shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_sigint)


# ------------------- HTML Template --------------------

TEMPLATE = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>CCTV Dashboard</title>
  <style>
    body{font-family:system-ui,Arial,sans-serif;margin:0;background:#0b0f17;color:#dfe6ef}
    header{padding:12px 16px;background:#101828;border-bottom:1px solid #1f2937}
    h1{margin:0;font-size:18px}
    .row{display:flex;gap:12px;flex-wrap:wrap;padding:12px}
    .card{background:#111827;border:1px solid #243244;border-radius:12px;padding:10px;flex:1;min-width:380px}
    .card h3{margin:6px 0 10px;font-size:16px;color:#e5e7eb}
    details{background:#0f172a;border:1px solid #1f2937;border-radius:10px;margin:8px 0;padding:8px}
    details>summary{cursor:pointer;list-style:none;font-weight:600}
    summary::-webkit-details-marker{display:none}
    img.thumb{width:300px;height:auto;border-radius:8px;border:1px solid #2b3543}
    .muted{color:#9aa7b3;font-size:12px}
    .pill{display:inline-block;background:#1f2937;border-radius:999px;padding:2px 8px;margin-left:8px;font-size:12px}
    .sightings{display:flex;gap:8px;flex-wrap:wrap}
    .sighting{display:flex;flex-direction:column;gap:4px;align-items:flex-start}
    .live{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}
    .live .panel{background:#0f172a;border:1px solid #1f2937;border-radius:10px;padding:8px}
    .live img{width:100%;border-radius:8px;border:1px solid #222}
    a, a:visited{color:#9dd1ff}
    table{border-collapse:collapse;width:100%;font-size:13px}
    th, td{border-bottom:1px solid #223042;padding:6px}
    .sticky{position:sticky;top:0;background:#101828;z-index:10;border-bottom:1px solid #263242}
  </style>
</head>
<body>
  <header class="sticky">
    <h1>Multi-Cam CCTV Dashboard</h1>
  </header>

  <div class="row">
    <div class="card" style="flex:2">
      <h3>Live Feeds</h3>
      <div class="live" id="liveFeeds"></div>
    </div>
    <div class="card" style="flex:3">
      <h3>Detections (Grouped)</h3>
      <div id="detections"></div>
    </div>
  </div>

<script>
const CAMS = {{ cams|tojson }};

function localKey(id){ return "ui.open."+id; }
function persistOpen(id, open){
  try{ localStorage.setItem(localKey(id), open ? "1":"0"); }catch(e){}
}
function restoreOpen(id, el){
  try{
    const v = localStorage.getItem(localKey(id));
    if(v === "1") el.setAttribute("open","open");
    else el.removeAttribute("open");
  }catch(e){}
}

// --- Live Feeds
function renderLive(){
  const wrap = document.getElementById("liveFeeds");
  wrap.innerHTML = "";
  for(const cam of CAMS){
    const div = document.createElement("div");
    div.className = "panel";
    div.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <div><strong>${cam}</strong></div>
        <div class="muted" id="ts_${cam}"></div>
      </div>
      <img id="img_${cam}" src="/stream/${cam}" alt="${cam}">
    `;
    wrap.appendChild(div);
  }
}

// --- Detections grouped: class -> gid -> sightings
async function fetchEvents(){
  const r = await fetch("/api/events");
  if(!r.ok) return {};
  return await r.json();
}

function groupByClass(events){
  const g = {};
  for(const gid in events){
    const e = events[gid];
    const cls = e.class || "object";
    if(!g[cls]) g[cls] = {};
    g[cls][gid] = e;
  }
  return g;
}

function renderDetections(data){
  const root = document.getElementById("detections");
  root.innerHTML = "";
  const byClass = groupByClass(data);

  for(const cls of Object.keys(byClass).sort()){
    const clsId = "cls_"+cls;
    const details = document.createElement("details");
    details.id = clsId;
    details.innerHTML = `<summary>${cls} <span class="pill">${Object.keys(byClass[cls]).length} IDs</span></summary>`;
    restoreOpen(clsId, details);
    details.addEventListener("toggle", () => persistOpen(clsId, details.open));

    // Per GID
    for(const gid of Object.keys(byClass[cls]).sort((a,b)=>parseInt(a)-parseInt(b))){
      const e = byClass[cls][gid];
      const gidId = clsId + "_gid_" + gid;
      const d2 = document.createElement("details");
      d2.id = gidId;

      // thumbnails: show up to 5
      const pics = (e.sightings||[]).filter(s=>s.image).slice(0,5);
      let thumbs = pics.map(s=>`<a href="/${s.image}" target="_self"><img class="thumb" src="/${s.image}"/></a>`).join("");

      // table for the rest sightings
      let rows = "";
      for(const s of e.sightings || []){
        const t = new Date((s.ts||0)*1000).toLocaleString();
        const vid = s.video ? `<a href="/recordings/${s.video}">${s.video}</a>` : "-";
        const img = s.image ? `<a href="/${s.image}" target="_self">view</a>` : "-";
        rows += `<tr><td>${s.cam||""}</td><td>${t}</td><td>${img}</td><td>${vid}</td></tr>`;
      }
      const table = `
        <table>
          <thead><tr><th>Camera</th><th>Time</th><th>Image</th><th>Recording</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      `;

      d2.innerHTML = `
        <summary>Object #${gid} <span class="pill">${(e.sightings||[]).length} sightings</span></summary>
        <div class="sightings">${thumbs}</div>
        ${table}
      `;
      restoreOpen(gidId, d2);
      d2.addEventListener("toggle", ()=>persistOpen(gidId, d2.open));
      details.appendChild(d2);
    }
    root.appendChild(details);
  }
}

async function refresh(){
  try{
    const data = await fetchEvents();
    renderDetections(data);
  }catch(e){ console.error(e); }
}

renderLive();
refresh();
setInterval(refresh, 4000);
</script>
</body>
</html>
"""
