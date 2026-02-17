#!/usr/bin/env python3
"""
Generate a single interactive HTML dashboard with V-VAD metrics for ALL recordings.
Includes a dropdown selector to switch between recordings — charts update dynamically.
Shows BOTH old (baseline) and new (ZCR-enhanced) V-VAD algorithms side by side.

Usage: python generate_vvad_dashboard.py
Output: vvad_dashboard.html
"""

import os
import json
import glob
import numpy as np
import pandas as pd
from collections import deque

# ── Algorithm functions ──────────────────────────────────────────────

def frontalize_mouth_landmarks(points_3d):
    required_stable = [8, 27, 36, 39, 42, 45]
    mouth_indices = list(range(48, 68))
    for idx in required_stable + mouth_indices:
        if idx not in points_3d:
            return None
    p = {k: np.array(v, dtype=np.float64) for k, v in points_3d.items()}
    left_eye_center = (p[36] + p[39]) / 2.0
    right_eye_center = (p[42] + p[45]) / 2.0
    x_axis = right_eye_center - left_eye_center
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        return None
    x_axis = x_axis / x_norm
    y_approx = p[8] - p[27]
    z_axis = np.cross(x_axis, y_approx)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-8:
        return None
    z_axis = z_axis / z_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    R_face = np.column_stack([x_axis, y_axis, z_axis])
    R_inv = R_face.T
    mouth_3d = np.array([list(points_3d[i]) for i in mouth_indices], dtype=np.float64)
    centroid = mouth_3d.mean(axis=0)
    mouth_centered = mouth_3d - centroid
    mouth_derotated = (R_inv @ mouth_centered.T).T
    mouth_frontalized = mouth_derotated + centroid
    result = {}
    for i, idx in enumerate(mouth_indices):
        result[idx] = (mouth_frontalized[i, 0], mouth_frontalized[i, 1])
    return result


def compute_mar(mouth_points):
    inner_indices = list(range(60, 68))
    for idx in inner_indices:
        if idx not in mouth_points:
            return None
    p = {k: np.array(v, dtype=np.float64) for k, v in mouth_points.items() if k in inner_indices}
    d1 = np.linalg.norm(p[61] - p[67])
    d2 = np.linalg.norm(p[62] - p[66])
    d3 = np.linalg.norm(p[63] - p[65])
    d_horiz = np.linalg.norm(p[60] - p[64])
    if d_horiz < 1e-8:
        return None
    return (d1 + d2 + d3) / (2.0 * d_horiz)


# ── OLD algorithm (baseline): var + mean, no ZCR ────────────────────
OLD_PARAMS = {
    'window_seconds': 0.5,
    'var_threshold': 0.0005,
    'mar_activity_threshold': 0.12,
    'hold_seconds': 1.0,
    'fps': 30.0,
    'label': 'OLD (baseline)',
}

# ── NEW algorithm (ZCR-enhanced, balanced recall) ───────────────────
NEW_PARAMS = {
    'window_seconds': 0.3,
    'var_threshold': 0.002,
    'mar_activity_threshold': 0.15,
    'hold_seconds': 1.0,
    'min_zcr': 3,
    'fps': 30.0,
    'label': 'NEW (ZCR, balanced recall)',
}


def _sigmoid(x, center, k=4.0):
    """Sigmoid function for soft thresholding. Returns 0.5 at x=center."""
    z = k * (x - center)
    # Clamp to avoid overflow in exp
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + np.exp(-z))


def _compute_speech_probability(var_value, mean_mar, params, zcr_value=None):
    """
    Compute continuous V-VAD speech probability [0, 1] from signal features.

    Uses sigmoid normalization for each feature — 0.5 at the threshold,
    smooth transition on both sides. The combined probability is the product
    of individual scores, so all features must contribute for high probability.

    For the OLD algorithm (no ZCR):  prob = s_var * s_mar
    For the NEW algorithm (with ZCR): prob = s_var * s_mar * s_zcr

    Args:
        var_value:  MAR variance in the current window
        mean_mar:   Mean MAR in the current window
        params:     Algorithm parameter dict (thresholds)
        zcr_value:  Zero-crossing rate (None for OLD algorithm)

    Returns:
        float: Speech probability in [0, 1]
    """
    vt = params['var_threshold']
    mt = params['mar_activity_threshold']
    k = 4.0  # steepness — gives smooth but decisive transition

    # Normalize each feature relative to its threshold, then apply sigmoid
    s_var = _sigmoid(var_value / vt, 1.0, k) if vt > 0 else 0.0
    s_mar = _sigmoid(mean_mar / mt, 1.0, k) if mt > 0 else 0.0

    if zcr_value is not None and 'min_zcr' in params and params['min_zcr'] > 0:
        s_zcr = _sigmoid(zcr_value / params['min_zcr'], 1.0, k)
        return float(s_var * s_mar * s_zcr)
    else:
        return float(s_var * s_mar)


def run_old_vvad(frontal_mars, params):
    """Run OLD baseline V-VAD: var + mean threshold + hold."""
    fps = params['fps']
    ws = max(3, int(params['window_seconds'] * fps))
    hf = max(1, int(params['hold_seconds'] * fps))
    vt = params['var_threshold']
    mt = params['mar_activity_threshold']

    hist = deque(maxlen=ws)
    hc = 0
    raw_out = []
    vvad_out = []
    var_out = []
    mean_out = []
    prob_out = []
    smoothed_prob_out = []

    prev_smoothed = 0.0
    decay = 1.0 - (1.0 / max(1, hf))  # decay matches hold time

    for m in frontal_mars:
        if m is not None:
            hist.append(m)
        active = False
        mv = 0.0
        mm = 0.0
        if len(hist) >= 3:
            v = np.array(hist)
            mv = float(np.var(v))
            mm = float(np.mean(v))
            active = (mv > vt) and (mm > mt)

        prob = _compute_speech_probability(mv, mm, params)
        smoothed = max(prob, prev_smoothed * decay)
        prev_smoothed = smoothed

        var_out.append(round(mv, 8))
        mean_out.append(round(mm, 6))
        prob_out.append(round(prob, 4))
        smoothed_prob_out.append(round(smoothed, 4))
        raw_out.append(1 if active else 0)
        if active:
            hc = hf
        elif hc > 0:
            hc -= 1
        vvad_out.append(1 if hc > 0 else 0)

    return raw_out, vvad_out, var_out, mean_out, prob_out, smoothed_prob_out


def run_new_vvad(frontal_mars, params):
    """Run NEW ZCR-enhanced V-VAD: var + mean + ZCR + hold."""
    fps = params['fps']
    ws = max(4, int(params['window_seconds'] * fps))
    hf = max(1, int(params['hold_seconds'] * fps))
    vt = params['var_threshold']
    mt = params['mar_activity_threshold']
    min_zcr = params['min_zcr']

    hist = deque(maxlen=ws)
    hc = 0
    raw_out = []
    vvad_out = []
    var_out = []
    mean_out = []
    zcr_out = []
    prob_out = []
    smoothed_prob_out = []

    prev_smoothed = 0.0
    decay = 1.0 - (1.0 / max(1, hf))  # decay matches hold time

    for m in frontal_mars:
        if m is not None:
            hist.append(m)
        active = False
        zcr_val = 0
        mv = 0.0
        mm = 0.0
        if len(hist) >= 4:
            v = np.array(hist)
            mv = float(np.var(v))
            mm = float(np.mean(v))
            delta = np.diff(v)
            signs = np.sign(delta)
            zcr_val = int(np.sum(np.abs(np.diff(signs)) > 0))
            active = (mv > vt) and (mm > mt) and (zcr_val >= min_zcr)

        prob = _compute_speech_probability(mv, mm, params, zcr_value=zcr_val)
        smoothed = max(prob, prev_smoothed * decay)
        prev_smoothed = smoothed

        var_out.append(round(mv, 8))
        mean_out.append(round(mm, 6))
        zcr_out.append(zcr_val)
        prob_out.append(round(prob, 4))
        smoothed_prob_out.append(round(smoothed, 4))
        raw_out.append(1 if active else 0)
        if active:
            hc = hf
        elif hc > 0:
            hc -= 1
        vvad_out.append(1 if hc > 0 else 0)

    return raw_out, vvad_out, var_out, mean_out, zcr_out, prob_out, smoothed_prob_out


def process_single_dump(dumps_path):
    df_mouth = pd.read_csv(os.path.join(dumps_path, 'mouth_position.csv'))
    df_pose = pd.read_csv(os.path.join(dumps_path, 'face_position.csv'))
    df_vad = pd.read_csv(os.path.join(dumps_path, 'vad.csv'))

    timestamps = sorted(df_mouth['seconds'].unique())

    times = []
    raw_mar_values = []
    frontal_mar_values = []

    for t in timestamps:
        frame_data = df_mouth[df_mouth['seconds'] == t]
        face_data = frame_data[frame_data['face_idx'] == 0]

        pts_2d = {}
        pts_3d = {}
        for _, row in face_data.iterrows():
            idx = int(row['point_type'])
            pts_2d[idx] = (row['x'], row['y'])
            pts_3d[idx] = (row['x'], row['y'], row['z'])

        raw_mar = compute_mar(pts_2d)
        frontalized = frontalize_mouth_landmarks(pts_3d)
        frontal_mar = compute_mar(frontalized) if frontalized else None

        times.append(round(t, 6))
        raw_mar_values.append(round(raw_mar, 6) if raw_mar is not None else None)
        frontal_mar_values.append(round(frontal_mar, 6) if frontal_mar is not None else None)

    # Run BOTH algorithms
    old_raw, old_vvad, old_var, old_mean, old_prob, old_smoothed = run_old_vvad(frontal_mar_values, OLD_PARAMS)
    new_raw, new_vvad, new_var, new_mean, new_zcr, new_prob, new_smoothed = run_new_vvad(frontal_mar_values, NEW_PARAMS)

    # Audio VAD
    vad_times = df_vad['seconds'].values
    vad_vals = df_vad['vadDagcDecFinal'].values
    audio_vad_values = []
    for t in times:
        idx = np.argmin(np.abs(vad_times - t))
        audio_vad_values.append(int(vad_vals[idx]))

    # Euler angles
    pose_times = df_pose['seconds'].values
    yaw_values, pitch_values, roll_values = [], [], []
    for t in times:
        idx = np.argmin(np.abs(pose_times - t))
        row = df_pose.iloc[idx]
        yaw_values.append(round(float(row['yaw']), 3))
        pitch_values.append(round(float(row['pitch']), 3))
        roll_values.append(round(float(row['roll']), 3))

    return {
        'times': times,
        'raw_mar': raw_mar_values,
        'frontal_mar': frontal_mar_values,
        # OLD algorithm
        'old_var': old_var,
        'old_mean': old_mean,
        'old_raw': old_raw,
        'old_vvad': old_vvad,
        'old_prob': old_prob,
        'old_smoothed': old_smoothed,
        # NEW algorithm
        'new_var': new_var,
        'new_mean': new_mean,
        'new_zcr': new_zcr,
        'new_raw': new_raw,
        'new_vvad': new_vvad,
        'new_prob': new_prob,
        'new_smoothed': new_smoothed,
        # Reference
        'audio_vad': audio_vad_values,
        'yaw': yaw_values,
        'pitch': pitch_values,
        'roll': roll_values,
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dump_dirs = sorted(glob.glob(os.path.join(base_dir, 'dumps_*')))
    dump_dirs = [d for d in dump_dirs if os.path.isdir(d)
                 and os.path.exists(os.path.join(d, 'mouth_position.csv'))
                 and os.path.exists(os.path.join(d, 'vad.csv'))]

    print(f"Found {len(dump_dirs)} dump directories")

    all_data = {}
    for dump_path in dump_dirs:
        name = os.path.basename(dump_path)
        label = name.replace('dumps_', '')
        print(f"  Processing: {name} ({label})...")
        all_data[label] = process_single_dump(dump_path)
        print(f"    → {len(all_data[label]['times'])} frames, "
              f"{all_data[label]['times'][-1]:.1f}s")

    data_json = json.dumps(all_data, indent=None)
    old_params_json = json.dumps(OLD_PARAMS)
    new_params_json = json.dumps(NEW_PARAMS)
    recording_keys = list(all_data.keys())
    options_html = '\n'.join(
        f'    <option value="{k}">{k.replace("_", " ").title()}</option>'
        for k in recording_keys
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V-VAD Dashboard — OLD vs NEW Algorithm Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.1.0/dist/chartjs-plugin-annotation.min.js"></script>
<style>
  :root {{
    --bg: #0f172a;
    --card: #1e293b;
    --accent: #38bdf8;
    --accent2: #a78bfa;
    --green: #4ade80;
    --red: #f87171;
    --white: #f1f5f9;
    --muted: #94a3b8;
    --yellow: #fbbf24;
    --orange: #fb923c;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--white);
    padding: 20px 30px;
    min-height: 100vh;
  }}
  h1 {{
    font-size: 1.8em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
    display: inline-block;
  }}
  h2 {{
    font-size: 1.1em;
    color: var(--accent);
    margin-bottom: 6px;
  }}
  .header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 14px;
  }}
  .header-left {{ flex: 1; }}
  .subtitle {{ color: var(--muted); font-size: 0.9em; }}

  .selector-box {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--card);
    border: 2px solid var(--accent);
    border-radius: 12px;
    padding: 10px 18px;
  }}
  .selector-box label {{
    color: var(--accent);
    font-weight: 600;
    font-size: 0.95em;
    white-space: nowrap;
  }}
  .selector-box select {{
    background: var(--bg);
    color: var(--white);
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 1em;
    font-family: 'Consolas', monospace;
    cursor: pointer;
    min-width: 220px;
  }}
  .selector-box select:hover {{ border-color: var(--accent); }}
  .selector-box select:focus {{ outline: none; border-color: var(--accent); box-shadow: 0 0 0 2px rgba(56,189,248,0.3); }}

  .nav-btns {{
    display: flex;
    gap: 6px;
  }}
  .nav-btns button {{
    background: var(--card);
    color: var(--muted);
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 7px 12px;
    cursor: pointer;
    font-size: 0.9em;
    transition: all 0.15s;
  }}
  .nav-btns button:hover {{ background: var(--accent); color: var(--bg); border-color: var(--accent); }}

  .chart-container {{
    background: var(--card);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
    border: 1px solid #334155;
  }}
  .chart-container canvas {{
    max-height: 180px;
  }}

  /* Two-column params */
  .params-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 12px;
  }}
  .params-bar {{
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    background: var(--card);
    border-radius: 10px;
    padding: 10px 16px;
    border: 1px solid #334155;
    font-size: 0.82em;
    align-items: center;
  }}
  .params-bar.old {{ border-color: #f87171; }}
  .params-bar.new {{ border-color: #4ade80; }}
  .params-title {{
    font-weight: 700;
    font-size: 0.9em;
    margin-right: 4px;
  }}
  .params-title.old {{ color: var(--red); }}
  .params-title.new {{ color: var(--green); }}
  .param {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .param-label {{ color: var(--muted); }}
  .param-value {{
    color: var(--yellow);
    font-weight: 700;
    font-family: 'Consolas', monospace;
  }}

  .legend-custom {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 5px;
    font-size: 0.78em;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 4px;
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 3px;
    flex-shrink: 0;
  }}

  /* Stats comparison */
  .stats-row {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 12px;
  }}
  .stat-card {{
    background: var(--card);
    border-radius: 10px;
    padding: 10px 14px;
    border: 1px solid #334155;
    flex: 1;
    min-width: 120px;
    text-align: center;
  }}
  .stat-value {{
    font-size: 1.4em;
    font-weight: 800;
    font-family: 'Consolas', monospace;
  }}
  .stat-label {{
    color: var(--muted);
    font-size: 0.76em;
    margin-top: 2px;
  }}
  .stat-compare {{
    display: flex;
    gap: 6px;
    justify-content: center;
    margin-top: 4px;
  }}
  .stat-old {{
    color: var(--red);
    font-size: 0.78em;
    font-family: 'Consolas', monospace;
    font-weight: 600;
  }}
  .stat-new {{
    color: var(--green);
    font-size: 0.78em;
    font-family: 'Consolas', monospace;
    font-weight: 600;
  }}
  .stat-arrow {{
    color: var(--muted);
    font-size: 0.78em;
  }}

  /* Two-column chart layout */
  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 12px;
  }}
  .charts-grid .chart-container canvas {{
    max-height: 160px;
  }}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <h1>V-VAD Algorithm Comparison Dashboard</h1>
    <p class="subtitle">OLD (baseline) vs NEW (ZCR, balanced recall) — select a recording to analyze</p>
  </div>
  <div class="selector-box">
    <label for="recording-select">Recording:</label>
    <select id="recording-select">
{options_html}
    </select>
    <div class="nav-btns">
      <button onclick="prevRecording()" title="Previous">&#9664;</button>
      <button onclick="nextRecording()" title="Next">&#9654;</button>
    </div>
  </div>
</div>

<div class="params-row">
  <div class="params-bar old">
    <span class="params-title old">OLD (baseline)</span>
    <div class="param"><span class="param-label">Window:</span><span class="param-value">{OLD_PARAMS['window_seconds']}s</span></div>
    <div class="param"><span class="param-label">Var thr:</span><span class="param-value">{OLD_PARAMS['var_threshold']}</span></div>
    <div class="param"><span class="param-label">MAR thr:</span><span class="param-value">{OLD_PARAMS['mar_activity_threshold']}</span></div>
    <div class="param"><span class="param-label">Hold:</span><span class="param-value">{OLD_PARAMS['hold_seconds']}s</span></div>
  </div>
  <div class="params-bar new">
    <span class="params-title new">NEW (ZCR, balanced recall)</span>
    <div class="param"><span class="param-label">Window:</span><span class="param-value">{NEW_PARAMS['window_seconds']}s</span></div>
    <div class="param"><span class="param-label">Var thr:</span><span class="param-value">{NEW_PARAMS['var_threshold']}</span></div>
    <div class="param"><span class="param-label">MAR thr:</span><span class="param-value">{NEW_PARAMS['mar_activity_threshold']}</span></div>
    <div class="param"><span class="param-label">Hold:</span><span class="param-value">{NEW_PARAMS['hold_seconds']}s</span></div>
    <div class="param"><span class="param-label">Min ZCR:</span><span class="param-value">{NEW_PARAMS['min_zcr']}</span></div>
  </div>
</div>

<div id="stats-row" class="stats-row"></div>

<div class="chart-container">
  <h2>1. Mouth Aspect Ratio (MAR) over Time</h2>
  <div class="legend-custom">
    <div class="legend-item"><div class="legend-dot" style="background:#94a3b8;"></div> Raw MAR</div>
    <div class="legend-item"><div class="legend-dot" style="background:#38bdf8;"></div> Frontalized MAR</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(248,113,113,0.4); border:1px dashed #f87171;"></div> OLD MAR thr ({OLD_PARAMS['mar_activity_threshold']})</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(74,222,128,0.4); border:1px dashed #4ade80;"></div> NEW MAR thr ({NEW_PARAMS['mar_activity_threshold']})</div>
  </div>
  <canvas id="chart-mar"></canvas>
</div>

<div class="charts-grid">
  <div class="chart-container">
    <h2>2a. MAR Variance — OLD (window={OLD_PARAMS['window_seconds']}s)</h2>
    <div class="legend-custom">
      <div class="legend-item"><div class="legend-dot" style="background:rgba(248,113,113,0.6);"></div> Variance</div>
      <div class="legend-item"><div class="legend-dot" style="background:rgba(248,113,113,0.3); border:1px dashed #f87171;"></div> Threshold ({OLD_PARAMS['var_threshold']})</div>
    </div>
    <canvas id="chart-var-old"></canvas>
  </div>
  <div class="chart-container">
    <h2>2b. MAR Variance — NEW (window={NEW_PARAMS['window_seconds']}s)</h2>
    <div class="legend-custom">
      <div class="legend-item"><div class="legend-dot" style="background:rgba(74,222,128,0.6);"></div> Variance</div>
      <div class="legend-item"><div class="legend-dot" style="background:rgba(74,222,128,0.3); border:1px dashed #4ade80;"></div> Threshold ({NEW_PARAMS['var_threshold']})</div>
    </div>
    <canvas id="chart-var-new"></canvas>
  </div>
</div>

<div class="chart-container">
  <h2>3. ZCR of MAR Derivative (NEW algorithm only)</h2>
  <div class="legend-custom">
    <div class="legend-item"><div class="legend-dot" style="background:#fb923c;"></div> Zero-Crossing Rate</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(251,191,36,0.3); border:1px dashed #fbbf24;"></div> min_zcr threshold ({NEW_PARAMS['min_zcr']})</div>
  </div>
  <canvas id="chart-zcr"></canvas>
</div>

<div class="charts-grid">
  <div class="chart-container">
    <h2>4a. Speech Probability — OLD (var &times; MAR)</h2>
    <div class="legend-custom">
      <div class="legend-item"><div class="legend-dot" style="background:rgba(248,113,113,0.7);"></div> Raw Probability</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f87171;"></div> Smoothed (hold decay)</div>
      <div class="legend-item"><div class="legend-dot" style="background:rgba(56,189,248,0.4); border:1px dashed #38bdf8;"></div> Audio VAD</div>
      <div class="legend-item"><div class="legend-dot" style="background:rgba(251,191,36,0.3); border:1px dashed #fbbf24;"></div> Threshold (0.5)</div>
    </div>
    <canvas id="chart-prob-old"></canvas>
  </div>
  <div class="chart-container">
    <h2>4b. Speech Probability — NEW (var &times; MAR &times; ZCR)</h2>
    <div class="legend-custom">
      <div class="legend-item"><div class="legend-dot" style="background:rgba(74,222,128,0.7);"></div> Raw Probability</div>
      <div class="legend-item"><div class="legend-dot" style="background:#4ade80;"></div> Smoothed (hold decay)</div>
      <div class="legend-item"><div class="legend-dot" style="background:rgba(56,189,248,0.4); border:1px dashed #38bdf8;"></div> Audio VAD</div>
      <div class="legend-item"><div class="legend-dot" style="background:rgba(251,191,36,0.3); border:1px dashed #fbbf24;"></div> Threshold (0.5)</div>
    </div>
    <canvas id="chart-prob-new"></canvas>
  </div>
</div>

<div class="chart-container">
  <h2>5. V-VAD Binary Detection: OLD vs NEW vs Audio VAD</h2>
  <div class="legend-custom">
    <div class="legend-item"><div class="legend-dot" style="background:rgba(248,113,113,0.5);"></div> OLD V-VAD</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(74,222,128,0.6);"></div> NEW V-VAD</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(56,189,248,0.4);"></div> Audio VAD (reference)</div>
  </div>
  <canvas id="chart-vad"></canvas>
</div>

<div class="chart-container">
  <h2>6. Head Pose (Euler Angles)</h2>
  <div class="legend-custom">
    <div class="legend-item"><div class="legend-dot" style="background:#f87171;"></div> Yaw</div>
    <div class="legend-item"><div class="legend-dot" style="background:#4ade80;"></div> Pitch</div>
    <div class="legend-item"><div class="legend-dot" style="background:#38bdf8;"></div> Roll</div>
  </div>
  <canvas id="chart-pose"></canvas>
</div>

<script>
const ALL_DATA = {data_json};
const OLD_P = {old_params_json};
const NEW_P = {new_params_json};

let charts = {{}};
const selector = document.getElementById('recording-select');

function prevRecording() {{
  const idx = selector.selectedIndex;
  if (idx > 0) {{ selector.selectedIndex = idx - 1; loadRecording(); }}
}}
function nextRecording() {{
  const idx = selector.selectedIndex;
  if (idx < selector.options.length - 1) {{ selector.selectedIndex = idx + 1; loadRecording(); }}
}}

selector.addEventListener('change', loadRecording);
document.addEventListener('keydown', e => {{
  if (e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') {{ e.preventDefault(); prevRecording(); }}
  if (e.key === 'ArrowRight') {{ e.preventDefault(); nextRecording(); }}
}});

function destroyCharts() {{
  Object.values(charts).forEach(c => c.destroy());
  charts = {{}};
}}

function computeStats(vvad, avad) {{
  const n = vvad.length;
  let tp=0, tn=0, fp=0, fn=0;
  for (let i=0; i<n; i++) {{
    if (vvad[i]===1 && avad[i]===1) tp++;
    else if (vvad[i]===0 && avad[i]===0) tn++;
    else if (vvad[i]===1 && avad[i]===0) fp++;
    else fn++;
  }}
  const agree = ((tp+tn)/n*100);
  const prec = (tp+fp)>0 ? (tp/(tp+fp)*100) : 100;
  const rec = (tp+fn)>0 ? (tp/(tp+fn)*100) : 100;
  const f1 = (2*tp+fp+fn)>0 ? (2*tp/(2*tp+fp+fn)*100) : 100;
  const fpr = (fp+tn)>0 ? (fp/(fp+tn)*100) : 0;
  return {{agree, prec, rec, f1, fpr, tp, tn, fp, fn}};
}}

function arrow(oldV, newV, higherBetter) {{
  const diff = newV - oldV;
  if (Math.abs(diff) < 0.1) return '<span class="stat-arrow">=</span>';
  const good = higherBetter ? diff > 0 : diff < 0;
  const color = good ? '#4ade80' : '#f87171';
  const sym = diff > 0 ? '&#9650;' : '&#9660;';
  return `<span style="color:${{color}};font-size:0.8em;">${{sym}} ${{Math.abs(diff).toFixed(1)}}</span>`;
}}

function loadRecording() {{
  const key = selector.value;
  const D = ALL_DATA[key];
  if (!D) return;

  destroyCharts();

  const labels = D.times.map(t => t.toFixed(3));
  const n = D.times.length;
  const dur = D.times[n-1];

  const oldS = computeStats(D.old_vvad, D.audio_vad);
  const newS = computeStats(D.new_vvad, D.audio_vad);
  const audioAct = D.audio_vad.filter(v=>v===1).length;

  document.getElementById('stats-row').innerHTML = `
    <div class="stat-card">
      <div class="stat-value" style="color:var(--accent);">${{dur.toFixed(1)}}s</div>
      <div class="stat-label">Duration (${{n}} frames)</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color:var(--accent2);">${{(audioAct/n*100).toFixed(1)}}%</div>
      <div class="stat-label">Audio VAD Active</div>
    </div>
    <div class="stat-card">
      <div class="stat-label" style="margin-bottom:4px">Agreement %</div>
      <div class="stat-compare">
        <span class="stat-old">${{oldS.agree.toFixed(1)}}%</span>
        <span class="stat-arrow">&#8594;</span>
        <span class="stat-new">${{newS.agree.toFixed(1)}}%</span>
        ${{arrow(oldS.agree, newS.agree, true)}}
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-label" style="margin-bottom:4px">Precision %</div>
      <div class="stat-compare">
        <span class="stat-old">${{oldS.prec.toFixed(1)}}%</span>
        <span class="stat-arrow">&#8594;</span>
        <span class="stat-new">${{newS.prec.toFixed(1)}}%</span>
        ${{arrow(oldS.prec, newS.prec, true)}}
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-label" style="margin-bottom:4px">Recall %</div>
      <div class="stat-compare">
        <span class="stat-old">${{oldS.rec.toFixed(1)}}%</span>
        <span class="stat-arrow">&#8594;</span>
        <span class="stat-new">${{newS.rec.toFixed(1)}}%</span>
        ${{arrow(oldS.rec, newS.rec, true)}}
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-label" style="margin-bottom:4px">F1 %</div>
      <div class="stat-compare">
        <span class="stat-old">${{oldS.f1.toFixed(1)}}%</span>
        <span class="stat-arrow">&#8594;</span>
        <span class="stat-new">${{newS.f1.toFixed(1)}}%</span>
        ${{arrow(oldS.f1, newS.f1, true)}}
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-label" style="margin-bottom:4px">False Positive %</div>
      <div class="stat-compare">
        <span class="stat-old">${{oldS.fpr.toFixed(1)}}%</span>
        <span class="stat-arrow">&#8594;</span>
        <span class="stat-new">${{newS.fpr.toFixed(1)}}%</span>
        ${{arrow(oldS.fpr, newS.fpr, false)}}
      </div>
    </div>
  `;

  const commonOpts = {{
    responsive: true,
    maintainAspectRatio: false,
    animation: {{ duration: 250 }},
    interaction: {{ mode: 'index', intersect: false }},
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{
        backgroundColor: '#1e293b',
        titleColor: '#f1f5f9',
        bodyColor: '#94a3b8',
        borderColor: '#334155',
        borderWidth: 1,
        callbacks: {{ title: (items) => `t = ${{items[0].label}}s` }}
      }}
    }},
    scales: {{
      x: {{
        title: {{ display: true, text: 'Time (s)', color: '#94a3b8' }},
        ticks: {{ color: '#64748b', maxTicksLimit: 20, callback: v => D.times[v]?.toFixed(1) }},
        grid: {{ color: 'rgba(100,116,139,0.12)' }},
      }},
      y: {{
        ticks: {{ color: '#94a3b8' }},
        grid: {{ color: 'rgba(100,116,139,0.12)' }},
      }}
    }},
    elements: {{ point: {{ radius: 0 }}, line: {{ borderWidth: 1.5 }} }}
  }};

  // 1. MAR with both thresholds
  charts.mar = new Chart(document.getElementById('chart-mar'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'Raw MAR', data: D.raw_mar, borderColor: 'rgba(148,163,184,0.5)', fill: false, borderWidth: 1 }},
        {{ label: 'Frontalized MAR', data: D.frontal_mar, borderColor: '#38bdf8', fill: false, borderWidth: 1.5 }},
      ]
    }},
    options: {{
      ...commonOpts,
      plugins: {{
        ...commonOpts.plugins,
        annotation: {{ annotations: {{
          oldThr: {{ type:'line', yMin: OLD_P.mar_activity_threshold, yMax: OLD_P.mar_activity_threshold,
                    borderColor:'rgba(248,113,113,0.5)', borderWidth:1.5, borderDash:[6,4],
                    label: {{ display:true, content:'OLD MAR='+OLD_P.mar_activity_threshold, position:'start',
                             backgroundColor:'rgba(248,113,113,0.12)', color:'#f87171', font:{{size:10}} }} }},
          newThr: {{ type:'line', yMin: NEW_P.mar_activity_threshold, yMax: NEW_P.mar_activity_threshold,
                    borderColor:'rgba(74,222,128,0.5)', borderWidth:1.5, borderDash:[6,4],
                    label: {{ display:true, content:'NEW MAR='+NEW_P.mar_activity_threshold, position:'end',
                             backgroundColor:'rgba(74,222,128,0.12)', color:'#4ade80', font:{{size:10}} }} }}
        }} }}
      }},
      scales: {{ ...commonOpts.scales, y: {{ ...commonOpts.scales.y, title: {{ display:true, text:'MAR', color:'#94a3b8' }} }} }}
    }}
  }});

  // 2a. Variance OLD
  charts.varOld = new Chart(document.getElementById('chart-var-old'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'Variance (OLD)', data: D.old_var, borderColor: '#f87171', backgroundColor: 'rgba(248,113,113,0.1)', fill: true, borderWidth: 1.2 }},
      ]
    }},
    options: {{
      ...commonOpts,
      plugins: {{
        ...commonOpts.plugins,
        annotation: {{ annotations: {{
          thr: {{ type:'line', yMin: OLD_P.var_threshold, yMax: OLD_P.var_threshold,
                 borderColor:'rgba(248,113,113,0.6)', borderWidth:1.5, borderDash:[5,3],
                 label: {{ display:true, content:'thr='+OLD_P.var_threshold, position:'start',
                          backgroundColor:'rgba(248,113,113,0.12)', color:'#f87171', font:{{size:10}} }} }}
        }} }}
      }},
      scales: {{ ...commonOpts.scales, y: {{ ...commonOpts.scales.y, title: {{ display:true, text:'Var', color:'#94a3b8' }} }} }}
    }}
  }});

  // 2b. Variance NEW
  charts.varNew = new Chart(document.getElementById('chart-var-new'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'Variance (NEW)', data: D.new_var, borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.1)', fill: true, borderWidth: 1.2 }},
      ]
    }},
    options: {{
      ...commonOpts,
      plugins: {{
        ...commonOpts.plugins,
        annotation: {{ annotations: {{
          thr: {{ type:'line', yMin: NEW_P.var_threshold, yMax: NEW_P.var_threshold,
                 borderColor:'rgba(74,222,128,0.6)', borderWidth:1.5, borderDash:[5,3],
                 label: {{ display:true, content:'thr='+NEW_P.var_threshold, position:'start',
                          backgroundColor:'rgba(74,222,128,0.12)', color:'#4ade80', font:{{size:10}} }} }}
        }} }}
      }},
      scales: {{ ...commonOpts.scales, y: {{ ...commonOpts.scales.y, title: {{ display:true, text:'Var', color:'#94a3b8' }} }} }}
    }}
  }});

  // 3. ZCR
  charts.zcr = new Chart(document.getElementById('chart-zcr'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'ZCR', data: D.new_zcr, borderColor: '#fb923c', backgroundColor: 'rgba(251,146,60,0.1)', fill: true, borderWidth: 1.5 }},
      ]
    }},
    options: {{
      ...commonOpts,
      plugins: {{
        ...commonOpts.plugins,
        annotation: {{ annotations: {{
          thr: {{ type:'line', yMin: NEW_P.min_zcr, yMax: NEW_P.min_zcr,
                 borderColor:'rgba(251,191,36,0.6)', borderWidth:1.5, borderDash:[5,3],
                 label: {{ display:true, content:'min_zcr='+NEW_P.min_zcr, position:'start',
                          backgroundColor:'rgba(251,191,36,0.12)', color:'#fbbf24', font:{{size:10}} }} }}
        }} }}
      }},
      scales: {{ ...commonOpts.scales, y: {{ ...commonOpts.scales.y, min: 0, title: {{ display:true, text:'ZCR', color:'#94a3b8' }} }} }}
    }}
  }});

  // 4a. Speech Probability — OLD
  charts.probOld = new Chart(document.getElementById('chart-prob-old'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'Raw Prob (OLD)', data: D.old_prob, borderColor: 'rgba(248,113,113,0.5)',
           backgroundColor: 'rgba(248,113,113,0.08)', fill: true, borderWidth: 1 }},
        {{ label: 'Smoothed Prob (OLD)', data: D.old_smoothed, borderColor: '#f87171',
           fill: false, borderWidth: 2 }},
        {{ label: 'Audio VAD', data: D.audio_vad, borderColor: 'rgba(56,189,248,0.35)',
           backgroundColor: 'rgba(56,189,248,0.05)', fill: true, stepped: true, borderWidth: 1, borderDash: [4,3] }},
      ]
    }},
    options: {{
      ...commonOpts,
      plugins: {{
        ...commonOpts.plugins,
        annotation: {{ annotations: {{
          thr: {{ type:'line', yMin: 0.5, yMax: 0.5,
                 borderColor:'rgba(251,191,36,0.5)', borderWidth:1.5, borderDash:[6,4],
                 label: {{ display:true, content:'Decision boundary (0.5)', position:'start',
                          backgroundColor:'rgba(251,191,36,0.12)', color:'#fbbf24', font:{{size:9}} }} }}
        }} }}
      }},
      scales: {{
        ...commonOpts.scales,
        y: {{ ...commonOpts.scales.y, min: 0, max: 1.05,
              title: {{ display:true, text:'P(speech)', color:'#94a3b8' }},
              ticks: {{ color: '#94a3b8', stepSize: 0.25 }} }}
      }}
    }}
  }});

  // 4b. Speech Probability — NEW
  charts.probNew = new Chart(document.getElementById('chart-prob-new'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'Raw Prob (NEW)', data: D.new_prob, borderColor: 'rgba(74,222,128,0.5)',
           backgroundColor: 'rgba(74,222,128,0.08)', fill: true, borderWidth: 1 }},
        {{ label: 'Smoothed Prob (NEW)', data: D.new_smoothed, borderColor: '#4ade80',
           fill: false, borderWidth: 2 }},
        {{ label: 'Audio VAD', data: D.audio_vad, borderColor: 'rgba(56,189,248,0.35)',
           backgroundColor: 'rgba(56,189,248,0.05)', fill: true, stepped: true, borderWidth: 1, borderDash: [4,3] }},
      ]
    }},
    options: {{
      ...commonOpts,
      plugins: {{
        ...commonOpts.plugins,
        annotation: {{ annotations: {{
          thr: {{ type:'line', yMin: 0.5, yMax: 0.5,
                 borderColor:'rgba(251,191,36,0.5)', borderWidth:1.5, borderDash:[6,4],
                 label: {{ display:true, content:'Decision boundary (0.5)', position:'start',
                          backgroundColor:'rgba(251,191,36,0.12)', color:'#fbbf24', font:{{size:9}} }} }}
        }} }}
      }},
      scales: {{
        ...commonOpts.scales,
        y: {{ ...commonOpts.scales.y, min: 0, max: 1.05,
              title: {{ display:true, text:'P(speech)', color:'#94a3b8' }},
              ticks: {{ color: '#94a3b8', stepSize: 0.25 }} }}
      }}
    }}
  }});

  // 5. V-VAD comparison: OLD vs NEW vs Audio
  charts.vad = new Chart(document.getElementById('chart-vad'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'OLD V-VAD', data: D.old_vvad.map(v => v * 0.85), borderColor: '#f87171',
           backgroundColor: 'rgba(248,113,113,0.12)', fill: true, stepped: true, borderWidth: 1.5 }},
        {{ label: 'NEW V-VAD', data: D.new_vvad, borderColor: '#4ade80',
           backgroundColor: 'rgba(74,222,128,0.12)', fill: true, stepped: true, borderWidth: 2 }},
        {{ label: 'Audio VAD', data: D.audio_vad.map(v => v * 1.1), borderColor: 'rgba(56,189,248,0.5)',
           backgroundColor: 'rgba(56,189,248,0.06)', fill: true, stepped: true, borderWidth: 1.5, borderDash: [4,3] }},
      ]
    }},
    options: {{
      ...commonOpts,
      scales: {{
        ...commonOpts.scales,
        y: {{
          ...commonOpts.scales.y,
          min: -0.1, max: 1.3,
          title: {{ display: true, text: 'Active / Inactive', color: '#94a3b8' }},
          ticks: {{ color: '#94a3b8', stepSize: 0.5, callback: v => v===0 ? 'Inactive' : v===1 ? 'Active' : '' }}
        }}
      }}
    }}
  }});

  // 5. Pose
  charts.pose = new Chart(document.getElementById('chart-pose'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'Yaw', data: D.yaw, borderColor: '#f87171', fill: false, borderWidth: 1.5 }},
        {{ label: 'Pitch', data: D.pitch, borderColor: '#4ade80', fill: false, borderWidth: 1.5 }},
        {{ label: 'Roll', data: D.roll, borderColor: '#38bdf8', fill: false, borderWidth: 1.5 }},
      ]
    }},
    options: {{
      ...commonOpts,
      scales: {{ ...commonOpts.scales, y: {{ ...commonOpts.scales.y, title: {{ display:true, text:'Degrees', color:'#94a3b8' }} }} }}
    }}
  }});
}}

loadRecording();
</script>

</body>
</html>"""

    output_path = os.path.join(base_dir, 'vvad_dashboard.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nDashboard saved to: {output_path}")
    print(f"Recordings included: {len(all_data)}")


if __name__ == '__main__':
    main()
