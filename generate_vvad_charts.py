#!/usr/bin/env python3
"""
Generate an interactive HTML dashboard with V-VAD algorithm metrics charts.
Reads dump CSVs and computes: MAR (raw + frontalized), variance, mean, V-VAD state.

Usage: python generate_vvad_charts.py [dumps_dir]
Default dumps_dir: dumps_marek_short_1
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from collections import deque

# ── Algorithm functions (same as in multiface_distance_render_v2.py) ──────

def frontalize_mouth_landmarks(points_3d):
    """Frontalize mouth landmarks using direct 3D geometry."""
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
    """Compute Mouth Aspect Ratio from inner mouth landmarks 60-67."""
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


def compute_raw_mar(df_frame, face_idx=0):
    """Compute MAR from raw (non-frontalized) 2D points."""
    face_data = df_frame[(df_frame['face_idx'] == face_idx)]
    points = {}
    for _, row in face_data.iterrows():
        pt_idx = int(row['point_type'])
        points[pt_idx] = (row['x'], row['y'])
    return compute_mar(points)


def compute_frontalized_mar(df_frame, face_idx=0):
    """Compute MAR from frontalized mouth landmarks."""
    face_data = df_frame[(df_frame['face_idx'] == face_idx)]
    points_3d = {}
    for _, row in face_data.iterrows():
        pt_idx = int(row['point_type'])
        points_3d[pt_idx] = (row['x'], row['y'], row['z'])
    frontalized = frontalize_mouth_landmarks(points_3d)
    if frontalized is None:
        return None
    return compute_mar(frontalized)


# ── Main processing ──────────────────────────────────────────────────────

def process_dumps(dumps_dir):
    """Process dump CSVs and compute all V-VAD metrics."""
    print(f"Reading data from: {dumps_dir}")
    
    df_mouth = pd.read_csv(os.path.join(dumps_dir, 'mouth_position.csv'))
    df_pose = pd.read_csv(os.path.join(dumps_dir, 'face_position.csv'))
    df_vad = pd.read_csv(os.path.join(dumps_dir, 'vad.csv'))
    
    timestamps = sorted(df_mouth['seconds'].unique())
    print(f"  Timestamps: {len(timestamps)}, range: {timestamps[0]:.3f} - {timestamps[-1]:.3f}s")
    
    # V-VAD parameters (same as in the renderer)
    WINDOW_SECONDS = 0.5
    FPS = 30.0
    VAR_THRESHOLD = 0.0005
    MAR_ACTIVITY_THRESHOLD = 0.12
    HOLD_SECONDS = 1.0
    
    window_size = max(3, int(WINDOW_SECONDS * FPS))
    hold_frames = max(1, int(HOLD_SECONDS * FPS))
    
    # Output arrays
    times = []
    raw_mar_values = []
    frontal_mar_values = []
    mar_variance_values = []
    mar_mean_values = []
    raw_speaking_values = []
    vvad_output_values = []
    
    # Sliding window state
    mar_history = deque(maxlen=window_size)
    hold_counter = 0
    
    print("  Computing metrics per frame...")
    for t in timestamps:
        frame_data = df_mouth[df_mouth['seconds'] == t]
        
        raw_mar = compute_raw_mar(frame_data, face_idx=0)
        frontal_mar = compute_frontalized_mar(frame_data, face_idx=0)
        
        times.append(round(t, 6))
        raw_mar_values.append(round(raw_mar, 6) if raw_mar is not None else None)
        frontal_mar_values.append(round(frontal_mar, 6) if frontal_mar is not None else None)
        
        # V-VAD sliding window
        if frontal_mar is not None:
            mar_history.append(frontal_mar)
        
        if len(mar_history) >= 3:
            values = np.array(mar_history)
            mar_var = float(np.var(values))
            mar_mean = float(np.mean(values))
            raw_speaking = (mar_var > VAR_THRESHOLD) and (mar_mean > MAR_ACTIVITY_THRESHOLD)
        else:
            mar_var = 0.0
            mar_mean = 0.0
            raw_speaking = False
        
        mar_variance_values.append(round(mar_var, 8))
        mar_mean_values.append(round(mar_mean, 6))
        raw_speaking_values.append(1 if raw_speaking else 0)
        
        # Hysteresis
        if raw_speaking:
            hold_counter = hold_frames
        else:
            if hold_counter > 0:
                hold_counter -= 1
        
        vvad_output_values.append(1 if hold_counter > 0 else 0)
    
    # Audio VAD — align to our timestamps
    audio_vad_values = []
    vad_times = df_vad['seconds'].values
    vad_vals = df_vad['vadDagcDecFinal'].values
    for t in times:
        idx = np.argmin(np.abs(vad_times - t))
        audio_vad_values.append(int(vad_vals[idx]))
    
    # Euler angles
    yaw_values = []
    pitch_values = []
    roll_values = []
    pose_times = df_pose['seconds'].values
    for t in times:
        idx = np.argmin(np.abs(pose_times - t))
        row = df_pose.iloc[idx]
        yaw_values.append(round(float(row['yaw']), 3))
        pitch_values.append(round(float(row['pitch']), 3))
        roll_values.append(round(float(row['roll']), 3))
    
    print(f"  Done. Frames processed: {len(times)}")
    
    return {
        'times': times,
        'raw_mar': raw_mar_values,
        'frontal_mar': frontal_mar_values,
        'mar_variance': mar_variance_values,
        'mar_mean': mar_mean_values,
        'raw_speaking': raw_speaking_values,
        'vvad_output': vvad_output_values,
        'audio_vad': audio_vad_values,
        'yaw': yaw_values,
        'pitch': pitch_values,
        'roll': roll_values,
        'params': {
            'window_seconds': WINDOW_SECONDS,
            'var_threshold': VAR_THRESHOLD,
            'mar_activity_threshold': MAR_ACTIVITY_THRESHOLD,
            'hold_seconds': HOLD_SECONDS,
            'fps': FPS,
        }
    }


def generate_html(data, output_path):
    """Generate interactive HTML dashboard with Chart.js."""
    
    params = data['params']
    data_json = json.dumps(data, indent=None)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V-VAD Algorithm Metrics Dashboard</title>
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
    font-size: 2em;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
  }}
  h2 {{
    font-size: 1.3em;
    color: var(--accent);
    margin-bottom: 8px;
  }}
  .subtitle {{ color: var(--muted); font-size: 0.95em; margin-bottom: 20px; }}
  .chart-container {{
    background: var(--card);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    border: 1px solid #334155;
    position: relative;
  }}
  .chart-container canvas {{
    max-height: 220px;
  }}
  .params-bar {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    background: var(--card);
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 16px;
    border: 1px solid #334155;
  }}
  .param {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.9em;
  }}
  .param-label {{ color: var(--muted); }}
  .param-value {{
    color: var(--yellow);
    font-weight: 700;
    font-family: 'Consolas', monospace;
  }}
  .legend-custom {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 6px;
    font-size: 0.85em;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 5px;
  }}
  .legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }}
  .stats-row {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 16px;
  }}
  .stat-card {{
    background: var(--card);
    border-radius: 10px;
    padding: 14px 20px;
    border: 1px solid #334155;
    flex: 1;
    min-width: 180px;
    text-align: center;
  }}
  .stat-value {{
    font-size: 1.8em;
    font-weight: 800;
    font-family: 'Consolas', monospace;
  }}
  .stat-label {{
    color: var(--muted);
    font-size: 0.85em;
    margin-top: 4px;
  }}
</style>
</head>
<body>

<h1>V-VAD Algorithm Metrics Dashboard</h1>
<p class="subtitle">Visual Voice Activity Detection — computed from dump data</p>

<div class="params-bar">
  <div class="param"><span class="param-label">Window:</span><span class="param-value">{params['window_seconds']}s</span></div>
  <div class="param"><span class="param-label">Var threshold:</span><span class="param-value">{params['var_threshold']}</span></div>
  <div class="param"><span class="param-label">MAR threshold:</span><span class="param-value">{params['mar_activity_threshold']}</span></div>
  <div class="param"><span class="param-label">Hold time:</span><span class="param-value">{params['hold_seconds']}s</span></div>
  <div class="param"><span class="param-label">FPS:</span><span class="param-value">{params['fps']}</span></div>
</div>

<div id="stats-row" class="stats-row"></div>

<!-- Chart 1: MAR over time -->
<div class="chart-container">
  <h2>Mouth Aspect Ratio (MAR) over Time</h2>
  <div class="legend-custom">
    <div class="legend-item"><div class="legend-dot" style="background:#94a3b8;"></div> Raw MAR</div>
    <div class="legend-item"><div class="legend-dot" style="background:#38bdf8;"></div> Frontalized MAR</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(251,191,36,0.15); border:1px dashed #fbbf24;"></div> MAR activity threshold</div>
  </div>
  <canvas id="chart-mar"></canvas>
</div>

<!-- Chart 2: MAR Variance -->
<div class="chart-container">
  <h2>MAR Variance (Sliding Window) vs Threshold</h2>
  <div class="legend-custom">
    <div class="legend-item"><div class="legend-dot" style="background:#a78bfa;"></div> MAR Variance</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(248,113,113,0.15); border:1px dashed #f87171;"></div> Variance threshold ({params['var_threshold']})</div>
  </div>
  <canvas id="chart-var"></canvas>
</div>

<!-- Chart 3: Sliding Window Mean MAR -->
<div class="chart-container">
  <h2>Sliding Window Mean MAR vs Threshold</h2>
  <div class="legend-custom">
    <div class="legend-item"><div class="legend-dot" style="background:#4ade80;"></div> Mean MAR (window)</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(251,191,36,0.15); border:1px dashed #fbbf24;"></div> MAR activity threshold ({params['mar_activity_threshold']})</div>
  </div>
  <canvas id="chart-mean"></canvas>
</div>

<!-- Chart 4: Detection stages -->
<div class="chart-container">
  <h2>V-VAD Detection Stages &amp; Audio VAD Comparison</h2>
  <div class="legend-custom">
    <div class="legend-item"><div class="legend-dot" style="background:rgba(251,191,36,0.6);"></div> Raw detection (Var &gt; thr AND Mean &gt; thr)</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(74,222,128,0.6);"></div> V-VAD output (with hysteresis)</div>
    <div class="legend-item"><div class="legend-dot" style="background:rgba(56,189,248,0.4);"></div> Audio VAD (reference)</div>
  </div>
  <canvas id="chart-vad"></canvas>
</div>

<!-- Chart 5: Head pose -->
<div class="chart-container">
  <h2>Head Pose (Euler Angles)</h2>
  <div class="legend-custom">
    <div class="legend-item"><div class="legend-dot" style="background:#f87171;"></div> Yaw (left/right)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#4ade80;"></div> Pitch (up/down)</div>
    <div class="legend-item"><div class="legend-dot" style="background:#38bdf8;"></div> Roll (tilt)</div>
  </div>
  <canvas id="chart-pose"></canvas>
</div>

<script>
const DATA = {data_json};

// Compute stats
const frontalMAR = DATA.frontal_mar.filter(v => v !== null);
const vvadActive = DATA.vvad_output.filter(v => v === 1).length;
const audioActive = DATA.audio_vad.filter(v => v === 1).length;
const totalFrames = DATA.times.length;
const duration = DATA.times[DATA.times.length - 1];

// Compute agreement between V-VAD and Audio VAD
let agree = 0;
for (let i = 0; i < totalFrames; i++) {{
  if (DATA.vvad_output[i] === DATA.audio_vad[i]) agree++;
}}
const agreementPct = ((agree / totalFrames) * 100).toFixed(1);

document.getElementById('stats-row').innerHTML = `
  <div class="stat-card"><div class="stat-value" style="color:var(--accent);">${{duration.toFixed(1)}}s</div><div class="stat-label">Duration</div></div>
  <div class="stat-card"><div class="stat-value" style="color:var(--white);">${{totalFrames}}</div><div class="stat-label">Total Frames</div></div>
  <div class="stat-card"><div class="stat-value" style="color:var(--green);">${{(vvadActive/totalFrames*100).toFixed(1)}}%</div><div class="stat-label">V-VAD Active</div></div>
  <div class="stat-card"><div class="stat-value" style="color:var(--accent2);">${{(audioActive/totalFrames*100).toFixed(1)}}%</div><div class="stat-label">Audio VAD Active</div></div>
  <div class="stat-card"><div class="stat-value" style="color:var(--yellow);">${{agreementPct}}%</div><div class="stat-label">V-VAD / Audio Agreement</div></div>
  <div class="stat-card"><div class="stat-value" style="color:var(--accent);">${{(frontalMAR.reduce((a,b)=>a+b,0)/frontalMAR.length).toFixed(3)}}</div><div class="stat-label">Mean Frontal MAR</div></div>
`;

// Common chart options
const commonOpts = {{
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  interaction: {{
    mode: 'index',
    intersect: false,
  }},
  plugins: {{
    legend: {{ display: false }},
    tooltip: {{
      backgroundColor: '#1e293b',
      titleColor: '#f1f5f9',
      bodyColor: '#94a3b8',
      borderColor: '#334155',
      borderWidth: 1,
      callbacks: {{
        title: (items) => `t = ${{items[0].label}}s`,
      }}
    }}
  }},
  scales: {{
    x: {{
      title: {{ display: true, text: 'Time (s)', color: '#94a3b8' }},
      ticks: {{ color: '#64748b', maxTicksLimit: 25, callback: v => DATA.times[v]?.toFixed(1) }},
      grid: {{ color: 'rgba(100,116,139,0.15)' }},
    }},
    y: {{
      ticks: {{ color: '#94a3b8' }},
      grid: {{ color: 'rgba(100,116,139,0.15)' }},
    }}
  }},
  elements: {{
    point: {{ radius: 0 }},
    line: {{ borderWidth: 1.5 }},
  }}
}};

const labels = DATA.times.map(t => t.toFixed(3));

// ── Chart 1: MAR ──
new Chart(document.getElementById('chart-mar'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'Raw MAR',
        data: DATA.raw_mar,
        borderColor: 'rgba(148,163,184,0.6)',
        backgroundColor: 'rgba(148,163,184,0.1)',
        fill: false,
        borderWidth: 1,
      }},
      {{
        label: 'Frontalized MAR',
        data: DATA.frontal_mar,
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56,189,248,0.1)',
        fill: false,
        borderWidth: 1.5,
      }},
    ]
  }},
  options: {{
    ...commonOpts,
    plugins: {{
      ...commonOpts.plugins,
      annotation: {{
        annotations: {{
          marThreshold: {{
            type: 'line',
            yMin: {params['mar_activity_threshold']},
            yMax: {params['mar_activity_threshold']},
            borderColor: 'rgba(251,191,36,0.5)',
            borderWidth: 2,
            borderDash: [6, 4],
            label: {{
              display: true,
              content: 'MAR threshold = {params['mar_activity_threshold']}',
              position: 'start',
              backgroundColor: 'rgba(251,191,36,0.15)',
              color: '#fbbf24',
              font: {{ size: 11 }},
            }}
          }}
        }}
      }}
    }},
    scales: {{
      ...commonOpts.scales,
      y: {{ ...commonOpts.scales.y, title: {{ display: true, text: 'MAR', color: '#94a3b8' }} }}
    }}
  }}
}});

// ── Chart 2: Variance ──
new Chart(document.getElementById('chart-var'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'MAR Variance',
        data: DATA.mar_variance,
        borderColor: '#a78bfa',
        backgroundColor: 'rgba(167,139,250,0.15)',
        fill: true,
        borderWidth: 1.5,
      }},
    ]
  }},
  options: {{
    ...commonOpts,
    plugins: {{
      ...commonOpts.plugins,
      annotation: {{
        annotations: {{
          varThreshold: {{
            type: 'line',
            yMin: {params['var_threshold']},
            yMax: {params['var_threshold']},
            borderColor: 'rgba(248,113,113,0.7)',
            borderWidth: 2,
            borderDash: [6, 4],
            label: {{
              display: true,
              content: 'var_threshold = {params['var_threshold']}',
              position: 'start',
              backgroundColor: 'rgba(248,113,113,0.15)',
              color: '#f87171',
              font: {{ size: 11 }},
            }}
          }}
        }}
      }}
    }},
    scales: {{
      ...commonOpts.scales,
      y: {{ ...commonOpts.scales.y, title: {{ display: true, text: 'Variance', color: '#94a3b8' }} }}
    }}
  }}
}});

// ── Chart 3: Mean MAR ──
new Chart(document.getElementById('chart-mean'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'Mean MAR (window)',
        data: DATA.mar_mean,
        borderColor: '#4ade80',
        backgroundColor: 'rgba(74,222,128,0.1)',
        fill: false,
        borderWidth: 1.5,
      }},
    ]
  }},
  options: {{
    ...commonOpts,
    plugins: {{
      ...commonOpts.plugins,
      annotation: {{
        annotations: {{
          marThreshold: {{
            type: 'line',
            yMin: {params['mar_activity_threshold']},
            yMax: {params['mar_activity_threshold']},
            borderColor: 'rgba(251,191,36,0.5)',
            borderWidth: 2,
            borderDash: [6, 4],
            label: {{
              display: true,
              content: 'mar_activity_threshold = {params['mar_activity_threshold']}',
              position: 'start',
              backgroundColor: 'rgba(251,191,36,0.15)',
              color: '#fbbf24',
              font: {{ size: 11 }},
            }}
          }}
        }}
      }}
    }},
    scales: {{
      ...commonOpts.scales,
      y: {{ ...commonOpts.scales.y, title: {{ display: true, text: 'Mean MAR', color: '#94a3b8' }} }}
    }}
  }}
}});

// ── Chart 4: VAD comparison ──
new Chart(document.getElementById('chart-vad'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'Raw Detection',
        data: DATA.raw_speaking.map(v => v * 0.9),
        borderColor: 'rgba(251,191,36,0.7)',
        backgroundColor: 'rgba(251,191,36,0.2)',
        fill: true,
        stepped: true,
        borderWidth: 1,
      }},
      {{
        label: 'V-VAD Output (hysteresis)',
        data: DATA.vvad_output,
        borderColor: '#4ade80',
        backgroundColor: 'rgba(74,222,128,0.15)',
        fill: true,
        stepped: true,
        borderWidth: 2,
      }},
      {{
        label: 'Audio VAD (reference)',
        data: DATA.audio_vad.map(v => v * 1.1),
        borderColor: 'rgba(56,189,248,0.5)',
        backgroundColor: 'rgba(56,189,248,0.08)',
        fill: true,
        stepped: true,
        borderWidth: 1.5,
        borderDash: [4, 3],
      }},
    ]
  }},
  options: {{
    ...commonOpts,
    scales: {{
      ...commonOpts.scales,
      y: {{
        ...commonOpts.scales.y,
        min: -0.1,
        max: 1.3,
        title: {{ display: true, text: 'Active (1) / Inactive (0)', color: '#94a3b8' }},
        ticks: {{
          color: '#94a3b8',
          stepSize: 0.5,
          callback: v => v === 0 ? 'Inactive' : v === 1 ? 'Active' : '',
        }}
      }}
    }}
  }}
}});

// ── Chart 5: Head Pose ──
new Chart(document.getElementById('chart-pose'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'Yaw',
        data: DATA.yaw,
        borderColor: '#f87171',
        fill: false,
        borderWidth: 1.5,
      }},
      {{
        label: 'Pitch',
        data: DATA.pitch,
        borderColor: '#4ade80',
        fill: false,
        borderWidth: 1.5,
      }},
      {{
        label: 'Roll',
        data: DATA.roll,
        borderColor: '#38bdf8',
        fill: false,
        borderWidth: 1.5,
      }},
    ]
  }},
  options: {{
    ...commonOpts,
    scales: {{
      ...commonOpts.scales,
      y: {{ ...commonOpts.scales.y, title: {{ display: true, text: 'Degrees (°)', color: '#94a3b8' }} }}
    }}
  }}
}});
</script>

</body>
</html>"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Dashboard saved to: {output_path}")


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    dumps_dir = sys.argv[1] if len(sys.argv) > 1 else 'dumps_marek_short_1'
    dumps_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dumps_dir)
    
    if not os.path.isdir(dumps_path):
        print(f"Error: Directory not found: {dumps_path}")
        sys.exit(1)
    
    data = process_dumps(dumps_path)
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'vvad_charts_{dumps_dir}.html')
    generate_html(data, output_path)
