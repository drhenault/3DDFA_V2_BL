#!/usr/bin/env python3
"""
Generate an HTML documentation page for the V-VAD DNN training results.

Reads the training history JSON produced by train_vvad_dnn.py and produces
a self-contained HTML file with interactive charts (Chart.js) and detailed
documentation of the model architecture, preprocessing pipeline, and results.

Usage:
    python generate_vvad_dnn_doc.py [--input vvad_dnn_training_history.json]
                                     [--output vvad_dnn_report.html]
"""

import argparse
import json
import os
import sys


def generate_html(data: dict) -> str:
    """Return the full HTML string with embedded training data."""

    cfg = data["config"]
    d = data["data"]
    epochs = data["epoch_history"]
    best_epoch = data["best_epoch"]
    last_epoch = data.get("last_epoch", len(epochs))
    t_time = data["training_time_s"]
    test = data["test_results"]
    cm = test["confusion_matrix"]
    no_split = cfg.get("no_split", False) or d.get("no_split", False)

    # Label used in charts / headings when there is no separate validation set
    eval_label = "Eval (train)" if no_split else "Val"
    eval_set_descr = "training set (all data)" if no_split else "held-out test set"

    # Pre-format epoch arrays for JS
    epoch_nums = [e["epoch"] for e in epochs]
    train_loss = [e["train_loss"] for e in epochs]
    val_loss = [e["val_loss"] for e in epochs]
    train_acc = [e["train_acc"] for e in epochs]
    val_acc = [e["val_acc"] for e in epochs]
    val_f1 = [e["val_f1"] for e in epochs]
    val_prec = [e["val_precision"] for e in epochs]
    val_rec = [e["val_recall"] for e in epochs]
    lr_vals = [e["lr"] for e in epochs]

    # ── Computational complexity ──────────────────────────────────
    layer_dims = [cfg["input_dim"]] + cfg["hidden_dims"] + [1]
    layer_macs = []
    for i in range(len(layer_dims) - 1):
        macs = layer_dims[i] * layer_dims[i + 1]
        layer_macs.append((layer_dims[i], layer_dims[i + 1], macs))
    total_macs = sum(m for _, _, m in layer_macs)
    total_mmacs = total_macs / 1e6
    fps = cfg.get("video_fps", 30)
    mmacs_per_sec = total_mmacs * fps
    # Add-ops ≈ MAC count (bias + BN), total FLOPs ≈ 2 × MACs
    total_flops = total_macs * 2
    total_mflops = total_flops / 1e6

    # Per-layer complexity rows for HTML table
    complexity_rows = ""
    prev = cfg["input_dim"]
    for i, h in enumerate(cfg["hidden_dims"]):
        macs = prev * h
        bn_ops = h * 2  # scale + shift
        complexity_rows += f"""<tr>
            <td>FC-{i+1} + BN + ReLU</td>
            <td>{prev:,} &rarr; {h:,}</td>
            <td>{macs:,}</td>
            <td>{macs/1e3:.1f} K</td>
        </tr>\n"""
        prev = h
    # Output layer
    macs_out = prev * 1
    complexity_rows += f"""<tr>
        <td>FC-out</td>
        <td>{prev:,} &rarr; 1</td>
        <td>{macs_out:,}</td>
        <td>{macs_out/1e3:.1f} K</td>
    </tr>\n"""
    complexity_rows += f"""<tr style="font-weight:700; border-top:2px solid var(--primary);">
        <td>Total</td>
        <td></td>
        <td>{total_macs:,}</td>
        <td>{total_mmacs:.4f} M</td>
    </tr>\n"""

    # Build hidden-layer diagram rows
    hidden_rows = ""
    prev = cfg["input_dim"]
    for i, h in enumerate(cfg["hidden_dims"]):
        hidden_rows += f"""
            <div class="layer">
                <div class="layer-box hidden">FC-{i+1}  ({prev} &rarr; {h})</div>
                <div class="layer-details">BatchNorm &rarr; ReLU &rarr; Dropout({cfg['dropout']})</div>
            </div>
            <div class="arrow">&darr;</div>"""
        prev = h

    # Per-recording table rows for DATA section
    rec_rows = ""
    for r in d["recordings"]:
        split = "Train"
        if not no_split:
            if r["rec_id"] in d["val_recordings"]:
                split = "Val"
            elif r["rec_id"] in d["test_recordings"]:
                split = "Test"
        badge = {"Train": "badge-train", "Val": "badge-val", "Test": "badge-test"}[split]
        split_col = "" if no_split else f'<td><span class="badge {badge}">{split}</span></td>'
        rec_rows += f"""<tr>
            <td>{r['rec_id']}</td>
            <td>{r['n_samples']:,}</td>
            <td>{r['n_active']:,}</td>
            <td>{r['active_pct']:.1f}%</td>
            {split_col}
        </tr>\n"""

    # Per-recording test results rows
    test_rec_rows = ""
    for r in test["per_recording"]:
        test_rec_rows += f"""<tr>
            <td>{r['rec_id']}</td>
            <td>{r['n_samples']:,}</td>
            <td>{r['n_active']:,}</td>
            <td>{r['accuracy']:.4f}</td>
            <td>{r['precision']:.4f}</td>
            <td>{r['recall']:.4f}</td>
            <td>{r['f1']:.4f}</td>
        </tr>\n"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V-VAD DNN &mdash; Training Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.1.0/dist/chartjs-plugin-annotation.min.js"></script>
<style>
:root {{
    --bg: #f5f7fa; --card: #fff; --border: #e1e5eb;
    --primary: #2563eb; --primary-light: #dbeafe;
    --accent: #f59e0b; --accent2: #10b981;
    --text: #1e293b; --text2: #64748b;
    --danger: #ef4444; --success: #22c55e;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.6; padding: 2rem 1rem;
}}
.container {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ font-size: 2rem; color: var(--primary); margin-bottom: .25rem; }}
h1 small {{ font-size: .9rem; color: var(--text2); font-weight: 400; }}
h2 {{
    font-size: 1.35rem; margin: 2.5rem 0 1rem;
    padding-bottom: .4rem; border-bottom: 2px solid var(--primary);
    color: var(--primary);
}}
h3 {{ font-size: 1.05rem; margin: 1.2rem 0 .5rem; color: var(--text); }}

.card {{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
.grid3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }}
@media (max-width: 768px) {{ .grid2, .grid3 {{ grid-template-columns: 1fr; }} }}

.kpi {{ text-align: center; padding: 1rem; }}
.kpi .value {{ font-size: 2rem; font-weight: 700; color: var(--primary); }}
.kpi .label {{ font-size: .8rem; color: var(--text2); text-transform: uppercase; letter-spacing: .05em; }}
.kpi.good .value {{ color: var(--accent2); }}
.kpi.warn .value {{ color: var(--accent); }}
.kpi.bad .value {{ color: var(--danger); }}

table {{
    width: 100%; border-collapse: collapse; font-size: .88rem;
}}
th, td {{
    padding: .55rem .75rem; text-align: left;
    border-bottom: 1px solid var(--border);
}}
th {{ background: var(--primary-light); color: var(--primary); font-weight: 600; }}
tr:hover {{ background: #f8fafc; }}

.badge {{
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: .75rem; font-weight: 600;
}}
.badge-train {{ background: #dbeafe; color: #1d4ed8; }}
.badge-val {{ background: #fef3c7; color: #b45309; }}
.badge-test {{ background: #d1fae5; color: #065f46; }}

/* Architecture diagram */
.arch-diagram {{ display: flex; flex-direction: column; align-items: center; gap: .15rem; padding: 1.5rem 0; }}
.layer {{ text-align: center; }}
.layer-box {{
    display: inline-block; padding: .6rem 1.6rem; border-radius: 8px;
    font-weight: 600; font-size: .9rem; min-width: 260px;
}}
.layer-box.input {{ background: #dbeafe; color: #1e40af; border: 2px solid #93c5fd; }}
.layer-box.hidden {{ background: #fef3c7; color: #92400e; border: 2px solid #fcd34d; }}
.layer-box.output {{ background: #d1fae5; color: #065f46; border: 2px solid #6ee7b7; }}
.layer-details {{ font-size: .78rem; color: var(--text2); margin-top: 2px; }}
.arrow {{ font-size: 1.2rem; color: var(--text2); line-height: 1; }}

/* Confusion matrix */
.cm-table {{ max-width: 360px; margin: 0 auto; }}
.cm-table td {{ text-align: center; font-weight: 600; font-size: 1.1rem; width: 120px; height: 60px; }}
.cm-table .cm-tn {{ background: #dcfce7; color: #166534; }}
.cm-table .cm-tp {{ background: #dbeafe; color: #1e40af; }}
.cm-table .cm-fp {{ background: #fee2e2; color: #991b1b; }}
.cm-table .cm-fn {{ background: #fef9c3; color: #854d0e; }}
.cm-table th {{ background: none; color: var(--text2); font-weight: 500; font-size: .8rem; text-align: center; }}

.chart-container {{ position: relative; height: 340px; }}

.pipeline-step {{
    display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1rem;
}}
.pipeline-num {{
    flex-shrink: 0; width: 32px; height: 32px; border-radius: 50%;
    background: var(--primary); color: #fff; display: flex;
    align-items: center; justify-content: center; font-weight: 700; font-size: .85rem;
}}
.pipeline-text {{ flex: 1; }}
.pipeline-text strong {{ color: var(--primary); }}

code {{ background: #f1f5f9; padding: 1px 6px; border-radius: 4px; font-size: .85em; }}
.mono {{ font-family: 'Cascadia Code', 'Fira Code', monospace; }}

footer {{ text-align: center; margin-top: 3rem; color: var(--text2); font-size: .8rem; }}
</style>
</head>
<body>
<div class="container">

<!-- ════════════════════ HEADER ════════════════════ -->
<h1>V-VAD DNN Training Report <br><small>Visual Voice Activity Detection using Facial Landmarks</small></h1>

<!-- ════════════════════ OVERVIEW ════════════════════ -->
<h2>1. Overview</h2>
<div class="card">
<p>
This report documents the training and evaluation of a <strong>feedforward Deep Neural Network (DNN)</strong>
for <strong>Visual Voice Activity Detection (V-VAD)</strong>. The model predicts whether a speaker is actively
speaking in a given video frame based solely on the movement of their <strong>68 facial landmarks</strong>.
</p>
<p style="margin-top:.7rem;">
The ground-truth labels are derived from audio-based Voice Activity Detection (VAD) decisions
aligned to video frames at <strong>{cfg['video_fps']:.0f} FPS</strong>.
A sliding window of <strong>{cfg['window_size']} consecutive frames</strong> provides temporal context,
and both absolute landmark positions and frame-to-frame deltas are used as features.
</p>
{"" if not no_split else '''<p style="margin-top:.7rem;"><strong>Note:</strong> In this training run, <em>all available data</em> was used for training (no train/val/test split). The best model was selected based on training-set metrics evaluated in inference mode (BatchNorm running statistics, Dropout disabled). All ''' + str(last_epoch) + ''' epochs were executed without early stopping.</p>'''}
</div>

<div class="grid3">
    <div class="card kpi"><div class="value">{d['total_recordings']}</div><div class="label">Recordings</div></div>
    <div class="card kpi"><div class="value">{d['total_samples']:,}</div><div class="label">Total Samples</div></div>
    <div class="card kpi"><div class="value">{d['total_active_pct']:.1f}%</div><div class="label">Active (Speaking)</div></div>
</div>

<!-- ════════════════════ PREPROCESSING ════════════════════ -->
<h2>2. Preprocessing Pipeline</h2>
<div class="card">
<p>Every frame undergoes the following four-stage preprocessing pipeline before features are extracted.
The goal is to remove variation caused by head pose and distance from the camera, isolating only
the information related to facial articulation (mouth movement).</p>

<div style="margin-top:1.2rem;">
    <div class="pipeline-step">
        <div class="pipeline-num">1</div>
        <div class="pipeline-text">
            <strong>Face Frontalization (Derotation)</strong><br>
            A face-aligned coordinate system is constructed from stable reference landmarks:
            eye corners (points 36, 39, 42, 45), nose bridge (27), and chin (8).
            The X-axis runs between the eye centers, the Y-axis from nose to chin, and
            the Z-axis is the face normal. The inverse rotation matrix is applied to all
            68 landmarks, effectively removing head rotation (yaw, pitch, roll).
        </div>
    </div>
    <div class="pipeline-step">
        <div class="pipeline-num">2</div>
        <div class="pipeline-text">
            <strong>Centering</strong><br>
            All landmarks are translated so that the midpoint between the two eye centers
            becomes the origin. This removes translational variation.
        </div>
    </div>
    <div class="pipeline-step">
        <div class="pipeline-num">3</div>
        <div class="pipeline-text">
            <strong>Scale Normalization</strong><br>
            All coordinates are divided by the inter-ocular distance (distance between
            the left and right eye centers). This makes the representation invariant to
            the speaker's distance from the camera.
        </div>
    </div>
    <div class="pipeline-step">
        <div class="pipeline-num">4</div>
        <div class="pipeline-text">
            <strong>Z-Score Standardization</strong><br>
            After all geometric normalization, the full feature vector is standardized
            (zero mean, unit variance) using statistics computed on the {"<em>full dataset</em> (no-split mode)" if no_split else "<em>training set only</em>. The same mean and standard deviation are applied to validation and test data"}.
        </div>
    </div>
</div>
</div>

<!-- ════════════════════ FEATURES ════════════════════ -->
<h2>3. Feature Engineering</h2>
<div class="card">
<p>For each sample, a sliding window of <strong>{cfg['window_size']} consecutive frames</strong>
is used. Two types of features are extracted:</p>

<h3>Absolute Landmark Positions</h3>
<p>The frontalized and normalized (x, y) coordinates of all 68 landmarks for each frame in the window.<br>
<code>{cfg['window_size']} frames &times; 68 landmarks &times; 2 coords = {cfg['n_position_features']} values</code></p>

<h3>Frame-to-Frame Deltas (Movement Features)</h3>
<p>The difference in coordinates between each pair of consecutive frames in the window.
These capture the <em>motion</em> of facial landmarks, which is the primary signal for speech detection.<br>
<code>({cfg['window_size']}&minus;1) frames &times; 68 landmarks &times; 2 coords = {cfg['n_delta_features']} values</code></p>

<h3 style="margin-top:1rem;">Total Input Dimension</h3>
<p><code>{cfg['n_position_features']} + {cfg['n_delta_features']} = <strong>{cfg['input_dim']}</strong> features per sample</code></p>
</div>

<!-- ════════════════════ ARCHITECTURE ════════════════════ -->
<h2>4. Model Architecture</h2>
<div class="card">
<p>The model is a fully-connected feedforward DNN with {len(cfg['hidden_dims'])} hidden layers.
Each hidden layer uses Batch Normalization, ReLU activation, and Dropout for regularization.
The output is a single logit, trained with <code>BCEWithLogitsLoss</code>.</p>

<div class="arch-diagram">
    <div class="layer">
        <div class="layer-box input">Input ({cfg['input_dim']} features)</div>
        <div class="layer-details">{cfg['window_size']}f &times; 68 landmarks &times; 2 coords &nbsp;+&nbsp; {cfg['window_size']-1}f &times; 68 &times; 2 deltas</div>
    </div>
    <div class="arrow">&darr;</div>
    {hidden_rows}
    <div class="layer">
        <div class="layer-box output">Output (1 logit &rarr; sigmoid)</div>
        <div class="layer-details">Binary: Speaking / Silent</div>
    </div>
</div>

<div class="grid3">
    <div class="card kpi"><div class="value">{cfg['n_params']:,}</div><div class="label">Trainable Parameters</div></div>
    <div class="card kpi"><div class="value">{cfg['dropout']}</div><div class="label">Dropout Rate</div></div>
    <div class="card kpi"><div class="value">{cfg['pos_weight']:.3f}</div><div class="label">Pos Weight (class balance)</div></div>
</div>

<h3>Computational Complexity</h3>
<p>Multiply-Accumulate (MAC) operations per single inference pass through the network.
One MAC equals one multiplication plus one addition (2 FLOPs).
The model processes one frame every {1000/fps:.1f}&thinsp;ms at {fps:.0f}&thinsp;FPS.</p>

<div class="grid3" style="margin-top:1rem;">
    <div class="card kpi"><div class="value">{total_mmacs:.4f}</div><div class="label">MMACs / inference</div></div>
    <div class="card kpi"><div class="value">{mmacs_per_sec:.2f}</div><div class="label">MMACs / second @{fps:.0f} FPS</div></div>
    <div class="card kpi"><div class="value">{total_mflops:.4f}</div><div class="label">MFLOPs / inference</div></div>
</div>

<table style="margin-top:1rem;">
<tr><th>Layer</th><th>Dimensions</th><th>MACs</th><th>MMACs</th></tr>
{complexity_rows}
</table>

<p style="margin-top:.8rem; font-size:.85rem; color: var(--text2);">
<strong>Note:</strong> The above counts cover only the linear (fully-connected) layers, which dominate
the computation. BatchNorm, ReLU, and Dropout add negligible overhead
(&lt;&thinsp;1% of total). At {mmacs_per_sec:.2f}&thinsp;MMACs/s the model is extremely lightweight
and suitable for real-time inference on embedded or mobile hardware.
</p>
</div>

<!-- ════════════════════ TRAINING CONFIG ════════════════════ -->
<h2>5. Training Configuration</h2>
<div class="card">
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Optimizer</td><td>Adam (weight_decay={cfg['weight_decay']})</td></tr>
<tr><td>Initial Learning Rate</td><td>{cfg['learning_rate']}</td></tr>
<tr><td>LR Scheduler</td><td>ReduceLROnPlateau (factor=0.5, patience=10)</td></tr>
<tr><td>Batch Size</td><td>{cfg['batch_size']}</td></tr>
<tr><td>Loss Function</td><td>BCEWithLogitsLoss (pos_weight={cfg['pos_weight']:.3f})</td></tr>
<tr><td>Epochs Trained</td><td>{last_epoch} (best&nbsp;at&nbsp;epoch&nbsp;{best_epoch})</td></tr>
<tr><td>Early Stopping</td><td>{"Disabled (all epochs executed)" if no_split else f"Patience = {cfg.get('early_stop_patience', 25)} epochs"}</td></tr>
<tr><td>Data Split</td><td>{"No split &mdash; all data used for training" if no_split else "Stratified by-recording (train/val/test)"}</td></tr>
<tr><td>Random Seed</td><td>{cfg['seed']}</td></tr>
<tr><td>Device</td><td>{cfg['device']}</td></tr>
<tr><td>Training Time</td><td>{t_time:.1f} seconds ({t_time/60:.1f} min)</td></tr>
</table>
</div>

<!-- ════════════════════ DATA SPLIT / USAGE ════════════════════ -->
<h2>6. {"Data Usage" if no_split else "Data Split"}</h2>
<div class="card">
{"<p><strong>All " + str(d['total_recordings']) + " recordings</strong> (" + f'{d["total_samples"]:,}' + " samples) are used for training. No separate validation or test set is held out. The best model checkpoint is selected by evaluating the training data in <em>inference mode</em> (model.eval() &mdash; BatchNorm uses running statistics, Dropout is disabled), so the selection metric reflects how the model would perform at deployment time.</p>" if no_split else "<p>Data is split <strong>by recording</strong> (not by frame) to prevent data leakage. Recordings are stratified by their activity ratio so that each split has a similar distribution of speaking vs.&nbsp;silent content.</p>"}

{"" if no_split else f"""<div class="grid3">
    <div class="card kpi"><div class="value">{d['train_samples']:,}</div><div class="label">Train Samples ({len(d['train_recordings'])} rec, {d['train_active_pct']:.1f}% active)</div></div>
    <div class="card kpi"><div class="value">{d['val_samples']:,}</div><div class="label">Val Samples ({len(d['val_recordings'])} rec, {d['val_active_pct']:.1f}% active)</div></div>
    <div class="card kpi"><div class="value">{d['test_samples']:,}</div><div class="label">Test Samples ({len(d['test_recordings'])} rec, {d['test_active_pct']:.1f}% active)</div></div>
</div>"""}

<h3>Per-Recording Breakdown</h3>
<table>
<tr><th>Recording</th><th>Samples</th><th>Active</th><th>Active %</th>{"" if no_split else "<th>Split</th>"}</tr>
{rec_rows}
</table>
</div>

<!-- ════════════════════ TRAINING CURVES ════════════════════ -->
<h2>7. Training Curves</h2>
<div class="grid2">
    <div class="card">
        <h3>Loss</h3>
        <div class="chart-container"><canvas id="chartLoss"></canvas></div>
    </div>
    <div class="card">
        <h3>Accuracy</h3>
        <div class="chart-container"><canvas id="chartAcc"></canvas></div>
    </div>
</div>
<div class="grid2">
    <div class="card">
        <h3>{eval_label} F1 / Precision / Recall</h3>
        <div class="chart-container"><canvas id="chartF1"></canvas></div>
    </div>
    <div class="card">
        <h3>Learning Rate Schedule</h3>
        <div class="chart-container"><canvas id="chartLR"></canvas></div>
    </div>
</div>

<!-- ════════════════════ RESULTS ════════════════════ -->
<h2>8. {"Evaluation Results (Training Set)" if no_split else "Test Set Results"}</h2>
<div class="card">
{"<p>The best model (epoch <strong>" + str(best_epoch) + "</strong> of " + str(last_epoch) + ", selected by training F1 in eval mode) was evaluated on the <strong>full training set</strong> (" + f'{d["train_samples"]:,}' + " samples from " + str(len(d['train_recordings'])) + " recordings). Both the best-epoch and last-epoch checkpoints are saved as <code>.pt</code> files.</p>" if no_split else "<p>The best model (epoch <strong>" + str(best_epoch) + "</strong> of " + str(last_epoch) + ", selected by validation F1) was evaluated on the held-out test set (" + f'{d["test_samples"]:,}' + " samples from " + str(len(d['test_recordings'])) + " recordings). Both the best-epoch and last-epoch checkpoints are saved as <code>.pt</code> files.</p>"}

<div class="grid3" style="margin-top:1rem;">
    <div class="card kpi"><div class="value">{test['accuracy']:.2%}</div><div class="label">Accuracy</div></div>
    <div class="card kpi"><div class="value">{test['f1']:.2%}</div><div class="label">F1-Score</div></div>
    <div class="card kpi"><div class="value">{test['precision']:.2%}</div><div class="label">Precision</div></div>
</div>
<div class="grid3">
    <div class="card kpi"><div class="value">{test['recall']:.2%}</div><div class="label">Recall</div></div>
    <div class="card kpi"><div class="value">{best_epoch}</div><div class="label">Best Epoch</div></div>
    <div class="card kpi"><div class="value">{last_epoch}</div><div class="label">Total Epochs</div></div>
</div>

<h3>Confusion Matrix</h3>
<table class="cm-table">
<tr><th></th><th></th><th colspan="2" style="text-align:center;">Predicted</th></tr>
<tr><th></th><th></th><th style="text-align:center;">Silent</th><th style="text-align:center;">Speaking</th></tr>
<tr><th rowspan="2" style="vertical-align:middle;">Actual</th><td><strong>Silent</strong></td>
    <td class="cm-tn">{cm['tn']:,}</td><td class="cm-fp">{cm['fp']:,}</td></tr>
<tr><td><strong>Speaking</strong></td>
    <td class="cm-fn">{cm['fn']:,}</td><td class="cm-tp">{cm['tp']:,}</td></tr>
</table>

<h3 style="margin-top:1.5rem;">Per-Recording {"Evaluation" if no_split else "Test"} Results</h3>
<table>
<tr><th>Recording</th><th>Samples</th><th>Active</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
{test_rec_rows}
</table>
</div>

<!-- ════════════════════ DISCUSSION ════════════════════ -->
<h2>9. Discussion &amp; Notes</h2>
<div class="card">
<ul style="margin-left:1.2rem;">
    <li>The model was trained on <strong>face_idx=0</strong> only (the primary / closest detected face per frame).</li>
    {"<li><strong>No data split (full-data training):</strong> All available recordings were used for training to maximise the amount of data the model sees. The best checkpoint was selected by evaluating the training data in <em>inference mode</em> (BatchNorm running stats, Dropout off). Early stopping was disabled &mdash; all " + str(last_epoch) + " epochs ran to completion.</li>" if no_split else "<li><strong>Data leakage prevention:</strong> The train/val/test split is performed at the recording level, ensuring that frames from the same video never appear in both training and evaluation sets.</li>"}
    <li><strong>Class imbalance:</strong> The <code>pos_weight</code> parameter in BCEWithLogitsLoss compensates
        for the imbalance between speaking and silent frames in the training set.</li>
    <li><strong>Standardization parameters</strong> (mean and std) are saved inside the model checkpoint
        (<code>.pt</code> file) to ensure consistent preprocessing at inference time.</li>
    <li><strong>Delta features</strong> (frame-to-frame landmark movement) are critical for performance &mdash;
        they capture the temporal dynamics of speech that static positions alone cannot convey.</li>
    <li>Recordings with <strong>0% activity</strong> (e.g., data_train5, data_train38&ndash;41) represent silence-only
        segments and are important for training the model to correctly classify non-speech.</li>
    {"<li><strong>Train vs. Eval metrics in charts:</strong> The <em>Train</em> line shows the loss/accuracy computed during training (with Dropout active and BatchNorm in training mode). The <em>Eval (train)</em> line shows the same data evaluated in inference mode. The gap between them reflects the effect of Dropout regularisation.</li>" if no_split else ""}
</ul>
</div>

<footer>
    Generated from <code>vvad_dnn_training_history.json</code> &mdash; V-VAD DNN Training Pipeline
</footer>

</div><!-- /container -->

<script>
// ── Chart Data ──────────────────────────────────────────────────────────────
const epochs   = {json.dumps(epoch_nums)};
const tLoss    = {json.dumps(train_loss)};
const vLoss    = {json.dumps(val_loss)};
const tAcc     = {json.dumps(train_acc)};
const vAcc     = {json.dumps(val_acc)};
const vF1      = {json.dumps(val_f1)};
const vPrec    = {json.dumps(val_prec)};
const vRec     = {json.dumps(val_rec)};
const lrVals   = {json.dumps(lr_vals)};
const bestEp   = {best_epoch};

// Best-epoch vertical annotation (red dashed line)
const bestAnnotation = {{
    annotations: {{
        bestLine: {{
            type: 'line',
            xMin: bestEp,
            xMax: bestEp,
            borderColor: 'rgba(239,68,68,0.55)',
            borderWidth: 2,
            borderDash: [6, 4],
            label: {{
                display: true,
                content: 'Best (ep ' + bestEp + ')',
                position: 'start',
                backgroundColor: 'rgba(239,68,68,0.8)',
                color: '#fff',
                font: {{ size: 10, weight: 'bold' }},
                padding: 3,
            }},
        }}
    }}
}};

// ── Loss Chart ──
new Chart(document.getElementById('chartLoss'), {{
    type: 'line',
    data: {{
        labels: epochs,
        datasets: [
            {{ label: 'Train Loss', data: tLoss, borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,.1)', tension: .3, pointRadius: 0 }},
            {{ label: '{eval_label} Loss',   data: vLoss, borderColor: '#f59e0b', backgroundColor: 'rgba(245,158,11,.1)', tension: .3, pointRadius: 0 }},
        ]
    }},
    options: {{ responsive: true, maintainAspectRatio: false,
        plugins: {{ annotation: bestAnnotation }},
        scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Loss' }} }} }},
    }}
}});

// ── Accuracy Chart ──
new Chart(document.getElementById('chartAcc'), {{
    type: 'line',
    data: {{
        labels: epochs,
        datasets: [
            {{ label: 'Train Acc', data: tAcc, borderColor: '#2563eb', tension: .3, pointRadius: 0 }},
            {{ label: '{eval_label} Acc',   data: vAcc, borderColor: '#f59e0b', tension: .3, pointRadius: 0 }},
        ]
    }},
    options: {{ responsive: true, maintainAspectRatio: false,
        plugins: {{ annotation: bestAnnotation }},
        scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Accuracy' }}, min: 0.4, max: 1.0 }} }},
    }}
}});

// ── F1 / Prec / Recall Chart ──
new Chart(document.getElementById('chartF1'), {{
    type: 'line',
    data: {{
        labels: epochs,
        datasets: [
            {{ label: 'F1',        data: vF1,   borderColor: '#2563eb', tension: .3, pointRadius: 0 }},
            {{ label: 'Precision',  data: vPrec, borderColor: '#10b981', tension: .3, pointRadius: 0, borderDash: [5,3] }},
            {{ label: 'Recall',     data: vRec,  borderColor: '#ef4444', tension: .3, pointRadius: 0, borderDash: [5,3] }},
        ]
    }},
    options: {{ responsive: true, maintainAspectRatio: false,
        plugins: {{ annotation: bestAnnotation }},
        scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'Score' }}, min: 0.0, max: 1.0 }} }},
    }}
}});

// ── Learning Rate Chart ──
new Chart(document.getElementById('chartLR'), {{
    type: 'line',
    data: {{
        labels: epochs,
        datasets: [
            {{ label: 'Learning Rate', data: lrVals, borderColor: '#8b5cf6', tension: .3, pointRadius: 0, fill: true, backgroundColor: 'rgba(139,92,246,.08)' }},
        ]
    }},
    options: {{ responsive: true, maintainAspectRatio: false,
        plugins: {{ annotation: bestAnnotation }},
        scales: {{ x: {{ title: {{ display: true, text: 'Epoch' }} }}, y: {{ title: {{ display: true, text: 'LR' }}, type: 'logarithmic' }} }},
    }}
}});
</script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate V-VAD DNN HTML report")
    parser.add_argument("--input", default="vvad_dnn_training_history.json",
                        help="Path to training history JSON")
    parser.add_argument("--output", default="vvad_dnn_report.html",
                        help="Output HTML file path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file '{args.input}' not found.")
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    html = generate_html(data)

    with open(args.output, "w") as f:
        f.write(html)

    print(f"Report generated: {os.path.abspath(args.output)}")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Size:   {len(html):,} bytes")


if __name__ == "__main__":
    main()
