#!/usr/bin/env python3
"""
Analyze V-VAD vs Audio VAD agreement across all recordings.
Grid-search optimal parameters to maximize agreement and minimize false positives.
"""

import os
import glob
import numpy as np
import pandas as pd
from collections import deque
from itertools import product

# ── Same frontalization / MAR functions ──────────────────────────────

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

# ── Precompute MAR time series for each recording ───────────────────

def load_recording(dumps_path):
    df_mouth = pd.read_csv(os.path.join(dumps_path, 'mouth_position.csv'))
    df_vad = pd.read_csv(os.path.join(dumps_path, 'vad.csv'))
    
    timestamps = sorted(df_mouth['seconds'].unique())
    
    frontal_mars = []
    audio_vad = []
    
    vad_times = df_vad['seconds'].values
    vad_vals = df_vad['vadDagcDecFinal'].values
    
    for t in timestamps:
        frame_data = df_mouth[df_mouth['seconds'] == t]
        face_data = frame_data[frame_data['face_idx'] == 0]
        
        pts_3d = {}
        pts_2d = {}
        for _, row in face_data.iterrows():
            idx = int(row['point_type'])
            pts_2d[idx] = (row['x'], row['y'])
            pts_3d[idx] = (row['x'], row['y'], row['z'])
        
        frontalized = frontalize_mouth_landmarks(pts_3d)
        frontal_mar = compute_mar(frontalized) if frontalized else None
        frontal_mars.append(frontal_mar)
        
        vidx = np.argmin(np.abs(vad_times - t))
        audio_vad.append(int(vad_vals[vidx]))
    
    return timestamps, frontal_mars, audio_vad

def run_vvad(frontal_mars, window_size, var_threshold, mar_threshold, hold_frames,
             use_delta=False, delta_threshold=0.0):
    """Run V-VAD with given parameters. Returns list of 0/1."""
    mar_history = deque(maxlen=window_size)
    hold_counter = 0
    results = []
    
    prev_mar = None
    
    for mar in frontal_mars:
        if mar is not None:
            mar_history.append(mar)
        
        if len(mar_history) >= 3:
            values = np.array(mar_history)
            mar_var = float(np.var(values))
            mar_mean = float(np.mean(values))
            
            is_active = (mar_var > var_threshold) and (mar_mean > mar_threshold)
            
            # Optional: delta-based check
            if use_delta and mar is not None and prev_mar is not None:
                delta = abs(mar - prev_mar)
                if delta < delta_threshold:
                    is_active = False
        else:
            is_active = False
        
        if is_active:
            hold_counter = hold_frames
        else:
            if hold_counter > 0:
                hold_counter -= 1
        
        results.append(1 if hold_counter > 0 else 0)
        if mar is not None:
            prev_mar = mar
    
    return results


def compute_metrics(vvad, audio_vad):
    """Compute agreement, precision, recall, F1, FP rate, FN rate."""
    n = len(vvad)
    tp = sum(1 for i in range(n) if vvad[i] == 1 and audio_vad[i] == 1)
    tn = sum(1 for i in range(n) if vvad[i] == 0 and audio_vad[i] == 0)
    fp = sum(1 for i in range(n) if vvad[i] == 1 and audio_vad[i] == 0)
    fn = sum(1 for i in range(n) if vvad[i] == 0 and audio_vad[i] == 1)
    
    agreement = (tp + tn) / n * 100 if n > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 100
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 100
    f1 = 2 * tp / (2*tp + fp + fn) * 100 if (2*tp + fp + fn) > 0 else 100
    fp_rate = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    
    return {
        'agreement': agreement,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dump_dirs = sorted(glob.glob(os.path.join(base_dir, 'dumps_*')))
    dump_dirs = [d for d in dump_dirs if os.path.isdir(d)
                 and os.path.exists(os.path.join(d, 'mouth_position.csv'))
                 and os.path.exists(os.path.join(d, 'vad.csv'))]
    
    FPS = 30.0
    
    # ── Step 1: Load all recordings ─────────────────────────────────
    print("=" * 90)
    print("STEP 1: Loading recordings...")
    print("=" * 90)
    
    recordings = {}
    for dp in dump_dirs:
        name = os.path.basename(dp).replace('dumps_', '')
        timestamps, frontal_mars, audio_vad = load_recording(dp)
        recordings[name] = {
            'timestamps': timestamps,
            'frontal_mars': frontal_mars,
            'audio_vad': audio_vad,
        }
        audio_active_pct = sum(audio_vad) / len(audio_vad) * 100
        print(f"  {name:30s}  {len(timestamps):5d} frames  "
              f"audio_active={audio_active_pct:.1f}%")
    
    # ── Step 2: Baseline analysis (current params) ──────────────────
    print("\n" + "=" * 90)
    print("STEP 2: Baseline analysis (current parameters)")
    print("  window=0.5s, var_thr=0.0005, mar_thr=0.12, hold=1.0s")
    print("=" * 90)
    
    ws = max(3, int(0.5 * FPS))
    hf = max(1, int(1.0 * FPS))
    
    print(f"  {'Recording':30s}  {'Agree%':>7s}  {'Prec%':>7s}  {'Rec%':>7s}  "
          f"{'F1%':>7s}  {'FP%':>7s}  {'FN%':>7s}  {'FP':>5s}  {'FN':>5s}")
    print("-" * 110)
    
    total_agree = []
    for name, rec in recordings.items():
        vvad = run_vvad(rec['frontal_mars'], ws, 0.0005, 0.12, hf)
        m = compute_metrics(vvad, rec['audio_vad'])
        total_agree.append(m['agreement'])
        print(f"  {name:30s}  {m['agreement']:6.1f}%  {m['precision']:6.1f}%  "
              f"{m['recall']:6.1f}%  {m['f1']:6.1f}%  {m['fp_rate']:6.1f}%  "
              f"{m['fn_rate']:6.1f}%  {m['fp']:5d}  {m['fn']:5d}")
    print(f"  {'AVERAGE':30s}  {np.mean(total_agree):6.1f}%")
    
    # ── Step 3: Detailed false positive analysis ────────────────────
    print("\n" + "=" * 90)
    print("STEP 3: False positive analysis — when does V-VAD fire but audio is silent?")
    print("=" * 90)
    
    for name, rec in recordings.items():
        vvad = run_vvad(rec['frontal_mars'], ws, 0.0005, 0.12, hf)
        mars = rec['frontal_mars']
        audio = rec['audio_vad']
        
        # Collect MAR values during FP frames
        fp_mars = [mars[i] for i in range(len(vvad)) if vvad[i] == 1 and audio[i] == 0 and mars[i] is not None]
        tp_mars = [mars[i] for i in range(len(vvad)) if vvad[i] == 1 and audio[i] == 1 and mars[i] is not None]
        tn_mars = [mars[i] for i in range(len(vvad)) if vvad[i] == 0 and audio[i] == 0 and mars[i] is not None]
        
        if len(fp_mars) > 0:
            print(f"\n  {name}:")
            print(f"    FP frames MAR:  mean={np.mean(fp_mars):.4f}  std={np.std(fp_mars):.4f}  "
                  f"min={np.min(fp_mars):.4f}  max={np.max(fp_mars):.4f}  (n={len(fp_mars)})")
            if len(tp_mars) > 0:
                print(f"    TP frames MAR:  mean={np.mean(tp_mars):.4f}  std={np.std(tp_mars):.4f}  "
                      f"min={np.min(tp_mars):.4f}  max={np.max(tp_mars):.4f}  (n={len(tp_mars)})")
            if len(tn_mars) > 0:
                print(f"    TN frames MAR:  mean={np.mean(tn_mars):.4f}  std={np.std(tn_mars):.4f}  "
                      f"min={np.min(tn_mars):.4f}  max={np.max(tn_mars):.4f}  (n={len(tn_mars)})")
    
    # ── Step 4: Grid search over parameters ─────────────────────────
    print("\n" + "=" * 90)
    print("STEP 4: Grid search — finding optimal parameters")
    print("=" * 90)
    
    window_options = [0.3, 0.5, 0.7, 1.0]
    var_thr_options = [0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01]
    mar_thr_options = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
    hold_options = [0.3, 0.5, 0.7, 1.0]
    
    best_f1_avg = 0
    best_agree_avg = 0
    best_params_f1 = None
    best_params_agree = None
    
    results_list = []
    
    total_combos = len(window_options) * len(var_thr_options) * len(mar_thr_options) * len(hold_options)
    print(f"  Testing {total_combos} parameter combinations...")
    
    for win_s, var_t, mar_t, hold_s in product(window_options, var_thr_options, mar_thr_options, hold_options):
        ws_i = max(3, int(win_s * FPS))
        hf_i = max(1, int(hold_s * FPS))
        
        f1_scores = []
        agree_scores = []
        fp_rates = []
        fn_rates = []
        
        for name, rec in recordings.items():
            vvad = run_vvad(rec['frontal_mars'], ws_i, var_t, mar_t, hf_i)
            m = compute_metrics(vvad, rec['audio_vad'])
            f1_scores.append(m['f1'])
            agree_scores.append(m['agreement'])
            fp_rates.append(m['fp_rate'])
            fn_rates.append(m['fn_rate'])
        
        avg_f1 = np.mean(f1_scores)
        avg_agree = np.mean(agree_scores)
        avg_fp = np.mean(fp_rates)
        avg_fn = np.mean(fn_rates)
        
        results_list.append({
            'window': win_s, 'var_thr': var_t, 'mar_thr': mar_t, 'hold': hold_s,
            'avg_f1': avg_f1, 'avg_agree': avg_agree, 'avg_fp': avg_fp, 'avg_fn': avg_fn,
        })
        
        if avg_f1 > best_f1_avg:
            best_f1_avg = avg_f1
            best_params_f1 = (win_s, var_t, mar_t, hold_s)
        
        if avg_agree > best_agree_avg:
            best_agree_avg = avg_agree
            best_params_agree = (win_s, var_t, mar_t, hold_s)
    
    # Sort by avg agreement DESC
    results_list.sort(key=lambda x: x['avg_agree'], reverse=True)
    
    print(f"\n  TOP 20 parameter sets by Agreement%:")
    print(f"  {'Window':>7s}  {'VarThr':>8s}  {'MARThr':>8s}  {'Hold':>6s}  "
          f"{'Agree%':>8s}  {'F1%':>7s}  {'FP%':>7s}  {'FN%':>7s}")
    print("  " + "-" * 80)
    for r in results_list[:20]:
        print(f"  {r['window']:7.1f}  {r['var_thr']:8.4f}  {r['mar_thr']:8.2f}  "
              f"{r['hold']:6.1f}  {r['avg_agree']:7.1f}%  {r['avg_f1']:6.1f}%  "
              f"{r['avg_fp']:6.1f}%  {r['avg_fn']:6.1f}%")
    
    # Sort by F1 DESC
    results_list.sort(key=lambda x: x['avg_f1'], reverse=True)
    
    print(f"\n  TOP 20 parameter sets by F1%:")
    print(f"  {'Window':>7s}  {'VarThr':>8s}  {'MARThr':>8s}  {'Hold':>6s}  "
          f"{'Agree%':>8s}  {'F1%':>7s}  {'FP%':>7s}  {'FN%':>7s}")
    print("  " + "-" * 80)
    for r in results_list[:20]:
        print(f"  {r['window']:7.1f}  {r['var_thr']:8.4f}  {r['mar_thr']:8.2f}  "
              f"{r['hold']:6.1f}  {r['avg_agree']:7.1f}%  {r['avg_f1']:6.1f}%  "
              f"{r['avg_fp']:6.1f}%  {r['avg_fn']:6.1f}%")
    
    # Sort by lowest FP rate (with minimum F1 > 30%)
    filtered = [r for r in results_list if r['avg_f1'] > 30]
    filtered.sort(key=lambda x: x['avg_fp'])
    
    print(f"\n  TOP 20 parameter sets by lowest FP% (with F1 > 30%):")
    print(f"  {'Window':>7s}  {'VarThr':>8s}  {'MARThr':>8s}  {'Hold':>6s}  "
          f"{'Agree%':>8s}  {'F1%':>7s}  {'FP%':>7s}  {'FN%':>7s}")
    print("  " + "-" * 80)
    for r in filtered[:20]:
        print(f"  {r['window']:7.1f}  {r['var_thr']:8.4f}  {r['mar_thr']:8.2f}  "
              f"{r['hold']:6.1f}  {r['avg_agree']:7.1f}%  {r['avg_f1']:6.1f}%  "
              f"{r['avg_fp']:6.1f}%  {r['avg_fn']:6.1f}%")
    
    # ── Step 5: Detailed results for best params ────────────────────
    print("\n" + "=" * 90)
    print("STEP 5: Detailed results with BEST parameters")
    print("=" * 90)
    
    for label, params in [("Best by Agreement%", best_params_agree),
                           ("Best by F1%", best_params_f1)]:
        win_s, var_t, mar_t, hold_s = params
        ws_i = max(3, int(win_s * FPS))
        hf_i = max(1, int(hold_s * FPS))
        
        print(f"\n  {label}: window={win_s}s, var_thr={var_t}, mar_thr={mar_t}, hold={hold_s}s")
        print(f"  {'Recording':30s}  {'Agree%':>7s}  {'Prec%':>7s}  {'Rec%':>7s}  "
              f"{'F1%':>7s}  {'FP%':>7s}  {'FN%':>7s}")
        print("  " + "-" * 95)
        
        all_a = []
        for name, rec in recordings.items():
            vvad = run_vvad(rec['frontal_mars'], ws_i, var_t, mar_t, hf_i)
            m = compute_metrics(vvad, rec['audio_vad'])
            all_a.append(m['agreement'])
            print(f"  {name:30s}  {m['agreement']:6.1f}%  {m['precision']:6.1f}%  "
                  f"{m['recall']:6.1f}%  {m['f1']:6.1f}%  {m['fp_rate']:6.1f}%  "
                  f"{m['fn_rate']:6.1f}%")
        print(f"  {'AVERAGE':30s}  {np.mean(all_a):6.1f}%")


if __name__ == '__main__':
    main()
