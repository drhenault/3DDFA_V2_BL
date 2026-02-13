#!/usr/bin/env python3
"""
Advanced V-VAD analysis: test algorithmic improvements beyond simple parameter tuning.
Tests: MAR delta, zero-crossing rate, adaptive thresholds, combined scoring.
"""

import os, glob, numpy as np, pandas as pd
from collections import deque
from itertools import product

# ── Frontalization + MAR (same as before) ────────────────────────────

def frontalize_mouth_landmarks(points_3d):
    required_stable = [8, 27, 36, 39, 42, 45]
    mouth_indices = list(range(48, 68))
    for idx in required_stable + mouth_indices:
        if idx not in points_3d:
            return None
    p = {k: np.array(v, dtype=np.float64) for k, v in points_3d.items()}
    lec = (p[36] + p[39]) / 2.0
    rec = (p[42] + p[45]) / 2.0
    x_axis = rec - lec
    xn = np.linalg.norm(x_axis)
    if xn < 1e-8: return None
    x_axis /= xn
    y_approx = p[8] - p[27]
    z_axis = np.cross(x_axis, y_approx)
    zn = np.linalg.norm(z_axis)
    if zn < 1e-8: return None
    z_axis /= zn
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    R_inv = np.column_stack([x_axis, y_axis, z_axis]).T
    m3d = np.array([list(points_3d[i]) for i in mouth_indices], dtype=np.float64)
    c = m3d.mean(axis=0)
    mf = (R_inv @ (m3d - c).T).T + c
    return {idx: (mf[i, 0], mf[i, 1]) for i, idx in enumerate(mouth_indices)}

def compute_mar(mp):
    for idx in range(60, 68):
        if idx not in mp: return None
    p = {k: np.array(v, dtype=np.float64) for k, v in mp.items() if 60 <= k < 68}
    d1 = np.linalg.norm(p[61]-p[67])
    d2 = np.linalg.norm(p[62]-p[66])
    d3 = np.linalg.norm(p[63]-p[65])
    dh = np.linalg.norm(p[60]-p[64])
    return (d1+d2+d3)/(2.0*dh) if dh > 1e-8 else None

def load_recording(path):
    dfm = pd.read_csv(os.path.join(path, 'mouth_position.csv'))
    dfv = pd.read_csv(os.path.join(path, 'vad.csv'))
    ts = sorted(dfm['seconds'].unique())
    mars = []
    vad_times = dfv['seconds'].values
    vad_vals = dfv['vadDagcDecFinal'].values
    audio_vad = []
    for t in ts:
        fd = dfm[(dfm['seconds']==t) & (dfm['face_idx']==0)]
        p3 = {}
        for _, r in fd.iterrows():
            idx = int(r['point_type'])
            p3[idx] = (r['x'], r['y'], r['z'])
        f = frontalize_mouth_landmarks(p3)
        mars.append(compute_mar(f) if f else None)
        audio_vad.append(int(vad_vals[np.argmin(np.abs(vad_times-t))]))
    return ts, mars, audio_vad

def metrics(vvad, avad):
    n = len(vvad)
    tp = sum(1 for i in range(n) if vvad[i]==1 and avad[i]==1)
    tn = sum(1 for i in range(n) if vvad[i]==0 and avad[i]==0)
    fp = sum(1 for i in range(n) if vvad[i]==1 and avad[i]==0)
    fn = sum(1 for i in range(n) if vvad[i]==0 and avad[i]==1)
    agree = (tp+tn)/n*100 if n else 0
    prec = tp/(tp+fp)*100 if tp+fp else 100
    rec = tp/(tp+fn)*100 if tp+fn else 100
    f1 = 2*tp/(2*tp+fp+fn)*100 if 2*tp+fp+fn else 100
    fpr = fp/(fp+tn)*100 if fp+tn else 0
    fnr = fn/(fn+tp)*100 if fn+tp else 0
    return {'agree':agree,'prec':prec,'rec':rec,'f1':f1,'fpr':fpr,'fnr':fnr}

# ── Algorithm variants ──────────────────────────────────────────────

def algo_baseline(mars, fps, window_s, var_thr, mar_thr, hold_s):
    """Current algorithm: var + mean MAR, with hold."""
    ws = max(3, int(window_s * fps))
    hf = max(1, int(hold_s * fps))
    hist = deque(maxlen=ws)
    hc = 0
    out = []
    for m in mars:
        if m is not None: hist.append(m)
        active = False
        if len(hist) >= 3:
            v = np.array(hist)
            active = (np.var(v) > var_thr) and (np.mean(v) > mar_thr)
        if active: hc = hf
        elif hc > 0: hc -= 1
        out.append(1 if hc > 0 else 0)
    return out

def algo_delta_enhanced(mars, fps, window_s, var_thr, mar_thr, hold_s,
                         delta_thr=0.01, min_zcr=2):
    """
    Enhanced: variance + mean MAR + delta MAR zero-crossing rate.
    Speech produces oscillating mouth movements → high zero-crossing rate of MAR derivative.
    Idle mouth opening → low ZCR.
    """
    ws = max(3, int(window_s * fps))
    hf = max(1, int(hold_s * fps))
    hist = deque(maxlen=ws)
    hc = 0
    out = []
    for m in mars:
        if m is not None: hist.append(m)
        active = False
        if len(hist) >= 4:
            v = np.array(hist)
            mar_var = np.var(v)
            mar_mean = np.mean(v)
            # Compute delta (first derivative)
            delta = np.diff(v)
            # Zero-crossing rate of delta
            signs = np.sign(delta)
            zcr = np.sum(np.abs(np.diff(signs)) > 0)
            active = (mar_var > var_thr) and (mar_mean > mar_thr) and (zcr >= min_zcr)
        if active: hc = hf
        elif hc > 0: hc -= 1
        out.append(1 if hc > 0 else 0)
    return out

def algo_adaptive(mars, fps, window_s, var_thr_mult, mar_thr_offset, hold_s,
                   calibration_s=2.0):
    """
    Adaptive: calibrate baseline MAR from initial silent period.
    var_threshold = var_thr_mult × baseline_variance
    mar_threshold = baseline_mean + mar_thr_offset
    """
    ws = max(3, int(window_s * fps))
    hf = max(1, int(hold_s * fps))
    cal_frames = max(5, int(calibration_s * fps))
    
    # Calibration: compute baseline from first N frames
    cal_mars = [m for m in mars[:cal_frames] if m is not None]
    if len(cal_mars) < 3:
        baseline_var = 0.001
        baseline_mean = 0.15
    else:
        baseline_var = max(np.var(cal_mars), 1e-6)
        baseline_mean = np.mean(cal_mars)
    
    var_thr = baseline_var * var_thr_mult
    mar_thr = baseline_mean + mar_thr_offset
    
    hist = deque(maxlen=ws)
    hc = 0
    out = []
    for m in mars:
        if m is not None: hist.append(m)
        active = False
        if len(hist) >= 3:
            v = np.array(hist)
            active = (np.var(v) > var_thr) and (np.mean(v) > mar_thr)
        if active: hc = hf
        elif hc > 0: hc -= 1
        out.append(1 if hc > 0 else 0)
    return out

def algo_energy_based(mars, fps, window_s, energy_thr, hold_s):
    """
    Energy-based: compute 'speech energy' as sum of |delta_MAR| in window.
    Speech produces higher total movement energy than idle.
    """
    ws = max(3, int(window_s * fps))
    hf = max(1, int(hold_s * fps))
    hist = deque(maxlen=ws)
    hc = 0
    out = []
    for m in mars:
        if m is not None: hist.append(m)
        active = False
        if len(hist) >= 4:
            v = np.array(hist)
            delta = np.abs(np.diff(v))
            energy = np.sum(delta) / len(delta)  # avg absolute delta
            active = energy > energy_thr
        if active: hc = hf
        elif hc > 0: hc -= 1
        out.append(1 if hc > 0 else 0)
    return out

def algo_combined(mars, fps, window_s, var_thr, mar_thr, energy_thr,
                   min_zcr, hold_s):
    """
    Combined: score-based approach.
    Score = weighted sum of: var_exceeded + mar_exceeded + energy_exceeded + zcr_exceeded
    Active if score >= 3 out of 4.
    """
    ws = max(3, int(window_s * fps))
    hf = max(1, int(hold_s * fps))
    hist = deque(maxlen=ws)
    hc = 0
    out = []
    for m in mars:
        if m is not None: hist.append(m)
        score = 0
        if len(hist) >= 4:
            v = np.array(hist)
            if np.var(v) > var_thr: score += 1
            if np.mean(v) > mar_thr: score += 1
            delta = np.diff(v)
            if np.mean(np.abs(delta)) > energy_thr: score += 1
            signs = np.sign(delta)
            zcr = np.sum(np.abs(np.diff(signs)) > 0)
            if zcr >= min_zcr: score += 1
        active = score >= 3
        if active: hc = hf
        elif hc > 0: hc -= 1
        out.append(1 if hc > 0 else 0)
    return out

# ── Main ────────────────────────────────────────────────────────────

def run_algo_on_all(recs, algo_fn, **kwargs):
    fps = 30.0
    all_m = []
    per_rec = {}
    for name, rec in recs.items():
        vvad = algo_fn(rec['frontal_mars'], fps, **kwargs)
        m = metrics(vvad, rec['audio_vad'])
        all_m.append(m)
        per_rec[name] = m
    avg = {k: np.mean([m[k] for m in all_m]) for k in all_m[0]}
    return per_rec, avg

def print_results(per_rec, avg, label=""):
    print(f"\n  {label}")
    print(f"  {'Recording':30s}  {'Agree%':>7s}  {'Prec%':>7s}  {'Rec%':>7s}  "
          f"{'F1%':>7s}  {'FP%':>7s}  {'FN%':>7s}")
    print("  " + "-" * 95)
    for name, m in per_rec.items():
        print(f"  {name:30s}  {m['agree']:6.1f}%  {m['prec']:6.1f}%  "
              f"{m['rec']:6.1f}%  {m['f1']:6.1f}%  {m['fpr']:6.1f}%  {m['fnr']:6.1f}%")
    print(f"  {'AVERAGE':30s}  {avg['agree']:6.1f}%  {avg['prec']:6.1f}%  "
          f"{avg['rec']:6.1f}%  {avg['f1']:6.1f}%  {avg['fpr']:6.1f}%  {avg['fnr']:6.1f}%")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dump_dirs = sorted(glob.glob(os.path.join(base_dir, 'dumps_*')))
    dump_dirs = [d for d in dump_dirs if os.path.isdir(d)
                 and os.path.exists(os.path.join(d, 'mouth_position.csv'))
                 and os.path.exists(os.path.join(d, 'vad.csv'))]

    recs = {}
    for dp in dump_dirs:
        name = os.path.basename(dp).replace('dumps_', '')
        recs[name] = dict(zip(
            ('timestamps','frontal_mars','audio_vad'),
            load_recording(dp)
        ))

    print("=" * 100)
    print("ALGORITHM COMPARISON")
    print("=" * 100)

    # 1) Baseline current
    pr, avg = run_algo_on_all(recs, algo_baseline,
        window_s=0.5, var_thr=0.0005, mar_thr=0.12, hold_s=1.0)
    print_results(pr, avg, "A) BASELINE (current): w=0.5, var=0.0005, mar=0.12, hold=1.0")

    # 2) Baseline with optimized params (best agreement from grid)
    pr, avg = run_algo_on_all(recs, algo_baseline,
        window_s=0.3, var_thr=0.005, mar_thr=0.30, hold_s=0.3)
    print_results(pr, avg, "B) TUNED BASELINE: w=0.3, var=0.005, mar=0.30, hold=0.3")

    # 3) Delta-enhanced: various params
    print("\n" + "=" * 100)
    print("DELTA-ENHANCED (ZCR) — Grid search")
    print("=" * 100)

    best_agree = 0; best_params_de = None
    for ws in [0.3, 0.5, 0.7]:
        for vt in [0.001, 0.003, 0.005, 0.01]:
            for mt in [0.15, 0.20, 0.25, 0.30]:
                for hs in [0.3, 0.5]:
                    for mzcr in [2, 3, 4]:
                        pr, avg = run_algo_on_all(recs, algo_delta_enhanced,
                            window_s=ws, var_thr=vt, mar_thr=mt,
                            hold_s=hs, min_zcr=mzcr)
                        if avg['agree'] > best_agree:
                            best_agree = avg['agree']
                            best_params_de = (ws, vt, mt, hs, mzcr)
                            best_pr_de, best_avg_de = pr, avg

    ws, vt, mt, hs, mzcr = best_params_de
    print_results(best_pr_de, best_avg_de,
        f"C) BEST DELTA-ENHANCED: w={ws}, var={vt}, mar={mt}, hold={hs}, min_zcr={mzcr}")

    # 4) Energy-based
    print("\n" + "=" * 100)
    print("ENERGY-BASED — Grid search")
    print("=" * 100)

    best_agree = 0; best_params_en = None
    for ws in [0.3, 0.5, 0.7, 1.0]:
        for et in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]:
            for hs in [0.3, 0.5, 0.7]:
                pr, avg = run_algo_on_all(recs, algo_energy_based,
                    window_s=ws, energy_thr=et, hold_s=hs)
                if avg['agree'] > best_agree:
                    best_agree = avg['agree']
                    best_params_en = (ws, et, hs)
                    best_pr_en, best_avg_en = pr, avg

    ws, et, hs = best_params_en
    print_results(best_pr_en, best_avg_en,
        f"D) BEST ENERGY-BASED: w={ws}, energy_thr={et}, hold={hs}")

    # 5) Combined
    print("\n" + "=" * 100)
    print("COMBINED (score-based) — Grid search")
    print("=" * 100)

    best_agree = 0; best_params_co = None
    for ws in [0.3, 0.5, 0.7]:
        for vt in [0.001, 0.003, 0.005]:
            for mt in [0.15, 0.20, 0.25]:
                for et in [0.01, 0.02, 0.03]:
                    for mzcr in [2, 3]:
                        for hs in [0.3, 0.5]:
                            pr, avg = run_algo_on_all(recs, algo_combined,
                                window_s=ws, var_thr=vt, mar_thr=mt,
                                energy_thr=et, min_zcr=mzcr, hold_s=hs)
                            if avg['agree'] > best_agree:
                                best_agree = avg['agree']
                                best_params_co = (ws, vt, mt, et, mzcr, hs)
                                best_pr_co, best_avg_co = pr, avg

    ws, vt, mt, et, mzcr, hs = best_params_co
    print_results(best_pr_co, best_avg_co,
        f"E) BEST COMBINED: w={ws}, var={vt}, mar={mt}, energy={et}, zcr={mzcr}, hold={hs}")

    # 6) Adaptive
    print("\n" + "=" * 100)
    print("ADAPTIVE (auto-calibrating) — Grid search")
    print("=" * 100)

    best_agree = 0; best_params_ad = None
    for ws in [0.3, 0.5, 0.7]:
        for vtm in [3, 5, 10, 20, 50]:
            for mto in [0.02, 0.05, 0.08, 0.10, 0.15]:
                for hs in [0.3, 0.5]:
                    for cal in [1.0, 2.0, 3.0]:
                        pr, avg = run_algo_on_all(recs, algo_adaptive,
                            window_s=ws, var_thr_mult=vtm, mar_thr_offset=mto,
                            hold_s=hs, calibration_s=cal)
                        if avg['agree'] > best_agree:
                            best_agree = avg['agree']
                            best_params_ad = (ws, vtm, mto, hs, cal)
                            best_pr_ad, best_avg_ad = pr, avg

    ws, vtm, mto, hs, cal = best_params_ad
    print_results(best_pr_ad, best_avg_ad,
        f"F) BEST ADAPTIVE: w={ws}, var_mult={vtm}, mar_offset={mto}, hold={hs}, cal={cal}s")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON")
    print("=" * 100)
    print(f"  {'Algorithm':50s}  {'Agree%':>8s}  {'F1%':>7s}  {'FP%':>7s}  {'FN%':>7s}")
    print("  " + "-" * 85)

    results_summary = [
        ("A) Baseline (current)", 0.5, 0.0005, 0.12, 1.0),
    ]
    # Re-compute for summary
    for label, ws, vt, mt, hs in results_summary:
        _, avg = run_algo_on_all(recs, algo_baseline,
            window_s=ws, var_thr=vt, mar_thr=mt, hold_s=hs)
        print(f"  {label:50s}  {avg['agree']:7.1f}%  {avg['f1']:6.1f}%  {avg['fpr']:6.1f}%  {avg['fnr']:6.1f}%")

    _, avg = run_algo_on_all(recs, algo_baseline,
        window_s=0.3, var_thr=0.005, mar_thr=0.30, hold_s=0.3)
    print(f"  {'B) Tuned baseline':50s}  {avg['agree']:7.1f}%  {avg['f1']:6.1f}%  {avg['fpr']:6.1f}%  {avg['fnr']:6.1f}%")

    print(f"  {'C) Delta-enhanced (ZCR)':50s}  {best_avg_de['agree']:7.1f}%  {best_avg_de['f1']:6.1f}%  {best_avg_de['fpr']:6.1f}%  {best_avg_de['fnr']:6.1f}%")
    print(f"  {'D) Energy-based':50s}  {best_avg_en['agree']:7.1f}%  {best_avg_en['f1']:6.1f}%  {best_avg_en['fpr']:6.1f}%  {best_avg_en['fnr']:6.1f}%")
    print(f"  {'E) Combined (score-based)':50s}  {best_avg_co['agree']:7.1f}%  {best_avg_co['f1']:6.1f}%  {best_avg_co['fpr']:6.1f}%  {best_avg_co['fnr']:6.1f}%")
    print(f"  {'F) Adaptive (auto-calibrating)':50s}  {best_avg_ad['agree']:7.1f}%  {best_avg_ad['f1']:6.1f}%  {best_avg_ad['fpr']:6.1f}%  {best_avg_ad['fnr']:6.1f}%")


if __name__ == '__main__':
    main()
