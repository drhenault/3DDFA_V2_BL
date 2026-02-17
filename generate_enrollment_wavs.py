#!/usr/bin/env python3
"""
Generate enrollment WAV files by extracting and concatenating active speech segments
from each recording based on three VAD criteria:

  1. Audio VAD only          → enrollment/{name}_audio_vad.wav
  2. Visual VAD (V-VAD) only → enrollment/{name}_vvad.wav
  3. Audio VAD + V-VAD (OR)  → enrollment/{name}_audio_vvad.wav

For each criterion, consecutive "active" frames are merged into time segments,
the corresponding audio is extracted from the source MP4, and all segments are
concatenated into a single WAV file.

Usage:  python generate_enrollment_wavs.py
Output: enrollment/ directory with 3 × N_recordings WAV files
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import soundfile as sf
from collections import deque

# ── V-VAD algorithm (must match generate_vvad_dashboard.py / multiface_distance_render.py) ──

VVAD_PARAMS = {
    'window_seconds': 0.3,
    'var_threshold': 0.002,
    'mar_activity_threshold': 0.15,
    'hold_seconds': 1.0,
    'min_zcr': 3,
    'fps': 30.0,
}


def frontalize_mouth_landmarks(points_3d):
    """Direct 3D geometry frontalization of mouth landmarks."""
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
    p = {k: np.array(v, dtype=np.float64) for k, v in mouth_points.items()
         if k in inner_indices}
    d1 = np.linalg.norm(p[61] - p[67])
    d2 = np.linalg.norm(p[62] - p[66])
    d3 = np.linalg.norm(p[63] - p[65])
    d_horiz = np.linalg.norm(p[60] - p[64])
    if d_horiz < 1e-8:
        return None
    return (d1 + d2 + d3) / (2.0 * d_horiz)


def run_vvad(frontal_mars, params):
    """Run ZCR-enhanced V-VAD and return per-frame binary decisions (with hold)."""
    fps = params['fps']
    ws = max(4, int(params['window_seconds'] * fps))
    hf = max(1, int(params['hold_seconds'] * fps))
    vt = params['var_threshold']
    mt = params['mar_activity_threshold']
    min_zcr = params['min_zcr']

    hist = deque(maxlen=ws)
    hc = 0
    vvad_out = []

    for m in frontal_mars:
        if m is not None:
            hist.append(m)
        active = False
        if len(hist) >= 4:
            v = np.array(hist)
            mv = float(np.var(v))
            mm = float(np.mean(v))
            delta = np.diff(v)
            signs = np.sign(delta)
            zcr_val = int(np.sum(np.abs(np.diff(signs)) > 0))
            active = (mv > vt) and (mm > mt) and (zcr_val >= min_zcr)
        if active:
            hc = hf
        elif hc > 0:
            hc -= 1
        vvad_out.append(1 if hc > 0 else 0)

    return vvad_out


# ── Audio utilities ──────────────────────────────────────────────────

SAMPLE_RATE = 48000  # 48 kHz output


def read_wav_samples(wav_path):
    """Read a WAV file and return (samples_array_float32, sample_rate)."""
    samples, sr = sf.read(wav_path, dtype='float32')
    # If stereo, take first channel
    if samples.ndim > 1:
        samples = samples[:, 0]
    return samples, sr


def write_wav_from_samples(wav_path, samples, sample_rate):
    """Write a mono 32-bit float WAV."""
    sf.write(wav_path, samples.astype(np.float32), sample_rate, subtype='FLOAT')


def decisions_to_segments(times, decisions):
    """
    Convert per-frame binary decisions (0/1) to a list of time segments.

    Returns list of (start_time, end_time) tuples.
    Adjacent active frames are merged. A small padding is added to avoid
    cutting speech at frame boundaries.
    """
    segments = []
    in_segment = False
    start_t = 0.0

    for i, dec in enumerate(decisions):
        t = times[i]
        if dec == 1 and not in_segment:
            start_t = t
            in_segment = True
        elif dec == 0 and in_segment:
            end_t = times[i - 1] if i > 0 else t
            # Add half-frame padding on each side
            dt = times[1] - times[0] if len(times) > 1 else 0.033
            segments.append((max(0.0, start_t - dt / 2), end_t + dt / 2))
            in_segment = False

    # Close last segment if still active
    if in_segment:
        dt = times[1] - times[0] if len(times) > 1 else 0.033
        segments.append((max(0.0, start_t - dt / 2), times[-1] + dt / 2))

    return segments


def extract_segments_from_audio(samples, sample_rate, segments):
    """Extract and concatenate audio samples for given time segments."""
    total_samples = len(samples)
    max_time = total_samples / sample_rate
    parts = []

    for start_t, end_t in segments:
        s_idx = max(0, int(start_t * sample_rate))
        e_idx = min(total_samples, int(end_t * sample_rate))
        if e_idx > s_idx:
            parts.append(samples[s_idx:e_idx])

    if parts:
        return np.concatenate(parts)
    else:
        return np.array([], dtype=np.float32)


# ── Main processing ─────────────────────────────────────────────────

def find_source_wav(dumps_path, name):
    """Find the source WAV file inside the dumps directory."""
    # Look for any .wav file in the dumps directory
    wav_files = glob.glob(os.path.join(dumps_path, '*.wav'))
    if wav_files:
        return wav_files[0]
    return None


def process_recording(name, dumps_path, output_dir):
    """Process a single recording: compute VADs, extract and save WAVs."""
    print(f"\n  [{name}]")

    # 1. Read source audio from dumps directory
    source_wav = find_source_wav(dumps_path, name)
    if source_wav is None:
        print(f"    SKIP: No WAV file found in {dumps_path}")
        return False

    samples, sr_orig = read_wav_samples(source_wav)
    print(f"    Source: {os.path.basename(source_wav)} "
          f"({len(samples)/sr_orig:.1f}s, {sr_orig} Hz, {len(samples)} samples)")

    # Resample to target rate if needed
    if sr_orig != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(SAMPLE_RATE, sr_orig)
        up, down = SAMPLE_RATE // g, sr_orig // g
        samples = resample_poly(samples, up, down).astype(np.float32)
        print(f"    Resampled: {sr_orig} Hz → {SAMPLE_RATE} Hz ({len(samples)} samples)")
    sr = SAMPLE_RATE
    audio_duration = len(samples) / sr
    print(f"    Audio: {audio_duration:.1f}s, {sr} Hz, {len(samples)} samples")

    # 2. Load data
    df_mouth = pd.read_csv(os.path.join(dumps_path, 'mouth_position.csv'))
    df_vad = pd.read_csv(os.path.join(dumps_path, 'vad.csv'))

    timestamps = sorted(df_mouth['seconds'].unique())
    n_frames = len(timestamps)
    print(f"    Frames: {n_frames}, span: {timestamps[0]:.2f}s – {timestamps[-1]:.2f}s")

    # 3. Compute frontalized MAR per frame
    frontal_mars = []
    for t in timestamps:
        frame_data = df_mouth[df_mouth['seconds'] == t]
        face_data = frame_data[frame_data['face_idx'] == 0]

        pts_3d = {}
        for _, row in face_data.iterrows():
            idx = int(row['point_type'])
            pts_3d[idx] = (row['x'], row['y'], row['z'])

        frontalized = frontalize_mouth_landmarks(pts_3d)
        frontal_mar = compute_mar(frontalized) if frontalized else None
        frontal_mars.append(frontal_mar)

    # 4. Run V-VAD
    vvad_decisions = run_vvad(frontal_mars, VVAD_PARAMS)

    # 5. Audio VAD (align to frame timestamps)
    vad_times = df_vad['seconds'].values
    vad_vals = df_vad['vadDagcDecFinal'].values
    audio_vad_decisions = []
    for t in timestamps:
        idx = np.argmin(np.abs(vad_times - t))
        audio_vad_decisions.append(int(vad_vals[idx]))

    # 6. Combined VAD (OR / union)
    combined_decisions = [
        1 if (a == 1 or v == 1) else 0
        for a, v in zip(audio_vad_decisions, vvad_decisions)
    ]

    # 7. Convert decisions to time segments
    seg_audio = decisions_to_segments(timestamps, audio_vad_decisions)
    seg_vvad = decisions_to_segments(timestamps, vvad_decisions)
    seg_combined = decisions_to_segments(timestamps, combined_decisions)

    # 8. Extract audio and save WAVs
    results = [
        ('audio_vad', seg_audio, audio_vad_decisions),
        ('vvad', seg_vvad, vvad_decisions),
        ('audio_vvad', seg_combined, combined_decisions),
    ]

    for suffix, segments, decisions in results:
        active_pct = sum(decisions) / len(decisions) * 100
        extracted = extract_segments_from_audio(samples, sr, segments)
        out_path = os.path.join(output_dir, f"{name}_{suffix}.wav")

        if len(extracted) > 0:
            write_wav_from_samples(out_path, extracted, sr)
            dur = len(extracted) / sr
            print(f"    ✓ {suffix:12s}: {len(segments):3d} segments, "
                  f"{dur:6.2f}s audio ({active_pct:5.1f}% active) → {os.path.basename(out_path)}")
        else:
            # Write empty/silent WAV (0.1s silence) so file always exists
            silence = np.zeros(int(sr * 0.1), dtype=np.float32)
            write_wav_from_samples(out_path, silence, sr)
            print(f"    ✗ {suffix:12s}: no active segments → {os.path.basename(out_path)} (silent)")

    return True


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'enrollment')
    os.makedirs(output_dir, exist_ok=True)

    # Find all dumps directories with required CSVs
    dump_dirs = sorted(glob.glob(os.path.join(base_dir, 'dumps_*')))
    dump_dirs = [
        d for d in dump_dirs
        if os.path.isdir(d)
        and os.path.exists(os.path.join(d, 'mouth_position.csv'))
        and os.path.exists(os.path.join(d, 'vad.csv'))
    ]

    print(f"Found {len(dump_dirs)} recordings")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {SAMPLE_RATE} Hz, 32-bit float")

    success = 0
    skipped = 0

    for dump_path in dump_dirs:
        name = os.path.basename(dump_path).replace('dumps_', '')

        if process_recording(name, dump_path, output_dir):
            success += 1
        else:
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Done! {success} recordings processed, {skipped} skipped.")
    print(f"Enrollment WAVs saved to: {output_dir}/")
    print(f"Files per recording: 3 (audio_vad, vvad, audio_vvad)")
    print(f"Total files: {success * 3}")


if __name__ == '__main__':
    main()
