#!/usr/bin/env python3
# coding: utf-8

"""
Reindex face IDs in dump CSVs using stable track assignment.

process_video.py assigns face_idx per frame based on detection order (tied to
face distance), so IDs can swap when people change relative position.  This
script computes persistent track IDs and rewrites the CSV dumps in-place so
that every downstream consumer (rendering scripts, analysis) sees stable
identities.

Two tracking strategies are available:
  1. Embedding-based:  Uses precomputed face embeddings (.npz) to match faces
     across frames by cosine similarity.  Requires a real face recognition
     model to produce meaningful embeddings.
  2. Position-based (fallback):  Uses the average X, Y landmark positions from
     mouth_position.csv to match faces by spatial proximity.  Activated
     automatically when the embeddings are all-zero placeholders or when
     --use-position-tracking is set.

Affected files (must contain a face_idx column):
  - mouth_position.csv
  - face_position.csv
  - speaker_identification.csv  (timestamps also snapped to face_count grid)

Usage (inside stitch_demo.sh, after extract_face_embeddings.py):
  python3 reindex_face_ids.py --dumps_dir dumps --embeddings face_embeddings.npz
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment


# ---- Embedding helpers (same algorithm used by the rendering scripts) ------

def _normalize_embeddings(emb):
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    n = np.where(n > 1e-12, n, 1.0)
    return emb / n


def _assignment_by_similarity(prev_track_embs, curr_face_embs,
                              similarity_threshold=0.0):
    n_curr = len(curr_face_embs)
    n_prev = len(prev_track_embs)
    if n_curr == 0:
        return []
    if n_prev == 0:
        return [(c[0], i) for i, c in enumerate(curr_face_embs)]

    prev_ids = [t[0] for t in prev_track_embs]
    prev_emb = np.array([t[1] for t in prev_track_embs], dtype=np.float64)
    curr_idx = [c[0] for c in curr_face_embs]
    curr_emb = np.array([c[1] for c in curr_face_embs], dtype=np.float64)
    prev_emb = _normalize_embeddings(prev_emb)
    curr_emb = _normalize_embeddings(curr_emb)
    sim = np.dot(curr_emb, prev_emb.T)
    cost = 1.0 - np.clip(sim, -1.0, 1.0)

    if n_curr <= n_prev:
        row_ind, col_ind = linear_sum_assignment(cost)
        return [(curr_idx[r], prev_ids[c]) for r, c in zip(row_ind, col_ind)]

    new_col_cost = 1.0 - similarity_threshold
    cost_mat = np.full((n_curr, n_curr), new_col_cost, dtype=np.float64)
    cost_mat[:, :n_prev] = cost
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    next_id = max(prev_ids) + 1
    result = []
    new_id = next_id
    for r, c in zip(row_ind, col_ind):
        if c < n_prev:
            track_id = prev_ids[c]
        else:
            track_id = new_id
            new_id += 1
        result.append((curr_idx[r], track_id))
    return result


def _are_embeddings_placeholder(embeddings):
    """
    Detect whether embeddings are placeholder (all-zero) vectors.
    When extract_face_embeddings.py runs without a real model, it writes
    np.zeros(128) for every face — these are useless for similarity matching.
    """
    if embeddings.size == 0:
        return True
    norms = np.linalg.norm(embeddings, axis=1)
    # If >90 % of vectors are near-zero the data is placeholder
    return float(np.mean(norms < 1e-8)) > 0.9


def compute_assignment_map(embeddings, seconds_arr, face_idx_arr,
                           df_face_count, similarity_threshold=0.3,
                           drop_after_frames=60):
    """
    Build (rounded_seconds, orig_face_idx) -> stable_track_id mapping
    by matching face embeddings across consecutive frames.
    """
    sec_key = np.round(seconds_arr, 6)
    key_to_emb = {}
    for i in range(len(sec_key)):
        key = (float(sec_key[i]), int(face_idx_arr[i]))
        key_to_emb[key] = embeddings[i]

    frames = df_face_count.sort_values('seconds')[['seconds', 'face_count']].values
    assignment_map = {}
    tracks = {}

    for frame_idx, (seconds, face_count) in enumerate(frames):
        sec_k = round(float(seconds), 6)
        curr_list = []
        for fi in range(int(face_count)):
            key = (sec_k, fi)
            if key in key_to_emb:
                curr_list.append((fi, key_to_emb[key].copy()))
        if not curr_list:
            to_drop = [tid for tid, (_, last) in tracks.items()
                       if frame_idx - last >= drop_after_frames]
            for tid in to_drop:
                del tracks[tid]
            continue

        prev_list = [(tid, emb) for tid, (emb, last) in tracks.items()]
        assignment = _assignment_by_similarity(
            prev_list, curr_list, similarity_threshold=similarity_threshold)

        for curr_face_idx, track_id in assignment:
            assignment_map[(sec_k, curr_face_idx)] = track_id
            key = (sec_k, curr_face_idx)
            if key in key_to_emb:
                tracks[track_id] = (key_to_emb[key].copy(), frame_idx)

        for tid in list(tracks.keys()):
            if frame_idx - tracks[tid][1] >= drop_after_frames:
                del tracks[tid]

    return assignment_map


# ---- Position-based tracking (fallback when embeddings are unavailable) ----

def compute_assignment_map_by_position(df_mouth, df_face_count,
                                       max_distance=300.0,
                                       ema_alpha=0.3,
                                       drop_after_frames=60):
    """
    Build (rounded_seconds, orig_face_idx) -> stable_track_id mapping
    by matching face centroid positions (average X, Y of all 68 landmarks)
    across consecutive frames.

    This is the fallback when face embeddings are placeholder (all-zero).
    Uses exponential moving average of centroids and the Hungarian algorithm
    for optimal assignment.

    Args:
        df_mouth: DataFrame from mouth_position.csv (columns: seconds,
                  face_idx, point_type, x, y, z)
        df_face_count: DataFrame from face_count.csv (seconds, face_count)
        max_distance: Maximum pixel distance to match a detection to a track.
        ema_alpha: EMA smoothing factor for tracked centroid positions.
        drop_after_frames: Remove a track after this many unseen frames.

    Returns:
        dict: (rounded_seconds, orig_face_idx) -> stable_track_id
    """
    # Pre-compute face centroids per (seconds, face_idx)
    # Use mean of all landmark X, Y coordinates as the centroid
    centroids = (
        df_mouth
        .groupby(['seconds', 'face_idx'])
        .agg(cx=('x', 'mean'), cy=('y', 'mean'))
        .reset_index()
    )
    centroid_lookup = {}
    for _, row in centroids.iterrows():
        key = (round(float(row['seconds']), 6), int(row['face_idx']))
        centroid_lookup[key] = (float(row['cx']), float(row['cy']))

    frames = df_face_count.sort_values('seconds')[['seconds', 'face_count']].values
    assignment_map = {}
    # tracks: {track_id: {'cx': float, 'cy': float, 'last_frame': int}}
    tracks = {}
    next_track_id = 0

    for frame_idx, (seconds, face_count) in enumerate(frames):
        sec_k = round(float(seconds), 6)
        n_det = int(face_count)

        # Collect centroids for all detected faces in this frame
        curr_centroids = []  # list of (face_idx, cx, cy)
        for fi in range(n_det):
            key = (sec_k, fi)
            if key in centroid_lookup:
                cx, cy = centroid_lookup[key]
                curr_centroids.append((fi, cx, cy))

        if not curr_centroids:
            # Cleanup stale tracks
            to_drop = [tid for tid, t in tracks.items()
                       if frame_idx - t['last_frame'] >= drop_after_frames]
            for tid in to_drop:
                del tracks[tid]
            continue

        n_curr = len(curr_centroids)

        # First frame or no active tracks → initialise
        active = [(tid, t) for tid, t in tracks.items()
                  if frame_idx - t['last_frame'] < drop_after_frames]
        n_active = len(active)

        if n_active == 0:
            for fi, cx, cy in curr_centroids:
                tid = next_track_id
                next_track_id += 1
                tracks[tid] = {'cx': cx, 'cy': cy, 'last_frame': frame_idx}
                assignment_map[(sec_k, fi)] = tid
            continue

        # Build cost matrix: curr_centroids x active_tracks (Euclidean dist)
        cost = np.empty((n_curr, n_active), dtype=np.float64)
        for i, (_, dcx, dcy) in enumerate(curr_centroids):
            for j, (_, t) in enumerate(active):
                cost[i, j] = np.sqrt((dcx - t['cx'])**2 +
                                     (dcy - t['cy'])**2)

        # Pad to square
        dim = max(n_curr, n_active)
        pad = np.full((dim, dim), max_distance * 2, dtype=np.float64)
        pad[:n_curr, :n_active] = cost
        rows, cols = linear_sum_assignment(pad)

        matched_dets = set()
        for r, c in zip(rows, cols):
            if r < n_curr and c < n_active and cost[r, c] <= max_distance:
                fi = curr_centroids[r][0]
                cx, cy = curr_centroids[r][1], curr_centroids[r][2]
                tid = active[c][0]
                assignment_map[(sec_k, fi)] = tid
                matched_dets.add(r)
                # Update EMA
                tracks[tid]['cx'] = (ema_alpha * cx +
                                     (1 - ema_alpha) * tracks[tid]['cx'])
                tracks[tid]['cy'] = (ema_alpha * cy +
                                     (1 - ema_alpha) * tracks[tid]['cy'])
                tracks[tid]['last_frame'] = frame_idx

        # New tracks for unmatched detections
        for i in range(n_curr):
            if i not in matched_dets:
                fi = curr_centroids[i][0]
                cx, cy = curr_centroids[i][1], curr_centroids[i][2]
                tid = next_track_id
                next_track_id += 1
                tracks[tid] = {'cx': cx, 'cy': cy, 'last_frame': frame_idx}
                assignment_map[(sec_k, fi)] = tid

        # Cleanup stale tracks
        to_drop = [tid for tid, t in tracks.items()
                   if frame_idx - t['last_frame'] >= drop_after_frames]
        for tid in to_drop:
            del tracks[tid]

    return assignment_map


# ---- Reindex helpers -------------------------------------------------------

def _reindex_df(df, assignment_map):
    """Replace face_idx using assignment_map keyed by (rounded seconds, face_idx)."""
    def _new_idx(row):
        key = (round(float(row['seconds']), 6), int(row['face_idx']))
        return assignment_map.get(key, row['face_idx'])

    df = df.copy()
    df['face_idx'] = df.apply(_new_idx, axis=1).astype(int)
    return df


def _snap_seconds(seconds, reference):
    """Snap a single timestamp to the nearest value in a sorted reference array."""
    idx = np.searchsorted(reference, seconds)
    candidates = []
    if idx > 0:
        candidates.append(reference[idx - 1])
    if idx < len(reference):
        candidates.append(reference[idx])
    return float(min(candidates, key=lambda x: abs(x - seconds)))


def _reindex_speaker_csv(df, assignment_map, ref_seconds):
    """
    Snap speaker_identification timestamps to the face_count grid, then
    reindex face_idx using the assignment_map.
    """
    df = df.copy()
    df['seconds'] = df['seconds'].apply(
        lambda s: _snap_seconds(float(s), ref_seconds))

    def _new_idx(row):
        key = (round(float(row['seconds']), 6), int(row['face_idx']))
        return assignment_map.get(key, row['face_idx'])

    df['face_idx'] = df.apply(_new_idx, axis=1).astype(int)
    return df


# ---- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Reindex face IDs in dump CSVs with stable '
                    'track IDs (embedding-based or position-based).')
    parser.add_argument('--dumps_dir', type=str, default='dumps',
                        help='Directory with CSV dump files')
    parser.add_argument('--embeddings', type=str, default='face_embeddings.npz',
                        help='Path to .npz with precomputed face embeddings')
    parser.add_argument('--similarity-threshold', type=float, default=0.3,
                        help='Min cosine similarity to match a face to an '
                             'existing track (default: 0.3)')
    parser.add_argument('--drop-after-frames', type=int, default=60,
                        help='Drop a track after this many unseen frames '
                             '(default: 60)')
    parser.add_argument('--use-position-tracking', action='store_true',
                        default=False,
                        help='Force position-based tracking instead of '
                             'embedding-based (uses X, Y landmark centroids)')
    parser.add_argument('--max-distance', type=float, default=300.0,
                        help='Max pixel distance for position-based matching '
                             '(default: 300)')
    parser.add_argument('--ema-alpha', type=float, default=0.3,
                        help='EMA smoothing factor for position tracking '
                             '(default: 0.3)')
    args = parser.parse_args()

    dumps = Path(args.dumps_dir)
    emb_path = Path(args.embeddings)

    print("Reindexing face IDs with stable track assignment")
    print("=" * 55)

    # Load face_count (needed for frame iteration order)
    fc_path = dumps / 'face_count.csv'
    if not fc_path.exists():
        print(f"  Error: {fc_path} not found")
        return
    df_face_count = pd.read_csv(fc_path)
    ref_seconds = np.sort(df_face_count['seconds'].values.astype(np.float64))

    # Load mouth_position (needed for position-based tracking fallback)
    mp_path = dumps / 'mouth_position.csv'
    df_mouth_orig = None
    if mp_path.exists():
        df_mouth_orig = pd.read_csv(mp_path)

    # Decide which tracking strategy to use
    use_position = args.use_position_tracking
    embeddings = None
    seconds_arr = None
    face_idx_arr = None

    if not use_position and emb_path.exists():
        data = np.load(emb_path, allow_pickle=False)
        embeddings = np.asarray(data['embeddings'], dtype=np.float64)
        seconds_arr = np.asarray(data['seconds'], dtype=np.float64).ravel()
        face_idx_arr = np.asarray(data['face_idx'], dtype=np.int64).ravel()
        print(f"  Loaded {len(seconds_arr)} embeddings from {emb_path}")

        # Auto-detect placeholder embeddings (all-zero from dummy model)
        if _are_embeddings_placeholder(embeddings):
            print("  ⚠ Embeddings are all-zero (placeholder) — they cannot "
                  "distinguish faces.")
            print("    Falling back to position-based tracking using "
                  "landmark centroids.")
            use_position = True
    elif not use_position:
        print(f"  Embeddings file not found: {emb_path}")
        print("  Falling back to position-based tracking.")
        use_position = True

    # Compute stable assignment
    if use_position:
        if df_mouth_orig is None:
            print(f"  Error: {mp_path} not found — cannot do position tracking")
            return
        print("  Strategy: POSITION-BASED (landmark centroid tracking)")
        assignment_map = compute_assignment_map_by_position(
            df_mouth_orig, df_face_count,
            max_distance=args.max_distance,
            ema_alpha=args.ema_alpha,
            drop_after_frames=args.drop_after_frames)
    else:
        print("  Strategy: EMBEDDING-BASED (cosine similarity)")
        assignment_map = compute_assignment_map(
            embeddings, seconds_arr, face_idx_arr, df_face_count,
            similarity_threshold=args.similarity_threshold,
            drop_after_frames=args.drop_after_frames)

    n_reassigned = sum(1 for (_, fi), tid in assignment_map.items() if fi != tid)
    print(f"  Computed assignment map: {len(assignment_map)} entries, "
          f"{n_reassigned} face_idx values changed")

    # Reindex mouth_position.csv
    if mp_path.exists():
        df = df_mouth_orig if df_mouth_orig is not None else pd.read_csv(mp_path)
        df = _reindex_df(df, assignment_map)
        df.to_csv(mp_path, index=False)
        print(f"  Reindexed {mp_path}")
    else:
        print(f"  Skipped (not found): {mp_path}")

    # Reindex face_position.csv
    fp_path = dumps / 'face_position.csv'
    if fp_path.exists():
        df = pd.read_csv(fp_path)
        df = _reindex_df(df, assignment_map)
        df.to_csv(fp_path, index=False)
        print(f"  Reindexed {fp_path}")
    else:
        print(f"  Skipped (not found): {fp_path}")

    # Reindex speaker_identification.csv (also snap timestamps)
    si_path = dumps / 'speaker_identification.csv'
    if si_path.exists():
        df = pd.read_csv(si_path)
        df = _reindex_speaker_csv(df, assignment_map, ref_seconds)
        df.to_csv(si_path, index=False)
        print(f"  Reindexed {si_path} (timestamps snapped to face_count grid)")
    else:
        print(f"  Skipped (not found): {si_path}")

    unique_ids = set(assignment_map.values())
    print(f"\n  Stable track IDs: {sorted(unique_ids)}")
    print("Done!")


if __name__ == '__main__':
    main()
