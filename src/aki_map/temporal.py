"""Method functions extracted from the final archived research code."""
import numpy as np

CHANNELS = ("sys", "dia", "map", "hr")
CENTERS = np.array([120, 70, 80, 75], dtype=np.float32)
SCALES = np.array([30, 20, 20, 20], dtype=np.float32)
WINDOWS = (15, 30, 60)

def patient_grid(records):
    """Same-source BP first; ART > NIBP > generic, direct MAP > derived within source."""
    r = records.drop_duplicates(["minute", "component", "source", "value"]).copy()
    if r.empty:
        raise ValueError("No valid observations")
    durations = r.duration.unique()
    if np.ptp(durations) > 0.02:
        raise ValueError("Multiple anesthesia durations within one locked visit")
    n = int(np.floor(durations[0] / 5)) + 1
    measured = np.full((n, 4), np.nan, dtype=np.float32)
    stamp = np.full((n, 4), np.nan, dtype=np.float32)
    source = np.zeros(n, dtype=np.int8)
    method = np.zeros(n, dtype=np.int8)
    counts = np.zeros((n, 4), dtype=np.float32)
    agg = r.groupby(["bin", "source", "component"]).agg(
        value=("value", "median"), minute=("minute", "max"), count=("value", "size"))
    for b in r.bin.unique():
        b = int(b)
        for rank, src in enumerate(("art", "nibp", "generic"), 1):
            rows = {c: agg.loc[(b, src, c)] if (b, src, c) in agg.index else None for c in CHANNELS[:3]}
            sbp, dbp, mp = (rows[c] for c in CHANNELS[:3])
            pair = sbp is not None and dbp is not None and 5 <= sbp.value - dbp.value <= 180
            derived = (sbp.value + 2 * dbp.value) / 3 if pair else np.nan
            if mp is None and not (pair and 30 <= derived <= 200):
                continue
            source[b] = rank
            method[b] = 1 if mp is not None else 2
            for j, c in enumerate(CHANNELS[:2]):
                row = rows[c]
                if row is not None and (sbp is None or dbp is None or pair):
                    measured[b, j], stamp[b, j], counts[b, j] = row.value, row.minute, row["count"]
            if mp is not None:
                measured[b, 2], stamp[b, 2], counts[b, 2] = mp.value, mp.minute, mp["count"]
            else:
                measured[b, 2] = derived
                # Both components must remain fresh; use the older last component time.
                stamp[b, 2] = min(sbp.minute, dbp.minute)
                counts[b, 2] = min(sbp["count"], dbp["count"])
            break
        if (b, "hr", "hr") in agg.index:
            row = agg.loc[(b, "hr", "hr")]
            measured[b, 3], stamp[b, 3], counts[b, 3] = row.value, row.minute, row["count"]
    return {"measured": measured, "stamp": stamp, "source": source,
            "method": method, "counts": counts}


def sequence_and_summary(grid):
    """All cumulative operations are causal. Placeholders never enter measured summaries."""
    x, stamp = grid["measured"], grid["stamp"]
    n = len(x)
    times = np.arange(n, dtype=float) * 5
    obs = np.isfinite(x)
    age = np.full_like(x, 1440)
    filled = np.tile(CENTERS, (n, 1))
    valid_locf = np.zeros_like(x)
    for j in range(4):
        last = -1
        for k in range(n):
            if obs[k, j]:
                last = k
            if last >= 0:
                age[k, j] = times[k] - stamp[last, j]
                if age[k, j] <= 10:
                    filled[k, j] = x[last, j]
                    valid_locf[k, j] = 1
    # Age and LOCF availability distinguish an imputed gap from a real observation.
    source_onehot = np.stack([grid["source"] == s for s in (1, 2, 3)], axis=1)
    seq = np.concatenate([(filled - CENTERS) / SCALES, obs, valid_locf,
                          np.minimum(age, 60) / 60, np.log1p(grid["counts"]) / 5,
                          source_onehot, (grid["method"] == 1)[:, None],
                          (grid["method"] == 2)[:, None]], axis=1).astype(np.float32)
    columns, names = [times / 60], ["elapsed_hours"]
    for j, channel in enumerate(CHANNELS):
        values = x[:, j]
        count = np.cumsum(obs[:, j])
        denom = np.maximum(count, 1)
        total = np.nancumsum(values)
        mean = total / denom
        sd = np.sqrt(np.maximum(np.nancumsum(values ** 2) / denom - mean ** 2, 0))
        minimum = np.minimum.accumulate(np.where(obs[:, j], values, np.inf))
        maximum = np.maximum.accumulate(np.where(obs[:, j], values, -np.inf))
        for label, v in (("mean", mean), ("sd", sd), ("min", minimum), ("max", maximum)):
            columns.append(np.where(count > 0, v, np.nan)); names.append(f"{channel}_{label}_observed")
        columns.extend([count / np.arange(1, n + 1), np.minimum(age[:, j], 1440),
                        np.where(age[:, j] <= 10, filled[:, j], np.nan)])
        names.extend([f"{channel}_coverage", f"{channel}_age", f"{channel}_recent"])
    for j, thresholds in ((2, (65, 60, 55, 50)), (0, (90,))):
        denom = np.maximum(np.cumsum(obs[:, j]), 1)
        for thr in thresholds:
            # Observed-bin fractions, NOT imputed minutes below threshold.
            columns.append(np.cumsum(obs[:, j] & (x[:, j] < thr)) / denom)
            names.append(f"{CHANNELS[j]}_observed_fraction_below{thr}")
            columns.append(np.nancumsum(np.maximum(thr - x[:, j], 0)) / denom)
            names.append(f"{CHANNELS[j]}_observed_deficit{thr}")
    summary = np.array(columns, dtype=np.float32).T
    landmarks = np.flatnonzero((times >= 60) & (age[:, 2] <= 10))
    # Audit only: this mask must never filter risk-model landmarks.
    future_complete = np.array([k + 3 < n and obs[k + 1:k + 4, 2].all() for k in landmarks], dtype=bool)
    audit = {"total_landmarks_after60": max(n - 12, 0), "risk_landmarks": len(landmarks),
             "recent_map_missing": max(n - 12, 0) - len(landmarks),
             "future15_complete": int(future_complete.sum()),
             "future15_end_truncated": int((landmarks + 3 >= n).sum()),
             "future15_missing_before_end": int(((landmarks + 3 < n) & ~future_complete).sum())}
    return seq, summary, names, landmarks, audit


def multiscale_features(measured):
    """Observed-only features from (t-window, t]; no future values or LOCF summaries."""
    d = np.asarray(measured, dtype=np.float64)
    count_past = np.isfinite(d).cumsum(0)
    past_mean = np.nancumsum(d, axis=0) / np.maximum(count_past, 1)
    values, names = [], []
    for minutes in WINDOWS:
        k = minutes // 5
        padded = np.pad(d, ((k - 1, 0), (0, 0)), constant_values=np.nan)
        window = np.lib.stride_tricks.sliding_window_view(padded, k, axis=0)
        observed = np.isfinite(window)
        count = observed.sum(-1)
        safe = np.maximum(count, 1)
        y = np.where(observed, window, 0)
        mean = y.sum(-1) / safe
        sd = np.sqrt(np.maximum((y * y).sum(-1) / safe - mean * mean, 0))
        minimum = np.where(observed, window, np.inf).min(-1)
        maximum = np.where(observed, window, -np.inf).max(-1)
        coverage = count / np.minimum(np.arange(1, len(d) + 1), k)[:, None]
        time = np.arange(k, dtype=float) * 5
        sx = (observed * time).sum(-1)
        sxx = (observed * time * time).sum(-1)
        denom = count * sxx - sx * sx
        slope = np.divide(count * (y * time).sum(-1) - sx * y.sum(-1), denom,
                          out=np.full_like(mean, np.nan), where=(count >= 2) & (denom > 0))
        for j, channel in enumerate(("sys", "dia", "map", "hr")):
            for label, feature in (("mean", mean), ("sd", sd), ("min", minimum),
                                   ("max", maximum), ("coverage", coverage),
                                   ("vs_cumulative", mean - past_mean), ("slope", slope)):
                valid = np.ones(len(d), bool) if label == "coverage" else count[:, j] > 0
                values.append(np.where(valid, feature[:, j], np.nan))
                names.append(f"{channel}_last{minutes}_{label}")
        for threshold in (60, 65):
            fraction = (window[:, 2] < threshold).sum(-1) / safe[:, 2]
            values.append(np.where(count[:, 2] > 0, fraction, np.nan))
            names.append(f"map_last{minutes}_observed_fraction_below{threshold}")
    return np.asarray(values, np.float32).T, names
