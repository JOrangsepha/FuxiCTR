#!/usr/bin/env python3
"""Summarize gate-collapse ablation into gc_out/results.md."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

OUT = Path(os.environ.get("OUT", "/root/autodl-tmp/gc_out"))
TOKENS = ["user_id", "user_profile", "item_id", "item_attr", "cross", "scenario"]

VARIANTS = [
    ("g6_gate_baseline", "MTR_gc_g6_gate", "gate EQ (baseline, reused rigor)"),
    ("g6_gate_ent0001", "MTR_gc_g6_gate_ent0001", "gate + entropy β=0.001"),
    ("g6_gate_ent0003", "MTR_gc_g6_gate_ent0003", "gate + entropy β=0.003"),
    ("g6_gate_ent001", "MTR_gc_g6_gate_ent001", "gate + entropy β=0.01"),
    ("g6_gate_T2", "MTR_gc_g6_gate_T2", "gate + temperature T=2"),
    ("g6_gate_T4", "MTR_gc_g6_gate_T4", "gate + temperature T=4"),
    ("g6_gate_ent001_T2", "MTR_gc_g6_gate_ent001_T2", "gate + β=0.01 × T=2"),
    ("g6_residual_ent001", "MTR_gc_g6_residual_ent001", "residual + entropy β=0.01"),
    ("g6_residual_ent0003", "MTR_gc_g6_residual_ent0003", "residual + entropy β=0.003"),
]

SEEDS = [2025, 2026, 2027, 2028, 2029]


def mean_std(vals):
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None, 0
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0, len(vals)
    var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
    return m, math.sqrt(var), len(vals)


def fmt(m, s, n=None):
    if m is None:
        return "—"
    if s is None or n == 1:
        return "{:.4f}".format(m)
    return "{:.4f} ± {:.4f}".format(m, s)


def parse_train_log(path: Path):
    """Best validation AUCs from MT RankMixer train log."""
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    # Look for evaluation blocks; take the last / best avg
    click = conv = avg = best_ep = None
    # Patterns used by FuxiCTR MultiTaskModel
    # e.g. [Metrics] click: AUC = 0.61 ... OR Evaluate metrics
    for m in re.finditer(
        r"Best[^\n]*epoch[=:\s]+(\d+)|best_epoch[=:\s]+(\d+)", text, re.I
    ):
        best_ep = int(m.group(1) or m.group(2))
    # Collect all validation AUC triples if present
    # Common: "click: AUC: 0.xxx" and "conversion: AUC: 0.xxx"
    clicks = [float(x) for x in re.findall(r"\bclick\b[^\n]{0,40}?AUC[=:\s]+([0-9.]+)", text, re.I)]
    convs = [float(x) for x in re.findall(r"\bconversion\b[^\n]{0,40}?AUC[=:\s]+([0-9.]+)", text, re.I)]
    # Also bare "AUC = x" monitor mean
    avgs = [float(x) for x in re.findall(r"(?:mean[_\s])?AUC[=:\s]+([0-9.]+)", text, re.I)]
    if clicks and convs:
        # pair by index; use last complete pair as final early-stop selection proxy
        n = min(len(clicks), len(convs))
        pairs = [(clicks[i], convs[i], 0.5 * (clicks[i] + convs[i])) for i in range(n)]
        # pick max avg (monitor)
        click, conv, avg = max(pairs, key=lambda t: t[2])
    elif avgs:
        avg = max(avgs)
    return {
        "click_auc": click,
        "conv_auc": conv,
        "avg_auc": avg,
        "best_epoch": best_ep,
    }


def load_cvr(expid: str):
    p = OUT / "eval_valid" / f"{expid}.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def gate_stats(tag: str, seed: int):
    p = OUT / "analysis" / f"{tag}_s{seed}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    out = {"path": str(p)}
    for e in d.get("entropy", []):
        if e.get("slice") == "all":
            out[f"ent_{e.get('task')}"] = e.get("entropy")
    # gate weights: prefer 'gates' or rows
    rows = d.get("gates") or d.get("rows") or []
    for row in rows:
        if row.get("slice") != "all":
            continue
        task = row.get("task")
        weights = {}
        for tok in TOKENS:
            cell = row.get(tok)
            if isinstance(cell, dict):
                weights[tok] = cell.get("mean")
            elif isinstance(cell, (int, float)):
                weights[tok] = float(cell)
        if not weights:
            continue
        # max over available
        valid = {k: v for k, v in weights.items() if v is not None}
        if not valid:
            continue
        arg = max(valid, key=valid.get)
        out[f"max_{task}"] = valid[arg]
        out[f"argmax_{task}"] = arg
        out[f"weights_{task}"] = valid
    return out


def collapsed(max_w, ent, max_thresh=0.9, ent_thresh=0.5):
    """True if still collapsed by either criterion (strict)."""
    if max_w is None and ent is None:
        return None
    if max_w is not None and max_w >= max_thresh:
        return True
    if ent is not None and ent <= ent_thresh and (max_w is None or max_w >= 0.85):
        # low entropy alone with near-peak mass
        if max_w is not None and max_w >= 0.85:
            return True
    if max_w is not None and max_w < max_thresh and ent is not None and ent > ent_thresh:
        return False
    if max_w is not None and max_w < max_thresh:
        return False
    if ent is not None and ent > ent_thresh:
        return False
    return True


def main():
    lines = []
    lines.append("# Gate-collapse ablation (validation only)")
    lines.append("")
    lines.append(f"OUT=`{OUT}`")
    lines.append("Protocol: `--skip_test`, patience=3, epochs≤10, AliCCP_x1 **validation only**.")
    lines.append("Success (mitigated): across seeds, conversion-gate max weight mean clearly below ~0.9 **or** entropy clearly above ~0.5, without destroying avg AUC vs baseline.")
    lines.append("")
    lines.append("## Summary (mean ± sample std over seeds)")
    lines.append("")
    lines.append("| variant | n | click AUC | conv AUC | avg AUC | clicked CVR | conv gate H | conv max w | collapsed |")
    lines.append("|---------|---|-----------|----------|---------|-------------|-------------|------------|-----------|")

    detail_rows = []
    summary_rows = []

    for tag, exp_prefix, label in VARIANTS:
        clicks, convs, avgs, cvrs = [], [], [], []
        ents, maxws, coll = [], [], []
        n_ok = 0
        for seed in SEEDS:
            expid = f"{exp_prefix}_s{seed}"
            cvr = load_cvr(expid)
            train = parse_train_log(OUT / "logs" / f"{expid}.log")
            g = gate_stats(tag, seed)
            click = conv = avg = cvr_auc = None
            if cvr:
                click = cvr.get("click_auc")
                conv = cvr.get("conv_auc")
                avg = cvr.get("avg_auc")
                cvr_auc = cvr.get("cvr_clicked_auc")
            elif train:
                click = train.get("click_auc")
                conv = train.get("conv_auc")
                avg = train.get("avg_auc")
            ent = maxw = arg = None
            if g:
                ent = g.get("ent_conversion")
                maxw = g.get("max_conversion")
                arg = g.get("argmax_conversion")
            is_coll = collapsed(maxw, ent)
            if click is not None:
                clicks.append(click)
            if conv is not None:
                convs.append(conv)
            if avg is not None:
                avgs.append(avg)
            if cvr_auc is not None:
                cvrs.append(cvr_auc)
            if ent is not None:
                ents.append(ent)
            if maxw is not None:
                maxws.append(maxw)
            if is_coll is not None:
                coll.append(1 if is_coll else 0)
            if avg is not None or g is not None:
                n_ok += 1
            detail_rows.append({
                "tag": tag, "label": label, "expid": expid, "seed": seed,
                "click": click, "conv": conv, "avg": avg, "cvr": cvr_auc,
                "ent": ent, "maxw": maxw, "argmax": arg, "collapsed": is_coll,
                "exit": (OUT / "logs" / f"{expid}.exit").read_text().strip()
                if (OUT / "logs" / f"{expid}.exit").is_file() else "?",
                "secs": (OUT / "logs" / f"{expid}.time").read_text().strip()
                if (OUT / "logs" / f"{expid}.time").is_file() else "?",
            })
        if n_ok == 0 and not any(
            (OUT / "checkpoints" / "AliCCP_x1" / f"{exp_prefix}_s{s}.model").is_file()
            for s in SEEDS
        ):
            continue
        mc, sc, nc = mean_std(clicks)
        mv, sv, nv = mean_std(convs)
        ma, sa, na = mean_std(avgs)
        mcv, scv, ncv = mean_std(cvrs)
        me, se, ne = mean_std(ents)
        mm, sm, nm = mean_std(maxws)
        n_coll = sum(coll) if coll else None
        n_g = len(coll) if coll else 0
        coll_cell = f"{n_coll}/{n_g}" if n_g else "—"
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                label,
                max(nc, nv, na, ne, nm),
                fmt(mc, sc, nc),
                fmt(mv, sv, nv),
                fmt(ma, sa, na),
                fmt(mcv, scv, ncv),
                fmt(me, se, ne),
                fmt(mm, sm, nm),
                coll_cell,
            )
        )
        summary_rows.append({
            "tag": tag, "label": label,
            "avg": ma, "ent": me, "maxw": mm,
            "n_coll": n_coll, "n_g": n_g,
        })

    lines.append("")
    lines.append("## Mitigation verdict")
    lines.append("")
    base = next((r for r in summary_rows if r["tag"] == "g6_gate_baseline"), None)
    for r in summary_rows:
        if r["tag"] == "g6_gate_baseline":
            continue
        parts = []
        mitigated = False
        if r["maxw"] is not None and r["maxw"] < 0.9:
            parts.append("max_w={:.3f} < 0.9".format(r["maxw"]))
            mitigated = True
        if r["ent"] is not None and r["ent"] > 0.5:
            parts.append("H={:.3f} > 0.5".format(r["ent"]))
            mitigated = True
        auc_ok = True
        if base and base["avg"] is not None and r["avg"] is not None:
            drop = base["avg"] - r["avg"]
            if drop > 0.01:
                auc_ok = False
                parts.append("avg AUC drop {:.4f} vs baseline".format(drop))
            else:
                parts.append("avg AUC Δ={:+.4f}".format(r["avg"] - base["avg"]))
        if mitigated and auc_ok:
            verdict = "**mitigated**"
        elif mitigated and not auc_ok:
            verdict = "gates improved but AUC hurt"
        else:
            verdict = "not mitigated"
        lines.append("- {}: {} ({})".format(r["label"], verdict, "; ".join(parts) if parts else "no gate stats"))
    lines.append("")
    lines.append("## Per-seed detail")
    lines.append("")
    lines.append("| expid | exit | secs | click | conv | avg | clicked_CVR | conv H | max w | argmax | collapsed |")
    lines.append("|-------|------|------|-------|------|-----|-------------|--------|-------|--------|-----------|")
    for row in detail_rows:
        if row["click"] is None and row["ent"] is None and row["exit"] == "?":
            # skip empty unused optional variants
            ck = OUT / "checkpoints" / "AliCCP_x1" / f"{row['expid']}.model"
            if not ck.is_file():
                continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["expid"],
                row["exit"],
                row["secs"],
                "—" if row["click"] is None else "{:.5f}".format(row["click"]),
                "—" if row["conv"] is None else "{:.5f}".format(row["conv"]),
                "—" if row["avg"] is None else "{:.5f}".format(row["avg"]),
                "—" if row["cvr"] is None else "{:.5f}".format(row["cvr"]),
                "—" if row["ent"] is None else "{:.4f}".format(row["ent"]),
                "—" if row["maxw"] is None else "{:.4f}".format(row["maxw"]),
                row["argmax"] or "—",
                "—" if row["collapsed"] is None else str(row["collapsed"]),
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Baseline checkpoints/logs/gate JSON are reused from `rg_out` when `REUSE_BASELINE=1`.")
    lines.append("- Impression-level conversion AUC is CTCVR; clicked CVR is conversion AUC on `click==1` rows.")
    lines.append("- Collapsed = conversion max weight ≥ 0.9 (or max≥0.85 with entropy ≤ 0.5).")
    lines.append("")

    out = OUT / "results.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
