#!/usr/bin/env python3
"""Follow-up suite bookkeeping: PLE config selection and the final report.

Subcommands
-----------
select-ple   Pick the PLE config with the highest mean validation avg AUC
             (mean of click and conversion AUC, restored best checkpoint, as
             logged by run_expid.py) over seeds 2025-2027. Validation only.
summarize    Build followup_summary.csv and followup_results.md from the
             training logs, the eval_cvr_clicked.py JSONs (valid and test),
             param counts, and the gradient audit.
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from summarize_multiseed import parse_log  # noqa: E402

PLE_CANDIDATES = {
    "PLE_rg": "baseline: experts [128,64], lr 1e-3, dropout 0",
    "PLE_fu_wide": "experts [256,128]",
    "PLE_fu_lr5e4": "lr 5e-4",
    "PLE_fu_drop01": "dropout 0.1",
}
TUNE_SEEDS = ["2025", "2026", "2027"]
ALL_SEEDS = ["2025", "2026", "2027", "2028", "2029"]

LABELS = [
    ("MTR_rg_g3_mean", "g3_mean", "EQ"),
    ("MTR_rg_g6_mean", "g6_mean", "EQ"),
    ("MTR_rg_g6_residual", "g6_residual", "EQ"),
    ("MTR_rg_g6_gate", "g6_gate", "EQ"),
    ("MTR_rg_g6_random", "g6_random", "EQ"),
    ("MTR_rg_g6_sequential", "g6_sequential", "EQ"),
    ("MTR_fu_g6_mean_norm", "g6_mean", "NORM"),
    ("MTR_fu_g6_residual_norm", "g6_residual", "NORM"),
    ("MTR_fu_g6_gate_norm", "g6_gate", "NORM"),
    ("MTR_fu_g6_mean_w10", "g6_mean", "W[1,10]"),
    ("MTR_fu_g6_residual_w10", "g6_residual", "W[1,10]"),
    ("MTR_fu_g6_gate_w10", "g6_gate", "W[1,10]"),
    ("PLE_rg", "PLE_base", "EQ"),
    ("PLE_fu_wide", "PLE_wide", "EQ"),
    ("PLE_fu_lr5e4", "PLE_lr5e4", "EQ"),
    ("PLE_fu_drop01", "PLE_drop01", "EQ"),
]
LABEL_OF = {stem: (name, loss) for stem, name, loss in LABELS}


def split_expid(expid):
    m = re.match(r"^(.*)_s(\d+)$", expid)
    return (m.group(1), m.group(2)) if m else (expid, "")


def display(stem):
    name, loss = LABEL_OF.get(stem, (stem, "?"))
    return "{} [{}]".format(name, loss)


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def training_rows(log_dirs):
    rows = {}
    for log_dir in log_dirs:
        for log in sorted(Path(log_dir).glob("*.log")):
            if log.name.endswith(".attempt1.log"):
                continue
            stem, seed = split_expid(log.stem)
            if stem not in LABEL_OF:
                continue
            text = log.read_text(errors="replace")
            best_epoch, valid, _test = parse_log(text)
            exit_code = read_text(log.with_suffix(".exit"))
            row = {"expid": log.stem, "stem": stem, "seed": seed, "exit_code": exit_code,
                   "seconds": read_text(log.with_suffix(".time")), "best_epoch": best_epoch,
                   "log_path": str(log),
                   "norm_train_loss_seen": ("Train loss: 2.000000" in text) if "NORM" in text else ""}
            if "click_auc" in valid and "conversion_auc" in valid:
                row["valid_click_auc"] = valid["click_auc"]
                row["valid_conv_auc"] = valid["conversion_auc"]
                row["valid_click_logloss"] = valid["click_logloss"]
                row["valid_conv_logloss"] = valid["conversion_logloss"]
                row["valid_avg_auc"] = (valid["click_auc"] + valid["conversion_auc"]) / 2.0
            rows[log.stem] = row
    return rows


def ok(row):
    return row.get("exit_code") == "0" and "valid_avg_auc" in row


def mean_sd(values):
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not values:
        return "n/a", 0
    arr = np.array(values, dtype=float)
    sd = arr.std(ddof=1) if arr.size > 1 else float("nan")
    return "{:.5f} ± {:.5f}".format(arr.mean(), sd), arr.size


def cmd_select_ple(args):
    rows = training_rows(args.logs)
    table = []
    for stem, desc in PLE_CANDIDATES.items():
        vals = [rows["{}_s{}".format(stem, s)]["valid_avg_auc"] for s in TUNE_SEEDS
                if "{}_s{}".format(stem, s) in rows and ok(rows["{}_s{}".format(stem, s)])]
        table.append({"stem": stem, "desc": desc, "n": len(vals),
                      "mean_valid_avg_auc": float(np.mean(vals)) if vals else None,
                      "values": vals})
    eligible = [t for t in table if t["n"] == len(TUNE_SEEDS)] or [t for t in table if t["n"] >= 2]
    if not eligible:
        print("NO_ELIGIBLE_PLE_CANDIDATE")
        Path(args.output).write_text(json.dumps({"best": None, "table": table}, indent=2))
        return 1
    best = max(eligible, key=lambda t: t["mean_valid_avg_auc"])
    result = {"best": best["stem"], "rule": "max mean valid avg AUC over seeds {} (validation only)".format(
        ",".join(TUNE_SEEDS)), "table": table}
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(best["stem"])
    return 0


def load_json_dir(path):
    out = {}
    p = Path(path)
    if not p.is_dir():
        return out
    for js in sorted(p.glob("*.json")):
        try:
            rec = json.loads(js.read_text())
        except (OSError, ValueError):
            continue
        out[rec["expid"]] = rec
    return out


def paired(rows_a, rows_b, key):
    from scipy import stats
    seeds = sorted(set(rows_a) & set(rows_b))
    a = np.array([rows_a[s][key] for s in seeds], dtype=float)
    b = np.array([rows_b[s][key] for s in seeds], dtype=float)
    if len(seeds) < 2:
        return None
    d = a - b
    t, p = stats.ttest_rel(a, b)
    return {"n": len(seeds), "mean": d.mean(), "sd": d.std(ddof=1), "wins": int((d > 0).sum()),
            "t": float(t), "p": float(p)}


def fmt_paired(label, metric, res):
    if res is None:
        return "| {} | {} | n<2 | | | |".format(label, metric)
    return "| {} | {} | {:+.5f} ± {:.5f} | {}/{} | {:+.2f} | {:.3f} |".format(
        label, metric, res["mean"], res["sd"], res["wins"], res["n"], res["t"], res["p"])


def cmd_summarize(args):
    out = Path(args.out)
    train = training_rows(args.logs)
    valid = load_json_dir(out / "eval_valid")
    test = load_json_dir(out / "final_test" / "json")
    params = {}
    pc = out / "param_counts.csv"
    if pc.is_file():
        for row in csv.DictReader(open(pc)):
            params[row["label"]] = row
    sel = {}
    if (out / "ple_selection.json").is_file():
        sel = json.loads((out / "ple_selection.json").read_text())
    best_ple = sel.get("best")

    expids = sorted(set(train) | set(valid) | set(test))
    fields = ["expid", "variant", "loss_weight", "seed", "exit_code", "seconds", "best_epoch",
              "valid_click_auc", "valid_conv_auc", "valid_avg_auc", "valid_click_logloss", "valid_conv_logloss",
              "valid_eval_click_auc", "valid_eval_conv_auc", "valid_cvr_clicked_auc", "valid_cvr_clicked_auc_ratio",
              "valid_cvr_clicked_logloss", "valid_n_click", "valid_n_conv_in_clicked",
              "test_click_auc", "test_conv_auc", "test_avg_auc", "test_cvr_clicked_auc",
              "test_cvr_clicked_auc_ratio", "test_click_logloss", "test_conv_logloss", "test_cvr_clicked_logloss",
              "params_total", "params_non_embedding", "in_frozen_set"]
    frozen = set()
    fz = out / "final_test" / "FROZEN_SET.txt"
    if fz.is_file():
        frozen = {line.split("|")[0] for line in fz.read_text().splitlines() if line and not line.startswith("#")}
    table = []
    for e in expids:
        stem, seed = split_expid(e)
        name, loss = LABEL_OF.get(stem, (stem, "?"))
        t = train.get(e, {})
        v = valid.get(e, {})
        s = test.get(e, {})
        r = {"expid": e, "variant": name, "loss_weight": loss, "seed": seed,
             "exit_code": t.get("exit_code", ""), "seconds": t.get("seconds", ""), "best_epoch": t.get("best_epoch", ""),
             "valid_click_auc": t.get("valid_click_auc", ""), "valid_conv_auc": t.get("valid_conv_auc", ""),
             "valid_avg_auc": t.get("valid_avg_auc", ""), "valid_click_logloss": t.get("valid_click_logloss", ""),
             "valid_conv_logloss": t.get("valid_conv_logloss", ""),
             "valid_eval_click_auc": v.get("click_auc", ""), "valid_eval_conv_auc": v.get("conv_auc", ""),
             "valid_cvr_clicked_auc": v.get("cvr_clicked_auc", ""),
             "valid_cvr_clicked_auc_ratio": v.get("cvr_clicked_auc_ratio", ""),
             "valid_cvr_clicked_logloss": v.get("cvr_clicked_logloss", ""),
             "valid_n_click": v.get("n_click", ""), "valid_n_conv_in_clicked": v.get("n_conv_in_clicked", ""),
             "test_click_auc": s.get("click_auc", ""), "test_conv_auc": s.get("conv_auc", ""),
             "test_avg_auc": s.get("avg_auc", ""), "test_cvr_clicked_auc": s.get("cvr_clicked_auc", ""),
             "test_cvr_clicked_auc_ratio": s.get("cvr_clicked_auc_ratio", ""),
             "test_click_logloss": s.get("click_logloss", ""), "test_conv_logloss": s.get("conv_logloss", ""),
             "test_cvr_clicked_logloss": s.get("cvr_clicked_logloss", ""),
             "params_total": v.get("params_total", s.get("params_total", "")),
             "params_non_embedding": v.get("params_non_embedding", s.get("params_non_embedding", "")),
             "in_frozen_set": int(e in frozen)}
        table.append(r)
    with open(out / "followup_summary.csv", "w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(table)

    by_stem = {}
    for r in table:
        stem, seed = split_expid(r["expid"])
        by_stem.setdefault(stem, {})[seed] = r

    def num(r, k):
        x = r.get(k, "")
        return float(x) if x not in ("", None) else None

    def stem_metric(stem, key, seeds=None):
        d = by_stem.get(stem, {})
        return [num(r, key) for s, r in sorted(d.items()) if (seeds is None or s in seeds) and num(r, key) is not None]

    L = ["# MT-RankMixer follow-up: loss normalisation, clicked-only CVR, PLE tuning, final test", ""]
    L.append("Generated by fu_report.py. Validation numbers select; test numbers are reported once and select nothing.")
    L.append("mean ± sample std over seeds; n = number of seeds with a result.")
    L.append("")
    L.append("## 1. Validation, all trained variants (restored best checkpoint, from training logs)")
    L.append("")
    L.append("| variant | n | click AUC | conv AUC (impression) | avg AUC | clicked-only CVR AUC | best epochs |")
    L.append("|---|---|---|---|---|---|---|")
    for stem, _n, _l in LABELS:
        if stem not in by_stem:
            continue
        c, n = mean_sd(stem_metric(stem, "valid_click_auc"))
        v, _ = mean_sd(stem_metric(stem, "valid_conv_auc"))
        a, _ = mean_sd(stem_metric(stem, "valid_avg_auc"))
        k, nk = mean_sd(stem_metric(stem, "valid_cvr_clicked_auc"))
        eps = ",".join(str(r["best_epoch"]) for _s, r in sorted(by_stem[stem].items()))
        L.append("| {} | {} | {} | {} | {} | {} (n={}) | {} |".format(display(stem), n, c, v, a, k, nk, eps))
    L.append("")
    L.append("## 2. Loss weighting vs EQ, paired by seed (validation)")
    L.append("")
    L.append("| comparison (A − EQ, same seed) | metric | mean diff ± sd | A wins | paired t | p |")
    L.append("|---|---|---|---|---|---|")
    for suffix, tag in (("norm", "NORM"), ("w10", "W[1,10]")):
        for var in ("g6_mean", "g6_gate", "g6_residual"):
            a = {s: r for s, r in by_stem.get("MTR_fu_{}_{}".format(var, suffix), {}).items()}
            if not a:
                continue
            b = {s: r for s, r in by_stem.get("MTR_rg_{}".format(var), {}).items()}
            for metric, key in (("click AUC", "valid_click_auc"), ("conv AUC", "valid_conv_auc"),
                                ("avg AUC", "valid_avg_auc"), ("clicked CVR AUC", "valid_cvr_clicked_auc")):
                aa = {s: {key: num(r, key)} for s, r in a.items() if num(r, key) is not None}
                bb = {s: {key: num(r, key)} for s, r in b.items() if num(r, key) is not None}
                L.append(fmt_paired("{} {} − EQ".format(var, tag), metric, paired(aa, bb, key)))
    L.append("")
    L.append("Two-sided paired t, df = n−1. n=5 is low power.")
    L.append("")
    L.append("## 3. PLE tuning (validation, seeds 2025–2027; selection rule: max mean avg AUC)")
    L.append("")
    L.append("| config | change vs baseline | non-emb params | n | click AUC | conv AUC | avg AUC | clicked CVR AUC | selected |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for stem, desc in PLE_CANDIDATES.items():
        if stem not in by_stem:
            continue
        c, n = mean_sd(stem_metric(stem, "valid_click_auc", TUNE_SEEDS))
        v, _ = mean_sd(stem_metric(stem, "valid_conv_auc", TUNE_SEEDS))
        a, _ = mean_sd(stem_metric(stem, "valid_avg_auc", TUNE_SEEDS))
        k, _ = mean_sd(stem_metric(stem, "valid_cvr_clicked_auc", TUNE_SEEDS))
        pn = params.get(stem, {}).get("non_embedding", "")
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            stem, desc, pn, n, c, v, a, k, "**yes**" if stem == best_ple else ""))
    if best_ple:
        a, n = mean_sd(stem_metric(best_ple, "valid_avg_auc"))
        L.append("")
        L.append("Selected PLE config: **{}**. With all seeds (incl. 2028–2029): avg AUC {} (n={}).".format(best_ple, a, n))
    L.append("")
    L.append("## 4. Clicked-only CVR AUC (validation, post-hoc eval of saved checkpoints)")
    L.append("")
    L.append("`conv AUC` = impression-level (CTCVR). `CVR AUC (clicked)` ranks clicked rows by the conversion head; "
             "`CVR AUC (p_conv/p_click)` ranks them by the implied pCVR. "
             "`|Δclick|` = max |eval click AUC − training-log click AUC| (reproducibility check).")
    L.append("")
    L.append("| variant | n | conv AUC (impr.) | CVR AUC (clicked) | CVR AUC (p_conv/p_click) | clicked rows | conversions in clicked | max abs Δclick |")
    L.append("|---|---|---|---|---|---|---|---|")
    for stem, _n, _l in LABELS:
        if stem not in by_stem:
            continue
        vals = stem_metric(stem, "valid_cvr_clicked_auc")
        if not vals:
            continue
        v, _ = mean_sd(stem_metric(stem, "valid_eval_conv_auc"))
        k, n = mean_sd(vals)
        kr, _ = mean_sd(stem_metric(stem, "valid_cvr_clicked_auc_ratio"))
        rows = list(by_stem[stem].values())
        nclick = rows[0].get("valid_n_click", "")
        nconv = rows[0].get("valid_n_conv_in_clicked", "")
        diffs = [abs(num(r, "valid_eval_click_auc") - num(r, "valid_click_auc")) for r in rows
                 if num(r, "valid_eval_click_auc") is not None and num(r, "valid_click_auc") is not None]
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            display(stem), n, v, k, kr, nclick, nconv, "{:.1e}".format(max(diffs)) if diffs else "n/a"))
    L.append("")
    L.append("## 5. Parameter counts and training budget")
    L.append("")
    L.append("| config | model | total params | embedding | non-embedding | emb dim | lr | batch | optimizer | emb reg | net reg | dropout | epochs | patience | loss |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for label, r in params.items():
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            label, r["model"], r["total"], r["embedding"], r["non_embedding"], r["embedding_dim"], r["learning_rate"],
            r["batch_size"], r["optimizer"], r["embedding_regularizer"], r["net_regularizer"], r["net_dropout"],
            r["epochs"], r["early_stop_patience"], r["loss_weight"]))
    L.append("")
    L.append("## 6. Gradient audit after loss re-weighting")
    L.append("")
    ga = out / "grad_audit_norm.md"
    L.append(ga.read_text() if ga.is_file() else "(missing)")
    ev = out.parent / "fu_evidence" / "NOTE.md"
    if ev.is_file():
        L.append("")
        L.append(ev.read_text())
    L.append("")
    L.append("## 7. Final test (read once, pre-registered frozen set, best-validation checkpoints, no retraining)")
    L.append("")
    if not test:
        L.append("(not run: see final_test/STATUS)")
    else:
        L.append("| variant | n | click AUC | conv AUC (impr.) | avg AUC | CVR AUC (clicked) | click logloss | conv logloss |")
        L.append("|---|---|---|---|---|---|---|---|")
        for stem, _n, _l in LABELS:
            vals = stem_metric(stem, "test_click_auc")
            if not vals:
                continue
            cells = [mean_sd(stem_metric(stem, k))[0] for k in
                     ("test_conv_auc", "test_avg_auc", "test_cvr_clicked_auc", "test_click_logloss", "test_conv_logloss")]
            c, n = mean_sd(vals)
            tag = " (selected PLE)" if stem == best_ple else ""
            L.append("| {}{} | {} | {} | {} |".format(display(stem), tag, n, c, " | ".join(cells)))
        L.append("")
        L.append("Pre-registered paired comparisons on test (reported, not used for any choice):")
        L.append("")
        L.append("| comparison | metric | mean diff ± sd | A wins | paired t | p |")
        L.append("|---|---|---|---|---|---|")
        comps = [("MTR_fu_{}_{}".format(v, suf), "MTR_rg_{}".format(v), "{} {} − EQ".format(v, tag))
                 for suf, tag in (("norm", "NORM"), ("w10", "W[1,10]"))
                 for v in ("g6_mean", "g6_gate", "g6_residual") if "MTR_fu_{}_{}".format(v, suf) in by_stem]
        if best_ple:
            comps += [("MTR_rg_g6_mean", best_ple, "g6_mean[EQ] − {}".format(best_ple)),
                      ("MTR_rg_g6_mean", "MTR_rg_g3_mean", "g6_mean − g3_mean [EQ]")]
        for sa, sb, label in comps:
            for metric, key in (("avg AUC", "test_avg_auc"), ("clicked CVR AUC", "test_cvr_clicked_auc")):
                aa = {s: {key: num(r, key)} for s, r in by_stem.get(sa, {}).items() if num(r, key) is not None}
                bb = {s: {key: num(r, key)} for s, r in by_stem.get(sb, {}).items() if num(r, key) is not None}
                L.append(fmt_paired(label, metric, paired(aa, bb, key)))
    L.append("")
    (out / "followup_results.md").write_text("\n".join(L))
    print("wrote {} and {}".format(out / "followup_summary.csv", out / "followup_results.md"))
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select-ple")
    p.add_argument("--logs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--logs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.exit(cmd_select_ple(args) if args.cmd == "select-ple" else cmd_summarize(args))


if __name__ == "__main__":
    main()
