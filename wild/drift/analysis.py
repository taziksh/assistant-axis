"""Analyze completed drift runs on CPU and plot all four conversation domains."""

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path

import numpy as np

from . import state as run_state

DOMAINS = ("coding", "writing", "therapy", "ai")
LABELS = {"coding": "Coding", "writing": "Writing", "therapy": "Emotional disclosure", "ai": "AI philosophy"}
COLORS = {"coding": "#0072B2", "writing": "#009E73", "therapy": "#E69F00", "ai": "#CC79A7"}
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0
MIN_PLOT_CONVERSATIONS = 10


def indexed_samples(rows):
    """Reject repeated keys rather than silently dropping sampled replies."""
    indexed = {}
    for row in rows:
        key = (row["hash"], row["k"], row["sample_index"])
        if key in indexed:
            raise ValueError(f"duplicate conversation/turn/sample: {key}")
        indexed[key] = row
    return indexed


def conversation_turns(conversations, generated, projections, config):
    """Average complete sets of reply samples before comparing conversations."""
    samples, horizon = config["samples_per_turn"], config["max_turns"]
    if type(samples) is not int or samples < 1 or type(horizon) is not int or horizon < 1:
        raise ValueError("positive integer sample count and turn horizon are required")
    gen, proj = indexed_samples(generated), indexed_samples(projections)
    if set(gen) != set(proj):
        raise ValueError("generation and projection sample keys must match")
    expected = {
        (c["hash"], k, sample)
        for c in conversations for k in range(1, min(len(c["users"]), horizon) + 1)
        for sample in range(samples)
    }
    if set(gen) - expected:
        raise ValueError("saved results contain an unexpected conversation, turn, or sample")
    turns = []
    for conversation in sorted(conversations, key=lambda c: c["hash"]):
        cid, domain = conversation["hash"], conversation["domain"]
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain: {domain}")
        baseline = None
        for k in range(1, min(len(conversation["users"]), horizon) + 1):
            keys = [(cid, k, index) for index in range(samples)]
            present = [key for key in keys if key in gen]
            if present and len(present) != samples:
                raise ValueError(f"incomplete sample set for {cid} turn {k}")
            values = [proj[key]["proj"] for key in present]
            finite = []
            for value in values:
                if value is not None:
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError(f"nonfinite projection for {cid} turn {k}")
                    finite.append(value)
            status = "not_generated" if not present else "no_content_tokens" if len(finite) < samples else "ok"
            mean = float(np.mean(finite)) if status == "ok" else None
            if k == 1:
                baseline = mean
            turns.append({
                "hash": cid, "domain": domain, "turn": k,
                "expected_samples": samples, "generated_samples": len(present),
                "valid_samples": len(finite),
                "truncated_samples": sum(bool(gen[key]["truncated"]) for key in present),
                "status": status, "projection": mean,
                "change_from_turn1": mean - baseline if mean is not None and baseline is not None else None,
            })
    return turns


def trajectory_summary(values, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Resample whole conversation rows, retaining their cross-turn dependence."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("trajectories must be a conversation-by-turn matrix")
    n_conversations, horizon = values.shape
    counts = np.isfinite(values).sum(axis=0)
    totals = np.nansum(values, axis=0)
    means = np.divide(totals, counts, out=np.full(horizon, np.nan), where=counts > 0)
    estimates = np.full((draws, horizon), np.nan)
    if n_conversations >= 2:
        rng = np.random.default_rng(seed)
        for index in range(draws):
            sampled = values[rng.integers(0, n_conversations, n_conversations)]
            n = np.isfinite(sampled).sum(axis=0)
            estimates[index] = np.divide(np.nansum(sampled, axis=0), n, out=np.full(horizon, np.nan), where=n > 0)
    result = []
    for k in range(horizon):
        valid_draws = estimates[:, k][np.isfinite(estimates[:, k])]
        interval = counts[k] >= 2 and len(valid_draws) > 0
        low, high = np.percentile(valid_draws, [2.5, 97.5]) if interval else (None, None)
        result.append({
            "turn": k + 1, "n_conversations": int(counts[k]),
            "plot_eligible": bool(counts[k] >= MIN_PLOT_CONVERSATIONS),
            "mean": float(means[k]) if counts[k] else None,
            "ci_low": float(low) if interval else None,
            "ci_high": float(high) if interval else None,
            "bootstrap_valid_draws": int(len(valid_draws)) if interval else 0,
        })
    return result


def summarize(conversations, turns, config):
    horizon = config["max_turns"]
    by_conversation = defaultdict(dict)
    for turn in turns:
        by_conversation[turn["hash"]][turn["turn"]] = turn
    summaries, cohorts = [], {}
    for domain in DOMAINS:
        ids = sorted(c["hash"] for c in conversations if c["domain"] == domain)
        values = np.full((len(ids), horizon), np.nan)
        changes = values.copy()
        for index, cid in enumerate(ids):
            for k, row in by_conversation[cid].items():
                if row["projection"] is not None:
                    values[index, k - 1] = row["projection"]
                if row["change_from_turn1"] is not None:
                    changes[index, k - 1] = row["change_from_turn1"]
        complete = np.isfinite(values).all(axis=1)
        cohorts[domain] = [cid for cid, keep in zip(ids, complete) if keep]
        for cohort, mask in (("available", np.ones(len(ids), dtype=bool)), ("fixed", complete)):
            for metric, matrix in (("projection", values), ("change", changes)):
                for result in trajectory_summary(matrix[mask]):
                    summaries.append({"domain": domain, "cohort": cohort, "metric": metric, **result})
    return summaries, cohorts


def write_csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    run_state.atomic_write(path, stream.getvalue().encode())


def plot_values(rows, field):
    """Mask means and bands by conversation count without changing raw estimates."""
    return [row[field] if row["n_conversations"] >= MIN_PLOT_CONVERSATIONS and row[field] is not None else np.nan for row in rows]


def render_plots(summaries, cohorts, horizon, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(domain, cohort, metric):
        return [r for r in summaries if r["domain"] == domain and r["cohort"] == cohort and r["metric"] == metric]

    def curve(ax, domain, cohort, metric, counts=False):
        rows = series(domain, cohort, metric)
        x = [r["turn"] for r in rows]
        y = [r["n_conversations"] for r in rows] if counts else plot_values(rows, "mean")
        label = LABELS[domain]
        if cohort == "fixed":
            label += f" (n={len(cohorts[domain])})"
            if len(cohorts[domain]) < MIN_PLOT_CONVERSATIONS:
                label += "; insufficient n"
        ax.plot(x, y, marker="o", markersize=3, color=COLORS[domain], label=label)
        if not counts:
            low = plot_values(rows, "ci_low")
            high = plot_values(rows, "ci_high")
            ax.fill_between(x, low, high, color=COLORS[domain], alpha=0.16)

    def insufficient(ax, text):
        ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", zorder=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95})

    def finish(fig, axes, name, note):
        for ax in np.atleast_1d(axes).flat:
            ax.set_xlabel("User turn")
            ax.set_xticks(range(1, horizon + 1))
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        handles, labels = np.atleast_1d(axes).flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.text(0.5, 0.02, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 0.85))
        for extension in ("png", "svg"):
            fig.savefig(out / f"{name}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    band_note = f"Points and bands require at least {MIN_PLOT_CONVERSATIONS} valid conversations per turn. Bands: 95% conversation bootstrap."
    for metric, name, ylabel in (
        ("projection", "projection_by_turn", "Assistant Axis projection"),
        ("change", "change_from_first_turn", "Change from turn 1"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for domain in DOMAINS:
            curve(ax, domain, "available", metric)
        if metric == "change":
            ax.axhline(0, color="gray", linewidth=0.7)
        if not any(r["n_conversations"] >= MIN_PLOT_CONVERSATIONS for domain in DOMAINS for r in series(domain, "available", metric)):
            insufficient(ax, f"Insufficient n: fewer than {MIN_PLOT_CONVERSATIONS} valid conversations at every turn")
        ax.set_ylabel(ylabel)
        finish(fig, ax, name, band_note)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, metric, title in zip(axes, ("projection", "change"), ("Absolute projection", "Change from turn 1")):
        for domain in DOMAINS:
            curve(ax, domain, "available", metric, counts=True)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axes[0].set_ylabel("Contributing conversations")
    finish(fig, axes, "conversation_counts", "A conversation contributes only when all requested replies have a measurable projection.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, metric, ylabel in zip(axes, ("projection", "change"), ("Assistant Axis projection", "Change from turn 1")):
        for domain in DOMAINS:
            curve(ax, domain, "fixed", metric)
        ax.set_ylabel(ylabel)
        if metric == "change":
            ax.axhline(0, color="gray", linewidth=0.7)
        if all(len(ids) < MIN_PLOT_CONVERSATIONS for ids in cohorts.values()):
            insufficient(ax, f"Insufficient n: no category has {MIN_PLOT_CONVERSATIONS} complete\nconversations through turn {horizon}")
    finish(fig, axes, "fixed_cohort", f"Same conversations at every turn, complete through turn {horizon}.\n" + band_note)


def write_report(out, result):
    lines = [
        "# Four-category Assistant Axis projections", "",
        f"Run: `{result['run_id']}`. Label review status: `{result['review_status']}`.", "",
        "The category assignments are the saved selection labels; this analysis does not validate them or infer a safety outcome.", "",
        "Each response was regenerated from the original recorded conversation prefix. Later turns retain the original assistant history; these trajectories do not describe an autonomous Llama conversation built from its own earlier replies.", "",
        f"Each conversation/turn averages all {result['samples_per_turn']} requested replies before conversations receive equal weight. Turns with any unmeasurable reply are excluded from the primary mean. Changes subtract that conversation's turn-one mean; a missing turn-one mean excludes it from change estimates.", "",
        f"Confidence intervals use {BOOTSTRAP_DRAWS:,} bootstrap resamples of whole conversations within each category, with seed {BOOTSTRAP_SEED}. The same resampled conversation IDs are used across turns. These are pointwise intervals, not a simultaneous test of the full trajectory. Fewer than two contributing conversations gives no interval.", "",
        f"Projection and change figures omit points and bands with fewer than {MIN_PLOT_CONVERSATIONS} valid conversations, after averaging replies within each conversation/turn. Three replies from one conversation still count as one conversation. Counts and raw table estimates remain available below this plotting threshold. Matching the paper's conversation count is not a power calculation or evidence of a safety effect.", "",
        f"The fixed cohort contains conversations with every turn measurable through turn {result['max_turns']}, the frozen selection horizon. Categories with fewer than {MIN_PLOT_CONVERSATIONS} complete conversations are marked insufficient n; the cutoff is not shortened to obtain a result.", "",
        "| Category | Selected | Fixed cohort | Fixed cohort status | Unmeasurable turns | Ungenerated turns | Truncated replies |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        counts = result["counts"][domain]
        fixed_n = len(result["fixed_cohort_ids"][domain])
        fixed_status = "plotted" if fixed_n >= MIN_PLOT_CONVERSATIONS else "insufficient n"
        lines.append(f"| {LABELS[domain]} | {counts['selected_conversations']} | {fixed_n} | {fixed_status} | {counts['unmeasurable_turns']} | {counts['ungenerated_turns']} | {counts['truncated_replies']} |")
    lines.extend([
        "", "Generated replies that reach the length limit remain in the primary analysis and are counted above. Later turns can have a different mix of conversations; compare the available-conversation and fixed-cohort figures before interpreting an apparent trend.", "",
        "The figures are `projection_by_turn`, `change_from_first_turn`, `conversation_counts`, and `fixed_cohort`, each in PNG and SVG. `conversation_turns.csv` contains the averaged measurements and exclusions; `summary.csv` contains all raw means, intervals, counts, and a `plot_eligible` flag, including estimates omitted from figures.", "",
    ])
    run_state.atomic_write(out / "REPORT.md", "\n".join(lines).encode())


def run_analysis(run_dir):
    run_dir = Path(run_dir)
    with run_state.run_lock(run_dir):
        conversations, run, generated, projections = run_state.load_completed_run(run_dir)
        sidecar = json.loads((run_dir / "selected.manifest.json").read_bytes())
        config = run["config"]
        turns = conversation_turns(conversations, generated, projections, config)
        summaries, cohorts = summarize(conversations, turns, config)
        counts = {}
        for domain in DOMAINS:
            domain_turns = [row for row in turns if row["domain"] == domain]
            statuses = Counter(row["status"] for row in domain_turns)
            counts[domain] = {
                "selected_conversations": sum(c["domain"] == domain for c in conversations),
                "measurable_turns": statuses["ok"], "unmeasurable_turns": statuses["no_content_tokens"],
                "ungenerated_turns": statuses["not_generated"],
                "generated_replies": sum(row["generated_samples"] for row in domain_turns),
                "truncated_replies": sum(row["truncated_samples"] for row in domain_turns),
            }
        result = {
            "run_id": run["run_id"], "review_status": sidecar.get("review_status", "unreviewed"),
            "samples_per_turn": config["samples_per_turn"], "max_turns": config["max_turns"],
            "model": config["model"], "layer": config["layer"],
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "conversation", "interval": "pointwise percentile 95%"},
            "plot_policy": {"min_valid_conversations": MIN_PLOT_CONVERSATIONS, "unit": "conversation", "raw_estimates_retained": True},
            "counts": counts, "fixed_cohort_ids": cohorts, "summary": summaries,
            "projections_sha256": run_state.checksum((run_dir / "projections.jsonl").read_bytes()),
            "analysis_code_sha256": run_state.checksum(Path(__file__).read_bytes()),
        }
        out = run_dir / "analysis"
        out.mkdir(exist_ok=True)
        write_csv(out / "conversation_turns.csv", turns, [
            "hash", "domain", "turn", "expected_samples", "generated_samples", "valid_samples",
            "truncated_samples", "status", "projection", "change_from_turn1",
        ])
        write_csv(out / "summary.csv", summaries, [
            "domain", "cohort", "metric", "turn", "n_conversations", "plot_eligible", "mean", "ci_low", "ci_high", "bootstrap_valid_draws",
        ])
        render_plots(summaries, cohorts, config["max_turns"], out)
        write_report(out, result)
        run_state.atomic_write(out / "results.json", run_state.encoded(result))
        print(f"drift analysis: wrote four-category summaries and figures to {out}")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    run_analysis(args.run_dir)


if __name__ == "__main__":
    main()
