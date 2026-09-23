#!/usr/bin/env python3
"""Turn bench_serving output into the Org table for README.org.

    scripts/bench-report.py /scratch/$PROJECT/$USER/sglang/bench

Expects one subdirectory per engine (sglang, vllm, optionally vllm-eager), each
holding c<N>.jsonl files written by scripts/bench-{sglang,vllm}.sh.

bench_serving appends to its --output-file, so running the sweep several times
into the same directory accumulates one line per run. Every line is read and the
median reported, with the spread printed underneath: a single serving run is not
stable enough to quote on its own, particularly for vLLM.
"""

import json
import statistics
import sys
from pathlib import Path

# Directory name -> column label. Order here is the column order in the table.
ENGINES = [("sglang", "SGLang"), ("vllm", "vLLM"), ("vllm-eager", "vLLM eager")]

METRICS = [
    ("output_throughput", "{} tok/s", "{:.0f}"),
    ("median_ttft_ms", "{} TTFT ms", "{:.0f}"),
    ("median_itl_ms", "{} ITL ms", "{:.1f}"),
]


def load_all(path):
    """Every JSON object in a JSONL file, oldest first."""
    out = []
    try:
        for line in path.read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as e:
        print(f"warning: {path}: {e}", file=sys.stderr)
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    root = Path(sys.argv[1])
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    # {concurrency: {engine: [run, ...]}}
    runs = {}
    present = []
    for name, label in ENGINES:
        files = sorted((root / name).glob("c*.jsonl"))
        if not files:
            continue
        present.append((name, label))
        for f in files:
            records = load_all(f)
            if not records:
                continue
            conc = records[-1].get("max_concurrency") or int(f.stem[1:])
            runs.setdefault(conc, {})[name] = records

    if not runs:
        sys.exit(f"no c*.jsonl results under {root}")

    # A point whose runs used different prompt counts is not a repeated
    # measurement of the same thing, and taking a median across them would
    # silently average two different workloads.
    for conc, by_engine in sorted(runs.items()):
        for name, records in by_engine.items():
            counts = {r.get("completed") for r in records}
            if len(counts) > 1:
                print(
                    f"warning: {name} c{conc} mixes prompt counts {sorted(counts)}; "
                    "clear the directory and re-run",
                    file=sys.stderr,
                )

    sample = next(iter(next(iter(runs.values())).values()))[-1]
    counts = [
        r["completed"] for by_e in runs.values() for recs in by_e.values() for r in recs
    ]
    span = f"{min(counts)}" if min(counts) == max(counts) else f"{min(counts)}-{max(counts)}"
    # Engines can end up with different run counts (an extra pass adds samples
    # to one and not the others), so report the range rather than the maximum,
    # which would overstate how well-sampled the thinnest column is.
    lens = [len(recs) for by_e in runs.values() for recs in by_e.values()]
    nruns = f"{min(lens)}" if min(lens) == max(lens) else f"{min(lens)}-{max(lens)}"

    # GCD count: vLLM records no tp field, so prefer the tag bench_once sets and
    # fall back to SGLang's server_info for results captured before the tag
    # existed. Hardcoding "one GCD" silently mislabels every multi-GCD table.
    gcds = None
    for by_e in runs.values():
        for recs in by_e.values():
            for r in recs:
                tag = r.get("tag") or ""
                if tag.startswith("tp") and tag[2:].isdigit():
                    gcds = int(tag[2:])
                elif gcds is None:
                    gcds = (r.get("server_info") or {}).get("tp_size")
    hw = f"{gcds} MI250x GCD{'s' if gcds and gcds > 1 else ''}" if gcds else "MI250x"

    print("* Benchmark\n")
    print(
        f"{hw}, Llama-3.1-8B-Instruct, "
        f"{sample.get('random_input_len')} in / {sample.get('random_output_len')} out, "
        f"{span} prompts scaled with concurrency, "
        f"median of {nruns} runs:\n"
    )

    headers = ["concurrency"]
    for _, label in present:
        headers += [m[1].format(label) for m in METRICS]

    rows = []
    for conc in sorted(runs):
        row = [str(conc)]
        for name, _ in present:
            records = runs[conc].get(name)
            for key, _, fmt in METRICS:
                vals = [r[key] for r in (records or []) if r.get(key) is not None]
                # A level the engine could not complete stays in the table as
                # "-" rather than vanishing: failing at concurrency 64 is itself
                # the result, and a dropped row reads as though it was never run.
                row.append(fmt.format(statistics.median(vals)) if vals else "-")
        rows.append(row)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print("|" + "+".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print("| " + " | ".join(c.rjust(w) for c, w in zip(row, widths)) + " |")

    # Run-to-run spread on throughput, so the reader knows how much of a
    # difference between the columns above is real.
    print("\nThroughput spread across runs (max deviation from median):")
    for name, label in present:
        worst = 0.0
        for conc in sorted(runs):
            vals = [
                r["output_throughput"]
                for r in runs[conc].get(name, [])
                if r.get("output_throughput") is not None
            ]
            if len(vals) > 1:
                med = statistics.median(vals)
                worst = max(worst, max(abs(v - med) / med for v in vals) * 100)
        print(f"  {label}: {worst:.1f}%" if worst else f"  {label}: single run")


if __name__ == "__main__":
    main()
