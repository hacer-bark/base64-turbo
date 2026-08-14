#!/usr/bin/env python3
import argparse
import math
import re
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

BENCH_RE = re.compile(
    r"[A-Za-z0-9_]+_Performances/(Encode|Decode)/([A-Za-z0-9_]+)/(\d+)"
)
THRPT_RE = re.compile(
    r"thrpt:\s*\[[\d.]+\s*[KMG]iB/s\s+([\d.]+)\s*([KMG]iB)/s\s+[\d.]+\s*[KMG]iB/s"
)

UNIT_TO_MIB = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0}

LIBRARY_ORDER = ["Turbo", "TurboBuff", "Simd", "Std"]
LIBRARY_LABEL = {
    "Turbo": "base64-turbo",
    "TurboBuff": "base64-turbo (zero-alloc)",
    "Simd": "base64-simd",
    "Std": "base64",
}
COLORS = {
    "Turbo": "#2a78d6",
    "TurboBuff": "#1baf7a",
    "Simd": "#eb6834",
    "Std": "#eda100",
}

THEME = {
    "paper": "rgba(0,0,0,0)",
    "plot": "rgba(0,0,0,0)",
    "ink": "#767671",
    "muted": "#8a8983",
    "grid": "rgba(137,135,129,0.35)",
}

NICE_MULTIPLES = [1, 2, 2.5, 5, 7.5, 10]


def nice_axis(max_value, target_ticks=6):
    """Round (step, top) to human-friendly numbers (…10, 25, 50, 75, 100…)."""
    if max_value <= 0:
        return 10, 10
    raw_step = max_value / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    step = next(
        (m * magnitude for m in NICE_MULTIPLES if residual <= m),
        10 * magnitude,
    )
    top = math.ceil(max_value / step) * step
    return step, top


def format_size(n):
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n // 1024} KB"
    return f"{n // (1024 * 1024)} MB"


def parse(text):
    """-> {phase: {size: {library: mib_per_s}}}"""
    data = {"Encode": {}, "Decode": {}}
    pending = None
    for line in text.splitlines():
        m = BENCH_RE.search(line)
        if m:
            pending = (m.group(1), m.group(2), int(m.group(3)))
            continue
        m = THRPT_RE.search(line)
        if m and pending:
            phase, library, size = pending
            value = float(m.group(1)) * UNIT_TO_MIB[m.group(2)]
            data[phase].setdefault(size, {})[library] = value
            pending = None
    return data


def render(data, out_path):
    theme = THEME

    sizes = sorted({s for phase in data.values() for s in phase})
    size_labels = [format_size(s) for s in sizes]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Encode", "Decode"),
        vertical_spacing=0.16,
    )

    bar_w = 0.9 / len(LIBRARY_ORDER)
    half_max = len(LIBRARY_ORDER) * bar_w / 2
    x_pad = 0.03

    for row, phase in enumerate(("Encode", "Decode"), start=1):
        series = {lib: {"x": [], "y": [], "text": []} for lib in LIBRARY_ORDER}
        for xi, size in enumerate(sizes):
            present = [lib for lib in LIBRARY_ORDER if lib in data[phase].get(size, {})]
            start = xi - (len(present) * bar_w) / 2
            # Label the fastest bar in each group, whichever library that
            # happens to be, so the annotation always marks the actual winner.
            winner = max(present, key=lambda lib: data[phase][size][lib])
            for j, lib in enumerate(present):
                value = data[phase][size][lib] / 1024  # MiB/s -> GiB/s
                series[lib]["x"].append(start + bar_w * (j + 0.5))
                series[lib]["y"].append(value)
                series[lib]["text"].append(f"{value:.1f}" if lib == winner else "")

        phase_max = max(v for s in series.values() for v in s["y"])
        _, y_top = nice_axis(phase_max)

        for lib in LIBRARY_ORDER:
            s = series[lib]
            if not s["x"]:
                continue
            fig.add_trace(
                go.Bar(
                    x=s["x"],
                    y=s["y"],
                    width=bar_w * 0.9,
                    name=LIBRARY_LABEL[lib],
                    marker_color=COLORS[lib],
                    text=s["text"],
                    textposition="outside",
                    textfont=dict(size=12, color=theme["muted"]),
                    legendgroup=lib,
                    showlegend=(row == 1),
                ),
                row=row,
                col=1,
            )

        fig.update_yaxes(
            range=[0, y_top * 1.12],
            title=dict(text="GiB/s", font=dict(color=theme["muted"])),
            gridcolor=theme["grid"],
            zerolinecolor=theme["grid"],
            tickfont=dict(color=theme["muted"]),
            automargin=True,
            row=row,
            col=1,
        )
        fig.update_xaxes(
            tickvals=list(range(len(sizes))),
            ticktext=size_labels,
            range=[-half_max - x_pad, len(sizes) - 1 + half_max + x_pad],
            title=dict(
                text="Payload size" if row == 2 else "",
                font=dict(color=theme["muted"]),
            ),
            tickfont=dict(color=theme["muted"]),
            automargin=True,
            row=row,
            col=1,
        )

    fig.update_layout(
        barmode="overlay",
        width=1150,
        height=900,
        paper_bgcolor=theme["paper"],
        plot_bgcolor=theme["plot"],
        font=dict(family="Arial, Helvetica, sans-serif", color=theme["ink"], size=15),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
        ),
        margin=dict(l=10, r=10, t=110, b=10, autoexpand=True),
    )

    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = dict(size=16, color=theme["ink"])

    fig.write_image(out_path, scale=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, help="raw `cargo bench` output (default: stdin)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "throughput.png",
    )
    args = parser.parse_args()

    text = args.input.read_text() if args.input else sys.stdin.read()
    data = parse(text)
    if not data["Encode"] and not data["Decode"]:
        sys.exit("No benchmark lines found in input.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    render(data, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
