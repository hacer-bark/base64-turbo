#!/usr/bin/env python3
"""Turn raw `cargo bench` output into a throughput-vs-size chart.

One line per implementation, two panels (encode, decode).

The throughput axis is linear with a constant step and starts at zero, so equal
vertical distance always means the same number of GiB/s -- an implementation
that sits near the floor is drawn near the floor. Both panels share one axis.
Payload size is log-spaced because the sweep doubles; that is the x axis only.
Nothing is normalised, clipped or rescaled per series.
"""
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

UNIT_TO_GIB = {"KiB": 1 / 1048576, "MiB": 1 / 1024, "GiB": 1.0}

LIBRARY_ORDER = ["Turbo", "Tb64", "Simd", "Ng", "Std"]
LIBRARY_LABEL = {
    "Turbo": "base64-turbo",
    "Tb64": "Turbo-Base64 (C)",
    "Simd": "base64-simd",
    "Ng": "base64-ng",
    "Std": "base64",
}
# Short forms for the direct labels, where the full crate names do not fit.
LIBRARY_TAG = {
    "Turbo": "turbo",
    "Tb64": "tb64",
    "Simd": "simd",
    "Ng": "ng",
    "Std": "std",
}

# Memcpy is not a codec -- it is the copy roof for the same buffers, drawn as a
# dashed neutral reference so the codec lines can be read against it.
CONTROL = "Memcpy"
CONTROL_LABEL = "memcpy (roof)"
CONTROL_TAG = "memcpy"
CONTROL_COLOR = "#8a8983"

# Validated against the data-viz six checks on both the light (#fcfcfb) and the
# dark (#1a1a19) surface, since the PNG is transparent and gets read on either:
# every slot clears 3:1 contrast on both, worst all-pairs normal-vision dE 16.3,
# worst all-pairs CVD dE 6.9. That CVD figure sits in the 6-8 relief band, so
# secondary encoding is mandatory and non-optional here: distinct marker symbols
# plus a direct label on every line.
COLORS = {
    "Turbo": "#256abf",  # blue
    "Tb64": "#9085e9",  # violet
    "Simd": "#d55181",  # magenta
    "Ng": "#008300",  # green
    "Std": "#c98500",  # yellow
}
SYMBOLS = {
    "Turbo": "circle",
    "Tb64": "square",
    "Simd": "diamond",
    "Ng": "triangle-up",
    "Std": "x",
}

THEME = {
    "paper": "rgba(0,0,0,0)",
    "plot": "rgba(0,0,0,0)",
    "ink": "#767671",
    "muted": "#8a8983",
    "grid": "rgba(137,135,129,0.22)",
}

NICE_MULTIPLES = [1, 2, 2.5, 5, 7.5, 10]


def nice_axis(max_value, target_ticks=8):
    """Round (step, top) to human-friendly numbers (...10, 25, 50, 75, 100...)."""
    if max_value <= 0:
        return 10, 10
    raw_step = max_value / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    residual = raw_step / magnitude
    step = next(
        (m * magnitude for m in NICE_MULTIPLES if residual <= m),
        10 * magnitude,
    )
    return step, math.ceil(max_value / step) * step


def format_size(n):
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n // 1024} KB"
    return f"{n // (1024 * 1024)} MB"


def format_rate(v):
    return f"{v:.2f}" if v < 1 else f"{v:.1f}"


def parse(text):
    """-> {phase: {size: {library: gib_per_s}}}"""
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
            value = float(m.group(1)) * UNIT_TO_GIB[m.group(2)]
            data[phase].setdefault(size, {})[library] = value
            pending = None
    return data



def place_label(value, others, band, y_top):
    """Put the label above or below its point, whichever side is less crowded.

    `others` are the other measurements at the same size; a label wants clear
    air between itself and any of them, and must stay inside the axis.
    """

    def clearance(side):
        y = value + side * band
        if not (band <= y <= y_top - band * 0.5):
            return -1.0
        return min((abs(y - o) for o in others), default=band)

    return max((1, -1), key=clearance)


def render(data, out_path, source=None):
    theme = THEME
    sizes = sorted({s for phase in data.values() for s in phase})
    xs = [math.log2(s) for s in sizes]
    x_lo, x_hi = xs[0], xs[-1]
    x_pad = (x_hi - x_lo) * 0.02

    # One linear GiB/s axis with a constant step, shared by both panels: equal
    # vertical distance always means the same number of GiB/s, and encode and
    # decode can be compared against each other directly.
    peak = max(v for phase in data.values() for row in phase.values() for v in row.values())
    y_step, y_top = nice_axis(peak)

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Encode", "Decode"),
        vertical_spacing=0.13,
    )

    for row, phase in enumerate(("Encode", "Decode"), start=1):
        order = [CONTROL] + LIBRARY_ORDER
        points = {
            lib: [
                (math.log2(s), data[phase][s][lib])
                for s in sizes
                if lib in data[phase].get(s, {})
            ]
            for lib in order
        }
        points = {lib: p for lib, p in points.items() if p}

        for lib, pts in points.items():
            control = lib == CONTROL
            fig.add_trace(
                go.Scatter(
                    x=[x for x, _ in pts],
                    y=[v for _, v in pts],
                    mode="lines" if control else "lines+markers",
                    name=CONTROL_LABEL if control else LIBRARY_LABEL[lib],
                    legendgroup=lib,
                    showlegend=(row == 1),
                    line=dict(
                        color=CONTROL_COLOR if control else COLORS[lib],
                        width=2,
                        dash="dash" if control else "solid",
                    ),
                    marker=dict(
                        size=8,
                        symbol=SYMBOLS.get(lib, "circle"),
                        color=COLORS.get(lib, CONTROL_COLOR),
                    ),
                ),
                row=row,
                col=1,
            )

        # Label the fastest codec at each size, whichever it happens to be --
        # the lead changes hands across the sweep. memcpy is excluded: it is the
        # roof, not a competitor. Doubles as the secondary encoding the CVD band
        # requires, so identity never rests on colour alone.
        for size in sizes:
            row_data = {
                lib: v for lib, v in data[phase].get(size, {}).items() if lib in COLORS
            }
            if not row_data:
                continue
            lib = max(row_data, key=row_data.get)
            value = row_data[lib]
            others = [v for k, v in data[phase][size].items() if k != lib]
            side = place_label(value, others, y_top * 0.055, y_top)
            fig.add_annotation(
                x=math.log2(size),
                y=value,
                text=f"{LIBRARY_TAG[lib]}&nbsp;&nbsp;<b>{format_rate(value)}</b>",
                showarrow=False,
                # Anchor the end labels inwards so they cannot clip the frame.
                xanchor={sizes[0]: "left", sizes[-1]: "right"}.get(size, "center"),
                yanchor="bottom" if side > 0 else "top",
                yshift=side * 9,
                font=dict(size=12, color=COLORS[lib]),
                row=row,
                col=1,
            )

        fig.update_yaxes(
            range=[0, y_top],
            dtick=y_step,
            title=dict(text="GiB/s", font=dict(color=theme["muted"], size=13)),
            gridcolor=theme["grid"],
            zeroline=False,
            showline=False,
            tickfont=dict(color=theme["muted"], size=12),
            automargin=True,
            row=row,
            col=1,
        )
        fig.update_xaxes(
            tickvals=xs,
            ticktext=[format_size(s) for s in sizes],
            range=[x_lo - x_pad, x_hi + x_pad],
            title=dict(
                text="Payload size (log scale)" if row == 2 else "",
                font=dict(color=theme["muted"], size=13),
            ),
            gridcolor=theme["grid"],
            zeroline=False,
            showline=False,
            tickfont=dict(color=theme["muted"], size=12),
            automargin=True,
            row=row,
            col=1,
        )

    caption = (
        f"Criterion median throughput, higher is better. Linear GiB/s axis, "
        f"constant {y_step:g} GiB/s per step, shared by both panels; no per-series scaling."
    )
    if source:
        caption += f"  ·  {source}"
    fig.add_annotation(
        text=caption,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.085,
        showarrow=False,
        xanchor="left",
        font=dict(size=12, color=theme["muted"]),
    )

    fig.update_layout(
        width=1180,
        height=950,
        paper_bgcolor=theme["paper"],
        plot_bgcolor=theme["plot"],
        font=dict(family="Arial, Helvetica, sans-serif", color=theme["ink"], size=14),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.10,
            xanchor="center",
            x=0.5,
            font=dict(size=13),
            itemsizing="constant",
        ),
        margin=dict(l=20, r=45, t=105, b=75, autoexpand=True),
    )

    for annotation in fig["layout"]["annotations"][:2]:
        annotation["font"] = dict(size=17, color=theme["ink"])
        annotation["x"] = 0
        annotation["xanchor"] = "left"

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
    render(data, args.out, source=args.input.name if args.input else None)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
