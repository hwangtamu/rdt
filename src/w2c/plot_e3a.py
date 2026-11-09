"""Minimal dependency-free SVG plots for the E3-A P2 analysis."""

from __future__ import annotations

import argparse
import csv
import os


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_curves(path):
    return read_rows(path)


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        return [lo]
    step = (hi - lo) / max(1, n - 1)
    return [lo + i * step for i in range(n)]


class Plot:
    def __init__(self, title, xlabel, ylabel, width=820, height=520,
                 xlim=None, ylim=None):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.width, self.height = width, height
        self.left, self.right, self.top, self.bottom = 90, 30, 55, 70
        self.xlim = xlim
        self.ylim = ylim
        self.elements = []

    def _map(self, x, y):
        l, r, t, b = self.left, self.right, self.top, self.bottom
        x0, x1 = self.xlim
        y0, y1 = self.ylim
        px = l + (x - x0) / (x1 - x0) * (self.width - l - r)
        py = self.height - b - (y - y0) / (y1 - y0) * (self.height - t - b)
        return px, py

    def line(self, x0, y0, x1, y1, color="#333", width=2, dash=None):
        a = self._map(x0, y0); b = self._map(x1, y1)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(
            f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" '
            f'stroke="{color}" stroke-width="{width}"{d}/>')

    def polyline(self, points, color="#1f5fbf", width=2, dash=None):
        pts = " ".join(f"{self._map(x, y)[0]:.2f},{self._map(x, y)[1]:.2f}"
                       for x, y in points)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{width}"{d}/>')

    def errorbar(self, x, y, lo, hi, color="#1f5fbf", cap=5, width=2):
        px, py = self._map(x, y)
        _, plo = self._map(x, lo)
        _, phi = self._map(x, hi)
        self.elements.append(
            f'<line x1="{px:.2f}" y1="{plo:.2f}" x2="{px:.2f}" y2="{phi:.2f}" '
            f'stroke="{color}" stroke-width="{width}"/>')
        for yy in (plo, phi):
            self.elements.append(
                f'<line x1="{px-cap:.2f}" y1="{yy:.2f}" x2="{px+cap:.2f}" y2="{yy:.2f}" '
                f'stroke="{color}" stroke-width="{width}"/>')

    def marker(self, x, y, color="#1f5fbf", kind="circle", size=4):
        px, py = self._map(x, y)
        if kind == "circle":
            self.elements.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{size}" fill="{color}"/>')
        else:
            self.elements.append(
                f'<polygon points="{px:.2f},{py-size:.2f} {px-size:.2f},{py+size:.2f} '
                f'{px+size:.2f},{py+size:.2f}" fill="{color}"/>')

    def text(self, x, y, s, size=13, anchor="start", color="#222", weight="normal"):
        px, py = self._map(x, y)
        self.elements.append(
            f'<text x="{px:.2f}" y="{py:.2f}" font-family="sans-serif" '
            f'font-size="{size}" text-anchor="{anchor}" fill="{color}" '
            f'font-weight="{weight}">{s}</text>')

    def save(self, path):
        l, r, t, b = self.left, self.right, self.top, self.bottom
        x0, x1 = self.xlim
        y0, y1 = self.ylim
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{self.width/2}" y="30" font-family="sans-serif" '
            f'font-size="18" text-anchor="middle" font-weight="bold">{self.title}</text>',
            f'<line x1="{l}" y1="{self.height-b}" x2="{self.width-r}" '
            f'y2="{self.height-b}" stroke="#333" stroke-width="1.5"/>',
            f'<line x1="{l}" y1="{t}" x2="{l}" y2="{self.height-b}" '
            f'stroke="#333" stroke-width="1.5"/>',
        ]
        for x in nice_ticks(x0, x1, 6):
            px, _ = self._map(x, y0)
            parts.append(f'<line x1="{px:.2f}" y1="{self.height-b}" x2="{px:.2f}" '
                         f'y2="{self.height-b+6}" stroke="#333"/>')
            parts.append(f'<text x="{px:.2f}" y="{self.height-b+22}" '
                         f'font-family="sans-serif" font-size="12" text-anchor="middle">'
                         f'{x:.2f}</text>')
        for y in nice_ticks(y0, y1, 6):
            _, py = self._map(x0, y)
            parts.append(f'<line x1="{l-6}" y1="{py:.2f}" x2="{l}" y2="{py:.2f}" stroke="#333"/>')
            parts.append(f'<text x="{l-10}" y="{py+4:.2f}" font-family="sans-serif" '
                         f'font-size="12" text-anchor="end">{y:.2f}</text>')
        parts.extend(self.elements)
        parts.append(f'<text x="{self.width/2}" y="{self.height-20}" '
                     f'font-family="sans-serif" font-size="14" text-anchor="middle">'
                     f'{self.xlabel}</text>')
        parts.append(f'<text x="22" y="{self.height/2}" font-family="sans-serif" '
                     f'font-size="14" text-anchor="middle" '
                     f'transform="rotate(-90 22 {self.height/2})">{self.ylabel}</text>')
        parts.append('</svg>')
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(parts))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default="runs/e3a_boolq_p2/analysis")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_rows(os.path.join(args.analysis_dir, "matched_expected.csv"))
    curves = read_curves(os.path.join(args.analysis_dir, "curves.csv"))
    if not rows:
        raise SystemExit("no matched rows")

    # Deterministic means.
    det = {"encoder": [], "decoder": []}
    seen = set()
    for r in curves:
        key = (r["backbone"], r["e2_seed"])
        if key in seen or r["det_ce_bits"] == "":
            continue
        seen.add(key)
        det[r["backbone"]].append(float(r["det_ce_bits"]))
    det_mean = {k: sum(v) / len(v) for k, v in det.items()}

    # Absolute RD.
    all_ce = [float(r["enc"]) for r in rows] + [float(r["dec"]) for r in rows]
    all_ce += [float(r["enc_lo"]) for r in rows] + [float(r["enc_hi"]) for r in rows]
    all_ce += [float(r["dec_lo"]) for r in rows] + [float(r["dec_hi"]) for r in rows]
    p = Plot("E3-A: absolute rate-distortion", "validation KL rate (bits/example)",
             "test CE (bits)", xlim=(0, 8.8),
             ylim=(min(all_ce) - 0.02, max(all_ce) + 0.02))
    p.line(0, det_mean["encoder"], 8.8, det_mean["encoder"], color="#1f5fbf", width=1.5, dash="6,5")
    p.line(0, det_mean["decoder"], 8.8, det_mean["decoder"], color="#c23b22", width=1.5, dash="6,5")
    for r in rows:
        x = float(r["target"])
        p.errorbar(x, float(r["enc"]), float(r["enc_lo"]), float(r["enc_hi"]), color="#1f5fbf")
        p.errorbar(x, float(r["dec"]), float(r["dec_lo"]), float(r["dec_hi"]), color="#c23b22")
    p.polyline([(float(r["target"]), float(r["enc"])) for r in rows], color="#1f5fbf")
    p.polyline([(float(r["target"]), float(r["dec"])) for r in rows], color="#c23b22")
    for r in rows:
        p.marker(float(r["target"]), float(r["enc"]), color="#1f5fbf", kind="circle")
        p.marker(float(r["target"]), float(r["dec"]), color="#c23b22", kind="triangle")
    p.save(os.path.join(args.out_dir, "e3a_absolute_rd.svg"))

    # Delta D.
    lo = min(float(r["delta_lo"]) for r in rows)
    hi = max(float(r["delta_hi"]) for r in rows)
    p = Plot("E3-A: matched-rate delta D", "validation KL rate (bits/example)",
             "decoder - encoder CE (bits)", xlim=(0, 8.8), ylim=(min(0, lo) - 0.02, hi + 0.02))
    p.line(0, 0, 8.8, 0, color="#555", width=1, dash="4,4")
    p.polyline([(float(r["target"]), float(r["delta"])) for r in rows], color="#4a3fbf")
    for r in rows:
        x = float(r["target"])
        p.errorbar(x, float(r["delta"]), float(r["delta_lo"]), float(r["delta_hi"]), color="#4a3fbf")
        p.marker(x, float(r["delta"]), color="#4a3fbf", kind="circle")
    p.save(os.path.join(args.out_dir, "e3a_delta.svg"))

    # Penalty delta.
    lo = min(float(r["dpen_lo"]) for r in rows)
    hi = max(float(r["dpen_hi"]) for r in rows)
    p = Plot("E3-A: compression-penalty delta", "validation KL rate (bits/example)",
             "decoder penalty - encoder penalty", xlim=(0, 8.8),
             ylim=(min(0, lo) - 0.02, max(0, hi) + 0.02))
    p.line(0, 0, 8.8, 0, color="#555", width=1, dash="4,4")
    p.polyline([(float(r["target"]), float(r["dpen"])) for r in rows], color="#2b7a4b")
    for r in rows:
        x = float(r["target"])
        p.errorbar(x, float(r["dpen"]), float(r["dpen_lo"]), float(r["dpen_hi"]), color="#2b7a4b")
        p.marker(x, float(r["dpen"]), color="#2b7a4b", kind="circle")
    p.save(os.path.join(args.out_dir, "e3a_penalty.svg"))
    print(f"wrote SVGs to {args.out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
