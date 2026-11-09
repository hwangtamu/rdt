"""Small dependency-free SVG plotting helpers for the RDT figures."""

from __future__ import annotations


def nice_ticks(lo, hi, n=5):
    if hi <= lo:
        return [lo]
    step = (hi - lo) / max(1, n - 1)
    return [lo + i * step for i in range(n)]


class Plot:
    def __init__(self, title, xlabel, ylabel, xlim, ylim,
                 width=900, height=560):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.width, self.height = width, height
        self.left, self.right, self.top, self.bottom = 90, 35, 55, 75
        self.xlim = xlim
        self.ylim = ylim
        self.xticks = None
        self.xtick_labels = None
        self.yticks = None
        self.elements = []

    def _map(self, x, y):
        l, r, t, b = self.left, self.right, self.top, self.bottom
        x0, x1 = self.xlim
        y0, y1 = self.ylim
        px = l + (x - x0) / (x1 - x0) * (self.width - l - r)
        py = self.height - b - (y - y0) / (y1 - y0) * (self.height - t - b)
        return px, py

    def line(self, x0, y0, x1, y1, color="#333", width=2, dash=None):
        a = self._map(x0, y0)
        b = self._map(x1, y1)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(
            f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" '
            f'stroke="{color}" stroke-width="{width}"{d}/>')

    def polyline(self, points, color="#1f5fbf", width=2, dash=None):
        if not points:
            return
        pts = " ".join(f"{self._map(x, y)[0]:.2f},{self._map(x, y)[1]:.2f}"
                       for x, y in points)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.elements.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{width}"{d}/>')

    def rect(self, x0, y0, x1, y1, fill="#888", stroke="none", width=1):
        a = self._map(x0, y0)
        b = self._map(x1, y1)
        x, y = min(a[0], b[0]), min(a[1], b[1])
        w, h = abs(b[0] - a[0]), abs(b[1] - a[1])
        self.elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')

    def bar(self, x, y, width, color="#1f5fbf", baseline=0.0):
        self.rect(x - width / 2, baseline, x + width / 2, y, fill=color)

    def errorbar(self, x, y, lo, hi, color="#1f5fbf", cap=5, width=2):
        px, _ = self._map(x, y)
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
        elif kind == "square":
            self.elements.append(
                f'<rect x="{px-size:.2f}" y="{py-size:.2f}" width="{2*size}" '
                f'height="{2*size}" fill="{color}"/>')
        elif kind == "diamond":
            self.elements.append(
                f'<polygon points="{px:.2f},{py-size:.2f} {px+size:.2f},{py:.2f} '
                f'{px:.2f},{py+size:.2f} {px-size:.2f},{py:.2f}" fill="{color}"/>')
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

    def text_px(self, x, y, s, size=13, anchor="start", color="#222", weight="normal"):
        self.elements.append(
            f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{s}</text>')

    def legend(self, entries, x=None, y=None):
        """entries: list of (color, label, dash, kind)."""
        if x is None:
            x = self.width - self.right - 210
        if y is None:
            y = self.top + 10
        for i, (color, label, dash, kind) in enumerate(entries):
            yy = y + i * 21
            d = f' stroke-dasharray="{dash}"' if dash else ""
            self.elements.append(
                f'<line x1="{x}" y1="{yy}" x2="{x+28}" y2="{yy}" stroke="{color}" '
                f'stroke-width="2.5"{d}/>')
            if kind == "circle":
                self.elements.append(f'<circle cx="{x+14}" cy="{yy}" r="3.5" fill="{color}"/>')
            elif kind == "square":
                self.elements.append(
                    f'<rect x="{x+10.5}" y="{yy-3.5}" width="7" height="7" fill="{color}"/>')
            elif kind == "diamond":
                self.elements.append(
                    f'<polygon points="{x+14},{yy-4} {x+18},{yy} {x+14},{yy+4} {x+10},{yy}" fill="{color}"/>')
            self.text_px(x + 36, yy + 4, label, size=12)

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
        xticks = nice_ticks(x0, x1, 6) if self.xticks is None else self.xticks
        for i, x in enumerate(xticks):
            px, _ = self._map(x, y0)
            parts.append(f'<line x1="{px:.2f}" y1="{self.height-b}" x2="{px:.2f}" '
                         f'y2="{self.height-b+6}" stroke="#333"/>')
            label = (self.xtick_labels[i] if self.xtick_labels is not None
                     else f"{x:g}")
            parts.append(f'<text x="{px:.2f}" y="{self.height-b+22}" '
                         f'font-family="sans-serif" font-size="12" text-anchor="middle">'
                         f'{label}</text>')
        yticks = nice_ticks(y0, y1, 6) if self.yticks is None else self.yticks
        for y in yticks:
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
