"""Genera los gráficos del README de perfil de GitHub de Maricel Meneses Gómez.

    python perfil/generar.py

Necesita Pillow y NumPy, y la foto en perfil/assets/source/retrato.jpg (cuadrada; si falta, el banner
usa un retrato provisional con sus iniciales). Produce, en claro y oscuro:

* banner-*.svg    terminal con el retrato en 1 bit (difusión de error de Floyd-Steinberg) que aparece con un
                  barrido de escáner, la ficha perfil.yml y un histograma que converge a la normal.
* card-synapse-*.svg  tarjeta del proyecto, en la línea de las tarjetas del perfil de Yoandy.

Las animaciones son CSS dentro del SVG (GitHub las muestra) y se detienen con «reducir movimiento».
"""
from __future__ import annotations

import html
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "assets" / "source" / "retrato.jpg"
OUT = ROOT / "assets"
MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,monospace"
SANS = "-apple-system,BlinkMacSystemFont,'SF Pro Display','Segoe UI',Helvetica,Arial,sans-serif"

THEMES = {
    "light": {"page": "#EEF3FF", "win": "#FFFFFF", "bar": "#F5F7FC", "line": "#D6E0F5", "ink": "#0B1B3F",
              "ink2": "#42506B", "muted": "#7A879E", "accent": "#0A5CFF", "accent2": "#0047D6", "mint": "#0FA697",
              "panel": "#F3F7FF", "dot": "#0A5CFF", "key": "#0047D6", "str": "#0B1B3F", "shadow": "#0A5CFF"},
    "dark": {"page": "#0D1117", "win": "#11161F", "bar": "#161C27", "line": "#243049", "ink": "#EAF0FF",
             "ink2": "#B8C4DE", "muted": "#7D8AA6", "accent": "#3D7DFF", "accent2": "#7FA6FF", "mint": "#2DD4BF",
             "panel": "#0E1420", "dot": "#7FA6FF", "key": "#7FA6FF", "str": "#EAF0FF", "shadow": "#000000"},
}

YAML = [
    (0, "perfil", None),
    (1, "nombre", "Maricel Meneses Gómez"),
    (1, "rol", "Bioestadística · Análisis de datos"),
    (1, "docencia", "10 años · Computación y Estadística"),
    (1, "universidad", "Ciencias Médicas de Villa Clara"),
    (1, "proyecto", "S.Y.N.A.P.S.E. (investigadora principal)"),
    (0, "metodos", None),
    (1, "supervivencia", "Kaplan-Meier · Cox · RMST"),
    (1, "inferencia", "IC · regresión · Monte Carlo"),
    (1, "datos", "Python · pandas · SQL · Jupyter"),
    (0, "contacto", None),
    (1, "linkedin", "/in/maricel9002"),
    (1, "ubicacion", "Lepe, Huelva · híbrido / remoto"),
]


# ─── Retrato en 1 bit ────────────────────────────────────────────────────────

def placeholder(size: int = 600) -> Image.Image:
    """Retrato provisional: silueta con iniciales, para poder generar el banner sin foto."""
    img = Image.new("L", (size, size), 235)
    d = ImageDraw.Draw(img)
    d.ellipse((size * .30, size * .14, size * .70, size * .56), fill=70)
    d.ellipse((size * .12, size * .58, size * .88, size * 1.25), fill=90)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(size * .16))
    except OSError:
        font = ImageFont.load_default()
    d.text((size / 2, size * .35), "MM", fill=235, font=font, anchor="mm")
    return img


def dither(img: Image.Image, w: int, h: int) -> np.ndarray:
    """Recorte centrado, contraste y difusión de error de Floyd-Steinberg → matriz booleana (True = punto)."""
    g = ImageOps.exif_transpose(img).convert("L")
    g = ImageOps.fit(g, (w * 4, h * 4), method=Image.LANCZOS, centering=(0.5, 0.35))
    g = ImageOps.autocontrast(g, cutoff=2)
    g = ImageEnhance.Contrast(g).enhance(1.25).filter(ImageFilter.UnsharpMask(radius=2, percent=120))
    a = np.asarray(g.resize((w, h), Image.LANCZOS), dtype=float) / 255.0
    out = np.zeros_like(a, dtype=bool)
    for y in range(h):
        for x in range(w):
            old = a[y, x]
            new = 1.0 if old > 0.5 else 0.0
            out[y, x] = new == 0.0
            err = old - new
            if x + 1 < w:
                a[y, x + 1] += err * 7 / 16
            if y + 1 < h:
                if x > 0:
                    a[y + 1, x - 1] += err * 3 / 16
                a[y + 1, x] += err * 5 / 16
                if x + 1 < w:
                    a[y + 1, x + 1] += err * 1 / 16
    return out


def dots_path(mask: np.ndarray, x0: float, y0: float, cell: float) -> list[str]:
    """Une los píxeles consecutivos de cada fila en rectángulos: un trazado por banda de 8 filas."""
    bands: list[str] = []
    h, w = mask.shape
    for b0 in range(0, h, 8):
        d = []
        for y in range(b0, min(b0 + 8, h)):
            x = 0
            while x < w:
                if mask[y, x]:
                    s = x
                    while x < w and mask[y, x]:
                        x += 1
                    d.append(f"M{x0 + s * cell:.1f} {y0 + y * cell:.1f}h{(x - s) * cell:.1f}v{cell * .82:.1f}h{-(x - s) * cell:.1f}z")
                else:
                    x += 1
        bands.append("".join(d))
    return bands


# ─── Banner ──────────────────────────────────────────────────────────────────

def banner(theme: str, mask: np.ndarray) -> str:
    t = THEMES[theme]
    W, H = 1180, 560
    px, py, pw, ph = 48, 104, 420, 412
    cell = min((pw - 28) / mask.shape[1], (ph - 56) / mask.shape[0])
    ox = px + (pw - mask.shape[1] * cell) / 2
    oy = py + 44 + (ph - 56 - mask.shape[0] * cell) / 2
    bands = dots_path(mask, ox, oy, cell)
    n_b = len(bands)
    band_svg = "".join(f'<path class="bd" style="animation-delay:{0.25 + i * 1.6 / n_b:.2f}s" d="{d}"/>' for i, d in enumerate(bands) if d)

    # Ficha perfil.yml
    yx, yy, lh = 520, 140, 25
    rows = []
    for i, (ind, k, v) in enumerate(YAML):
        x = yx + 30 + ind * 22
        delay = 1.0 + i * 0.12
        val = "" if v is None else f'<tspan fill="{t["str"]}"> {html.escape(v)}</tspan>'
        rows.append(f'<text class="ln" style="animation-delay:{delay:.2f}s" x="{x}" y="{yy + i * lh}" font-family="{MONO}" font-size="15">'
                    f'<tspan fill="{t["muted"]}" x="{yx - 2}" text-anchor="end">{i + 1:>2}</tspan>'
                    f'<tspan x="{x}" fill="{t["key"]}" font-weight="600">{k}</tspan><tspan fill="{t["muted"]}">:</tspan>{val}</text>')

    # Histograma que converge a la normal (binomial(12, 1/2)), 13 barras
    hx, hy, hw, hh = 548, 470, 560, 54
    probs = [math.comb(12, k) / 4096 for k in range(13)]
    bw = hw / 13
    bars = "".join(
        f'<rect class="bar" style="animation-delay:{2.6 + k * .06:.2f}s" x="{hx + k * bw + 3:.1f}" y="{hy - hh * p / probs[6]:.1f}" '
        f'width="{bw - 6:.1f}" height="{hh * p / probs[6]:.1f}" rx="3" fill="{t["accent"]}" fill-opacity=".85"/>' for k, p in enumerate(probs))
    curve = " ".join(
        f'{"M" if s == 0 else "L"}{hx + (s / 120 * 12 + .5) * bw:.1f} {hy - hh * math.exp(-((s / 120 * 12 - 6) ** 2) / 6):.1f}'
        for s in range(121))

    css = f"""<style>
  .bd {{ opacity: 0; animation: on .5s ease forwards; }}
  .scan {{ animation: scan 2.2s cubic-bezier(.32,.72,0,1) .2s both; }}
  .ln {{ opacity: 0; animation: on .45s ease forwards; }}
  .bar {{ transform-box: fill-box; transform-origin: bottom; transform: scaleY(0); animation: grow .9s cubic-bezier(.23,1,.32,1) forwards; }}
  .curve {{ stroke-dasharray: 900; stroke-dashoffset: 900; animation: draw 1.4s ease 3.4s forwards; }}
  .cursor {{ animation: blink 1s steps(1) infinite; }}
  @keyframes on {{ to {{ opacity: 1; }} }}
  @keyframes grow {{ to {{ transform: scaleY(1); }} }}
  @keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
  @keyframes scan {{ from {{ transform: translateY(0); opacity: .9; }} 90% {{ opacity: .9; }} to {{ transform: translateY({ph - 56}px); opacity: 0; }} }}
  @keyframes blink {{ 50% {{ opacity: 0; }} }}
  @media (prefers-reduced-motion: reduce) {{
    .bd, .ln {{ animation: none; opacity: 1; }} .bar {{ animation: none; transform: none; }}
    .curve {{ animation: none; stroke-dashoffset: 0; }} .scan, .cursor {{ animation: none; opacity: 0; }}
  }}
</style>"""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">
<title id="t">Maricel Meneses Gómez · Bioestadística y análisis de datos</title>
<desc id="d">Terminal con el retrato de Maricel en puntos, su ficha perfil.yml y un histograma que converge a la distribución normal.</desc>
{css}
<defs><filter id="sh" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="14" stdDeviation="18" flood-color="{t["shadow"]}" flood-opacity=".18"/></filter>
<clipPath id="pc"><rect x="{px}" y="{py + 36}" width="{pw}" height="{ph - 36}" rx="6"/></clipPath></defs>
<rect width="{W}" height="{H}" rx="22" fill="{t["page"]}"/>
<rect x="16" y="16" width="{W - 32}" height="{H - 32}" rx="16" fill="{t["win"]}" stroke="{t["line"]}" filter="url(#sh)"/>
<path d="M16 64H{W - 16}" stroke="{t["line"]}"/>
<circle cx="42" cy="40" r="6.5" fill="#FF5F57"/><circle cx="64" cy="40" r="6.5" fill="#FEBC2E"/><circle cx="86" cy="40" r="6.5" fill="#28C840"/>
<text x="{W / 2}" y="45" text-anchor="middle" font-family="{MONO}" font-size="13.5" fill="{t["muted"]}">maricel@datos:~ — vim perfil.yml</text>
<rect x="{px}" y="{py}" width="{pw}" height="{ph}" rx="8" fill="{t["panel"]}" stroke="{t["line"]}"/>
<path d="M{px} {py + 36}H{px + pw}" stroke="{t["line"]}"/>
<text x="{px + 16}" y="{py + 24}" font-family="{MONO}" font-size="12.5" font-weight="700" letter-spacing="1.4" fill="{t["accent2"]}">RETRATO.PNG</text>
<text x="{px + pw - 16}" y="{py + 24}" text-anchor="end" font-family="{MONO}" font-size="11.5" fill="{t["muted"]}">{mask.shape[1]}×{mask.shape[0]} · 1 BIT · FLOYD-STEINBERG</text>
<g clip-path="url(#pc)"><g fill="{t["dot"]}" shape-rendering="crispEdges">{band_svg}</g>
<rect class="scan" x="{px}" y="{py + 36}" width="{pw}" height="3" fill="{t["mint"]}"/></g>
<text x="{yx - 2}" y="{yy - 34}" font-family="{MONO}" font-size="12.5" font-weight="700" letter-spacing="1.4" fill="{t["accent2"]}">PERFIL.YML</text>
{"".join(rows)}
<rect class="cursor" x="{yx + 30}" y="{yy + len(YAML) * lh - 15}" width="9" height="18" fill="{t["accent"]}"/>
<path d="M{hx} {hy + .5}H{hx + hw}" stroke="{t["line"]}"/>
{bars}
<path class="curve" d="{curve}" fill="none" stroke="{t["mint"]}" stroke-width="2.5"/>
<text x="{hx + hw}" y="{hy + 20}" text-anchor="end" font-family="{MONO}" font-size="11.5" fill="{t["muted"]}">binomial(12; 0,5) → N(6; 3)</text>
</svg>
"""


# ─── Tarjeta de proyecto ─────────────────────────────────────────────────────

def km_steps(seed: int, n: int, hz: float) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    times = sorted(min(rng.expovariate(hz), 1.0) for _ in range(n))
    s, pts, risk = 1.0, [(0.0, 1.0)], n
    for t in times:
        if t >= 1:
            break
        s *= 1 - 1 / risk
        risk -= 1
        pts.append((t, s))
    return pts


def card(theme: str) -> str:
    dark = theme == "dark"
    a, b = ("#0A3BB8", "#0A5CFF") if not dark else ("#071B57", "#0A4BD6")
    W, H = 1200, 300

    def km(seed: int, hz: float, color: str, op: float) -> str:
        pts = km_steps(seed, 70, hz)
        x0, x1, y0, y1 = 640, 1160, 70, 250
        d = f"M{x0} {y0}"
        for t, s in pts[1:]:
            d += f"H{x0 + (x1 - x0) * t:.1f}V{y1 - (y1 - y0) * s:.1f}"
        return f'<path d="{d}H{x1}" fill="none" stroke="{color}" stroke-opacity="{op}" stroke-width="3"/>'
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="S.Y.N.A.P.S.E.: validez inferencial y riesgo de reidentificación de datos sanitarios sintéticos">
<defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="{a}"/><stop offset="1" stop-color="{b}"/></linearGradient>
<clipPath id="c"><rect width="{W}" height="{H}" rx="20"/></clipPath></defs>
<g clip-path="url(#c)">
<rect width="{W}" height="{H}" fill="url(#g)"/>
{km(4, 1.6, "#FFFFFF", .55)}{km(23, 1.8, "#7DF0E2", .8)}
<path d="M0 248 C 300 214, 600 278, 900 238 S 1200 250, 1200 250 V300 H0 Z" fill="#000000" fill-opacity=".18"/>
</g>
<circle cx="34" cy="34" r="7" fill="#FF5F57"/><circle cx="56" cy="34" r="7" fill="#FEBC2E"/><circle cx="78" cy="34" r="7" fill="#28C840"/>
<text x="56" y="128" font-family="{SANS}" font-weight="800" font-size="52" letter-spacing="-1.5" fill="#FFFFFF">S.Y.N.A.P.S.E.</text>
<text x="58" y="170" font-family="{SANS}" font-size="21" fill="#FFFFFF" fill-opacity=".92">Datos sanitarios sintéticos · validez inferencial y privacidad</text>
<text x="58" y="206" font-family="{SANS}" font-size="16" fill="#FFFFFF" fill-opacity=".78">Kaplan-Meier · Cox · 320 evaluaciones · investigadora principal</text>
<text x="58" y="270" font-family="{SANS}" font-size="14" fill="#FFFFFF" fill-opacity=".85">Maricel Meneses Gómez · Yoandy Ramírez Delgado</text>
</svg>
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    img = Image.open(SRC) if SRC.exists() else placeholder()
    mask = dither(img, 150, 160)
    for theme in THEMES:
        (OUT / f"banner-{theme}.svg").write_text(banner(theme, mask), encoding="utf-8")
        (OUT / f"card-synapse-{theme}.svg").write_text(card(theme), encoding="utf-8")
    sizes = {p.name: f"{p.stat().st_size / 1024:.0f} KB" for p in sorted(OUT.glob("*.svg"))}
    print(("foto: " + SRC.name) if SRC.exists() else "foto: provisional (falta assets/source/retrato.jpg)", sizes)


if __name__ == "__main__":
    main()
