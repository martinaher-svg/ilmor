#!/usr/bin/env python3
"""
Static site builder: fetches the three locations once and writes anonymised HTML pages.

Environment (GitHub Actions secrets):
  WEATHER_API_KEY   OpenWeatherMap key
  LOC_BE, LOC_IT, LOC_EE   "lat,lon" per location (never written to output)
  SITE_TOKEN        random URL segment the pages are published under

Output layout (--out, default _site):
  index.html, 404.html, robots.txt, .nojekyll     decoys / crawler opt-out
  <SITE_TOKEN>/index.html                          overview of all three
  <SITE_TOKEN>/{be,it,ee}/index.html               detail pages

Visitors only ever read these files; nothing on the site can trigger an API call.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from engine import WeatherPayload, evaluate_vibe, fetch_weather, nearest_step

REGIONS: List[Tuple[str, str]] = [("BE", "🇧🇪"), ("IT", "🇮🇹"), ("EE", "🇪🇪")]
RETRY_DELAYS = (0, 5, 15)  # seconds before each attempt

# Rich colour names used by the vibe engine -> CSS (tuned for a dark background)
COLORS: Dict[str, str] = {
    "bright_yellow": "#f9e27d", "bright_red": "#ff6b6b", "bright_white": "#f0f6fc", "khaki": "#d7d787",
    "khaki1": "#f0e68c", "light_slate_grey": "#8a99ad", "bright_blue": "#79a6ff", "bright_cyan": "#67e8f9",
    "sea_green2": "#6ee7a8", "light_goldenrod1": "#ffe08a", "dark_violet": "#b36bdb", "bright_magenta": "#ff79c6",
    "dim white": "#9da7b3", "dark_orange": "#ff8c1a", "sandy_brown": "#f4a460", "grey50": "#8b949e",
    "orange3": "#e0a040", "bright_green": "#7ee787", "dark_olive_green3": "#a9d36b", "sky_blue2": "#87afff",
    "gold1": "#ffd700",
}
AQI = {1: ("Good", "#7ee787"), 2: ("Fair", "#67e8f9"), 3: ("Moderate", "#f9e27d"), 4: ("Poor", "#ff6b6b"), 5: ("Hazardous", "#ff3b3b")}
AQI_ET = {1: ("Hea", "#7ee787"), 2: ("Rahuldav", "#67e8f9"), 3: ("Mõõdukas", "#f9e27d"), 4: ("Halb", "#ff6b6b"), 5: ("Ohtlik", "#ff3b3b")}

LIGHTING_ET = {
    "Gloomy / Low Diffuse Lux": "Sombune / Hajus hämarus",
    "Filtered Sun / Soft Glare": "Filtreeritud päike / Pehme kuma",
    "Direct Bright Sun": "Otsene ere päike",
    "Darkness": "Pimedus",
}

from vibe_ee import get_ee_vibe, translate_weather_desc


@dataclass
class Location:
    region: str
    flag: str
    lat: float
    lon: float
    raw: str  # the secret exactly as supplied, for the leak guard


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def load_config() -> Tuple[str, str, List[Location]]:
    required = ["WEATHER_API_KEY", "SITE_TOKEN", *(f"LOC_{r}" for r, _ in REGIONS)]
    missing = [k for k in required if not os.getenv(k, "").strip()]
    if missing:
        hint = (" (none are set: they must be repository *Secrets*, not Variables, "
                "under Settings → Secrets and variables → Actions)"
                if len(missing) == len(required) else "")
        die(f"not set: {', '.join(missing)}{hint}")
    api_key = os.getenv("WEATHER_API_KEY", "").strip()
    token = os.getenv("SITE_TOKEN", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,}", token):
        die("SITE_TOKEN must be at least 16 URL-safe characters")

    locations = []
    for region, flag in REGIONS:
        raw = os.getenv(f"LOC_{region}", "").strip()
        try:
            lat_s, lon_s = (p.strip() for p in raw.split(","))
            lat, lon = float(lat_s), float(lon_s)
        except ValueError:
            die(f"LOC_{region} must look like 'lat,lon'")
        locations.append(Location(region, flag, lat, lon, raw))

    if os.getenv("GITHUB_ACTIONS") == "true":
        # GitHub only masks whole secret values; also mask the individual parts.
        for loc in locations:
            for part in {loc.raw, *loc.raw.split(","), f"{loc.lat}", f"{loc.lon}"}:
                if part.strip():
                    print(f"::add-mask::{part.strip()}")
        print(f"::add-mask::{token}")
    return api_key, token, locations


async def fetch_one(client: httpx.AsyncClient, api_key: str, loc: Location) -> Tuple[Location, Optional[WeatherPayload], Optional[str]]:
    last_err = None
    for delay in RETRY_DELAYS:
        if delay:
            await asyncio.sleep(delay)
        try:
            return loc, await fetch_weather(client, api_key, loc.lat, loc.lon, loc.region), None
        except Exception as e:  # never echo request URLs: they contain coordinates
            last_err = type(e).__name__
            if isinstance(e, httpx.HTTPStatusError):
                last_err = f"HTTP {e.response.status_code}"
    return loc, None, last_err


# ---------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------

E = html.escape

CSS = """
:root{--bg:#0d1117;--panel:#131a24;--line:#21262d;--txt:#c9d1d9;--dim:#8b949e;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:15px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  padding:env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left)}
main{max-width:680px;margin:0 auto;padding:12px}
header{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:12px}
header .brand{color:var(--accent);font-weight:700;letter-spacing:.08em;font-size:13px;text-decoration:none}
nav{display:flex;gap:6px}
nav a{color:var(--dim);text-decoration:none;border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-weight:700}
nav a.on{color:var(--bg);background:var(--accent);border-color:var(--accent)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:12px}
.vibe h1{margin:0 0 6px;font-size:22px;letter-spacing:.03em;color:var(--c)}
.vibe p{margin:0;font-style:italic;color:var(--c);opacity:.92}
.now{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.now .big{grid-column:1/-1;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.now .big b{font-size:40px;color:#67e8f9;line-height:1}
.tile{background:#0f141c;border:1px solid var(--line);border-radius:8px;padding:8px}
.tile small{display:block;color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
h2{font-size:12px;letter-spacing:.1em;color:var(--dim);margin:0 0 8px;text-transform:uppercase}
.row{display:grid;grid-template-columns:3.4em 4.2em 1fr 4.6em;gap:6px;align-items:center;padding:6px 0;border-top:1px solid var(--line)}
.row:first-of-type{border-top:0}
.row.tw{grid-template-columns:3.4em 4.2em 1fr 2.6em}
.row .d{grid-column:1/-1;color:var(--dim);font-size:12px;margin-top:-4px}
.bar{position:relative;height:18px;border-radius:4px;background:#1b2330;overflow:hidden;font-size:12px}
.bar i{position:absolute;inset:0 auto 0 0;background:#2f5fb8}
.bar span{position:relative;padding:0 6px;line-height:18px;white-space:nowrap}
.wet{color:#79a6ff}.warn{color:#ff6b6b;font-weight:700}.windy{color:#ff6b6b}
.kv{display:grid;grid-template-columns:6em 1fr;gap:4px 8px}
.kv span:nth-child(odd){color:var(--dim)}
footer{color:var(--dim);font-size:12px;text-align:center;padding:8px 0 20px}
.stale footer time{color:#ff6b6b;font-weight:700}
.err{color:#ff6b6b}
.ov a{display:block;text-decoration:none;color:inherit}
.ov .top{display:flex;justify-content:space-between;align-items:baseline}
.ov .reg{font-weight:700;font-size:18px}
.ov .t{font-size:26px;color:#67e8f9}
.ov h1{font-size:17px;margin:6px 0 2px}
"""

JS = """
(()=>{const t=document.querySelector('time[data-ts]');if(!t)return;const ts=+t.dataset.ts;
const f=()=>{const m=Math.max(0,Math.round((Date.now()/1e3-ts)/60));
t.textContent=m<1?'just now':m<60?m+' min ago':Math.floor(m/60)+' h '+(m%60)+' min ago';
document.body.classList.toggle('stale',m>45)};f();setInterval(f,30000)})();
"""


def page(title: str, body: str, generated: int, lang: str = "en", app_title: str = "Atmospheric") -> str:
    if lang == "et":
        footer_html = f'<footer>Uuendatud <time data-ts="{generated}" datetime="{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(generated))}">{time.strftime("%H:%M UTC", time.gmtime(generated))}</time> · värskendub iga 15 min järel</footer>'
        js_code = """(()=>{const t=document.querySelector('time[data-ts]');if(!t)return;const ts=+t.dataset.ts;
const f=()=>{const m=Math.max(0,Math.round((Date.now()/1e3-ts)/60));
t.textContent=m<1?'just praegu':m<60?m+' min tagasi':Math.floor(m/60)+' t '+(m%60)+' min tagasi';
document.body.classList.toggle('stale',m>45)};f();setInterval(f,30000)})();"""
    else:
        footer_html = f'<footer>Updated <time data-ts="{generated}" datetime="{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(generated))}">{time.strftime("%H:%M UTC", time.gmtime(generated))}</time> · refreshes every 15 min</footer>'
        js_code = JS

    return f"""<!doctype html>
<html lang="{lang}"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="robots" content="noindex,nofollow,noarchive,nosnippet">
<meta name="referrer" content="no-referrer">
<meta http-equiv="refresh" content="300">
<meta name="theme-color" content="#0d1117">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data: 'self';">
<title>{E(title)}</title>
<link rel="icon" type="image/png" sizes="32x32" href="favicon-32.png">
<link rel="icon" type="image/png" sizes="192x192" href="icon.png">
<link rel="apple-touch-icon" sizes="180x180" href="apple-touch-icon.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="{E(app_title)}">
<style>{CSS}</style>
</head><body><main>
{body}
{footer_html}
</main><script>{js_code}</script></body></html>
"""


def nav(active: Optional[str], prefix: str) -> str:
    links = [f'<a href="{prefix}{r.lower()}/" class="{"on" if r == active else ""}">{r}</a>' for r, _ in REGIONS]
    return f'<header><a class="brand" href="{prefix}">ATMOSPHERIC DISPATCH</a><nav>{"".join(links)}</nav></header>'


def render_detail(loc: Location, w: Optional[WeatherPayload], err: Optional[str], generated: int) -> str:
    is_ee = (loc.region == "EE")
    lang = "et" if is_ee else "en"
    head = nav(loc.region, "../")
    if w is None:
        err_msg = (f'Andmed pole sellel tsüklil saadaval ({E(err or "tundmatu viga")}). Järgmine katse ≤15 min pärast.'
                   if is_ee else f'Data unavailable this cycle ({E(err or "unknown error")}). Next attempt in ≤15 min.')
        body = f'{head}<section class="card"><h2>{loc.flag} {loc.region}</h2><p class="err">{err_msg}</p></section>'
        return page(f"{loc.region} · {'pole saadaval' if is_ee else 'unavailable'}", body, generated, lang=lang)

    v = evaluate_vibe(w)
    if is_ee:
        v_name, v_quote = get_ee_vibe(v.name)
    else:
        v_name, v_quote = v.name, v.quote

    color = COLORS.get(v.color, "#f0f6fc")
    uv = f"~{w.uv_index:.1f}" if w.uv_estimated else f"{w.uv_index:.1f}"
    precip = f"{w.rain_amount:.1f} mm" + (f" · ❄ {w.snow_amount:.1f} mm" if w.snow_amount > 0 else "")

    desc_disp = translate_weather_desc(w.desc) if is_ee else w.desc.title()
    feels_label = "tuntav" if is_ee else "feels"
    wind_label = "Tuul" if is_ee else "Wind"
    hum_label = "Õhuniiskus" if is_ee else "Humidity"
    uv_label = "UV" if is_ee else "UV"
    rain_now_label = "Sadu praegu" if is_ee else "Rain now"
    rain_chance_label = "Saju tõenäosus" if is_ee else "Rain chance"
    press_label = "Õhurõhk" if is_ee else "Pressure"

    parts = [head]
    parts.append(f'<section class="card vibe" style="--c:{color}"><h1>{E(v_name)}</h1><p>{E(v_quote)}</p></section>')
    parts.append(f'''<section class="card now">
<div class="big"><b>{w.temp:+.1f}°C</b><span>{feels_label} {w.feels_like:+.1f}°C · {E(desc_disp)}</span></div>
<div class="tile"><small>{wind_label}</small>{w.wind_speed:.1f} m/s<br><span style="color:var(--dim)">{w.wind_deg}°</span></div>
<div class="tile"><small>{hum_label}</small>{w.humidity}%</div>
<div class="tile"><small>{uv_label}</small>{uv}</div>
<div class="tile"><small>{rain_now_label}</small>{precip}</div>
<div class="tile"><small>{rain_chance_label}</small>{w.rain_prob:.0f}%</div>
<div class="tile"><small>{press_label}</small>{w.pressure} hPa</div>
</section>''')

    rows = []
    tw_targets = (("Hommik", 11), ("Pärastlõuna", 14), ("Õhtu", 17)) if is_ee else (("Morning", 11), ("Afternoon", 14), ("Evening", 17))
    for label, hr in tw_targets:
        s = nearest_step(w.forecast_steps, hr)
        if s:
            flag = '<span class="warn">[!]</span>' if (s.rain_mm > 0.4 or s.pop > 0.5) else "[-]"
            s_desc = translate_weather_desc(s.desc) if is_ee else s.desc
            rows.append(f'<div class="row tw"><b>{s.time_str}</b><span>{s.temp:+.1f}°</span>'
                        f'<span class="{"wet" if s.rain_mm > 0 else ""}">{int(s.pop*100)}% · {s.rain_mm:.1f}mm</span>'
                        f'<span>{flag}</span><span class="d">{label} · {E(s_desc)}</span></div>')
    if rows:
        tw_title = "Võtmeajad" if is_ee else "Target windows"
        parts.append(f'<section class="card"><h2>{tw_title}</h2>{"".join(rows)}</section>')

    frows = []
    for s in w.forecast_steps:
        pct = int(s.pop * 100)
        s_desc = translate_weather_desc(s.desc) if is_ee else s.desc
        frows.append(f'<div class="row"><b>{s.time_str}</b><span>{s.temp:+.1f}°</span>'
                     f'<div class="bar"><i style="width:{pct}%"></i><span>{pct}% · {s.rain_mm:.1f}mm</span></div>'
                     f'<span class="{"windy" if s.wind_speed > 10 else ""}">{s.wind_speed:.1f}m/s</span>'
                     f'<span class="d">{E(s_desc)}</span></div>')
    fc_title = "24 h prognoos" if is_ee else "24-hour forecast"
    parts.append(f'<section class="card"><h2>{fc_title}</h2>{"".join(frows)}</section>')

    eph = w.ephemeris
    air = ""
    if w.aqi_data:
        if is_ee:
            label, ac = AQI_ET.get(w.aqi_data.aqi, ("Tundmatu", "#f0f6fc"))
            air = (f'<span>Õhk</span><span><b style="color:{ac}">{w.aqi_data.aqi} {label}</b> · '
                   f'PM2.5 {w.aqi_data.pm2_5:.1f} · PM10 {w.aqi_data.pm10:.1f} µg/m³</span>')
        else:
            label, ac = AQI.get(w.aqi_data.aqi, ("Unknown", "#f0f6fc"))
            air = (f'<span>Air</span><span><b style="color:{ac}">{w.aqi_data.aqi} {label}</b> · '
                   f'PM2.5 {w.aqi_data.pm2_5:.1f} · PM10 {w.aqi_data.pm10:.1f} µg/m³</span>')

    if is_ee:
        eph_title = "Päev & Valgus"
        daylight_label = "Valget aega"
        dl_str = eph.daylight_left_str
        if dl_str == "Night Cycle":
            dl_disp = "Öö"
        elif "h " in dl_str and "m" in dl_str:
            dl_disp = dl_str.replace("h ", " t ").replace("m", " min jäänud")
        else:
            dl_disp = dl_str
        clouds_label = "Pilvisus"
        lighting_disp = LIGHTING_ET.get(eph.lighting_condition, eph.lighting_condition)
    else:
        eph_title = "Ephemeris"
        daylight_label = "Daylight"
        dl_disp = eph.daylight_left_str
        clouds_label = "Clouds"
        lighting_disp = eph.lighting_condition

    parts.append(f'''<section class="card"><h2>{eph_title}</h2><div class="kv">
<span>{daylight_label}</span><span>{E(dl_disp)}</span>
<span>{clouds_label}</span><span>{eph.cloud_pct}% · {E(lighting_disp)}</span>
{air}</div></section>''')
    app_titles = {"EE": "Ilm EE", "BE": "Weather BE", "IT": "Meteo IT"}
    app_title = app_titles.get(loc.region, f"Weather {loc.region}")
    return page(f"{loc.region} · {v_name}", "\n".join(parts), generated, lang=lang, app_title=app_title)


def render_overview(results, generated: int) -> str:
    cards = [nav(None, "./")]
    for loc, w, err in results:
        is_ee = (loc.region == "EE")
        if w is None:
            err_text = "Pole sellel tsüklil saadaval" if is_ee else "Unavailable this cycle"
            cards.append(f'<section class="card ov"><a href="./{loc.region.lower()}/"><div class="top"><span class="reg">{loc.flag} {loc.region}</span></div>'
                         f'<p class="err">{err_text}</p></a></section>')
            continue
        v = evaluate_vibe(w)
        if is_ee:
            v_name, _ = get_ee_vibe(v.name)
            desc_text = f'{E(translate_weather_desc(w.desc))} · {w.wind_speed:.0f} m/s · {w.rain_prob:.0f}% vihma'
        else:
            v_name = v.name
            desc_text = f'{E(w.desc.title())} · {w.wind_speed:.0f} m/s · {w.rain_prob:.0f}% rain'

        cards.append(f'''<section class="card ov"><a href="./{loc.region.lower()}/">
<div class="top"><span class="reg">{loc.flag} {loc.region}</span><span class="t">{w.temp:+.0f}°</span></div>
<h1 style="color:{COLORS.get(v.color, "#f0f6fc")}">{E(v_name)}</h1>
<span style="color:var(--dim)">{desc_text}</span></a></section>''')
    return page("Atmospheric Dispatch", "\n".join(cards), generated, app_title="Atmospheric")


# ---------------------------------------------------------------------
# Leak guard: refuse to publish anything containing coordinates or the token
# ---------------------------------------------------------------------

def forbidden_strings(locations: List[Location], token: str) -> List[Tuple[str, str]]:
    """(label, value) pairs that must never appear in page contents."""
    out = {("SITE_TOKEN", token)}
    for loc in locations:
        out.add((f"LOC_{loc.region} raw", loc.raw))
        for axis, v in (("lat", loc.lat), ("lon", loc.lon)):
            for dp in (3, 4):
                out.add((f"LOC_{loc.region} {axis} {dp}dp", f"{v:.{dp}f}"))
            # Only reprs with ≥3 decimals are specific; '10.2' would collide with a +10.2° reading
            if len(str(v).partition(".")[2]) >= 3:
                out.add((f"LOC_{loc.region} {axis} repr", str(v)))
    return sorted(p for p in out if len(p[1]) >= 4)


def assert_clean(files: Dict[str, str], bad: List[Tuple[str, str]]) -> None:
    for name, content in files.items():
        for label, s in bad:
            i = content.find(s)
            if i >= 0:
                ctx = (content[max(0, i - 40):i] + "█" * len(s) + content[i + len(s):i + len(s) + 40]).replace("\n", " ")
                die(f"leak guard: refusing to publish, {name!r} contains {label}: …{ctx}…")


# ---------------------------------------------------------------------

async def build(out: Path) -> int:
    api_key, token, locations = load_config()
    generated = int(time.time())
    async with httpx.AsyncClient(timeout=10.0) as client:
        results = await asyncio.gather(*(fetch_one(client, api_key, loc) for loc in locations))

    ok = sum(1 for _, w, _ in results if w is not None)
    for loc, w, err in results:
        print(f"{loc.region}: {'ok' if w else 'FAILED (' + str(err) + ')'}")
    if ok == 0:
        print("All locations failed; not publishing (the previous deployment stays live).", file=sys.stderr)
        return 1

    files: Dict[str, str] = {
        f"{token}/index.html": render_overview(results, generated),
        "index.html": page("Nothing here", "<p style='color:#8b949e'>Nothing to see here.</p>", generated).replace('<meta http-equiv="refresh" content="300">', ""),
        "404.html": page("Not found", "<p style='color:#8b949e'>Not found.</p>", generated).replace('<meta http-equiv="refresh" content="300">', ""),
        "robots.txt": "User-agent: *\nDisallow: /\n",
        ".nojekyll": "",
    }
    for loc, w, err in results:
        files[f"{token}/{loc.region.lower()}/index.html"] = render_detail(loc, w, err, generated)

    # Token is allowed in file *paths* but must never appear in file *contents*
    assert_clean(files, forbidden_strings(locations, token))

    for rel, content in files.items():
        path = out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    # Copy region-specific and overview icons
    icons_src = Path(__file__).resolve().parent / "icons"
    if icons_src.exists():
        import shutil
        for loc, _, _ in results:
            reg = loc.region.lower()
            src_reg = icons_src / reg
            if src_reg.exists():
                dst_reg = out / token / reg
                dst_reg.mkdir(parents=True, exist_ok=True)
                for f in src_reg.glob("*.png"):
                    shutil.copy2(f, dst_reg / f.name)
        # Overview icons
        src_ov = icons_src / "ov"
        if src_ov.exists():
            dst_ov = out / token
            for f in src_ov.glob("*.png"):
                shutil.copy2(f, dst_ov / f.name)
                shutil.copy2(f, out / f.name)

    print(f"Wrote {len(files)} files ({ok}/3 locations live).")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="_site", type=Path)
    args = ap.parse_args()
    sys.exit(asyncio.run(build(args.out)))


if __name__ == "__main__":
    main()
