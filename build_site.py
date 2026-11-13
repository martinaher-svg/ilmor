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

from engine import ForecastStep, WeatherPayload, deg_to_cardinal, evaluate_vibe, fetch_weather, nearest_step

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
    "Fog / Low Visibility": "Udu / Piiratud nähtavus",
    "Gloomy / Low Diffuse Lux": "Sombune / Hajus hämarus",
    "Hazy / Diffused Mist": "Uduvine / Hajus valgus",
    "Filtered Sun / Soft Glare": "Filtreeritud päike / Pehme kuma",
    "Direct Bright Sun": "Otsene ere päike",
    "Darkness": "Pimedus",
}

from vibe_be import get_be_vibe
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
.row{display:grid;grid-template-columns:3.2em 4.0em 1fr 6.0em;gap:6px;align-items:center;padding:6px 0;border-top:1px solid var(--line)}
.row:first-of-type{border-top:0}
.row.tw{grid-template-columns:3.2em 4.0em 1fr 2.6em}
.row .wcol{text-align:right;font-size:13px;white-space:nowrap}
.row .d{grid-column:1/-1;color:var(--dim);font-size:12px;margin-top:-4px}
.bar{position:relative;height:18px;border-radius:4px;background:#1b2330;overflow:hidden;font-size:12px}
.bar i{position:absolute;inset:0 auto 0 0;background:#2f5fb8}
.bar.ice i{background:#9a3412}
.bar.snow i{background:#0e7490}
.bar span{position:relative;padding:0 6px;line-height:18px;white-space:nowrap}
.wet{color:#79a6ff}.warn{color:#ff6b6b;font-weight:700}.windy{color:#ff6b6b}
.ice{color:#f59e0b;font-weight:700}.fog{color:#d7d787}.frost{color:#67e8f9}
.badge{display:inline-block;font-size:11px;padding:1px 6px;border-radius:4px;border:1px solid currentColor;letter-spacing:.04em}
.day-sep{margin:8px 0 2px;padding-top:6px;border-top:1px dashed var(--line);font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
.day-sep+.row{border-top:0}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 8px}
.kv span:nth-child(odd){color:var(--dim)}
footer{color:var(--dim);font-size:12px;text-align:center;padding:8px 0 20px}
.stale footer time{color:#ff6b6b;font-weight:700}
.err{color:#ff6b6b}
.ov a{display:block;text-decoration:none;color:inherit}
.ov .top{display:flex;justify-content:space-between;align-items:baseline}
.ov .reg{font-weight:700;font-size:18px}
.ov .t{font-size:26px;color:#67e8f9}
.ov .t small{font-size:14px;color:var(--dim);margin-left:4px}
.ov h1{font-size:17px;margin:6px 0 2px}
"""

JS = """
(()=>{const f=window.updateTime=()=>{const t=document.querySelector('time[data-ts]');if(!t)return;const ts=+t.dataset.ts;
const m=Math.max(0,Math.round((Date.now()/1e3-ts)/60));
t.textContent=m<1?'just now':m<60?m+' min ago':Math.floor(m/60)+' h '+(m%60)+' min ago';
document.body.classList.toggle('stale',m>45)};f();setInterval(f,30000)})();
"""

POLL_JS = """(() => {
  let busy = false;
  const refresh = async () => {
    if (busy) return;
    busy = true;
    try {
      const res = await fetch(window.location.href, { cache: 'no-store' });
      if (res.ok) {
        const html = await res.text();
        const doc = new DOMParser().parseFromString(html, 'text/html');
        const newMain = doc.querySelector('main');
        if (newMain) {
          document.querySelector('main').replaceWith(newMain);
          // Footer holds the "Updated" timestamp; swap it too so the age resets
          const newFooter = doc.querySelector('footer');
          const oldFooter = document.querySelector('footer');
          if (newFooter && oldFooter) oldFooter.replaceWith(newFooter);
          if (window.updateTime) window.updateTime();
        }
      }
    } catch (e) {
      // Offline: keep existing rendered data intact without crashing
    } finally {
      busy = false;
    }
  };
  setInterval(refresh, 300000);
  // Phones suspend timers in background tabs / home-screen apps: refresh on return
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') refresh();
  });
  window.addEventListener('pageshow', (e) => { if (e.persisted) refresh(); });
})();"""


def page(title: str, body: str, generated: int, lang: str = "en", app_title: str = "Atmospheric") -> str:
    if lang == "et":
        footer_html = f'<footer>Uuendatud <time data-ts="{generated}" datetime="{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(generated))}">{time.strftime("%H:%M UTC", time.gmtime(generated))}</time> · värskendub iga 15 min järel</footer>'
        time_js = """(()=>{const f=window.updateTime=()=>{const t=document.querySelector('time[data-ts]');if(!t)return;const ts=+t.dataset.ts;
const m=Math.max(0,Math.round((Date.now()/1e3-ts)/60));
t.textContent=m<1?'just praegu':m<60?m+' min tagasi':Math.floor(m/60)+' t '+(m%60)+' min tagasi';
document.body.classList.toggle('stale',m>45)};f();setInterval(f,30000)})();"""
    else:
        footer_html = f'<footer>Updated <time data-ts="{generated}" datetime="{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(generated))}">{time.strftime("%H:%M UTC", time.gmtime(generated))}</time> · refreshes every 15 min</footer>'
        time_js = JS

    js_code = f"{time_js}\n{POLL_JS}"

    return f"""<!doctype html>
<html lang="{lang}"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="robots" content="noindex,nofollow,noarchive,nosnippet">
<meta name="referrer" content="no-referrer">
<meta name="theme-color" content="#0d1117">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; connect-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data: 'self';">
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


def _format_descs(tags: Tuple[str, ...], fallback: str, is_ee: bool) -> str:
    items = tags if tags else ((fallback,) if fallback else ("Unknown",))
    if is_ee:
        parts = [translate_weather_desc(t) for t in items]
        return " + ".join([parts[0], *(p.lower() for p in parts[1:])])
    return " + ".join(t.title() for t in items)


def _format_vis(meters: int) -> str:
    if meters < 1000:
        return f"{meters} m"
    if meters < 10000:
        return f"{meters / 1000.0:.1f} km"
    return "10+ km"


def _vis_label(meters: int, is_ee: bool) -> str:
    base = _format_vis(meters)
    if meters < 1000:
        return f"{base} ({'Tihe udu' if is_ee else 'Dense fog'})"
    if meters < 4000:
        return f"{base} ({'Uduvine' if is_ee else 'Mist'})"
    return base


def _format_step_precip(s: ForecastStep) -> str:
    if s.rain_mm > 0 and s.snow_mm > 0:
        return f"❄{s.total_precip_mm:.1f}mm"
    if s.snow_mm > 0:
        return f"❄{s.snow_mm:.1f}mm"
    return f"{s.rain_mm:.1f}mm"


def _step_context(s: ForecastStep, is_ee: bool) -> Tuple[str, str, str, str]:
    """Return (tw_flag_html, precip_cls, bar_cls, extra_notes_html) for a forecast step."""
    is_stormy = s.wind_speed >= 10.0 or s.effective_wind_gust >= 13.0
    is_freezing = s.is_freezing_wet
    is_snowy = s.snow_mm >= 0.3 or "snow" in s.desc_lower or "sleet" in s.desc_lower
    is_foggy = s.is_dense_fog
    is_bitter_cold = s.temp <= -5.0 or (s.feels_like is not None and s.feels_like <= -8.0)
    is_wet_risk = s.total_precip_mm > 0.4 or s.pop > 0.5

    if is_freezing:
        flag = '<span class="ice">[!]</span>'
    elif is_wet_risk or is_stormy or is_snowy:
        flag = '<span class="warn">[!]</span>'
    elif is_foggy:
        flag = '<span class="fog">[!]</span>'
    elif is_bitter_cold:
        flag = '<span class="frost">[!]</span>'
    else:
        flag = "[-]"

    bar_cls = "bar ice" if is_freezing else ("bar snow" if (s.snow_mm > 0 or is_snowy) else "bar")
    precip_cls = "ice" if is_freezing else ("frost" if s.snow_mm > 0 else ("wet" if s.rain_mm > 0 else ""))

    extras: List[str] = []
    if is_freezing:
        extras.append(f'<span class="ice">{"Libeduse oht" if is_ee else "Ice risk"}</span>')
    if is_foggy and "fog" not in s.desc_lower:
        extras.append(f'<span class="fog">{"Tihe udu" if is_ee else "Dense fog"}</span>')
    if is_bitter_cold:
        extras.append(f'<span class="frost">{"Karm külm" if is_ee else "Bitter cold"}</span>')
    if s.rain_mm > 0 and s.snow_mm > 0:
        extras.append(
            f'{"vihm" if is_ee else "rain"} {s.rain_mm:.1f} + {"lumi" if is_ee else "snow"} {s.snow_mm:.1f}mm'
        )

    if s.visibility is not None and s.visibility < 4000:
        extras.append(f'{"näht." if is_ee else "vis"} {_format_vis(s.visibility)}')
    if s.feels_like is not None and abs(s.feels_like - s.temp) >= 2.5:
        extras.append(f'{"tuntav" if is_ee else "feels"} {s.feels_like:+.0f}°')
    if s.effective_wind_gust >= s.wind_speed + 2.0 and s.effective_wind_gust >= 10.0:
        extras.append(f'{"iilid" if is_ee else "gusts"} {s.effective_wind_gust:.0f} m/s')

    return flag, precip_cls, bar_cls, " · ".join(extras)


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
    elif loc.region == "BE":
        v_name, v_quote = get_be_vibe(v.name)
    else:
        v_name, v_quote = v.name, v.quote

    color = COLORS.get(v.color, "#f0f6fc")
    uv = f"~{w.uv_index:.1f}" if w.uv_estimated else f"{w.uv_index:.1f}"
    if w.snow_amount > 0 and w.rain_amount > 0:
        precip = f"{w.rain_amount:.1f} mm · ❄ {w.snow_amount:.1f} mm"
    elif w.snow_amount > 0:
        precip = f"❄ {w.snow_amount:.1f} mm"
    else:
        precip = f"{w.rain_amount:.1f} mm"

    desc_disp = _format_descs(w.effective_descs, w.desc, is_ee)
    feels_label = "tuntav" if is_ee else "feels"
    wind_label = "Tuul" if is_ee else "Wind"
    hum_label = "Õhuniiskus" if is_ee else "Humidity"
    dew_label = "Kastep." if is_ee else "Dew"
    uv_label = "UV"
    rain_now_label = "Sadu praegu" if is_ee else "Precip now"
    rain_chance_label = "Saju tõenäosus" if is_ee else "Rain chance"
    press_label = "Õhurõhk" if is_ee else "Pressure"

    gust_val = w.effective_wind_gust
    if gust_val >= w.wind_speed + 1.5:
        gust_lbl = f"iilid {gust_val:.0f} m/s" if is_ee else f"gusts {gust_val:.0f} m/s"
        gust_cls = ' class="windy"' if gust_val >= 13.0 else ""
        wind_sub = f'{deg_to_cardinal(w.wind_deg)} · <span{gust_cls}>{gust_lbl}</span>'
    else:
        wind_sub = f'{deg_to_cardinal(w.wind_deg)} · {w.wind_deg}°'
    wind_val_cls = ' class="windy"' if (w.wind_speed >= 10.0 or gust_val >= 13.0) else ""

    vis_inline = (f' · {"näht." if is_ee else "vis"} {_format_vis(w.visibility)}'
                  if (w.visibility is not None and w.visibility < 4000) else "")
    freeze_badge = (f' <span class="badge ice">{"JÄÄTUV SADU" if is_ee else "FREEZING PRECIP"}</span>'
                    if w.is_freezing_wet else "")

    if w.is_freezing_wet:
        precip_sub = f'<br><span class="ice">⚠ {"Jäätuv / libe" if is_ee else "Freezing / glaze"}</span>'
    elif w.rain_amount == 0.0 and w.snow_amount == 0.0 and w.is_wet:
        precip_sub = f'<br><span style="color:var(--dim)">{"Nõrk tibutus" if is_ee else "Trace drizzle"}</span>'
    else:
        precip_sub = ""

    next_precip_sub = ""
    if w.forecast_steps and w.forecast_steps[0].total_precip_mm > 0:
        next_precip_sub = f'<br><span style="color:var(--dim)">3h: {E(_format_step_precip(w.forecast_steps[0]))}</span>'

    parts = [head]
    parts.append(f'<section class="card vibe" style="--c:{color}"><h1>{E(v_name)}</h1><p>{E(v_quote)}</p></section>')
    parts.append(f'''<section class="card now">
<div class="big"><b>{w.temp:+.1f}°C</b><span>{feels_label} {w.feels_like:+.1f}°C · {E(desc_disp)}{vis_inline}{freeze_badge}</span></div>
<div class="tile"><small>{wind_label}</small><span{wind_val_cls}>{w.wind_speed:.1f} m/s</span><br><span style="color:var(--dim)">{wind_sub}</span></div>
<div class="tile"><small>{hum_label}</small>{w.humidity}%<br><span style="color:var(--dim)">{dew_label} {w.dew_point:+.1f}°C</span></div>
<div class="tile"><small>{uv_label}</small>{uv}</div>
<div class="tile"><small>{rain_now_label}</small>{precip}{precip_sub}</div>
<div class="tile"><small>{rain_chance_label}</small>{w.rain_prob:.0f}%{next_precip_sub}</div>
<div class="tile"><small>{press_label}</small>{w.pressure} hPa</div>
</section>''')

    tw_matches = []
    tw_targets = (("Hommik", 11), ("Pärastlõuna", 14), ("Õhtu", 17)) if is_ee else (("Morning", 11), ("Afternoon", 14), ("Evening", 17))
    for label, hr in tw_targets:
        s = nearest_step(w.forecast_steps, hr)
        if s:
            tw_matches.append((s, label))

    tw_matches.sort(key=lambda pair: (pair[0].day_offset, pair[0].time_str))

    rows = []
    for s, label in tw_matches:
        day_prefix = ("Homme · " if is_ee else "Tomorrow · ") if s.day_offset > 0 else ""
        tw_label = f"{day_prefix}{label}"
        flag, precip_cls, _, extras = _step_context(s, is_ee)
        s_desc = _format_descs(s.effective_descs, s.desc, is_ee)
        detail_parts = [f"{tw_label} · {E(s_desc)}"]
        if extras:
            detail_parts.append(extras)
        rows.append(f'<div class="row tw"><b>{s.time_str}</b><span>{s.temp:+.1f}°</span>'
                    f'<span class="{precip_cls}">{int(s.pop*100)}% · {_format_step_precip(s)}</span>'
                    f'<span>{flag}</span><span class="d">{" · ".join(detail_parts)}</span></div>')
    if rows:
        tw_title = "Võtmeajad" if is_ee else "Target windows"
        parts.append(f'<section class="card"><h2>{tw_title}</h2>{"".join(rows)}</section>')

    frows = []
    prev_day = w.forecast_steps[0].day_offset if w.forecast_steps else 0
    for s in w.forecast_steps:
        if s.day_offset > prev_day:
            frows.append(f'<h3 class="day-sep">{"Homme" if is_ee else "Tomorrow"}</h3>')
            prev_day = s.day_offset
        pct = int(s.pop * 100)
        s_desc = _format_descs(s.effective_descs, s.desc, is_ee)
        _, _, bar_cls, extras = _step_context(s, is_ee)
        is_windy_row = s.wind_speed >= 10.0 or s.effective_wind_gust >= 13.0
        detail = E(s_desc) + (f" · {extras}" if extras else "")
        frows.append(f'<div class="row"><b>{s.time_str}</b><span>{s.temp:+.1f}°</span>'
                     f'<div class="{bar_cls}"><i style="width:{pct}%"></i><span>{pct}% · {_format_step_precip(s)}</span></div>'
                     f'<span class="wcol{" windy" if is_windy_row else ""}">{deg_to_cardinal(s.wind_deg)} {s.wind_speed:.1f}m/s</span>'
                     f'<span class="d">{detail}</span></div>')
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
        vis_label = "Nähtavus"
        lighting_disp = LIGHTING_ET.get(eph.lighting_condition, eph.lighting_condition)
    else:
        eph_title = "Ephemeris"
        daylight_label = "Daylight"
        dl_disp = eph.daylight_left_str
        clouds_label = "Clouds"
        vis_label = "Visibility"
        lighting_disp = eph.lighting_condition

    vis_row = f'\n<span>{vis_label}</span><span>{E(_vis_label(w.visibility, is_ee))}</span>' if w.visibility is not None else ""
    parts.append(f'''<section class="card"><h2>{eph_title}</h2><div class="kv">
<span>{daylight_label}</span><span>{E(dl_disp)}</span>
<span>{clouds_label}</span><span>{eph.cloud_pct}% · {E(lighting_disp)}</span>{vis_row}
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
        gust_val = w.effective_wind_gust
        has_gust = gust_val >= w.wind_speed + 2.0 and gust_val >= 11.0
        gust_kw = "iilid" if is_ee else "gusts"
        wind_str = f"{w.wind_speed:.0f} ({gust_kw} {gust_val:.0f}) m/s" if has_gust else f"{w.wind_speed:.0f} m/s"
        if is_ee:
            v_name, _ = get_ee_vibe(v.name)
        elif loc.region == "BE":
            v_name, _ = get_be_vibe(v.name)
        else:
            v_name = v.name

        desc_disp = _format_descs(w.effective_descs, w.desc, is_ee)
        if w.snow_amount > 0 and w.rain_amount > 0:
            precip_str = f"{w.rain_prob:.0f}% {'sadu' if is_ee else 'precip'} ({w.rain_amount:.1f}+❄{w.snow_amount:.1f} mm)"
        elif w.snow_amount > 0:
            precip_str = f"{w.rain_prob:.0f}% {'lumi' if is_ee else 'snow'} (❄{w.snow_amount:.1f} mm)"
        elif w.rain_amount > 0:
            precip_str = f"{w.rain_prob:.0f}% {'sadu' if is_ee else 'rain'} ({w.rain_amount:.1f} mm)"
        else:
            precip_str = f"{w.rain_prob:.0f}% {'vihma' if is_ee else 'rain'}"

        ov_parts = [E(desc_disp), wind_str, precip_str]
        if w.visibility is not None and w.visibility < 4000:
            ov_parts.append(f"{'näht.' if is_ee else 'vis'} {_format_vis(w.visibility)}")
        if w.is_freezing_wet:
            ov_parts.append(f'<span class="ice">{"Jäätuv sadu" if is_ee else "Freezing precip"}</span>')
        desc_text = " · ".join(ov_parts)

        feels_small = ""
        if abs(w.feels_like - w.temp) >= 2.5:
            feels_small = f'<small>({"tuntav" if is_ee else "feels"} {w.feels_like:+.0f}°)</small>'

        cards.append(f'''<section class="card ov"><a href="./{loc.region.lower()}/">
<div class="top"><span class="reg">{loc.flag} {loc.region}</span><span class="t">{w.temp:+.0f}°{feels_small}</span></div>
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
        "index.html": page("Nothing here", "<p style='color:#8b949e'>Nothing to see here.</p>", generated),
        "404.html": page("Not found", "<p style='color:#8b949e'>Not found.</p>", generated),
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
