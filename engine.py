"""
Headless weather + vibe engine for the static web dashboard.

Logic is identical to weatherlive20.py (rules, priorities, thresholds, data fetching), but this copy
is safe to publish: no coordinates, no place names finer than country level, no station names.
Locations are supplied at runtime (see build_site.py) and identified only by country code.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, List, Optional

import httpx

log = logging.getLogger("weatherweb")

OWM_BASE = "https://api.openweathermap.org/data/2.5"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_STEP_HOURS = 3.0  # OWM /forecast granularity

BE: FrozenSet[str] = frozenset({"BE"})
IT: FrozenSet[str] = frozenset({"IT"})
EE: FrozenSet[str] = frozenset({"EE"})
DEFAULT_REGION = "BE"

# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class AirQualityData:
    aqi: int
    pm2_5: float
    pm10: float

@dataclass
class ForecastStep:
    time_str: str
    temp: float
    pop: float         # 0.0 to 1.0
    rain_mm: float     # mm accumulated over the 3h step
    wind_speed: float  # m/s
    wind_deg: int
    desc: str
    day_offset: int = 0  # 0 for today (relative to local time), 1 for tomorrow

    @property
    def rain_rate(self) -> float:
        """Average intensity in mm/h, comparable with WeatherPayload.rain_amount."""
        return self.rain_mm / FORECAST_STEP_HOURS

@dataclass
class EphemerisData:
    daylight_left_str: str
    is_day: bool
    cloud_pct: int
    lighting_condition: str

@dataclass
class WeatherPayload:
    location_name: str            # web: always the region code, never a place name
    temp: float
    feels_like: float
    humidity: int
    pressure: int
    wind_speed: float
    wind_deg: int
    desc: str
    forecast_steps: List[ForecastStep]
    ephemeris: EphemerisData
    aqi_data: Optional[AirQualityData]
    rain_amount: float = 0.0      # mm current 1h rain
    rain_prob: float = 0.0        # % immediate precipitation probability
    uv_index: float = 0.0         # Current UV index
    local_hour: int = 12          # 0 - 23 local time at location
    snow_amount: float = 0.0      # mm current 1h snow
    uv_estimated: bool = False    # True when UV comes from the solar-geometry fallback
    country: str = ""             # ISO 3166 alpha-2 (e.g. "BE", "IT", "EE")

    @property
    def desc_lower(self) -> str:
        return self.desc.lower()

    @property
    def region(self) -> str:
        return self.country if self.country in SUPPORTED_REGIONS else DEFAULT_REGION

    @property
    def is_wet(self) -> bool:
        d = self.desc_lower
        return self.rain_amount > 0.0 or "rain" in d or "drizzle" in d or "shower" in d

    @property
    def is_night(self) -> bool:
        return self.local_hour > 20 or self.local_hour < 6

    @property
    def is_morning(self) -> bool:
        return 6 <= self.local_hour < 10

@dataclass(frozen=True)
class VibeRule:
    name: str
    color: str
    quote: str
    priority: int
    condition: Callable[[WeatherPayload], bool]
    regions: Optional[FrozenSet[str]] = None

    def applies_to(self, region: str) -> bool:
        return self.regions is None or region in self.regions

HEAVY_RAIN_DESCS = ("moderate rain", "heavy intensity rain", "very heavy rain", "extreme rain", "heavy intensity shower rain")

def _heavy_desc(w: WeatherPayload) -> bool:
    return any(k in w.desc_lower for k in HEAVY_RAIN_DESCS)

CARDINALS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")

def deg_to_cardinal(deg: int) -> str:
    idx = int((deg + 11.25) / 22.5) % 16
    return CARDINALS[idx]

def wind_from(w: WeatherPayload, lo: int, hi: int) -> bool:
    d = w.wind_deg % 360
    return lo <= d <= hi if lo <= hi else (d >= lo or d <= hi)

# =====================================================================
# VIBE ENGINE (mirrors weatherlive20.py; quotes generalised to country level)
# =====================================================================

VIBE_RULES: List[VibeRule] = [
    # 1. EXTREMES & POP CULTURE
    VibeRule("GHOSTBUSTERS", "bright_yellow", '"There is no Dana, only Zuul!" Unplug electronics.',
             priority=1000, condition=lambda w: "thunder" in w.desc_lower),
    VibeRule("THE EYEWALL", "bright_red", '"Hold onto your hats, your neighbors\' sheds, and your dignity."',
             priority=990, condition=lambda w: w.wind_speed > 20.0),
    VibeRule("HOTH", "bright_white", '"Your Tauntaun will freeze. Stay inside and expand the factory."',
             priority=980, condition=lambda w: "snow" in w.desc_lower or "sleet" in w.desc_lower or w.snow_amount > 0.0),
    VibeRule("THE STEAM ROOM", "khaki", '"Opaque AND sweaty. The worst kind of sauna."',
             priority=970, condition=lambda w: ("fog" in w.desc_lower or "mist" in w.desc_lower) and w.temp > 20.0),
    VibeRule("SILENT HILL", "light_slate_grey", '"Watch out for Pyramid Head."',
             priority=960, condition=lambda w: "fog" in w.desc_lower or "mist" in w.desc_lower),

    # 2. SEVERE PRECIPITATION
    VibeRule("TROPICAL DEPRESSION", "bright_blue", '"Warm and aggressively wet. Basically a jungle minus the exotic birds."',
             priority=955, condition=lambda w: w.temp >= 20.0 and (w.rain_amount > 4.0 or (w.is_wet and any(s.rain_rate > 4.0 for s in w.forecast_steps[:1])))),
    VibeRule("NOAH'S ARK", "bright_cyan", '"Might want to build that ark."',
             priority=950, condition=lambda w: w.rain_amount > 4.0 or (w.rain_prob > 85 and w.rain_amount > 2.0) or any(s.rain_rate > 4.0 or (s.pop > 0.8 and s.rain_rate > 2.0) for s in w.forecast_steps[:3])),
    VibeRule("THE LUKEWARM WASH", "bright_cyan", '"Like standing in a mildly neglected public shower."',
             priority=930, condition=lambda w: 16.0 <= w.temp <= 20.0 and (w.rain_amount > 2.0 or ("rain" in w.desc_lower and w.rain_amount > 1.0))),
    VibeRule("PROPER RAIN", "bright_blue", '"Get the umbrella. The heavy-duty one."',
             priority=920, condition=lambda w: 1.5 < w.rain_amount <= 4.0 or _heavy_desc(w)),

    # 3. EXTREME HEAT
    VibeRule("BOILED ALIVE", "bright_red", '"Remember that one time you hiked a volcano in the Philippines..."',
             priority=900, condition=lambda w: w.temp > 31.0 and w.humidity > 75),
    VibeRule("THE SURFACE OF THE SUN", "bright_red", '"The asphalt is melting. Stay inside or suffer."',
             priority=890, condition=lambda w: w.temp > 31.0),
    VibeRule("THE DRY HEAT", "bright_yellow", '"It\'s like a hair dryer is constantly blowing on your face. Moisturize now."',
             priority=880, condition=lambda w: w.temp > 25.0 and w.humidity < 30),
    VibeRule("THE SAUNA SESSION", "bright_yellow", '"You aren\'t breathing air as much as warm soup."',
             priority=870, condition=lambda w: 26.0 <= w.temp <= 31.0 and w.humidity > 80),
    VibeRule("THE DEADLY LASER", "bright_yellow", '"The sun is a deadly laser. Don\'t look directly at it."',
             priority=860, condition=lambda w: w.uv_index > 7.0 and w.temp > 25.0),

    # 4. EXTREME COLD & NIGHT CONDITIONS
    VibeRule("THE WHITE WALKER", "bright_cyan", '"Winter is here. Don\'t go outside unless you have Valyrian steel."',
             priority=855, condition=lambda w: w.is_night and w.temp < 0.0 and w.wind_speed > 5.0),
    VibeRule("BALTIC FREEZE", "bright_cyan", '"Nose hairs freezing on contact. Thermal base layers mandatory."',
             priority=850, condition=lambda w: w.temp <= -3.0),
    VibeRule("LIBECCIO", "bright_blue", '"The Libeccio is hurling the Tyrrhenian at the coast. Stay off the pier and tie down the ombrellone."',
             priority=845, regions=IT, condition=lambda w: wind_from(w, 200, 250) and w.wind_speed > 9.0),
    VibeRule("THE WIND TUNNEL", "bright_cyan", '"It\'s not just cold; the air is actively trying to peel the skin off your face."',
             priority=840, condition=lambda w: w.temp < 10.0 and w.wind_speed > 10.0),
    VibeRule("BONE CHILL", "bright_white", '"Excellent weather to stay indoors and stare at your Steam library."',
             priority=830, condition=lambda w: w.temp < 5.0 and w.humidity > 90 and not w.is_wet),
    VibeRule("SWAMP NIGHT", "sea_green2", '"Turn the fans on high and pray for morning."',
             priority=810, condition=lambda w: w.is_night and w.temp > 20.0 and w.humidity > 80),
    VibeRule("SÄÄSED", "dark_olive_green3", '"The true apex predator of the Estonian forest has hatched. Long sleeves, DEET, and accept your fate."',
             priority=806, regions=EE, condition=lambda w: 16.0 <= w.temp <= 28.0 and w.humidity > 70 and w.wind_speed < 2.5 and not w.is_wet and 17 <= w.local_hour <= 22),
    VibeRule("VALGED ÖÖD", "light_goldenrod1", '"White nights. It\'s past 21:00 and the sun refuses to clock out. Neither do the mosquitoes."',
             priority=805, regions=EE, condition=lambda w: w.is_night and w.ephemeris.is_day and not w.is_wet and w.temp > 8.0),
    VibeRule("THE RARE EVENING", "dark_violet", '"Actually pleasant after dark. A statistical anomaly."',
             priority=800, condition=lambda w: w.is_night and 19.0 <= w.temp <= 22.0 and w.humidity < 60 and not w.is_wet),
    VibeRule("MIDNIGHT MAGIC", "bright_magenta", '"The perfect night for a walk you\'ll regret by 2 AM. Stay hydrated."',
             priority=790, condition=lambda w: w.is_night and not w.is_wet and w.temp > 15.0),
    VibeRule("NIGHT CRAWLER", "dim white", '"Go out if you must, but consider the cost."',
             priority=780, condition=lambda w: w.is_night and not w.is_wet and 10.0 <= w.temp <= 15.0),
    VibeRule("PIMEDUS", "grey50", '"Pimedus: the darkness. It\'s mid-afternoon and pitch black. Vitamin D is now a personality trait."',
             priority=775, regions=EE, condition=lambda w: not w.ephemeris.is_day and 14 <= w.local_hour <= 19),
    VibeRule("COCOA TIME", "sandy_brown", '"Warm cup of chocolate and back to bed."',
             priority=770, condition=lambda w: w.is_night and not w.is_wet and 3.0 < w.temp < 10.0),
    VibeRule("SUITSUSAUN", "dark_orange", '"Smoke sauna weather. Fire up the suitsusaun, then roll in the snow (or the pond) like a true local."',
             priority=765, regions=EE, condition=lambda w: w.temp < 5.0 and 16 <= w.local_hour <= 23 and w.wind_speed < 10.0 and not w.is_wet),
    VibeRule("SCIROCCO", "orange3", '"Hot, sticky, Saharan dust on the car and everyone is irritable. It\'s not you, it\'s the Scirocco."',
             priority=760, regions=IT, condition=lambda w: wind_from(w, 100, 170) and w.wind_speed >= 3.0 and w.temp >= 18.0 and w.humidity >= 50 and not w.is_wet),
    VibeRule("TRAMONTANA", "bright_cyan", '"Crystal clear and freezing, straight down off the mountains. Even the statues are shivering."',
             priority=757, regions=IT, condition=lambda w: wind_from(w, 330, 30) and w.wind_speed >= 5.0 and w.temp < 14.0 and not w.is_wet),

    # 5. MORNING CONDITIONS
    VibeRule("EARLY OVEN", "bright_yellow", '"It\'s 8 AM and you are already sweating. Cancel all afternoon plans."',
             priority=750, condition=lambda w: w.is_morning and w.temp >= 22.0),
    VibeRule("MORNING SOUP", "bright_white", '"You can chew the air this morning. Coffee won\'t save you."',
             priority=740, condition=lambda w: w.is_morning and 17.0 <= w.temp < 22.0 and w.humidity > 80),
    VibeRule("THE OLYMPIC SPORT", "bright_blue", '"Getting a toddler into a car seat in this drizzle deserves a gold medal."',
             priority=730, condition=lambda w: w.is_morning and w.temp < 18.0 and w.is_wet),
    VibeRule("THE MORNING FROST", "bright_cyan", '"The only thing colder than the air is the realization that you have to be productive."',
             priority=720, condition=lambda w: w.is_morning and not w.is_wet and w.temp <= 3.0),
    VibeRule("BRISK MORROW", "bright_cyan", '"The air has a bite, but the coffee is hot. A productive day awaits."',
             priority=710, condition=lambda w: w.is_morning and not w.is_wet and 3.0 < w.temp < 18.0),

    # 6. HIGH-SPECIFICITY EDGE CASES
    VibeRule("THE UNICORN", "bright_magenta", '"Exactly 20 degrees and calm. Take a picture, it will be gone in 5 minutes."',
             priority=690, condition=lambda w: 19.8 <= w.temp <= 20.2 and not w.is_wet and w.wind_speed < 2.0),
    VibeRule("THE TERRARIUM", "bright_green", '"Welcome to the enclosure. You are the iguana now."',
             priority=680, condition=lambda w: 20.0 <= w.temp <= 24.0 and w.wind_speed < 1.5 and w.humidity > 70 and not w.is_wet),
    VibeRule("THE MUGGY MIDDLE", "bright_yellow", '"Not hot, not cold, just aggressively moist. Good time to stay in and chill a Burgundy."',
             priority=670, condition=lambda w: 18.0 <= w.temp <= 22.0 and w.humidity > 80 and not w.is_wet),
    VibeRule("SPITTING COBWEBS", "bright_white", '"It\'s humid enough to ruin your hair, but not wet enough to justify an umbrella."',
             priority=660, condition=lambda w: 18.0 <= w.temp <= 22.0 and 0.1 <= w.rain_amount <= 1.0 and w.wind_speed < 3.0),
    VibeRule("SNEAKY SUNBURN", "bright_red", '"The wind masks the heat. You will return looking like a boiled lobster."',
             priority=650, condition=lambda w: 17.0 <= w.temp <= 24.0 and w.uv_index > 5.0 and w.wind_speed > 5.0 and not w.is_wet),
    VibeRule("THE DECEPTIVE BREEZE", "bright_cyan", '"Looks perfect from the window. Outside, your patio furniture is plotting an escape."',
             priority=640, condition=lambda w: 17.0 <= w.temp <= 23.0 and not w.is_wet and w.wind_speed > 7.0),
    VibeRule("SCHRÖDINGER'S SHOWER", "bright_magenta", '"It might rain. It might not. Carry a jacket just to spite the atmosphere."',
             priority=630, condition=lambda w: 12.0 <= w.temp <= 22.0 and 35.0 <= w.rain_prob <= 65.0 and not w.is_wet),
    VibeRule("THE CRISP BAKE", "bright_yellow", '"Air is cool, sun is a laser beam. The ultimate thermostat confusion."',
             priority=620, condition=lambda w: 16.0 <= w.temp <= 20.0 and w.uv_index > 5.0 and w.wind_speed < 2.5 and not w.is_wet),

    # 7. THE COOL-TO-MILD PURGATORY (~10°C - 21°C)
    VibeRule("THE THREAT", "light_slate_grey", '"The sky is aggressively gray and looming, but doing absolutely nothing."',
             priority=600, condition=lambda w: 12.0 <= w.temp <= 19.0 and w.rain_prob > 60.0 and not w.is_wet),
    VibeRule("THE INVISIBLE HAND", "bright_cyan", '"Mother Nature is trying to physically shove you back indoors."',
             priority=590, condition=lambda w: 13.0 <= w.temp <= 21.0 and w.wind_speed > 14.0 and not w.is_wet),
    VibeRule("BAD HAIR DAY", "bright_yellow", '"Your hairgel has no power here."',
             priority=580, condition=lambda w: 13.0 <= w.temp <= 20.0 and w.wind_speed > 10.0 and not w.is_wet),
    VibeRule("FOOL'S SUMMER", "bright_red", '"Looks like July, feels like October. Don\'t pack the coats away yet."',
             priority=570, condition=lambda w: 14.0 <= w.temp <= 18.0 and w.uv_index > 4.5 and w.wind_speed < 3.0 and not w.is_wet),
    VibeRule("THE HORIZONTAL SHOWER", "bright_blue", '"Umbrellas are useless here. Embrace the suck."',
             priority=560, condition=lambda w: w.temp < 18.0 and w.wind_speed > 8.0 and w.is_wet),
    VibeRule("GARDEN WARRIOR", "bright_green", '"Crisp, clear, and calm. Prime weather to test out a heavy-duty shredder on some branches."',
             priority=550, condition=lambda w: 13.0 <= w.temp <= 16.0 and "clear" in w.desc_lower and w.wind_speed < 2.0),
    VibeRule("THE WIND CHILL", "bright_cyan", '"The sun is trying its best, but the wind hates you."',
             priority=540, condition=lambda w: 10.0 <= w.temp <= 16.0 and w.uv_index > 3.5 and w.wind_speed > 6.0 and not w.is_wet),
    VibeRule("PERPETUAL GLOOM", "dim white", '"It\'s barely raining, but your soul is still getting wet."',
             priority=530, condition=lambda w: 10.0 <= w.temp <= 15.0 and w.rain_prob > 75.0 and w.rain_amount < 0.5),
    VibeRule("THE VOID", "bright_white", '"Absolutely nothing is happening out there. Perfect stasis."',
             priority=520, condition=lambda w: 8.0 <= w.temp <= 13.0 and w.wind_speed < 1.0 and not w.is_wet),

    # 8. GENERAL PRECIPITATION & FALSE SEASONS
    VibeRule("THE SPONGE", "bright_yellow", '"Warm drizzle that instantly vaporizes and coats you in a fine layer of misery."',
             priority=500, condition=lambda w: 22.0 <= w.temp <= 31.0 and w.is_wet),
    VibeRule("THE BELGIAN DRIZZLE", "sky_blue2", '"It\'s not really raining, but you will still be soaked. Peak Belgium."',
             priority=490, regions=BE, condition=lambda w: w.temp < 22.0 and w.is_wet),
    VibeRule("LA PIOVIGGINE", "sky_blue2", '"Not enough rain to cancel the aperitivo, just enough to ruin your hair. Spritz under the portico."',
             priority=490, regions=IT, condition=lambda w: w.temp < 22.0 and w.is_wet),
    VibeRule("UDUVIHM", "sky_blue2", '"Uduvihm: literally \'fog-rain\'. Estonian has a word for it because it is always happening."',
             priority=490, regions=EE, condition=lambda w: w.temp < 22.0 and w.is_wet),
    VibeRule("THE MATTE FINISH", "light_slate_grey", '"No sun, no rain, dry air. The weather is buffering."',
             priority=480, condition=lambda w: 17.0 <= w.temp <= 22.0 and "cloud" in w.desc_lower and w.humidity < 50 and not w.is_wet),
    VibeRule("THE FALSE SPRING", "bright_green", '"Nature is gaslighting you. It looks warm, but the air will still bite."',
             priority=470, condition=lambda w: 3.0 < w.temp < 13.0 and w.uv_index > 3.0 and not w.is_wet),

    # 9. GOLDILOCKS & BASELINE CONDITIONS
    VibeRule("LA PASSEGGIATA", "gold1", '"Golden hour, gelato in hand, silently judging everyone\'s shoes. The passeggiata is mandatory."',
             priority=405, regions=IT, condition=lambda w: 18.0 <= w.temp <= 26.0 and w.wind_speed < 5.0 and not w.is_wet and 17 <= w.local_hour <= 21),
    VibeRule("THE GOLDILOCKS ZONE", "bright_green", '"Are you not entertained?! Nature is putting on a masterclass."',
             priority=400, condition=lambda w: 22.0 <= w.temp <= 25.0 and w.rain_prob < 15.0 and w.wind_speed <= 2.5 and not w.is_wet),
    VibeRule("TERRACE ESPRESSO", "bright_green", '"Ideal luminosity and gentle breeze. Step away from the monitor."',
             priority=395, condition=lambda w: 18.0 <= w.temp <= 25.0 and w.wind_speed < 4.5 and not w.is_wet and w.ephemeris.is_day and w.local_hour < 15),
    VibeRule("BIG GREEN EGG TERRITORY", "bright_yellow", '"Looks like meat\'s back on the menu, boys!"',
             priority=390, condition=lambda w: 18.0 <= w.temp <= 25.0 and w.rain_prob < 20.0 and w.wind_speed < 5.0 and not w.is_wet),
    VibeRule("OVERCAST VELVET", "light_slate_grey", '"Zero glare, uniform diffuse lighting. Optimal hacking state."',
             priority=387, condition=lambda w: "overcast" in w.desc_lower and not w.is_wet and w.wind_speed < 5.0 and 10.0 <= w.temp <= 25.0),
    VibeRule("THE PATIO GAMBLE", "khaki1", '"Lovely right now, but the forecast has trust issues. Eat outside, but sit near the door."',
             priority=385, condition=lambda w: 18.0 <= w.temp <= 25.0 and w.wind_speed < 5.0 and not w.is_wet),
    VibeRule("THE BRISK BREEZE", "bright_yellow", '"Nice temperature, but the wind is trying to steal your napkins."',
             priority=380, condition=lambda w: 17.0 <= w.temp <= 25.0 and w.wind_speed >= 5.0 and not w.is_wet),
    VibeRule("THE ROASTING SPIT", "bright_yellow", '"Hot enough to complain, not hot enough to melt. Stay in the shade."',
             priority=370, condition=lambda w: 25.0 < w.temp <= 31.0 and not w.is_wet),
    VibeRule("LAYERING SEASON", "dim white", '"A jacket is too much. No jacket is too little. Good luck."',
             priority=360, condition=lambda w: 10.0 <= w.temp < 18.0),
    VibeRule("THE BELGIAN STANDARD", "dim white", '"Grey, chilly, unapologetically mediocre. Peak Tuesday weather."',
             priority=350, regions=BE, condition=lambda w: 3.0 < w.temp < 10.0),
    VibeRule("CASTAGNE E CAMINO", "sandy_brown", '"Chestnuts on the fire, polenta on the stove. Hibernation protocol engaged."',
             priority=350, regions=IT, condition=lambda w: 3.0 < w.temp < 10.0),
    VibeRule("THE ESTONIAN STANDARD", "dim white", '"Grey, chilly, and silent. Locally described as \'normal\', if anything is said at all."',
             priority=350, regions=EE, condition=lambda w: 3.0 < w.temp < 10.0),
    VibeRule("WINTER IS COMING", "bright_cyan", '"The North remembers." Put on the thick woolen socks.',
             priority=340, condition=lambda w: w.temp <= 3.0),
]

DEFAULT_VIBES: Dict[str, VibeRule] = {
    "BE": VibeRule("FORREST GUMP", "bright_white", '"Belgian weather is like a box of chocolates."', 0, lambda w: True),
    "IT": VibeRule("FORREST GUMP", "bright_white", '"Italian weather is like a box of Baci."', 0, lambda w: True),
    "EE": VibeRule("FORREST GUMP", "bright_white", '"Estonian weather is like a box of Kalev chocolates."', 0, lambda w: True),
}
SUPPORTED_REGIONS: FrozenSet[str] = frozenset(DEFAULT_VIBES)

_SORTED_RULES = sorted(VIBE_RULES, key=lambda r: r.priority, reverse=True)
_RULES_BY_REGION = {reg: [r for r in _SORTED_RULES if r.applies_to(reg)] for reg in SUPPORTED_REGIONS}

def evaluate_vibe(payload: WeatherPayload) -> VibeRule:
    region = payload.region
    for rule in _RULES_BY_REGION[region]:
        try:
            if rule.condition(payload):
                return rule
        except Exception:
            log.exception("Vibe rule %r raised", rule.name)
    return DEFAULT_VIBES[region]

# =====================================================================
# SOLAR GEOMETRY (UV FALLBACK)
# =====================================================================

def solar_elevation_deg(lat: float, lon: float, ts: float) -> float:
    """NOAA low-precision solar position."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    g = 2.0 * math.pi / 365.0 * (dt.timetuple().tm_yday - 1 + (hour - 12.0) / 24.0)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
            - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
            - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                       - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    hour_angle = math.radians((hour * 60.0 + eqtime + 4.0 * lon) / 4.0 - 180.0)
    lat_r = math.radians(lat)
    cos_zen = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle)
    return 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zen))))

def estimate_uv(lat: float, lon: float, ts: float, cloud_pct: int) -> float:
    lat = round(lat, 1)
    lon = round(lon, 1)
    elev = solar_elevation_deg(lat, lon, ts)
    if elev <= 0.0:
        return 0.0
    clear_sky = 12.5 * math.sin(math.radians(elev)) ** 2.42
    return round(clear_sky * (1.0 - 0.75 * (cloud_pct / 100.0) ** 3.4), 1)

# =====================================================================
# DATA FETCHING
# =====================================================================

async def _optional_json(client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        res = await client.get(url, params=params)
        res.raise_for_status()
        return res.json()
    except Exception:
        return None

async def fetch_weather(client: httpx.AsyncClient, api_key: str, lat: float, lon: float, country: str) -> WeatherPayload:
    """Same data pipeline as weatherlive20's TelemetryController, without any naming of the location."""
    lat = round(lat, 1)
    lon = round(lon, 1)
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    om_params = {"latitude": lat, "longitude": lon, "current": "uv_index",
                 "hourly": "precipitation_probability", "forecast_hours": 1, "timezone": "GMT"}

    res_curr, res_fore, d_aqi, d_om = await asyncio.gather(
        client.get(f"{OWM_BASE}/weather", params=params),
        client.get(f"{OWM_BASE}/forecast", params=params),
        _optional_json(client, f"{OWM_BASE}/air_pollution", params),
        _optional_json(client, OPEN_METEO_URL, om_params),
        return_exceptions=True,
    )
    for res in (res_curr, res_fore):
        if isinstance(res, BaseException):
            raise res
        res.raise_for_status()
    d_curr, d_fore = res_curr.json(), res_fore.json()

    aqi_data = None
    if isinstance(d_aqi, dict):
        try:
            item = d_aqi["list"][0]
            aqi_data = AirQualityData(item["main"]["aqi"], item["components"].get("pm2_5", 0.0), item["components"].get("pm10", 0.0))
        except (KeyError, IndexError, TypeError):
            pass

    uv_val: Optional[float] = None
    om_pop: Optional[float] = None
    if isinstance(d_om, dict):
        try:
            uv_raw = d_om.get("current", {}).get("uv_index")
            if uv_raw is not None:
                uv_val = round(float(uv_raw), 1)
            pops = d_om.get("hourly", {}).get("precipitation_probability") or []
            if pops and pops[0] is not None:
                om_pop = float(pops[0])
        except (TypeError, ValueError, AttributeError):
            pass

    now_ts = time.time()
    sr = d_curr.get("sys", {}).get("sunrise", now_ts)
    ss = d_curr.get("sys", {}).get("sunset", now_ts + 86400)
    cloud_pct = d_curr.get("clouds", {}).get("all", 0)
    is_day = sr <= now_ts <= ss
    tz_offset = d_curr.get("timezone", 0)
    now_local_dt = datetime.fromtimestamp(now_ts + tz_offset, tz=timezone.utc)
    local_hour = now_local_dt.hour

    uv_estimated = uv_val is None
    if uv_val is None:
        uv_val = estimate_uv(lat, lon, now_ts, cloud_pct)

    if is_day:
        rem = max(0, ss - now_ts)
        daylight_str = f"{int(rem // 3600):02d}h {int((rem % 3600) // 60):02d}m"
        lighting = ("Gloomy / Low Diffuse Lux" if cloud_pct > 80 else
                    "Filtered Sun / Soft Glare" if cloud_pct > 40 else "Direct Bright Sun")
    else:
        daylight_str, lighting = "Night Cycle", "Darkness"
    ephemeris = EphemerisData(daylight_str, is_day, cloud_pct, lighting)

    steps: List[ForecastStep] = []
    for entry in d_fore.get("list", [])[:8]:
        step_dt = datetime.fromtimestamp(entry.get("dt", 0) + tz_offset, tz=timezone.utc)
        day_offset = (step_dt.date() - now_local_dt.date()).days

        def _step_precip(block_key: str) -> float:
            block = entry.get(block_key)
            if isinstance(block, dict):
                if "3h" in block:
                    return float(block["3h"])
                if "1h" in block:
                    return float(block["1h"]) * FORECAST_STEP_HOURS
            return 0.0

        rain_val = _step_precip("rain") + _step_precip("snow")
        steps.append(ForecastStep(
            time_str=step_dt.strftime("%H:%M"),
            temp=float(entry.get("main", {}).get("temp", 0.0)),
            pop=float(entry.get("pop", 0.0)),
            rain_mm=rain_val,
            wind_speed=float(entry.get("wind", {}).get("speed", 0.0)),
            wind_deg=int(entry.get("wind", {}).get("deg", 0)),
            desc=entry.get("weather", [{}])[0].get("description", "").title(),
            day_offset=day_offset,
        ))

    def _current_mm(key: str) -> float:
        block = d_curr.get(key)
        if isinstance(block, dict):
            if "1h" in block:
                return float(block["1h"])
            if "3h" in block:
                return float(block["3h"]) / FORECAST_STEP_HOURS
        return 0.0

    immediate_pop = om_pop if om_pop is not None else ((steps[0].pop * 100.0) if steps else 0.0)

    return WeatherPayload(
        location_name=country,
        temp=float(d_curr.get("main", {}).get("temp", 0.0)),
        feels_like=float(d_curr.get("main", {}).get("feels_like", 0.0)),
        humidity=int(d_curr.get("main", {}).get("humidity", 0)),
        pressure=int(d_curr.get("main", {}).get("pressure", 1013)),
        wind_speed=float(d_curr.get("wind", {}).get("speed", 0.0)),
        wind_deg=int(d_curr.get("wind", {}).get("deg", 0)),
        desc=d_curr.get("weather", [{}])[0].get("description", "Unknown"),
        forecast_steps=steps,
        ephemeris=ephemeris,
        aqi_data=aqi_data,
        rain_amount=_current_mm("rain"),
        rain_prob=immediate_pop,
        uv_index=uv_val,
        local_hour=local_hour,
        snow_amount=_current_mm("snow"),
        uv_estimated=uv_estimated,
        country=country,
    )

def nearest_step(steps: List[ForecastStep], target_hr: int, tolerance_hr: int = 1) -> Optional[ForecastStep]:
    def dist(s: ForecastStep) -> int:
        d = abs(int(s.time_str[:2]) - target_hr) % 24
        return min(d, 24 - d)
    best = min(steps, key=dist, default=None)
    return best if best is not None and dist(best) <= tolerance_hr else None
