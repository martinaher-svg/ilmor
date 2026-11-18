"""
Headless weather + synergy vibe engine for the static web dashboard.

Decomposes raw weather telemetry into orthogonal physical/atmospheric dimensions
(`AtmosphericProfile`) and resolves emergent multi-factor compound conditions
strictly BEFORE single-factor fallbacks within each severity band. Shares the
exact same wind, precipitation, fog, freezing, and thermal thresholds with
`classify_step()` for 3-hour forecast steps.

Safe to publish: no coordinates, no place names finer than country level, no
station names. Locations are supplied at runtime (see build_site.py) and
identified only by country code.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

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
# SHARED VOCABULARIES, WIND SCALE & TELEMETRY HELPERS
# =====================================================================

FACTOR_VOCAB: Tuple[str, ...] = (
    "thunder",
    "freezing",
    "snow",
    "sleet",
    "rain",
    "wind",
    "fog",
    "mist",
    "cold",
    "heat",
    "humidity",
    "dry",
    "uv",
    "sun",
    "clouds",
    "night",
    "morning",
    "air",
    "front",
)

SEVERITY_LEVELS: Tuple[str, ...] = ("calm", "advisory", "warning", "severe")

WIND_RANKS: Dict[str, int] = {
    "calm": 0,
    "light": 1,
    "breezy": 2,
    "strong": 3,
    "storm": 4,
    "gale": 5,
}

SYNERGY_LABELS: Dict[str, str] = {
    # Band A — Severe compound & extreme hazards
    "thunder_squall": "Thunder squall",
    "ice_storm": "Ice storm",
    "blizzard": "Blizzard",
    "wet_gale": "Wet gale",
    "arctic_gale": "Arctic gale",
    "wintry_gale": "Wintry gale",
    "freezing_fog": "Freezing fog",
    "thunderstorm": "Thunderstorm",
    "freezing_rain": "Freezing rain",
    "gale": "Gale",
    "mixed_precip": "Wintry mix",
    "snowfall": "Snowfall",
    # Band B — Active precip & visibility compounds + pure states
    "tropical_squall": "Tropical squall",
    "stormy_rain": "Stormy downpour",
    "libeccio": "Coastal SW gale",
    "cold_lashing_rain": "Cold lashing rain",
    "horizontal_rain": "Horizontal rain",
    "foggy_rain": "Fog-bound rain",
    "windy_fog": "Wind-driven fog",
    "smog_inversion": "Stagnant smog",
    "warm_fog": "Humid warm fog",
    "sunshower": "Sunshower",
    "warm_downpour": "Warm downpour",
    "cold_rain": "Raw cold rain",
    "misty_drizzle": "Misty drizzle",
    "warm_drizzle": "Warm humid rain",
    "mild_rain": "Mild steady rain",
    "cold_fog": "Cold dense fog",
    "heavy_rain": "Heavy rain",
    "moderate_rain": "Steady rain",
    "dense_fog": "Dense fog",
    "gloomy_drizzle": "Gloomy drizzle",
    "calm_drizzle": "Fine calm drizzle",
    "morning_rain": "Morning rain",
    "light_drizzle": "Light drizzle",
    # Band C — Dry atmospheric synergies
    "storm_approaching": "Storm approaching",
    "scorching_humid": "Oppressive humid heat",
    "scirocco": "Warm humid SE wind",
    "muggy_night": "Muggy night",
    "tropical_night": "Tropical night",
    "tramontana": "Cold mountain wind",
    "freezing_night_wind": "Sub-zero night wind",
    "arctic_windchill": "Arctic windchill",
    "damp_windchill": "Raw damp gale",
    "bitter_wind": "Biting cold wind",
    "sunny_frost": "Sunny frost",
    "starry_frost": "Clear frost night",
    "bitter_cold": "Deep freeze",
    "damp_cold": "Damp bone chill",
    "hot_dry_wind": "Hot dry wind",
    "extreme_heat": "Extreme heat",
    "arid_heat": "Dry still heat",
    "muggy_heat": "Still muggy heat",
    "breezy_heat": "Breezy heat",
    "high_uv_heat": "Intense UV heat",
    "storm_wind": "Dry storm wind",
    "strong_wind": "Strong dry wind",
    "windy_uv": "Breezy high UV",
    "sunny_windchill": "Sunny cold wind",
    "gusty_breeze": "Deceptive gusts",
    "looming_rain": "Looming rain clouds",
    "humid_evening": "Still humid evening",
    "white_night": "White night",
    "still_humid": "Still humid warmth",
    "muggy_mild": "Muggy mild air",
    "cool_high_uv": "Cool air, intense sun",
    "sunny_cool": "Bright cool illusion",
    "uncertain_showers": "Uncertain showers",
    # Band D & E — Diurnal, pleasant & breezy moods
    "warm_morning": "Hot morning",
    "humid_morning": "Humid morning",
    "frost_morning": "Frosty morning",
    "crisp_morning": "Crisp morning",
    "winter_darkness": "Mid-afternoon darkness",
    "crisp_sauna_evening": "Crisp cold evening",
    "pleasant_night": "Pleasant dry evening",
    "mild_night": "Mild dry night",
    "cool_night": "Cool night",
    "chilly_night": "Chilly night",
    "perfect_calm": "Calm 20°C equilibrium",
    "golden_evening": "Warm calm evening",
    "ideal_warmth": "Ideal calm warmth",
    "crisp_clear": "Crisp clear calm",
    "bright_chilly": "Sunny chill",
    "dry_cloudy": "Dry mild clouds",
    "bright_mild": "Bright gentle warmth",
    "calm_mild": "Dry calm warmth",
    "uncertain_patio": "Mild with shower risk",
    "dead_calm": "Cool dead calm",
    "soft_overcast": "Soft dry overcast",
    "brisk_breeze": "Brisk warm breeze",
    "hot_still": "Still heat",
}

STEP_KIND_LABELS: Dict[str, str] = {
    "thunder_squall": "Thunder squall",
    "thunderstorm": "Thunderstorm",
    "ice_storm": "Ice storm",
    "freezing_rain": "Freezing rain",
    "blizzard": "Blizzard",
    "wintry_gale": "Wintry gale",
    "wet_gale": "Wet gale",
    "arctic_gale": "Arctic gale",
    "freezing_fog": "Freezing fog",
    "stormy_rain": "Stormy rain",
    "cold_lashing_rain": "Cold lashing rain",
    "foggy_rain": "Fog-bound rain",
    "windy_fog": "Wind-driven fog",
    "mixed_precip": "Wintry mix",
    "cold_rain": "Raw cold rain",
    "misty_drizzle": "Misty drizzle",
    "bitter_wind": "Biting wind",
    "gale": "Gale",
    "heavy_rain": "Heavy rain",
    "heavy_snow": "Heavy snow",
    "dense_fog": "Dense fog",
    "bitter_cold": "Bitter cold",
    "hot_dry_wind": "Hot dry wind",
    "muggy_heat": "Muggy heat",
    "extreme_heat": "Extreme heat",
    "rain": "Rain",
    "snow": "Snow",
    "windy": "Strong wind",
    "calm": "Calm",
}

SEVERE_RAIN_DESCS = (
    "heavy intensity rain",
    "very heavy rain",
    "extreme rain",
    "heavy intensity shower rain",
    "heavy shower rain and drizzle",
)
HEAVY_SNOW_DESCS = ("heavy snow", "heavy shower snow", "heavy sleet")
MODERATE_RAIN_DESCS = ("moderate rain", "heavy intensity drizzle", "heavy intensity drizzle rain")
MIST_HAZE_TOKENS = ("mist", "haze", "smoke", "dust", "sand", "ash")


def compute_dew_point(temp_c: float, humidity: int) -> float:
    """Approximate dew point in °C using the Magnus-Tetens formula."""
    rh = max(1.0, min(100.0, float(humidity)))
    gamma = math.log(rh / 100.0) + (17.625 * temp_c) / (243.04 + temp_c)
    return round((243.04 * gamma) / (17.625 - gamma), 1)


def parse_weather_tags(weather_list: Any, default: str = "") -> Tuple[str, Tuple[str, ...]]:
    """Extract the primary OWM description plus deduplicated lowercase tags."""
    if not isinstance(weather_list, list):
        return (default, (default.lower(),)) if default else ("", ())
    primary = ""
    seen: List[str] = []
    for item in weather_list:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("description") or "").strip()
        if not raw:
            continue
        if not primary:
            primary = raw
        low = raw.lower()
        if low not in seen:
            seen.append(low)
    if not primary:
        return (default, (default.lower(),)) if default else ("", ())
    return primary, tuple(seen)


def _merge_descs(primary: str, extra: Tuple[str, ...]) -> Tuple[str, ...]:
    seen: List[str] = []
    for item in extra:
        low = item.strip().lower()
        if low and low not in seen:
            seen.append(low)
    p = primary.strip().lower()
    if p and p not in seen:
        seen.insert(0, p)
    return tuple(seen)


def _has_liquid_token(desc_blob: str) -> bool:
    if "rain" in desc_blob or "drizzle" in desc_blob:
        return True
    return "shower" in desc_blob and "snow" not in desc_blob and "sleet" not in desc_blob


def _has_sleet_or_mix(desc_blob: str, rain_mm: float, snow_mm: float) -> bool:
    if "sleet" in desc_blob:
        return True
    if rain_mm > 0.0 and snow_mm > 0.0:
        return True
    return "snow" in desc_blob and ("rain" in desc_blob or "drizzle" in desc_blob)


def _is_freezing_wet(temp: float, rain_mm: float, desc_blob: str) -> bool:
    has_liquid = rain_mm > 0.0 or _has_liquid_token(desc_blob)
    if "freezing" in desc_blob and (has_liquid or "fog" not in desc_blob):
        return True
    is_frozen_type = "snow" in desc_blob or "sleet" in desc_blob
    return temp <= 0.5 and has_liquid and not is_frozen_type


def _is_dense_fog(desc_blob: str, visibility: Optional[int]) -> bool:
    return "fog" in desc_blob or (visibility is not None and visibility < 1000)


def _is_mist_or_haze(desc_blob: str, visibility: Optional[int]) -> bool:
    if _is_dense_fog(desc_blob, visibility):
        return False
    if any(tok in desc_blob for tok in MIST_HAZE_TOKENS):
        return True
    return visibility is not None and visibility < 3000


def classify_wind_tier(wind_speed: float, wind_gust: float = 0.0) -> str:
    """Unified Beaufort-aligned wind tier using both sustained wind and peak gusts."""
    gust = max(wind_speed, wind_gust)
    if wind_speed >= 15.0 or gust >= 21.0 or (wind_speed >= 12.0 and gust >= 17.5):
        return "gale"
    if wind_speed >= 11.5 or gust >= 16.0 or (wind_speed >= 9.5 and gust >= 13.5):
        return "storm"
    if wind_speed >= 8.0 or gust >= 12.0 or (wind_speed >= 6.5 and gust >= 10.5):
        return "strong"
    if wind_speed >= 5.0 or gust >= 8.5:
        return "breezy"
    if wind_speed >= 2.0 or gust >= 4.0:
        return "light"
    return "calm"


def is_notable_gust(wind_speed: float, wind_gust: float) -> bool:
    """Shared gust-display threshold across detail, overview, and forecast steps."""
    gust = max(wind_speed, wind_gust)
    return gust >= 9.5 and (gust - wind_speed) >= 2.0


def is_wind_elevated(wind_speed: float, wind_gust: float = 0.0) -> bool:
    """True when wind reaches strong/storm/gale tier (shared UI emphasis predicate)."""
    return WIND_RANKS[classify_wind_tier(wind_speed, wind_gust)] >= WIND_RANKS["strong"]


CARDINALS = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)


def deg_to_cardinal(deg: int) -> str:
    idx = int((deg + 11.25) / 22.5) % 16
    return CARDINALS[idx]


def wind_from(w: WeatherPayload, lo: int, hi: int) -> bool:
    d = w.wind_deg % 360
    return lo <= d <= hi if lo <= hi else (d >= lo or d <= hi)


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
    rain_mm: float     # mm liquid rain/drizzle accumulated over the 3h step
    wind_speed: float  # m/s
    wind_deg: int
    desc: str
    day_offset: int = 0   # 0 for today (relative to local time), 1 for tomorrow
    wind_gust: float = 0.0  # m/s peak gust in the 3h step
    snow_mm: float = 0.0    # mm snow accumulated over the 3h step
    feels_like: Optional[float] = None
    humidity: Optional[int] = None
    visibility: Optional[int] = None
    all_descs: Tuple[str, ...] = ()

    @property
    def total_precip_mm(self) -> float:
        """Total precipitation (liquid + snow) in mm over the 3h step."""
        return self.rain_mm + self.snow_mm

    @property
    def rain_rate(self) -> float:
        """Average liquid rain intensity in mm/h, comparable with WeatherPayload.rain_amount."""
        return self.rain_mm / FORECAST_STEP_HOURS

    @property
    def effective_wind_gust(self) -> float:
        """Peak gust in m/s (at least wind_speed)."""
        return max(self.wind_speed, self.wind_gust)

    @property
    def effective_descs(self) -> Tuple[str, ...]:
        """Deduplicated lowercase condition tags for this step."""
        return _merge_descs(self.desc, self.all_descs)

    @property
    def desc_lower(self) -> str:
        """Combined lowercase condition string for tag matching."""
        return " ".join(self.effective_descs)

    @property
    def is_dense_fog(self) -> bool:
        """True when fog or <1000 m visibility is forecast."""
        return _is_dense_fog(self.desc_lower, self.visibility)

    @property
    def is_freezing_wet(self) -> bool:
        """True when liquid precipitation falls at or below 0.5°C or freezing rain is flagged."""
        return _is_freezing_wet(self.temp, self.rain_mm, self.desc_lower)


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
    wind_gust: float = 0.0        # m/s current wind gust
    visibility: Optional[int] = None   # meters (OWM caps at 10000)
    all_descs: Tuple[str, ...] = ()    # all current OWM weather tags (lowercase)

    @property
    def effective_wind_gust(self) -> float:
        """Peak gust (at least wind_speed)."""
        return max(self.wind_speed, self.wind_gust)

    @property
    def effective_descs(self) -> Tuple[str, ...]:
        """Deduplicated lowercase weather tags reported for current conditions."""
        return _merge_descs(self.desc, self.all_descs)

    @property
    def desc_lower(self) -> str:
        return self.desc.lower()

    @property
    def all_descs_lower(self) -> str:
        """All reported current condition tags joined for multi-tag checks."""
        return " ".join(self.effective_descs)

    @property
    def region(self) -> str:
        return self.country if self.country in SUPPORTED_REGIONS else DEFAULT_REGION

    @property
    def is_wet(self) -> bool:
        return self.rain_amount > 0.0 or _has_liquid_token(self.all_descs_lower)

    @property
    def dew_point(self) -> float:
        """Dew point in °C computed from air temperature and relative humidity."""
        return compute_dew_point(self.temp, self.humidity)

    @property
    def dew_spread(self) -> float:
        """Temperature minus dew point in °C."""
        return round(self.temp - self.dew_point, 1)

    @property
    def is_dense_fog(self) -> bool:
        """True when fog or <1000 m visibility is present."""
        return _is_dense_fog(self.all_descs_lower, self.visibility)

    @property
    def is_freezing_wet(self) -> bool:
        """True when freezing rain is reported or liquid precipitation falls at <= 0.5°C."""
        return _is_freezing_wet(self.temp, self.rain_amount, self.all_descs_lower)

    @property
    def is_night(self) -> bool:
        return self.local_hour > 20 or self.local_hour < 6

    @property
    def is_after_dark(self) -> bool:
        """True after local sunset or during late-night clock hours."""
        return self.is_night or not self.ephemeris.is_day

    @property
    def is_morning(self) -> bool:
        return 6 <= self.local_hour < 10


# =====================================================================
# ATMOSPHERIC PROFILE & SHARED FORECAST STEP CLASSIFIER
# =====================================================================


@dataclass(frozen=True)
class AtmosphericProfile:
    thermal: str           # deep_freeze | freezing | cold | cool | mild | warm | hot | scorching
    precip_kind: str       # none | thunder | freezing_rain | sleet | snow | rain
    precip_level: str      # none | trace | moderate | heavy
    wind: str              # calm | light | breezy | strong | storm | gale
    wind_rank: int         # 0..5 matching WIND_RANKS[wind]
    is_gusty: bool
    visibility_state: str  # dense_fog | mist | clear
    moisture: str          # arid | dry | normal | humid | saturated
    sky: str               # dark | blazing_sun | sunny | partly_cloudy | overcast
    diurnal: str           # morning | day | evening | night
    is_after_dark: bool
    is_white_night: bool
    air: str               # good | moderate | poor
    trend: str             # storm_approaching | rain_imminent | uncertain_showers | steady
    is_showery: bool
    is_precipitating: bool
    is_cold_air: bool
    is_subzero: bool
    wind_chill_gap: float


@dataclass(frozen=True)
class StepCondition:
    kind: str
    label: str
    factors: Tuple[str, ...]
    severity: str
    is_compound: bool


def _heavy_desc(w: WeatherPayload) -> bool:
    d = w.all_descs_lower
    if any(k in d for k in SEVERE_RAIN_DESCS):
        return True
    return "moderate rain" in d and w.rain_amount >= 1.0


def _is_gale(w: WeatherPayload) -> bool:
    """Severe storm / gale-force sustained wind or damaging gusts."""
    return classify_wind_tier(w.wind_speed, w.effective_wind_gust) == "gale"


def _is_storm_wind(w: WeatherPayload) -> bool:
    """Strong storm wind or higher."""
    return WIND_RANKS[classify_wind_tier(w.wind_speed, w.effective_wind_gust)] >= WIND_RANKS["storm"]


def _is_high_wind(w: WeatherPayload) -> bool:
    """Strong/blustery wind or higher."""
    return WIND_RANKS[classify_wind_tier(w.wind_speed, w.effective_wind_gust)] >= WIND_RANKS["strong"]


def _is_wet_windy(w: WeatherPayload) -> bool:
    """Rain/drizzle driven by strong wind or gusts (sideways rain)."""
    return w.is_wet and _is_high_wind(w)


def _step_has_severe_incoming(step: ForecastStep) -> bool:
    """True when an upcoming forecast step carries severe weather warranting a front warning."""
    d = step.desc_lower
    wind = classify_wind_tier(step.wind_speed, step.effective_wind_gust)
    if "thunder" in d or step.is_freezing_wet:
        return True
    if wind in ("storm", "gale"):
        return True
    if step.rain_rate > 4.0 or any(k in d for k in SEVERE_RAIN_DESCS):
        return True
    if step.snow_mm >= 2.5 or any(k in d for k in HEAVY_SNOW_DESCS):
        return True
    return False


def analyze_atmosphere(w: WeatherPayload) -> AtmosphericProfile:
    """Decompose raw weather telemetry into orthogonal physical dimensions."""
    d = w.all_descs_lower
    gust = w.effective_wind_gust
    wind = classify_wind_tier(w.wind_speed, gust)
    wind_rank = WIND_RANKS[wind]
    is_gusty = is_notable_gust(w.wind_speed, gust)

    # Thermal regime
    if w.temp <= -3.0 or (w.temp <= 1.0 and w.feels_like <= -6.0):
        thermal = "deep_freeze"
    elif w.temp <= 0.5:
        thermal = "freezing"
    elif w.temp < 5.0:
        thermal = "cold"
    elif w.temp < 12.0:
        thermal = "cool"
    elif w.temp < 19.0:
        thermal = "mild"
    elif w.temp <= 25.0:
        thermal = "warm"
    elif w.temp <= 31.0 and w.feels_like < 35.0:
        thermal = "hot"
    else:
        thermal = "scorching"

    # Precipitation phase
    if "thunder" in d:
        precip_kind = "thunder"
    elif w.is_freezing_wet:
        precip_kind = "freezing_rain"
    elif _has_sleet_or_mix(d, w.rain_amount, w.snow_amount):
        precip_kind = "sleet"
    elif "snow" in d or w.snow_amount > 0.0:
        precip_kind = "snow"
    elif w.is_wet:
        precip_kind = "rain"
    else:
        precip_kind = "none"

    is_precipitating = precip_kind != "none"
    is_showery = "shower" in d

    # Precipitation intensity
    if not is_precipitating:
        precip_level = "none"
    elif (
        w.rain_amount > 4.0
        or any(k in d for k in SEVERE_RAIN_DESCS)
        or (w.rain_prob > 85.0 and w.rain_amount > 2.5)
        or (w.rain_amount >= 1.5 and any(s.rain_rate > 4.0 for s in w.forecast_steps[:2]))
        or w.snow_amount > 2.0
        or any(k in d for k in HEAVY_SNOW_DESCS)
    ):
        precip_level = "heavy"
    elif (
        w.rain_amount >= 1.0
        or w.snow_amount >= 0.5
        or any(k in d for k in MODERATE_RAIN_DESCS)
        or ("shower rain" in d and "light" not in d)
        or (precip_kind in ("snow", "sleet") and "light" not in d)
    ):
        precip_level = "moderate"
    else:
        precip_level = "trace"

    # Visibility / optical obstruction
    if w.is_dense_fog:
        visibility_state = "dense_fog"
    elif _is_mist_or_haze(d, w.visibility):
        visibility_state = "mist"
    else:
        visibility_state = "clear"

    # Hygrometric / moisture regime
    if w.humidity >= 90 or w.dew_spread <= 1.5:
        moisture = "saturated"
    elif w.humidity > 75 or (w.temp >= 18.0 and w.dew_point >= 16.0):
        moisture = "humid"
    elif w.humidity < 30:
        moisture = "arid"
    elif w.humidity < 50:
        moisture = "dry"
    else:
        moisture = "normal"

    # Sky & solar state
    if not w.ephemeris.is_day:
        sky = "dark"
    elif w.uv_index > 7.0 and visibility_state != "dense_fog":
        sky = "blazing_sun"
    elif visibility_state == "dense_fog" or "overcast" in d or w.ephemeris.cloud_pct >= 80:
        sky = "overcast"
    elif "clear" in d or w.ephemeris.cloud_pct <= 35 or w.uv_index >= 4.0:
        sky = "sunny"
    else:
        sky = "partly_cloudy"

    # Diurnal context
    if w.is_night:
        diurnal = "night"
    elif w.is_morning:
        diurnal = "morning"
    elif 17 <= w.local_hour <= 20:
        diurnal = "evening"
    else:
        diurnal = "day"

    is_after_dark = w.is_after_dark
    is_white_night = w.is_night and w.ephemeris.is_day

    # Air quality
    if w.aqi_data is None:
        air = "good"
    elif w.aqi_data.aqi >= 4 or w.aqi_data.pm2_5 >= 35.0 or w.aqi_data.pm10 >= 75.0:
        air = "poor"
    elif w.aqi_data.aqi == 3 or w.aqi_data.pm2_5 >= 20.0 or w.aqi_data.pm10 >= 45.0:
        air = "moderate"
    else:
        air = "good"

    # Barometric & short-term forecast trajectory
    if (
        not is_precipitating
        and visibility_state != "dense_fog"
        and (
            any(_step_has_severe_incoming(s) for s in w.forecast_steps[:2])
            or (
                w.pressure <= 1000
                and w.rain_prob >= 60.0
                and (wind_rank >= WIND_RANKS["breezy"] or w.ephemeris.cloud_pct >= 75)
            )
        )
    ):
        trend = "storm_approaching"
    elif not is_precipitating and w.rain_prob >= 65.0:
        trend = "rain_imminent"
    elif not is_precipitating and 35.0 <= w.rain_prob < 65.0:
        trend = "uncertain_showers"
    else:
        trend = "steady"

    return AtmosphericProfile(
        thermal=thermal,
        precip_kind=precip_kind,
        precip_level=precip_level,
        wind=wind,
        wind_rank=wind_rank,
        is_gusty=is_gusty,
        visibility_state=visibility_state,
        moisture=moisture,
        sky=sky,
        diurnal=diurnal,
        is_after_dark=is_after_dark,
        is_white_night=is_white_night,
        air=air,
        trend=trend,
        is_showery=is_showery,
        is_precipitating=is_precipitating,
        is_cold_air=w.temp < 10.0,
        is_subzero=w.temp <= 0.5,
        wind_chill_gap=max(0.0, round(w.temp - w.feels_like, 1)),
    )


def _step_cond(kind: str, factors: Tuple[str, ...], severity: str, is_compound: bool) -> StepCondition:
    return StepCondition(
        kind=kind,
        label=STEP_KIND_LABELS[kind],
        factors=factors,
        severity=severity,
        is_compound=is_compound,
    )


def classify_step(step: ForecastStep) -> StepCondition:
    """Classify a 3-hour forecast step into a compound-first StepCondition."""
    d = step.desc_lower
    gust = step.effective_wind_gust
    wind = classify_wind_tier(step.wind_speed, gust)
    wind_rank = WIND_RANKS[wind]

    has_thunder = "thunder" in d
    is_freezing = step.is_freezing_wet
    is_sleet = _has_sleet_or_mix(d, step.rain_mm, step.snow_mm)
    is_snow = (not is_sleet) and (step.snow_mm > 0.0 or "snow" in d)
    has_rain = (not is_freezing) and (not is_sleet) and (not is_snow) and (
        step.rain_mm > 0.0 or _has_liquid_token(d)
    )
    is_heavy_rain = has_rain and (
        step.rain_rate > 4.0
        or any(k in d for k in SEVERE_RAIN_DESCS)
    )
    is_heavy_snow = is_snow and (
        step.snow_mm >= 3.0 or any(k in d for k in HEAVY_SNOW_DESCS)
    )
    is_dense_fog = step.is_dense_fog
    is_mist = _is_mist_or_haze(d, step.visibility)
    is_bitter_cold = (
        step.temp <= -3.0
        or (step.feels_like is not None and step.feels_like <= -6.0 and step.temp <= 1.0)
    )
    is_subzero = step.temp <= 0.5
    is_below_zero = step.temp <= 0.0
    hum = step.humidity

    # 1. Severe compound & extreme hazards
    if has_thunder:
        has_any_wet = step.rain_mm > 0.0 or _has_liquid_token(d)
        has_severe_wind = wind_rank >= WIND_RANKS["strong"]
        has_heavy_thunder_precip = step.rain_rate > 4.0 or any(k in d for k in SEVERE_RAIN_DESCS)
        has_frozen_thunder = any(tok in d for tok in ("snow", "sleet", "hail", "freezing"))
        if has_severe_wind or has_heavy_thunder_precip or has_frozen_thunder:
            sq_list: List[str] = ["thunder"]
            if has_severe_wind:
                sq_list.append("wind")
            if has_any_wet:
                sq_list.append("rain")
            if "snow" in d:
                sq_list.append("snow")
            elif "sleet" in d:
                sq_list.append("sleet")
            elif "freezing" in d:
                sq_list.append("freezing")
            if len(sq_list) < 2:
                sq_list.append("front")
            return _step_cond("thunder_squall", tuple(sq_list), "severe", True)
        factors = ("thunder", "rain") if has_any_wet else ("thunder",)
        return _step_cond("thunderstorm", factors, "severe", False)

    if is_freezing:
        if wind_rank >= WIND_RANKS["breezy"]:
            return _step_cond("ice_storm", ("freezing", "rain", "wind"), "severe", True)
        return _step_cond("freezing_rain", ("freezing", "rain"), "severe", True)

    if is_snow and wind_rank >= WIND_RANKS["strong"]:
        return _step_cond("blizzard", ("snow", "wind"), "severe", True)

    if is_sleet and wind_rank >= WIND_RANKS["strong"]:
        return _step_cond("wintry_gale", ("sleet", "wind", "cold"), "severe", True)

    if has_rain and (wind == "gale" or (wind == "storm" and is_heavy_rain)):
        return _step_cond("wet_gale", ("wind", "rain"), "severe", True)

    if wind == "gale" and (is_subzero or (step.feels_like is not None and step.feels_like <= -8.0)):
        return _step_cond("arctic_gale", ("wind", "cold", "freezing"), "severe", True)

    if wind == "gale":
        return _step_cond("gale", ("wind",), "severe", False)

    # 2. Active compound conditions
    if is_dense_fog and (is_below_zero or "freezing fog" in d) and not has_rain and not is_snow and not is_sleet:
        factors = ("freezing", "cold", "fog") if is_bitter_cold else ("freezing", "fog")
        return _step_cond("freezing_fog", factors, "warning", True)

    if is_sleet:
        return _step_cond("mixed_precip", ("sleet", "cold"), "warning", True)

    if has_rain and wind_rank >= WIND_RANKS["strong"]:
        if 0.5 < step.temp < 6.0:
            return _step_cond("cold_lashing_rain", ("cold", "rain", "wind"), "warning", True)
        return _step_cond("stormy_rain", ("wind", "rain"), "warning", True)

    if has_rain and is_heavy_rain and wind_rank >= WIND_RANKS["breezy"]:
        return _step_cond("stormy_rain", ("wind", "rain"), "warning", True)

    if has_rain and is_dense_fog:
        return _step_cond("foggy_rain", ("fog", "rain"), "warning", True)

    if is_dense_fog and wind_rank >= WIND_RANKS["strong"] and not is_snow:
        return _step_cond("windy_fog", ("fog", "wind"), "warning", True)

    if is_heavy_snow:
        return _step_cond("heavy_snow", ("snow",), "warning", False)

    if is_heavy_rain:
        return _step_cond("heavy_rain", ("rain",), "warning", False)

    if has_rain and 0.5 < step.temp < 6.0:
        return _step_cond("cold_rain", ("cold", "rain"), "advisory", True)

    if has_rain and is_mist and step.rain_rate < 1.5 and (step.visibility is None or step.visibility < 3000):
        return _step_cond("misty_drizzle", ("mist", "rain"), "advisory", True)

    if is_snow:
        return _step_cond("snow", ("snow",), "advisory", False)

    if has_rain:
        return _step_cond("rain", ("rain",), "advisory", False)

    if is_dense_fog:
        return _step_cond("dense_fog", ("fog",), "advisory", False)

    # 3. Dry thermal & wind compounds
    if wind_rank >= WIND_RANKS["strong"] and step.temp < 10.0:
        return _step_cond("bitter_wind", ("cold", "wind"), "warning", True)

    if is_bitter_cold:
        return _step_cond("bitter_cold", ("cold", "freezing"), "warning", False)

    if step.temp > 31.0 or (step.feels_like is not None and step.feels_like >= 35.0):
        if hum is not None and hum > 70:
            return _step_cond("muggy_heat", ("heat", "humidity"), "warning", True)
        return _step_cond("extreme_heat", ("heat",), "warning", False)

    if step.temp > 25.0 and hum is not None and hum < 35 and wind_rank >= WIND_RANKS["breezy"]:
        return _step_cond("hot_dry_wind", ("heat", "dry", "wind"), "advisory", True)

    if step.temp >= 25.0 and hum is not None and hum > 75:
        return _step_cond("muggy_heat", ("heat", "humidity"), "advisory", True)

    if wind_rank >= WIND_RANKS["strong"]:
        return _step_cond("windy", ("wind",), "advisory", False)

    if step.pop >= 0.65:
        if is_subzero:
            return _step_cond("snow", ("snow",), "advisory", False)
        return _step_cond("rain", ("rain",), "advisory", False)

    return _step_cond("calm", (), "calm", False)


# =====================================================================
# SYNERGY VIBE ENGINE
# =====================================================================


@dataclass(frozen=True)
class VibeRule:
    name: str
    color: str
    quote: str
    priority: int
    condition: Callable[[WeatherPayload], bool]
    regions: Optional[FrozenSet[str]] = None
    synergy_key: str = ""
    synergy_label: str = ""
    factors: Tuple[str, ...] = ()
    severity: str = "calm"
    is_compound: bool = False
    _profile_fn: Optional[Callable[[WeatherPayload, AtmosphericProfile], bool]] = None

    def applies_to(self, region: str) -> bool:
        return self.regions is None or region in self.regions

    def matches(self, payload: WeatherPayload, profile: Optional[AtmosphericProfile] = None) -> bool:
        if self._profile_fn is not None:
            prof = profile if profile is not None else analyze_atmosphere(payload)
            return self._profile_fn(payload, prof)
        return self.condition(payload)


def _rule(
    name: str,
    color: str,
    quote: str,
    priority: int,
    fn: Callable[[WeatherPayload, AtmosphericProfile], bool],
    *,
    regions: Optional[FrozenSet[str]] = None,
    synergy_key: str = "",
    factors: Tuple[str, ...] = (),
    severity: str = "calm",
    is_compound: bool = False,
) -> VibeRule:
    label = SYNERGY_LABELS[synergy_key] if synergy_key else ""
    return VibeRule(
        name=name,
        color=color,
        quote=quote,
        priority=priority,
        condition=lambda w: fn(w, analyze_atmosphere(w)),
        regions=regions,
        synergy_key=synergy_key,
        synergy_label=label,
        factors=factors,
        severity=severity,
        is_compound=is_compound,
        _profile_fn=fn,
    )


VIBE_RULES: List[VibeRule] = [
    # =================================================================
    # BAND A (1000–900): SEVERE COMPOUND HAZARDS -> PURE EXTREMES
    # =================================================================
    _rule(
        "SUPERCELL SQUALL", "bright_yellow",
        '"Lightning, lashing rain, and howling gusts all clocking in at once. Unplug everything and stay inside."',
        priority=995,
        fn=lambda w, p: (
            p.precip_kind == "thunder"
            and (
                p.wind_rank >= WIND_RANKS["strong"]
                or p.precip_level == "heavy"
                or any(tok in w.all_descs_lower for tok in ("snow", "sleet", "hail", "freezing"))
            )
        ),
        synergy_key="thunder_squall", factors=("thunder", "wind", "rain"), severity="severe", is_compound=True,
    ),
    _rule(
        "ICE STORM", "bright_red",
        '"Freezing rain driven by wind. Every branch, wire, and sidewalk is turning into glass."',
        priority=990,
        fn=lambda w, p: p.precip_kind == "freezing_rain" and p.wind_rank >= WIND_RANKS["breezy"],
        synergy_key="ice_storm", factors=("freezing", "rain", "wind"), severity="severe", is_compound=True,
    ),
    _rule(
        "BLIZZARD", "bright_white",
        '"Sideways snow and zero horizon. Even the snowplows are reconsidering their life choices."',
        priority=985,
        fn=lambda w, p: p.precip_kind == "snow" and p.wind_rank >= WIND_RANKS["strong"],
        synergy_key="blizzard", factors=("snow", "wind"), severity="severe", is_compound=True,
    ),
    _rule(
        "THE TEMPEST", "bright_red",
        '"Full gale-force wind hurling sheets of rain across the street. Battening down the hatches is no longer a metaphor."',
        priority=980,
        fn=lambda w, p: (
            p.precip_kind == "rain"
            and (p.wind == "gale" or (p.wind == "storm" and p.precip_level == "heavy"))
        ),
        synergy_key="wet_gale", factors=("wind", "rain"), severity="severe", is_compound=True,
    ),
    _rule(
        "ARCTIC GALE", "bright_cyan",
        '"Sub-zero air moving at hurricane ambition. Exposed skin has roughly ninety seconds."',
        priority=975,
        fn=lambda w, p: (
            not p.is_precipitating
            and p.wind == "gale"
            and (w.temp <= 0.5 or w.feels_like <= -8.0)
        ),
        synergy_key="arctic_gale", factors=("wind", "cold", "freezing"), severity="severe", is_compound=True,
    ),
    _rule(
        "WINTRY GALE", "bright_cyan",
        '"Wind-whipped sleet stinging like wet gravel. Peak atmospheric hostility."',
        priority=970,
        fn=lambda w, p: p.precip_kind == "sleet" and p.wind_rank >= WIND_RANKS["strong"],
        synergy_key="wintry_gale", factors=("sleet", "wind", "cold"), severity="severe", is_compound=True,
    ),
    _rule(
        "FROSTBOUND FOG", "bright_white",
        '"Sub-zero pea soup. Every branch, windshield, and eyelash is slowly plating in rime ice."',
        priority=965,
        fn=lambda w, p: (
            not p.is_precipitating
            and p.visibility_state == "dense_fog"
            and (w.temp <= 0.0 or "freezing fog" in w.all_descs_lower)
            and p.wind != "gale"
        ),
        synergy_key="freezing_fog", factors=("freezing", "fog"), severity="severe", is_compound=True,
    ),

    # Pure single-factor extremes (only reached when no second severe force combined above)
    _rule(
        "GHOSTBUSTERS", "bright_yellow",
        '"There is no Dana, only Zuul!" Unplug electronics.',
        priority=935,
        fn=lambda w, p: p.precip_kind == "thunder",
        synergy_key="thunderstorm", factors=("thunder",), severity="severe", is_compound=False,
    ),
    _rule(
        "BLACK ICE PROTOCOL", "bright_yellow",
        '"Liquid rain at freezing point. Every sidewalk is now an audition for Olympic figure skating."',
        priority=930,
        fn=lambda w, p: p.precip_kind == "freezing_rain",
        synergy_key="freezing_rain", factors=("freezing", "rain"), severity="severe", is_compound=True,
    ),
    _rule(
        "THE EYEWALL", "bright_red",
        '"Hold onto your hats, your neighbors\' sheds, and your dignity."',
        priority=925,
        fn=lambda w, p: not p.is_precipitating and p.wind == "gale",
        synergy_key="gale", factors=("wind",), severity="severe", is_compound=False,
    ),
    _rule(
        "SLUSH SEASON", "sky_blue2",
        '"Rain and snow arguing over which one gets to ruin your shoes first. Both are winning."',
        priority=920,
        fn=lambda w, p: p.precip_kind == "sleet",
        synergy_key="mixed_precip", factors=("sleet", "cold"), severity="warning", is_compound=True,
    ),
    _rule(
        "HOTH", "bright_white",
        '"Your Tauntaun will freeze. Stay inside and expand the factory."',
        priority=915,
        fn=lambda w, p: p.precip_kind == "snow",
        synergy_key="snowfall", factors=("snow",), severity="warning", is_compound=False,
    ),

    # =================================================================
    # BAND B (899–780): ACTIVE PRECIPITATION, VISIBILITY & SMOG SYNERGIES
    # Invariant: 100% of wet & dense-fog weather resolves by priority 780.
    # =================================================================
    _rule(
        "TROPICAL SQUALL", "bright_blue",
        '"Warm rain blasted sideways by storm gusts. A summer squall that ignores every umbrella."',
        priority=890,
        fn=lambda w, p: p.precip_kind == "rain" and w.temp >= 22.0 and p.wind_rank >= WIND_RANKS["strong"],
        synergy_key="tropical_squall", factors=("heat", "rain", "wind"), severity="warning", is_compound=True,
    ),
    _rule(
        "NORTH SEA LASHING", "bright_cyan",
        '"Near-freezing rain fired horizontally into your face. Too warm for snow, far too violent for mercy."',
        priority=887,
        fn=lambda w, p: p.precip_kind == "rain" and 0.5 < w.temp < 6.0 and p.wind_rank >= WIND_RANKS["strong"],
        synergy_key="cold_lashing_rain", factors=("cold", "rain", "wind"), severity="warning", is_compound=True,
    ),
    _rule(
        "STORMY DOWNPOUR", "bright_blue",
        '"Sheets of heavy rain angled by brisk gusts. Drains are losing and coats are surrendering."',
        priority=885,
        fn=lambda w, p: (
            p.precip_kind == "rain"
            and p.precip_level == "heavy"
            and p.wind_rank >= WIND_RANKS["breezy"]
        ),
        synergy_key="stormy_rain", factors=("rain", "wind"), severity="warning", is_compound=True,
    ),
    _rule(
        "LIBECCIO", "bright_blue",
        '"The Libeccio is hurling the Tyrrhenian at the coast. Stay off the pier and tie down the ombrellone."',
        priority=883,
        regions=IT,
        fn=lambda w, p: (
            wind_from(w, 200, 250)
            and p.wind_rank >= WIND_RANKS["strong"]
            and p.precip_kind in ("none", "rain")
            and p.precip_level != "heavy"
        ),
        synergy_key="libeccio", factors=("wind",), severity="warning", is_compound=False,
    ),
    _rule(
        "THE HORIZONTAL SHOWER", "bright_blue",
        '"Umbrellas are useless here. Embrace the suck."',
        priority=880,
        fn=lambda w, p: p.precip_kind == "rain" and 6.0 <= w.temp < 22.0 and p.wind_rank >= WIND_RANKS["strong"],
        synergy_key="horizontal_rain", factors=("rain", "wind"), severity="warning", is_compound=True,
    ),
    _rule(
        "FOGBOUND RAIN", "light_slate_grey",
        '"Dense fog soaked with falling rain. Zero visibility, 100% damp, pure maritime limbo."',
        priority=875,
        fn=lambda w, p: p.precip_kind == "rain" and p.visibility_state == "dense_fog",
        synergy_key="foggy_rain", factors=("fog", "rain"), severity="warning", is_compound=True,
    ),
    _rule(
        "SEA FOG DRIFT", "light_slate_grey",
        '"Thick fog racing past on strong gusts. Like standing inside a fast-moving cloud."',
        priority=872,
        fn=lambda w, p: (
            not p.is_precipitating
            and p.visibility_state == "dense_fog"
            and w.temp > 0.0
            and "freezing fog" not in w.all_descs_lower
            and p.wind_rank >= WIND_RANKS["strong"]
        ),
        synergy_key="windy_fog", factors=("fog", "wind"), severity="warning", is_compound=True,
    ),
    _rule(
        "SMOG TRAP", "orange3",
        '"Still air trapping particulates under a low lid. Keep the windows shut and skip the outdoor run."',
        priority=870,
        fn=lambda w, p: (
            not p.is_precipitating
            and p.wind_rank <= WIND_RANKS["light"]
            and p.air == "poor"
            and p.thermal not in ("scorching", "deep_freeze")
            and (
                (w.aqi_data is not None and w.aqi_data.aqi >= 5)
                or p.visibility_state in ("dense_fog", "mist")
                or w.pressure >= 1018
                or w.humidity >= 70
            )
        ),
        synergy_key="smog_inversion", factors=("air", "mist"), severity="warning", is_compound=True,
    ),
    _rule(
        "THE STEAM ROOM", "khaki",
        '"Opaque AND sweaty. The worst kind of sauna."',
        priority=868,
        fn=lambda w, p: (
            not p.is_precipitating
            and w.temp > 20.0
            and p.wind_rank < WIND_RANKS["strong"]
            and (
                p.visibility_state == "dense_fog"
                or ("mist" in w.all_descs_lower and (w.visibility is None or w.visibility < 3000))
            )
        ),
        synergy_key="warm_fog", factors=("fog", "heat", "humidity"), severity="advisory", is_compound=True,
    ),
    _rule(
        "SUNSHOWER", "gold1",
        '"Bright sunshine and falling raindrops at the same time. Look opposite the sun for the rainbow."',
        priority=865,
        fn=lambda w, p: (
            p.precip_kind == "rain"
            and w.ephemeris.is_day
            and p.precip_level != "heavy"
            and w.temp >= 6.0
            and w.ephemeris.cloud_pct <= 70
            and "overcast" not in w.all_descs_lower
            and (
                w.uv_index >= 3.0
                or (w.ephemeris.cloud_pct <= 55 and (p.is_showery or w.rain_amount <= 2.0))
            )
        ),
        synergy_key="sunshower", factors=("sun", "rain"), severity="advisory", is_compound=True,
    ),
    _rule(
        "TROPICAL DEPRESSION", "bright_blue",
        '"Warm and aggressively wet. Basically a jungle minus the exotic birds."',
        priority=855,
        fn=lambda w, p: (
            p.precip_kind == "rain"
            and w.temp >= 20.0
            and (p.precip_level == "heavy" or any(s.rain_rate > 4.0 for s in w.forecast_steps[:1]))
        ),
        synergy_key="warm_downpour", factors=("heat", "rain"), severity="warning", is_compound=True,
    ),
    _rule(
        "THE MARROW SOAK", "bright_blue",
        '"Too warm for snow, too cold to forgive. Raw rain that goes straight through wool and into your bones."',
        priority=850,
        fn=lambda w, p: p.precip_kind == "rain" and 0.5 < w.temp < 6.0 and p.precip_level != "heavy",
        synergy_key="cold_rain", factors=("cold", "rain"), severity="advisory", is_compound=True,
    ),
    _rule(
        "MISTY DRIZZLE", "sky_blue2",
        '"Hanging mist blurring into fine drizzle. Not quite rain, not quite fog—just damp air."',
        priority=845,
        fn=lambda w, p: (
            p.precip_kind == "rain"
            and w.temp >= 6.0
            and p.precip_level == "trace"
            and p.visibility_state == "mist"
            and (w.visibility is None or w.visibility < 3000)
        ),
        synergy_key="misty_drizzle", factors=("mist", "rain"), severity="advisory", is_compound=True,
    ),
    _rule(
        "THE SPONGE", "bright_yellow",
        '"Warm drizzle that instantly vaporizes and coats you in a fine layer of misery."',
        priority=840,
        fn=lambda w, p: p.precip_kind == "rain" and w.temp >= 22.0,
        synergy_key="warm_drizzle", factors=("heat", "rain", "humidity"), severity="advisory", is_compound=True,
    ),
    _rule(
        "THE LUKEWARM WASH", "bright_cyan",
        '"Like standing in a mildly neglected public shower."',
        priority=835,
        fn=lambda w, p: (
            p.precip_kind == "rain"
            and p.precip_level != "heavy"
            and 16.0 <= w.temp < 22.0
            and (w.rain_amount > 2.0 or ("rain" in w.all_descs_lower and w.rain_amount > 1.0))
        ),
        synergy_key="mild_rain", factors=("rain",), severity="advisory", is_compound=False,
    ),
    _rule(
        "CHILL FOG", "bright_white",
        '"Raw, bone-damp fog hovering just above freezing. Quiet, clammy, and unforgiving."',
        priority=830,
        fn=lambda w, p: not p.is_precipitating and p.visibility_state == "dense_fog" and 0.0 < w.temp < 5.0,
        synergy_key="cold_fog", factors=("cold", "fog"), severity="advisory", is_compound=True,
    ),

    # Pure active precipitation & fog fallbacks
    _rule(
        "NOAH'S ARK", "bright_cyan",
        '"Might want to build that ark."',
        priority=815,
        fn=lambda w, p: p.precip_kind == "rain" and p.precip_level == "heavy",
        synergy_key="heavy_rain", factors=("rain",), severity="warning", is_compound=False,
    ),
    _rule(
        "PROPER RAIN", "bright_blue",
        '"Get the umbrella. The heavy-duty one."',
        priority=810,
        fn=lambda w, p: p.precip_kind == "rain" and (w.rain_amount > 1.5 or _heavy_desc(w)),
        synergy_key="moderate_rain", factors=("rain",), severity="advisory", is_compound=False,
    ),
    _rule(
        "SILENT HILL", "light_slate_grey",
        '"Watch out for Pyramid Head."',
        priority=805,
        fn=lambda w, p: (
            not p.is_precipitating
            and w.temp <= 20.0
            and p.wind_rank < WIND_RANKS["strong"]
            and (
                p.visibility_state == "dense_fog"
                or (
                    w.temp >= 5.0
                    and (
                        ("mist" in w.all_descs_lower and (w.visibility is None or w.visibility < 2500))
                        or (w.visibility is not None and w.visibility < 2000)
                    )
                )
            )
        ),
        synergy_key="dense_fog", factors=("fog",), severity="advisory", is_compound=False,
    ),
    _rule(
        "PERPETUAL GLOOM", "dim white",
        '"It\'s barely raining, but your soul is still getting wet."',
        priority=795,
        fn=lambda w, p: (
            p.precip_kind == "rain"
            and 6.0 <= w.temp <= 15.0
            and w.rain_prob > 70.0
            and w.rain_amount < 0.5
            and w.ephemeris.cloud_pct >= 70
        ),
        synergy_key="gloomy_drizzle", factors=("rain", "clouds"), severity="advisory", is_compound=True,
    ),
    _rule(
        "SPITTING COBWEBS", "bright_white",
        '"It\'s humid enough to ruin your hair, but not wet enough to justify an umbrella."',
        priority=790,
        fn=lambda w, p: (
            p.precip_kind == "rain"
            and 16.0 <= w.temp < 22.0
            and w.rain_amount <= 1.0
            and p.wind_rank <= WIND_RANKS["light"]
            and w.wind_speed < 3.0
        ),
        synergy_key="calm_drizzle", factors=("rain", "humidity"), severity="advisory", is_compound=True,
    ),
    _rule(
        "THE OLYMPIC SPORT", "bright_blue",
        '"Getting a toddler into a car seat in this drizzle deserves a gold medal."',
        priority=785,
        fn=lambda w, p: p.precip_kind == "rain" and w.is_morning and 6.0 <= w.temp < 18.0,
        synergy_key="morning_rain", factors=("morning", "rain"), severity="advisory", is_compound=True,
    ),
    _rule(
        "THE BELGIAN DRIZZLE", "sky_blue2",
        '"It\'s not really raining, but you will still be soaked. Peak Belgium."',
        priority=780,
        regions=BE,
        fn=lambda w, p: p.precip_kind == "rain",
        synergy_key="light_drizzle", factors=("rain",), severity="advisory", is_compound=False,
    ),
    _rule(
        "LA PIOVIGGINE", "sky_blue2",
        '"Not enough rain to cancel the aperitivo, just enough to ruin your hair. Spritz under the portico."',
        priority=780,
        regions=IT,
        fn=lambda w, p: p.precip_kind == "rain",
        synergy_key="light_drizzle", factors=("rain",), severity="advisory", is_compound=False,
    ),
    _rule(
        "UDUVIHM", "sky_blue2",
        '"Uduvihm: literally \'fog-rain\'. Estonian has a word for it because it is always happening."',
        priority=780,
        regions=EE,
        fn=lambda w, p: p.precip_kind == "rain",
        synergy_key="light_drizzle", factors=("rain",), severity="advisory", is_compound=False,
    ),

    # =================================================================
    # BAND C (779–550): DRY ATMOSPHERIC SYNERGIES (FRONTS, HEAT, COLD, WIND, UV)
    # Guaranteed dry, non-dense-fog, non-gale.
    # =================================================================
    _rule(
        "STORM BREWING", "dark_violet",
        '"Eerily dry for now, but the barometer and radar are flashing red. Bring everything indoors."',
        priority=770,
        fn=lambda w, p: (
            p.trend == "storm_approaching"
            and p.wind_rank < WIND_RANKS["strong"]
            and p.thermal not in ("scorching", "deep_freeze")
        ),
        synergy_key="storm_approaching", factors=("front", "clouds"), severity="warning", is_compound=True,
    ),
    _rule(
        "BOILED ALIVE", "bright_red",
        '"Remember that one time you hiked a volcano in the Philippines..."',
        priority=765,
        fn=lambda w, p: w.temp > 31.0 and w.humidity > 70,
        synergy_key="scorching_humid", factors=("heat", "humidity"), severity="severe", is_compound=True,
    ),
    _rule(
        "SCIROCCO", "orange3",
        '"Hot, sticky, Saharan dust on the car and everyone is irritable. It\'s not you, it\'s the Scirocco."',
        priority=762,
        regions=IT,
        fn=lambda w, p: (
            wind_from(w, 100, 170)
            and w.wind_speed >= 3.0
            and 18.0 <= w.temp <= 31.0
            and w.humidity >= 50
            and p.wind_rank <= WIND_RANKS["strong"]
            and not (w.ephemeris.is_day and not w.is_night and w.uv_index > 7.0 and w.temp > 25.0)
        ),
        synergy_key="scirocco", factors=("heat", "humidity", "wind"), severity="advisory", is_compound=True,
    ),
    _rule(
        "SWAMP NIGHT", "sea_green2",
        '"Turn the fans on high and pray for morning."',
        priority=760,
        fn=lambda w, p: p.is_after_dark and w.temp > 20.0 and w.humidity > 75,
        synergy_key="muggy_night", factors=("heat", "humidity", "night"), severity="advisory", is_compound=True,
    ),
    _rule(
        "TROPICAL NIGHT", "dark_orange",
        '"Past sunset and still over 25 degrees. The walls are radiating back everything they absorbed all day."',
        priority=755,
        fn=lambda w, p: p.is_after_dark and w.temp > 25.0,
        synergy_key="tropical_night", factors=("heat", "night"), severity="advisory", is_compound=True,
    ),
    _rule(
        "THE WHITE WALKER", "bright_cyan",
        '"Winter is here. Don\'t go outside unless you have Valyrian steel."',
        priority=750,
        fn=lambda w, p: p.is_after_dark and w.temp < 0.0 and p.wind_rank >= WIND_RANKS["breezy"],
        synergy_key="freezing_night_wind", factors=("cold", "freezing", "wind", "night"), severity="warning", is_compound=True,
    ),
    _rule(
        "ARCTIC WINDCHILL", "bright_cyan",
        '"Deep sub-zero air sharpened by howling wind. Every exposed square inch is filing a complaint."',
        priority=745,
        fn=lambda w, p: p.thermal == "deep_freeze" and p.wind_rank >= WIND_RANKS["strong"],
        synergy_key="arctic_windchill", factors=("cold", "freezing", "wind"), severity="warning", is_compound=True,
    ),
    _rule(
        "TRAMONTANA", "bright_cyan",
        '"Crystal clear and freezing, straight down off the mountains. Even the statues are shivering."',
        priority=742,
        regions=IT,
        fn=lambda w, p: (
            wind_from(w, 330, 30)
            and p.wind_rank >= WIND_RANKS["breezy"]
            and w.temp < 14.0
            and w.ephemeris.cloud_pct <= 70
        ),
        synergy_key="tramontana", factors=("cold", "wind"), severity="warning", is_compound=True,
    ),
    _rule(
        "RAW DAMP GALE", "bright_white",
        '"Saturated coastal cold driven straight through your layers by strong gusts."',
        priority=738,
        fn=lambda w, p: (
            0.5 <= w.temp < 7.0
            and (w.humidity >= 85 or w.dew_spread <= 2.0)
            and p.wind_rank >= WIND_RANKS["strong"]
        ),
        synergy_key="damp_windchill", factors=("cold", "humidity", "wind"), severity="warning", is_compound=True,
    ),
    _rule(
        "THE WIND TUNNEL", "bright_cyan",
        '"It\'s not just cold; the air is actively trying to peel the skin off your face."',
        priority=735,
        fn=lambda w, p: w.temp < 10.0 and p.wind_rank >= WIND_RANKS["strong"],
        synergy_key="bitter_wind", factors=("cold", "wind"), severity="warning", is_compound=True,
    ),
    _rule(
        "BLUEBIRD FREEZE", "bright_cyan",
        '"Crystal-clear sky, sparkling frost, and crisp sub-zero calm. Cold, yes—but dazzling."',
        priority=730,
        fn=lambda w, p: (
            w.temp <= 0.5
            and w.ephemeris.is_day
            and w.ephemeris.cloud_pct <= 35
            and p.wind_rank <= WIND_RANKS["light"]
        ),
        synergy_key="sunny_frost", factors=("cold", "freezing", "sun"), severity="advisory", is_compound=True,
    ),
    _rule(
        "STARRY FREEZE", "bright_white",
        '"Clear dark sky radiating heat straight into space. Quiet, crisp, and frosting over fast."',
        priority=725,
        fn=lambda w, p: (
            p.is_after_dark
            and w.temp <= 1.0
            and w.ephemeris.cloud_pct <= 40
            and p.wind_rank <= WIND_RANKS["light"]
        ),
        synergy_key="starry_frost", factors=("cold", "freezing", "night"), severity="advisory", is_compound=True,
    ),
    _rule(
        "BALTIC FREEZE", "bright_cyan",
        '"Nose hairs freezing on contact. Thermal base layers mandatory."',
        priority=720,
        fn=lambda w, p: p.thermal == "deep_freeze",
        synergy_key="bitter_cold", factors=("cold", "freezing"), severity="warning", is_compound=False,
    ),
    _rule(
        "BONE CHILL", "bright_white",
        '"Excellent weather to stay indoors and stare at your Steam library."',
        priority=715,
        fn=lambda w, p: w.temp < 5.0 and p.moisture == "saturated",
        synergy_key="damp_cold", factors=("cold", "humidity"), severity="advisory", is_compound=True,
    ),

    # Daytime/evening heat x humidity x wind x UV synergies (strictly ordered by severity)
    _rule(
        "THE SURFACE OF THE SUN", "bright_red",
        '"The asphalt is melting. Stay inside or suffer."',
        priority=696,
        fn=lambda w, p: not p.is_after_dark and w.temp > 31.0,
        synergy_key="extreme_heat", factors=("heat",), severity="severe", is_compound=False,
    ),
    _rule(
        "HOT DRY WIND", "dark_orange",
        '"Like standing directly in front of a giant hair dryer. Hot, bone-dry gusts pulling moisture out of everything."',
        priority=692,
        fn=lambda w, p: (
            not p.is_after_dark
            and 25.0 < w.temp <= 31.0
            and w.humidity < 35
            and p.wind_rank >= WIND_RANKS["breezy"]
        ),
        synergy_key="hot_dry_wind", factors=("heat", "dry", "wind"), severity="warning", is_compound=True,
    ),
    _rule(
        "THE DEADLY LASER", "bright_yellow",
        '"The sun is a deadly laser. Don\'t look directly at it."',
        priority=690,
        fn=lambda w, p: not p.is_after_dark and w.uv_index > 7.0 and w.temp > 25.0,
        synergy_key="high_uv_heat", factors=("heat", "uv", "sun"), severity="warning", is_compound=True,
    ),
    _rule(
        "THE DRY HEAT", "bright_yellow",
        '"Baked, bone-dry stillness. Drink twice as much water as you think you need."',
        priority=688,
        fn=lambda w, p: (
            not p.is_after_dark
            and 25.0 < w.temp <= 31.0
            and w.humidity < 30
            and p.wind_rank <= WIND_RANKS["light"]
        ),
        synergy_key="arid_heat", factors=("heat", "dry"), severity="advisory", is_compound=True,
    ),
    _rule(
        "THE SAUNA SESSION", "bright_yellow",
        '"You aren\'t breathing air as much as warm soup."',
        priority=684,
        fn=lambda w, p: (
            not p.is_after_dark
            and 25.0 < w.temp <= 31.0
            and w.humidity > 75
            and p.wind_rank <= WIND_RANKS["light"]
        ),
        synergy_key="muggy_heat", factors=("heat", "humidity"), severity="advisory", is_compound=True,
    ),
    _rule(
        "BREEZY HEAT", "gold1",
        '"Hot sun tempered by a steady warm wind. Find some shade and let the breeze do the heavy lifting."',
        priority=680,
        fn=lambda w, p: not p.is_after_dark and 25.0 < w.temp <= 31.0 and p.wind_rank >= WIND_RANKS["breezy"],
        synergy_key="breezy_heat", factors=("heat", "wind"), severity="advisory", is_compound=True,
    ),

    # Dry mild/cool wind, solar-chill, humidity & front-building synergies
    _rule(
        "THE INVISIBLE HAND", "bright_cyan",
        '"Mother Nature is trying to physically shove you back indoors."',
        priority=615,
        fn=lambda w, p: 10.0 <= w.temp <= 25.0 and p.wind == "storm",
        synergy_key="storm_wind", factors=("wind",), severity="warning", is_compound=False,
    ),
    _rule(
        "BAD HAIR DAY", "bright_yellow",
        '"Your hairgel has no power here."',
        priority=610,
        fn=lambda w, p: 10.0 <= w.temp <= 25.0 and p.wind == "strong",
        synergy_key="strong_wind", factors=("wind",), severity="advisory", is_compound=False,
    ),
    _rule(
        "SNEAKY SUNBURN", "bright_red",
        '"The wind masks the heat. You will return looking like a boiled lobster."',
        priority=605,
        fn=lambda w, p: w.ephemeris.is_day and 17.0 <= w.temp <= 25.0 and w.uv_index > 5.0 and p.wind == "breezy",
        synergy_key="windy_uv", factors=("uv", "sun", "wind"), severity="advisory", is_compound=True,
    ),
    _rule(
        "THE WIND CHILL", "bright_cyan",
        '"The sun is trying its best, but the wind hates you."',
        priority=600,
        fn=lambda w, p: (
            w.ephemeris.is_day
            and w.ephemeris.cloud_pct <= 75
            and 8.0 <= w.temp < 17.0
            and w.uv_index > 3.5
            and p.wind == "breezy"
        ),
        synergy_key="sunny_windchill", factors=("sun", "cold", "wind"), severity="advisory", is_compound=True,
    ),
    _rule(
        "THE DECEPTIVE BREEZE", "bright_cyan",
        '"Looks perfect from the window. Outside, your patio furniture is plotting an escape."',
        priority=595,
        fn=lambda w, p: (
            12.0 <= w.temp <= 25.0
            and p.wind == "breezy"
            and (w.wind_speed >= 6.5 or w.effective_wind_gust >= 10.5)
        ),
        synergy_key="gusty_breeze", factors=("wind",), severity="advisory", is_compound=False,
    ),
    _rule(
        "THE THREAT", "light_slate_grey",
        '"The sky is aggressively gray and looming, but doing absolutely nothing."',
        priority=590,
        fn=lambda w, p: (
            5.0 <= w.temp <= 25.0
            and w.rain_prob >= 65.0
            and (w.ephemeris.cloud_pct >= 65 or w.pressure <= 1005)
        ),
        synergy_key="looming_rain", factors=("front", "clouds"), severity="advisory", is_compound=True,
    ),
    _rule(
        "SÄÄSED", "dark_olive_green3",
        '"The true apex predator of the Estonian forest has hatched. Long sleeves, DEET, and accept your fate."',
        priority=585,
        regions=EE,
        fn=lambda w, p: 17 <= w.local_hour <= 22 and 16.0 <= w.temp <= 28.0 and w.humidity > 70 and p.wind == "calm",
        synergy_key="humid_evening", factors=("humidity", "night"), severity="advisory", is_compound=True,
    ),
    _rule(
        "VALGED ÖÖD", "light_goldenrod1",
        '"White nights. It\'s past 21:00 and the sun refuses to clock out. Neither do the mosquitoes."',
        priority=580,
        regions=EE,
        fn=lambda w, p: p.is_white_night and w.temp > 8.0 and p.wind_rank <= WIND_RANKS["breezy"],
        synergy_key="white_night", factors=("sun", "night"), severity="calm", is_compound=True,
    ),
    _rule(
        "THE TERRARIUM", "bright_green",
        '"Welcome to the enclosure. You are the iguana now."',
        priority=575,
        fn=lambda w, p: (
            not w.is_morning
            and not p.is_after_dark
            and 20.0 <= w.temp <= 25.0
            and p.wind == "calm"
            and w.humidity > 70
        ),
        synergy_key="still_humid", factors=("heat", "humidity"), severity="advisory", is_compound=True,
    ),
    _rule(
        "THE MUGGY MIDDLE", "bright_yellow",
        '"Not hot, not cold, just aggressively moist. Good time to stay in and chill a Burgundy."',
        priority=570,
        fn=lambda w, p: (
            not w.is_morning
            and not p.is_after_dark
            and 17.0 <= w.temp <= 22.0
            and w.humidity > 80
            and p.wind_rank <= WIND_RANKS["light"]
        ),
        synergy_key="muggy_mild", factors=("humidity",), severity="advisory", is_compound=False,
    ),
    _rule(
        "THE CRISP BAKE", "bright_yellow",
        '"Air is cool, sun is a laser beam. The ultimate thermostat confusion."',
        priority=565,
        fn=lambda w, p: (
            w.ephemeris.is_day
            and w.ephemeris.cloud_pct <= 75
            and 16.5 <= w.temp <= 20.0
            and w.uv_index > 5.0
            and p.wind_rank <= WIND_RANKS["light"]
            and w.wind_speed < 3.0
        ),
        synergy_key="cool_high_uv", factors=("sun", "uv"), severity="advisory", is_compound=True,
    ),
    _rule(
        "FOOL'S SUMMER", "bright_red",
        '"Looks like July, feels like October. Don\'t pack the coats away yet."',
        priority=560,
        fn=lambda w, p: (
            w.ephemeris.is_day
            and w.ephemeris.cloud_pct <= 75
            and 13.0 <= w.temp < 16.5
            and w.uv_index > 4.0
            and p.wind_rank <= WIND_RANKS["light"]
            and w.wind_speed < 3.5
        ),
        synergy_key="sunny_cool", factors=("sun", "cold"), severity="calm", is_compound=True,
    ),
    _rule(
        "SCHRÖDINGER'S SHOWER", "bright_magenta",
        '"It might rain. It might not. Carry a jacket just to spite the atmosphere."',
        priority=555,
        fn=lambda w, p: 10.0 <= w.temp <= 24.0 and 35.0 <= w.rain_prob < 65.0 and p.wind_rank <= WIND_RANKS["light"],
        synergy_key="uncertain_showers", factors=("front", "clouds"), severity="advisory", is_compound=True,
    ),

    # =================================================================
    # BAND D (549–400): DIURNAL (MORNING / DARK / NIGHT) & PLEASANT MOODS
    # =================================================================
    _rule(
        "EARLY OVEN", "bright_yellow",
        '"It\'s 8 AM and you are already sweating. Cancel all afternoon plans."',
        priority=545,
        fn=lambda w, p: w.is_morning and w.temp >= 22.0 and p.wind_rank <= WIND_RANKS["breezy"],
        synergy_key="warm_morning", factors=("morning", "heat"), severity="advisory", is_compound=True,
    ),
    _rule(
        "MORNING SOUP", "bright_white",
        '"You can chew the air this morning. Coffee won\'t save you."',
        priority=540,
        fn=lambda w, p: w.is_morning and 16.0 <= w.temp < 22.0 and w.humidity > 75 and p.wind_rank <= WIND_RANKS["light"],
        synergy_key="humid_morning", factors=("morning", "humidity"), severity="advisory", is_compound=True,
    ),
    _rule(
        "THE MORNING FROST", "bright_cyan",
        '"The only thing colder than the air is the realization that you have to be productive."',
        priority=535,
        fn=lambda w, p: w.is_morning and w.temp <= 3.0,
        synergy_key="frost_morning", factors=("morning", "cold", "freezing"), severity="advisory", is_compound=True,
    ),
    _rule(
        "BRISK MORROW", "bright_cyan",
        '"The air has a bite, but the coffee is hot. A productive day awaits."',
        priority=530,
        fn=lambda w, p: w.is_morning and 3.0 < w.temp < 18.0 and p.wind_rank <= WIND_RANKS["breezy"],
        synergy_key="crisp_morning", factors=("morning", "cold"), severity="calm", is_compound=True,
    ),

    # Regional EE darkness / sauna evening & general night rules
    _rule(
        "PIMEDUS", "grey50",
        '"Pimedus: the darkness. It\'s mid-afternoon and pitch black. Vitamin D is now a personality trait."',
        priority=505,
        regions=EE,
        fn=lambda w, p: not w.ephemeris.is_day and 14 <= w.local_hour <= 17 and p.wind_rank <= WIND_RANKS["breezy"],
        synergy_key="winter_darkness", factors=("night", "clouds"), severity="calm", is_compound=True,
    ),
    _rule(
        "SUITSUSAUN", "dark_orange",
        '"Smoke sauna weather. Fire up the suitsusaun, then roll in the snow (or the pond) like a true local."',
        priority=500,
        regions=EE,
        fn=lambda w, p: w.temp < 5.0 and 18 <= w.local_hour <= 23 and p.wind_rank <= WIND_RANKS["light"],
        synergy_key="crisp_sauna_evening", factors=("cold", "night"), severity="calm", is_compound=True,
    ),
    _rule(
        "THE RARE EVENING", "dark_violet",
        '"Actually pleasant after dark. A statistical anomaly."',
        priority=495,
        fn=lambda w, p: p.is_after_dark and 19.0 <= w.temp <= 22.0 and w.humidity < 60 and p.wind_rank <= WIND_RANKS["light"],
        synergy_key="pleasant_night", factors=("night", "dry"), severity="calm", is_compound=True,
    ),
    _rule(
        "MIDNIGHT MAGIC", "bright_magenta",
        '"The perfect night for a walk you\'ll regret by 2 AM. Stay hydrated."',
        priority=490,
        fn=lambda w, p: p.is_after_dark and w.temp > 15.0 and p.wind_rank <= WIND_RANKS["breezy"],
        synergy_key="mild_night", factors=("night",), severity="calm", is_compound=False,
    ),
    _rule(
        "NIGHT CRAWLER", "dim white",
        '"Go out if you must, but consider the cost."',
        priority=485,
        fn=lambda w, p: p.is_after_dark and 10.0 <= w.temp <= 15.0 and p.wind_rank <= WIND_RANKS["breezy"],
        synergy_key="cool_night", factors=("night",), severity="calm", is_compound=False,
    ),
    _rule(
        "COCOA TIME", "sandy_brown",
        '"Warm cup of chocolate and back to bed."',
        priority=480,
        fn=lambda w, p: p.is_after_dark and 0.5 < w.temp < 10.0 and p.wind_rank <= WIND_RANKS["breezy"],
        synergy_key="chilly_night", factors=("cold", "night"), severity="calm", is_compound=True,
    ),

    # Pleasant & atmospheric calm daytime moods (all gust-checked!)
    _rule(
        "THE UNICORN", "bright_magenta",
        '"Exactly 20 degrees and calm. Take a picture, it will be gone in 5 minutes."',
        priority=445,
        fn=lambda w, p: 19.8 <= w.temp <= 20.2 and p.wind == "calm",
        synergy_key="perfect_calm", factors=(), severity="calm", is_compound=False,
    ),
    _rule(
        "LA PASSEGGIATA", "gold1",
        '"Golden hour, gelato in hand, silently judging everyone\'s shoes. The passeggiata is mandatory."',
        priority=440,
        regions=IT,
        fn=lambda w, p: (
            w.ephemeris.is_day
            and w.ephemeris.cloud_pct <= 75
            and 17 <= w.local_hour <= 21
            and 18.0 <= w.temp <= 26.0
            and p.wind_rank <= WIND_RANKS["light"]
        ),
        synergy_key="golden_evening", factors=("sun",), severity="calm", is_compound=False,
    ),
    _rule(
        "THE GOLDILOCKS ZONE", "bright_green",
        '"Are you not entertained?! Nature is putting on a masterclass."',
        priority=435,
        fn=lambda w, p: (
            w.ephemeris.is_day
            and 21.0 <= w.temp <= 25.0
            and w.rain_prob < 15.0
            and p.wind_rank <= WIND_RANKS["light"]
            and w.wind_speed <= 2.5
            and w.ephemeris.cloud_pct <= 75
        ),
        synergy_key="ideal_warmth", factors=("sun",), severity="calm", is_compound=False,
    ),
    _rule(
        "GARDEN WARRIOR", "bright_green",
        '"Crisp, clear, and calm. Prime weather to test out a heavy-duty shredder on some branches."',
        priority=430,
        fn=lambda w, p: (
            w.ephemeris.is_day
            and w.ephemeris.cloud_pct <= 75
            and 12.0 <= w.temp < 16.5
            and "clear" in w.all_descs_lower
            and p.wind == "calm"
            and w.uv_index <= 4.0
        ),
        synergy_key="crisp_clear", factors=("sun",), severity="calm", is_compound=False,
    ),
    _rule(
        "THE FALSE SPRING", "bright_green",
        '"Nature is gaslighting you. It looks warm, but the air will still bite."',
        priority=425,
        fn=lambda w, p: (
            w.ephemeris.is_day
            and w.ephemeris.cloud_pct <= 75
            and 3.0 < w.temp < 13.0
            and w.uv_index > 3.0
            and p.wind_rank <= WIND_RANKS["light"]
        ),
        synergy_key="bright_chilly", factors=("sun", "cold"), severity="calm", is_compound=True,
    ),
    _rule(
        "THE MATTE FINISH", "light_slate_grey",
        '"No sun, no rain, dry air. The weather is buffering."',
        priority=420,
        fn=lambda w, p: (
            w.ephemeris.is_day
            and 16.0 <= w.temp <= 22.0
            and "cloud" in w.all_descs_lower
            and w.humidity < 50
            and p.wind_rank <= WIND_RANKS["light"]
        ),
        synergy_key="dry_cloudy", factors=("clouds", "dry"), severity="calm", is_compound=True,
    ),
    _rule(
        "TERRACE ESPRESSO", "bright_green",
        '"Ideal luminosity and gentle breeze. Step away from the monitor."',
        priority=415,
        fn=lambda w, p: (
            w.ephemeris.is_day
            and w.local_hour < 15
            and 18.0 <= w.temp <= 25.0
            and p.wind_rank <= WIND_RANKS["light"]
            and w.wind_speed < 4.5
            and w.ephemeris.cloud_pct <= 75
            and w.rain_prob < 20.0
        ),
        synergy_key="bright_mild", factors=("sun",), severity="calm", is_compound=False,
    ),
    _rule(
        "BIG GREEN EGG TERRITORY", "bright_yellow",
        '"Looks like meat\'s back on the menu, boys!"',
        priority=410,
        fn=lambda w, p: (
            18.0 <= w.temp <= 25.0
            and w.rain_prob < 20.0
            and p.wind_rank <= WIND_RANKS["light"]
            and w.ephemeris.cloud_pct <= 75
        ),
        synergy_key="calm_mild", factors=(), severity="calm", is_compound=False,
    ),
    _rule(
        "THE PATIO GAMBLE", "khaki1",
        '"Lovely right now, but the forecast has trust issues. Eat outside, but sit near the door."',
        priority=407,
        fn=lambda w, p: 18.0 <= w.temp <= 25.0 and p.wind_rank <= WIND_RANKS["light"] and w.rain_prob >= 20.0,
        synergy_key="uncertain_patio", factors=("front",), severity="calm", is_compound=False,
    ),
    _rule(
        "THE VOID", "bright_white",
        '"Absolutely nothing is happening out there. Perfect stasis."',
        priority=406,
        fn=lambda w, p: 6.0 <= w.temp <= 13.0 and p.wind == "calm" and w.wind_speed < 1.0 and w.uv_index <= 3.0,
        synergy_key="dead_calm", factors=(), severity="calm", is_compound=False,
    ),
    _rule(
        "OVERCAST VELVET", "light_slate_grey",
        '"Zero glare, uniform diffuse lighting. Optimal hacking state."',
        priority=405,
        fn=lambda w, p: (
            w.ephemeris.is_day
            and ("overcast" in w.all_descs_lower or w.ephemeris.cloud_pct > 75)
            and 10.0 <= w.temp <= 25.0
            and p.wind_rank <= WIND_RANKS["light"]
        ),
        synergy_key="soft_overcast", factors=("clouds",), severity="calm", is_compound=False,
    ),

    # =================================================================
    # BAND E (399–0): PURE BASELINES & FALLBACK
    # =================================================================
    _rule(
        "THE BRISK BREEZE", "bright_yellow",
        '"Nice temperature, but the wind is trying to steal your napkins."',
        priority=380,
        fn=lambda w, p: 14.0 <= w.temp <= 25.0 and p.wind == "breezy",
        synergy_key="brisk_breeze", factors=("wind",), severity="calm", is_compound=False,
    ),
    _rule(
        "THE ROASTING SPIT", "bright_yellow",
        '"Hot enough to complain, not hot enough to melt. Stay in the shade."',
        priority=370,
        fn=lambda w, p: 25.0 < w.temp <= 31.0,
        synergy_key="hot_still", factors=("heat",), severity="advisory", is_compound=False,
    ),
    _rule(
        "LAYERING SEASON", "dim white",
        '"A jacket is too much. No jacket is too little. Good luck."',
        priority=360,
        fn=lambda w, p: 10.0 <= w.temp < 18.0,
    ),
    _rule(
        "THE BELGIAN STANDARD", "dim white",
        '"Grey, chilly, unapologetically mediocre. Peak Tuesday weather."',
        priority=350,
        regions=BE,
        fn=lambda w, p: 3.0 < w.temp < 10.0,
    ),
    _rule(
        "CASTAGNE E CAMINO", "sandy_brown",
        '"Chestnuts on the fire, polenta on the stove. Hibernation protocol engaged."',
        priority=350,
        regions=IT,
        fn=lambda w, p: 3.0 < w.temp < 10.0,
    ),
    _rule(
        "THE ESTONIAN STANDARD", "dim white",
        '"Grey, chilly, and silent. Locally described as \'normal\', if anything is said at all."',
        priority=350,
        regions=EE,
        fn=lambda w, p: 3.0 < w.temp < 10.0,
    ),
    _rule(
        "WINTER IS COMING", "bright_cyan",
        '"The North remembers." Put on the thick woolen socks.',
        priority=340,
        fn=lambda w, p: w.temp <= 3.0,
        factors=("cold",),
    ),
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
    profile = analyze_atmosphere(payload)
    for rule in _RULES_BY_REGION[region]:
        try:
            if rule.matches(payload, profile):
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
    decl = (
        0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g)
        - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
        - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g)
    )
    eqtime = 229.18 * (
        0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g)
    )
    hour_angle = math.radians((hour * 60.0 + eqtime + 4.0 * lon) / 4.0 - 180.0)
    lat_r = math.radians(lat)
    cos_zen = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(hour_angle)
    return 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zen))))


def estimate_uv(
    lat: float,
    lon: float,
    ts: float,
    cloud_pct: int,
    has_fog: bool = False,
    has_mist: bool = False,
) -> float:
    lat = round(lat, 1)
    lon = round(lon, 1)
    elev = solar_elevation_deg(lat, lon, ts)
    if elev <= 0.0:
        return 0.0
    eff_cloud = max(cloud_pct, 85 if has_fog else (60 if has_mist else cloud_pct))
    clear_sky = 12.5 * math.sin(math.radians(elev)) ** 2.42
    return round(clear_sky * (1.0 - 0.75 * (eff_cloud / 100.0) ** 3.4), 1)


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
    om_params = {
        "latitude": lat, "longitude": lon, "current": "uv_index,wind_gusts_10m",
        "hourly": "precipitation_probability", "wind_speed_unit": "ms",
        "forecast_hours": 1, "timezone": "GMT",
    }

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
            aqi_data = AirQualityData(
                item["main"]["aqi"],
                item["components"].get("pm2_5", 0.0),
                item["components"].get("pm10", 0.0),
            )
        except (KeyError, IndexError, TypeError):
            pass

    uv_val: Optional[float] = None
    om_pop: Optional[float] = None
    om_gust: float = 0.0
    if isinstance(d_om, dict):
        try:
            curr_om = d_om.get("current", {})
            uv_raw = curr_om.get("uv_index")
            if uv_raw is not None:
                uv_val = round(float(uv_raw), 1)
            gust_raw = curr_om.get("wind_gusts_10m")
            if gust_raw is not None:
                om_gust = float(gust_raw)
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

    primary_desc, curr_descs = parse_weather_tags(d_curr.get("weather"), default="Unknown")
    curr_vis_raw = d_curr.get("visibility")
    curr_vis = int(curr_vis_raw) if curr_vis_raw is not None else None
    curr_blob = " ".join(curr_descs)
    has_fog = _is_dense_fog(curr_blob, curr_vis)
    has_mist = (not has_fog) and (
        "mist" in curr_blob or "haze" in curr_blob or (curr_vis is not None and curr_vis < 4000)
    )

    uv_estimated = uv_val is None
    if uv_val is None:
        uv_val = estimate_uv(lat, lon, now_ts, cloud_pct, has_fog=has_fog, has_mist=has_mist)

    if is_day:
        rem = max(0, ss - now_ts)
        daylight_str = f"{int(rem // 3600):02d}h {int((rem % 3600) // 60):02d}m"
        if has_fog:
            lighting = "Fog / Low Visibility"
        elif cloud_pct > 80:
            lighting = "Gloomy / Low Diffuse Lux"
        elif has_mist:
            lighting = "Hazy / Diffused Mist"
        elif cloud_pct > 40:
            lighting = "Filtered Sun / Soft Glare"
        else:
            lighting = "Direct Bright Sun"
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

        step_wind = float(entry.get("wind", {}).get("speed", 0.0))
        step_gust = max(step_wind, float(entry.get("wind", {}).get("gust") or 0.0))
        step_vis_raw = entry.get("visibility")
        step_vis = int(step_vis_raw) if step_vis_raw is not None else None
        step_primary_desc, step_descs = parse_weather_tags(entry.get("weather"))
        step_main = entry.get("main", {})
        step_feels_raw = step_main.get("feels_like")
        step_hum_raw = step_main.get("humidity")
        steps.append(ForecastStep(
            time_str=step_dt.strftime("%H:%M"),
            temp=float(step_main.get("temp", 0.0)),
            pop=float(entry.get("pop", 0.0)),
            rain_mm=_step_precip("rain"),
            wind_speed=step_wind,
            wind_deg=int(entry.get("wind", {}).get("deg", 0)),
            desc=step_primary_desc.title(),
            day_offset=day_offset,
            wind_gust=step_gust,
            snow_mm=_step_precip("snow"),
            feels_like=float(step_feels_raw) if step_feels_raw is not None else None,
            humidity=int(step_hum_raw) if step_hum_raw is not None else None,
            visibility=step_vis,
            all_descs=step_descs,
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
    curr_wind = float(d_curr.get("wind", {}).get("speed", 0.0))
    owm_gust = float(d_curr.get("wind", {}).get("gust") or 0.0)
    curr_gust = max(curr_wind, owm_gust, om_gust)

    return WeatherPayload(
        location_name=country,
        temp=float(d_curr.get("main", {}).get("temp", 0.0)),
        feels_like=float(d_curr.get("main", {}).get("feels_like", 0.0)),
        humidity=int(d_curr.get("main", {}).get("humidity", 0)),
        pressure=int(d_curr.get("main", {}).get("pressure", 1013)),
        wind_speed=curr_wind,
        wind_deg=int(d_curr.get("wind", {}).get("deg", 0)),
        desc=primary_desc,
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
        wind_gust=curr_gust,
        visibility=curr_vis,
        all_descs=curr_descs,
    )


def nearest_step(steps: List[ForecastStep], target_hr: int, tolerance_hr: int = 1) -> Optional[ForecastStep]:
    def dist(s: ForecastStep) -> int:
        d = abs(int(s.time_str[:2]) - target_hr) % 24
        return min(d, 24 - d)
    best = min(steps, key=dist, default=None)
    return best if best is not None and dist(best) <= tolerance_hr else None
