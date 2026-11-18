"""Comprehensive unit & synergy test suite for weatherweb engine, localizations, and HTML rendering."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx

import build_site
import engine
import vibe_be
import vibe_ee


# =====================================================================
# SHARED TEST FIXTURE BUILDERS
# =====================================================================


def make_step(
    *,
    time_str: str = "12:00",
    temp: float = 12.0,
    pop: float = 0.0,
    rain_mm: float = 0.0,
    snow_mm: float = 0.0,
    wind_speed: float = 2.0,
    wind_gust: float = 0.0,
    wind_deg: int = 280,
    desc: str = "few clouds",
    day_offset: int = 0,
    feels_like: float | None = None,
    humidity: int | None = 65,
    visibility: int | None = 10000,
    all_descs: tuple[str, ...] = (),
) -> engine.ForecastStep:
    """Build a ForecastStep with clean defaults that avoid unintended compound triggers."""
    return engine.ForecastStep(
        time_str=time_str,
        temp=temp,
        pop=pop,
        rain_mm=rain_mm,
        wind_speed=wind_speed,
        wind_deg=wind_deg,
        desc=desc,
        day_offset=day_offset,
        wind_gust=wind_gust,
        snow_mm=snow_mm,
        feels_like=feels_like,
        humidity=humidity,
        visibility=visibility,
        all_descs=all_descs,
    )


def make_payload(
    *,
    temp: float = 15.0,
    feels_like: float | None = None,
    humidity: int = 60,
    pressure: int = 1015,
    wind_speed: float = 2.0,
    wind_gust: float | None = None,
    wind_deg: int = 280,  # WNW: neutral w.r.t. IT LIBECCIO (200-250), SCIROCCO (100-170), TRAMONTANA (330-30)
    desc: str = "scattered clouds",
    all_descs: tuple[str, ...] = (),
    rain_amount: float = 0.0,
    snow_amount: float = 0.0,
    rain_prob: float | None = None,
    uv_index: float = 2.0,
    local_hour: int = 13,
    is_day: bool = True,
    cloud_pct: int = 40,
    lighting: str | None = None,
    country: str = "BE",
    visibility: int | None = 10000,
    aqi: int | None = None,
    pm2_5: float = 5.0,
    pm10: float = 10.0,
    forecast_steps: list[engine.ForecastStep] | None = None,
    uv_estimated: bool = False,
) -> engine.WeatherPayload:
    """Build a WeatherPayload with explicit, deterministic defaults."""
    if feels_like is None:
        feels_like = temp
    if wind_gust is None:
        wind_gust = wind_speed
    if rain_prob is None:
        rain_prob = 80.0 if (rain_amount > 0.0 or snow_amount > 0.0) else 10.0
    if lighting is None:
        lighting = "Filtered Sun / Soft Glare" if is_day else "Darkness"
    eph = engine.EphemerisData(
        daylight_left_str="03h 00m" if is_day else "Night Cycle",
        is_day=is_day,
        cloud_pct=cloud_pct,
        lighting_condition=lighting,
    )
    aqi_obj = engine.AirQualityData(aqi, pm2_5, pm10) if aqi is not None else None
    return engine.WeatherPayload(
        location_name=country,
        temp=temp,
        feels_like=feels_like,
        humidity=humidity,
        pressure=pressure,
        wind_speed=wind_speed,
        wind_deg=wind_deg,
        desc=desc,
        forecast_steps=list(forecast_steps) if forecast_steps is not None else [],
        ephemeris=eph,
        aqi_data=aqi_obj,
        rain_amount=rain_amount,
        rain_prob=rain_prob,
        uv_index=uv_index,
        local_hour=local_hour,
        snow_amount=snow_amount,
        uv_estimated=uv_estimated,
        country=country,
        wind_gust=wind_gust,
        visibility=visibility,
        all_descs=all_descs,
    )


# =====================================================================
# 1. TELEMETRY, API PIPELINE, SECURITY & RENDERING INVARIANTS
# =====================================================================


class TelemetryPrecisionTests(unittest.TestCase):
    def test_dew_point_and_clamp(self) -> None:
        dp = engine.compute_dew_point(4.0, 95)
        self.assertAlmostEqual(dp, 3.3, delta=0.2)
        self.assertIsInstance(engine.compute_dew_point(-2.0, 0), float)
        self.assertAlmostEqual(engine.compute_dew_point(10.0, 100), 10.0, delta=0.1)
        self.assertAlmostEqual(engine.compute_dew_point(10.0, 150), 10.0, delta=0.1)

    def test_parse_weather_tags_preserves_order_and_deduplicates(self) -> None:
        primary, tags = engine.parse_weather_tags([
            {"id": 500, "description": "light rain"},
            {"id": 701, "description": "mist"},
            {"id": 500, "description": "Light Rain"},
            {"id": 701, "description": "   "},
            "ignored-non-dict",
        ])
        self.assertEqual(primary, "light rain")
        self.assertEqual(tags, ("light rain", "mist"))

        primary_fb, tags_fb = engine.parse_weather_tags([], default="Unknown")
        self.assertEqual(primary_fb, "Unknown")
        self.assertEqual(tags_fb, ("unknown",))

        primary_bad, tags_bad = engine.parse_weather_tags(None, default="Unknown")
        self.assertEqual(primary_bad, "Unknown")
        self.assertEqual(tags_bad, ("unknown",))

    def test_estimate_uv_attenuates_for_fog_mist_and_returns_zero_at_night(self) -> None:
        # Mid-day summer UTC timestamp at 50N, 4E -> high solar elevation
        ts_day = 1718884800.0  # 2024-06-20 12:00:00 UTC
        clear_uv = engine.estimate_uv(50.0, 4.0, ts_day, cloud_pct=10)
        mist_uv = engine.estimate_uv(50.0, 4.0, ts_day, cloud_pct=10, has_mist=True)
        fog_uv = engine.estimate_uv(50.0, 4.0, ts_day, cloud_pct=10, has_fog=True)
        self.assertGreater(clear_uv, mist_uv)
        self.assertGreater(mist_uv, fog_uv)
        self.assertGreater(fog_uv, 0.0)

        # Mid-night winter UTC timestamp at 50N, 4E -> sun below horizon
        ts_night = 1703116800.0  # 2023-12-21 00:00:00 UTC
        self.assertEqual(engine.estimate_uv(50.0, 4.0, ts_night, cloud_pct=0), 0.0)

    def test_deg_to_cardinal_and_wind_from_wraparound(self) -> None:
        expected = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW", 355: "N"}
        for deg, cardinal in expected.items():
            self.assertEqual(engine.deg_to_cardinal(deg), cardinal)

        self.assertTrue(engine.wind_from(make_payload(wind_deg=225), 200, 250))
        self.assertFalse(engine.wind_from(make_payload(wind_deg=199), 200, 250))
        # Wrap-around sector across North (330..30)
        for deg in (330, 359, 0, 15, 30, 360, -10):
            self.assertTrue(engine.wind_from(make_payload(wind_deg=deg), 330, 30), f"deg={deg}")
        for deg in (31, 180, 329):
            self.assertFalse(engine.wind_from(make_payload(wind_deg=deg), 330, 30), f"deg={deg}")

    def test_nearest_step_tolerance_and_midnight_wrap(self) -> None:
        steps = [
            make_step(time_str="00:00"),
            make_step(time_str="11:00"),
            make_step(time_str="14:00"),
            make_step(time_str="23:00"),
        ]
        self.assertEqual(engine.nearest_step(steps, 11).time_str, "11:00")
        self.assertEqual(engine.nearest_step(steps, 12).time_str, "11:00")
        self.assertIsNone(engine.nearest_step(steps, 6, tolerance_hr=1))
        # Midnight wrap-around: 23:00 is 1h from 00:00 and vice versa
        self.assertEqual(engine.nearest_step([make_step(time_str="23:00")], 0, tolerance_hr=1).time_str, "23:00")
        self.assertEqual(engine.nearest_step([make_step(time_str="00:00")], 23, tolerance_hr=1).time_str, "00:00")
        self.assertIsNone(engine.nearest_step([], 12))

    def test_wind_tier_notable_gust_and_elevated_unity(self) -> None:
        cases = [
            (0.0, 0.0, "calm", False),
            (1.9, 3.9, "calm", False),
            (2.0, 2.0, "light", False),
            (1.0, 4.0, "light", False),
            (5.0, 5.0, "breezy", False),
            (3.0, 8.5, "breezy", False),
            (8.0, 8.0, "strong", True),
            (4.0, 12.0, "strong", True),
            (6.5, 10.5, "strong", True),
            (11.5, 11.5, "storm", True),
            (7.0, 16.0, "storm", True),
            (9.5, 13.5, "storm", True),
            (15.0, 15.0, "gale", True),
            (10.0, 21.0, "gale", True),
            (12.0, 17.5, "gale", True),
        ]
        for ws, wg, tier, elevated in cases:
            with self.subTest(wind_speed=ws, wind_gust=wg):
                self.assertEqual(engine.classify_wind_tier(ws, wg), tier)
                self.assertEqual(engine.is_wind_elevated(ws, wg), elevated)

        # Shared notable-gust rule: gust >= 9.5 AND gust - wind_speed >= 2.0
        self.assertFalse(engine.is_notable_gust(7.0, 9.4))
        self.assertFalse(engine.is_notable_gust(8.0, 9.5))
        self.assertTrue(engine.is_notable_gust(7.5, 9.5))
        self.assertTrue(engine.is_notable_gust(5.0, 11.0))

    def test_unknown_country_fallback_and_evaluate_vibe_fault_tolerance(self) -> None:
        p = make_payload(country="XX", temp=14.0)
        self.assertEqual(p.region, engine.DEFAULT_REGION)
        self.assertIsInstance(engine.evaluate_vibe(p), engine.VibeRule)

        # Even if a rule raises, evaluate_vibe logs and safely falls back to DEFAULT_VIBES
        broken = engine.VibeRule("BROKEN", "bright_white", '"boom"', 9999, lambda w: 1 / 0)
        with mock.patch.dict(engine._RULES_BY_REGION, {"BE": [broken]}):
            with self.assertLogs("weatherweb", level="ERROR"):
                res = engine.evaluate_vibe(make_payload(country="BE"))
            self.assertEqual(res.name, engine.DEFAULT_VIBES["BE"].name)


class FetchWeatherPipelineTests(unittest.TestCase):
    def test_fetch_weather_parses_visibility_multitag_fog_lighting_and_snow_steps(self) -> None:
        now = int(time.time())
        seen_coords: list[tuple[str, str]] = []
        curr_payload = {
            "dt": now,
            "timezone": 3600,
            "sys": {"sunrise": now - 3600, "sunset": now + 10800},
            "clouds": {"all": 10},  # clear sky aloft despite surface radiation fog
            "visibility": 350,
            "main": {"temp": 1.0, "feels_like": -2.5, "humidity": 97, "pressure": 1018},
            "wind": {"speed": 2.0, "deg": 210, "gust": 3.0},
            "weather": [
                {"id": 500, "description": "light rain"},
                {"id": 741, "description": "fog"},
            ],
            "rain": {"1h": 0.4},
        }
        fore_payload = {
            "list": [
                {
                    "dt": now + 3600,
                    "main": {"temp": 0.2, "feels_like": -3.5, "humidity": 96},
                    "wind": {"speed": 3.5, "deg": 200, "gust": 5.0},
                    "pop": 0.85,
                    "rain": {"3h": 1.2},
                    "snow": {"3h": 2.4},
                    "visibility": 600,
                    "weather": [
                        {"id": 511, "description": "freezing rain"},
                        {"id": 741, "description": "fog"},
                    ],
                },
                {
                    "dt": now + 86400,  # next day
                    "main": {"temp": -6.0, "feels_like": -10.5, "humidity": 82},
                    "wind": {"speed": 4.0, "deg": 45, "gust": 11.0},
                    "pop": 0.2,
                    "snow": {"3h": 0.6},
                    "visibility": 3200,
                    "weather": [{"id": 600, "description": "light snow"}],
                },
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            qp = request.url.params
            if "lat" in qp and "lon" in qp:
                seen_coords.append((qp["lat"], qp["lon"]))
            if "latitude" in qp and "longitude" in qp:
                seen_coords.append((qp["latitude"], qp["longitude"]))
            if "/weather" in url:
                return httpx.Response(200, content=json.dumps(curr_payload))
            if "/forecast" in url:
                return httpx.Response(200, content=json.dumps(fore_payload))
            if "/air_pollution" in url:
                return httpx.Response(200, json={"list": [{"main": {"aqi": 2}, "components": {"pm2_5": 6.2, "pm10": 11.4}}]})
            if "open-meteo.com" in url:
                return httpx.Response(503, json={})  # verify solar UV fallback with fog
            return httpx.Response(404)

        async def run() -> engine.WeatherPayload:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                return await engine.fetch_weather(client, "dummy-key", 51.2345, -17.8765, "BE")

        payload = asyncio.run(run())
        # Coordinates must be rounded to 1 decimal place before leaving the host
        self.assertTrue(seen_coords)
        for lat_s, lon_s in seen_coords:
            self.assertEqual((lat_s, lon_s), ("51.2", "-17.9"))

        self.assertEqual(payload.location_name, "BE")
        self.assertEqual(payload.visibility, 350)
        self.assertEqual(payload.all_descs, ("light rain", "fog"))
        self.assertTrue(payload.is_dense_fog)
        self.assertEqual(payload.ephemeris.lighting_condition, "Fog / Low Visibility")
        self.assertTrue(payload.uv_estimated)
        self.assertEqual(len(payload.forecast_steps), 2)

        step0 = payload.forecast_steps[0]
        self.assertAlmostEqual(step0.rain_mm, 1.2)
        self.assertAlmostEqual(step0.snow_mm, 2.4)
        self.assertAlmostEqual(step0.total_precip_mm, 3.6)
        self.assertAlmostEqual(step0.rain_rate, 0.4)
        self.assertEqual(step0.visibility, 600)
        self.assertEqual(step0.all_descs, ("freezing rain", "fog"))
        self.assertTrue(step0.is_dense_fog)
        self.assertTrue(step0.is_freezing_wet)

        step1 = payload.forecast_steps[1]
        self.assertGreaterEqual(step1.day_offset, 1)
        self.assertAlmostEqual(step1.snow_mm, 0.6)
        self.assertEqual(step1.feels_like, -10.5)

    def test_fetch_weather_open_meteo_override_and_optional_aqi_resilience(self) -> None:
        now = int(time.time())
        curr_payload = {
            "dt": now,
            "timezone": 0,
            "sys": {"sunrise": now - 3600, "sunset": now + 3600},
            "clouds": {"all": 25},
            "visibility": 10000,
            "main": {"temp": 21.0, "feels_like": 21.5, "humidity": 55, "pressure": 1016},
            "wind": {"speed": 4.0, "deg": 240, "gust": 6.5},
            "weather": [{"id": 500, "description": "light rain"}],
            "rain": {"3h": 3.0},  # 3h fallback -> 1.0 mm/h
        }
        fore_payload = {
            "list": [
                {
                    "dt": now + 10800,
                    "main": {"temp": 19.0, "feels_like": 19.0, "humidity": 65},
                    "wind": {"speed": 5.0, "deg": 240},
                    "pop": 0.4,
                    "rain": {"1h": 0.5},  # 1h fallback -> 1.5 mm / 3h
                    "weather": [{"id": 500, "description": "light rain"}],
                }
            ]
        }

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "open-meteo.com" in url:
                return httpx.Response(200, json={
                    "current": {"uv_index": 5.44, "wind_gusts_10m": 12.2},
                    "hourly": {"precipitation_probability": [72]},
                })
            if "/weather" in url:
                return httpx.Response(200, json=curr_payload)
            if "/forecast" in url:
                return httpx.Response(200, json=fore_payload)
            if "/air_pollution" in url:
                return httpx.Response(500, json={})  # optional -> None
            return httpx.Response(404)

        async def run() -> engine.WeatherPayload:
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                return await engine.fetch_weather(client, "dummy-key", 14.56, -29.34, "IT")

        payload = asyncio.run(run())
        self.assertIsNone(payload.aqi_data)
        self.assertFalse(payload.uv_estimated)
        self.assertAlmostEqual(payload.uv_index, 5.4)
        self.assertAlmostEqual(payload.wind_gust, 12.2)  # max(4.0, 6.5, 12.2)
        self.assertAlmostEqual(payload.rain_prob, 72.0)  # Open-Meteo POP overrides OWM
        self.assertAlmostEqual(payload.rain_amount, 1.0)  # 3h / 3.0
        self.assertAlmostEqual(payload.forecast_steps[0].rain_mm, 1.5)  # 1h * 3.0


class SecurityAndLeakGuardTests(unittest.TestCase):
    def test_forbidden_strings_collects_secrets_without_short_temp_collisions(self) -> None:
        locs = [
            build_site.Location("BE", "🇧🇪", 51.2345, -17.8765, "51.2345, -17.8765"),
            build_site.Location("EE", "🇪🇪", 10.2, -31.7654, "10.2,-31.7654"),
        ]
        token = "SuperSecretToken_123456"
        bad = build_site.forbidden_strings(locs, token)
        values = {v for _, v in bad}

        self.assertIn(token, values)
        self.assertIn("51.2345, -17.8765", values)
        self.assertIn("51.234", values)
        self.assertIn("51.2345", values)
        self.assertIn("-17.877", values)
        self.assertIn("-31.7654", values)
        # 1-decimal repr '10.2' must NOT be treated as forbidden (would collide with +10.2°C)
        self.assertNotIn("10.2", values)

    def test_assert_clean_allows_clean_html_and_blocks_leaks(self) -> None:
        locs = [build_site.Location("BE", "🇧🇪", 51.2345, -17.8765, "51.2345, -17.8765")]
        token = "SuperSecretToken_123456"
        bad = build_site.forbidden_strings(locs, token)

        # Token is allowed in the output file path key, never in file contents
        clean_files = {
            f"{token}/index.html": "<html><body>BE +10.2°C</body></html>",
            "index.html": "<html><body>Nothing here</body></html>",
        }
        build_site.assert_clean(clean_files, bad)

        for leaked_snippet in (token, "51.2345, -17.8765", "51.234", "-17.8765"):
            with self.subTest(leaked_snippet=leaked_snippet):
                dirty = {"index.html": f"<html><body>debug {leaked_snippet}</body></html>"}
                err_buf = io.StringIO()
                with contextlib.redirect_stderr(err_buf):
                    with self.assertRaises(SystemExit) as ctx:
                        build_site.assert_clean(dirty, bad)
                self.assertEqual(ctx.exception.code, 2)
                err_text = err_buf.getvalue()
                self.assertIn("leak guard", err_text)
                self.assertIn("███", err_text)
                # Full raw leaked snippet must never be echoed back to stderr
                self.assertNotIn(leaked_snippet, err_text)

    def test_fetch_one_redacts_urls_coords_and_api_key_on_error(self) -> None:
        loc = build_site.Location("BE", "🇧🇪", 51.2345, -17.8765, "51.2345,-17.8765")
        secret_key = "TOP_SECRET_OWM_KEY_999"

        def fail_http(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, request=request)

        def fail_net(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"failed to reach {request.url}")

        async def run_with(handler):
            transport = httpx.MockTransport(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                return await build_site.fetch_one(client, secret_key, loc)

        with mock.patch.object(build_site, "RETRY_DELAYS", (0,)):
            _, w1, err1 = asyncio.run(run_with(fail_http))
            self.assertIsNone(w1)
            self.assertEqual(err1, "HTTP 502")
            self.assertNotIn(secret_key, err1)
            self.assertNotIn("51.2", err1)

            _, w2, err2 = asyncio.run(run_with(fail_net))
            self.assertIsNone(w2)
            self.assertEqual(err2, "ConnectError")
            self.assertNotIn(secret_key, err2)
            self.assertNotIn("51.2", err2)

    def test_render_escapes_html_and_includes_security_and_poll_anchors(self) -> None:
        loc = build_site.Location("BE", "🇧🇪", 0.0, 0.0, "0,0")
        xss = '<script>alert("xss")</script>'
        w = make_payload(country="BE", desc=xss, all_descs=(xss,))

        detail = build_site.render_detail(loc, w, None, 1700000000)
        overview = build_site.render_overview([(loc, w, None)], 1700000000)
        for doc in (detail, overview):
            self.assertNotIn(xss, doc)
            self.assertIn("&lt;script&gt;", doc.lower())
            self.assertIn("Content-Security-Policy", doc)
            self.assertIn("noindex,nofollow", doc)
            self.assertIn("<main>", doc)
            self.assertIn("<footer>", doc)


class RenderPrecisionTests(unittest.TestCase):
    def _sample_payload(
        self,
        country: str = "BE",
        temp: float = -0.5,
        rain_amount: float = 0.6,
        snow_amount: float = 0.0,
        desc: str = "freezing rain",
        all_descs: tuple[str, ...] = ("freezing rain", "fog"),
        visibility: int = 350,
        wind_speed: float = 3.5,
        wind_gust: float = 5.0,
        aqi: int | None = None,
    ) -> engine.WeatherPayload:
        eph = engine.EphemerisData("03h 15m", True, 15, "Fog / Low Visibility")
        steps = [
            make_step(
                time_str="11:00", temp=-0.5, pop=0.8, rain_mm=1.5, snow_mm=0.0,
                wind_speed=3.5, wind_deg=210, desc="Freezing Rain", day_offset=0,
                wind_gust=11.0, feels_like=-5.0, humidity=96, visibility=600,
                all_descs=("freezing rain",),
            ),
            make_step(
                time_str="14:00", temp=1.0, pop=0.3, rain_mm=0.4, snow_mm=0.8,
                wind_speed=4.0, wind_deg=220, desc="Rain And Snow", day_offset=0,
                wind_gust=6.0, feels_like=-2.5, humidity=92, visibility=800,
            ),
            make_step(
                time_str="17:00", temp=-6.0, pop=0.1, rain_mm=0.0, snow_mm=0.0,
                wind_speed=5.0, wind_deg=30, desc="Fog", day_offset=1,
                wind_gust=7.5, feels_like=-11.5, humidity=96, visibility=400,
            ),
        ]
        return engine.WeatherPayload(
            location_name=country, temp=temp, feels_like=temp - 3.5, humidity=96, pressure=1014,
            wind_speed=wind_speed, wind_deg=210, desc=desc, forecast_steps=steps, ephemeris=eph,
            aqi_data=engine.AirQualityData(aqi, 55.0, 90.0) if aqi is not None else None,
            rain_amount=rain_amount, rain_prob=80.0, uv_index=0.2,
            local_hour=10, snow_amount=snow_amount, uv_estimated=False, country=country,
            wind_gust=wind_gust, visibility=visibility, all_descs=all_descs,
        )

    def test_detail_page_surfaces_synergy_fog_dewpoint_freezing_snow_and_day_boundary(self) -> None:
        loc_be = build_site.Location("BE", "🇧🇪", 0.0, 0.0, "0,0")
        w_be = self._sample_payload("BE")
        v_be = engine.evaluate_vibe(w_be)
        self.assertEqual(v_be.name, "BLACK ICE PROTOCOL")
        be_title, _ = vibe_be.get_be_vibe(v_be.name, v_be.quote)

        html_be = build_site.render_detail(loc_be, w_be, None, 1700000000)
        self.assertIn(f"<title>BE · {be_title}</title>", html_be)
        self.assertIn("FREEZING PRECIP", html_be)
        self.assertIn("Freezing Rain + Fog", html_be)
        self.assertIn("vis 350 m", html_be)
        self.assertIn("Dew -1.1°C", html_be)
        self.assertIn("350 m (Dense fog)", html_be)
        self.assertIn("Fog / Low Visibility", html_be)
        self.assertIn("Precip chance", html_be)
        self.assertIn("0.4+❄0.8mm", html_be)
        self.assertIn('class="bar mix"', html_be)
        self.assertIn('class="bar ice"', html_be)
        # Synergy strip + factor chips + classify_step-backed compound forecast labels
        self.assertIn('class="synergy"', html_be)
        self.assertIn(v_be.synergy_label, html_be)
        self.assertIn('<span class="chip">Freezing</span>', html_be)
        self.assertIn('<span class="chip">Rain</span>', html_be)
        self.assertIn("Ice storm", html_be)
        self.assertIn("Wintry mix", html_be)
        self.assertIn("Freezing fog", html_be)
        # Target Windows wind visibility: step0 has gusts 11 m/s + ice_storm ("wind" factor)
        self.assertIn("wind SSW 3.5 m/s (gusts 11)", html_be)
        self.assertIn("gusts 11 m/s", html_be)
        self.assertIn('<h3 class="day-sep">Tomorrow</h3>', html_be)

        loc_ee = build_site.Location("EE", "🇪🇪", 0.0, 0.0, "0,0")
        w_ee = self._sample_payload("EE")
        ee_title, _ = vibe_ee.get_ee_vibe(v_be.name, v_be.quote)
        html_ee = build_site.render_detail(loc_ee, w_ee, None, 1700000000)
        self.assertIn('lang="et"', html_ee)
        self.assertIn(f"<title>EE · {ee_title}</title>", html_ee)
        self.assertIn("JÄÄTUV SADU", html_ee)
        self.assertIn("Jäätuv vihm + udu", html_ee)
        self.assertIn("Kastep. -1.1°C", html_ee)
        self.assertIn("350 m (Tihe udu)", html_ee)
        self.assertIn("Udu / Piiratud nähtavus", html_ee)
        self.assertIn("0.4+❄0.8mm", html_ee)
        self.assertIn(vibe_ee.translate_synergy_et(v_be.synergy_key, v_be.synergy_label), html_ee)
        self.assertIn(vibe_ee.translate_step_kind_et("ice_storm", "Ice storm"), html_ee)
        self.assertIn(vibe_ee.translate_step_kind_et("mixed_precip", "Wintry mix"), html_ee)
        self.assertIn(vibe_ee.translate_step_kind_et("freezing_fog", "Freezing fog"), html_ee)
        self.assertIn('<h3 class="day-sep">Homme</h3>', html_ee)

    def test_detail_page_it_and_unavailable_fallback_pages(self) -> None:
        loc_it = build_site.Location("IT", "🇮🇹", 0.0, 0.0, "0,0")
        w_it = make_payload(country="IT", temp=16.0, wind_deg=225, wind_speed=10.0, desc="few clouds")
        html_it = build_site.render_detail(loc_it, w_it, None, 1700000000)
        self.assertIn('lang="en"', html_it)
        self.assertIn("<title>IT · LIBECCIO</title>", html_it)
        self.assertIn("Meteo IT", html_it)

        loc_ee = build_site.Location("EE", "🇪🇪", 0.0, 0.0, "0,0")
        err_en = build_site.render_detail(loc_it, None, "HTTP 503", 1700000000)
        self.assertIn("unavailable", err_en)
        self.assertIn("HTTP 503", err_en)
        err_ee = build_site.render_detail(loc_ee, None, "TimeoutException", 1700000000)
        self.assertIn('lang="et"', err_ee)
        self.assertIn("pole saadaval", err_ee)
        self.assertIn("TimeoutException", err_ee)

    def test_trace_drizzle_subtitle_and_poor_aqi_hazard_badge(self) -> None:
        loc_be = build_site.Location("BE", "🇧🇪", 0.0, 0.0, "0,0")
        w = self._sample_payload(
            "BE", temp=4.0, rain_amount=0.0, desc="light intensity drizzle",
            all_descs=("light intensity drizzle",), visibility=6000, aqi=5,
        )
        html_be = build_site.render_detail(loc_be, w, None, 1700000000)
        self.assertIn("Trace drizzle", html_be)
        self.assertIn("Hazardous", html_be)

    def test_overview_shows_synergy_feels_like_precip_visibility_and_partial_failure(self) -> None:
        loc_be = build_site.Location("BE", "🇧🇪", 0.0, 0.0, "0,0")
        loc_it = build_site.Location("IT", "🇮🇹", 0.0, 0.0, "0,0")
        loc_ee = build_site.Location("EE", "🇪🇪", 0.0, 0.0, "0,0")
        ov = build_site.render_overview([
            (loc_be, self._sample_payload("BE"), None),
            (loc_it, None, "HTTP 500"),
            (loc_ee, self._sample_payload(
                "EE", temp=-2.0, rain_amount=0.0, snow_amount=1.4,
                desc="light snow", all_descs=("light snow",), visibility=2500,
                wind_speed=9.5, wind_gust=13.0,
            ), None),
        ], 1700000000)
        self.assertIn("(feels -4°)", ov)
        self.assertIn("0.6 mm", ov)
        self.assertIn("vis 350 m", ov)
        self.assertIn("Freezing precip", ov)
        self.assertIn("Freezing rain", ov)  # compound synergy pill on BE overview card
        self.assertIn("Unavailable this cycle", ov)  # partial failure for IT
        self.assertIn("❄1.4 mm", ov)
        self.assertIn("näht. 2.5 km", ov)
        self.assertIn(vibe_ee.translate_synergy_et("blizzard", "Blizzard"), ov)


# =====================================================================
# 2. COMBINATORIAL CONDITION-SYNERGY MATRIX
# =====================================================================


class CombinatorialSynergyMatrixTests(unittest.TestCase):
    def _assert_vibe(
        self,
        expected_name: str,
        *,
        expected_key: str | None = None,
        expect_compound: bool | None = None,
        required_factors: tuple[str, ...] = (),
        **kwargs,
    ) -> engine.VibeRule:
        v = engine.evaluate_vibe(make_payload(**kwargs))
        self.assertEqual(v.name, expected_name, f"inputs={kwargs} -> got {v.name} ({v.synergy_key})")
        if expected_key is not None:
            self.assertEqual(v.synergy_key, expected_key)
        if expect_compound is not None:
            self.assertEqual(v.is_compound, expect_compound)
        for f in required_factors:
            self.assertIn(f, v.factors, f"{v.name} missing factor {f!r}")
        return v

    def test_wind_x_rain_matrix(self) -> None:
        # Gale + light rain -> THE TEMPEST (never dry THE EYEWALL)
        self._assert_vibe(
            "THE TEMPEST", expected_key="wet_gale", expect_compound=True, required_factors=("wind", "rain"),
            temp=12.0, desc="light rain", rain_amount=0.5, wind_speed=16.0,
        )
        # Storm wind + heavy rain -> THE TEMPEST
        self._assert_vibe(
            "THE TEMPEST", expected_key="wet_gale", expect_compound=True, required_factors=("wind", "rain"),
            temp=12.0, desc="heavy intensity rain", rain_amount=5.0, wind_speed=12.0,
        )
        # Strong wind + warm rain -> TROPICAL SQUALL
        self._assert_vibe(
            "TROPICAL SQUALL", expected_key="tropical_squall", expect_compound=True, required_factors=("heat", "rain", "wind"),
            temp=24.0, desc="moderate rain", rain_amount=1.8, wind_speed=9.0,
        )
        # Breezy + heavy cool rain -> STORMY DOWNPOUR
        self._assert_vibe(
            "STORMY DOWNPOUR", expected_key="stormy_rain", expect_compound=True, required_factors=("rain", "wind"),
            temp=14.0, desc="heavy intensity rain", rain_amount=4.8, wind_speed=6.0,
        )
        # Strong wind + near-freezing rain -> NORTH SEA LASHING
        self._assert_vibe(
            "NORTH SEA LASHING", expected_key="cold_lashing_rain", expect_compound=True, required_factors=("cold", "rain", "wind"),
            temp=3.0, desc="light rain", rain_amount=0.8, wind_speed=9.0,
        )
        # Strong wind + mild rain -> THE HORIZONTAL SHOWER
        self._assert_vibe(
            "THE HORIZONTAL SHOWER", expected_key="horizontal_rain", expect_compound=True, required_factors=("rain", "wind"),
            temp=12.0, desc="light rain", rain_amount=0.8, wind_speed=9.0,
        )
        # Calm heavy warm rain -> TROPICAL DEPRESSION
        self._assert_vibe(
            "TROPICAL DEPRESSION", expected_key="warm_downpour", expect_compound=True, required_factors=("heat", "rain"),
            temp=23.0, desc="heavy intensity rain", rain_amount=5.0, wind_speed=2.0, cloud_pct=95, uv_index=1.0,
        )
        # Calm heavy cool rain -> NOAH'S ARK
        self._assert_vibe(
            "NOAH'S ARK", expected_key="heavy_rain", expect_compound=False, required_factors=("rain",),
            temp=12.0, desc="heavy intensity rain", rain_amount=5.0, wind_speed=2.0, cloud_pct=95, uv_index=0.5,
        )
        # Calm light drizzle resolves regionally
        for reg, expected in (("BE", "THE BELGIAN DRIZZLE"), ("IT", "LA PIOVIGGINE"), ("EE", "UDUVIHM")):
            self._assert_vibe(
                expected, expected_key="light_drizzle", expect_compound=False,
                country=reg, temp=11.0, desc="light rain", rain_amount=0.6, rain_prob=60.0,
                wind_speed=2.0, cloud_pct=90, uv_index=0.5,
            )

    def test_wind_x_snow_and_sleet_matrix(self) -> None:
        # Strong wind or gale + snow -> BLIZZARD (never dry THE EYEWALL or calm HOTH)
        for ws in (9.0, 12.0, 16.5):
            self._assert_vibe(
                "BLIZZARD", expected_key="blizzard", expect_compound=True, required_factors=("snow", "wind"),
                temp=-3.0, desc="snow", snow_amount=1.0, wind_speed=ws,
            )
        # Calm snow -> HOTH
        self._assert_vibe(
            "HOTH", expected_key="snowfall", expect_compound=False, required_factors=("snow",),
            temp=-3.0, desc="light snow", snow_amount=0.6, wind_speed=2.0,
        )
        # Strong wind or gale + sleet -> WINTRY GALE
        for ws in (9.0, 16.5):
            self._assert_vibe(
                "WINTRY GALE", expected_key="wintry_gale", expect_compound=True, required_factors=("sleet", "wind", "cold"),
                temp=0.5, desc="sleet", rain_amount=0.5, snow_amount=0.5, wind_speed=ws,
            )
        # Calm sleet / rain+snow -> SLUSH SEASON
        self._assert_vibe(
            "SLUSH SEASON", expected_key="mixed_precip", expect_compound=True, required_factors=("sleet", "cold"),
            temp=0.5, desc="light rain and snow", rain_amount=0.4, snow_amount=0.4, wind_speed=2.0,
        )

    def test_rain_x_freezing_and_cold_matrix(self) -> None:
        # Freezing rain + breezy/strong/gale wind -> ICE STORM
        for ws in (5.5, 9.5, 16.0):
            self._assert_vibe(
                "ICE STORM", expected_key="ice_storm", expect_compound=True, required_factors=("freezing", "rain", "wind"),
                temp=-1.0, desc="freezing rain", rain_amount=1.0, wind_speed=ws,
            )
        # Freezing rain + calm/light wind -> BLACK ICE PROTOCOL (also at temp<=0.5 without "freezing" tag)
        self._assert_vibe(
            "BLACK ICE PROTOCOL", expected_key="freezing_rain", expect_compound=True, required_factors=("freezing", "rain"),
            temp=-1.0, desc="freezing rain", rain_amount=1.0, wind_speed=2.0,
        )
        self._assert_vibe(
            "BLACK ICE PROTOCOL", expected_key="freezing_rain", expect_compound=True, required_factors=("freezing", "rain"),
            temp=0.0, desc="light rain", rain_amount=0.6, wind_speed=2.0,
        )
        # Near-freezing raw rain (0.5..6.0) calm -> THE MARROW SOAK (even in morning or dark EE afternoon)
        self._assert_vibe(
            "THE MARROW SOAK", expected_key="cold_rain", expect_compound=True, required_factors=("cold", "rain"),
            temp=2.0, desc="light intensity drizzle", rain_amount=0.3, wind_speed=2.0, local_hour=8,
        )
        self._assert_vibe(
            "THE MARROW SOAK", expected_key="cold_rain", expect_compound=True, required_factors=("cold", "rain"),
            country="EE", temp=1.5, desc="light rain", rain_amount=0.6, wind_speed=2.0, local_hour=16, is_day=False,
        )

    def test_thunder_x_wind_rain_hail_matrix(self) -> None:
        # Thunder + strong wind, heavy rain, or hail -> SUPERCELL SQUALL
        self._assert_vibe(
            "SUPERCELL SQUALL", expected_key="thunder_squall", expect_compound=True,
            temp=20.0, desc="thunderstorm with rain", rain_amount=1.0, wind_speed=9.0,
        )
        self._assert_vibe(
            "SUPERCELL SQUALL", expected_key="thunder_squall", expect_compound=True,
            temp=20.0, desc="thunderstorm with heavy rain", rain_amount=5.0, wind_speed=2.0,
        )
        self._assert_vibe(
            "SUPERCELL SQUALL", expected_key="thunder_squall", expect_compound=True,
            temp=18.0, desc="thunderstorm with hail", rain_amount=1.0, wind_speed=2.0,
        )
        # Calm light thunderstorm -> GHOSTBUSTERS
        self._assert_vibe(
            "GHOSTBUSTERS", expected_key="thunderstorm", expect_compound=False,
            temp=20.0, desc="light thunderstorm", rain_amount=0.5, wind_speed=2.0,
        )

    def test_fog_x_rain_wind_cold_heat_matrix(self) -> None:
        # Sub-zero fog stays FROSTBOUND FOG even under strong wind (only gale -> ARCTIC GALE)
        self._assert_vibe(
            "FROSTBOUND FOG", expected_key="freezing_fog", expect_compound=True, required_factors=("freezing", "fog"),
            temp=-2.0, desc="fog", visibility=350, wind_speed=1.5,
        )
        self._assert_vibe(
            "FROSTBOUND FOG", expected_key="freezing_fog", expect_compound=True, required_factors=("freezing", "fog"),
            temp=-3.0, desc="freezing fog", visibility=350, wind_speed=10.0,
        )
        self._assert_vibe(
            "ARCTIC GALE", expected_key="arctic_gale", expect_compound=True,
            temp=-3.0, desc="freezing fog", visibility=350, wind_speed=16.0,
        )
        # Fog + rain -> FOGBOUND RAIN
        self._assert_vibe(
            "FOGBOUND RAIN", expected_key="foggy_rain", expect_compound=True, required_factors=("fog", "rain"),
            temp=8.0, desc="light rain", all_descs=("light rain", "fog"), rain_amount=1.2, visibility=400, wind_speed=2.0,
        )
        # Dry non-freezing fog + strong wind -> SEA FOG DRIFT
        self._assert_vibe(
            "SEA FOG DRIFT", expected_key="windy_fog", expect_compound=True, required_factors=("fog", "wind"),
            temp=8.0, desc="fog", visibility=400, wind_speed=9.0,
        )
        # Cold fog (0.5..5.0) calm -> CHILL FOG
        self._assert_vibe(
            "CHILL FOG", expected_key="cold_fog", expect_compound=True, required_factors=("cold", "fog"),
            temp=2.5, desc="fog", visibility=400, wind_speed=2.0,
        )
        # Mild fog -> SILENT HILL
        self._assert_vibe(
            "SILENT HILL", expected_key="dense_fog", expect_compound=False,
            temp=12.0, desc="fog", visibility=400, wind_speed=2.0,
        )
        # Warm dry fog -> THE STEAM ROOM
        self._assert_vibe(
            "THE STEAM ROOM", expected_key="warm_fog", expect_compound=True, required_factors=("fog", "heat", "humidity"),
            temp=24.0, desc="fog", visibility=500, wind_speed=2.0,
        )
        # Mist + trace drizzle -> MISTY DRIZZLE
        self._assert_vibe(
            "MISTY DRIZZLE", expected_key="misty_drizzle", expect_compound=True, required_factors=("mist", "rain"),
            temp=10.0, desc="light intensity drizzle", all_descs=("mist", "light intensity drizzle"),
            rain_amount=0.3, visibility=1800, wind_speed=2.0, cloud_pct=90, uv_index=0.5,
        )

    def test_wind_x_cold_dry_day_and_night_matrix(self) -> None:
        self._assert_vibe(
            "ARCTIC GALE", expected_key="arctic_gale", expect_compound=True,
            temp=-2.0, wind_speed=16.0,
        )
        self._assert_vibe(
            "THE EYEWALL", expected_key="gale", expect_compound=False,
            temp=14.0, wind_speed=16.0,
        )
        self._assert_vibe(
            "THE WHITE WALKER", expected_key="freezing_night_wind", expect_compound=True,
            temp=-2.0, local_hour=22, is_day=False, wind_speed=6.0,
        )
        self._assert_vibe(
            "ARCTIC WINDCHILL", expected_key="arctic_windchill", expect_compound=True,
            temp=-5.0, wind_speed=9.0,
        )
        self._assert_vibe(
            "RAW DAMP GALE", expected_key="damp_windchill", expect_compound=True, required_factors=("cold", "humidity", "wind"),
            temp=3.0, humidity=92, wind_speed=9.0,
        )
        self._assert_vibe(
            "THE WIND TUNNEL", expected_key="bitter_wind", expect_compound=True,
            temp=4.0, humidity=60, wind_speed=9.0,
        )
        self._assert_vibe(
            "BLUEBIRD FREEZE", expected_key="sunny_frost", expect_compound=True, required_factors=("cold", "freezing", "sun"),
            temp=-2.0, is_day=True, cloud_pct=15, wind_speed=2.0,
        )
        self._assert_vibe(
            "STARRY FREEZE", expected_key="starry_frost", expect_compound=True, required_factors=("cold", "freezing", "night"),
            temp=-1.0, local_hour=23, is_day=False, cloud_pct=15, wind_speed=2.0,
        )
        self._assert_vibe(
            "BALTIC FREEZE", expected_key="bitter_cold", expect_compound=False,
            temp=-5.0, is_day=True, cloud_pct=90, wind_speed=2.0,
        )
        self._assert_vibe(
            "BONE CHILL", expected_key="damp_cold", expect_compound=True,
            temp=2.0, humidity=95, wind_speed=2.0, cloud_pct=90,
        )

    def test_heat_x_humidity_x_wind_x_uv_and_sunshower_matrix(self) -> None:
        self._assert_vibe("BOILED ALIVE", expected_key="scorching_humid", temp=33.0, humidity=80, wind_speed=2.0)
        # Severity monotonicity: >31°C scorching stays THE SURFACE OF THE SUN; 25..31°C arid+breezy -> HOT DRY WIND
        self._assert_vibe("THE SURFACE OF THE SUN", expected_key="extreme_heat", temp=34.0, humidity=25, wind_speed=6.0)
        self._assert_vibe("HOT DRY WIND", expected_key="hot_dry_wind", temp=29.0, humidity=25, wind_speed=6.0)
        self._assert_vibe("THE SURFACE OF THE SUN", expected_key="extreme_heat", temp=33.0, humidity=50, wind_speed=2.0)
        self._assert_vibe("THE DRY HEAT", expected_key="arid_heat", temp=28.0, humidity=20, wind_speed=2.0)
        self._assert_vibe("THE SAUNA SESSION", expected_key="muggy_heat", temp=28.0, humidity=82, wind_speed=2.0)
        self._assert_vibe("BREEZY HEAT", expected_key="breezy_heat", temp=28.0, humidity=55, wind_speed=6.0)
        self._assert_vibe("THE DEADLY LASER", expected_key="high_uv_heat", temp=28.0, humidity=55, wind_speed=2.0, uv_index=8.5)
        self._assert_vibe("THE ROASTING SPIT", expected_key="hot_still", temp=28.0, humidity=55, wind_speed=2.0, uv_index=4.0)

        # Sun + rain (sunshower) during day with high UV or broken sky, never at night
        self._assert_vibe(
            "SUNSHOWER", expected_key="sunshower", expect_compound=True, required_factors=("sun", "rain"),
            temp=16.0, desc="light intensity shower rain", rain_amount=0.6, wind_speed=2.0, uv_index=4.0, cloud_pct=45,
        )
        v_night_shower = engine.evaluate_vibe(make_payload(
            temp=16.0, desc="light intensity shower rain", rain_amount=0.6, wind_speed=2.0,
            uv_index=0.0, cloud_pct=45, is_day=False, local_hour=22,
        ))
        self.assertNotEqual(v_night_shower.name, "SUNSHOWER")

    def test_adversarial_edge_cases_and_engine_step_alignment(self) -> None:
        # 1. Calm 16–19.9°C heavy rain hits NOAH'S ARK (not shadowed by THE LUKEWARM WASH)
        self._assert_vibe(
            "NOAH'S ARK", expected_key="heavy_rain",
            temp=18.0, rain_amount=12.0, desc="extreme rain", wind_speed=2.0, cloud_pct=100, uv_index=0.5,
        )
        # 2. 95% overcast summer rain with UV>=3 never mislabels as SUNSHOWER
        v_ovc_rain = engine.evaluate_vibe(make_payload(
            temp=21.0, rain_amount=3.0, desc="moderate rain", cloud_pct=95, uv_index=3.5, is_day=True, wind_speed=2.0,
        ))
        self.assertNotEqual(v_ovc_rain.name, "SUNSHOWER")
        # 3. Shower snow / shower sleet are not liquid (is_wet == False) and don't add "rain" to thunder_squall
        self.assertFalse(make_payload(temp=-3.0, desc="light shower snow").is_wet)
        self.assertFalse(make_payload(temp=0.0, desc="shower sleet").is_wet)
        ts_snow = engine.classify_step(make_step(
            temp=-2.0, desc="thunderstorm", all_descs=("thunderstorm", "light shower snow"), wind_speed=9.0,
        ))
        self.assertEqual(ts_snow.kind, "thunder_squall")
        self.assertNotIn("rain", ts_snow.factors)
        # 4. NORTH SEA LASHING & STORMY DOWNPOUR align with classify_step across 1.5°C and 22°C
        self._assert_vibe(
            "NORTH SEA LASHING", expected_key="cold_lashing_rain",
            temp=1.5, rain_amount=6.0, desc="heavy intensity rain", wind_speed=10.0,
        )
        self.assertEqual(
            engine.classify_step(make_step(temp=1.5, rain_mm=18.0, desc="heavy intensity rain", wind_speed=10.0)).kind,
            "cold_lashing_rain",
        )
        self._assert_vibe(
            "STORMY DOWNPOUR", expected_key="stormy_rain",
            temp=22.0, rain_amount=6.0, desc="heavy intensity rain", wind_speed=6.0,
        )
        self.assertEqual(
            engine.classify_step(make_step(temp=22.0, rain_mm=18.0, desc="heavy intensity rain", wind_speed=6.0)).kind,
            "stormy_rain",
        )
        # 5. Severity monotonicity across SMOG TRAP, STORM BREWING, DEADLY LASER, HOT DRY WIND, SCIROCCO, TRAMONTANA
        self._assert_vibe("BOILED ALIVE", temp=38.0, humidity=80, wind_speed=1.0, aqi=4)
        self._assert_vibe(
            "BOILED ALIVE", temp=38.0, humidity=80, wind_speed=1.0,
            forecast_steps=[make_step(desc="thunderstorm", rain_mm=5.0, wind_speed=12.0)],
        )
        self._assert_vibe(
            "ARCTIC WINDCHILL", temp=-10.0, feels_like=-22.0, wind_speed=13.0,
            forecast_steps=[make_step(desc="thunderstorm", rain_mm=5.0, wind_speed=12.0)],
        )
        self._assert_vibe("THE DEADLY LASER", temp=29.0, uv_index=10.0, wind_speed=6.0, humidity=55)
        self._assert_vibe("THE DEADLY LASER", temp=29.0, uv_index=10.0, wind_speed=2.0, humidity=82)
        self._assert_vibe("THE DEADLY LASER", temp=29.0, uv_index=10.0, wind_speed=2.0, humidity=20)
        self.assertEqual(
            engine.classify_step(make_step(temp=40.0, humidity=20, wind_speed=5.5)).kind,
            "extreme_heat",
        )
        self._assert_vibe("THE SURFACE OF THE SUN", country="IT", temp=38.0, humidity=60, wind_deg=140, wind_speed=3.5)
        self._assert_vibe("THE INVISIBLE HAND", country="IT", temp=20.0, humidity=55, wind_deg=140, wind_speed=13.0)
        self._assert_vibe("THE WHITE WALKER", country="IT", temp=-10.0, wind_deg=0, wind_speed=14.0, local_hour=22, is_day=False)
        self.assertNotEqual(
            engine.evaluate_vibe(make_payload(country="IT", temp=13.5, wind_deg=0, wind_speed=6.0, cloud_pct=100)).name,
            "TRAMONTANA",
        )
        # 6. Zero accidental FORREST GUMP fallthroughs & after-dark 20:00 heat handling
        self._assert_vibe("THE PATIO GAMBLE", temp=22.0, wind_speed=1.5, rain_prob=70.0, cloud_pct=50, pressure=1012, local_hour=13)
        self.assertNotEqual(
            engine.evaluate_vibe(make_payload(temp=20.5, wind_speed=2.5, cloud_pct=85, local_hour=20, is_day=False, rain_prob=10.0)).name,
            "FORREST GUMP",
        )
        self._assert_vibe("TROPICAL NIGHT", temp=33.0, humidity=50, local_hour=20, is_day=False, wind_speed=2.0)
        # 7. classify_step 4.0 mm/h heavy rain, truthful thunder_squall factors, FROSTBOUND FOG <= 0.0°C & sun cloud guards
        self.assertEqual(engine.classify_step(make_step(temp=14.0, desc="moderate rain", rain_mm=10.0, wind_speed=2.0)).kind, "rain")
        self.assertEqual(engine.classify_step(make_step(temp=14.0, desc="moderate rain", rain_mm=13.0, wind_speed=2.0)).kind, "heavy_rain")
        ts_calm = engine.classify_step(make_step(desc="thunderstorm with heavy rain", rain_mm=15.0, wind_speed=1.0))
        self.assertEqual(ts_calm.kind, "thunder_squall")
        self.assertNotIn("wind", ts_calm.factors)
        self._assert_vibe("CHILL FOG", temp=0.4, desc="fog", visibility=400, wind_speed=2.0)
        self._assert_vibe("FROSTBOUND FOG", temp=0.0, desc="fog", visibility=400, wind_speed=2.0)
        self.assertEqual(engine.classify_step(make_step(temp=0.4, desc="fog", visibility=400, wind_speed=2.0)).kind, "dense_fog")
        self.assertEqual(engine.classify_step(make_step(temp=0.0, desc="fog", visibility=400, wind_speed=2.0)).kind, "freezing_fog")
        self.assertNotEqual(
            engine.evaluate_vibe(make_payload(country="IT", temp=22.0, local_hour=20, is_day=False, wind_speed=2.0)).name,
            "LA PASSEGGIATA",
        )
        self.assertNotIn(
            "sun",
            engine.evaluate_vibe(make_payload(temp=9.0, uv_index=3.5, cloud_pct=90, wind_speed=2.0)).factors,
        )

    def test_aqi_stagnation_and_incoming_front_synergies(self) -> None:
        # Hazardous AQI=5 on a calm sunny 23°C day must NEVER return THE GOLDILOCKS ZONE
        self._assert_vibe(
            "SMOG TRAP", expected_key="smog_inversion", expect_compound=True, required_factors=("air",),
            temp=23.0, desc="haze", visibility=2500, wind_speed=1.5, rain_prob=0.0, cloud_pct=10, aqi=5,
        )
        # Poor AQI=4 + high-pressure calm -> SMOG TRAP; strong wind disperses it
        self._assert_vibe(
            "SMOG TRAP", expected_key="smog_inversion",
            temp=20.0, pressure=1022, wind_speed=1.5, aqi=4,
        )
        self.assertNotEqual(
            engine.evaluate_vibe(make_payload(temp=20.0, pressure=1022, wind_speed=9.0, aqi=4)).name,
            "SMOG TRAP",
        )
        # Active rain washes out smog (rain rule takes precedence)
        self.assertNotEqual(
            engine.evaluate_vibe(make_payload(temp=20.0, desc="moderate rain", rain_amount=2.5, wind_speed=1.5, aqi=5)).name,
            "SMOG TRAP",
        )
        # Incoming severe step -> STORM BREWING
        self._assert_vibe(
            "STORM BREWING", expected_key="storm_approaching", expect_compound=True, required_factors=("front",),
            temp=18.0, wind_speed=4.0, forecast_steps=[make_step(desc="thunderstorm", rain_mm=3.0, wind_speed=10.0)],
        )
        # Low pressure + high rain_prob + overcast -> STORM BREWING
        self._assert_vibe(
            "STORM BREWING", expected_key="storm_approaching",
            temp=15.0, pressure=998, rain_prob=70.0, cloud_pct=85, wind_speed=3.0,
        )

    def test_regional_and_diurnal_synergies(self) -> None:
        # IT: LIBECCIO wins on strong SW dry or light rain, yields to STORMY DOWNPOUR on heavy rain
        self._assert_vibe("LIBECCIO", country="IT", temp=16.0, wind_deg=225, wind_speed=10.0)
        self._assert_vibe("LIBECCIO", country="IT", temp=16.0, wind_deg=225, wind_speed=10.0, desc="light rain", rain_amount=0.8)
        self._assert_vibe(
            "STORMY DOWNPOUR", country="IT", temp=16.0, wind_deg=225, wind_speed=10.0,
            desc="heavy intensity rain", rain_amount=6.0,
        )
        # IT: TRAMONTANA beats THE WIND TUNNEL / BALTIC FREEZE; SCIROCCO beats heat/night
        self._assert_vibe("TRAMONTANA", country="IT", temp=5.0, wind_deg=10, wind_speed=9.5)
        self._assert_vibe("SCIROCCO", country="IT", temp=27.0, humidity=70, wind_deg=140, wind_speed=4.5, local_hour=22, is_day=False)
        # Night hot dry -> TROPICAL NIGHT (never daytime SURFACE OF THE SUN)
        self._assert_vibe("TROPICAL NIGHT", temp=33.0, humidity=50, local_hour=23, is_day=False, wind_speed=2.0)
        # Night 0.5-3.0 overcast dry closes old gap -> COCOA TIME
        self._assert_vibe("COCOA TIME", temp=1.5, humidity=70, local_hour=23, is_day=False, cloud_pct=90, wind_speed=2.0)
        # EE: PIMEDUS (14-17h dark) cleanly separated from SUITSUSAUN (18-23h cold)
        self._assert_vibe("PIMEDUS", country="EE", temp=2.0, local_hour=16, is_day=False, humidity=75, wind_speed=2.0)
        self._assert_vibe("SUITSUSAUN", country="EE", temp=2.0, local_hour=19, is_day=False, humidity=75, wind_speed=2.0, cloud_pct=85)


# =====================================================================
# 3. MONOTONICITY, GUST-AWARENESS & TAG-ORDER SYMMETRY
# =====================================================================


class MonotonicityAndSymmetryTests(unittest.TestCase):
    def test_owm_tag_order_invariance_for_vibe_and_step(self) -> None:
        tag_pairs = [
            (("mist", "light rain"), dict(temp=10.0, rain_amount=0.3, visibility=1800, cloud_pct=90, uv_index=0.5)),
            (("fog", "light rain"), dict(temp=10.0, rain_amount=1.0, visibility=400, cloud_pct=95, uv_index=0.5)),
            (("freezing rain", "fog"), dict(temp=-1.0, rain_amount=0.8, visibility=400, wind_speed=2.0)),
            (("snow", "light rain"), dict(temp=0.5, rain_amount=0.4, snow_amount=0.4, wind_speed=2.0)),
            (("thunderstorm", "heavy intensity rain"), dict(temp=20.0, rain_amount=5.0, wind_speed=2.0)),
            (("light intensity drizzle", "mist"), dict(temp=10.0, rain_amount=0.2, visibility=1800, cloud_pct=90, uv_index=0.5)),
        ]
        for (t1, t2), kw in tag_pairs:
            with self.subTest(tags=(t1, t2)):
                va = engine.evaluate_vibe(make_payload(desc=t1, all_descs=(t1, t2), **kw))
                vb = engine.evaluate_vibe(make_payload(desc=t2, all_descs=(t2, t1), **kw))
                self.assertEqual(va.name, vb.name)
                self.assertEqual(va.synergy_key, vb.synergy_key)

                sa = engine.classify_step(make_step(
                    desc=t1, all_descs=(t1, t2), temp=kw["temp"],
                    rain_mm=kw.get("rain_amount", 0.0) * 3.0,
                    snow_mm=kw.get("snow_amount", 0.0) * 3.0,
                    wind_speed=kw.get("wind_speed", 2.0),
                    visibility=kw.get("visibility", 10000),
                ))
                sb = engine.classify_step(make_step(
                    desc=t2, all_descs=(t2, t1), temp=kw["temp"],
                    rain_mm=kw.get("rain_amount", 0.0) * 3.0,
                    snow_mm=kw.get("snow_amount", 0.0) * 3.0,
                    wind_speed=kw.get("wind_speed", 2.0),
                    visibility=kw.get("visibility", 10000),
                ))
                self.assertEqual(sa.kind, sb.kind)

    def test_calm_and_pleasant_rules_respect_wind_gusts(self) -> None:
        # Every calm/pleasant rule must be suppressed when wind_speed is low but wind_gust is strong (13 m/s)
        calm_cases = [
            ("THE UNICORN", dict(temp=20.0, wind_speed=1.0, humidity=55)),
            ("THE VOID", dict(temp=10.0, wind_speed=0.5, uv_index=1.5, cloud_pct=50)),
            ("THE GOLDILOCKS ZONE", dict(temp=23.0, wind_speed=2.0, rain_prob=5.0, cloud_pct=30)),
            ("TERRACE ESPRESSO", dict(temp=19.0, local_hour=13, wind_speed=2.5, rain_prob=10.0, cloud_pct=50)),
            ("GARDEN WARRIOR", dict(temp=14.0, desc="clear sky", wind_speed=1.0, uv_index=3.0)),
            ("THE TERRARIUM", dict(temp=22.0, humidity=78, wind_speed=1.0)),
            ("SÄÄSED", dict(country="EE", temp=21.0, humidity=80, local_hour=19, wind_speed=1.0)),
            ("SUITSUSAUN", dict(country="EE", temp=3.0, local_hour=20, is_day=False, wind_speed=1.5, cloud_pct=85)),
            ("BLUEBIRD FREEZE", dict(temp=-2.0, is_day=True, cloud_pct=15, wind_speed=1.5)),
            ("STARRY FREEZE", dict(temp=-1.0, local_hour=23, is_day=False, cloud_pct=15, wind_speed=1.5)),
            ("THE DRY HEAT", dict(temp=28.0, humidity=20, wind_speed=1.5)),
        ]
        for rule_name, kw in calm_cases:
            with self.subTest(rule=rule_name):
                self.assertEqual(engine.evaluate_vibe(make_payload(**kw)).name, rule_name)
                gusty = engine.evaluate_vibe(make_payload(**kw, wind_gust=13.0))
                self.assertNotEqual(gusty.name, rule_name, f"{rule_name} ignored 13 m/s gust!")
                self.assertIn("wind", gusty.factors)

    def test_gusts_trigger_wet_snow_sleet_and_gale_compounds(self) -> None:
        # Moderate sustained 6.5 m/s + 11.0 m/s gust -> "strong" tier
        self.assertEqual(
            engine.evaluate_vibe(make_payload(temp=12.0, desc="light rain", rain_amount=0.8, wind_speed=6.5, wind_gust=11.0)).name,
            "THE HORIZONTAL SHOWER",
        )
        self.assertEqual(
            engine.evaluate_vibe(make_payload(temp=-2.0, desc="snow", snow_amount=0.8, wind_speed=6.5, wind_gust=11.0)).name,
            "BLIZZARD",
        )
        self.assertEqual(
            engine.evaluate_vibe(make_payload(temp=0.5, desc="sleet", rain_amount=0.4, snow_amount=0.4, wind_speed=6.5, wind_gust=11.0)).name,
            "WINTRY GALE",
        )
        # Low sustained 3.0 m/s + 22.0 m/s gale gust -> "gale" tier
        self.assertEqual(
            engine.evaluate_vibe(make_payload(temp=12.0, desc="light rain", rain_amount=0.5, wind_speed=3.0, wind_gust=22.0)).name,
            "THE TEMPEST",
        )
        self.assertEqual(
            engine.evaluate_vibe(make_payload(temp=-2.0, wind_speed=3.0, wind_gust=22.0)).name,
            "ARCTIC GALE",
        )

    def test_monotonic_escalation_never_drops_overlapping_factor(self) -> None:
        # 1. Increasing wind during rain keeps "rain" factor all the way to gale
        for ws in (1.0, 5.5, 9.0, 12.5, 16.5):
            v = engine.evaluate_vibe(make_payload(temp=12.0, desc="moderate rain", rain_amount=2.0, wind_speed=ws, cloud_pct=95, uv_index=0.5))
            self.assertIn("rain", v.factors, f"rain erased at wind={ws} -> {v.name}")
            if ws >= 8.0:
                self.assertIn("wind", v.factors, f"wind missing at wind={ws} -> {v.name}")
                self.assertTrue(v.is_compound)

        # 2. Adding rain to strong/gale wind keeps "wind" factor and adds "rain"
        for ws in (9.0, 12.0, 16.5):
            dry_v = engine.evaluate_vibe(make_payload(temp=12.0, wind_speed=ws))
            wet_v = engine.evaluate_vibe(make_payload(temp=12.0, desc="light rain", rain_amount=1.0, wind_speed=ws))
            self.assertIn("wind", dry_v.factors)
            self.assertIn("wind", wet_v.factors)
            self.assertIn("rain", wet_v.factors)
            self.assertTrue(wet_v.is_compound)

        # 3. Increasing wind during snow keeps "snow" factor all the way to gale
        for ws in (1.0, 5.5, 9.0, 16.5):
            v = engine.evaluate_vibe(make_payload(temp=-2.0, desc="snow", snow_amount=1.0, wind_speed=ws))
            self.assertIn("snow", v.factors, f"snow erased at wind={ws} -> {v.name}")
            if ws >= 8.0:
                self.assertEqual(v.name, "BLIZZARD")

        # 4. Dropping temperature during rain progresses smoothly to cold rain -> freezing rain
        sev_rank = {s: i for i, s in enumerate(engine.SEVERITY_LEVELS)}
        prev_rank = -1
        for t in (24.0, 18.0, 11.0, 3.0, 0.0):
            v = engine.evaluate_vibe(make_payload(temp=t, desc="light rain", rain_amount=0.8, wind_speed=2.0, cloud_pct=90, uv_index=0.5))
            self.assertIn("rain", v.factors, f"rain erased at temp={t} -> {v.name}")
            r = sev_rank[v.severity]
            self.assertGreaterEqual(r, prev_rank, f"severity regressed at temp={t}: {v.name}")
            prev_rank = r
        self.assertEqual(v.name, "BLACK ICE PROTOCOL")

    def test_band_totality_invariants_across_combinatorial_grid(self) -> None:
        daytime_only = {
            "THE SURFACE OF THE SUN", "HOT DRY WIND", "THE DRY HEAT", "THE SAUNA SESSION",
            "BREEZY HEAT", "THE DEADLY LASER", "SUNSHOWER", "BLUEBIRD FREEZE",
            "THE CRISP BAKE", "FOOL'S SUMMER", "TERRACE ESPRESSO",
        }
        temps = (-5.0, 0.0, 3.0, 10.0, 20.0, 28.0, 33.0)
        winds = (1.0, 6.0, 9.5, 16.0)
        precip_modes = (
            ("scattered clouds", 0.0, 0.0, 10000),
            ("fog", 0.0, 0.0, 400),
            ("light rain", 0.8, 0.0, 8000),
            ("heavy intensity rain", 5.0, 0.0, 4000),
            ("snow", 0.0, 1.0, 3000),
            ("sleet", 0.5, 0.5, 3000),
        )
        hours = ((13, True), (23, False))

        checked = 0
        for t in temps:
            for ws in winds:
                for desc, r_mm, s_mm, vis in precip_modes:
                    for hr, is_day in hours:
                        w = make_payload(
                            temp=t, wind_speed=ws, desc=desc,
                            rain_amount=r_mm, snow_amount=s_mm, visibility=vis,
                            local_hour=hr, is_day=is_day,
                            cloud_pct=90 if (r_mm > 0 or s_mm > 0 or vis < 1000) else 40,
                            uv_index=0.0 if not is_day else 3.0,
                        )
                        v = engine.evaluate_vibe(w)
                        checked += 1
                        # Wet or snowy/sleet/thunder must resolve in Band A/B (>= 780)
                        if w.is_wet or s_mm > 0.0 or desc in ("snow", "sleet"):
                            self.assertGreaterEqual(v.priority, 780, f"wet leaked to {v.name}: {w}")
                        # Dense fog must resolve in Band A/B (>= 805)
                        if w.is_dense_fog:
                            self.assertGreaterEqual(v.priority, 805, f"fog leaked to {v.name}: {w}")
                        # Gale must resolve in Band A (>= 925)
                        if ws >= 15.0:
                            self.assertGreaterEqual(v.priority, 925, f"gale leaked to {v.name}: {w}")
                        # Night / after-dark never triggers daytime-only heat/sun rules
                        if w.is_after_dark:
                            self.assertNotIn(v.name, daytime_only, f"after-dark hit daytime rule {v.name}: {w}")
                        # Zero accidental FORREST GUMP fallthroughs
                        self.assertNotEqual(v.name, "FORREST GUMP", f"unexpected FORREST GUMP fallthrough: {w}")
        self.assertGreaterEqual(checked, 300)


# =====================================================================
# 4. FORECAST STEP SYNERGY & 100% STEP_KIND REACHABILITY
# =====================================================================


STEP_KIND_WITNESSES: dict[str, engine.ForecastStep] = {
    "thunder_squall": make_step(desc="thunderstorm with rain", rain_mm=3.0, wind_speed=9.0),
    "thunderstorm": make_step(desc="light thunderstorm", rain_mm=0.6, wind_speed=2.0),
    "ice_storm": make_step(temp=-1.0, desc="freezing rain", rain_mm=1.5, wind_speed=6.0),
    "freezing_rain": make_step(temp=-1.0, desc="freezing rain", rain_mm=1.5, wind_speed=2.0),
    "blizzard": make_step(temp=-3.0, desc="snow", snow_mm=2.0, wind_speed=9.0),
    "wintry_gale": make_step(temp=0.5, desc="sleet", rain_mm=0.5, snow_mm=0.5, wind_speed=9.0),
    "wet_gale": make_step(temp=12.0, desc="light rain", rain_mm=1.5, wind_speed=16.0),
    "arctic_gale": make_step(temp=-2.0, wind_speed=16.0),
    "freezing_fog": make_step(temp=-2.0, desc="fog", visibility=350, wind_speed=2.0),
    "stormy_rain": make_step(temp=12.0, desc="moderate rain", rain_mm=3.0, wind_speed=9.0),
    "cold_lashing_rain": make_step(temp=3.0, desc="light rain", rain_mm=1.5, wind_speed=9.0),
    "foggy_rain": make_step(temp=10.0, desc="light rain", all_descs=("light rain", "fog"), rain_mm=1.5, visibility=400, wind_speed=2.0),
    "windy_fog": make_step(temp=10.0, desc="fog", visibility=400, wind_speed=9.0),
    "mixed_precip": make_step(temp=0.5, desc="rain and snow", rain_mm=0.5, snow_mm=0.5, wind_speed=2.0),
    "cold_rain": make_step(temp=3.0, desc="light rain", rain_mm=1.0, wind_speed=2.0),
    "misty_drizzle": make_step(temp=10.0, desc="light intensity drizzle", all_descs=("light intensity drizzle", "mist"), rain_mm=0.9, visibility=1800, wind_speed=2.0),
    "bitter_wind": make_step(temp=4.0, wind_speed=9.0),
    "gale": make_step(temp=12.0, wind_speed=16.0),
    "heavy_rain": make_step(temp=14.0, desc="heavy intensity rain", rain_mm=13.0, wind_speed=2.0),
    "heavy_snow": make_step(temp=-2.0, desc="heavy snow", snow_mm=4.0, wind_speed=2.0),
    "dense_fog": make_step(temp=10.0, desc="fog", visibility=400, wind_speed=2.0),
    "bitter_cold": make_step(temp=-6.0, wind_speed=2.0),
    "hot_dry_wind": make_step(temp=29.0, humidity=25, wind_speed=6.0),
    "muggy_heat": make_step(temp=28.0, humidity=82, wind_speed=2.0),
    "extreme_heat": make_step(temp=33.0, humidity=50, wind_speed=2.0),
    "rain": make_step(temp=14.0, desc="light rain", rain_mm=1.5, wind_speed=2.0),
    "snow": make_step(temp=-2.0, desc="light snow", snow_mm=1.0, wind_speed=2.0),
    "windy": make_step(temp=16.0, wind_speed=9.0),
    "calm": make_step(temp=18.0, desc="few clouds", wind_speed=2.0),
}


class ForecastStepSynergyTests(unittest.TestCase):
    def test_all_step_kinds_are_reachable_and_valid(self) -> None:
        self.assertEqual(set(STEP_KIND_WITNESSES), set(engine.STEP_KIND_LABELS))
        for expected_kind, step in STEP_KIND_WITNESSES.items():
            with self.subTest(kind=expected_kind):
                sc = engine.classify_step(step)
                self.assertEqual(sc.kind, expected_kind)
                self.assertEqual(sc.label, engine.STEP_KIND_LABELS[expected_kind])
                self.assertIn(sc.severity, engine.SEVERITY_LEVELS)
                for f in sc.factors:
                    self.assertIn(f, engine.FACTOR_VOCAB)
                if sc.is_compound:
                    self.assertGreaterEqual(len(sc.factors), 2)

    def test_step_precedence_gusts_and_high_pop_fallback(self) -> None:
        # Gust-triggered ice storm and blizzard
        self.assertEqual(
            engine.classify_step(make_step(temp=-0.5, desc="freezing rain", rain_mm=1.0, wind_speed=3.0, wind_gust=9.5)).kind,
            "ice_storm",
        )
        self.assertEqual(
            engine.classify_step(make_step(temp=-2.0, desc="snow", snow_mm=1.0, wind_speed=6.5, wind_gust=11.0)).kind,
            "blizzard",
        )
        # Gale + snow -> blizzard, gale + rain -> wet_gale (never dry gale)
        self.assertEqual(
            engine.classify_step(make_step(temp=-2.0, desc="snow", snow_mm=1.0, wind_speed=16.0)).kind,
            "blizzard",
        )
        self.assertEqual(
            engine.classify_step(make_step(temp=12.0, desc="light rain", rain_mm=1.0, wind_speed=16.0)).kind,
            "wet_gale",
        )
        # High POP (>= 0.65) dry step reflects thermal phase
        self.assertEqual(engine.classify_step(make_step(temp=-2.0, pop=0.75, desc="overcast clouds")).kind, "snow")
        self.assertEqual(engine.classify_step(make_step(temp=10.0, pop=0.75, desc="overcast clouds")).kind, "rain")

    def test_step_context_renders_cohesive_compound_labels_in_en_and_et(self) -> None:
        compound_steps = [
            ("stormy_rain", "bar storm", make_step(temp=12.0, desc="moderate rain", rain_mm=3.0, wind_speed=9.5, wind_gust=14.0)),
            ("blizzard", "bar snow", make_step(temp=-3.0, desc="snow", snow_mm=2.5, wind_speed=10.0, wind_gust=14.5)),
            ("ice_storm", "bar ice", make_step(temp=-1.0, desc="freezing rain", rain_mm=1.8, wind_speed=6.0, wind_gust=11.0)),
            ("mixed_precip", "bar mix", make_step(temp=0.5, desc="rain and snow", rain_mm=0.6, snow_mm=0.8, wind_speed=3.0)),
            ("freezing_fog", "bar", make_step(temp=-4.0, desc="fog", visibility=350, wind_speed=2.0)),
            ("foggy_rain", "bar", make_step(temp=9.0, desc="light rain", all_descs=("light rain", "fog"), rain_mm=1.5, visibility=400)),
        ]
        for expected_kind, expected_bar, step in compound_steps:
            with self.subTest(kind=expected_kind):
                flag_en, _, bar_en, extras_en = build_site._step_context(step, is_ee=False, include_wind=True)
                flag_et, _, bar_et, extras_et = build_site._step_context(step, is_ee=True, include_wind=True)
                self.assertIn("[!]", flag_en)
                self.assertIn("[!]", flag_et)
                self.assertEqual(bar_en, expected_bar)
                self.assertEqual(bar_et, expected_bar)
                self.assertIn(engine.STEP_KIND_LABELS[expected_kind], extras_en)
                self.assertIn(vibe_ee.translate_step_kind_et(expected_kind, engine.STEP_KIND_LABELS[expected_kind]), extras_et)

        # Tiered indicators: calm -> [-], advisory -> [~], and snow whiteout doesn't tack on false Dense fog
        calm_flag, _, _, _ = build_site._step_context(make_step(temp=18.0, desc="few clouds"), is_ee=False)
        adv_flag, _, _, _ = build_site._step_context(make_step(temp=12.0, desc="light rain", rain_mm=1.2), is_ee=False)
        self.assertEqual(calm_flag, "[-]")
        self.assertIn("[~]", adv_flag)
        _, _, _, bliz_detail = build_site._step_context(
            make_step(temp=-4.0, desc="heavy snow", snow_mm=4.0, wind_speed=11.0, visibility=400),
            is_ee=False,
        )
        self.assertIn("Blizzard", bliz_detail)
        self.assertNotIn("Dense fog", bliz_detail)


# =====================================================================
# 5. CATALOG INTEGRITY, PARITY & 100% VIBE_RULES REACHABILITY
# =====================================================================


VIBE_RULE_WITNESSES: dict[str, dict] = {
    # Band A (12)
    "SUPERCELL SQUALL": dict(temp=20.0, desc="thunderstorm with rain", rain_amount=1.0, wind_speed=9.0),
    "ICE STORM": dict(temp=-1.0, desc="freezing rain", rain_amount=1.0, wind_speed=6.0),
    "BLIZZARD": dict(temp=-3.0, desc="snow", snow_amount=1.0, wind_speed=9.0),
    "THE TEMPEST": dict(temp=12.0, desc="light rain", rain_amount=1.0, wind_speed=16.0),
    "ARCTIC GALE": dict(temp=-2.0, wind_speed=16.0),
    "WINTRY GALE": dict(temp=0.5, desc="sleet", rain_amount=0.5, snow_amount=0.5, wind_speed=9.0),
    "FROSTBOUND FOG": dict(temp=-2.0, desc="fog", visibility=350, wind_speed=2.0),
    "GHOSTBUSTERS": dict(temp=18.0, desc="light thunderstorm", rain_amount=0.5, wind_speed=2.0),
    "BLACK ICE PROTOCOL": dict(temp=-1.0, desc="freezing rain", rain_amount=0.8, wind_speed=2.0),
    "THE EYEWALL": dict(temp=12.0, wind_speed=16.0),
    "SLUSH SEASON": dict(temp=0.5, desc="light rain and snow", rain_amount=0.4, snow_amount=0.4, wind_speed=2.0),
    "HOTH": dict(temp=-2.0, desc="light snow", snow_amount=0.5, wind_speed=2.0),
    # Band B (25)
    "TROPICAL SQUALL": dict(temp=24.0, desc="moderate rain", rain_amount=2.0, wind_speed=9.0),
    "STORMY DOWNPOUR": dict(temp=14.0, desc="heavy intensity rain", rain_amount=5.0, wind_speed=6.0),
    "LIBECCIO": dict(temp=16.0, wind_deg=225, wind_speed=10.0),
    "NORTH SEA LASHING": dict(temp=3.0, desc="light rain", rain_amount=0.8, wind_speed=9.0),
    "THE HORIZONTAL SHOWER": dict(temp=12.0, desc="light rain", rain_amount=0.8, wind_speed=9.0),
    "FOGBOUND RAIN": dict(temp=10.0, desc="light rain", all_descs=("light rain", "fog"), rain_amount=1.0, visibility=400, wind_speed=2.0),
    "SEA FOG DRIFT": dict(temp=8.0, desc="fog", visibility=400, wind_speed=9.0),
    "SMOG TRAP": dict(temp=18.0, desc="haze", visibility=2500, wind_speed=1.5, aqi=5),
    "THE STEAM ROOM": dict(temp=24.0, desc="fog", visibility=500, wind_speed=2.0),
    "SUNSHOWER": dict(temp=16.0, desc="light intensity shower rain", rain_amount=0.6, wind_speed=2.0, uv_index=4.0, cloud_pct=40),
    "TROPICAL DEPRESSION": dict(temp=23.0, desc="heavy intensity rain", rain_amount=5.0, wind_speed=2.0, uv_index=1.0, cloud_pct=95),
    "THE MARROW SOAK": dict(temp=3.0, desc="light rain", rain_amount=0.5, wind_speed=2.0, uv_index=0.5, cloud_pct=90),
    "MISTY DRIZZLE": dict(temp=10.0, desc="light intensity drizzle", all_descs=("light intensity drizzle", "mist"), rain_amount=0.3, visibility=1800, wind_speed=2.0, uv_index=0.5, cloud_pct=90),
    "THE SPONGE": dict(temp=24.0, desc="light rain", rain_amount=0.5, wind_speed=2.0, uv_index=1.0, cloud_pct=90),
    "THE LUKEWARM WASH": dict(temp=18.0, desc="moderate rain", rain_amount=2.2, wind_speed=2.0, uv_index=1.0, cloud_pct=90),
    "CHILL FOG": dict(temp=2.5, desc="fog", visibility=400, wind_speed=2.0),
    "NOAH'S ARK": dict(temp=12.0, desc="heavy intensity rain", rain_amount=5.0, wind_speed=2.0, uv_index=0.5, cloud_pct=95),
    "PROPER RAIN": dict(temp=12.0, desc="moderate rain", rain_amount=2.0, wind_speed=2.0, uv_index=0.5, cloud_pct=95),
    "SILENT HILL": dict(temp=12.0, desc="fog", visibility=400, wind_speed=2.0),
    "PERPETUAL GLOOM": dict(temp=10.0, desc="light intensity drizzle", rain_amount=0.2, rain_prob=85.0, wind_speed=2.0, uv_index=0.5, cloud_pct=90),
    "SPITTING COBWEBS": dict(temp=18.0, desc="light intensity drizzle", rain_amount=0.3, rain_prob=50.0, wind_speed=2.0, uv_index=1.0, cloud_pct=85),
    "THE OLYMPIC SPORT": dict(temp=10.0, desc="light rain", rain_amount=0.6, rain_prob=60.0, wind_speed=2.0, local_hour=8, uv_index=0.5, cloud_pct=90),
    "THE BELGIAN DRIZZLE": dict(temp=10.0, desc="light rain", rain_amount=0.6, rain_prob=60.0, wind_speed=2.0, local_hour=13, uv_index=0.5, cloud_pct=90),
    "LA PIOVIGGINE": dict(temp=10.0, desc="light rain", rain_amount=0.6, rain_prob=60.0, wind_speed=2.0, local_hour=13, uv_index=0.5, cloud_pct=90),
    "UDUVIHM": dict(temp=10.0, desc="light rain", rain_amount=0.6, rain_prob=60.0, wind_speed=2.0, local_hour=13, uv_index=0.5, cloud_pct=90),
    # Band C (33)
    "STORM BREWING": dict(temp=16.0, wind_speed=4.0, forecast_steps=[make_step(desc="thunderstorm", rain_mm=3.0, wind_speed=10.0)]),
    "BOILED ALIVE": dict(temp=33.0, humidity=80, wind_speed=2.0),
    "SCIROCCO": dict(temp=26.0, humidity=65, wind_deg=135, wind_speed=4.5),
    "SWAMP NIGHT": dict(temp=23.0, humidity=82, local_hour=23, is_day=False, wind_speed=2.0),
    "TROPICAL NIGHT": dict(temp=27.0, humidity=55, local_hour=23, is_day=False, wind_speed=2.0),
    "TRAMONTANA": dict(temp=6.0, wind_deg=10, wind_speed=7.0),
    "THE WHITE WALKER": dict(temp=-2.0, local_hour=23, is_day=False, wind_speed=6.0),
    "ARCTIC WINDCHILL": dict(temp=-5.0, wind_speed=9.0),
    "RAW DAMP GALE": dict(temp=3.0, humidity=92, wind_speed=9.0),
    "THE WIND TUNNEL": dict(temp=4.0, humidity=60, wind_speed=9.0),
    "BLUEBIRD FREEZE": dict(temp=-2.0, is_day=True, cloud_pct=15, wind_speed=2.0),
    "STARRY FREEZE": dict(temp=-1.0, local_hour=23, is_day=False, cloud_pct=15, wind_speed=2.0),
    "BALTIC FREEZE": dict(temp=-5.0, is_day=True, cloud_pct=90, wind_speed=2.0),
    "BONE CHILL": dict(temp=3.0, humidity=95, wind_speed=2.0, cloud_pct=90),
    "HOT DRY WIND": dict(temp=29.0, humidity=25, wind_speed=6.0),
    "THE SURFACE OF THE SUN": dict(temp=33.0, humidity=50, wind_speed=2.0),
    "THE DRY HEAT": dict(temp=28.0, humidity=20, wind_speed=2.0),
    "THE SAUNA SESSION": dict(temp=28.0, humidity=82, wind_speed=2.0),
    "BREEZY HEAT": dict(temp=28.0, humidity=55, wind_speed=6.0),
    "THE DEADLY LASER": dict(temp=28.0, humidity=55, wind_speed=2.0, uv_index=8.5),
    "THE INVISIBLE HAND": dict(temp=16.0, wind_speed=12.0),
    "BAD HAIR DAY": dict(temp=16.0, wind_speed=9.0),
    "SNEAKY SUNBURN": dict(temp=21.0, wind_speed=6.0, uv_index=6.0),
    "THE WIND CHILL": dict(temp=12.0, wind_speed=6.0, uv_index=4.5),
    "THE DECEPTIVE BREEZE": dict(temp=18.0, wind_speed=7.0, uv_index=2.0),
    "THE THREAT": dict(temp=15.0, rain_prob=70.0, cloud_pct=85, wind_speed=2.0),
    "SÄÄSED": dict(temp=21.0, humidity=80, local_hour=19, wind_speed=1.0),
    "VALGED ÖÖD": dict(temp=14.0, local_hour=22, is_day=True, wind_speed=2.0),
    "THE TERRARIUM": dict(temp=22.0, humidity=78, wind_speed=1.0, local_hour=13),
    "THE MUGGY MIDDLE": dict(temp=19.0, humidity=85, wind_speed=3.0, local_hour=13),
    "THE CRISP BAKE": dict(temp=18.0, uv_index=6.0, wind_speed=2.0),
    "FOOL'S SUMMER": dict(temp=15.0, uv_index=4.5, wind_speed=2.0),
    "SCHRÖDINGER'S SHOWER": dict(temp=16.0, rain_prob=50.0, wind_speed=2.0, uv_index=2.0),
    # Band D (21)
    "EARLY OVEN": dict(temp=23.0, local_hour=8, wind_speed=2.0, uv_index=2.0, humidity=55),
    "MORNING SOUP": dict(temp=19.0, local_hour=8, humidity=82, wind_speed=2.0, uv_index=1.0),
    "THE MORNING FROST": dict(temp=1.5, local_hour=8, wind_speed=2.0, humidity=70, cloud_pct=80),
    "BRISK MORROW": dict(temp=10.0, local_hour=8, wind_speed=2.0, uv_index=1.5, humidity=60),
    "PIMEDUS": dict(temp=4.0, local_hour=15, is_day=False, wind_speed=2.0, humidity=75),
    "SUITSUSAUN": dict(temp=3.0, local_hour=20, is_day=False, wind_speed=2.0, humidity=75, cloud_pct=80),
    "THE RARE EVENING": dict(temp=20.5, humidity=50, local_hour=22, is_day=False, wind_speed=2.0),
    "MIDNIGHT MAGIC": dict(temp=17.0, humidity=65, local_hour=22, is_day=False, wind_speed=2.0),
    "NIGHT CRAWLER": dict(temp=12.0, humidity=65, local_hour=22, is_day=False, wind_speed=2.0),
    "COCOA TIME": dict(temp=6.0, humidity=70, local_hour=2, is_day=False, wind_speed=2.0, cloud_pct=80),
    "THE UNICORN": dict(temp=20.0, wind_speed=1.0, humidity=55, uv_index=2.0),
    "LA PASSEGGIATA": dict(temp=22.0, local_hour=19, wind_speed=2.5, humidity=55, uv_index=2.0),
    "THE GOLDILOCKS ZONE": dict(temp=23.0, wind_speed=2.0, rain_prob=5.0, cloud_pct=30, uv_index=3.0),
    "GARDEN WARRIOR": dict(temp=14.0, desc="clear sky", wind_speed=1.0, uv_index=3.0),
    "THE FALSE SPRING": dict(temp=9.0, uv_index=3.5, wind_speed=2.0),
    "THE MATTE FINISH": dict(temp=19.0, desc="broken clouds", humidity=40, wind_speed=2.5, uv_index=2.0),
    "TERRACE ESPRESSO": dict(temp=19.0, local_hour=13, wind_speed=3.0, rain_prob=10.0, cloud_pct=50, uv_index=2.0),
    "BIG GREEN EGG TERRITORY": dict(temp=19.0, local_hour=16, wind_speed=3.0, rain_prob=10.0, cloud_pct=50, uv_index=2.0),
    "THE PATIO GAMBLE": dict(temp=19.0, local_hour=16, wind_speed=3.0, rain_prob=25.0, cloud_pct=50, uv_index=2.0),
    "THE VOID": dict(temp=10.0, wind_speed=0.5, uv_index=1.5, cloud_pct=50),
    "OVERCAST VELVET": dict(temp=15.0, desc="overcast clouds", cloud_pct=90, wind_speed=3.0, uv_index=1.0),
    # Band E (7)
    "THE BRISK BREEZE": dict(temp=18.0, wind_speed=5.5, uv_index=2.0),
    "THE ROASTING SPIT": dict(temp=28.0, humidity=55, wind_speed=2.0, uv_index=4.0),
    "LAYERING SEASON": dict(temp=14.0, desc="scattered clouds", cloud_pct=50, wind_speed=3.0, uv_index=2.0),
    "THE BELGIAN STANDARD": dict(temp=7.0, desc="broken clouds", cloud_pct=60, wind_speed=3.0, uv_index=1.5),
    "CASTAGNE E CAMINO": dict(temp=7.0, desc="broken clouds", cloud_pct=60, wind_speed=3.0, uv_index=1.5),
    "THE ESTONIAN STANDARD": dict(temp=7.0, desc="broken clouds", cloud_pct=60, wind_speed=3.0, uv_index=1.5),
    "WINTER IS COMING": dict(temp=1.5, desc="overcast clouds", cloud_pct=85, wind_speed=3.0, humidity=70, uv_index=0.5),
}


class CatalogIntegrityAndReachabilityTests(unittest.TestCase):
    def test_all_vibe_rules_are_reachable_in_every_eligible_region(self) -> None:
        rule_names = [r.name for r in engine.VIBE_RULES]
        self.assertEqual(len(rule_names), len(set(rule_names)), "Duplicate rule names in VIBE_RULES")
        self.assertEqual(set(VIBE_RULE_WITNESSES), set(rule_names))

        for rule in engine.VIBE_RULES:
            kw = VIBE_RULE_WITNESSES[rule.name]
            eligible_regions = [reg for reg in sorted(engine.SUPPORTED_REGIONS) if rule.applies_to(reg)]
            self.assertTrue(eligible_regions, f"Rule {rule.name!r} applies to no region")
            for reg in eligible_regions:
                with self.subTest(rule=rule.name, region=reg):
                    payload = make_payload(country=reg, **kw)
                    matched = engine.evaluate_vibe(payload)
                    self.assertEqual(
                        matched.name, rule.name,
                        f"{rule.name} shadowed by {matched.name} in {reg}",
                    )
                    # Backwards-compatible condition(payload) must agree with matches(payload, profile)
                    self.assertTrue(rule.condition(payload))

    def test_engine_metadata_priorities_palette_and_synergy_labels(self) -> None:
        # Unique priorities within each region
        for reg, rules in engine._RULES_BY_REGION.items():
            priorities = [r.priority for r in rules]
            self.assertEqual(len(priorities), len(set(priorities)), f"Duplicate priority in {reg}")

        used_synergy_keys: set[str] = set()
        for rule in (*engine.VIBE_RULES, *engine.DEFAULT_VIBES.values()):
            self.assertIn(rule.color, build_site.COLORS, f"{rule.name} color {rule.color!r} missing from build_site.COLORS")
            self.assertIn(rule.severity, engine.SEVERITY_LEVELS, f"{rule.name} invalid severity {rule.severity!r}")
            for f in rule.factors:
                self.assertIn(f, engine.FACTOR_VOCAB, f"{rule.name} unknown factor {f!r}")
            if rule.synergy_key:
                self.assertIn(rule.synergy_key, engine.SYNERGY_LABELS)
                self.assertEqual(rule.synergy_label, engine.SYNERGY_LABELS[rule.synergy_key])
                used_synergy_keys.add(rule.synergy_key)
            if rule.is_compound:
                self.assertGreaterEqual(len(rule.factors), 2, f"Compound rule {rule.name} has <2 factors")
                self.assertTrue(rule.synergy_key, f"Compound rule {rule.name} missing synergy_key")

        # Zero dead SYNERGY_LABELS entries
        self.assertEqual(set(engine.SYNERGY_LABELS) - used_synergy_keys, set())

    def test_be_vibe_catalog_strict_parity_and_quality(self) -> None:
        be_eligible = {r.name for r in engine.VIBE_RULES if r.applies_to("BE")} | {engine.DEFAULT_VIBES["BE"].name}
        be_keys = set(vibe_be._BY_ORIG_NAME)
        self.assertEqual(be_eligible - be_keys, set(), "Missing BE vibe entries")
        self.assertEqual(be_keys - be_eligible, set(), "Orphan/stale BE vibe entries")

        orig_names = [v["orig_name"] for v in vibe_be.VIBES_BE.values()]
        self.assertEqual(len(orig_names), len(set(orig_names)), "Duplicate orig_name in VIBES_BE")
        self.assertEqual(set(orig_names), be_eligible)

        picked_titles: dict[str, str] = {}
        for vid, entry in vibe_be.VIBES_BE.items():
            with self.subTest(be_id=vid, orig_name=entry["orig_name"]):
                self.assertTrue(entry.get("trigger", "").strip())
                self.assertTrue(entry["orig_quote"].strip())
                pick = entry.get("pick", "").lower()
                self.assertIn(pick, ("orig", "a", "b", "c", "d", "e", "f", "g"))
                for slot in ("a", "b", "c", "d", "e", "f", "g"):
                    alt = entry[slot]
                    self.assertIsInstance(alt, tuple)
                    self.assertEqual(len(alt), 2)
                    self.assertTrue(alt[0].strip() and alt[1].strip())
                title, quote = vibe_be.get_be_vibe(entry["orig_name"], "DEFAULT_Q")
                self.assertTrue(title.strip() and quote.strip())
                self.assertNotIn(title, picked_titles, f"Duplicate BE picked title {title!r}: {picked_titles.get(title)} vs {entry['orig_name']}")
                picked_titles[title] = entry["orig_name"]

        # Fallback signature check
        self.assertEqual(vibe_be.get_be_vibe("__UNKNOWN__", "fallback quote"), ("__UNKNOWN__", "fallback quote"))

    def test_ee_vibe_catalog_strict_parity_translations_and_quality(self) -> None:
        ee_eligible = {r.name for r in engine.VIBE_RULES if r.applies_to("EE")} | {engine.DEFAULT_VIBES["EE"].name}
        ee_keys = set(vibe_ee._BY_ORIG_NAME)
        self.assertEqual(ee_eligible - ee_keys, set(), "Missing EE vibe entries")
        self.assertEqual(ee_keys - ee_eligible, set(), "Orphan/stale EE vibe entries")

        orig_names = [v["orig_name"] for v in vibe_ee.VIBES_EE.values()]
        self.assertEqual(len(orig_names), len(set(orig_names)), "Duplicate orig_name in VIBES_EE")
        self.assertEqual(set(orig_names), ee_eligible)

        picked_titles: dict[str, str] = {}
        for vid, entry in vibe_ee.VIBES_EE.items():
            with self.subTest(ee_id=vid, orig_name=entry["orig_name"]):
                self.assertTrue(entry.get("trigger", "").strip())
                self.assertTrue(entry["orig_quote"].strip())
                pick = entry.get("pick", "").lower()
                self.assertIn(pick, ("a", "b", "c"))
                for slot in ("a", "b", "c"):
                    alt = entry[slot]
                    self.assertIsInstance(alt, tuple)
                    self.assertEqual(len(alt), 2)
                    self.assertTrue(alt[0].strip() and alt[1].strip())
                title, quote = vibe_ee.get_ee_vibe(entry["orig_name"], "DEFAULT_Q")
                self.assertTrue(title.strip() and quote.strip())
                self.assertNotIn(title, picked_titles, f"Duplicate EE picked title {title!r}: {picked_titles.get(title)} vs {entry['orig_name']}")
                picked_titles[title] = entry["orig_name"]

        # Fallback signature check
        self.assertEqual(vibe_ee.get_ee_vibe("__UNKNOWN__", "fallback quote"), ("__UNKNOWN__", "fallback quote"))

        # 100% Estonian label coverage for FACTOR_VOCAB, SYNERGY_LABELS, STEP_KIND_LABELS
        self.assertEqual(set(engine.FACTOR_VOCAB) - set(vibe_ee.FACTOR_LABELS_ET), set())
        for f in engine.FACTOR_VOCAB:
            self.assertTrue(vibe_ee.translate_factor_et(f).strip())

        self.assertEqual(set(engine.SYNERGY_LABELS) - set(vibe_ee.SYNERGY_LABELS_ET), set())
        for k, en_lbl in engine.SYNERGY_LABELS.items():
            self.assertTrue(vibe_ee.translate_synergy_et(k, en_lbl).strip())

        self.assertEqual(set(engine.STEP_KIND_LABELS) - set(vibe_ee.STEP_KIND_LABELS_ET), set())
        for k, en_lbl in engine.STEP_KIND_LABELS.items():
            self.assertTrue(vibe_ee.translate_step_kind_et(k, en_lbl).strip())

        # Weather description translation fixes (longest-first thunder preservation, unknown, typos)
        self.assertEqual(vibe_ee.translate_weather_desc("freezing fog"), "Jäätuv udu")
        self.assertEqual(vibe_ee.translate_weather_desc("freezing drizzle"), "Jäätuv uduvihm")
        self.assertNotEqual(vibe_ee.translate_weather_desc("unknown").lower(), "unknown")
        self.assertIn("äike", vibe_ee.translate_weather_desc("thunderstorm with heavy shower rain").lower())
        self.assertIn("äike", vibe_ee.translate_weather_desc("heavy thunderstorm with rain").lower())
        self.assertNotEqual(vibe_ee.translate_weather_desc("ragged shower rain").lower(), "hooline vihm")
        self.assertNotEqual(vibe_ee.translate_weather_desc("sand/ dust whirls").lower(), "liiv")

        # UI translation & styling tables coverage
        self.assertEqual(set(engine.FACTOR_VOCAB) - set(build_site.FACTOR_LABELS_EN), set())
        self.assertEqual(set(engine.STEP_KIND_LABELS) - set(build_site.STEP_STYLES), set())
        expected_lighting = {
            "Fog / Low Visibility",
            "Gloomy / Low Diffuse Lux",
            "Hazy / Diffused Mist",
            "Filtered Sun / Soft Glare",
            "Direct Bright Sun",
            "Darkness",
        }
        self.assertEqual(expected_lighting - set(build_site.LIGHTING_ET), set())
        self.assertEqual(set(range(1, 6)) - set(build_site.AQI_ET), set())


if __name__ == "__main__":
    unittest.main()
