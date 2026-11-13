"""Unit tests for weatherweb engine telemetry, forecast precision, and HTML rendering."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx

import build_site
import engine
import vibe_be
import vibe_ee


class TelemetryPrecisionTests(unittest.TestCase):
    def test_dew_point_and_clamp(self) -> None:
        dp = engine.compute_dew_point(4.0, 95)
        self.assertAlmostEqual(dp, 3.3, delta=0.2)
        self.assertIsInstance(engine.compute_dew_point(-2.0, 0), float)

    def test_parse_weather_tags_preserves_order_and_deduplicates(self) -> None:
        primary, tags = engine.parse_weather_tags([
            {"id": 500, "description": "light rain"},
            {"id": 701, "description": "mist"},
            {"id": 500, "description": "Light Rain"},
        ])
        self.assertEqual(primary, "light rain")
        self.assertEqual(tags, ("light rain", "mist"))

        primary_fb, tags_fb = engine.parse_weather_tags([], default="Unknown")
        self.assertEqual(primary_fb, "Unknown")
        self.assertEqual(tags_fb, ("unknown",))

    def test_estimate_uv_attenuates_for_fog_and_mist(self) -> None:
        # Mid-day summer UTC timestamp at 50N, 4E -> high solar elevation
        ts = 1718884800.0  # 2024-06-20 12:00:00 UTC
        clear_uv = engine.estimate_uv(50.0, 4.0, ts, cloud_pct=10)
        mist_uv = engine.estimate_uv(50.0, 4.0, ts, cloud_pct=10, has_mist=True)
        fog_uv = engine.estimate_uv(50.0, 4.0, ts, cloud_pct=10, has_fog=True)
        self.assertGreater(clear_uv, mist_uv)
        self.assertGreater(mist_uv, fog_uv)


class FetchWeatherPipelineTests(unittest.TestCase):
    def test_fetch_weather_parses_visibility_multitag_fog_lighting_and_snow_steps(self) -> None:
        now = int(time.time())
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
                return await engine.fetch_weather(client, "dummy-key", 50.0, 4.0, "BE")

        payload = asyncio.run(run())
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


class RenderPrecisionTests(unittest.TestCase):
    def _sample_payload(self, country: str = "BE", temp: float = -0.5, rain_amount: float = 0.6,
                        snow_amount: float = 0.0, desc: str = "freezing rain",
                        all_descs: tuple[str, ...] = ("freezing rain", "fog"),
                        visibility: int = 350) -> engine.WeatherPayload:
        eph = engine.EphemerisData("03h 15m", True, 15, "Fog / Low Visibility")
        steps = [
            engine.ForecastStep(
                time_str="11:00", temp=-0.5, pop=0.8, rain_mm=1.5, snow_mm=0.0,
                wind_speed=3.5, wind_deg=210, desc="Freezing Rain", day_offset=0,
                wind_gust=11.0, feels_like=-5.0, humidity=96, visibility=600,
                all_descs=("freezing rain",),
            ),
            engine.ForecastStep(
                time_str="14:00", temp=1.0, pop=0.3, rain_mm=0.4, snow_mm=0.8,
                wind_speed=4.0, wind_deg=220, desc="Rain And Snow", day_offset=0,
                wind_gust=6.0, feels_like=-2.5, humidity=92, visibility=800,
            ),
            engine.ForecastStep(
                time_str="17:00", temp=-6.0, pop=0.1, rain_mm=0.0, snow_mm=0.0,
                wind_speed=5.0, wind_deg=30, desc="Fog", day_offset=1,
                wind_gust=7.5, feels_like=-11.5, humidity=96, visibility=400,
            ),
        ]
        return engine.WeatherPayload(
            location_name=country, temp=temp, feels_like=temp - 3.5, humidity=96, pressure=1014,
            wind_speed=3.5, wind_deg=210, desc=desc, forecast_steps=steps, ephemeris=eph,
            aqi_data=None, rain_amount=rain_amount, rain_prob=80.0, uv_index=0.2,
            local_hour=10, snow_amount=snow_amount, uv_estimated=False, country=country,
            wind_gust=5.0, visibility=visibility, all_descs=all_descs,
        )

    def test_detail_page_surfaces_fog_dewpoint_freezing_snow_and_day_boundary(self) -> None:
        loc_be = build_site.Location("BE", "🇧🇪", 0.0, 0.0, "0,0")
        html_be = build_site.render_detail(loc_be, self._sample_payload("BE"), None, 1700000000)
        self.assertIn("<title>BE · BLACK ICE PROTOCOL</title>", html_be)
        self.assertIn("FREEZING PRECIP", html_be)
        self.assertIn("Freezing Rain + Fog", html_be)
        self.assertIn("vis 350 m", html_be)
        self.assertIn("Dew -1.1°C", html_be)
        self.assertIn("350 m (Dense fog)", html_be)
        self.assertIn("Fog / Low Visibility", html_be)
        self.assertIn("❄1.2mm", html_be)
        self.assertIn("rain 0.4 + snow 0.8mm", html_be)
        self.assertIn('<span class="ice">Ice risk</span>', html_be)
        self.assertIn('<span class="fog">Dense fog</span>', html_be)
        self.assertIn('<span class="frost">Bitter cold</span> · vis 400 m', html_be)
        self.assertIn('<h3 class="day-sep">Tomorrow</h3>', html_be)
        self.assertIn('<span class="ice">[!]</span>', html_be)
        self.assertIn('<span class="warn">[!]</span>', html_be)
        self.assertIn('<span class="fog">[!]</span>', html_be)

        loc_ee = build_site.Location("EE", "🇪🇪", 0.0, 0.0, "0,0")
        html_ee = build_site.render_detail(loc_ee, self._sample_payload("EE"), None, 1700000000)
        self.assertIn("<title>EE · MUSTA JÄÄ HÄIRE</title>", html_ee)
        self.assertIn("JÄÄTUV SADU", html_ee)
        self.assertIn("Jäätuv vihm + udu", html_ee)
        self.assertIn("Kastep. -1.1°C", html_ee)
        self.assertIn("350 m (Tihe udu)", html_ee)
        self.assertIn("Udu / Piiratud nähtavus", html_ee)
        self.assertIn("vihm 0.4 + lumi 0.8mm", html_ee)
        self.assertIn('<span class="ice">Libeduse oht</span>', html_ee)
        self.assertIn('<span class="fog">Tihe udu</span>', html_ee)
        self.assertIn('<span class="frost">Karm külm</span> · näht. 400 m', html_ee)
        self.assertIn('<h3 class="day-sep">Homme</h3>', html_ee)

    def test_trace_drizzle_subtitle_when_gauge_reads_zero(self) -> None:
        loc_be = build_site.Location("BE", "🇧🇪", 0.0, 0.0, "0,0")
        w = self._sample_payload("BE", temp=4.0, rain_amount=0.0, desc="light intensity drizzle",
                                 all_descs=("light intensity drizzle",), visibility=6000)
        html_be = build_site.render_detail(loc_be, w, None, 1700000000)
        self.assertIn("Trace drizzle", html_be)

    def test_overview_shows_feels_like_precip_visibility_and_freezing_hazard(self) -> None:
        loc_be = build_site.Location("BE", "🇧🇪", 0.0, 0.0, "0,0")
        loc_ee = build_site.Location("EE", "🇪🇪", 0.0, 0.0, "0,0")
        ov = build_site.render_overview([
            (loc_be, self._sample_payload("BE"), None),
            (loc_ee, self._sample_payload("EE", temp=2.0, rain_amount=0.0, snow_amount=1.4,
                                          desc="light snow", all_descs=("light snow",), visibility=2500), None),
        ], 1700000000)
        self.assertIn("(feels -4°)", ov)
        self.assertIn("0.6 mm", ov)
        self.assertIn("vis 350 m", ov)
        self.assertIn("Freezing precip", ov)
        self.assertIn("❄1.4 mm", ov)
        self.assertIn("näht. 2.5 km", ov)


class VibeMatrixColdRainFogTests(unittest.TestCase):
    def _mk(
        self,
        temp: float = 2.0,
        humidity: int = 95,
        wind_speed: float = 2.0,
        desc: str = "overcast clouds",
        rain_amount: float = 0.0,
        local_hour: int = 13,
        is_day: bool = True,
        country: str = "BE",
        rain_prob: float | None = None,
        feels_like: float | None = None,
        visibility: int | None = None,
        all_descs: tuple[str, ...] = (),
        snow_amount: float = 0.0,
    ) -> engine.WeatherPayload:
        if feels_like is None:
            feels_like = temp - wind_speed * 0.6
        if rain_prob is None:
            rain_prob = 85.0 if rain_amount > 0 else 20.0
        eph = engine.EphemerisData("02h 00m" if is_day else "Night Cycle", is_day, 90, "Gloomy")
        return engine.WeatherPayload(
            location_name=country,
            temp=temp,
            feels_like=feels_like,
            humidity=humidity,
            pressure=1012,
            wind_speed=wind_speed,
            wind_deg=220,
            desc=desc,
            forecast_steps=[],
            ephemeris=eph,
            aqi_data=None,
            rain_amount=rain_amount,
            rain_prob=rain_prob,
            uv_index=0.5,
            local_hour=local_hour,
            snow_amount=snow_amount,
            country=country,
            wind_gust=wind_speed * 1.4,
            visibility=visibility,
            all_descs=all_descs,
        )

    def test_region_vibe_dictionaries_cover_all_eligible_rules(self) -> None:
        be_eligible = {r.name for r in engine.VIBE_RULES if r.applies_to("BE")} | {engine.DEFAULT_VIBES["BE"].name}
        ee_eligible = {r.name for r in engine.VIBE_RULES if r.applies_to("EE")} | {engine.DEFAULT_VIBES["EE"].name}
        self.assertEqual(be_eligible - set(vibe_be._BY_ORIG_NAME), set())
        self.assertEqual(ee_eligible - set(vibe_ee._BY_ORIG_NAME), set())
        self.assertEqual(vibe_ee.translate_weather_desc("freezing fog"), "Jäätuv udu")
        self.assertEqual(vibe_ee.translate_weather_desc("freezing drizzle"), "Jäätuv uduvihm")

    def test_freezing_rain_and_freezing_fog_precedence(self) -> None:
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=-1.0, desc="freezing rain", rain_amount=1.2)).name, "BLACK ICE PROTOCOL")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=-2.0, desc="freezing rain", rain_amount=2.5)).name, "BLACK ICE PROTOCOL")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=-1.0, wind_speed=9.0, desc="light rain", rain_amount=1.0)).name, "BLACK ICE PROTOCOL")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=-1.0, local_hour=16, is_day=False, country="EE", desc="freezing rain", rain_amount=0.5)).name, "BLACK ICE PROTOCOL")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=-1.0, desc="light rain and snow", rain_amount=0.5, snow_amount=0.5)).name, "HOTH")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=-2.0, desc="fog", visibility=350)).name, "FROSTBOUND FOG")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=-2.0, desc="freezing fog", visibility=350)).name, "FROSTBOUND FOG")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=-4.0, desc="mist", visibility=2500)).name, "BALTIC FREEZE")

    def test_rain_fog_multitag_order_and_visibility_thresholds(self) -> None:
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=4.0, desc="fog", rain_amount=5.0, visibility=300)).name, "NOAH'S ARK")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=4.0, desc="fog", rain_amount=2.5, visibility=300)).name, "PROPER RAIN")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=2.0, desc="fog", visibility=400)).name, "SILENT HILL")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=2.0, desc="mist", visibility=2000, humidity=95)).name, "BONE CHILL")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=7.0, desc="mist", visibility=4500, humidity=80)).name, "THE BELGIAN STANDARD")
        # Order of OWM tags must not flip the verdict
        v_a = engine.evaluate_vibe(self._mk(temp=7.0, desc="mist", all_descs=("mist", "light rain"), rain_amount=0.6, rain_prob=60.0, visibility=2000, country="EE"))
        v_b = engine.evaluate_vibe(self._mk(temp=7.0, desc="light rain", all_descs=("light rain", "mist"), rain_amount=0.6, rain_prob=60.0, visibility=2000, country="EE"))
        self.assertEqual(v_a.name, "UDUVIHM")
        self.assertEqual(v_b.name, "UDUVIHM")

    def test_cold_drizzle_night_windchill_and_ee_darkness(self) -> None:
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=2.0, desc="light intensity drizzle", rain_amount=0.3)).name, "THE MARROW SOAK")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=2.0, local_hour=22, is_day=False, desc="light intensity drizzle", rain_amount=0.3)).name, "THE MARROW SOAK")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=1.5, local_hour=8, desc="light intensity drizzle", rain_amount=0.3)).name, "THE MARROW SOAK")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=4.0, local_hour=8, desc="light intensity drizzle", rain_amount=0.3)).name, "THE OLYMPIC SPORT")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=7.0, desc="light intensity drizzle", rain_amount=0.3, rain_prob=85.0)).name, "PERPETUAL GLOOM")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=-2.0, local_hour=17, is_day=False, wind_speed=6.5, desc="few clouds")).name, "THE WHITE WALKER")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=0.5, feels_like=-6.5, wind_speed=8.0, humidity=85)).name, "BALTIC FREEZE")
        # Estonia winter dark afternoon separation
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=2.0, humidity=80, local_hour=16, is_day=False, country="EE")).name, "SUITSUSAUN")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=1.0, rain_amount=0.8, desc="light rain", local_hour=16, is_day=False, country="EE")).name, "THE MARROW SOAK")
        self.assertEqual(engine.evaluate_vibe(self._mk(temp=7.0, humidity=80, local_hour=16, is_day=False, country="EE")).name, "PIMEDUS")

        # Verify BE/EE localization entries for all new rules render cleanly
        for orig in ("BLACK ICE PROTOCOL", "FROSTBOUND FOG", "THE MARROW SOAK"):
            self.assertTrue(vibe_be.get_be_vibe(orig)[0])
            self.assertTrue(vibe_ee.get_ee_vibe(orig)[0])


if __name__ == "__main__":
    unittest.main()
