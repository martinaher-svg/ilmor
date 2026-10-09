# Atmospheric Dispatch — web edition

A static, phone-friendly version of the terminal weather dashboard.

- **Updates:** a GitHub Actions job runs every 15 minutes, fetches the weather once per location, renders plain HTML and deploys it to GitHub Pages.
- **No API calls from visitors:** visitors only ever download those static files. No visitor action can call the weather APIs, so there is no cost exposure.
- **Anonymised:** locations are identified by country code only (BE / IT / EE). Coordinates live in encrypted repository secrets and never appear in the code, logs or pages. A leak guard aborts the build if they would.

## Setup

1. Create a **public** repository and push this folder to it. Free GitHub Pages and free Actions minutes both require a public repo.
2. **Settings → Pages → Build and deployment → Source: GitHub Actions**
3. **Settings → Secrets and variables → Actions → New repository secret**. Add these:

   | Secret | Value |
   |---|---|
   | `WEATHER_API_KEY` | OpenWeatherMap API key |
   | `LOC_BE` | `lat,lon` |
   | `LOC_IT` | `lat,lon` |
   | `LOC_EE` | `lat,lon` |
   | `SITE_TOKEN` | random string, e.g. output of `python -c "import secrets; print(secrets.token_urlsafe(18))"` |

4. **Actions → Publish dashboard → Run workflow** to trigger the first build.
5. Open `https://<user>.github.io/<repo>/<SITE_TOKEN>/` on your phone and add it to your home screen.

## Notes

- **Schedule drift:** GitHub's scheduler is best-effort, so runs can be a few minutes late. The page shows its age and turns the timestamp red after 45 minutes.
- **Auto-disable:** in public repos, GitHub disables scheduled workflows after 60 days without repository activity. It emails a warning first, and one click re-enables it.
- **Failures:** if one location fails, its page says so and the others still update. If all fail, nothing is deployed and the previous version stays online.
- **Cost:** 3 locations × 4 requests × 96 runs/day ≈ 1,150 requests/day, well inside the OpenWeatherMap and Open-Meteo free tiers.
