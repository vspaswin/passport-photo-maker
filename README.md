# Passport Photo Maker

**Production-ready** service to validate and convert photos into **Indian passport / visa / OCI** format (print + digital upload).

- Strict automated QC (as-is + convertible + final)
- White background cutout (`u2net_human_seg`)
- Face detection: MediaPipe when available, OpenCV Haar fallback
- Print sheets: **Letter 8.5×11** (Canon GP-701), A4, 4×6, single 2×2
- **Job-based downloads** (multi-user safe, TTL)
- **Freemium credits** + optional **Stripe Checkout**
- Web UI + **CLI** + **Docker**

> Automated QC is **not** official government approval. Final acceptance is decided by VFS / passport authorities.

## Quick start (local)

```bash
cd passport-photo-maker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
# open http://127.0.0.1:8765
```

Optional MediaPipe (better faces):

```bash
pip install mediapipe
```

## CLI

```bash
# Check only
python cli.py check photo.jpg

# Convert one
python cli.py convert photo.jpg -o ./out

# Batch folder
python cli.py batch ./photos -o ./batch_out
```

## Monetization (freemium + Stripe)

Defaults (env-overridable):

| | Free / day | After free |
|--|------------|------------|
| Checks (per cookie) | 20 | Unlimited with credits |
| Converts (per cookie) | 3 | 1 credit each |
| IP caps (cookie rotation) | 40 checks / 6 converts | Buy credits |

Hardening (v1.0.1+):

- **Atomic** `reserve_convert` before work; **refund** if processing fails
- **Idempotent** Stripe fulfill (no double-credit)
- **Job ownership** — downloads require matching client cookie
- Production **fails to start** if `SECRET_KEY` is still the default

1. Create products/prices in [Stripe Dashboard](https://dashboard.stripe.com/)
2. Copy `.env.example` → `.env` and set:

```env
APP_ENV=production
APP_URL=https://your-domain.com
SECRET_KEY=long-random-string
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_STARTER=price_...
STRIPE_PRICE_PRO=price_...
```

3. Webhook endpoint: `POST /api/billing/webhook` → event `checkout.session.completed`

Without Stripe keys the app still runs; Buy buttons stay disabled and free daily quotas apply.

## Docker

```bash
cp .env.example .env   # edit secrets
docker compose up --build -d
```

Data (jobs + credits DB) persists in volume `ppm-data`.

## API overview

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Liveness |
| GET | `/api/status` | Model + usage + pricing |
| POST | `/api/validate` | Check as-is + convertible |
| POST | `/api/convert` | Convert (costs free slot or credit) |
| POST | `/api/batch` | Multi-file convert |
| GET | `/api/jobs/{id}` | Job metadata |
| GET | `/api/jobs/{id}/files/{name}` | Download |
| POST | `/api/billing/checkout` | Stripe session |
| POST | `/api/billing/webhook` | Stripe webhook |

## Print on Canon GP-701 (Letter glossy)

Use **`*_sheet_letter.jpg`**:

- Paper: **Letter** + **Photo Glossy** + **High/Best**
- Scale: **100% / Actual size**
- Load glossy side correctly; no Draft; dry before stacking

## Project layout

```
app/
  core/config.py      # env settings
  jobs/store.py       # SQLite jobs + credits
  billing/            # Stripe
  engine/             # face, validate, process, specs
  main.py             # FastAPI
cli.py
Dockerfile
docker-compose.yml
```

## License

MIT — see [LICENSE](LICENSE).
