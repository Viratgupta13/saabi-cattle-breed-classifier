# Deploying the demo

The repo ships a `Dockerfile` (gunicorn, non-root, health check) that listens on `$PORT` (default `7860`).

```bash
docker build -t saabi .
docker run -p 7860:7860 saabi        # http://localhost:7860
```

## Option A: Hugging Face Spaces (recommended, free)

Spaces gives free CPU instances with 16 GB RAM, which is plenty for TensorFlow.

1. Create a new Space at huggingface.co/new-space → SDK: **Docker**.
2. Add this front matter to the **top of the Space's own `README.md`** (not needed in the GitHub README):
   ```yaml
   ---
   title: SAABI Cattle Breed Identifier
   emoji: 🐄
   sdk: docker
   app_port: 7860
   ---
   ```
3. (Shortcut) run `scripts/push_to_space.sh <user>/<space>`, which adds the front matter for you. Or push manually:
   push the repo contents to the Space (`git remote add space https://huggingface.co/spaces/<user>/<space>` then `git push space main`).
   `models/` (≈22 MB) must be included; `data/` is excluded by `.gitignore`.
4. Optional: set a Space variable `GITHUB_URL` to show a GitHub link in the page nav.

## Option B: Fly.io / Render

TensorFlow inference needs roughly 0.7–1 GB of RAM. Render's free 512 MB tier will likely run out of memory;
use a paid instance, or Fly.io with a 1 GB machine (`fly launch`, internal port 7860).

## After deploying

Add the live URL to the top of `README.md`, your resume and your portfolio.
