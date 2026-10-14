# Deploying the demo (free, no credit card)

The server runs the **TFLite** model (8.9 MB, ~75 MB of RAM, ~11 ms per image), so it fits free hosting tiers.
For comparison, the same model under full TensorFlow/Keras used ~630 MB of RAM.

## Render (free)

1. Sign in at render.com with GitHub.
2. **New +** → **Blueprint** → pick `saabi-cattle-breed-classifier`. Render reads `render.yaml`
   (Docker, free plan, health check at `/api/health`) and builds the image.
3. After a few minutes you get a URL like `https://saabi-xxxx.onrender.com`.

Free services sleep after ~15 minutes without traffic and take ~30-60 s to wake on the next visit.
Open the URL once before showing it to anyone.

## Run it yourself

```bash
docker build -t saabi .
docker run -p 10000:10000 saabi        # http://localhost:10000
```

> Hugging Face now requires a paid plan for Docker Spaces, which is why this project deploys on Render instead.
