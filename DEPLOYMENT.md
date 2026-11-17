# OralVision AI — GitHub + Render Deployment

## 1. Push to GitHub

Open a terminal in this folder and run:

```bash
git init
git branch -M main
git add .
git status
git commit -m "Initial release of OralVision AI"
git remote add origin https://github.com/YOUR_USERNAME/oralvision-ai.git
git push -u origin main
```

The trained model is included at `model/densenet201_model.keras` and is below GitHub's 100 MiB single-file limit. Do not upload the project through the GitHub browser uploader; use Git from a terminal.

Before pushing, verify that `.env`, `instance/`, databases, uploaded images, and datasets are not staged:

```bash
git status --short
```

## 2. Deploy on Render

Create a Render Web Service from the GitHub repository.

- Runtime: Docker
- Dockerfile: `Dockerfile`
- Health check: `/health`
- Branch: `main`

The included `render.yaml` can also be used with Render Blueprint deployment.

## 3. Required environment variables

Set these in Render's Environment settings:

```text
FLASK_ENV=production
FLASK_SECRET_KEY=<long-random-secret>
```

`FLASK_SECRET_KEY` must never be committed to GitHub.

The chatbot works with its built-in educational FAQ without an API key. `OPENAI_API_KEY` is optional.

## 4. After deployment

Check:

```text
https://YOUR-SERVICE.onrender.com/health
```

Then test the complete flow:

1. Register.
2. Log in.
3. Create a patient.
4. Upload a valid image.
5. Run prediction.
6. Confirm Grad-CAM appears when generated successfully.
7. Open the chatbot.
8. Generate a PDF report.

## 5. Important persistence limitation

The default deployment uses SQLite and the local `instance/uploads` directory. These are suitable for a college/demo deployment but should not be represented as durable production medical storage on an ephemeral host.

For a real production system, migrate to PostgreSQL and object storage for uploaded images.

## 6. Local Docker test

```bash
docker build -t oralvision-ai .
docker run --rm -p 10000:10000 -e FLASK_SECRET_KEY="dev-secret-change-me" oralvision-ai
```

Open `http://127.0.0.1:10000/health`.

## 7. Model checksum

The expected SHA-256 checksum is stored in `model/SHA256.txt`.

Windows PowerShell:

```powershell
Get-FileHash model/densenet201_model.keras -Algorithm SHA256
```

Linux/macOS:

```bash
sha256sum model/densenet201_model.keras
```
