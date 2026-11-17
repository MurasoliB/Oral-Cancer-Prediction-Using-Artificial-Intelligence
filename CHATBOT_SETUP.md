# OralVision Chatbot

The chatbot works offline using a built-in educational knowledge base. No API key and no OpenAI Python SDK are required. The knowledge base answers common questions about result interpretation, confidence scores, Grad-CAM, model classes, warning signs, prevention, biopsy, and the project workflow. It uses a saved analysis result only when the server verifies ownership; it does not trust result data sent by the browser.

## Optional AI responses

To enable an OpenAI-compatible Chat Completions API, set these environment variables in PowerShell before starting the app:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_MODEL = "gpt-4o-mini"
# Optional for a compatible endpoint:
# $env:OPENAI_BASE_URL = "https://api.openai.com/v1"
```

Do not commit API keys to source control. API access may incur charges. If the API is unavailable, the chatbot automatically falls back to the offline knowledge base.

## Safety

This is an educational project, not a medical device or diagnostic service. Model predictions and confidence scores do not establish or exclude cancer. Persistent mouth ulcers, lumps, red/white patches, unexplained bleeding, numbness, or swallowing difficulty should be assessed by a qualified clinician. Breathing difficulty, severe bleeding, or inability to swallow fluids requires emergency care.
