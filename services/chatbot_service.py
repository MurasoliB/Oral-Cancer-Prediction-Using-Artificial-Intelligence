"""OralVision educational assistant with result-aware offline answers.

The assistant is educational only; it must not diagnose or prescribe. An optional
OpenAI-compatible endpoint can be enabled with OPENAI_API_KEY. The built-in
knowledge base works without internet or API credentials.
"""
import logging
import os
import re
import requests

logger = logging.getLogger(__name__)

SYSTEM = """You are OralVision's clear, careful educational oral-health assistant.
Explain general oral health and this project's model in plain language. Never diagnose,
claim a prediction confirms or rules out cancer, estimate prognosis/survival, prescribe
medication, or invent patient details. A model prediction is a screening/research aid,
not a medical diagnosis. For a persistent mouth ulcer, lump, red/white patch, unexplained
bleeding, numbness, difficulty swallowing, or a symptom lasting more than two weeks,
recommend prompt assessment by a dentist/oral medicine specialist/ENT. For breathing
trouble, severe bleeding, or inability to swallow fluids, recommend emergency care.
Be empathetic but concise, answer the actual question, and state uncertainty clearly."""

TOPICS = {
    "results": "The result is the model's image classification, not a diagnosis. The three output classes are Normal, Pre-Cancer, and Oral Cancer. Image quality, the type of image used for training, and differences between the uploaded image and training data can affect the result. A dentist or oral medicine/ENT specialist must assess concerning lesions; a biopsy may be needed to confirm a diagnosis.",
    "next_steps": "Do not use this result alone to decide whether you have cancer. If you have a persistent ulcer, lump, red/white patch, unexplained bleeding, numbness, or trouble swallowing, arrange an appointment with a dentist or oral medicine/ENT specialist promptly—especially if it lasts longer than two weeks. They may examine the area and decide whether further tests or a biopsy are appropriate. Seek emergency care for breathing difficulty, severe bleeding, or inability to swallow fluids.",
    "confidence": "The confidence score is the model's score for its predicted class. It is not the probability that you have cancer, does not measure disease severity, and is not proof that the prediction is correct. A low score or a high score cannot replace clinical assessment.",
    "gradcam": "Grad-CAM creates a heatmap showing image regions that influenced a model prediction. It is an explanation aid—not a lesion boundary, proof of cancer, or evidence that the model focused on medically meaningful tissue. Interpret it alongside the original image and professional assessment.",
    "classes": "This project has three output classes: Normal, Pre-Cancer, and Oral Cancer. These are the model's learned image categories, not definitive clinical diagnoses. Real diagnosis depends on examination and, when indicated, histopathology/biopsy.",
    "diagnosis": "Oral cancer is assessed by a qualified clinician through examination of the mouth and neck. If an area looks suspicious, a biopsy with histopathological examination is commonly used to establish the diagnosis. Imaging or other tests may be used depending on the clinical situation. This app cannot diagnose or rule out cancer.",
    "symptoms": "Possible warning signs include a mouth ulcer that does not heal, a persistent red or white patch, a lump or thickened area, unexplained bleeding, persistent pain, numbness, difficulty chewing or swallowing, or a neck lump. These signs can have causes other than cancer, so they need professional assessment rather than self-diagnosis.",
    "prevention": "Risk reduction includes avoiding tobacco in every form and areca/betel nut products, limiting or avoiding alcohol, maintaining oral hygiene, attending dental check-ups, and seeking assessment for persistent mouth changes. No prevention step eliminates all risk.",
    "tobacco": "Tobacco products—including smokeless tobacco—and areca/betel nut products increase oral disease and cancer risk. Stopping is beneficial; a clinician or local tobacco-cessation service can help with a practical quit plan. Avoid replacing one tobacco product with another.",
    "diet": "A balanced diet with vegetables, fruits, whole grains, and adequate protein supports general health, but no food or supplement can prevent or treat oral cancer by itself. Do not delay a clinical assessment or substitute supplements for treatment.",
    "treatment": "Treatment depends on the confirmed diagnosis, site, stage, overall health, and specialist assessment. Options may include surgery, radiotherapy, systemic medicines, or combinations. Only the treating specialist can recommend an appropriate plan after diagnostic work-up; this chatbot cannot choose treatment.",
    "biopsy": "A biopsy means taking a small tissue sample for examination by a pathologist. It may be recommended when a clinician needs to determine what a suspicious area is. The clinician can explain the procedure, risks, and alternatives for your situation.",
    "worried": "It is understandable to feel worried. An AI result cannot tell you for certain whether you have cancer. The most useful next step is to arrange a professional examination if you have a persistent or concerning mouth change, and bring the result/image with you. If you tell me what part of the report is confusing, I can explain its wording in plain language.",
    "about": "OralVision is an academic image-classification project using a trained DenseNet201 model. It resizes an image to the model's input size (224×224 RGB), produces scores for three classes, and may display Grad-CAM to show regions influencing the output. Performance depends on the training data and validation; it is not a substitute for a clinician or biopsy.",
    "privacy": "Avoid entering names, phone numbers, addresses, or other identifying details into the chatbot. Share only the minimum information needed for a general explanation. Follow the project's privacy policy for how uploaded images and records are stored.",
}


def _normalise(message):
    return re.sub(r"[^a-z0-9% ]", " ", str(message).lower()).strip()


def _has(text, *phrases):
    # Match whole words/phrases so e.g. "obscure" does not accidentally match "cure".
    return any(re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text) for phrase in phrases)


def faq(message, info=None):
    """Return a relevant, safe offline response, using only server-held result context."""
    text = _normalise(message)
    if not text:
        return "Ask me a question about the report, confidence score, Grad-CAM, oral-health warning signs, prevention, or how the model works."

    # Emergency symptom triage must take priority over ordinary FAQ matching.
    if _has(text, "cant breathe", "cannot breathe", "trouble breathing", "difficulty breathing", "severe bleeding", "cannot swallow fluids", "unable to swallow fluids"):
        return "Those symptoms may need urgent medical care. Please contact your local emergency service or go to an emergency department now. This chatbot cannot assess emergencies."

    if _has(text, "what does my result", "what do these results", "explain my result", "explain the result", "my prediction", "what does it mean", "what do the results"):
        if info and info.get("prediction"):
            prediction = str(info.get("prediction"))
            confidence = info.get("confidence")
            conf_text = f" The displayed model score is {confidence}%" if isinstance(confidence, (int, float)) else ""
            return (f"Your saved model output is **{prediction}**.{conf_text}. This is an image-classification result, not a diagnosis or a reliable measure of cancer probability. Image quality and training-data limitations can affect it. If you have a persistent ulcer, lump, red/white patch, unexplained bleeding, numbness, or swallowing difficulty, arrange prompt assessment with a dentist or oral medicine/ENT specialist. A clinician may recommend a biopsy if needed.")
        return TOPICS["results"]
    if _has(text, "what should i do", "next step", "next steps", "what now", "should i see", "who should i contact", "do i need a doctor"):
        return TOPICS["next_steps"]
    if _has(text, "confidence", "score mean", "percentage", "probability", "how accurate"):
        return TOPICS["confidence"]
    if _has(text, "grad cam", "gradcam", "heat map", "heatmap", "red area", "highlight"):
        return TOPICS["gradcam"]
    if _has(text, "class", "classes", "normal pre cancer", "pre cancer", "oral cancer class"):
        return TOPICS["classes"]
    if _has(text, "biopsy", "histopath", "how is cancer diagnosed", "diagnosis", "diagnosed"):
        return TOPICS["biopsy"] if "biopsy" in text else TOPICS["diagnosis"]
    if _has(text, "symptom", "warning sign", "mouth ulcer", "white patch", "red patch", "lump", "bleeding", "not heal", "swallow"):
        return TOPICS["symptoms"]
    if _has(text, "prevent", "avoid cancer", "reduce risk", "lower risk"):
        return TOPICS["prevention"]
    if _has(text, "tobacco", "smoking", "smokeless", "betel", "areca", "gutka", "paan"):
        return TOPICS["tobacco"]
    if _has(text, "diet", "food", "nutrition", "eat", "supplement", "vitamin"):
        return TOPICS["diet"]
    if _has(text, "treatment", "cure", "medicine", "medication", "chemotherapy", "radiation", "surgery"):
        return TOPICS["treatment"]
    if _has(text, "worried", "scared", "afraid", "anxious", "panic", "fear"):
        return TOPICS["worried"]
    if _has(text, "how does this work", "how does the model", "what model", "densenet", "224", "machine learning", "project", "how it works"):
        return TOPICS["about"]
    if _has(text, "privacy", "stored", "save my image", "my data", "personal information"):
        return TOPICS["privacy"]
    if _has(text, "hello", "hi", "hey", "good morning", "good evening") and len(text.split()) <= 4:
        return "Hi! I'm the OralVision educational assistant. I can explain your model result, confidence score, Grad-CAM heatmap, warning signs, prevention, or how the project works. What would you like to understand?"
    if _has(text, "thank", "thanks"):
        return "You're welcome. If you have a specific line from the result report, paste just that wording and I can explain it. Remember that the model cannot diagnose or rule out oral cancer."
    return "I can help with your result, confidence score, Grad-CAM heatmap, model classes, oral-health warning signs, prevention, biopsy/diagnosis, or how the project works. Try asking: ‘What does my result mean?’, ‘Is confidence the same as cancer probability?’, or ‘What does Grad-CAM show?’ I can't diagnose or recommend a personal treatment plan."


def generate_chat_response(message, info=None):
    """Use optional chat-completions API; use the offline knowledge base by default."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return faq(message, info)

    safe_context = "No saved analysis result is available."
    if info and info.get("prediction"):
        safe_context = "Saved model output (not a diagnosis): prediction={}; confidence={}; status={}.".format(
            info.get("prediction"), info.get("confidence"), info.get("status")
        )
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model_name, "temperature": 0.2, "max_tokens": 400,
                  "messages": [{"role": "system", "content": SYSTEM},
                               {"role": "system", "content": safe_context},
                               {"role": "user", "content": str(message)[:1000]}]},
            timeout=(4, 18),
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        if isinstance(answer, str) and answer.strip():
            return answer.strip()[:5000]
        logger.warning("Chat API returned an empty response; using offline answer.")
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("Chat API unavailable; using offline answer: %s", exc)
    return faq(message, info)
