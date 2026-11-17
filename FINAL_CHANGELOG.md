# OralVision AI — Final Engineering Release

## Fixed in this release

### Prediction system
- Correct 3-class labels: Normal / Pre-Cancer / Oral Cancer.
- Added an explicit `Inconclusive` outcome.
- Added abstention when top probability is below 60% or top-two class separation is below 10 percentage points.
- Stored the complete probability vector for each patient analysis.
- Confidence is displayed as model confidence, never as disease severity.
- Grad-CAM is generated only for a classified result and targets the predicted class.
- Training scripts and model metadata use the same class ordering and 1/255 preprocessing convention.

### Security
- CSRF protection on form and JSON POST requests.
- Login/signup/prediction/chatbot rate limits.
- Security response headers.
- Password hashing.
- UUID server-side filenames.
- Real image validation and dimension limits.
- Uploaded images moved outside `static/`.
- Authenticated ownership checks before serving uploaded images.
- Production secret key requirement.
- Non-root Docker runtime.

### Testing
- Added pytest smoke tests.
- Added security-header test.
- Added chatbot validation test.
- Removed legacy scripts from pytest discovery and moved them under `scripts/`.

### CI/CD
- GitHub Actions now runs:
  - Ruff linting
  - Python compilation
  - pytest
  - Docker image build
- Python 3.11 is used consistently in CI and Docker.

### Code quality
- Removed the previous large chatbot implementation from `app.py`.
- Chatbot logic is isolated in `services/chatbot_service.py`.
- Centralized upload validation/saving.
- Centralized confidence-band presentation.
- Added lightweight database schema migration/compatibility handling.
- Added `.env.example`.
- Updated documentation and removed obsolete Benign/Malignant terminology.

### Chatbot
- No fake/mock AI responses.
- Deterministic educational FAQ works without an API key.
- Optional OpenAI integration is environment-variable based.
- Prompt explicitly prevents diagnosis, treatment prescriptions, and disease-severity claims.
- Patient analysis context is only attached after ownership verification.
