import os
import re
import numpy as np
from datetime import datetime

# Apply TensorFlow optimizations before importing TensorFlow
try:
    import optimize_startup  # Apply optimizations
except ImportError:
    # Fallback optimizations if file doesn't exist
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, session, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image
import cv2
import uuid
import json
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
import base64
import secrets

# Import TensorFlow and Keras for real model loading
import tensorflow as tf
from tensorflow import keras

from services.chatbot_service import generate_chat_response

# Global variable to store the model
model = None

# Initialize Flask app
app = Flask(__name__)
# Reads from the FLASK_SECRET_KEY environment variable in production.
# Falls back to a fixed dev-only value so the app still runs out of the box
# locally - but ALWAYS set FLASK_SECRET_KEY yourself before deploying,
# otherwise sessions/logins can be forged.
SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
if not SECRET_KEY and os.getenv('FLASK_ENV', '').lower() == 'production':
    raise RuntimeError('FLASK_SECRET_KEY must be set in production')
app.secret_key = SECRET_KEY or secrets.token_hex(32)

# Database Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
os.makedirs(INSTANCE_DIR, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///' + os.path.join(INSTANCE_DIR, 'oral_cancer_detection.db'))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configuration
UPLOAD_FOLDER = os.path.join(INSTANCE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['WTF_CSRF_TIME_LIMIT'] = 3600
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
if os.getenv('FLASK_ENV') == 'production':
    app.config['SESSION_COOKIE_SECURE'] = True

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
csrf = CSRFProtect(app)
limiter = Limiter(key_func=get_remote_address, app=app, default_limits=['300 per hour'], storage_uri='memory://')

# Add custom Jinja2 filters
@app.template_filter('from_json')
def from_json_filter(value):
    """Convert JSON string to Python object"""
    if not value:
        return {}
    try:
        import json
        if isinstance(value, str):
            return json.loads(value)
        return value
    except (json.JSONDecodeError, TypeError):
        return {}

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Verify upload directory exists and is writable
if not os.path.exists(UPLOAD_FOLDER):
    print(f"❌ Upload directory does not exist: {UPLOAD_FOLDER}")
else:
    print(f"✅ Upload directory exists: {UPLOAD_FOLDER}")
    
# Test write permissions
try:
    test_file = os.path.join(UPLOAD_FOLDER, 'test_write.tmp')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    print(f"✅ Upload directory is writable")
except Exception as e:
    print(f"❌ Upload directory is not writable: {e}")

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with patients
    patients = db.relationship('Patient', backref='user', lazy=True)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    sugar_level = db.Column(db.Float, nullable=True)
    blood_pressure = db.Column(db.String(20), nullable=True)
    weight = db.Column(db.Float, nullable=True)
    height = db.Column(db.Float, nullable=True)
    doctor_name = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    email = db.Column(db.String(120), nullable=True)
    doctor_phone = db.Column(db.String(20), nullable=True)
    address = db.Column(db.Text, nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign key to user
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationship with analyses
    analyses = db.relationship('Analysis', backref='patient', lazy=True)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(200), nullable=False)
    gradcam_filename = db.Column(db.String(200), nullable=True)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    severity = db.Column(db.String(20), nullable=False)  # legacy column; stores confidence band only
    status = db.Column(db.String(20), nullable=False, default='classified')
    probability_json = db.Column(db.Text, nullable=True)
    medical_suggestion = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign key to patient
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

CLASS_NAMES = ["Normal", "Oral Cancer", "Pre-Cancer"]
MODEL_PATH = os.path.join(BASE_DIR, "model", "densenet201_model.keras")
MODEL_LOAD_ERROR = None

def load_model():
    """Load the real trained DenseNet201 model. No synthetic fallback is used."""
    global model, MODEL_LOAD_ERROR
    if model is not None:
        return model
    try:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        model = keras.models.load_model(MODEL_PATH, compile=False)
        MODEL_LOAD_ERROR = None
        print("DenseNet201 model loaded successfully")
        return model
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)
        model = None
        print(f"Model loading failed: {exc}")
        return None

def get_model():
    return load_model() if model is None else model

def allowed_file(filename):
    """Check whether the extension is an accepted image format."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Use the same preprocessing as the bundled training pipeline: RGB, 224x224, float32 scaling to [0,1]."""
    try:
        with Image.open(image_path) as source:
            img = source.convert('RGB')
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)
    except Exception as exc:
        print(f"Image preprocessing error: {exc}")
        return None

def validate_image_file(path):
    """Validate real image content, dimensions, and decompression safety."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if img.width < 64 or img.height < 64:
                return False, "Image is too small. Please upload an image at least 64x64 pixels."
            if img.width > 8000 or img.height > 8000:
                return False, "Image dimensions are too large. Please upload a smaller image."
        return True, None
    except Exception:
        return False, "The uploaded file is not a valid image."

def generate_gradcam(filepath, predicted_index, output_filename):
    """Generate a Grad-CAM heatmap for the predicted class.

    This is an explainability visualization, not a segmentation or proof of
    where disease is located. Failures here never block the classification.
    """
    try:
        current_model = get_model()
        if current_model is None or len(current_model.inputs) != 1:
            return None

        # Prefer the last 4-D feature tensor exposed by the top-level model.
        target_layer = None
        for layer in reversed(current_model.layers):
            try:
                shape = layer.output_shape
            except Exception:
                continue
            if isinstance(shape, tuple) and len(shape) == 4:
                target_layer = layer
                break
        if target_layer is None:
            return None

        image_tensor = preprocess_image(filepath)
        if image_tensor is None:
            return None

        grad_model = keras.Model(current_model.inputs, [target_layer.output, current_model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_tensor, training=False)
            class_score = predictions[:, predicted_index]
        grads = tape.gradient(class_score, conv_outputs)
        if grads is None:
            return None

        conv_outputs = conv_outputs[0]
        grads = grads[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        heatmap = tf.reduce_sum(conv_outputs * weights, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)
        heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)
        heatmap = (heatmap.numpy() * 255).astype(np.uint8)
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        with Image.open(filepath) as source:
            original = np.asarray(source.convert('RGB').resize((224, 224), Image.Resampling.LANCZOS))
        original_bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_bgr, 0.55, heatmap_color, 0.45, 0)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, overlay)
        return output_filename if os.path.isfile(output_path) else None
    except Exception as exc:
        print(f"Grad-CAM generation skipped: {exc}")
        return None

def predict_image(filepath):
    """Run DenseNet201 inference with an abstention rule for uncertain outputs."""
    current_model = get_model()
    if current_model is None:
        raise RuntimeError("AI model is unavailable. Check the model file and TensorFlow installation.")
    processed = preprocess_image(filepath)
    if processed is None:
        raise ValueError("Could not preprocess the image.")
    probs = np.asarray(current_model.predict(processed, verbose=0)[0], dtype=np.float64)
    if probs.ndim != 1 or len(probs) != len(CLASS_NAMES) or not np.isfinite(probs).all():
        raise RuntimeError("Model returned an invalid prediction vector.")
    probs = np.clip(probs, 0, None)
    total = probs.sum()
    if total <= 0:
        raise RuntimeError("Model returned zero probabilities.")
    probs /= total
    order = np.argsort(probs)[::-1]
    idx = int(order[0])
    confidence = float(probs[idx])
    margin = float(probs[idx] - probs[order[1]])
    classified = confidence >= 0.60 and margin >= 0.10
    prediction = CLASS_NAMES[idx] if classified else 'Inconclusive'
    confidence_band = 'High' if confidence >= 0.80 else 'Moderate' if confidence >= 0.60 else 'Low'
    return {
        'class_index': idx,
        'prediction': prediction,
        'confidence': round(confidence * 100, 2),
        'confidence_band': confidence_band,
        'status': 'classified' if classified else 'needs_review',
        'margin': round(margin * 100, 2),
        'probabilities': {name: round(float(probs[i]) * 100, 2) for i, name in enumerate(CLASS_NAMES)}
    }

def get_medical_suggestion(confidence_band, prediction, status='classified'):
    if status != 'classified':
        return {
            'title': 'Inconclusive AI Result',
            'message': 'The model did not produce a sufficiently separated prediction for this image.',
            'advice': 'Review the image manually and do not treat this result as a diagnosis.'
        }
    return {
        'title': 'AI Classification Result',
        'message': f'The model classified the uploaded image as {prediction}.',
        'advice': 'This is an academic AI classification result, not a medical diagnosis. Professional assessment is required for real medical decisions.'
    }

def confidence_css_class(band):
    return {'High': 'success', 'Moderate': 'warning', 'Low': 'danger'}.get(band, 'warning')

def secure_upload_filename(original_filename):
    ext = original_filename.rsplit('.', 1)[1].lower()
    return f"{uuid.uuid4().hex}.{ext}"

def save_upload(file):
    if file is None or not file.filename:
        raise ValueError('No file selected.')
    if not allowed_file(file.filename):
        raise ValueError('Invalid file type. Use PNG, JPG, JPEG, GIF, or BMP.')
    filename = secure_upload_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    ok, reason = validate_image_file(path)
    if not ok:
        try: os.remove(path)
        except OSError: pass
        raise ValueError(reason)
    return filename, path

def generate_pdf_report(patient, analysis):
    """Generate PDF report for patient analysis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    # Title
    story.append(Paragraph("AI Oral Lesion Classification Report", title_style))
    story.append(Spacer(1, 20))
    
    # Patient Information
    story.append(Paragraph("Patient Information", heading_style))
    patient_data = [
        ['Name:', patient.name],
        ['Age:', f"{patient.age} years"],
        ['Gender:', patient.gender],
        ['Phone:', patient.phone or 'Not provided'],
        ['Doctor:', patient.doctor_name or 'Not assigned'],
        ['Date of Analysis:', analysis.created_at.strftime('%B %d, %Y at %I:%M %p')]
    ]
    
    if patient.sugar_level:
        patient_data.append(['Sugar Level:', f"{patient.sugar_level} mg/dL"])
    if patient.blood_pressure:
        patient_data.append(['Blood Pressure:', patient.blood_pressure])
    if patient.weight:
        patient_data.append(['Weight:', f"{patient.weight} kg"])
    if patient.height:
        patient_data.append(['Height:', f"{patient.height} cm"])
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Analysis Results
    story.append(Paragraph("Analysis Results", heading_style))
    
    # Determine result styling from the three model classes
    if analysis.prediction == "Oral Cancer":
        result_color = colors.red
    else:
        result_color = colors.green
    
    result_data = [
        ['Prediction:', analysis.prediction],
        ['Confidence:', f"{analysis.confidence}%"],
        ['Decision Status:', 'Classified' if analysis.status == 'classified' else 'Inconclusive / Needs Review'],
        ['Analysis Date:', analysis.created_at.strftime('%B %d, %Y at %I:%M %p')]
    ]
    
    result_table = Table(result_data, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, 0), result_color),
        ('BACKGROUND', (1, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(result_table)
    story.append(Spacer(1, 20))
    
    # Project Guidance
    story.append(Paragraph("Project Guidance", heading_style))
    
    if analysis.medical_suggestion:
        import json
        try:
            suggestion = json.loads(analysis.medical_suggestion)
            story.append(Paragraph(f"<b>{suggestion.get('title', 'Medical Advice')}</b>", styles['Normal']))
            story.append(Spacer(1, 10))
            story.append(Paragraph(suggestion.get('message', ''), styles['Normal']))
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"<b>Recommended Action:</b> {suggestion.get('advice', '')}", styles['Normal']))
        except:
            story.append(Paragraph(analysis.medical_suggestion, styles['Normal']))
    
    story.append(Spacer(1, 30))
    
    # Disclaimer
    story.append(Paragraph("Important Disclaimer", heading_style))
    disclaimer_text = """
    This AI-powered analysis is designed for screening and educational purposes only. 
    It should not replace professional medical diagnosis, treatment, or advice from qualified healthcare professionals. 
    Always consult with your doctor or dentist for proper medical evaluation and treatment decisions. 
    Early detection and professional medical care are crucial for the best outcomes.
    """
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 20))
    footer_text = f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | AI Oral Lesion Classification System"
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_quick_pdf_report(result_data):
    """Generate PDF report for quick upload analysis (without patient data)"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    # Title
    story.append(Paragraph("AI Oral Lesion Classification Report", title_style))
    story.append(Spacer(1, 20))
    
    # Analysis Information
    story.append(Paragraph("Analysis Information", heading_style))
    
    # Get user info safely
    try:
        user_name = current_user.full_name if current_user and current_user.is_authenticated else 'Anonymous'
    except:
        user_name = 'Anonymous'
    
    analysis_data = [
        ['Analysis Type:', 'Quick Upload Analysis'],
        ['Image File:', result_data.get('image_path', 'Unknown')],
        ['Analysis Date:', result_data.get('timestamp', 'Unknown')],
        ['User:', user_name]
    ]
    
    analysis_table = Table(analysis_data, colWidths=[2*inch, 4*inch])
    analysis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(analysis_table)
    story.append(Spacer(1, 20))
    
    # Analysis Results
    story.append(Paragraph("Analysis Results", heading_style))
    
    # Determine result styling from the three model classes
    prediction = result_data.get('prediction', '')
    if prediction == "Oral Cancer":
        result_color = colors.red
    else:
        result_color = colors.green
    
    result_data_table = [
        ['Prediction:', prediction],
        ['Confidence:', f"{result_data.get('confidence', 0)}%"],
        ['Analysis Method:', 'AI Deep Learning (DenseNet201)']
    ]
    
    result_table = Table(result_data_table, colWidths=[2*inch, 4*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, 0), result_color),
        ('BACKGROUND', (1, 1), (1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(result_table)
    story.append(Spacer(1, 20))
    
    # Project Guidance
    story.append(Paragraph("Project Guidance", heading_style))
    
    medical_suggestion = result_data.get('medical_suggestion', {})
    if medical_suggestion:
        if isinstance(medical_suggestion, str):
            try:
                import json
                medical_suggestion = json.loads(medical_suggestion)
            except:
                pass
        
        if isinstance(medical_suggestion, dict):
            story.append(Paragraph(f"<b>{medical_suggestion.get('title', 'Medical Advice')}</b>", styles['Normal']))
            story.append(Spacer(1, 10))
            story.append(Paragraph(medical_suggestion.get('message', ''), styles['Normal']))
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"<b>Recommended Action:</b> {medical_suggestion.get('advice', '')}", styles['Normal']))
        else:
            story.append(Paragraph(str(medical_suggestion), styles['Normal']))
    
    story.append(Spacer(1, 30))
    
    # Important Notes
    story.append(Paragraph("Important Notes", heading_style))
    notes_text = """
    • This analysis was performed using a quick upload without patient information
    • For comprehensive medical records, please use the full patient analysis system
    • Results should be reviewed by qualified healthcare professionals
    • This screening tool is for educational and preliminary assessment purposes only
    """
    story.append(Paragraph(notes_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Disclaimer
    story.append(Paragraph("Important Disclaimer", heading_style))
    disclaimer_text = """
    This AI-powered analysis is designed for screening and educational purposes only. 
    It should not replace professional medical diagnosis, treatment, or advice from qualified healthcare professionals. 
    Always consult with your doctor or dentist for proper medical evaluation and treatment decisions. 
    Early detection and professional medical care are crucial for the best outcomes.
    """
    story.append(Paragraph(disclaimer_text, styles['Normal']))
    
    # Footer
    story.append(Spacer(1, 20))
    footer_text = f"Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | AI Oral Lesion Classification System"
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

# Routes
@app.after_request
def add_security_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
    response.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    response.headers.setdefault('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
    if request.is_secure:
        response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return response

@app.route('/health')
def health():
    """Deployment health endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": os.path.basename(MODEL_PATH),
        "classes": CLASS_NAMES
    })

@app.route('/')
def index():
    """Home page"""
    from datetime import datetime
    today = datetime.now().date()
    return render_template('index.html', today=today)

@app.route('/signup', methods=['GET', 'POST'])
@limiter.limit('10 per hour', methods=['POST'])
def signup():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        full_name = request.form['full_name']
        if len(password) < 8:
            flash('Password must be at least 8 characters.', 'error')
            return render_template('signup.html')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('signup.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return render_template('signup.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            full_name=full_name
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit('10 per 15 minutes', methods=['POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash(f'Welcome back, {user.full_name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/quick-analysis', methods=['GET', 'POST'])
@login_required
def quick_analysis():
    """Quick analysis flow - Form + Analysis in one page"""
    from datetime import datetime
    
    if request.method == 'POST':
        # Handle patient form submission
        if 'patient_data' in request.form:
            # Create patient from form data
            patient = Patient(
                name=request.form['name'],
                age=int(request.form['age']),
                gender=request.form['gender'],
                phone=request.form['phone'],
                email=request.form.get('email', ''),
                address=request.form.get('address', ''),
                medical_history=request.form.get('medical_history', ''),
                doctor_name=request.form.get('doctor_name', ''),
                doctor_phone=request.form.get('doctor_phone', ''),
                user_id=current_user.id
            )
            
            db.session.add(patient)
            db.session.commit()
            
            flash('Patient information saved successfully!', 'success')
            return redirect(url_for('quick_analysis_upload', patient_id=patient.id))
    
    return render_template('quick_analysis.html')

@app.route('/quick-analysis/<int:patient_id>/upload', methods=['GET', 'POST'])
@login_required
@limiter.limit('20 per hour', methods=['POST'])
def quick_analysis_upload(patient_id):
    patient = Patient.query.filter_by(id=patient_id, user_id=current_user.id).first_or_404()
    if request.method == 'POST':
        try:
            filename, filepath = save_upload(request.files.get('file'))
            result = predict_image(filepath)
            gradcam_filename = generate_gradcam(filepath, result['class_index'], f"gradcam_{uuid.uuid4().hex}.jpg") if result['status'] == 'classified' else None
            suggestion = get_medical_suggestion(result['confidence_band'], result['prediction'], result['status'])
            analysis = Analysis(image_filename=filename, gradcam_filename=gradcam_filename, prediction=result['prediction'], confidence=result['confidence'], severity=result['confidence_band'], status=result['status'], probability_json=json.dumps(result['probabilities']), medical_suggestion=json.dumps(suggestion), patient_id=patient_id)
            db.session.add(analysis); db.session.commit()
            return redirect(url_for('quick_results', analysis_id=analysis.id))
        except ValueError as exc:
            flash(str(exc), 'error')
        except Exception:
            db.session.rollback(); app.logger.exception('Quick analysis failed'); flash('Analysis failed. Please try again.', 'error')
        return redirect(url_for('quick_analysis_upload', patient_id=patient_id))
    return render_template('quick_upload.html', patient=patient)

@app.route('/quick-results/<int:analysis_id>')
@login_required
def quick_results(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    if analysis.patient.user_id != current_user.id:
        abort(403)
    suggestion = get_medical_suggestion(analysis.severity, analysis.prediction, analysis.status)
    probs = None
    if analysis.probability_json:
        try:
            probs = json.loads(analysis.probability_json)
        except Exception:
            probs = None
    return render_template('quick_results.html', patient=analysis.patient, analysis=analysis, image_path=analysis.image_filename, gradcam_path=analysis.gradcam_filename, prediction=analysis.prediction, confidence=analysis.confidence, confidence_band=analysis.severity, confidence_class=confidence_css_class(analysis.severity), status=analysis.status, medical_suggestion=suggestion, probabilities=probs)

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    patients = Patient.query.filter_by(user_id=current_user.id).all()
    from datetime import datetime
    today = datetime.now().date()
    return render_template('dashboard.html', patients=patients, today=today)

@app.route('/patient/new', methods=['GET', 'POST'])
@login_required
def new_patient():
    """Add new patient"""
    if request.method == 'POST':
        patient = Patient(
            name=request.form['name'],
            age=int(request.form['age']),
            gender=request.form['gender'],
            sugar_level=float(request.form['sugar_level']) if request.form['sugar_level'] else None,
            blood_pressure=request.form['blood_pressure'] if request.form['blood_pressure'] else None,
            weight=float(request.form['weight']) if request.form['weight'] else None,
            height=float(request.form['height']) if request.form['height'] else None,
            doctor_name=request.form['doctor_name'] if request.form['doctor_name'] else None,
            phone=request.form['phone'] if request.form['phone'] else None,
            address=request.form['address'] if request.form['address'] else None,
            medical_history=request.form.get('medical_history', ''),
            user_id=current_user.id
        )
        
        db.session.add(patient)
        db.session.commit()
        
        flash(f'Patient {patient.name} added successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('patient_form.html')

@app.route('/patient/<int:patient_id>')
@login_required
def patient_detail(patient_id):
    """Patient detail page"""
    patient = Patient.query.filter_by(id=patient_id, user_id=current_user.id).first_or_404()
    analyses = Analysis.query.filter_by(patient_id=patient_id).order_by(Analysis.created_at.desc()).all()
    return render_template('patient_detail.html', patient=patient, analyses=analyses)

@app.route('/analysis/<int:patient_id>')
@login_required
def analysis_page(patient_id):
    """Analysis page for patient"""
    patient = Patient.query.filter_by(id=patient_id, user_id=current_user.id).first_or_404()
    return render_template('analysis.html', patient=patient)

@app.route('/predict/<int:patient_id>', methods=['POST'])
@login_required
@limiter.limit('20 per hour', methods=['POST'])
def predict(patient_id):
    patient = Patient.query.filter_by(id=patient_id, user_id=current_user.id).first_or_404()
    try:
        filename, filepath = save_upload(request.files.get('file'))
        result = predict_image(filepath)
        gradcam_filename = generate_gradcam(filepath, result['class_index'], f"gradcam_{uuid.uuid4().hex}.jpg") if result['status'] == 'classified' else None
        suggestion = get_medical_suggestion(result['confidence_band'], result['prediction'], result['status'])
        analysis = Analysis(image_filename=filename, gradcam_filename=gradcam_filename, prediction=result['prediction'], confidence=result['confidence'], severity=result['confidence_band'], status=result['status'], probability_json=json.dumps(result['probabilities']), medical_suggestion=json.dumps(suggestion), patient_id=patient_id)
        db.session.add(analysis); db.session.commit()
        return render_template('result.html', patient=patient, analysis=analysis, image_path=filename, gradcam_path=gradcam_filename, prediction=result['prediction'], confidence=result['confidence'], confidence_band=result['confidence_band'], confidence_class=confidence_css_class(result['confidence_band']), status=result['status'], medical_suggestion=suggestion, probabilities=result.get('probabilities'))
    except ValueError as exc:
        flash(str(exc), 'error')
    except Exception:
        db.session.rollback(); app.logger.exception('Prediction failed'); flash('Prediction failed. Please try again.', 'error')
    return redirect(url_for('analysis_page', patient_id=patient_id))

@app.route('/protected-image/<path:filename>')
@login_required
def protected_image(filename):
    safe_name = os.path.basename(filename)
    analysis = Analysis.query.filter((Analysis.image_filename == safe_name) | (Analysis.gradcam_filename == safe_name)).first()
    if analysis and analysis.patient.user_id != current_user.id:
        abort(403)
    quick = session.get('quick_result') or {}
    if not analysis and safe_name not in {quick.get('image_path'), quick.get('gradcam_path')}:
        abort(403)
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
    if not os.path.isfile(full_path):
        abort(404)
    return send_file(full_path, conditional=True)

@app.route('/download_report/<int:analysis_id>')
@login_required
def download_report(analysis_id):
    """Download PDF report for analysis"""
    analysis = Analysis.query.get_or_404(analysis_id)
    patient = Patient.query.filter_by(id=analysis.patient_id, user_id=current_user.id).first_or_404()
    
    # Generate PDF
    pdf_buffer = generate_pdf_report(patient, analysis)
    
    # Create filename
    filename = f"oral_cancer_report_{patient.name.replace(' ', '_')}_{analysis.created_at.strftime('%Y%m%d_%H%M%S')}.pdf"
    
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )

@app.route('/upload')
@login_required
def upload_page():
    """Upload page for image analysis"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
@login_required
@limiter.limit('20 per hour', methods=['POST'])
def upload_and_predict():
    try:
        filename, filepath = save_upload(request.files.get('file'))
        result = predict_image(filepath)
        gradcam_filename = generate_gradcam(filepath, result['class_index'], f"gradcam_{uuid.uuid4().hex}.jpg") if result['status'] == 'classified' else None
        suggestion = get_medical_suggestion(result['confidence_band'], result['prediction'], result['status'])
        session['quick_result'] = {'image_path': filename, 'gradcam_path': gradcam_filename, 'prediction': result['prediction'], 'confidence': result['confidence'], 'confidence_band': result['confidence_band'], 'status': result['status'], 'confidence_class': confidence_css_class(result['confidence_band']), 'medical_suggestion': suggestion, 'probabilities': result['probabilities'], 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        return redirect(url_for('upload_result'))
    except ValueError as exc:
        flash(str(exc), 'error')
    except Exception:
        app.logger.exception('Upload prediction failed'); flash('Analysis failed. Please try again.', 'error')
    return redirect(url_for('upload_page'))

@app.route('/upload/result')
@login_required
def upload_result():
    """Show results from upload analysis"""
    result_data = session.get('quick_result')
    if not result_data:
        flash('No analysis results found. Please upload an image first.', 'error')
        return redirect(url_for('upload_page'))
    
    return render_template('result.html', **result_data, patient=None, analysis=None)

@app.route('/upload/download-pdf')
@login_required
def download_upload_pdf():
    """Download PDF report for upload analysis"""
    result_data = session.get('quick_result')
    if not result_data:
        flash('No analysis results found. Please upload an image first.', 'error')
        return redirect(url_for('upload_page'))
    
    # Generate PDF
    pdf_buffer = generate_quick_pdf_report(result_data)
    
    # Create filename
    timestamp = result_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    safe_timestamp = timestamp.replace(':', '-').replace(' ', '_')
    filename = f"oral_cancer_quick_analysis_{safe_timestamp}.pdf"
    
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=filename,
        mimetype='application/pdf'
    )

@app.route('/recommendations')
def recommendations():
    """Recommendations page"""
    return render_template('recommendations.html')

@app.route('/about')
def about():
    """About page with CNN/Keras information"""
    return render_template('about.html')

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Please upload a file smaller than 16MB.', 'error')
    return redirect(request.referrer or url_for('dashboard'))

@app.route('/test-prediction')
@login_required
def test_prediction():
    """Simple deployment/model diagnostic endpoint."""
    current_model = get_model()
    if current_model is None:
        return jsonify({"status": "error", "model_loaded": False, "error": MODEL_LOAD_ERROR}), 503
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model": os.path.basename(MODEL_PATH),
        "classes": CLASS_NAMES
    })

@app.route('/chatbot', methods=['POST'])
@limiter.limit('30 per hour', methods=['POST'])
def chatbot():
    """Safe educational chatbot endpoint with server-verified result context."""
    payload = request.get_json(silent=True) or {}
    message = str(payload.get('message', '')).strip()
    if not message or len(message) > 1000:
        return jsonify({'error': 'Message must contain 1-1000 characters.'}), 400

    info = None
    analysis_id = payload.get('analysis_id')
    if analysis_id and current_user.is_authenticated:
        try:
            analysis = db.session.get(Analysis, int(analysis_id))
        except (TypeError, ValueError):
            analysis = None
        # Only use records owned by the signed-in user.
        if analysis and analysis.patient and analysis.patient.user_id == current_user.id:
            info = {
                'prediction': analysis.prediction,
                'confidence': analysis.confidence,
                'confidence_band': analysis.severity,
                'status': analysis.status,
            }
    elif current_user.is_authenticated:
        quick = session.get('quick_result') or {}
        if quick:
            info = {
                'prediction': quick.get('prediction'),
                'confidence': quick.get('confidence'),
                'confidence_band': quick.get('confidence_band'),
                'status': quick.get('status'),
            }

    try:
        answer = generate_chat_response(message, info)
        return jsonify({'response': answer, 'mode': 'ai' if os.getenv('OPENAI_API_KEY', '').strip() else 'knowledge_base'})
    except Exception:
        app.logger.exception('Chatbot request failed')
        return jsonify({'error': 'The assistant is temporarily unavailable. Please try again.'}), 503

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    flash('An internal server error occurred. Please try again.', 'error')
    return redirect(url_for('index'))

def ensure_schema():
    """Create tables and add lightweight SQLite columns needed by current releases."""
    db.create_all()
    try:
        from sqlalchemy import inspect, text
        inspector = inspect(db.engine)
        if inspector.has_table('analysis'):
            columns = {c['name'] for c in inspector.get_columns('analysis')}
            with db.engine.begin() as conn:
                if 'status' not in columns:
                    conn.execute(text("ALTER TABLE analysis ADD COLUMN status VARCHAR(20) DEFAULT 'classified'"))
                if 'probability_json' not in columns:
                    conn.execute(text('ALTER TABLE analysis ADD COLUMN probability_json TEXT'))
    except Exception:
        app.logger.exception('Database schema check failed')
        raise

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        ensure_schema()
    
    # Model will be loaded lazily when first needed
    print("🚀 Starting Flask app - model will load on first prediction")
    
    # Run the app. PORT and FLASK_DEBUG are read from the environment so this
    # works both locally (defaults: port 5000, debug off) and on hosting
    # platforms that assign a dynamic port (Render, Railway, Heroku, etc).
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.getenv('PORT', 5000))

    if os.getenv('AUTO_OPEN_BROWSER', 'true').lower() == 'true':
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f'http://127.0.0.1:{port}')).start()

    app.run(debug=debug_mode, host='0.0.0.0', port=port)
else:
    # When running under a production WSGI server (e.g. `gunicorn app:app`),
    # this module is imported rather than executed as __main__, so the
    # block above never runs. Create tables here too so a fresh deployment
    # doesn't 500 on its first request.
    with app.app_context():
        ensure_schema()

