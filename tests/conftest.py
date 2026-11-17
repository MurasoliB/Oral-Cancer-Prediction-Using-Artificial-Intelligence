import os, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
os.environ.setdefault('FLASK_SECRET_KEY','test-secret-key-1234567890')
os.environ.setdefault('FLASK_ENV','testing')
@pytest.fixture()
def client(tmp_path):
    import app as application
    application.app.config.update(TESTING=True,WTF_CSRF_ENABLED=False,SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path/'test.db'}",UPLOAD_FOLDER=str(tmp_path/'uploads'))
    os.makedirs(application.app.config['UPLOAD_FOLDER'],exist_ok=True)
    with application.app.app_context(): application.db.drop_all(); application.db.create_all()
    with application.app.test_client() as c: yield c
