from app import app, db
from sqlalchemy import inspect, text
with app.app_context():
    inspector=inspect(db.engine)
    cols={c['name'] for c in inspector.get_columns('analysis')} if inspector.has_table('analysis') else set()
    with db.engine.begin() as conn:
        if 'status' not in cols: conn.execute(text("ALTER TABLE analysis ADD COLUMN status VARCHAR(20) DEFAULT 'classified'"))
        if 'probability_json' not in cols: conn.execute(text('ALTER TABLE analysis ADD COLUMN probability_json TEXT'))
    db.create_all()
    print('Migration complete.')
