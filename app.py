import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from app.utils import init_db
from app.routes import bp, get_model


def create_app():
    """
    Creates and configures the Flask application.
    """

    app = Flask(
        __name__,
        template_folder="app/templates",
        static_folder="app/static"
    )

    # Secret key
    app.secret_key = os.environ.get(
        "SECRET_KEY",
        "dev-secret-key-change-in-production"
    )

    # Register blueprint routes
    app.register_blueprint(bp)

    return app


# ==========================================
# GLOBAL APP OBJECT FOR GUNICORN / RENDER
# ==========================================

app = create_app()

# Initialize database
init_db()

# Preload ML model
model, scaler = get_model()

if model:
    print("[STARTUP] Pre-trained model loaded successfully")
else:
    print("[STARTUP] No trained model found")


# ==========================================
# LOCAL DEVELOPMENT SERVER
# ==========================================

if __name__ == "__main__":

    PORT = int(os.environ.get("PORT", 5000))

    print(f"[STARTUP] Flask server running on port {PORT}")

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=True
    )