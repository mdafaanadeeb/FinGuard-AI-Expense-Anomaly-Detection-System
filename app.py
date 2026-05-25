import os
import sys

# Ensure /app is importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from app.utils import init_db
from app.routes import bp, get_model


def create_app():
    """
    Application factory pattern.
    Creates and configures the Flask app.
    """
    app = Flask(
        __name__,
        template_folder="app/templates",
        static_folder="app/static"
    )

    # Secret key for session (use env variable in production)
    app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")

    # Register routes blueprint
    app.register_blueprint(bp)

    return app


# 🚀 RUN BLOCK (LOCAL DEVELOPMENT ONLY)
if __name__ == "__main__":
    print("[STARTUP] Initializing application...")

    # Initialize DB
    init_db()
    print("[STARTUP] Database initialized")

    # Pre-load model (optional optimization)
    model, scaler = get_model()
    if model:
        print("[STARTUP] Pre-trained model loaded successfully")
    else:
        print("[STARTUP] No pre-trained model found — will train on first upload")

    # Create Flask app
    app = create_app()

    # Get port dynamically (important for deployment)
    PORT = int(os.environ.get("PORT", 5000))

    print(f"[STARTUP] Server starting on http://0.0.0.0:{PORT}")

    # Run server
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=True,
        use_reloader=False
    )
