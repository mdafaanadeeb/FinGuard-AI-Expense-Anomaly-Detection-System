import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from app.utils import init_db
from app.routes import bp, get_model


def create_app():

    app = Flask(
        __name__,
        template_folder="app/templates",
        static_folder="app/static"
    )

    app.secret_key = os.environ.get(
        "SECRET_KEY",
        "dev-secret-key"
    )

    app.register_blueprint(bp)

    return app


# Global Flask app for Gunicorn
app = create_app()

# Initialize database
init_db()

# Preload model
model, scaler = get_model()


if __name__ == "__main__":

    PORT = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=True
    )