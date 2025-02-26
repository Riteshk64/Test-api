import os

from flask import Flask 

from .extensions import db
from .routes import main

def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
    # postgresql://postgres_test_rl52_user:g1nOSLY9BN98KaWtZnLIpJjKt7Dov1fl@dpg-cuvjgc9u0jms739lkrrg-a.singapore-postgres.render.com/postgres_test_rl52

    db.init_app(app)

    app.register_blueprint(main)

    return app