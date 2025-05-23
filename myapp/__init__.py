import os

from flask import Flask 

from .extensions import db, migrate
from .routes import api
from . import models

def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")

    db.init_app(app)
    migrate.init_app(app, db)

    app.register_blueprint(api)

    @app.cli.command("update-embeddings")
    def cli_update_embeddings():
        from myapp.utils import update_scheme_embeddings
        update_scheme_embeddings()

    return app