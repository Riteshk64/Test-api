from myapp import create_app
from myapp.extensions import db
from flask_migrate import Migrate
from myapp.models import Scheme, SchemeRating

app = create_app()
migrate = Migrate(app, db)