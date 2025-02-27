from .extensions import db 
from datetime import datetime

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firebase_id = db.Column(db.String(255), nullable=False, unique=True)
    name = db.Column(db.String(255), nullable=False)
    dob = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(50), nullable=True)
    occupation = db.Column(db.String(255), nullable=True)  # Can include "student" as a value
    marital_status = db.Column(db.Boolean, nullable=True)
    city = db.Column(db.String(100), nullable=True)
    residence_type = db.Column(db.Boolean, nullable=True)  # 'rural' or 'urban'
    category = db.Column(db.String(50), nullable=True)  # 'general', 'OBC', etc.
    differently_abled = db.Column(db.Boolean, nullable=True, default=False)
    disability_percentage = db.Column(db.Float, nullable=True)
    minority = db.Column(db.Boolean, nullable=True)
    bpl_category = db.Column(db.Boolean, nullable=True)
    income = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Scheme(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    scheme_name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    launch_date = db.Column(db.Date, nullable=True)
    expiry_date = db.Column(db.Date, nullable=True)
    age_range = db.Column(db.String(50), nullable=True)
    income = db.Column(db.Float, nullable=True)
    occupation = db.Column(db.String(100), nullable=True)  # Can include "student" as a value
    residence_type = db.Column(db.Boolean, nullable=True)  # Changed to match User table: 'rural' or 'urban'
    city = db.Column(db.String(100), nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    caste = db.Column(db.String(100), nullable=True)
    benefit_type = db.Column(db.String(100), nullable=True)
    differently_abled = db.Column(db.Boolean, nullable=True)
    marital_status = db.Column(db.String(50), nullable=True)
    disability_percentage = db.Column(db.Float, nullable=True)
    minority = db.Column(db.Boolean, nullable=True)  # Added to match User table
    bpl_category = db.Column(db.Boolean, nullable=True)  # Added to match User table
    department = db.Column(db.String(255), nullable=True)
    application_link = db.Column(db.String(500), nullable=True)
    required_documents = db.Column(db.Text, nullable=True)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
