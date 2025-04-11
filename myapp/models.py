from .extensions import db 
from datetime import datetime
from sqlalchemy.dialects.postgresql import BYTEA

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    firebase_id = db.Column(db.String(255), nullable=False, unique=True)
    name = db.Column(db.String(255), nullable=False)
    dob = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(50), nullable=True)
    occupation = db.Column(db.String(255), nullable=True)
    marital_status = db.Column(db.String(50), nullable=True)
    city = db.Column(db.String(100), nullable=True)
    residence_type = db.Column(db.String(20), nullable=True)  # 'rural' or 'urban'
    category = db.Column(db.String(50), nullable=True)  # 'general', 'OBC', etc.
    differently_abled = db.Column(db.Boolean, nullable=True, default=False)
    disability_percentage = db.Column(db.Float, nullable=True)
    minority = db.Column(db.Boolean, nullable=True)
    bpl_category = db.Column(db.Boolean, nullable=True)
    income = db.Column(db.Float, nullable=True)
    education_level = db.Column(db.String(100), nullable=True)  # Added for faceted search
    preferred_language = db.Column(db.String(50), nullable=True)  # For multilingual support
    notification_preferences = db.Column(db.Boolean, default=True)  # For notification system
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, nullable=True)  # Track user activity


class Scheme(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    scheme_name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    launch_date = db.Column(db.Date, nullable=True)
    expiry_date = db.Column(db.Date, nullable=True)
    age_range = db.Column(db.String(50), nullable=True)
    income = db.Column(db.Float, nullable=True)
    occupation = db.Column(db.String(100), nullable=True)
    residence_type = db.Column(db.String(20), nullable=True)  # 'rural' or 'urban'
    city = db.Column(db.String(100), nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    caste = db.Column(db.String(100), nullable=True)
    benefit_type = db.Column(db.String(100), nullable=True)
    differently_abled = db.Column(db.Boolean, nullable=True)
    marital_status = db.Column(db.String(50), nullable=True)
    disability_percentage = db.Column(db.Float, nullable=True)
    minority = db.Column(db.Boolean, nullable=True)
    bpl_category = db.Column(db.Boolean, nullable=True)
    department = db.Column(db.String(255), nullable=True)
    application_link = db.Column(db.String(500), nullable=True)
    scheme_details = db.Column(db.Text, nullable=True)  # More detailed information
    keywords = db.Column(db.Text, nullable=True)  # For better NLP matching
    popularity_score = db.Column(db.Float, default=0.0)  # For relevance ranking
    local_body = db.Column(db.String(100), nullable=True)  # For filtering by local governing bodies
    education_criteria = db.Column(db.String(255), nullable=True)  # Specific education requirements
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    embedding = db.Column(BYTEA, nullable=True)
    
    # Optional fields for multilingual support
    description_marathi = db.Column(db.Text, nullable=True)
    scheme_details_marathi = db.Column(db.Text, nullable=True)

class UserBookmark(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    scheme_id = db.Column(db.Integer, db.ForeignKey('scheme.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)  # Optional user notes about the bookmarked scheme
    
    # Relationships
    user = db.relationship('User', backref=db.backref('bookmarks', lazy=True))
    scheme = db.relationship('Scheme', backref=db.backref('bookmarked_by', lazy=True))
    
    # Ensure a user can bookmark a scheme only once
    _table_args_ = (db.UniqueConstraint('user_id', 'scheme_id', name='unique_user_scheme_bookmark'),)