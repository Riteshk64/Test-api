from flask import Blueprint, request, jsonify
from datetime import datetime
from sqlalchemy import or_
import math
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from sqlalchemy.sql import func

from .extensions import db
from .models import User, Scheme, UserBookmark, SchemeRating
from marshmallow import Schema, fields, validate, ValidationError

# Create Blueprint
api = Blueprint('api', __name__)

# --------------------- Validation Schemas ---------------------

class UserSchema(Schema):
    firebase_id = fields.String(required=True)
    name = fields.String(required=True)
    dob = fields.Date(format='%Y-%m-%d', required=False, allow_none=True)
    gender = fields.String(required=False, allow_none=True, validate=validate.OneOf(['male', 'female', 'other', None]))
    occupation = fields.String(required=False, allow_none=True)
    marital_status = fields.String(required=False, allow_none=True, 
                                  validate=validate.OneOf(['single', 'married', 'divorced', 'widowed', None]))
    city = fields.String(required=False, allow_none=True)
    residence_type = fields.String(required=False, allow_none=True, 
                                  validate=validate.OneOf(['urban', 'rural', 'semi-urban', None]))
    category = fields.String(required=False, allow_none=True)
    differently_abled = fields.Boolean(required=False, allow_none=True)
    disability_percentage = fields.Integer(required=False, allow_none=True, 
                                         validate=validate.Range(min=0, max=100))
    minority = fields.Boolean(required=False, allow_none=True)
    bpl_category = fields.Boolean(required=False, allow_none=True)
    income = fields.Float(required=False, allow_none=True, validate=validate.Range(min=0))
    education_level = fields.String(required=False, allow_none=True)
    preferred_language = fields.String(required=False, default='en')

class BookmarkSchema(Schema):
    scheme_id = fields.Integer(required=True)
    notes = fields.String(required=False, allow_none=True)

class PaginationParamsSchema(Schema):
    page = fields.Integer(required=False, missing=1, validate=validate.Range(min=1))
    per_page = fields.Integer(required=False, missing=10, validate=validate.Range(min=1, max=100))

# --------------------- Helper Functions ---------------------

def paginate_results(query, page, per_page):
    """Helper function to paginate SQLAlchemy query results"""
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    total_pages = math.ceil(pagination.total / per_page) if pagination.total > 0 else 0
    
    return {
        "items": pagination.items,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total_items": pagination.total,
            "total_pages": total_pages,
            "has_next": pagination.has_next,
            "has_prev": pagination.has_prev
        }
    }

def handle_validation_error(error):
    """Helper function to format validation errors"""
    return jsonify({"error": "Validation error", "details": error.messages}), 400

# Load spaCy model - you'll need to install it first with:
# python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md')

def preprocess_text(text):
    """Preprocess text by removing stopwords and punctuation"""
    if not text or not isinstance(text, str):
        return ""
    
    doc = nlp(text.lower())
    # Remove stopwords and punctuation
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def get_text_embedding(text):
    """Convert text to embedding vector using spaCy"""
    if not text or not isinstance(text, str):
        return None
    
    processed_text = preprocess_text(text)
    if not processed_text:
        return None
    
    # Get document vector
    doc = nlp(processed_text)
    return doc.vector

def extract_keywords(text, num_keywords=5):
    """Extract important keywords from text"""
    if not text or not isinstance(text, str):
        return []
    
    doc = nlp(text.lower())
    
    # Get all tokens that are not stopwords or punctuation
    keywords = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.pos_ in ('NOUN', 'PROPN', 'ADJ'):
            keywords.append(token.text)
    
    # Count frequencies
    keyword_freq = {}
    for word in keywords:
        if word in keyword_freq:
            keyword_freq[word] += 1
        else:
            keyword_freq[word] = 1
    
    # Sort by frequency and return top N
    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_keywords[:num_keywords]]

# Enhanced search function with NLP
def search_schemes_nlp(query_text, schemes, max_results=20):
    """Search schemes using precomputed embeddings and semantic similarity."""
    if not query_text:
        return []

    query_embedding = get_text_embedding(query_text)
    if query_embedding is None:
        return []

    results = []

    for scheme in schemes:
        if not scheme.embedding:
            continue  # Skip if missing embedding

        try:
            scheme_embedding = np.frombuffer(scheme.embedding, dtype=np.float32)
        except Exception:
            continue

        similarity = cosine_similarity(
            query_embedding.reshape(1, -1),
            scheme_embedding.reshape(1, -1)
        )[0][0]

        if similarity >= 0.2:
            total_ratings = SchemeRating.query.filter_by(scheme_id=scheme.id).count()
            results.append({
                "id": scheme.id,
                "scheme_name": scheme.scheme_name,
                "category": scheme.category,
                "description": scheme.description,
                "similarity": float(similarity),
                "keywords": extract_keywords(scheme.description, 3),
                "average_rating": round(scheme.average_rating or 0.0, 2),
                "total_ratings": total_ratings,
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:max_results]

def detect_categories_from_query(query, categories, synonyms):
    detected = set()
    query_lower = query.lower()

    for word, category in synonyms.items():
        if word in query_lower and category in categories:
            detected.add(category)

    # Optional: Match exact category names if used directly
    for category in categories:
        if category in query_lower:
            detected.add(category)

    return list(detected)

def normalize_number_phrase(text):
    """
    Convert number phrases like '3 lakhs', '50k', '2 crore' to a float value.
    """
    text = text.lower().replace(',', '').strip()

    match = re.search(r'(\d+\.?\d*)\s*(lakh|lakhs|k|thousand|crore)', text)
    if match:
        num = float(match.group(1))
        unit = match.group(2)

        if unit in ['lakh', 'lakhs']:
            return num * 100000
        elif unit in ['k', 'thousand']:
            return num * 1000
        elif unit == 'crore':
            return num * 10000000

    plain_num = re.search(r'\d{5,}', text)
    if plain_num:
        return float(plain_num.group())

    return None

def detect_filters_from_query(query_text):
    """Extract gender, residence_type, city, income, and age from query"""
    doc = nlp(query_text.lower())
    residence_options = {'rural', 'urban', 'semi-urban'}
    gender_options = {'male', 'female', 'other'}

    filters = {
        'residence_type': None,
        'gender': None,
        'city': None,
        'income': None,
        'age': None
    }

    # --- Gender & Residence Detection ---
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in residence_options:
            filters['residence_type'] = lemma
        elif lemma in gender_options:
            filters['gender'] = lemma

    # --- City Detection from DB ---
    known_cities = {row.city.lower() for row in db.session.query(Scheme.city).distinct() if row.city}
    for token in doc:
        if token.text.lower() in known_cities:
            filters['city'] = token.text
            break

    # --- Income Detection ---
    income_match = re.search(
        r'(under|below|less than|upto|up to|above|income of|income is|income)\s+([^\s]+(?:\s*(?:lakh|lakhs|k|crore|thousand))?)',
        query_text.lower()
    )
    if income_match:
        value = normalize_number_phrase(income_match.group(2))
        if value:
            filters['income'] = value

    # --- Age Detection ---
    age_match = re.search(r'(above|over|older than)?\s*(\d{2,3})\s*(years|year|yo)?', query_text.lower())
    if age_match:
        try:
            filters['age'] = int(age_match.group(2))
        except:
            pass

    # --- Age keywords ---
    if 'senior citizen' in query_text.lower():
        filters['age'] = 60
    elif 'youth' in query_text.lower():
        filters['age'] = 25

    return filters

def scheme_matches_age(age, age_range_str):
    if not age_range_str or not isinstance(age_range_str, str):
        return True  # No restriction

    try:
        age_range_str = age_range_str.strip().lower()

        if '+' in age_range_str:
            min_age = int(age_range_str.replace('+', '').strip())
            return age >= min_age

        if 'to' in age_range_str:
            parts = age_range_str.split('to')
            if len(parts) == 2:
                min_age = int(parts[0].strip())
                max_age = int(parts[1].strip())
                return min_age <= age <= max_age

        if age_range_str.isdigit():
            return age == int(age_range_str)

    except Exception:
        return False  # If parsing fails, exclude it

    return False

# --------------------- User Routes ---------------------

@api.route('/user', methods=['POST'])
def create_user():
    try:
        # Validate input data
        schema = UserSchema()
        data = schema.load(request.get_json())
        
        # Check if user with this firebase_id already exists
        existing_user = User.query.filter_by(firebase_id=data.get('firebase_id')).first()
        if existing_user:
            return jsonify({"error": "User with this firebase_id already exists"}), 400
        
        # Create new user
        new_user = User(
            firebase_id=data.get('firebase_id'),
            name=data.get('name'),
            dob=data.get('dob'),
            gender=data.get('gender'),
            occupation=data.get('occupation'),
            marital_status=data.get('marital_status'),
            city=data.get('city'),
            residence_type=data.get('residence_type'),
            category=data.get('category'),
            differently_abled=data.get('differently_abled'),
            disability_percentage=data.get('disability_percentage'),
            minority=data.get('minority'),
            bpl_category=data.get('bpl_category'),
            income=data.get('income'),
            education_level=data.get('education_level'),
            preferred_language=data.get('preferred_language', 'en')
        )
        
        db.session.add(new_user)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        
        return jsonify({"message": "User created successfully", "user_id": new_user.id}), 201
    
    except ValidationError as err:
        return handle_validation_error(err)
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@api.route('/user/<string:firebase_id>', methods=['GET'])
def get_user(firebase_id):
    try:
        user = User.query.filter_by(firebase_id=firebase_id).first()
        if not user:
            return jsonify({"error": f"User with firebase_id {firebase_id} not found"}), 404
        
        # Convert user object to dictionary
        user_data = {
            "id": user.id,
            "firebase_id": user.firebase_id,
            "name": user.name,
            "dob": user.dob.strftime('%Y-%m-%d') if user.dob else None,
            "gender": user.gender,
            "occupation": user.occupation,
            "marital_status": user.marital_status,
            "city": user.city,
            "residence_type": user.residence_type,
            "category": user.category,
            "differently_abled": user.differently_abled,
            "disability_percentage": user.disability_percentage,
            "minority": user.minority,
            "bpl_category": user.bpl_category,
            "income": user.income,
            "education_level": user.education_level,
            "preferred_language": user.preferred_language,
            "created_at": user.created_at.strftime('%Y-%m-%d %H:%M:%S') if user.created_at else None,
            "last_active": user.last_active.strftime('%Y-%m-%d %H:%M:%S') if user.last_active else None
        }
        
        return jsonify(user_data), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@api.route('/user/<string:firebase_id>', methods=['PUT'])
def update_user(firebase_id):
    try:
        user = User.query.filter_by(firebase_id=firebase_id).first()
        if not user:
            return jsonify({"error": f"User with firebase_id {firebase_id} not found"}), 404
        
        # Get JSON data directly without schema validation
        data = request.get_json()
        
        # Update user fields if provided in request
        if 'name' in data:
            user.name = data['name']
        if 'dob' in data:
            user.dob = data['dob']
        if 'gender' in data:
            user.gender = data['gender']
        if 'occupation' in data:
            user.occupation = data['occupation']
        if 'marital_status' in data:
            user.marital_status = data['marital_status']
        if 'city' in data:
            user.city = data['city']
        if 'residence_type' in data:
            user.residence_type = data['residence_type']
        if 'category' in data:
            user.category = data['category']
        if 'differently_abled' in data:
            user.differently_abled = data['differently_abled']
        if 'disability_percentage' in data:
            user.disability_percentage = data['disability_percentage']
        if 'minority' in data:
            user.minority = data['minority']
        if 'bpl_category' in data:
            user.bpl_category = data['bpl_category']
        if 'income' in data:
            user.income = data['income']
        if 'education_level' in data:
            user.education_level = data['education_level']
        if 'preferred_language' in data:
            user.preferred_language = data['preferred_language']
        
        user.last_active = datetime.utcnow()
        
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        
        return jsonify({"message": "User updated successfully"}), 200
    
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@api.route('/user/<string:firebase_id>', methods=['DELETE'])
def delete_user(firebase_id):
    try:
        user = User.query.filter_by(firebase_id=firebase_id).first()
        if not user:
            return jsonify({"error": f"User with firebase_id {firebase_id} not found"}), 404
        
        # Delete related records first (bookmarks)
        UserBookmark.query.filter_by(user_id=user.id).delete()
        
        db.session.delete(user)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        
        return jsonify({"message": "User deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --------------------- Bookmark Routes ---------------------

@api.route('/bookmarks', methods=['GET'])
def get_bookmarks():
    try:
        # Parse and validate pagination parameters
        pagination_schema = PaginationParamsSchema()
        try:
            pagination_params = pagination_schema.load({
            "page": request.args.get("page", type=int),
            "per_page": request.args.get("per_page", type=int)
        })
        except ValidationError as err:
            return handle_validation_error(err)
        
        page = pagination_params['page']
        per_page = pagination_params['per_page']
        
        user_id = request.args.get('user_id', type=int)
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": f"User with firebase_id {firebase_id} not found"}), 404
        
        # Query with ordering by timestamp
        query = UserBookmark.query.filter_by(user_id=user.id).order_by(UserBookmark.timestamp.desc())
        
        # Paginate results
        pagination_result = paginate_results(query, page, per_page)
        bookmarks = pagination_result['items']
        
        result = []
        for bookmark in bookmarks:
            scheme = bookmark.scheme
            total_ratings = SchemeRating.query.filter_by(scheme_id=scheme.id).count()
            result.append({
                "bookmark_id": bookmark.id,
                "scheme_id": scheme.id,
                "scheme_name": scheme.scheme_name,
                "category": scheme.category,
                "description": scheme.description,
                "bookmarked_at": bookmark.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "notes": bookmark.notes,
                "total_ratings": total_ratings,
                "average_rating": round(scheme.average_rating or 0.0, 2),
            })
        
        # Return paginated response
        return jsonify({
            "bookmarks": result,
            "pagination": pagination_result['pagination']
        }), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@api.route('/bookmarks', methods=['POST'])
def create_bookmark():
    try:
        # Get firebase_id from request body
        data = request.get_json()
        firebase_id = data.get('firebase_id')
        
        user = User.query.filter_by(firebase_id=firebase_id).first()
        if not user:
            return jsonify({"error": f"User with firebase_id {firebase_id} not found"}), 404
        
        # Validate input data
        schema = BookmarkSchema()
        try:
            bookmark_data = schema.load({
                "scheme_id": data.get("scheme_id"),
                "notes": data.get("notes")
            })
        except ValidationError as err:
            return handle_validation_error(err)
        
        scheme_id = bookmark_data.get('scheme_id')
        notes = bookmark_data.get('notes')
        
        # Check if scheme exists
        scheme = Scheme.query.get(scheme_id)
        if not scheme:
            return jsonify({"error": f"Scheme with ID {scheme_id} not found"}), 404
        
        # Check if bookmark already exists
        existing_bookmark = UserBookmark.query.filter_by(user_id=user.id, scheme_id=scheme_id).first()
        if existing_bookmark:
            return jsonify({"error": "Bookmark already exists", "bookmark_id": existing_bookmark.id}), 400
        
        # Create bookmark
        bookmark = UserBookmark(
            user_id=user.id,
            scheme_id=scheme_id,
            notes=notes
        )
        
        db.session.add(bookmark)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        
        return jsonify({
            "message": "Bookmark created successfully", 
            "bookmark_id": bookmark.id
        }), 201
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@api.route('/bookmarks/<int:bookmark_id>', methods=['PUT'])
def update_bookmark(bookmark_id):
    try:
        # Get firebase_id from request body
        data = request.get_json()
        firebase_id = data.get('firebase_id')
        
        user = User.query.filter_by(firebase_id=firebase_id).first()
        if not user:
            return jsonify({"error": f"User with firebase_id {firebase_id} not found"}), 404
        
        # Get bookmark and verify ownership
        bookmark = UserBookmark.query.get(bookmark_id)
        if not bookmark:
            return jsonify({"error": f"Bookmark with ID {bookmark_id} not found"}), 404
        
        if bookmark.user_id != user.id:
            return jsonify({"error": "Unauthorized access"}), 403
        
        # Update notes
        if 'notes' in data:
            bookmark.notes = data['notes']
            bookmark.timestamp = datetime.utcnow()  # Update timestamp to reflect the change
            
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                return jsonify({"error": f"Database error: {str(e)}"}), 500
            
            return jsonify({"message": "Bookmark updated successfully"}), 200
        else:
            return jsonify({"error": "No update data provided"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@api.route('/bookmarks/<int:bookmark_id>', methods=['DELETE'])
def delete_bookmark(bookmark_id):
    try:
        # Get firebase_id from request body
        data = request.get_json()
        firebase_id = data.get('firebase_id')
        
        user = User.query.filter_by(firebase_id=firebase_id).first()
        if not user:
            return jsonify({"error": f"User with firebase_id {firebase_id} not found"}), 404
        
        # Get bookmark and verify ownership
        bookmark = UserBookmark.query.get(bookmark_id)
        if not bookmark:
            return jsonify({"error": f"Bookmark with ID {bookmark_id} not found"}), 404
        
        if bookmark.user_id != user.id:
            return jsonify({"error": "Unauthorized access"}), 403
        
        db.session.delete(bookmark)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        
        return jsonify({"message": "Bookmark deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --------------------- Scheme Routes ---------------------

@api.route('/schemes', methods=['GET'])
def get_schemes():
    try:
        # Parse and validate only pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Validate pagination params separately instead of validating all request.args
        pagination_schema = PaginationParamsSchema()
        try:
            pagination_params = pagination_schema.load({
                'page': page,
                'per_page': per_page
            })
            page = pagination_params['page']
            per_page = pagination_params['per_page']
        except ValidationError as err:
            return handle_validation_error(err)
        
        # Basic filtering parameters
        category = request.args.get('category')
        gender = request.args.get('gender')
        residence_type = request.args.get('residence_type')
        city = request.args.get('city')
        income = request.args.get('income', type=float)  # Directly take income as parameter
        differently_abled = request.args.get('differently_abled')
        if differently_abled is not None:
            differently_abled = differently_abled.lower() == 'true'
        minority = request.args.get('minority')
        if minority is not None:
            minority = minority.lower() == 'true'
        bpl_category = request.args.get('bpl_category')
        if bpl_category is not None:
            bpl_category = bpl_category.lower() == 'true'
        
        # Start with base query
        query = Scheme.query
        
        # Apply filters if provided
        if category:
            query = query.filter(Scheme.category == category)
        if gender:
            query = query.filter(or_(Scheme.gender == gender, Scheme.gender == None))
        if residence_type:
            query = query.filter(or_(Scheme.residence_type == residence_type, Scheme.residence_type == None))
        if city:
            query = query.filter(or_(Scheme.city == city, Scheme.city == None))
        if income is not None:  # Only filter by income if provided
            query = query.filter(Scheme.income <= income)
        if differently_abled is not None:
            query = query.filter(or_(Scheme.differently_abled == differently_abled, Scheme.differently_abled == None))
        if minority is not None:
            query = query.filter(or_(Scheme.minority == minority, Scheme.minority == None))
        if bpl_category is not None:
            query = query.filter(or_(Scheme.bpl_category == bpl_category, Scheme.bpl_category == None))
        
        # Order by scheme name
        query = query.order_by(Scheme.scheme_name)
        
        # Paginate results
        pagination_result = paginate_results(query, page, per_page)
        schemes = pagination_result['items']
        
        # Format response
        result = []
        for scheme in schemes:
            total_ratings = SchemeRating.query.filter_by(scheme_id=scheme.id).count()
            result.append({
                "id": scheme.id,
                "scheme_name": scheme.scheme_name,
                "category": scheme.category,
                "description": scheme.description,
                "benefit_type": scheme.benefit_type,
                "application_link": scheme.application_link,
                "department": scheme.department,
                "launch_date": scheme.launch_date.strftime('%Y-%m-%d') if scheme.launch_date else None,
                "expiry_date": scheme.expiry_date.strftime('%Y-%m-%d') if scheme.expiry_date else None,
                "average_rating": scheme.average_rating or 0.0,
                "total_ratings": total_ratings,
            })
        
        # Return paginated response
        return jsonify({
            "schemes": result,
            "pagination": pagination_result['pagination']
        }), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@api.route('/schemes/<int:scheme_id>', methods=['GET'])
def get_scheme(scheme_id):
    try:
        scheme = Scheme.query.get(scheme_id)
        if not scheme:
            return jsonify({"error": f"Scheme with ID {scheme_id} not found"}), 404
        
        # Check bookmark status
        firebase_id = request.args.get('firebase_id')
        bookmark_id = None
        user_rating = None

        if firebase_id:
            user = User.query.filter_by(firebase_id=firebase_id).first()
            if user:
                bookmark = UserBookmark.query.filter_by(user_id=user.id, scheme_id=scheme.id).first()
                bookmark_id = bookmark.id if bookmark else None
            
                rating_entry = SchemeRating.query.filter_by(user_id=user.id, scheme_id=scheme.id).first()
                if rating_entry:
                    user_rating = float(rating_entry.rating)
        
        total_ratings = SchemeRating.query.filter_by(scheme_id=scheme.id).count()
        
        # Format response
        result = {
            "id": scheme.id,
            "scheme_name": scheme.scheme_name,
            "category": scheme.category,
            "description": scheme.description,
            "launch_date": scheme.launch_date.strftime('%Y-%m-%d') if scheme.launch_date else None,
            "expiry_date": scheme.expiry_date.strftime('%Y-%m-%d') if scheme.expiry_date else None,
            "age_range": scheme.age_range,
            "income": scheme.income,
            "occupation": scheme.occupation,
            "residence_type": scheme.residence_type,
            "city": scheme.city,
            "gender": scheme.gender,
            "caste": scheme.caste,
            "benefit_type": scheme.benefit_type,
            "differently_abled": scheme.differently_abled,
            "marital_status": scheme.marital_status,
            "disability_percentage": scheme.disability_percentage,
            "minority": scheme.minority,
            "bpl_category": scheme.bpl_category,
            "department": scheme.department,
            "application_link": scheme.application_link,
            "scheme_details": scheme.scheme_details,
            "local_body": scheme.local_body,
            "education_criteria": scheme.education_criteria,
            "keywords": scheme.keywords,
            "average_rating": round(scheme.average_rating or 0.0, 2),
            "total_ratings": total_ratings,
            "bookmark_id": bookmark_id,
            "user_rating": user_rating
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --------------------- Recommendation Routes ---------------------

@api.route('/recommendations', methods=['GET'])
def get_top_rated_schemes():
    try:
        top_schemes = Scheme.query.order_by(Scheme.average_rating.desc().nullslast()).limit(5).all()

        result = []
        for scheme in top_schemes:
            total_ratings = SchemeRating.query.filter_by(scheme_id=scheme.id).count()
            result.append({
                "id": scheme.id,
                "scheme_name": scheme.scheme_name,
                "category": scheme.category,
                "description": scheme.description,
                "average_rating": scheme.average_rating or 0.0,
                "total_ratings": total_ratings,
                "department": scheme.department,
                "benefit_type": scheme.benefit_type
            })

        return jsonify({"top_rated_schemes": result}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

    try:
        # Parse and validate pagination parameters
        pagination_schema = PaginationParamsSchema()
        try:
            pagination_params = pagination_schema.load(request.args, unknown='exclude')
        except ValidationError as err:
            return handle_validation_error(err)
            
        page = pagination_params['page']
        per_page = pagination_params['per_page']
            
        firebase_id = request.args.get('firebase_id')
        user = User.query.filter_by(firebase_id=firebase_id).first()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        # Query schemes based on user profile
        query = Scheme.query
        
        # Filter schemes based on user preferences
        if user.gender:
            query = query.filter(or_(Scheme.gender == user.gender, Scheme.gender == None))
        
        if user.residence_type:
            query = query.filter(or_(Scheme.residence_type == user.residence_type, Scheme.residence_type == None))
        
        if user.city:
            query = query.filter(or_(Scheme.city == user.city, Scheme.city == None))
        
        if user.income is not None:
            query = query.filter(or_(Scheme.income >= user.income, Scheme.income == None))
        
        if user.differently_abled is not None:
            query = query.filter(or_(Scheme.differently_abled == user.differently_abled, Scheme.differently_abled == None))
        
        if user.minority is not None:
            query = query.filter(or_(Scheme.minority == user.minority, Scheme.minority == None))
        
        if user.bpl_category is not None:
            query = query.filter(or_(Scheme.bpl_category == user.bpl_category, Scheme.bpl_category == None))
        
        # Order by relevance (you can implement more sophisticated ordering)
        query = query.order_by(Scheme.scheme_name)
        
        # Paginate results
        pagination_result = paginate_results(query, page, per_page)
        schemes = pagination_result['items']
        
        # Format response
        result = []
        for scheme in schemes:
            # Calculate match score and similarity
            match_score = calculate_match_score(user, scheme)
            
            result.append({
                "id": scheme.id,
                "scheme_name": scheme.scheme_name,
                "category": scheme.category,
                "description": scheme.description,
                "match_score": match_score,
                "benefit_type": scheme.benefit_type,
                "department": scheme.department
            })
        
        # Sort result by match score
        result.sort(key=lambda x: {"High": 3, "Medium": 2, "Low": 1}[x['match_score']], reverse=True)
        
        # Return paginated response
        return jsonify({
            "recommendations": result,
            "pagination": pagination_result['pagination']
        }), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@api.route('/schemes/search/enhanced', methods=['GET'])
def search_schemes_enhanced():
    try:
        query_text = request.args.get('q', '').strip()
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        # Manual filters
        manual_category_raw = request.args.get('category')
        residence_type = request.args.get('residence_type')
        gender = request.args.get('gender')
        city = request.args.get('city')
        income = request.args.get('income', type=float)
        age = request.args.get('age', type=int)
        differently_abled = request.args.get('differently_abled')
        if differently_abled is not None:
            differently_abled = differently_abled.lower() == 'true'
        minority = request.args.get('minority')
        if minority is not None:
            minority = minority.lower() == 'true'
        bpl_category = request.args.get('bpl_category')
        if bpl_category is not None:
            bpl_category = bpl_category.lower() == 'true'

        # Categories + NLP synonym mapping
        categories = ['Health', 'Insurance', 'Employment', 'Agriculture', 'Housing',
                      'Financial Assistance', 'Safety', 'Subsidy', 'Education', 'Pension',
                      'Business', 'Loan']

        synonyms = {
            "medical": "Health", "hospital": "Health", "doctor": "Health",
            "crop": "Agriculture", "farmer": "Agriculture", "farming": "Agriculture",
            "home": "Housing", "residence": "Housing", "rent": "Housing",
            "financial": "Financial Assistance", "money": "Financial Assistance",
            "loan": "Loan",
            "student": "Education", "college": "Education", "school": "Education", "teacher": "Education",
            "pensioner": "Pension", "retired": "Pension",
            "business": "Business", "startup": "Business",
            "insurance": "Insurance", "premium": "Insurance",
            "job": "Employment", "employment": "Employment", "unemployed": "Employment",
            "protection": "Safety", "violence": "Safety",
            "assistance": "Financial Assistance", "grant": "Subsidy", "subsidy": "Subsidy"
        }

        matched_categories = detect_categories_from_query(query_text, categories, synonyms) if query_text else []
        filters_from_query = detect_filters_from_query(query_text) if query_text else {}

        # Combine NLP + manual filters
        manual_categories = [cat.strip() for cat in manual_category_raw.split(',')] if manual_category_raw else []
        combined_categories = list(set(matched_categories + manual_categories))

        # NLP fallback values if manual not provided
        residence_type = residence_type or filters_from_query.get('residence_type')
        gender = gender or filters_from_query.get('gender')
        city = city or filters_from_query.get('city')
        income = income if income is not None else filters_from_query.get('income')
        age = age if age is not None else filters_from_query.get('age')

        scheme_query = Scheme.query

        # CATEGORY filter
        if combined_categories:
            scheme_query = scheme_query.filter(
                or_(*[Scheme.category.ilike(f"%{cat}%") for cat in combined_categories])
            )

        # Filter with NULL fallback
        if gender:
            scheme_query = scheme_query.filter(or_(Scheme.gender == gender, Scheme.gender == None))
        if residence_type:
            scheme_query = scheme_query.filter(or_(Scheme.residence_type == residence_type, Scheme.residence_type == None))
        if city:
            scheme_query = scheme_query.filter(or_(Scheme.city == city, Scheme.city == None))
        if income is not None:
            scheme_query = scheme_query.filter(or_(Scheme.income <= income, Scheme.income == None))
        if differently_abled is not None:
            scheme_query = scheme_query.filter(or_(Scheme.differently_abled == differently_abled, Scheme.differently_abled == None))
        if minority is not None:
            scheme_query = scheme_query.filter(or_(Scheme.minority == minority, Scheme.minority == None))
        if bpl_category is not None:
            scheme_query = scheme_query.filter(or_(Scheme.bpl_category == bpl_category, Scheme.bpl_category == None))

        # Age filter post-query
        if age is not None:
            filtered = scheme_query.all()
            matching_ids = [s.id for s in filtered if scheme_matches_age(age, s.age_range)]
            scheme_query = Scheme.query.filter(Scheme.id.in_(matching_ids))

        # Paginate
        scheme_query = scheme_query.order_by(Scheme.scheme_name)
        pagination = scheme_query.paginate(page=page, per_page=per_page, error_out=False)
        schemes = pagination.items

        # NLP similarity if query is provided
        if query_text:
            results = search_schemes_nlp(query_text, schemes)
        else:
            results = []

        # Fallback if NLP match fails
        if not results:
            for scheme in schemes:
                total_ratings = SchemeRating.query.filter_by(scheme_id=scheme.id).count()
                results.append({
                    "id": scheme.id,
                    "scheme_name": scheme.scheme_name,
                    "category": scheme.category,
                    "description": scheme.description,
                    "similarity": 0.0,
                    "keywords": extract_keywords(scheme.description, 3),
                    "average_rating": round(scheme.average_rating or 0.0, 2),
                    "total_ratings": total_ratings,
                })

        return jsonify({
            "query": query_text,
            "detected_categories": matched_categories,
            "detected_filters": filters_from_query,
            "applied_filters": {
                "category": combined_categories,
                "residence_type": residence_type,
                "gender": gender,
                "city": city,
                "income": income,
                "age": age,
                "differently_abled": differently_abled,
                "minority": minority,
                "bpl_category": bpl_category
            },
            "schemes": results,
            "pagination": {
                "page": pagination.page,
                "per_page": pagination.per_page,
                "total_items": pagination.total,
                "total_pages": pagination.pages,
                "has_next": pagination.has_next,
                "has_prev": pagination.has_prev
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --------------------- Ratings Routes ---------------------

@api.route('/schemes/<int:scheme_id>/rate', methods=['POST'])
def rate_scheme(scheme_id):
    try:
        data = request.json
        user_id = data.get('user_id')
        rating = float(data.get('rating'))

        if not (1.0 <= rating <= 5.0):
            return jsonify({"error": "Rating must be between 1 and 5"}), 400

        # Check if user already rated
        existing = SchemeRating.query.filter_by(user_id=user_id, scheme_id=scheme_id).first()

        if existing:
            existing.rating = rating
        else:
            new_rating = SchemeRating(user_id=user_id, scheme_id=scheme_id, rating=rating)
            db.session.add(new_rating)

        db.session.commit()

        # Recalculate average
        avg = db.session.query(func.avg(SchemeRating.rating))\
                        .filter_by(scheme_id=scheme_id).scalar() or 0.0

        scheme = Scheme.query.get(scheme_id)
        scheme.average_rating = round(avg, 2)
        db.session.commit()

        return jsonify({"message": "Rating submitted", "new_average": scheme.average_rating}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500