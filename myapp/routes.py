from flask import Blueprint, request, jsonify
from datetime import datetime
from sqlalchemy import or_
import math
import os
import json
import base64
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .extensions import db
from .models import User, Scheme, UserBookmark
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

def calculate_text_similarity(text1, text2):
    """Calculate cosine similarity between two text strings"""
    if not text1 or not text2:
        return 0.0
    
    vec1 = get_text_embedding(text1)
    vec2 = get_text_embedding(text2)
    
    if vec1 is None or vec2 is None:
        return 0.0
    
    # Reshape vectors for cosine_similarity function
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(vec1, vec2)[0][0]
    return float(similarity)

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

# Enhanced recommendation function that could replace or supplement the existing one
def get_enhanced_recommendations(user, schemes, max_results=10):
    """Get NLP + rule-based scheme recommendations"""
    results = []

    # Create user profile description
    user_profile = f"{user.gender or ''} {user.age or ''} {user.occupation or ''} {user.city or ''} " \
                   f"{user.residence_type or ''} {user.education_level or ''} {user.category or ''}"
    user_embedding = get_text_embedding(user_profile)

    if user_embedding is None:
        return []

    for scheme in schemes:
        scheme_desc = f"{scheme.scheme_name or ''} {scheme.description or ''} {scheme.keywords or ''} " \
                      f"{scheme.benefit_type or ''} {scheme.occupation or ''} {scheme.gender or ''}"
        scheme_embedding = get_text_embedding(scheme_desc)

        if scheme_embedding is None:
            continue

        semantic_score = cosine_similarity(
            user_embedding.reshape(1, -1),
            scheme_embedding.reshape(1, -1)
        )[0][0]

        rule_score = calculate_match_score(user, scheme)
        rule_score_numeric = {"High": 1.0, "Medium": 0.6, "Low": 0.3}[rule_score]

        combined_score = 0.7 * rule_score_numeric + 0.3 * semantic_score

        results.append({
            "id": scheme.id,
            "scheme_name": scheme.scheme_name,
            "category": scheme.category,
            "description": scheme.description,
            "semantic_score": float(semantic_score),
            "rule_score": rule_score,
            "combined_score": float(combined_score),
            "keywords": extract_keywords(scheme.description, 3)
        })

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results[:max_results]

    """Get scheme recommendations with NLP-enhanced matching"""
    results = []
    
    # Create a user profile description
    user_profile = f"{user.gender or ''} {user.age or ''} {user.occupation or ''} {user.city or ''} " \
                   f"{user.residence_type or ''} {user.education_level or ''} {user.category or ''}"
    user_embedding = get_text_embedding(user_profile)
    
    for scheme in schemes:
        # Create scheme description for matching
        scheme_desc = f"{scheme.scheme_name or ''} {scheme.description or ''} {scheme.keywords or ''} " \
                     f"{scheme.benefit_type or ''} {scheme.occupation or ''} {scheme.gender or ''}"
        
        # Calculate text similarity
        scheme_embedding = get_text_embedding(scheme_desc)
        
        if user_embedding is not None and scheme_embedding is not None:
            # Calculate semantic similarity between user profile and scheme
            semantic_score = cosine_similarity(
                user_embedding.reshape(1, -1), 
                scheme_embedding.reshape(1, -1)
            )[0][0]
            
            # Combine with rule-based matching from original function
            rule_score = calculate_match_score(user, scheme)
            rule_score_numeric = {"High": 1.0, "Medium": 0.6, "Low": 0.3}[rule_score]
            
            # Combine scores (you can adjust the weights)
            combined_score = 0.7 * rule_score_numeric + 0.3 * semantic_score
            
            results.append({
                "id": scheme.id,
                "scheme_name": scheme.scheme_name,
                "category": scheme.category,
                "description": scheme.description,
                "semantic_score": float(semantic_score),
                "rule_score": rule_score,
                "combined_score": float(combined_score),
                "keywords": extract_keywords(scheme.description, 3)
            })
    
    # Sort by combined score
    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results[:max_results]

# Enhanced search function with NLP
def search_schemes_nlp(query_text, schemes, max_results=20):
    """Search schemes using NLP techniques"""
    if not query_text:
        return []
    
    results = []
    query_embedding = get_text_embedding(query_text)
    
    if query_embedding is None:
        return []
    
    for scheme in schemes:
        # Create scheme text for searching
        scheme_text = f"{scheme.scheme_name or ''} {scheme.description or ''} {scheme.keywords or ''}"
        scheme_embedding = get_text_embedding(scheme_text)
        
        if scheme_embedding is not None:
            # Calculate similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                scheme_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity > 0.3:  # Only include somewhat relevant results
                results.append({
                    "id": scheme.id,
                    "scheme_name": scheme.scheme_name,
                    "category": scheme.category,
                    "description": scheme.description,
                    "similarity": float(similarity),
                    "keywords": extract_keywords(scheme.description, 3)
                })
    
    # Sort by similarity score
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:max_results]

def detect_categories_from_query(query_text, category_list, synonym_map):
    """Detect all relevant categories from a query"""
    doc = nlp(query_text.lower())
    matched_categories = set()

    # Keyword-based detection using synonym mapping
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in synonym_map:
            matched_categories.add(synonym_map[lemma])

    # SpaCy similarity-based fallback
    for category in category_list:
        cat_doc = nlp(category.lower())
        similarity = doc.similarity(cat_doc)
        if similarity >= 0.60:
            matched_categories.add(category)

    return list(matched_categories)

def detect_filters_from_query(query_text):
    """Extract gender, residence_type, and city from query"""
    doc = nlp(query_text.lower())

    # Defined options
    residence_options = {'rural', 'urban', 'semi-urban'}
    gender_options = {'male', 'female', 'other'}

    # Detect filters
    filters = {
        'residence_type': None,
        'gender': None,
        'city': None
    }

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in residence_options:
            filters['residence_type'] = lemma
        elif lemma in gender_options:
            filters['gender'] = lemma

    # Try to find a city match from known cities in DB
    known_cities = {row.city.lower() for row in db.session.query(Scheme.city).distinct() if row.city}
    for token in doc:
        if token.text.lower() in known_cities:
            filters['city'] = token.text
            break

    return filters

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
            pagination_params = pagination_schema.load(request.args)
        except ValidationError as err:
            return handle_validation_error(err)
        
        page = pagination_params['page']
        per_page = pagination_params['per_page']
        
        firebase_id = request.args.get('firebase_id')
        user = User.query.filter_by(firebase_id=firebase_id).first()
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
            result.append({
                "bookmark_id": bookmark.id,
                "scheme_id": scheme.id,
                "scheme_name": scheme.scheme_name,
                "category": scheme.category,
                "description": scheme.description,
                "bookmarked_at": bookmark.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "notes": bookmark.notes
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
            bookmark_data = schema.load(data)
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
            "keywords": scheme.keywords
        }
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@api.route('/schemes/search', methods=['GET'])
def search_schemes():
    try:
        # Parse and validate pagination parameters
        pagination_schema = PaginationParamsSchema()
        try:
            pagination_params = pagination_schema.load(request.args)
        except ValidationError as err:
            return handle_validation_error(err)
        
        page = pagination_params['page']
        per_page = pagination_params['per_page']
        
        # Get search query
        query_text = request.args.get('q', '')
        if not query_text or len(query_text.strip()) == 0:
            return jsonify({"error": "Search query cannot be empty"}), 400
        
        # Simple search implementation
        query = Scheme.query.filter(
            or_(
                Scheme.scheme_name.ilike(f'%{query_text}%'),
                Scheme.description.ilike(f'%{query_text}%'),
                Scheme.keywords.ilike(f'%{query_text}%'),
                Scheme.benefit_type.ilike(f'%{query_text}%'),
                Scheme.department.ilike(f'%{query_text}%')
            )
        ).order_by(Scheme.scheme_name)
        
        # Paginate results
        pagination_result = paginate_results(query, page, per_page)
        schemes = pagination_result['items']
        
        # Format response
        result = []
        for scheme in schemes:
            result.append({
                "id": scheme.id,
                "scheme_name": scheme.scheme_name,
                "category": scheme.category,
                "description": scheme.description,
                "benefit_type": scheme.benefit_type,
                "application_link": scheme.application_link,
                "department": scheme.department
            })
        
        # Return paginated response
        return jsonify({
            "query": query_text,
            "schemes": result,
            "pagination": pagination_result['pagination']
        }), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@api.route('/schemes/unified', methods=['GET'])
def unified_schemes():
    try:
        # Parse and validate pagination parameters
        pagination_schema = PaginationParamsSchema()
        try:
            pagination_params = pagination_schema.load(request.args)
        except ValidationError as err:
            return handle_validation_error(err)
        
        page = pagination_params['page']
        per_page = pagination_params['per_page']
        
        # Get search query (optional)
        query_text = request.args.get('q', '').strip()
        
        # Get filter parameters (all optional)
        category = request.args.get('category')
        gender = request.args.get('gender')
        residence_type = request.args.get('residence_type')
        city = request.args.get('city')
        income = request.args.get('income', type=float)  # Directly use income as parameter
        
        # Convert string boolean parameters to actual booleans
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
        
        # Apply search if query text is provided
        if query_text:
            query = query.filter(
                or_(
                    Scheme.scheme_name.ilike(f'%{query_text}%'),
                    Scheme.description.ilike(f'%{query_text}%'),
                    Scheme.keywords.ilike(f'%{query_text}%'),
                    Scheme.benefit_type.ilike(f'%{query_text}%'),
                    Scheme.department.ilike(f'%{query_text}%')
                )
            )
        
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
            })
        
        # Return paginated response with search query in response if provided
        response_data = {
            "schemes": result,
            "pagination": pagination_result['pagination']
        }
        
        if query_text:
            response_data["query"] = query_text
            
        return jsonify(response_data), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --------------------- Recommendation Routes ---------------------

@api.route('/recommendations', methods=['GET'])
def get_recommendations():
    try:
        # Parse and validate pagination parameters
        pagination_schema = PaginationParamsSchema()
        try:
            pagination_params = pagination_schema.load(request.args)
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

def calculate_match_score(user, scheme):
    """Calculate how well a scheme matches a user profile"""
    score = 0
    total_criteria = 0
    
    # Match gender
    if scheme.gender is not None:
        total_criteria += 1
        if user.gender == scheme.gender:
            score += 1
    
    # Match residence type
    if scheme.residence_type is not None:
        total_criteria += 1
        if user.residence_type == scheme.residence_type:
            score += 1
    
    # Match city
    if scheme.city is not None:
        total_criteria += 1
        if user.city == scheme.city:
            score += 1
    
    # Match income
    if scheme.income is not None and user.income is not None:
        total_criteria += 1
        if user.income <= scheme.income:
            score += 1
    
    # Match differently_abled status
    if scheme.differently_abled is not None:
        total_criteria += 1
        if user.differently_abled == scheme.differently_abled:
            score += 1
    
    # Match minority status
    if scheme.minority is not None:
        total_criteria += 1
        if user.minority == scheme.minority:
            score += 1
    
    # Match BPL category
    if scheme.bpl_category is not None:
        total_criteria += 1
        if user.bpl_category == scheme.bpl_category:
            score += 1
    
    # Match occupation
    if scheme.occupation is not None:
        total_criteria += 1
        if user.occupation == scheme.occupation:
            score += 1
    
    # Match marital status
    if scheme.marital_status is not None:
        total_criteria += 1
        if user.marital_status == scheme.marital_status:
            score += 1
    
    # Match caste/category
    if scheme.caste is not None:
        total_criteria += 1
        if user.category == scheme.caste:
            score += 1
    
    # Calculate age if dob is available and scheme has age criteria
    if user.dob and scheme.age_range:
        total_criteria += 1
        try:
            today = datetime.now().date()
            age = today.year - user.dob.year - ((today.month, today.day) < (user.dob.month, user.dob.day))
            
            age_min, age_max = map(int, scheme.age_range.split('-'))
            if age_min <= age <= age_max:
                score += 1
        except (ValueError, AttributeError):
            # If age range format is incorrect, ignore this criterion
            total_criteria -= 1
    
    # Calculate percentage match
    match_percentage = (score / total_criteria * 100) if total_criteria > 0 else 0
    
    # Return match label based on percentage
    if match_percentage >= 80:
        return "High"
    elif match_percentage >= 50:
        return "Medium"
    else:
        return "Low"

@api.route('/recommendations/enhanced', methods=['GET'])
def get_enhanced_recommendations_route():
    try:
        firebase_id = request.args.get('firebase_id')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        user = User.query.filter_by(firebase_id=firebase_id).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Compute age inline if dob is available
        user.age = None
        if user.dob:
            today = datetime.today().date()
            user.age = today.year - user.dob.year - ((today.month, today.day) < (user.dob.month, user.dob.day))

        # Apply basic filters
        query = Scheme.query
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

        # Use pagination instead of query.all()
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        schemes = pagination.items

        # Generate recommendations only for current page
        recommendations = get_enhanced_recommendations(user, schemes)

        return jsonify({
            "recommendations": recommendations,
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

    try:
        firebase_id = request.args.get('firebase_id')
        user = User.query.filter_by(firebase_id=firebase_id).first()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
            
        # Get all schemes - you might want to limit this in production
        schemes = Scheme.query.all()
        
        # Use the NLP-enhanced recommendation function
        recommendations = get_enhanced_recommendations(user, schemes)
        
        return jsonify({
            "recommendations": recommendations
        }), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@api.route('/schemes/search/enhanced', methods=['GET'])
def search_schemes_enhanced():
    try:
        query_text = request.args.get('q', '').strip()
        if not query_text:
            return jsonify({"error": "Search query cannot be empty"}), 400

        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        # Categories and synonyms
        categories = ['Education', 'Scholarship', 'Hospital', 'Agriculture', 'Insurance', 'Housing', 'Fund support']
        synonyms = {
            "educational": "Education",
            "college": "Education",
            "student": "Scholarship",
            "financial": "Fund support",
            "loan": "Fund support",
            "money": "Fund support",
            "medical": "Hospital",
            "health": "Hospital",
            "farmer": "Agriculture",
            "crop": "Agriculture",
            "farming": "Agriculture",
            "insurance": "Insurance",
            "home": "Housing",
            "residence": "Housing"
        }

        # NLP-based category detection
        matched_categories = detect_categories_from_query(query_text, categories, synonyms)

        filters = detect_filters_from_query(query_text)
        residence_type = filters['residence_type']
        gender = filters['gender']
        city = filters['city']

        # Filter schemes based on matched categories
        if matched_categories:
            scheme_query = Scheme.query.filter(Scheme.category.in_(matched_categories)).order_by(Scheme.scheme_name)
        else:
            scheme_query = Scheme.query.order_by(Scheme.scheme_name)
        
        if residence_type:
            scheme_query = scheme_query.filter(or_(Scheme.residence_type == residence_type, Scheme.residence_type == None))
        if gender:
            scheme_query = scheme_query.filter(or_(Scheme.gender == gender, Scheme.gender == None))
        if city:
            scheme_query = scheme_query.filter(or_(Scheme.city == city, Scheme.city == None))


        pagination = scheme_query.paginate(page=page, per_page=per_page, error_out=False)
        schemes = pagination.items

        # NLP similarity ranking (on page results)
        results = search_schemes_nlp(query_text, schemes)

        return jsonify({
            "query": query_text,
            "detected_categories": matched_categories,
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