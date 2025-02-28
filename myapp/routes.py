from flask import Blueprint, request, jsonify, redirect, url_for
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
from sqlalchemy import or_, and_

from .extensions import db
from .models import User, Scheme, UserBookmark

# Create Blueprint
api = Blueprint('api', __name__)

# --------------------- User Routes ---------------------

@api.route('/user', methods=['POST'])
def create_user():
    data = request.get_json()
    
    # Check if user with this firebase_id already exists
    existing_user = User.query.filter_by(firebase_id=data.get('firebase_id')).first()
    if existing_user:
        return jsonify({"error": "User with this firebase_id already exists"}), 400
    
    # Create new user
    new_user = User(
        firebase_id=data.get('firebase_id'),
        name=data.get('name'),
        dob=datetime.strptime(data.get('dob'), '%Y-%m-%d').date() if data.get('dob') else None,
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
    db.session.commit()
    
    return jsonify({"message": "User created successfully", "user_id": new_user.id}), 201

@api.route('/user/<int:user_id>', methods=['GET'])
@jwt_required()
def get_user(user_id):
    # Verify user can only access their own profile
    current_user_id = get_jwt_identity()
    if current_user_id != user_id:
        return jsonify({"error": "Unauthorized access"}), 403
        
    user = User.query.get_or_404(user_id)
    
    # Convert user object to dictionary
    user_data = {
        "id": user.id,
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
        "preferred_language": user.preferred_language
    }
    
    return jsonify(user_data), 200

@api.route('/user/<int:user_id>', methods=['PUT'])
@jwt_required()
def update_user(user_id):
    # Verify user can only update their own profile
    current_user_id = get_jwt_identity()
    if current_user_id != user_id:
        return jsonify({"error": "Unauthorized access"}), 403
        
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    # Update user fields if provided in request
    if 'name' in data:
        user.name = data['name']
    if 'dob' in data:
        user.dob = datetime.strptime(data['dob'], '%Y-%m-%d').date()
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
    db.session.commit()
    
    return jsonify({"message": "User updated successfully"}), 200

@api.route('/user/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    # Verify user can only delete their own profile
    current_user_id = get_jwt_identity()
    if current_user_id != user_id:
        return jsonify({"error": "Unauthorized access"}), 403
        
    user = User.query.get_or_404(user_id)
    
    # Delete related records first (bookmarks)
    UserBookmark.query.filter_by(user_id=user_id).delete()
    
    db.session.delete(user)
    db.session.commit()
    
    return jsonify({"message": "User deleted successfully"}), 200

# --------------------- Scheme Routes ---------------------

@api.route('/schemes', methods=['GET'])
def get_schemes():
    # Basic filtering parameters
    category = request.args.get('category')
    gender = request.args.get('gender')
    residence_type = request.args.get('residence_type')
    city = request.args.get('city')
    income_max = request.args.get('income_max', type=float)
    differently_abled = request.args.get('differently_abled', type=bool)
    minority = request.args.get('minority', type=bool)
    bpl_category = request.args.get('bpl_category', type=bool)
    
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
    if income_max:
        query = query.filter(or_(Scheme.income >= income_max, Scheme.income == None))
    if differently_abled is not None:
        query = query.filter(or_(Scheme.differently_abled == differently_abled, Scheme.differently_abled == None))
    if minority is not None:
        query = query.filter(or_(Scheme.minority == minority, Scheme.minority == None))
    if bpl_category is not None:
        query = query.filter(or_(Scheme.bpl_category == bpl_category, Scheme.bpl_category == None))
    
    # Execute query
    schemes = query.all()
    
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
            # Add other fields as needed
        })
    
    return jsonify(result), 200

@api.route('/schemes/<int:scheme_id>', methods=['GET'])
def get_scheme(scheme_id):
    scheme = Scheme.query.get_or_404(scheme_id)
    
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
        "education_criteria": scheme.education_criteria
    }
    
    return jsonify(result), 200

@api.route('/schemes/search', methods=['GET'])
def search_schemes():
    # Get search query
    query_text = request.args.get('q', '')
    
    # Simple search implementation (can be enhanced with NLP)
    schemes = Scheme.query.filter(
        or_(
            Scheme.scheme_name.ilike(f'%{query_text}%'),
            Scheme.description.ilike(f'%{query_text}%'),
            Scheme.keywords.ilike(f'%{query_text}%'),
            Scheme.benefit_type.ilike(f'%{query_text}%'),
            Scheme.department.ilike(f'%{query_text}%')
        )
    ).all()
    
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
            # Add other fields as needed
        })
    
    return jsonify(result), 200

# --------------------- Bookmark Routes ---------------------

@api.route('/bookmarks', methods=['GET'])
@jwt_required()
def get_bookmarks():
    user_id = get_jwt_identity()
    
    bookmarks = UserBookmark.query.filter_by(user_id=user_id).all()
    
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
    
    return jsonify(result), 200

@api.route('/bookmarks', methods=['POST'])
@jwt_required()
def create_bookmark():
    user_id = get_jwt_identity()
    data = request.get_json()
    
    scheme_id = data.get('scheme_id')
    notes = data.get('notes')
    
    # Check if scheme exists
    scheme = Scheme.query.get_or_404(scheme_id)
    
    # Check if bookmark already exists
    existing_bookmark = UserBookmark.query.filter_by(user_id=user_id, scheme_id=scheme_id).first()
    if existing_bookmark:
        return jsonify({"error": "Bookmark already exists"}), 400
    
    # Create bookmark
    bookmark = UserBookmark(
        user_id=user_id,
        scheme_id=scheme_id,
        notes=notes
    )
    
    db.session.add(bookmark)
    db.session.commit()
    
    return jsonify({
        "message": "Bookmark created successfully", 
        "bookmark_id": bookmark.id
    }), 201

@api.route('/bookmarks/<int:bookmark_id>', methods=['DELETE'])
@jwt_required()
def delete_bookmark(bookmark_id):
    user_id = get_jwt_identity()
    
    # Get bookmark and verify ownership
    bookmark = UserBookmark.query.get_or_404(bookmark_id)
    if bookmark.user_id != user_id:
        return jsonify({"error": "Unauthorized access"}), 403
    
    db.session.delete(bookmark)
    db.session.commit()
    
    return jsonify({"message": "Bookmark deleted successfully"}), 200

# --------------------- Recommendation Routes ---------------------

@api.route('/recommendations', methods=['GET'])
@jwt_required()
def get_recommendations():
    user_id = get_jwt_identity()
    user = User.query.get_or_404(user_id)
    
    # Basic recommendation logic based on user profile
    # This can be enhanced with more sophisticated algorithms
    query = Scheme.query
    
    # Filter by basic matching criteria
    if user.gender:
        query = query.filter(or_(Scheme.gender == user.gender, Scheme.gender == None))
    if user.residence_type:
        query = query.filter(or_(Scheme.residence_type == user.residence_type, Scheme.residence_type == None))
    if user.city:
        query = query.filter(or_(Scheme.city == user.city, Scheme.city == None))
    if user.income:
        query = query.filter(or_(Scheme.income >= user.income, Scheme.income == None))
    if user.differently_abled:
        query = query.filter(or_(Scheme.differently_abled == user.differently_abled, Scheme.differently_abled == None))
    if user.minority:
        query = query.filter(or_(Scheme.minority == user.minority, Scheme.minority == None))
    if user.bpl_category:
        query = query.filter(or_(Scheme.bpl_category == user.bpl_category, Scheme.bpl_category == None))
    if user.occupation:
        query = query.filter(or_(Scheme.occupation == user.occupation, Scheme.occupation == None))
    if user.marital_status:
        query = query.filter(or_(Scheme.marital_status == user.marital_status, Scheme.marital_status == None))
    if user.category:
        query = query.filter(or_(Scheme.caste == user.category, Scheme.caste == None))
    
    # Calculate age if dob is available
    if user.dob:
        from datetime import date
        today = date.today()
        age = today.year - user.dob.year - ((today.month, today.day) < (user.dob.month, user.dob.day))
        
        # Filter schemes by age range
        schemes = query.all()
        result = []
        
        for scheme in schemes:
            # Handle age range filtering manually
            if scheme.age_range:
                try:
                    age_min, age_max = map(int, scheme.age_range.split('-'))
                    if age < age_min or age > age_max:
                        continue
                except (ValueError, AttributeError):
                    # If age range format is incorrect, include the scheme anyway
                    pass
            
            result.append({
                "id": scheme.id,
                "scheme_name": scheme.scheme_name,
                "category": scheme.category,
                "description": scheme.description,
                "benefit_type": scheme.benefit_type,
                "application_link": scheme.application_link,
                "match_score": "High"  # This could be calculated based on how many criteria match
            })
    else:
        # If no DOB provided, just return the filtered schemes
        schemes = query.all()
        result = [{
            "id": scheme.id,
            "scheme_name": scheme.scheme_name,
            "category": scheme.category,
            "description": scheme.description,
            "benefit_type": scheme.benefit_type,
            "application_link": scheme.application_link,
            "match_score": "Medium"
        } for scheme in schemes]
    
    return jsonify(result), 200

# --------------------- NLP Query Route ---------------------

@api.route('/nlp-query', methods=['POST'])
def nlp_query():
    data = request.get_json()
    query_text = data.get('query')
    
    # Extract keywords from the query
    # In a real implementation, this would use more sophisticated NLP techniques
    keywords = extract_keywords_from_query(query_text)
    
    # Search for schemes based on extracted keywords
    schemes = []
    for keyword in keywords:
        keyword_schemes = Scheme.query.filter(
            or_(
                Scheme.scheme_name.ilike(f'%{keyword}%'),
                Scheme.description.ilike(f'%{keyword}%'),
                Scheme.keywords.ilike(f'%{keyword}%'),
                Scheme.benefit_type.ilike(f'%{keyword}%'),
                Scheme.department.ilike(f'%{keyword}%'),
                Scheme.occupation.ilike(f'%{keyword}%')
            )
        ).all()
        schemes.extend(keyword_schemes)
    
    # Remove duplicates
    unique_schemes = list({scheme.id: scheme for scheme in schemes}.values())
    
    # Format response
    result = []
    for scheme in unique_schemes:
        result.append({
            "id": scheme.id,
            "scheme_name": scheme.scheme_name,
            "category": scheme.category,
            "description": scheme.description,
            "benefit_type": scheme.benefit_type,
            "application_link": scheme.application_link
        })
    
    return jsonify({
        "query": query_text,
        "extracted_keywords": keywords,
        "results": result
    }), 200

# Helper function for simple keyword extraction
def extract_keywords_from_query(query):
    # Simple implementation - split by spaces and remove common words
    common_words = {'what', 'which', 'how', 'is', 'are', 'the', 'for', 'in', 'of', 'and', 'to', 'a', 'an'}
    words = query.lower().split()
    keywords = [word for word in words if word not in common_words and len(word) > 2]
    
    # Add specific entity recognition for common categories
    if 'farmer' in query.lower() or 'farming' in query.lower() or 'agriculture' in query.lower():
        keywords.append('agriculture')
    if 'student' in query.lower() or 'education' in query.lower() or 'study' in query.lower():
        keywords.append('education')
    if 'women' in query.lower() or 'woman' in query.lower() or 'female' in query.lower():
        keywords.append('women')
    if 'disable' in query.lower() or 'disability' in query.lower() or 'handicap' in query.lower():
        keywords.append('disability')
    
    return keywords