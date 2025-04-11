import numpy as np
from myapp.extensions import db
from myapp.models import Scheme
from myapp.routes import get_text_embedding

def update_scheme_embeddings():
    schemes = Scheme.query.all()
    count = 0

    for scheme in schemes:
        if scheme.embedding:  # Skip if already populated
            continue

        desc = f"{scheme.scheme_name or ''} {scheme.description or ''} {scheme.keywords or ''} " \
               f"{scheme.benefit_type or ''} {scheme.occupation or ''} {scheme.gender or ''}"

        embedding = get_text_embedding(desc)
        if embedding is not None:
            scheme.embedding = embedding.astype(np.float32).tobytes()
            count += 1

    db.session.commit()
    print(f"{count} embeddings updated.")
