import os
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from typing import Annotated
# Google Auth libraries
from google.oauth2 import id_token
from google.auth.transport import requests

# get the allowed caller service account
ALLOWED_CALLER_EMAIL = os.getenv("ALLOWED_CALLER_EMAIL")
# Get the current environment (defaults to 'production' if not set)
APP_ENV = os.getenv("APP_ENV", "production")

print(f"INFO:     App environment: {APP_ENV}. You can skip auth when running locally by setting APP_ENV=local")

# ---- Configuration ----
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO:     Using device: {device}")

# ---- Model Loading ----
print("INFO:     Loading BAAI/bge-m3 model...")
model = SentenceTransformer('BAAI/bge-m3', device=device)
print("INFO:     Model loaded successfully.")

# ---- FastAPI App ----
app = FastAPI()

# ---- Authentication Dependency ----
async def validate_token(authorization: Annotated[str, Header()]=None):
    """
    A FastAPI dependency that validates the GCP ID Token in the Authorization header.
    """

    # Bypass auth for local development
    if APP_ENV == "local":
        print("INFO:     Skipping token validation for local environment.")
        return
    
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Missing Bearer token")
    
    token = authorization.split("Bearer ")[1]
    
    try:
        # Validate the token: checks signature, expiration, and audience
        id_info = id_token.verify_oauth2_token(
            token, 
            requests.Request()
            # Add an 'audience' parameter here for an extra layer of security
            # audience="http://your-gce-internal-ip:8000/embed" 
        )

        # Check that the token was issued to the allowed service account
        if id_info.get("email") != ALLOWED_CALLER_EMAIL:
            raise HTTPException(status_code=403, detail="Forbidden: Caller not permitted")

    except ValueError as e:
        # This catches invalid tokens
        raise HTTPException(status_code=401, detail=f"Unauthorized: Invalid token ({e})")

# ---- Pydantic model for request body validation ----
# This class is now defined BEFORE it is used below.
class TextInput(BaseModel):
    text: str

# ---- API Endpoints ----
@app.get("/")
def read_root():
    return {"status": "Embedding service is online"}

@app.post("/embed", dependencies=[Depends(validate_token)])
def create_embedding(item: TextInput):
    embedding = model.encode(item.text, normalize_embeddings=True)

    print(f"Generated embedding for '{item.text}':\n{embedding}\n")

    return {"text": item.text, "embedding": embedding.tolist()}
