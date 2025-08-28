import os
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer
import torch
from typing import Annotated, List
# Google Auth libraries
from google.oauth2 import id_token
from google.auth.transport import requests

# --- Configuration ----
# Service account email of the main API that is allowed to call this service
ALLOWED_CALLER_EMAIL = os.getenv("ALLOWED_CALLER_EMAIL")
APP_ENV = os.getenv("APP_ENV", "production")

print(f"INFO:     App environment: {APP_ENV}. You can skip auth when running locally by setting APP_ENV=local")

# Use GPU if available, otherwise CPU. Cloud Run can be configured with GPUs.
MODEL_NAME = 'Qwen/Qwen3-Embedding-4B'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"INFO:     Using device: {device}")

# --- Model Loading ---
# This happens once when the container starts up.
print(f"INFO:     Loading {MODEL_NAME} model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 
).to(device).eval()

# Apply torch.compile for a significant speed-up during inference
# This is a best-practice for modern PyTorch.
# try:
#     model = torch.compile(model)
#     print("INFO:     Model compiled successfully for faster inference.")
# except Exception as e:
#     print(f"WARNING:  Could not compile model. It will run without compilation. Error: {e}")

print("INFO:     Model loaded successfully.")

# --- FastAPI App ---
app = FastAPI()

# --- Authentication Dependency ---
# This function is unchanged and works perfectly for Cloud Run.
async def validate_token(authorization: Annotated[str, Header()] = None):
    if APP_ENV == "local":
        print("INFO:     Skipping token validation for local environment.")
        return
    
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized: Missing Bearer token")
    
    token = authorization.split("Bearer ")[1]
    
    try:
        id_info = id_token.verify_oauth2_token(token, requests.Request())
        if id_info.get("email") != ALLOWED_CALLER_EMAIL:
            raise HTTPException(status_code=403, detail="Forbidden: Caller not permitted")
    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Unauthorized: Invalid token ({e})")

# --- Pydantic Models ----
class TextInput(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    text: str
    embedding: List[float]

# --- API Endpoints ----
@app.get("/")
async def read_root():
    return {"status": "Qwen embedding service is online"}

@app.post("/embed", response_model=EmbeddingResponse, dependencies=[Depends(validate_token)])
async def create_embedding(item: TextInput):
    """
    Generates an embedding for the input text using the Qwen model.
    The endpoint is now async.
    """
    # Tokenize the input text
    inputs = tokenizer([item.text], return_tensors='pt', padding=True, truncation=True).to(device)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Use the last hidden state's mean pooling as the embedding
        hidden_states = outputs.hidden_states[-1]
        embedding = torch.mean(hidden_states, dim=1).squeeze().cpu().numpy()

    print(f"Generated embedding for text: '{item.text}'")
    return {"text": item.text, "embedding": embedding.tolist()}