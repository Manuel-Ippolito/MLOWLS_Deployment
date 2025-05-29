from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set")

# Create API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key