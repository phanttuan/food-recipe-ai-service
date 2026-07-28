from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, Any, Dict, List
from services.llm_service import extract_recipe, batch_extract_recipes

router = APIRouter(prefix="/extract", tags=["Extract"])

class ExtractRequest(BaseModel):
    text: Optional[str] = ""
    audioUrl: Optional[str] = ""
    schema: Optional[Dict[str, Any]] = None

class ExtractResponse(BaseModel):
    data: Dict[str, Any]

class VideoItem(BaseModel):
    title: str = ""
    description: str = ""
    transcript: str = ""

class BatchExtractRequest(BaseModel):
    videos: List[VideoItem]

class BatchExtractResponse(BaseModel):
    recipes: List[Dict[str, Any]]

@router.post("", response_model=ExtractResponse)
def extract_recipe_endpoint(request: ExtractRequest):
    content = request.text or f"Audio URL: {request.audioUrl}"
    result = extract_recipe(text_content=content, schema=request.schema)
    return ExtractResponse(data=result.get("data", {}))

@router.post("/batch", response_model=BatchExtractResponse)
def batch_extract_endpoint(request: BatchExtractRequest):
    """Extract recipes from multiple videos in a SINGLE LLM call."""
    videos = [v.model_dump() for v in request.videos]
    recipes = batch_extract_recipes(videos)
    return BatchExtractResponse(recipes=recipes)
