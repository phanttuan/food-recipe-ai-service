from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from services.llm_service import rank_candidates

router = APIRouter(prefix="/rank", tags=["Rank"])

class Candidate(BaseModel):
    id: str
    title: str
    description: Optional[str] = ""
    duration: Optional[str] = ""
    hasCaption: Optional[bool] = False
    thumbnailUrl: Optional[str] = ""

class RankRequest(BaseModel):
    query: str = Field(..., description="Comma-separated ingredient list or search query")
    candidates: List[Candidate]
    topK: Optional[int] = 3

class RankResponse(BaseModel):
    selectedIds: List[str]
    reasoning: str

@router.post("", response_model=RankResponse)
def rank_videos(request: RankRequest):
    if not request.candidates:
        return RankResponse(selectedIds=[], reasoning="No candidates provided.")

    candidates_dict = [c.model_dump() for c in request.candidates]
    result = rank_candidates(
        query=request.query,
        candidates=candidates_dict,
        top_k=request.topK or 3
    )
    return RankResponse(
        selectedIds=result.get("selectedIds", []),
        reasoning=result.get("reasoning", "")
    )
