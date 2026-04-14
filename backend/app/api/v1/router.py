from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List, Any

router = APIRouter()

# --- Schemas ---

class AnalysisRequest(BaseModel):
    latitude: float
    longitude: float
    use_case: str

class AnalysisResponse(BaseModel):
    composite_score: float
    shap_breakdown: Dict[str, float]
    bounding_box: List[List[float]]

# --- Routes ---

@router.post("/analyze-site", response_model=AnalysisResponse)
async def analyze_site(request: AnalysisRequest):
    """
    Mocked endpoint returning site readiness analysis for a given coordinate.
    """
    # Mocking the response with a 85 score as requested
    mock_response = AnalysisResponse(
        composite_score=85.0,
        shap_breakdown={
            "distance_to_roads": 15.0,
            "nearby_competitors": -5.0,
            "population_density": 20.0
        },
        bounding_box=[
            [request.longitude - 0.01, request.latitude - 0.01],
            [request.longitude + 0.01, request.latitude + 0.01]
        ]
    )
    
    return mock_response
