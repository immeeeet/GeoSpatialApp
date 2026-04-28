"""
Pydantic request/response schemas for the site analysis API.
"""

from typing import Literal, Any
from pydantic import BaseModel, Field

# Supported business types
BusinessType = Literal[
    "restaurant", "gym", "clinic", "pharmacy", "retail_store",
    "ev_charging", "warehouse", "cafe", "school", "bank",
]


# ── Request Models ───────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    """Input for the /analyze endpoint."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lng: float = Field(..., ge=-180, le=180, description="Longitude")
    business_type: BusinessType
    radius_km: float = Field(default=3.0, gt=0, le=50, description="Search radius in km")


class HeatmapRequest(BaseModel):
    """Query params for the /heatmap endpoint."""
    city: str | None = Field(default=None, description="City name (e.g. 'ahmedabad')")
    bbox: str | None = Field(default=None, description="Bounding box: 'lat_min,lng_min,lat_max,lng_max'")
    business_type: BusinessType


# ── Response Models ──────────────────────────────────────────


class FeatureBreakdown(BaseModel):
    """Single feature score + weight + raw description."""
    score: float
    weight: float
    raw_value: str


class DemandGap(BaseModel):
    status: str  # HIGH, MODERATE, LOW
    message: str


class Competitor(BaseModel):
    name: str
    lat: float
    lng: float
    distance_m: int


class CatchmentInfo(BaseModel):
    isochrone_15min: dict | None = None
    population_within_15min: int = 0


class HexScore(BaseModel):
    h3_index: str
    score: float
    center: list[float]


class ClusteredHex(BaseModel):
    """A single hex with DBSCAN cluster info and color for map rendering."""
    h3_index: str
    score: float
    cluster_label: int
    cluster_type: str
    center: list[float]
    color: str


class HeatmapPoint(BaseModel):
    lat: float
    lng: float
    intensity: float


class ClusterInfo(BaseModel):
    """DBSCAN cluster context for the queried point."""
    type: str           # CORE, EDGE, SMALL_CLUSTER, ISOLATED
    size: int           # Number of hexes in the cluster
    avg_score: float    # Average score across cluster hexes
    adjustment: str     # Human-readable reason for score adjustment
    neighboring_clusters: int


class MapData(BaseModel):
    """Enhanced map data with clustered hexes and cluster boundaries."""
    clustered_hexes: list[ClusteredHex] = []
    heatmap_points: list[HeatmapPoint] = []
    cluster_boundaries: dict | None = None  # GeoJSON FeatureCollection


class AnalyzeResponse(BaseModel):
    """Full response from the /analyze endpoint with DBSCAN cluster context."""
    score: int                                  # Cluster-adjusted score
    raw_score: int | None = None                # Original model score before adjustment
    grade: str
    verdict: str
    breakdown: dict[str, FeatureBreakdown]
    demand_gap: DemandGap
    competitors: list[Competitor]
    cluster: ClusterInfo | None = None          # DBSCAN cluster context
    catchment: CatchmentInfo
    map_data: MapData


class SuggestedBusiness(BaseModel):
    """A single business type suggestion."""
    business_type: str
    underserved_score: float
    message: str


class SuggestResponse(BaseModel):
    """Response from the /suggest endpoint."""
    location: dict[str, float]
    radius_km: float
    suggestions: list[SuggestedBusiness]
