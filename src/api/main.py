"""
FastAPI - Maritime Route Optimizer API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

from src.models.optimizer import MaritimeRouteOptimizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Maritime Route Optimizer",
    description="Optimize maritime routes using GNN + A* pathfinding on real AIS data.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load optimizer once at startup ────────────────────────────────────────────

optimizer: Optional[MaritimeRouteOptimizer] = None

@app.on_event("startup")
async def startup():
    global optimizer
    log.info("Loading Maritime Route Optimizer...")
    optimizer = MaritimeRouteOptimizer()
    log.info("✅ API ready!")


# ── Schemas ───────────────────────────────────────────────────────────────────

class RouteRequest(BaseModel):
    origin: str
    destination: str

    class Config:
        json_schema_extra = {
            "example": {
                "origin": "Los Angeles",
                "destination": "Long Beach"
            }
        }

class RouteResponse(BaseModel):
    found: bool
    origin: str
    destination: str
    path: list[str]
    coordinates: list[tuple[float, float]]
    total_distance_km: float
    total_cost: float
    n_hops: int
    message: str


class PortInfo(BaseModel):
    name: str
    country: str
    lat: float
    lon: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check API status."""
    return {
        "status": "ok",
        "model": "GNN + A*",
        "ports_loaded": len(optimizer.nodes_df) if optimizer else 0,
    }


@app.get("/ports", response_model=list[PortInfo])
def list_ports(country: Optional[str] = None):
    """
    List all available ports.
    Optionally filter by country name (partial match).
    """
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not loaded")

    ports = optimizer.list_ports()

    if country:
        ports = ports[
            ports["country"].str.lower().str.contains(country.lower(), na=False)
        ]

    return ports.to_dict(orient="records")


@app.post("/optimize", response_model=RouteResponse)
def optimize_route(request: RouteRequest):
    """
    Find the optimal maritime route between two ports.

    Uses GNN-predicted edge costs with A* pathfinding.
    """
    if optimizer is None:
        raise HTTPException(status_code=503, detail="Optimizer not loaded")

    if not request.origin.strip() or not request.destination.strip():
        raise HTTPException(status_code=400, detail="Origin and destination required")

    result = optimizer.optimize(request.origin, request.destination)

    if not result.found:
        return RouteResponse(
            found=False,
            origin=request.origin,
            destination=request.destination,
            path=[],
            coordinates=[],
            total_distance_km=0.0,
            total_cost=0.0,
            n_hops=0,
            message=f"No route found between '{request.origin}' and '{request.destination}'. Check /ports for available ports."
        )

    return RouteResponse(
        found=True,
        origin=result.origin,
        destination=result.destination,
        path=result.path_ports,
        coordinates=result.path_coords,
        total_distance_km=result.total_distance_km,
        total_cost=result.total_cost,
        n_hops=result.n_hops,
        message=f"Route found: {len(result.path_ports)} ports, {result.total_distance_km} km"
    )