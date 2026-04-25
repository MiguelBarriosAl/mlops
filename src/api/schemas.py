"""
Pydantic schemas for the California Housing prediction API.

Defines the input model (HousingFeatures) and the output model (PredictionResponse)
used by FastAPI to validate requests and serialise responses automatically.
"""

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Feature columns — ordered list that matches train_features_v1.csv (no target)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "rooms_per_household",
    "bedrooms_per_room",
    "log_population",
]

# ---------------------------------------------------------------------------
# Schemas — request body (input) and response body (output)
# ---------------------------------------------------------------------------


class HousingFeatures(BaseModel):
    """Input: the 11 engineered feature values for one house (excluding MedHouseVal)."""

    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    rooms_per_household: float
    bedrooms_per_room: float
    log_population: float


class PredictionResponse(BaseModel):
    """Output: predicted median house value (in $100k) and the model version that answered."""

    prediction: float
    model_version: str
