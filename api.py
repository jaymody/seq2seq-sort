import os
import secrets

from fastapi import APIRouter, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from train import load_model

API_KEY = os.environ["API_KEY"]
API_KEY_NAME = "X-API-KEY"
api_key_header_auth = APIKeyHeader(name="X-API-Key")


def authenticate(api_key_header: str = Security(api_key_header_auth)):
    if not secrets.compare_digest(api_key_header, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key",
        )


model = load_model("models/run1")
api = APIRouter(prefix="/api", dependencies=[Security(authenticate)])


class PredictRequest(BaseModel):
    input_sequence: str


class PredictResponse(BaseModel):
    output_sequence: str


@api.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    """Model predict route.

    Example:
    -------
    ```python
    import requests

    input_sequence = "bcdaefg"
    res = requests.post(
        "http://localhost:8000/api/predict",
        json={"input_sequence": input_sequence},
        headers={"X-API-Key": "d61c6c40-f372-4cf7-8de8-e8f3706bc83b"},
    )

    assert res.ok
    output_sequence = res.json()["output_sequence"]
    assert output_sequence == "abcdefg"
    ```
    """
    prd_sequences, _, _ = model.predict([data.input_sequence], batch_size=1)
    return PredictResponse(output_sequence=prd_sequences[0])
