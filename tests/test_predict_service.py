"""
Production FastAPI Prediction Service Test Suite
=================================================
Validates the persistent FastAPI service, lifespan startup lifecycle,
in-memory inference, and HTTP endpoint error boundaries.
"""

import time
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from predict_service import (
    DroughtPredictionService,
    InvalidInputError,
    ModelNotFoundError,
    app,
    lifespan,
    predict_drought,
)


# =============================================================================
# 1. Pure Python Domain / Service Tests
# =============================================================================

class TestPredictionServiceDomain:
    def test_predict_drought_valid_inputs(self):
        res = predict_drought(latitude=9.63, longitude=39.53, year=2005)

        assert isinstance(res, dict)
        assert res["predicted_drought_class"] in {0, 1, 2}
        assert res["severity_label"] in {"Normal", "Moderate Drought", "Severe Drought"}
        assert "confidence_probabilities" in res
        assert "grid_cell" in res
        assert res["year"] == 2005
        assert res["service_mode"] == "retrospective_reconstruction"

        probs = res["confidence_probabilities"]
        assert 0.0 <= probs["class_0"] <= 1.0
        assert 0.0 <= probs["class_1"] <= 1.0
        assert 0.0 <= probs["class_2"] <= 1.0
        assert pytest.approx(sum(probs.values()), abs=1e-3) == 1.0

    def test_predict_drought_future_year(self):
        res = predict_drought(latitude=13.01, longitude=37.80, year=2028)
        assert res["year"] == 2028
        assert res["service_mode"] == "prospective_solar_projection"
        assert res["predicted_drought_class"] in {0, 1, 2}

    def test_invalid_latitude_raises(self):
        with pytest.raises(InvalidInputError):
            predict_drought(latitude=95.0, longitude=39.53, year=2005)
        with pytest.raises(InvalidInputError):
            predict_drought(latitude=-91.0, longitude=39.53, year=2005)

    def test_invalid_longitude_raises(self):
        with pytest.raises(InvalidInputError):
            predict_drought(latitude=9.63, longitude=185.0, year=2005)
        with pytest.raises(InvalidInputError):
            predict_drought(latitude=9.63, longitude=-181.0, year=2005)

    def test_invalid_year_raises(self):
        with pytest.raises(InvalidInputError):
            predict_drought(latitude=9.63, longitude=39.53, year=1600)
        with pytest.raises(InvalidInputError):
            predict_drought(latitude=9.63, longitude=39.53, year=2200)

    def test_singleton_instance_caching(self):
        s1 = DroughtPredictionService.get_instance()
        s2 = DroughtPredictionService.get_instance()
        assert s1 is s2
        assert s1.is_ready


# =============================================================================
# 2. FastAPI HTTP Endpoint Tests
# =============================================================================

class TestFastAPIEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        """Initializes FastAPI client inside lifespan context."""
        with TestClient(app) as test_client:
            yield test_client

    def test_health_liveness(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "service" in data

    def test_readiness_probe_healthy(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True
        assert data["model_loaded"] is True
        assert data["solar_data_loaded"] is True
        assert data["spei_data_loaded"] is True

    def test_get_predict_valid_query(self, client):
        resp = client.get("/predict?latitude=4.88&longitude=38.08&year=2026")
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_drought_class"] in {0, 1, 2}
        assert data["year"] == 2026
        assert data["service_mode"] == "prospective_solar_projection"
        assert data["grid_cell"]["requested_lat"] == 4.88
        assert data["grid_cell"]["requested_lon"] == 38.08
        assert "selected_lat" in data["grid_cell"]
        assert "distance_km" in data["grid_cell"]

    def test_get_predict_missing_parameters(self, client):
        # Missing year
        resp = client.get("/predict?latitude=4.88&longitude=38.08")
        assert resp.status_code == 422

        # Missing latitude and longitude
        resp = client.get("/predict")
        assert resp.status_code == 422

    def test_get_predict_invalid_coordinates(self, client):
        # Latitude out of bounds
        resp = client.get("/predict?latitude=95.0&longitude=38.08&year=2026")
        assert resp.status_code == 422

        # Longitude out of bounds
        resp = client.get("/predict?latitude=4.88&longitude=190.0&year=2026")
        assert resp.status_code == 422

    def test_get_predict_invalid_year(self, client):
        resp = client.get("/predict?latitude=4.88&longitude=38.08&year=1500")
        assert resp.status_code == 422

        resp = client.get("/predict?latitude=4.88&longitude=38.08&year=2500")
        assert resp.status_code == 422

    def test_post_predict_valid_json(self, client):
        payload = {"latitude": 9.63, "longitude": 39.53, "year": 2005}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["predicted_drought_class"] in {0, 1, 2}
        assert data["year"] == 2005
        assert data["service_mode"] == "retrospective_reconstruction"

    def test_post_predict_invalid_payload(self, client):
        # Bad types
        resp = client.post("/predict", json={"latitude": "invalid", "longitude": 38.08, "year": 2026})
        assert resp.status_code == 422

        # Out of bounds
        resp = client.post("/predict", json={"latitude": 100.0, "longitude": 38.08, "year": 2026})
        assert resp.status_code == 422

    def test_cold_start_elimination(self, client):
        """Verifies that consecutive predictions complete rapidly without re-initialization overhead."""
        # Issue first prediction
        t0 = time.time()
        r1 = client.get("/predict?latitude=4.88&longitude=38.08&year=2026")
        dt1 = (time.time() - t0) * 1000
        assert r1.status_code == 200

        # Issue second prediction
        t1 = time.time()
        r2 = client.get("/predict?latitude=9.63&longitude=39.53&year=2005")
        dt2 = (time.time() - t1) * 1000
        assert r2.status_code == 200

        # Both must be well below the old cold-start 16,000ms threshold
        assert dt1 < 2000  # < 2 seconds
        assert dt2 < 2000  # < 2 seconds


# =============================================================================
# 3. Startup Failure & Degraded State Tests
# =============================================================================

class TestStartupFailureBehavior:
    def test_startup_missing_model_fails_gracefully(self):
        """Verifies that an uninitialized/failed app returns 503 on /ready."""
        mock_app = FastAPI()
        mock_app.state.is_ready = False
        mock_app.state.service = None
        mock_app.state.startup_error = "Model file corrupted or missing."

        from predict_service import readiness_check
        mock_app.get("/ready")(readiness_check)

        with TestClient(mock_app) as client:
            resp = client.get("/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["detail"]["ready"] is False
            assert "Model file corrupted" in data["detail"]["error"]
