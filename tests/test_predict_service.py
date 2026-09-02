"""
Test suite for Production Prediction Service (predict_service.py).
==================================================================
"""

import json
import pytest
from predict_service import (
    DroughtPredictionService,
    InvalidInputError,
    predict_drought,
)


class TestPredictionService:
    def test_predict_drought_valid_inputs(self):
        res = predict_drought(latitude=9.63, longitude=39.53, year=2005)

        assert isinstance(res, dict)
        assert "predicted_drought_class" in res
        assert res["predicted_drought_class"] in {0, 1, 2}
        assert "severity_label" in res
        assert res["severity_label"] in {"Normal", "Moderate Drought", "Severe Drought"}
        assert "confidence_probabilities" in res
        assert "grid_cell" in res
        assert res["year"] == 2005
        assert res["service_mode"] == "retrospective_reconstruction"

        # Check probabilities
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
        assert s1._model is not None


import threading
import urllib.request
import urllib.error
from http.server import HTTPServer
from predict_service import DroughtHTTPRequestHandler


@pytest.fixture(scope="module")
def http_server():
    server = HTTPServer(("127.0.0.1", 0), DroughtHTTPRequestHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


class TestDroughtHTTPServer:
    def test_get_root_health_check(self, http_server):
        req = urllib.request.Request(f"{http_server}/")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "online"
            assert "POST /predict" in data["endpoints"]
            assert "GET /predict" in data["endpoints"]

    def test_get_predict_valid_query(self, http_server):
        url = f"{http_server}/predict?lat=9.63&lon=39.53&year=2005"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "predicted_drought_class" in data
            assert data["year"] == 2005
            assert data["severity_label"] in {"Normal", "Moderate Drought", "Severe Drought"}

    def test_get_predict_missing_params(self, http_server):
        url = f"{http_server}/predict"
        req = urllib.request.Request(url)
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400

    def test_post_predict_valid_json(self, http_server):
        url = f"{http_server}/predict"
        payload = json.dumps({"latitude": 9.63, "longitude": 39.53, "year": 2005}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "predicted_drought_class" in data
            assert data["year"] == 2005

    def test_options_cors(self, http_server):
        url = f"{http_server}/predict"
        req = urllib.request.Request(url, method="OPTIONS")
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 204
            assert resp.headers.get("Access-Control-Allow-Origin") == "*"
