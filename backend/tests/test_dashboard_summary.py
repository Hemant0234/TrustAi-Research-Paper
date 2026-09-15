from fastapi.testclient import TestClient

from app.main import app


def test_dashboard_summary_uses_live_backend_metrics():
    client = TestClient(app)
    response = client.get('/api/dashboard/summary')

    assert response.status_code == 200, response.text

    data = response.json()
    assert 'accuracy' in data
    assert 'calibration' in data
    assert 'xqi' in data
    assert 'reliability' in data
    assert 0 <= data['accuracy'] <= 100
    assert 0 <= data['calibration'] <= 1
    assert 0 <= data['xqi'] <= 100
    assert 0 <= data['reliability'] <= 100
