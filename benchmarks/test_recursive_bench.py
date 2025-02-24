import requests

def test_recursive_infer():
    url = "http://localhost:8080/api/v1/recursive_infer"
    data = {"query": "Expand recursive intelligence.", "depth": 0}
    response = requests.post(url, json=data)
    assert response.status_code == 200

test_recursive_infer()