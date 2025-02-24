import requests
import time

API_URL = "http://localhost:8080/api/v1/recursive_ai/infer"

def benchmark_recursive_ai():
    depths_to_test = [1, 5, 10, 20, 50]
    for depth in depths_to_test:
        start_time = time.time()
        response = requests.post(API_URL, json={"query": "Define recursion", "depth": depth})
        elapsed_time = time.time() - start_time

        print(f"Depth: {depth}, Response: {response.json()}, Time: {elapsed_time:.2f}s")

if __name__ == "__main__":
    benchmark_recursive_ai()
