import time
from orion_model import ORIONModel

def run_benchmark():
    model = ORIONModel()
    queries = [f"Test Query {i}" for i in range(100)]
    start_time = time.time()
    
    results = [model.infer(query) for query in queries]
    
    total_time = time.time() - start_time
    avg_time = total_time / len(queries)

    print(f"Total Inference Time: {total_time:.4f} seconds")
    print(f"Average Query Time: {avg_time:.4f} seconds")

if __name__ == "__main__":
    run_benchmark()
