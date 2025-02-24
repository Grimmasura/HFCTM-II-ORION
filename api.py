from fastapi import FastAPI
from src.orion_model import ORIONModel

app = FastAPI()
model = ORIONModel()

@app.get("/")
def read_root():
    return {"message": "O.R.I.O.N. ∞ API is running!"}

@app.get("/infer")
def infer(query: str):
    response = model.infer(query)
    return {"query": query, "response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
