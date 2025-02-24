from src.orion_model import ORIONModel

if __name__ == "__main__":
    model = ORIONModel()
    query = input("Enter your query: ")
    response = model.infer(query)
    print("O.R.I.O.N. ∞ Response:", response)
