from stable_baselines3 import PPO

agent = PPO.load("recursive_live_optimization_model")

def recursive_model_live(query, depth):
    optimal_depth = agent.predict(depth)[0]
    if optimal_depth == 0:
        return f"Base case: {query}"
    response = f"Recursive Expansion of '{query}' at depth {optimal_depth}"
    return response + "\n" + recursive_model_live(query, optimal_depth - 1)
