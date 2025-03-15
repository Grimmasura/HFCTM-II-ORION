import requests
import os

repo_owner = "Grimmasura"
repo_name = "HFCTM-II-ORION"
pull_number = 1  # Update dynamically
github_token = os.getenv("GITHUB_TOKEN")

headers = {"Authorization": f"token {github_token}"}
url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pull_number}/merge"

data = {"commit_title": "🤖 AI Auto-Merge: Approved"}

response = requests.put(url, json=data, headers=headers)
print(response.json())
