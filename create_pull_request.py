import requests
import os

repo_owner = "Grimmasura"
repo_name = "HFCTM-II-ORION"
branch_name = "feature-ai-update"
github_token = os.getenv("GITHUB_TOKEN")  # Set this in GitHub Secrets

headers = {"Authorization": f"token {github_token}"}
url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"

data = {
    "title": "🤖 AI-Generated PR: Auto-update",
    "head": branch_name,
    "base": "main",
    "body": "This is an AI-generated PR for automation testing.",
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
