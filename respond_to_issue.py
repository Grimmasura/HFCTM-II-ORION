import requests
import os

repo_owner = "Grimmasura"
repo_name = "HFCTM-II-ORION"
issue_number = 1  # Change this dynamically if needed
github_token = os.getenv("GITHUB_TOKEN")

headers = {"Authorization": f"token {github_token}"}
url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments"

data = {"body": "🤖 AI Auto-Response: Thanks for your issue report!"}

response = requests.post(url, json=data, headers=headers)
print(response.json())
