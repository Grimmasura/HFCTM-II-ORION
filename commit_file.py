import os
import datetime
from git import Repo

# Initialize the repo
repo = Repo(os.getcwd())

# Create a test file
file_path = "test_ai_commit.txt"
with open(file_path, "w") as f:
    f.write(f"AI-generated commit at {datetime.datetime.now()}")

# Add, commit, and push
repo.git.add(file_path)
repo.index.commit("🤖 AI Auto-Commit: Updated test file")

# Only attempt to push when a GitHub token is available
if os.getenv("GITHUB_TOKEN"):
    origin = repo.remote(name="origin")
    origin.push()
    print("✅ File committed and pushed successfully.")
else:
    print("⚠️  GITHUB_TOKEN not found. Commit created locally but not pushed.")
