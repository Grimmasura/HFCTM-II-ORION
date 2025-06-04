import os
import datetime
from git import Repo

# Initialize the repo
repo = Repo(os.getcwd())
BRANCH = "main"

if repo.head.is_detached:
    # Ensure we are on the desired branch when running in detached HEAD state
    repo.git.checkout(BRANCH)

# Create a test file
file_path = "test_ai_commit.txt"
with open(file_path, "w") as f:
    f.write(f"AI-generated commit at {datetime.datetime.now()}")

# Add, commit, and push
repo.git.add(file_path)
repo.index.commit("🤖 AI Auto-Commit: Updated test file")

# Only attempt to push when a GitHub token is available
token = os.getenv("GITHUB_TOKEN")
if token:
    origin = repo.remote(name="origin")
    # Temporarily use a token-authenticated URL to avoid storing credentials
    original_url = next(origin.urls, None)
    if original_url and original_url.startswith("https://"):
        authed_url = original_url.replace("https://", f"https://{token}@")
        origin.set_url(authed_url)
    try:
        origin.push(refspec=f"HEAD:{BRANCH}")
        print("✅ File committed and pushed successfully.")
    finally:
        if original_url:
            origin.set_url(original_url)
else:
    print("⚠️  GITHUB_TOKEN not found. Commit created locally but not pushed.")
