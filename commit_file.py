import os
from git import Repo

repo = Repo(os.getcwd())
repo.git.add(all=True)

if repo.is_dirty():
    repo.index.commit("🤖 Auto-commit: routine update")
    token = os.getenv("GITHUB_TOKEN")
    if token:
        origin = repo.remote(name='origin')
        url = origin.url
        if token not in url:
            url = url.replace("https://", f"https://x-access-token:{token}@")
        origin.set_url(url)
        origin.push()
        print("Pushed changes to remote.")
    else:
        print("Committed locally. No token provided, skipping push.")
else:
    print("No changes to commit.")
