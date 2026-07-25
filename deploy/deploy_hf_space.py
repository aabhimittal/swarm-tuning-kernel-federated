"""One-command deploy of the parameter server to a Hugging Face Space.

Requires an HF *write* token (https://huggingface.co/settings/tokens):

    export HF_TOKEN=hf_xxx
    python deploy/deploy_hf_space.py --repo-id YOUR_USERNAME/swarm-server

The script stages the files into the layout a Docker Space expects (Dockerfile +
README.md at the root) and uploads them. Re-running it redeploys (the Space
rebuilds automatically).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def stage(staging: str) -> None:
    # Dockerfile + README.md must live at the Space repo root.
    shutil.copy(os.path.join(REPO_ROOT, "deploy", "Dockerfile"), os.path.join(staging, "Dockerfile"))
    shutil.copy(os.path.join(REPO_ROOT, "deploy", "README_SPACE.md"), os.path.join(staging, "README.md"))
    shutil.copy(os.path.join(REPO_ROOT, "requirements.txt"), os.path.join(staging, "requirements.txt"))
    for pkg in ("swarm", "server", "worker"):
        shutil.copytree(
            os.path.join(REPO_ROOT, pkg),
            os.path.join(staging, pkg),
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy the swarm parameter server to a HF Space.")
    p.add_argument("--repo-id", required=True, help="e.g. your-username/swarm-server")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    p.add_argument("--private", action="store_true", help="Create the Space as private.")
    p.add_argument(
        "--skip-create",
        action="store_true",
        help="Upload to an existing Space without calling create_repo. Use when the "
        "repo already exists but repo creation is refused (e.g. HF returns 402 "
        "because Docker Spaces need PRO) — the code still lands in the repo.",
    )
    args = p.parse_args()

    if not args.token:
        raise SystemExit("No HF token. Set HF_TOKEN or pass --token (needs *write* scope).")

    from huggingface_hub import HfApi  # imported here so --help works without the dep

    api = HfApi(token=args.token)
    if not args.skip_create:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="docker",
            private=args.private,
            exist_ok=True,
        )

    with tempfile.TemporaryDirectory() as staging:
        stage(staging)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="space",
            folder_path=staging,
            commit_message="Deploy swarm parameter server",
        )

    url = f"https://huggingface.co/spaces/{args.repo_id}"
    host = f"https://{args.repo_id.replace('/', '-')}.hf.space"
    print(f"Deployed: {url}")
    print(f"Once the build finishes, point workers at: {host}")
    print(f"  python -m worker.run_worker --server {host} --worker-id w0 --steps 100")


if __name__ == "__main__":
    main()
