#!/usr/bin/env python3
"""
One-Click Uploader for Hugging Face Spaces.
Uploads the complete hf_space_deployment folder directly to your Hugging Face Space repository.

Usage:
  ./venv/bin/python scripts/upload_to_huggingface.py --token <YOUR_HF_TOKEN> --repo-id <USERNAME/SPACE_NAME>
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("❌ huggingface_hub is not installed! Run: pip install huggingface_hub")
    sys.exit(1)

DEPLOY_DIR = Path.home() / "Desktop" / "hf_space_deployment"


def main():
    parser = argparse.ArgumentParser(description="Upload STT Model & Gradio App to Hugging Face Spaces")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Access Token (with WRITE permissions)")
    parser.add_argument("--repo-id", type=str, required=True, help="Target Space ID, e.g. 'username/gujarati-stt'")
    args = parser.parse_args()

    if not DEPLOY_DIR.exists():
        print(f"❌ Deployment folder not found at: {DEPLOY_DIR}")
        sys.exit(1)

    print(f"🚀 Connecting to Hugging Face Hub...")
    api = HfApi(token=args.token)

    # Verify space existence or create it
    try:
        api.space_info(repo_id=args.repo_id)
        print(f"✅ Found existing Space: {args.repo_id}")
    except Exception:
        print(f"📦 Space '{args.repo_id}' not found, creating new Gradio Space...")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
        )
        print(f"✅ Created new Space: {args.repo_id}")

    print(f"📤 Uploading deployment folder ({DEPLOY_DIR}) to Space '{args.repo_id}'...")
    print("This may take 1-2 minutes depending on your internet upload speed...")

    api.upload_folder(
        folder_path=str(DEPLOY_DIR),
        repo_id=args.repo_id,
        repo_type="space",
        commit_message="Deploy fine-tuned Multilingual STT with Gradio Web App",
    )

    space_url = f"https://huggingface.co/spaces/{args.repo_id}"
    print("\n🎉 SUCCESS! Your Speech-to-Text Model is deployed to Hugging Face Spaces!")
    print(f"👉 Live URL: {space_url}")


if __name__ == "__main__":
    main()
