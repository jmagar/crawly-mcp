#!/usr/bin/env python3
"""
Manual extraction trigger for GitHub PR AI prompts.

Usage:
    python scripts/manual_extraction.py <owner> <repo> <pr_number>

Example:
    python scripts/manual_extraction.py jmagar crawler-mcp 10
"""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def trigger_manual_extraction(
    owner: str, repo: str, pr_number: int, webhook_url: str | None = None
) -> bool:
    """Trigger manual extraction via webhook server API.

    Args:
        owner: GitHub repository owner
        repo: Repository name
        pr_number: Pull request number
        webhook_url: Webhook server URL (defaults to localhost)

    Returns:
        True if successful, False otherwise
    """
    if not webhook_url:
        # Try to get from environment or use default
        webhook_url = os.getenv("WEBHOOK_SERVER_URL", "http://localhost:38080")

    endpoint = f"{webhook_url.rstrip('/')}/manual"

    # Prepare payload
    payload = {"owner": owner, "repo": repo, "pr_number": pr_number}

    # Prepare request
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Content-Length": str(len(data))}

    req = Request(endpoint, data=data, headers=headers, method="POST")

    try:
        print(f"📤 Triggering extraction for {owner}/{repo}#{pr_number}")
        print(f"   Endpoint: {endpoint}")

        with urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))

        print(f"✅ {result.get('message', 'Extraction queued successfully')}")
        print(f"   Status: {result.get('status', 'unknown')}")
        return True

    except HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_data = json.loads(error_body)
            error_msg = error_data.get("detail", str(e))
        except json.JSONDecodeError:
            error_msg = error_body

        print(f"❌ HTTP Error {e.code}: {error_msg}")
        return False

    except URLError as e:
        print(f"❌ Connection error: {e.reason}")
        print(f"   Make sure the webhook server is running at {webhook_url}")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manually trigger GitHub PR AI prompt extraction"
    )
    parser.add_argument("owner", help="GitHub repository owner")
    parser.add_argument("repo", help="Repository name")
    parser.add_argument("pr_number", type=int, help="Pull request number")
    parser.add_argument(
        "--url",
        default=None,
        help="Webhook server URL (default: http://localhost:38080 or WEBHOOK_SERVER_URL env var)",
    )

    args = parser.parse_args()

    # Trigger extraction
    success = trigger_manual_extraction(
        owner=args.owner, repo=args.repo, pr_number=args.pr_number, webhook_url=args.url
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
