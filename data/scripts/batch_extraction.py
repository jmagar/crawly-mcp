#!/usr/bin/env python3
"""
Batch extraction trigger for GitHub PR AI prompts across all repositories.

Usage:
    python scripts/batch_extraction.py [options]

Examples:
    # Auto-discover and process all recent PRs from last 7 days
    python scripts/batch_extraction.py --auto-discover

    # Auto-discover with custom timeframe
    python scripts/batch_extraction.py --auto-discover --days 14

    # List recent PRs without processing
    python scripts/batch_extraction.py --list-only --days 3

    # Use production webhook server
    python scripts/batch_extraction.py --auto-discover --url https://githook.tootie.tv
"""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def fetch_recent_prs(webhook_url: str, days: int = 7) -> dict:
    """Fetch recent PRs from webhook server.

    Args:
        webhook_url: Webhook server URL
        days: Days to look back

    Returns:
        API response with recent PRs
    """
    endpoint = f"{webhook_url.rstrip('/')}/recent"
    params = f"?days={days}"

    req = Request(f"{endpoint}{params}", method="GET")

    try:
        print(f"🔍 Fetching recent PRs from last {days} days...")
        print(f"   Endpoint: {endpoint}{params}")

        with urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))

        return result

    except HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_data = json.loads(error_body)
            error_msg = error_data.get("detail", str(e))
        except json.JSONDecodeError:
            error_msg = error_body

        print(f"❌ HTTP Error {e.code}: {error_msg}")
        return {}

    except URLError as e:
        print(f"❌ Connection error: {e.reason}")
        print(f"   Make sure the webhook server is running at {webhook_url}")
        return {}

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {}


def trigger_batch_extraction(
    webhook_url: str,
    prs: list | None = None,
    auto_discover: bool = False,
    days: int = 7,
) -> bool:
    """Trigger batch extraction via webhook server API.

    Args:
        webhook_url: Webhook server URL
        prs: List of PR dictionaries with owner, repo, pr_number
        auto_discover: Auto-discover recent PRs
        days: Days to look back for auto-discovery

    Returns:
        True if successful, False otherwise
    """
    endpoint = f"{webhook_url.rstrip('/')}/batch"

    # Prepare payload
    payload = {"auto_discover": auto_discover}

    if auto_discover:
        payload["days"] = days
        print(f"📤 Triggering batch extraction with auto-discovery ({days} days)")
    else:
        if not prs:
            print("❌ No PRs provided and auto-discovery disabled")
            return False
        payload["prs"] = prs
        print(f"📤 Triggering batch extraction for {len(prs)} PRs")

    # Prepare request
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Content-Length": str(len(data))}

    req = Request(endpoint, data=data, headers=headers, method="POST")

    try:
        print(f"   Endpoint: {endpoint}")

        with urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))

        print(f"✅ {result.get('message', 'Batch extraction queued successfully')}")
        print(f"   Total PRs: {result.get('total_prs', 0)}")
        print(f"   Successfully queued: {result.get('successfully_queued', 0)}")
        if result.get("failed", 0) > 0:
            print(f"   Failed: {result.get('failed', 0)}")

        # Show details of queued tasks
        tasks = result.get("tasks", [])
        if tasks:
            print("\n📋 Queued tasks:")
            for task in tasks[:10]:  # Show first 10
                status_icon = "✅" if task["status"] == "queued" else "❌"
                print(f"   {status_icon} {task['repo']}#{task['pr_number']}")
            if len(tasks) > 10:
                print(f"   ... and {len(tasks) - 10} more")

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


def display_recent_prs(prs_data: dict):
    """Display recent PRs in a formatted table."""
    recent_prs = prs_data.get("recent_prs", [])
    total_prs = prs_data.get("total_prs", 0)
    days_back = prs_data.get("days_back", 7)

    print(f"\n📊 Found {total_prs} PRs updated in the last {days_back} days:")
    print("=" * 80)

    if not recent_prs:
        print("No recent PRs found.")
        return

    for pr in recent_prs:
        state_icon = "🟢" if pr["state"] == "open" else "🔵"
        print(f"{state_icon} {pr['repo']}#{pr['pr_number']}: {pr['title']}")
        print(f"   State: {pr['state']} | Updated: {pr['updated_at']}")
        print(f"   URL: {pr['url']}")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch trigger GitHub PR AI prompt extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="Auto-discover recent PRs from all repositories",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back (default: 7)",
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list recent PRs without triggering extraction",
    )

    parser.add_argument(
        "--url",
        default=None,
        help="Webhook server URL (default: http://localhost:38080 or WEBHOOK_SERVER_URL env var)",
    )

    args = parser.parse_args()

    # Get webhook URL
    webhook_url = args.url or os.getenv("WEBHOOK_SERVER_URL", "http://localhost:38080")

    success = True

    if args.list_only or args.auto_discover:
        # Fetch recent PRs
        prs_data = fetch_recent_prs(webhook_url, args.days)
        if not prs_data:
            sys.exit(1)

        display_recent_prs(prs_data)

        if args.list_only:
            print("✅ List completed (no extraction triggered)")
            sys.exit(0)

    if args.auto_discover:
        # Trigger batch extraction with auto-discovery
        success = trigger_batch_extraction(
            webhook_url=webhook_url,
            auto_discover=True,
            days=args.days,
        )
    else:
        print("❌ No action specified. Use --auto-discover or --list-only")
        parser.print_help()
        success = False

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
