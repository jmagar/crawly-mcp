import os
import re
import sys

import requests  # You may need to run: pip install requests
from dotenv import load_dotenv  # You may need to run: pip install python-dotenv

# --- Configuration ---
OWNER = sys.argv[1]  # e.g., "your-github-org"
REPO = sys.argv[2]  # e.g., "your-repo-name"
PR_NUMBER = sys.argv[3]  # e.g., "123"
OUTPUT_FILE = f"{REPO}-pr{PR_NUMBER}-fixes.md"
PROMPT_HEADER = "🤖 Prompt for AI Agents"

# Enhanced Filtering Configuration
FILTER_DISMISSED_REVIEWS = True  # Filter out dismissed/pending reviews
FILTER_RESOLVED_THREADS = True  # Filter resolved conversations
FILTER_STALE_CODE = True  # Filter stale code suggestions
FILTER_OLD_COMMENTS = True  # Filter very old comments (30+ days)
FILTER_PRE_FORCE_PUSH = True  # Filter comments before force-push events
ATTEMPT_TIMELINE_FETCH = True  # Try to fetch timeline for force-push detection
VERBOSE_FILTERING = True  # Show detailed filtering messages

# For security, get your token from an environment variable
# On Mac/Linux: export GITHUB_TOKEN="your_token_here"
# On Windows: set GITHUB_TOKEN="your_token_here"

# Load environment variables from .env file in the project root
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
# ---------------------


def _parse_file_reference_from_body(body):
    """Parse file and line references from comment body text."""
    if not body:
        return None

    # Pattern 1: HTML summary tags "<summary>crawler_mcp/path/file.py (2)</summary>"
    html_summary_pattern = r"<summary>([^/\s]+/[^/\s]+\.py)"
    match = re.search(html_summary_pattern, body)
    if match:
        # Look for line numbers in backticks nearby
        line_backtick_pattern = r"`(\d+(?:-\d+)?)`:"
        line_match = re.search(line_backtick_pattern, body)
        if line_match:
            return f"{match.group(1)}:{line_match.group(1)}"
        return match.group(1)

    # Pattern 2: "In crawler_mcp/path/file.py around lines X-Y" or "around line X"
    file_in_text_pattern = (
        r"In ([^/\s]+/[^/\s]+\.py) around lines? (\d+(?:-\d+)?|\d+ to \d+)"
    )
    match = re.search(file_in_text_pattern, body)
    if match:
        return f"{match.group(1)}:{match.group(2)}"

    # Pattern 3: "In file.py around lines X-Y" (just filename without path)
    file_simple_pattern = r"In ([^/\s]+\.py) around lines? (\d+(?:-\d+)?|\d+ to \d+)"
    match = re.search(file_simple_pattern, body)
    if match:
        return f"{match.group(1)}:{match.group(2)}"

    # Pattern 4: Line number in backticks "`445-447`:" at start or after newline
    line_backtick_pattern = r"`(\d+(?:-\d+)?)`:"
    match = re.search(line_backtick_pattern, body)
    if match:
        return f"line:{match.group(1)}"

    # Pattern 5: Line number prefix at start of body "414-447:" or "414:"
    line_prefix_pattern = r"^(\d+(?:-\d+)?):"
    match = re.match(line_prefix_pattern, body.strip())
    if match:
        return f"line:{match.group(1)}"

    # Pattern 6: Just file path mention "In crawler_mcp/path/file.py" without line numbers
    file_only_pattern = r"In ([^/\s]+/[^/\s]+\.py)"
    match = re.search(file_only_pattern, body)
    if match:
        return match.group(1)

    return None


def get_coderabbit_prompts():
    """Fetches and compiles coderabbit prompts from a GitHub PR."""
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN environment variable not set.")
        sys.exit(1)

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    # API endpoints for PR reviews and review comments
    urls = [
        f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}/reviews",
        f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}/comments",
    ]

    all_prompts = []
    filtered_count = 0

    # Track filtered items for improved filtering
    dismissed_review_ids = set()
    filtered_comment_ids = set()

    print(f"🔍 Searching for prompts in PR #{PR_NUMBER} of {OWNER}/{REPO}...")

    # First pass: Process reviews to identify dismissed ones
    reviews_url = (
        f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}/reviews"
    )
    try:
        response = requests.get(reviews_url, headers=headers)
        response.raise_for_status()
        reviews = response.json()

        for review in reviews:
            # Track dismissed or pending reviews to filter out their comments
            if FILTER_DISMISSED_REVIEWS and review.get("state") in [
                "DISMISSED",
                "PENDING",
            ]:
                dismissed_review_ids.add(review.get("id"))
                filtered_count += 1
                if VERBOSE_FILTERING:
                    print(
                        f"⚠️  Filtered dismissed/pending review {review.get('id')}: {review.get('state')}"
                    )
    except requests.exceptions.RequestException as e:
        print(f"Error fetching reviews: {e}")
        sys.exit(1)

    # Get PR timeline to identify force-push events and significant changes
    last_force_push_date = None
    if ATTEMPT_TIMELINE_FETCH and FILTER_PRE_FORCE_PUSH:
        try:
            timeline_url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/{PR_NUMBER}/timeline"

            # Timeline API requires preview header for full access
            timeline_headers = headers.copy()
            timeline_headers["Accept"] = (
                "application/vnd.github.mockingbird-preview+json"
            )

            response = requests.get(timeline_url, headers=timeline_headers)
            response.raise_for_status()
            timeline_events = response.json()

            # Find the most recent force-push or head_ref_force_pushed event
            for event in reversed(
                timeline_events
            ):  # Process in reverse chronological order
                if event.get("event") in [
                    "head_ref_force_pushed",
                    "synchronize",
                ] and event.get("created_at"):
                    from datetime import datetime

                    last_force_push_date = datetime.fromisoformat(
                        event["created_at"].replace("Z", "+00:00")
                    )
                    if VERBOSE_FILTERING:
                        print(f"📅 Found force-push event at {last_force_push_date}")
                    break
        except requests.exceptions.RequestException as e:
            if VERBOSE_FILTERING:
                print(f"⚠️  Could not fetch timeline (non-critical): {e}")
            # Continue without timeline filtering

    # Second pass: Process all items from both endpoints
    for url in urls:
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
            items = response.json()

            for item in items:
                body = item.get("body")
                user_login = item.get("user", {}).get("login", "")
                file_path = item.get("path")
                line_number = item.get("line")
                position = item.get("position")
                original_position = item.get("original_position")
                pull_request_review_id = item.get("pull_request_review_id")
                in_reply_to_id = item.get("in_reply_to_id")
                item_id = item.get("id")

                # Skip empty comments
                if not body:
                    continue

                # Enhanced filtering logic
                should_filter = False
                filter_reason = ""

                # Filter comments from dismissed/pending reviews
                if (
                    FILTER_DISMISSED_REVIEWS
                    and pull_request_review_id
                    and pull_request_review_id in dismissed_review_ids
                ):
                    should_filter = True
                    filter_reason = f"comment from dismissed/pending review {pull_request_review_id}"

                # Filter comments that are replies to already filtered comments
                elif in_reply_to_id and in_reply_to_id in filtered_comment_ids:
                    should_filter = True
                    filter_reason = f"reply to filtered comment {in_reply_to_id}"

                # Original outdated logic (position is null but original_position exists)
                elif position is None and original_position is not None:
                    should_filter = True
                    filter_reason = "outdated position in diff"

                # Enhanced resolved detection - check API resolved field and text patterns
                elif FILTER_RESOLVED_THREADS and (
                    item.get("resolved") is True
                    or (
                        body
                        and any(
                            phrase in body.lower()
                            for phrase in [
                                "resolved",
                                "fixed",
                                "addressed",
                                "done",
                                "completed",
                                "no longer relevant",
                                "outdated",
                            ]
                        )
                    )
                ):
                    should_filter = True
                    filter_reason = "resolved conversation"

                # Filter stale code suggestions (diff_hunk indicates outdated context)
                elif (
                    FILTER_STALE_CODE
                    and item.get("diff_hunk")
                    and item.get("original_commit_id")
                ):
                    # If there's a diff_hunk but no current position, likely stale
                    if not position and item.get("diff_hunk"):
                        should_filter = True
                        filter_reason = "stale code suggestion (diff context changed)"

                # Filter very old comments (more than 30 days old and no recent activity)
                elif FILTER_OLD_COMMENTS and item.get("created_at"):
                    try:
                        from datetime import datetime

                        created_date = datetime.fromisoformat(
                            item["created_at"].replace("Z", "+00:00")
                        )
                        updated_date = datetime.fromisoformat(
                            item.get("updated_at", item["created_at"]).replace(
                                "Z", "+00:00"
                            )
                        )
                        now = datetime.now().astimezone()

                        # Filter comments older than 30 days with no recent updates
                        if (now - created_date).days > 30 and (
                            now - updated_date
                        ).days > 7:
                            should_filter = True
                            filter_reason = f"stale comment (created {(now - created_date).days} days ago, last updated {(now - updated_date).days} days ago)"

                        # Filter comments created before the last force-push (likely outdated)
                        elif (
                            FILTER_PRE_FORCE_PUSH
                            and last_force_push_date
                            and created_date < last_force_push_date
                        ):
                            should_filter = True
                            filter_reason = (
                                "comment predates force-push (likely outdated)"
                            )
                    except (ValueError, TypeError):
                        pass  # Skip filtering if date parsing fails

                if should_filter:
                    filtered_count += 1
                    if item_id:
                        filtered_comment_ids.add(item_id)
                    if VERBOSE_FILTERING:
                        print(f"⚠️  Filtered comment {item_id}: {filter_reason}")
                    continue

                # Create file reference string for output
                file_ref = ""
                parsed_file_info = None

                if file_path:
                    if line_number:
                        file_ref = f" - {file_path}:{line_number}"
                    else:
                        file_ref = f" - {file_path}"
                else:
                    # Try to extract file information from comment body text
                    parsed_file_info = _parse_file_reference_from_body(body)
                    if parsed_file_info:
                        file_ref = f" - {parsed_file_info}"

                # Check if this comment has AI Prompt sections (only from CodeRabbit)
                if PROMPT_HEADER in body and user_login == "coderabbitai[bot]":
                    # Extract all details blocks that contain the prompt header
                    details_pattern = r"<details>\s*<summary>🤖 Prompt for AI Agents</summary>\s*(.*?)\s*</details>"
                    details_matches = re.findall(details_pattern, body, re.DOTALL)

                    for details_content in details_matches:
                        # Extract content between triple backticks
                        code_pattern = r"```\s*(.*?)\s*```"
                        code_matches = re.findall(
                            code_pattern, details_content, re.DOTALL
                        )

                        for code_content in code_matches:
                            cleaned_prompt = code_content.strip()
                            if cleaned_prompt:  # Only add non-empty prompts
                                # Add file reference info as a prefix for AI prompts
                                if file_ref:
                                    all_prompts.append(
                                        f"- [ ] [AI PROMPT{file_ref}]\n{cleaned_prompt}"
                                    )
                                else:
                                    all_prompts.append(f"- [ ] {cleaned_prompt}")

                # If no AI Prompt sections, check for Committable suggestions (from any user)
                elif "📝 Committable suggestion" in body:
                    # Extract the suggestion content from the suggestion block
                    suggestion_pattern = r"```suggestion\s*(.*?)\s*```"
                    suggestion_matches = re.findall(suggestion_pattern, body, re.DOTALL)

                    for suggestion_content in suggestion_matches:
                        cleaned_suggestion = suggestion_content.strip()
                        if cleaned_suggestion:  # Only add non-empty suggestions
                            # Add a prefix to distinguish from AI prompts, include author and file
                            all_prompts.append(
                                f"- [ ] [COMMITTABLE SUGGESTION - {user_login}{file_ref}]\n{cleaned_suggestion}"
                            )

                # Also check for Copilot suggestions (even if they don't have the formal committable suggestion header)
                elif (
                    user_login in ["Copilot", "copilot-pull-request-reviewer[bot]"]
                    and "```suggestion" in body
                ):
                    # Extract the suggestion content from Copilot suggestion blocks
                    suggestion_pattern = r"```suggestion\s*(.*?)\s*```"
                    suggestion_matches = re.findall(suggestion_pattern, body, re.DOTALL)

                    for suggestion_content in suggestion_matches:
                        cleaned_suggestion = suggestion_content.strip()
                        if cleaned_suggestion:  # Only add non-empty suggestions
                            # Add a prefix to distinguish from AI prompts, include author and file
                            all_prompts.append(
                                f"- [ ] [COPILOT SUGGESTION - {user_login}{file_ref}]\n{cleaned_suggestion}"
                            )

                # Also capture Copilot review overviews (without suggestions)
                elif (
                    user_login == "copilot-pull-request-reviewer[bot]"
                    and body
                    and "```suggestion" not in body
                ):
                    # Extract meaningful review content (skip generic footers)
                    if len(body.strip()) > 50 and not body.strip().startswith("---"):
                        # Clean up the review content
                        cleaned_review = body.strip()
                        # Remove the footer tip section
                        if "**Tip:** Customize your code reviews" in cleaned_review:
                            cleaned_review = cleaned_review.split("---")[0].strip()
                        if cleaned_review:
                            all_prompts.append(
                                f"- [ ] [COPILOT REVIEW - {user_login}]\n{cleaned_review}"
                            )

                # Additionally, check for any other code blocks (diff, python, etc.) in any comment
                # Look for code blocks with various languages that haven't been captured yet
                code_block_patterns = [
                    (r"```diff\s*(.*?)\s*```", "DIFF"),
                    (r"```patch\s*(.*?)\s*```", "PATCH"),
                    (r"```python\s*(.*?)\s*```", "PYTHON"),
                    (r"```javascript\s*(.*?)\s*```", "JAVASCRIPT"),
                    (r"```typescript\s*(.*?)\s*```", "TYPESCRIPT"),
                    (r"```json\s*(.*?)\s*```", "JSON"),
                    (r"```yaml\s*(.*?)\s*```", "YAML"),
                    (r"```sql\s*(.*?)\s*```", "SQL"),
                    (r"```shell\s*(.*?)\s*```", "SHELL"),
                    (r"```bash\s*(.*?)\s*```", "BASH"),
                ]

                # Only extract if we haven't already processed this comment for AI prompts or suggestions
                if not (
                    PROMPT_HEADER in body
                    or "📝 Committable suggestion" in body
                    or (
                        user_login in ["Copilot", "copilot-pull-request-reviewer[bot]"]
                        and "```suggestion" in body
                    )
                ):
                    for pattern, lang_type in code_block_patterns:
                        matches = re.findall(pattern, body, re.DOTALL)
                        for match in matches:
                            cleaned_code = match.strip()
                            if (
                                cleaned_code and len(cleaned_code) > 20
                            ):  # Only meaningful code blocks
                                all_prompts.append(
                                    f"- [ ] [{lang_type} BLOCK - {user_login}{file_ref}]\n{cleaned_code}"
                                )

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            sys.exit(1)

    if not all_prompts:
        print(
            "No '🤖 Prompt for AI Agents', 'Committable suggestion', 'Copilot' content, or code blocks found."
        )
        return

    # Write all compiled prompts to the output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# AI Review Content from PR #{PR_NUMBER}\n\n")
        f.write(
            f"**Extracted from PR:** https://github.com/{OWNER}/{REPO}/pull/{PR_NUMBER}\n"
        )
        f.write(f"**Original items found:** {len(all_prompts) + filtered_count}\n")
        f.write(f"**Items filtered out:** {filtered_count}\n")
        f.write(f"**Final items kept:** {len(all_prompts)}\n")
        if last_force_push_date:
            f.write(
                f"**Last significant change:** {last_force_push_date.strftime('%Y-%m-%d %H:%M UTC')}\n"
            )
        f.write("\n---\n\n")
        f.write("\n\n---\n\n".join(all_prompts))

    print(f"✅ Success! Found {len(all_prompts) + filtered_count} total items")
    if filtered_count > 0:
        print(f"🗑️ Filtered out {filtered_count} outdated/resolved/stale comments")
        print(
            "   • Filtered: dismissed reviews, resolved threads, stale suggestions, outdated comments"
        )
    print(f"✅ Kept {len(all_prompts)} relevant items in '{OUTPUT_FILE}'")
    if last_force_push_date:
        print("📅 Used timeline data to filter pre-force-push comments")
    print("📋 Output includes metadata and filtering statistics")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python get_prompts.py <OWNER> <REPO> <PR_NUMBER>")
        sys.exit(1)
    get_coderabbit_prompts()
