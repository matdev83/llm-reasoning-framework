import argparse
import datetime
import re

def update_changelog(commit_message: str, changed_files: list):
    changelog_path = "CHANGELOG.md"
    
    with open(changelog_path, "r") as f:
        content = f.readlines()

    # Find the line number for "## Commit History"
    commit_history_index = -1
    for i, line in enumerate(content):
        if line.strip() == "## Commit History":
            commit_history_index = i
            break

    if commit_history_index == -1:
        print("Error: '## Commit History' section not found in CHANGELOG.md")
        return

    # Prepare the new entry
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    
    # Placeholder for commit hash, will be updated later
    new_entry_lines = [
        f"### <COMMIT_HASH> - {current_date} - {commit_message}\n",
        f"**Changes:**\n"
    ]
    for f in changed_files:
        status = f[0] # M, A, D, R, etc.
        file_path = f[1:].strip() # Remove status and leading space
        
        # Determine action based on status
        action = "Modified"
        if status == "A":
            action = "Added"
        elif status == "D":
            action = "Deleted"
        elif status == "R":
            action = "Renamed"
        elif status == "C":
            action = "Copied"
        elif status == "?": # Untracked, should not happen if using git status --porcelain
            action = "Untracked"
        
        new_entry_lines.append(f"- {action} {file_path}\n")
    new_entry_lines.append("\n") # Add an empty line for spacing

    # Insert the new entry after "## Commit History"
    content.insert(commit_history_index + 1, "\n") # Add an empty line before the new entry
    for line in reversed(new_entry_lines): # Insert in reverse to maintain order
        content.insert(commit_history_index + 1, line)

    with open(changelog_path, "w") as f:
        f.writelines(content)
    
    print(f"CHANGELOG.md updated successfully with commit message: '{commit_message}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update CHANGELOG.md with a new commit entry.")
    parser.add_argument("--commit-message", required=True, help="The commit message.")
    parser.add_argument("--changed-files", nargs='*', help="List of changed files from git status --porcelain output.")
    
    args = parser.parse_args()
    
    update_changelog(args.commit_message, args.changed_files)
