#!/bin/bash

# Pull the latest changes from the remote repository
git pull origin main

# Add the changes
git add .

# Check if a commit message argument was provided
if [ -z "$1" ]; then
    # If no commit message was provided, use a default message
    commit_message="update"
else
    # If a commit message was provided, use it
    commit_message="$1"
fi

# Commit the changes with the specified message
git commit -m "$commit_message"

# Push the changes to the remote repository
git push -u origin main
