@echo off

REM Pull the latest changes from the remote repository
git pull origin main

REM Add the changes
git add .

REM Check if a commit message argument was provided
IF "%~1"=="" (
    REM If no commit message was provided, use a default message
    set commit_message=update
) ELSE (
    REM If a commit message was provided, use it
    set commit_message=%~1
)

REM Commit the changes with the specified message
git commit -m "%commit_message%"

REM Push the changes to the remote repository
git push -u origin main
