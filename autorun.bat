@echo off

REM Pull the latest changes from the remote repository
git pull origin main

REM Set the Python script to execute
set PYTHON_SCRIPT=main.py

REM Execute your Python script with all the arguments passed to the batch script
python %PYTHON_SCRIPT% %*

REM Add and commit the changes
git add .
git commit -m "update"

REM Push the changes to the remote repository
git push -u origin main

REM Example of using this file
REM batch_script.bat --model DKT_AUG --dataset ednet_qs --batch_size 16
