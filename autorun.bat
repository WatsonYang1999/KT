@echo off

REM Pull the latest changes from the remote repository
git pull origin main

REM Execute your Python script
python main.py --model DKT_AUG --dataset ednet_qs --batch_size 16

REM Add and commit the changes
git add .
git commit -m "update"

REM Push the changes to the remote repository
git push -u origin main
