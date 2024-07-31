#!/bin/bash

# Activate the virtual environment
source /home/apps/forkedFloorplanToBlender3d/myenv/bin/activate

# Define the log file
LOGFILE="/home/apps/forkedFloorplanToBlender3d/logs/process_pending_jobs.log"

# Run the Python script and redirect stdout and stderr to the log file
python /home/apps/forkedFloorplanToBlender3d/Server/process_pending_jobs_to_firebase.py >> $LOGFILE 2>&1