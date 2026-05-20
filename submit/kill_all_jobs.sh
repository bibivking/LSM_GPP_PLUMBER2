#!/bin/bash

# Account name
account="mm3972"

cd /srv/ccrc/LandAP/z5218916/script/PLUMBER2/LSM_VPD_PLUMBER2/submit
# Get the list of job IDs for the account and save to a file
qstat -u "$account" | grep 'express' | awk '{print $1}' > job_ids.txt

# Loop through the job IDs and delete each one
while read job_id; do
    qdel "$job_id"
done < job_ids.txt

# Optionally, you can clean up by removing the temporary file
rm job_ids.txt

