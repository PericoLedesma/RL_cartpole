#!/bin/bash

clear 
echo "Running models.... "

# Using find command to get all directories
directories=$(find . -maxdepth 1 -type d)

# Loop through each directory
for dir in $directories; do
    if [[ -f "$dir/main.py" ]]; then
        echo " "
        echo "=============================================================="
        echo "=============================================================="
        echo "                       Running $dir"
        echo "=============================================================="
        echo "=============================================================="

        python3 "$dir/main.py"

        echo "=============================================================="
        echo "=============================================================="
        echo "       Finished running main.py in directory: $dir"
        echo "=============================================================="
        echo "=============================================================="
        echo " "
    else
        echo "No main.py found in directory: $dir"
    fi
done