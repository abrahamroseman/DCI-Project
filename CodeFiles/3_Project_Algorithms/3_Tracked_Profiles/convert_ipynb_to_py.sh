#!/bin/bash

# --- LIST YOUR NOTEBOOKS HERE ---
# Only include the names, no extension needed
NOTEBOOKS=(
    # "Tracked_Profiles"
    # "Tracked_Profiles_UpdraftArea"
    # "Tracked_Profiles_Plotting_TZContours"
    # "Tracked_Ascent_Trajectories"
    # "EntrainmentTrackback"
    "EulerianReconstruction"
)

# Optional: Load your environment if jupyter isn't in your base path
# source activate work 

echo "Starting conversion of ${#NOTEBOOKS[@]} notebooks..."
echo "--------------------------------------------"

for nb in "${NOTEBOOKS[@]}"; do
    FILE="${nb}.ipynb"
    
    if [ -f "$FILE" ]; then
        echo "[PROCESSING] $FILE..."
        
        # Convert to script
        # --loglevel ERROR keeps the output clean
        jupyter nbconvert --to script "$FILE" --log-level ERROR
        
        if [ $? -eq 0 ]; then
            echo "[SUCCESS] Created ${nb}.py"
        else
            echo "[ERROR] Failed to convert $nb"
        fi
    else
        echo "[SKIPPED] $FILE not found in current directory."
    fi
done

echo "--------------------------------------------"
echo "Done!"
