#!/bin/bash

# Set the path to the folder whose files you want to iterate
path=""

# Create the output file
outputfile=".csv"
echo "file,last line" > "$outputfile"

# Iterate over the CSV files in the folder
for file in $path/*.csv; do
    # Extract the numeric part from the filename using "sed" and "cut"
    filename=$(echo "$file" | grep -oE '[0-9]+-[0-9]+')
    # Execute the "tail" command to get the last line of the file
    lastline=$(tail -n 1 "$file")
    # Appends the last line to the output file with the source file name as the identifier
    echo "$filename,$lastline" >> "$outputfile"
done
