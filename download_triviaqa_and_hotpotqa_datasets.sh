#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Target directory for datasets
DATASET_DIR="collection"

# Array of dataset information
# Format: "FILE_ID|FILENAME"
# Add more entries as needed
declare -a DATASETS=(
    "1NAv33zn37UZvKq_38avY0K7Lvx03-DxW|hotpot_train_v1.1.json"
    # Add more datasets in the same format:
    # "GOOGLE_DRIVE_FILE_ID|FILENAME"
)

echo -e "${YELLOW}Starting download of datasets...${NC}"

# Function to clean up on error
cleanup() {
    local filename=$1
    echo -e "${RED}Error occurred. Cleaning up...${NC}"
    rm -f "$filename"
    exit 1
}

# Create dataset directory if it doesn't exist
mkdir -p "$DATASET_DIR"

# Function to download and verify a single file
download_dataset() {
    local file_id=$1
    local filename=$2
    
    echo -e "${YELLOW}Downloading ${filename}...${NC}"
    
    # Download the file using gdown
    if gdown "https://drive.google.com/uc?id=$file_id" -O "${DATASET_DIR}/${filename}"; then
        echo -e "${GREEN}Successfully downloaded ${filename}${NC}"
        
        # Verify file was downloaded and has content
        if [ ! -s "${DATASET_DIR}/${filename}" ]; then
            echo -e "${RED}Error: Downloaded file ${filename} is empty${NC}"
            cleanup "${DATASET_DIR}/${filename}"
        fi
    else
        echo -e "${RED}Failed to download ${filename}${NC}"
        cleanup "${DATASET_DIR}/${filename}"
    fi
}

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    # Split the dataset string into ID and filename
    IFS="|" read -r file_id filename <<< "$dataset"
    
    # Check if file already exists
    if [ -f "${DATASET_DIR}/${filename}" ]; then
        echo -e "${YELLOW}${filename} already exists in ${DATASET_DIR}. Skipping...${NC}"
        continue
    fi
    
    download_dataset "$file_id" "$filename"
done

echo -e "${GREEN}All datasets have been downloaded to ${DATASET_DIR}/${NC}"

# Print instructions for adding more datasets
echo -e "\n${YELLOW}To add more datasets, edit this script and add entries to the DATASETS array in the format:${NC}"
echo -e "${YELLOW}\"GOOGLE_DRIVE_FILE_ID|FILENAME\"${NC}"
