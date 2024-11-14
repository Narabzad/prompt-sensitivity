#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# File information
FILE_ID="108BlwzDom4gq9A66Wbv4r5IMHu-GMYMv"
ZIP_FILE="pre-trained_text_classification_models.zip"
EXTRACT_DIR="baselines/text_classification"

echo -e "${YELLOW}Starting download of pre-trained models...${NC}"

# Function to clean up on error
cleanup() {
    echo -e "${RED}Error occurred. Cleaning up...${NC}"
    rm -f "$ZIP_FILE"
    exit 1
}

# Set up error handling
trap cleanup ERR

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo -e "${RED}pip is not installed. Please install Python and pip first.${NC}"
    exit 1
fi

# Check if gdown is installed, install if not
if ! command -v gdown &> /dev/null; then
    echo -e "${YELLOW}Installing gdown...${NC}"
    pip install gdown
fi

# Create directories if they don't exist
mkdir -p "$EXTRACT_DIR"

# Download the file using gdown
echo -e "${YELLOW}Downloading model files...${NC}"
gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$ZIP_FILE" || cleanup

# Check if file was downloaded successfully and is a valid zip file
if [ ! -f "$ZIP_FILE" ]; then
    echo -e "${RED}Failed to download $ZIP_FILE${NC}"
    cleanup
fi

# Test if the file is a valid zip archive
if ! unzip -t "$ZIP_FILE" > /dev/null 2>&1; then
    echo -e "${RED}Downloaded file is not a valid zip archive. Please check the Google Drive link and try again.${NC}"
    cleanup
fi

# Extract the zip file
echo -e "${YELLOW}Extracting model files to $EXTRACT_DIR...${NC}"
unzip -o "$ZIP_FILE" -d "$EXTRACT_DIR" || cleanup

# Clean up zip file
echo -e "${YELLOW}Cleaning up...${NC}"
rm -f "$ZIP_FILE"

echo -e "${GREEN}Successfully downloaded and extracted pre-trained models!${NC}"
echo -e "${GREEN}Models are now available in $EXTRACT_DIR${NC}"