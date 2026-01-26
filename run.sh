#!/bin/bash

# Skin Cancer Detection - Main Run Script
# This script handles setup and execution in one command

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        Skin Cancer Detection - AI Application          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}\n"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed!${NC}"
    echo -e "${BLUE}Please install Python 3.9+ and try again.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found.${NC}"
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv venv
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Virtual environment created${NC}"
    else
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${GREEN}✓ Activating virtual environment...${NC}"
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" &> /dev/null; then
    echo -e "${YELLOW}⚠️  Dependencies not installed.${NC}"
    echo -e "${BLUE}Installing required packages...${NC}"
    pip install --upgrade pip -q
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
    else
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        exit 1
    fi
fi

# Check if Dataset exists
if [ ! -d "Dataset" ]; then
    echo -e "${RED}❌ Dataset folder not found!${NC}"
    echo -e "${BLUE}Please download the dataset from Google Drive and place it in the 'Dataset' folder.${NC}"
    echo -e "${BLUE}Dataset link: https://drive.google.com/drive/folders/1xO8lKv2eVZtjN-6uXAiGumiDM9JLMkq7?usp=drive_link${NC}"
    echo -e "${YELLOW}The Dataset folder should contain: AK/, BCC/, and SK/ subfolders${NC}\n"
    exit 1
fi

# Check if model file exists
if [ ! -f "skin_model.pth" ]; then
    echo -e "${YELLOW}⚠️  Trained model 'skin_model.pth' not found!${NC}"
    echo -e "${BLUE}The model needs to be trained first (one-time process).${NC}"
    echo -e "${YELLOW}This will take approximately 10-30 minutes.${NC}\n"
    
    read -p "Do you want to train the model now? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "\n${BLUE}🚀 Starting model training...${NC}\n"
        python train_model.py
        
        if [ $? -eq 0 ]; then
            echo -e "\n${GREEN}✅ Model training completed successfully!${NC}\n"
        else
            echo -e "\n${RED}❌ Model training failed!${NC}"
            echo -e "${BLUE}Please check the error messages above.${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Cannot run the app without a trained model.${NC}"
        echo -e "${BLUE}To train the model later, run:${NC}"
        echo -e "${BLUE}  source venv/bin/activate${NC}"
        echo -e "${BLUE}  python train_model.py${NC}"
        exit 1
    fi
fi

# Run Streamlit app
echo -e "${GREEN}✓ Model file found!${NC}"
echo -e "${GREEN}✓ Launching Streamlit app...${NC}"
echo -e "${BLUE}The app will open in your browser automatically.${NC}"
echo -e "${BLUE}Press Ctrl+C to stop the server.${NC}\n"
echo -e "${YELLOW}═══════════════════════════════════════════════════════${NC}\n"

streamlit run app.py
