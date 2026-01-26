# 🔬 Skin Cancer Detection with Deep Learning

An AI-powered web application for detecting skin cancer types using **EfficientNet-B0** and **GradCAM** visualization. This system classifies dermoscopic images into three categories: **AK (Actinic Keratosis)**, **BCC (Basal Cell Carcinoma)**, and **SK (Seborrheic Keratosis)**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)

---

## ✨ Features

- 🧠 **Deep Learning Model**: EfficientNet-B0 with transfer learning from ImageNet
- 🎯 **3-Class Classification**: Detects AK, BCC, and SK skin conditions
- 🔥 **GradCAM Visualization**: Shows which image regions the model focuses on
- 📊 **Confidence Scores**: Displays prediction confidence percentages
- 🖥️ **Interactive Web UI**: Clean Streamlit interface for easy image upload
- 📈 **Training Progress**: Real-time training metrics with progress bars

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- 2GB+ free disk space
- (Optional) GPU for faster training

### Installation & Running

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vanshbadjate07/Skin-Cancer-Detection.git
   cd Skin-Cancer-Detection
   ```

2. **Download the Dataset**:
   
   📂 **[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1xO8lKv2eVZtjN-6uXAiGumiDM9JLMkq7?usp=drive_link)**
   
   After downloading, extract and place the dataset in the project directory:
   ```
   Skin-Cancer-Detection/
   └── Dataset/
       ├── AK/     # Actinic Keratosis images
       ├── BCC/    # Basal Cell Carcinoma images
       └── SK/     # Seborrheic Keratosis images
   ```

3. **Make the run script executable**:
   ```bash
   chmod +x run.sh
   ```

4. **Run the application**:
   ```bash
   ./run.sh
   ```

That's it! The script will:
- ✅ Create a virtual environment (if needed)
- ✅ Install all dependencies automatically
- ✅ Train the model on first run (one-time, ~10-30 minutes)
- ✅ Launch the Streamlit web app

**Note**: The model trains only once. Subsequent runs will use the saved model and start immediately.

---

## 📁 Project Structure

```
Skin-Cancer-Detection/
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── run.sh                  # Main execution script (all-in-one)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── skin_model.pth         # Trained model (generated after training)
└── Dataset/               # Training dataset (download separately)
    ├── AK/                # ~226 images
    ├── BCC/               # ~193 images
    └── SK/                # ~327 images
```

---

## 🎨 Using the Web Application

1. **Launch the app** using `./run.sh`
2. **Upload an image**:
   - Click "Browse files" or drag & drop
   - Supported formats: JPG, JPEG, PNG
3. **View results**:
   - **Left panel**: Original uploaded image
   - **Right panel**: GradCAM heatmap showing model attention
   - **Bottom**: Prediction class and confidence percentage

---

## 📊 Dataset Information

The dataset contains dermoscopic images organized into three classes:

- **AK (Actinic Keratosis)**: Pre-cancerous skin lesions (~226 images)
- **BCC (Basal Cell Carcinoma)**: Most common type of skin cancer (~193 images)
- **SK (Seborrheic Keratosis)**: Benign skin growths (~327 images)

**Total Images**: ~746 images

📂 **[Download Dataset from Google Drive](https://drive.google.com/drive/folders/1xO8lKv2eVZtjN-6uXAiGumiDM9JLMkq7?usp=drive_link)**

---

## 🎓 Model Training

### Automatic Training (Recommended)

The first time you run `./run.sh`, it will automatically prompt you to train the model if it doesn't exist.

### Manual Training

```bash
source venv/bin/activate
python train_model.py
```

**Training Configuration**:
- **Epochs**: 10
- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Train/Val/Test Split**: 70%/15%/15%
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

**Expected Performance**:
- **Validation Accuracy**: ~80%
- **Test Accuracy**: ~82%

**Training Time**:
- CPU: 20-40 minutes
- GPU: 5-15 minutes

**Important**: The model trains only once. After training, the model file (`skin_model.pth`) is saved and reused for all future runs.

---

## 🔧 Model Architecture

- **Base Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Input Size**: 224×224 pixels
- **Output Classes**: 3 (AK, BCC, SK)
- **Final Layer**: Linear(1280 → 3)
- **Activation**: Softmax for probability distribution

### GradCAM Explainability

GradCAM (Gradient-weighted Class Activation Mapping) highlights the regions of the image that most influenced the model's decision. This provides:
- **Transparency**: See what the model is "looking at"
- **Trust**: Verify the model focuses on relevant features
- **Debugging**: Identify potential issues with predictions

---

## 📦 Dependencies

Core libraries (see `requirements.txt` for full list):
- `torch==2.2.0` - Deep learning framework
- `torchvision==0.17.0` - Computer vision utilities
- `streamlit==1.28.0` - Web application framework
- `opencv-python==4.8.1.78` - Image processing
- `Pillow==10.2.0` - Image handling
- `numpy==1.26.4` - Numerical operations
- `tqdm==4.66.1` - Progress bars

---

## 🛠️ Troubleshooting

### Dataset not found
**Error**: `Dataset folder not found`

**Solution**: Download the dataset from Google Drive and place it in the project directory:
```bash
# Your folder structure should look like:
Skin-Cancer-Detection/
└── Dataset/
    ├── AK/
    ├── BCC/
    └── SK/
```

### Model training failed
**Solution**: Ensure you have enough disk space and the Dataset folder is properly set up. Check error messages for specific issues.

### Permission denied on run.sh
**Solution**: Make the script executable:
```bash
chmod +x run.sh
```

### Memory errors during training
**Solution**: Reduce batch size in `train_model.py`:
```python
BATCH_SIZE = 8  # Reduce from 16 to 8
```

---

## 📝 Command Reference

| Command | Description |
|---------|-------------|
| `./run.sh` | **Main command** - Sets up environment and runs the app |
| `python train_model.py` | Train the model manually (after activating venv) |
| `streamlit run app.py` | Run the web app directly (after setup) |
| `source venv/bin/activate` | Activate virtual environment |
| `deactivate` | Deactivate virtual environment |

---

## ⚠️ Important Disclaimers

> **Medical Disclaimer**: This application is for **educational and research purposes only**. It is **NOT** a medical diagnostic tool and should **NOT** be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

> **Accuracy Note**: Model performance depends on training data quality and quantity. Results should be interpreted with caution.

---

## 🚀 Future Improvements

- [ ] Add more skin cancer types (melanoma, etc.)
- [ ] Implement data augmentation for better generalization
- [ ] Add batch image processing
- [ ] Deploy to cloud (Hugging Face Spaces, Streamlit Cloud)
- [ ] Add lesion segmentation
- [ ] Implement ensemble models
- [ ] Add ROC/AUC curves and confusion matrix
- [ ] Support for video input

---

## 📄 License

This project is provided for educational purposes.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---

## 📧 Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Ensure the Dataset folder is properly set up
3. Verify Python 3.9+ is installed
4. Check that all dependencies are installed

---

**Made with ❤️ using PyTorch, Streamlit, and EfficientNet**
