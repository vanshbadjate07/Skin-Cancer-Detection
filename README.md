# Skin Cancer Detection using EfficientNet and GradCAM

A Deep Learning based Skin Cancer Classification system that detects **AK (Actinic Keratosis)**, **BCC (Basal Cell Carcinoma)**, and **SK (Seborrheic Keratosis)** using dermoscopic images.
The project also includes a **Streamlit UI** that shows:

* Predicted class
* Confidence score
* GradCAM heatmap highlighting cancer regions

This project was fully built and tested on **Google Colab**.

---

## Features

* Train a CNN model using **EfficientNet-B0 (Transfer Learning)**
* Predict 3 skin cancer types
* Compute **confidence percentage**
* Generate **GradCAM heatmaps** to visualize "where" the model is looking
* Clean Streamlit UI for image upload and instant prediction
* Very beginner friendly, structured, and reproducible

---

## Project Structure

```
Skin_Cancer/
│
├── Skin_Cancer_Detection_(AK,_SK,_BCC).ipynb   # Main Colab notebook
├── streamlit_ui_skin_cancer.py                 # Streamlit App
├── skin_model.pth                              # Saved model (not pushed to GitHub)
├── README.md                                   # Project documentation
└── requirements.txt (optional)
```

---

## Dataset

This project uses a 3-class dermoscopy image dataset:

* **AK:** 288 images
* **BCC:** 338 images
* **SK:** 328 images

You must download the dataset manually because it is too large for GitHub.

### Dataset Link (Google Drive)
📂 [Click Here](https://drive.google.com/drive/folders/1xO8lKv2eVZtjN-6uXAiGumiDM9JLMkq7?usp=sharing)


---

## How to Run the Project (Google Colab Recommended)

### Step 1: Open the Notebook

Upload the `.ipynb` file to Google Colab.

### Step 2: Mount Google Drive

Modify path:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Update Dataset Path

Place dataset in:

```
/content/drive/MyDrive/SkinCancerData/
```

Folder structure:

```
SkinCancerData/
  ├── AK/
  ├── BCC/
  └── SK/
```

### Step 4: Run All Cells

This will:

* Load dataset
* Train the model
* Save the model as `skin_model.pth`
* Prepare everything for UI

---

## Running Streamlit App in Colab

Install Cloudflare tunnel:

```python
!pip install streamlit
!wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb
```

Run Streamlit:

```python
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &>/dev/null &
```

Start Cloudflare Tunnel:

```python
!cloudflared tunnel --url http://localhost:8501 --no-autoupdate
```

You will receive a public URL.
Open it to use your app.

---

## Streamlit App (UI)

When launched, the app provides:

### • Original Uploaded Image

### • Predicted Cancer Class

### • Confidence Percent

### • GradCAM Heatmap

All side-by-side with clean layout.

---

## Model Architecture

* EfficientNet-B0 backbone
* Final classifier changed to 3 output nodes
* Optimizer: Adam
* Loss: CrossEntropy
* Input Size: 224x224

---

## GradCAM Explainability

GradCAM highlights the exact regions the model focuses on.
Useful for:

* Medical explainability
* Debugging wrong predictions
* Trust building

---

## Future Improvements

* Add lesion segmentation
* Deploy on HuggingFace Spaces
* Add multi-class ROC/AUC
* Add batch image prediction

---

## Disclaimer

This project is for **educational and research purposes only**.
It is **NOT a medical diagnostic tool**.
