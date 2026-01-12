PCB DIFFERENTIAL DEFECT DETECTION SYSTEM

A deep learning–based PCB defect detection system that combines classical image comparison techniques with CNN-based classification to accurately identify manufacturing defects on Printed Circuit Boards (PCBs).
This project is developed as part of an industry internship and focuses on building an end-to-end AI pipeline including model training, inference, and a user-friendly web interface.
------------------------------------------------------------------------------------------------------------------------------------------------------------------
KEY FEATURES

1. Differential Detection using Golden Reference PCB images
2. Deep Learning Classification using CNN (EfficientNet / ResNet)
3. Detects multiple PCB defects:
   A) Missing Hole
   B) Mouse Bite
   C) Open Circuit
   D) Short
   E) Spur
   F) Spurious Copper
4. High model accuracy (~99%)
5. Interactive Web Interface using Streamlit
6. Modular & production-ready code structure
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
SYSTEM ARCHITECTURE

The system follows a two-stage hybrid approach:
1️. Differential Comparison (Classical Computer Vision)
   a) Perceptual Hashing (pHash)
   b) Structural Similarity Index (SSIM)
   c) Identifies suspicious regions by comparing input PCB with a golden reference PCB

2️. Deep Learning Classification
   a) CNN-based model (EfficientNet-B0 / ResNet-50)
   b) Classifies the detected abnormal regions into defect categories
----------------------------------------------------------------------------------------------------------------------------------------------------------------
TECH STACK

| Category         | Tools                      |
| ---------------- | -------------------------- |
| Programming      | Python                     |
| Deep Learning    | PyTorch                    |
| Models           | EfficientNet-B0, ResNet-50 |
| Computer Vision  | OpenCV, SSIM               |
| Web Framework    | Streamlit                  |
| Image Processing | PIL, torchvision           |
| Deployment       | Local (Windows/Linux)      |
----------------------------------------------------------------------------------------------------------------------------------------------------------------
PROJECT STRUCTURE

Website/
├── app.py                     # Streamlit web application
├── inference_new.py           # Inference & detection pipeline

Data prep
|── image processing.ipynb     # Data prep & image processing

Model/
|── model.ipynb                # Model Training
|── Inference.ipynb            # Model Testing

|── README.md
----------------------------------------------------------------------------------------------------------------------------------------------------------------
INSTALLATION & SETUP

1. Clone the repository:
   git clone https://github.com/yuvraj-makani/pcb-defect-detection.git
   cd pcb-defect-detection

2. Install Dependencies:
   pip install -r requirements.txt

Required packages:
torch
torchvision
streamlit
numpy
matplotlib
scikit-image
imagehash

3. How to run the application:
   streamlit run app.py
----------------------------------------------------------------------------------------------------------------------------------------------------------------
HOW DETECTION WORKS

1. Upload a PCB image
2. System finds the best matching golden PCB
3. SSIM highlights anomalous regions
4. CNN classifies defect type
5. Results are displayed with bounding boxes & confidence scores
----------------------------------------------------------------------------------------------------------------------------------------------------------------
MODEL DETAILS

Training Type: Supervised Learning
Loss Function: Cross-Entropy Loss
Optimizer: Adam
Input Size: 224×224 (cropped patches)
Accuracy: ~98–99% on validation set
----------------------------------------------------------------------------------------------------------------------------------------------------------------
KNOWN LIMMITATIONS

Model performance depends on alignment with golden PCB
Dataset imbalance may bias predictions
Industrial lighting variations can affect SSIM
----------------------------------------------------------------------------------------------------------------------------------------------------------------
FUTURE IMPROVEMENTS

Replace sliding-window approach with YOLO-based detection
Add real-time camera inspection support
Deploy as cloud-based API
Improve robustness with data augmentation
----------------------------------------------------------------------------------------------------------------------------------------------------------------
AUTHOR

Yuvraj Makani
AI/ML Intern
PCB Defect Detection | Computer Vision | Deep Learning
----------------------------------------------------------------------------------------------------------------------------------------------------------------
LICENSE

This project is intended for academic and research purposes.
For commercial use, please contact the author.
----------------------------------------------------------------------------------------------------------------------------------------------------------------
