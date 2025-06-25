# 🩺 PheumoScan

**PheumoScan** is an AI-powered diagnostic tool that detects **pneumonia** from chest X-ray images using deep learning. The system includes **Grad-CAM** visualizations to show where the model is focusing, and it gives voice feedback in **English**, **Hausa**, and **Pidgin**, making it accessible and multilingual.

---

## 🚀 Key Features

- 🔍 **CNN-based Pneumonia Detection** trained on PneumoniaMNIST
- 🎯 **Grad-CAM Visual Explanations** to highlight areas of interest
- 🗣️ **Voice Feedback** using gTTS (English, Hausa, and Pidgin)
- 🌐 **Gradio Interface** for simple and fast interaction
- 📱 Lightweight and ready for low-resource deployment

---

## 🧪 Dataset

- **Name:** PneumoniaMNIST  
- **Source:** [MedMNIST](https://medmnist.com/)
- **Images:** 28x28 grayscale chest X-rays  
- **Classes:** 0 = Normal, 1 = Pneumonia

---

## 🛠️ Tech Stack

- `PyTorch` for deep learning
- `Gradio` for interactive web interface
- `gTTS` for multilingual voice output
- `OpenCV` and `matplotlib` for image processing
- `pytorch-grad-cam` for visualizing CNN focus areas

---

## ⚙️ Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/pheumoscan.git
cd pheumoscan
