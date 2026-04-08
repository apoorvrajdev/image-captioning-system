# 🖼️ Image Captioning System (CNN + Transformer)

## 📌 Overview
This project is an end-to-end deep learning system that generates natural language captions from images using a CNN-based encoder and a Transformer-based decoder.

---

## 🧠 Problem Statement
Automatically generate human-like captions for images using deep learning. This is useful in accessibility, search, and content understanding systems.

---

## ⚙️ Architecture

- **Encoder:** CNN (InceptionNet) for feature extraction  
- **Decoder:** Transformer for sequence generation  
- **Attention Mechanism:** Helps model focus on relevant parts of the image  

---

## 🔄 Pipeline

Image → CNN Encoder → Feature Vector → Transformer Decoder → Caption

---

## 📊 Results

- BLEU Score: ~24  
- Model generates context-aware captions for unseen images  

---

## 🧪 Dataset

- COCO 2017 Dataset  
- Preprocessing:
  - Tokenization  
  - Vocabulary building  
  - Padding sequences  

---

## 📸 Sample Outputs

![Sample Outputs](assets/sample_outputs.png)

---

## 🧱 Architecture Diagram

![Architecture](assets/architecture.png)

---

## 🚀 Live Notebook (Kaggle)

👉 [https://kaggle.com/YOUR-NOTEBOOK-LINK](https://www.kaggle.com/code/apoorvujjwal/image-captioning-using-ai)

- Full training pipeline  
- GPU-enabled  
- Reproducible workflow  

---

## ⚡ Improvements

- Replace CNN with Vision Transformer (ViT)  
- Use pre-trained models like BLIP / CLIP  
- Optimize inference with TensorRT  
- Deploy using FastAPI + Docker  

---

## 🏗️ Future Work

- Real-time captioning API  
- Multi-GPU distributed training  
- Low-latency production deployment  

---

## 🧑‍💻 Tech Stack

- Python  
- TensorFlow / PyTorch  
- CNN, Transformer  
- NumPy, Pandas  

---
