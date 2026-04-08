# 🖼️ Image Captioning System (CNN + Transformer)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN-red)
![NLP](https://img.shields.io/badge/NLP-Transformer-green)
![Dataset](https://img.shields.io/badge/Dataset-COCO-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

This project builds an **AI-powered image captioning system** that generates **natural language descriptions from images** using a hybrid **CNN + Transformer architecture**.

The system understands visual content and produces **context-aware captions**, bridging the gap between **computer vision and natural language processing**.

---

# 🚀 Live Demo

[![Open Notebook](https://img.shields.io/badge/Open%20Kaggle%20Notebook-GPU-blue)](https://www.kaggle.com/code/apoorvujjwal/image-captionin-using-dl)

OR explore the full pipeline here:

👉 https://www.kaggle.com/code/apoorvujjwal/image-captionin-using-dl

The notebook includes:

- End-to-end training pipeline  
- COCO dataset integration  
- Transformer-based caption generation  
- GPU-enabled execution  

---

# 📄 Research Paper (IEEE Publication)

This project is backed by an **IEEE published research paper**:

[![IEEE Paper](https://img.shields.io/badge/View%20Research%20Paper-IEEE-blue)](https://ieeexplore.ieee.org/document/10675203)

📄 **Title:** AI Narratives: Bridging Visual Content and Linguistic Expression  

---

### 🧠 Key Contributions

- Designed a hybrid **CNN + Transformer architecture** for image captioning  
- Leveraged **InceptionV3** for visual feature extraction  
- Implemented **attention-based sequence generation**  
- Achieved improved caption quality using **BLEU evaluation**  
- Compared multiple CNN backbones (VGG, ResNet, Inception)  

---

### 🚀 Why This Matters

- Demonstrates **research-level understanding of deep learning systems**  
- Shows ability to **design and evaluate AI architectures**  
- Validated on **real-world benchmark dataset (COCO)**  

# 🧠 Model Overview

The system uses a **two-stage architecture**:

### 🔹 Encoder (Vision)
- **InceptionV3 (CNN)**
- Extracts high-level spatial features from images
- Converts image → feature vector

### 🔹 Decoder (Language)
- **Transformer Decoder**
- Generates captions word-by-word using attention
- Captures long-range dependencies in text

---

# 🔄 Caption Generation Pipeline

Image → CNN Encoder → Feature Embeddings → Transformer Decoder → Caption

---

# 📸 Sample Outputs

### 🟢 Example 1
**Generated Caption:**  
`a man is standing on a beach with a surfboard`

*<img width="923" height="906" alt="image" src="https://github.com/user-attachments/assets/64e8412b-1d49-404c-a5b2-1da121b224e2" />
*

---

### 🟢 Example 2
**Generated Caption:**  
`a man riding a motorcycle on a street`
*<img width="832" height="857" alt="image" src="https://github.com/user-attachments/assets/c802d420-a1c1-48be-8e79-599f193c72cd" />
*

---

# 📊 Model Performance

The model was evaluated using **BLEU Score**, a standard NLP metric for text generation.

| Metric | Value |
|--------|------|
| BLEU Score | ~24 |

### Key Observations:
- Generates **semantically meaningful captions**
- Performs well on **common objects and scenes**
- Slight limitations on **complex multi-object scenes**

---

# 📂 Dataset

The model is trained on the **COCO 2017 Dataset**, a large-scale benchmark dataset for image captioning.

Dataset characteristics:

- 200,000+ images  
- 80 object categories  
- Multiple captions per image  
- Rich annotations for training  

---

# ⚙️ Deep Learning Pipeline

The project follows a complete deep learning workflow:

1. Image preprocessing (resize, normalization)
2. Feature extraction using InceptionV3
3. Caption preprocessing (tokenization, padding)
4. Vocabulary creation
5. Transformer model training
6. Loss optimization (Cross-Entropy)
7. Model evaluation using BLEU score
8. Inference on unseen images

---

# 🧰 Technologies Used

- Python  
- TensorFlow / Keras  
- CNN (InceptionV3)  
- Transformer Architecture  
- NumPy, Pandas  
- Matplotlib  
- Jupyter Notebook  

---

# 📁 Project Structure

```

image-captioning-system
│
├── image_captioning.ipynb
├── assets/
├── requirements.txt
└── README.md

---

# 🧪 Research Contribution

This project is based on an **IEEE research publication**:

📄 AI Narratives: Bridging Visual Content and Linguistic Expression

Key contributions:

- Integration of **CNN + Transformer architecture**
- Improved caption generation using **attention mechanisms**
- Comparative analysis of CNN encoders (VGG, ResNet, Inception)
- Enhanced tokenization strategies for better language modeling  

---

# ⚠️ Limitations

- Struggles with highly complex or cluttered scenes  
- May generate generic captions for rare objects  
- Requires large datasets and compute for training  

---

# 🚀 Future Improvements

- Replace CNN with **Vision Transformer (ViT)**  
- Use pretrained models like **BLIP / CLIP**  
- Optimize inference using **TensorRT / ONNX**  
- Deploy as **FastAPI-based real-time API**  
- Multi-GPU distributed training  

---

# 👨‍💻 Author

**Apoorv Raj**  
AI Systems Engineer | Deep Learning | ML Infrastructure  

---

⭐ If you found this project useful, consider giving it a **star** on GitHub.
