
# 🌾 Wheat Disease Detection using Transfer Learning

This project is a web-based **wheat leaf disease detection app** built with **Streamlit** and **Keras (TensorFlow)**.  
Users can upload an image of a wheat leaf, and the trained model predicts whether it is healthy or affected by a specific disease.

---

## 🚀 Features

- Upload wheat leaf images directly through the web interface  
- Real-time prediction using a pre-trained deep learning model (`model.h5`)  
- Clean and simple Streamlit UI  
- Lightweight and deployable on **Streamlit Cloud**

---

## 🧠 Model Overview

The model is built using **Transfer Learning** with a pretrained CNN such as **MobileNetV2**, **EfficientNet**, or **ResNet50**.  
It has been fine-tuned on a custom dataset of wheat leaf disease images to classify multiple disease types.

---

## 📁 Project Structure

📦 wheat-disease-classification/
┣ 📜 app.py
┣ 📜 model.h5
┣ 📜 requirements.txt
┣ 📜 runtime.txt
┗ 📜 README.md

yaml
Copy code

---

## ⚙️ Installation (Run Locally)

### 1. Clone this repository
```bash
git clone https://github.com/your-username/wheat-disease-classification.git
cd wheat-disease-classification
2. Create a virtual environment (optional but recommended)
bash
Copy code
python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate         # For Windows
3. Install dependencies
bash
Copy code
pip install -r requirements.txt
4. Run the Streamlit app
bash
Copy code
streamlit run app.py