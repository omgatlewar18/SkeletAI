# 🦴 SkeletAI  
**AI-Powered Gender Estimation from Hand Radiographs**

---


## 🖥️ Live Demo  
Try it here 👉 [**skeletai.streamlit.app**](https://skeletai.streamlit.app)

---

## 🚀 Mission  

In forensic anthropology and post-disaster identification, determining gender from skeletal remains is a key step in restoring identity to victims.  
Traditional markers like the skull or pelvis are often unavailable — yet **hand bones remain more resilient**.  

**SkeletAI** bridges this gap using deep learning to estimate gender directly from hand X-rays, offering a **fast, reliable, and interpretable** solution for forensic experts and researchers.

---

## ✨ Key Features  

- 🧠 **High-Accuracy Model**  
  Built on **ResNet50** with transfer learning from ImageNet and fine-tuned on the **RSNA Bone Age dataset**, achieving robust performance through a two-stage training process.

- 🔍 **Explainable AI (XAI)**  
  Uses **Grad-CAM** heatmaps to visualize the bone regions (e.g., carpals, metacarpals) influencing predictions, enhancing interpretability and trust.

- 💻 **Interactive Web App**  
  Powered by **Streamlit**, providing a simple and intuitive interface — upload a hand X-ray, get an instant prediction with confidence scores.

---

## 🧩 Tech Stack  

| **Component** | **Technology** |
|----------------|----------------|
| **Model** | ResNet50 (Transfer Learning) |
| **Framework** | TensorFlow / Keras |
| **Visualization** | Grad-CAM |
| **Frontend** | Streamlit |
| **Dataset** | RSNA Bone Age Dataset |
| **Deployment** | Streamlit Cloud |

---

## 🧠 How It Works  

1. **Input:** Upload a hand X-ray (PNG/JPG).  
2. **Preprocessing:** Image is resized and normalized.  
3. **Prediction:** ResNet50 extracts features and classifies gender.  
4. **Explainability:** Grad-CAM highlights key bone regions driving the decision.  

---

## 🧑‍💻 Developer  
**Developed by:** Om Gatlewar  
**Email:** omgatlewar18@gmail.com 

---

## 🤝 Contributions  
Open for collaboration, improvements, and research partnerships.  
If you’d like to contribute, submit a PR or reach out via email.  

---
