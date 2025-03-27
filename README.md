
# 🐶 Dog Breed & Age Identification with Health Recommendations

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-green)](https://24zphikhhtygx4jsh7r6ze.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project provides an intelligent image classification tool that predicts **dog breed** (from 120 classes) and **dog age group** (Adult, Senior, Young) using deep learning models. It also recommends healthcare guidelines based on the predicted breed or age category.

---

## 🚀 Live Demo

👉 [Try the App on Streamlit](https://24zphikhhtygx4jsh7r6ze.streamlit.app/)

---

## 📸 Sample Output (Demo)

After uploading a dog image:

- 🐕 **Breed Prediction**: Labrador Retriever (90.23%)  
  🩺 **Health Tip**: “Watch for hip dysplasia. Regular exercise and healthy diet recommended.”

- 👶 **Age Group**: Young (61.45%)  
  🩺 **Age Tip**: “Ensure timely vaccinations and training.”

---

## 📈 Model Training Performance

### ✅ Accuracy Over Epochs
![Accuracy](accuracy_plot.png)

### ✅ Loss Over Epochs
![Loss](loss_plot.png)

---

## 🧠 Model Overview

### ✅ Dog Breed Classifier
- **Architecture**: InceptionResNetV2 (Transfer Learning)
- **Training Accuracy**: 81.66%
- **Validation Accuracy**: 85.83%
- **Classes**: 120 dog breeds
- **Dataset**: [Stanford Dog Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

### ✅ Dog Age Classifier
- **Architecture**: EfficientNetB3
- **Training Accuracy**: ~54%
- **Validation Accuracy**: ~57%
- **Classes**: `Young`, `Adult`, `Senior`
- **Dataset**: [DogAge Dataset (Kaggle)](https://www.kaggle.com/datasets/user164919/the-dogage-dataset)

---

## 🩺 Health Recommendations

After prediction, the system provides **custom healthcare advice**:
- **Breed-based**: Common health concerns and care tips (e.g., hip dysplasia, grooming needs).
- **Age-based**: Preventive care suggestions for different life stages.

---

## 📁 Repository Structure

```
.
├── app.py                          # Streamlit app (UI and model inference)
├── dog_breed_classifier_final.keras
├── dog_age_classifier.keras
├── breed_class_indices.json       # Maps breed index to class name and health info
├── age_class_indices.json         # Maps age group index to label and health info
├── requirements.txt               # Dependencies
```

---

## 🛠️ Installation & Usage

### 1. Clone the Repo
```bash
git clone https://github.com/Luaim/Doge-breed-and-age-identification-.git
cd Doge-breed-and-age-identification-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
streamlit run app.py
```

---

## 📌 Datasets Used

- [Stanford Dog Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) – For breed classification
- [DogAge Dataset on Kaggle](https://www.kaggle.com/datasets/user164919/the-dogage-dataset) – For age classification

---

## 💡 Future Enhancements

- Improve age classifier accuracy with better age-labeled data
- Integrate object detection to crop dog faces automatically
- Add breed facts and history in UI
- Connect to vet APIs for real-time medical advice

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

- [Stanford Vision Lab](http://vision.stanford.edu/)
- [Kaggle Community](https://www.kaggle.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
