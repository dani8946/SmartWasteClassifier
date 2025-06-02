# 🗑️ Smart Waste Classifier

An intelligent waste classification system that uses deep learning to identify waste types such as plastic, paper, metal, glass, cardboard, and trash from images. Built using MobileNetV2 and Streamlit, this project aims to support smarter recycling and waste management.

---

 🔍 Overview

> 🚮 Upload an image of waste and the model will tell you if it's **plastic**, **paper**, **metal**, **trash**, **glass**, or **cardboard**.  
> 🎯 Powered by Transfer Learning with MobileNetV2 .

---

 🧠 Tech Stack

- Python, TensorFlow, Keras
- MobileNetV2 (Transfer Learning)
- OpenCV for image processing
- TrashNet dataset

---

 🚀 Features

- 🧠 High-accuracy image classification with MobileNetV2
- 📤 Upload trash image and get prediction with class probability
- 📊 Confusion matrix & performance metrics

---

 📦 How to Use

1. **Clone the Repository**
```bash
git clone https://github.com/dani8946/SmartWasteClassifier.git
cd SmartWasteClassifier
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the App**
```bash
streamlit run app.py
```

4. **Upload a Trash Image**  
View predicted class and confidence score instantly!

---

 📁 Dataset

This model was trained on the **TrashNet** dataset.
- Categories: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- [🔗 Dataset Source](https://github.com/garythung/trashnet)

---

## 🎯 Model Performance

- ✅ Accuracy: 73.96%
- 📈 Precision: 78.79%
- 📉 Recall: 70.18%

---

🙌 Acknowledgements

- [TrashNet Dataset by Gary Thung](https://github.com/garythung/trashnet)

---
