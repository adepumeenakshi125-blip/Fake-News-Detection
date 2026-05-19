# 📰 Fake News Detection System

A Machine Learning based web application that detects whether a news article is REAL or FAKE using Natural Language Processing (NLP) techniques.

---

# 🚀 Features

- Detects fake and real news
- Interactive Streamlit web interface
- Machine Learning model using Logistic Regression
- TF-IDF text vectorization
- Confidence score prediction
- Simple and beginner-friendly project

---

# 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- NLP (TF-IDF Vectorizer)

---

# 📂 Project Structure

```text
fake-news-detector/
│
├── app.py
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── dataset/
    └── news.csv
```

---

# 📊 Dataset

Dataset used:
Fake and Real News Dataset from Kaggle

The dataset contains:
- News title
- News content
- Labels (REAL / FAKE)

---

# ⚙️ Installation

## Step 1: Clone Repository

```bash
git clone https://github.com/your-username/fake-news-detector.git
```

## Step 2: Open Project Folder

```bash
cd fake-news-detector
```

## Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Model Training

```bash
python train_model.py
```

This will generate:
- model.pkl
- vectorizer.pkl

---

# ▶️ Run Streamlit App

```bash
streamlit run app.py
```

---

# 🧠 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
TF-IDF Vectorization
   ↓
Model Training
   ↓
Prediction
   ↓
Web Application
```

---

# 📈 Model Used

- Logistic Regression

---

# 🎯 Future Improvements

- Add Deep Learning models
- Improve UI design
- Add news category detection
- Add live news API integration
- Deploy online

---

# 👩‍💻 Author

Adepu Meenakshi

---

# ⭐ If you like this project

Give it a star on GitHub!