# fake-news-detection
# 📰 Fake News Detection using Machine Learning

## 📌 Project Overview

This project is a **Fake News Detection System** built using Machine Learning.
It classifies news articles as **FAKE** or **REAL** based on their content.

The model uses:

* TF-IDF Vectorization
* SGD Classifier (Passive Aggressive equivalent)

---

## 🚀 Features

* Clean and preprocess news text
* Convert text into numerical features using TF-IDF
* Train a machine learning model
* Predict whether a news article is FAKE or REAL
* Interactive user input for real-time prediction

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Regular Expressions

---

## 📂 Dataset

Dataset used from:

* Kaggle Fake and Real News Dataset

Files:

* `Fake.csv`
* `True.csv`

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn
```

### 3. Prepare dataset

Download dataset from Kaggle and place:

* `Fake.csv`
* `True.csv`

Then run:

```bash
python merge_data.py
```

This will generate:

```
news.csv
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 💡 Example Usage

```
Enter news (type exit to stop):
Government announces new policy

Prediction: REAL
```

---

## 📊 Model Details

* Vectorizer: TF-IDF
* Model: SGD Classifier (hinge loss)
* Accuracy: ~90% (depends on dataset)

---

## 📈 Future Improvements

* Add GUI using Tkinter / Streamlit
* Deploy as a web application
* Improve preprocessing with NLP techniques
* Use deep learning models (LSTM, BERT)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request

---

## 🙌 Acknowledgements

* Kaggle for dataset
* Scikit-learn for ML tools
