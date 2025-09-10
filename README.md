# Mini Opinion Analyzer

A lightweight sentiment analysis tool that classifies short text samples (movie reviews and news-style headlines) as **Positive**, **Negative**, or **Neutral**, and visualizes the sentiment distribution.

---

## 🔧 Features

* Collects a small dataset of reviews and headlines.
* Preprocesses text (lowercasing, punctuation/stopword removal, tokenization).
* Uses **TF-IDF vectorization + Logistic Regression** for sentiment classification.
* Outputs predictions for each text sample.
* Generates a **bar chart** showing sentiment distribution.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/malik8154/Mini-Opinion-Analyzer.git
cd Mini-Opinion-Analyzer
```

Set up a virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate   # On Windows PowerShell
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the analyzer:

```bash
python analyzer.py
```

## 📊 Example Visualization

Bar chart showing distribution of predicted sentiments:

![Sentiment Chart](chart.png)

---

## 📂 Project Structure

```
Mini-Opinion-Analyzer/
│── analyzer.py          # Main script
│── requirements.txt     # Required Python packages
│── README.md            # Project documentation
│── LICENSE              # Open-source license (MIT, Apache, etc.)
│── chart.png            # Output sentiment distribution chart (generated after run)

```

---

## 🛠 Requirements

* Python 3.8+
* nltk
* scikit-learn
* pandas
* matplotlib

---

## ✨ Author

Developed by **Malik M Shahmeer Rashid** 🚀

---
