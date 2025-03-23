# 🚀 Streamlit App Setup Guide

Follow these steps to set up a virtual environment and run your Streamlit app.

## 🔧 1. Create a Virtual Environment
```bash
python -m venv venv
```

## ⚡️ 2. Activate the Virtual Environment

### macOS/Linux
```bash
source venv/bin/activate
```

### Windows
```bash
venv\Scripts\activate
```

## 📦 3. Install Dependencies

If you already have a `requirements.txt` file:
```bash
pip install -r requirements.txt
```

If not, install Streamlit manually:
```bash
pip install streamlit
```

(Optional) To generate a `requirements.txt` after installing packages:
```bash
pip freeze > requirements.txt
```

## ▶️ 4. Run the Streamlit App

Make sure you're in the same directory as your `app.py`, then run:
```bash
streamlit run app.py
```

## 🛑 5. Deactivate the Virtual Environment

When you're done, you can deactivate it with:
```bash
deactivate
```
