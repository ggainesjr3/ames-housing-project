# 🏗️ Ames Housing Regression Engine v1.1

## Project Overview
A high-performance machine learning pipeline designed to predict residential home prices in Ames, Iowa. This project moves beyond simple regression by implementing **Defensive Engineering** patterns and a **Log-Normal** transformation strategy to handle high-value outliers in the real estate market.

**Current Performance:** * **Log-RMSE:** 0.0498 (Verified on rosencrantz)
* **Status:** Operational / Live Dashboard Integrated

---

## 🛠️ Technical Architecture

### 1. The Preprocessing Pipeline (The Bouncer)
The ingestion layer is built with a "Bouncer Check" philosophy. It doesn't just load data; it validates the integrity of the stream:
* **Ordinal Mapping:** Converts qualitative assessments (`Ex`, `Gd`, `TA`, `Fa`) into a quantitative hierarchy ($0-5$).
* **Logarithmic Scaling:** Applies $log(1+p)$ to the target price. This normalizes the distribution, ensuring that $500k luxury properties don't disproportionately skew the model's weightings against $150k starter homes.
* **Defensive Sanitization:** Handles "dirty data" by mapping missing categorical values to a baseline `None` state rather than allowing `NaN` to crash the training gate.

### 2. Model Architecture (The Engine)
* **Core:** `XGBoost` (Extreme Gradient Boosting)
* **Optimization:** Implemented **Early Stopping** logic (50 rounds) to prevent over-fitting during high-iteration training.
* **Serialization:** The trained "Brain" is serialized via `joblib` into `ames_model.pkl` for instant deployment without re-training.

---

## 📂 Project Structure
```text
.
├── app.py                # Streamlit Live Dashboard
├── train_model.py        # Automated training & serialization pipeline
├── src/
│   └── model.py          # Core processing & XGBoost logic
├── data/                 # Local data storage (Git ignored)
├── requirements.txt      # Dependency Manifest
└── README.md             # Project Documentation

🚀 Deployment & Usage
Prerequisites

    Environment: Linux (rosencrantz)

    Language: Python 3.10+

    Stack: xgboost, pandas, streamlit, scikit-learn

Installation
Bash

# Clone the repository
git clone [https://github.com/ggainesjr3/ames-housing-project.git](https://github.com/ggainesjr3/ames-housing-project.git)
cd ames-housing-project

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Operation

To re-train the model and update the serialized brain:
Bash

python3 train_model.py

To launch the tactical dashboard:
Bash

streamlit run app.py

⚖️ The Gaines Philosophy

As an engineer and bartender, I build software that treats data like a high-volume shift:

    Clean Workspace: Preprocessing is non-negotiable.

    Clear the Line: Early stopping prevents wasted compute.

    The Final Pour: Accuracy is the only metric that pays the bills.

Developed on rosencrantz.