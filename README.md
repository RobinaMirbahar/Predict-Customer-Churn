
# Predict Customer Churn

> **Author & Architect:** [Robina Mirbahar](https://github.com/RobinaMirbahar) (Google Developer Expert in Cloud & AI)  
> **Focus:** End-to-End Feature Engineering, Out-of-Fold Stacking (HistGradientBoosting, LightGBM, XGBoost, Logistic Regression) & Probability Calibration

---

## 🎯 Why I Built Customer Churn Prediction — Stacked Ensemble Pipeline (Engineering Purpose & Real-World Value)

### 1. 📈 Beating Single-Model Baselines with Out-of-Fold Stacking
Single gradient-boosted trees often overfit noisy customer telemetry. I engineered a multi-model stacked ensemble combining `HistGradientBoosting`, `LightGBM`, `XGBoost`, and a calibrated `LogisticRegression` meta-learner.

### 2. 🛠️ Domain-Driven Behavioral Feature Engineering
I designed interaction ratios, tenure velocity features, and service-adoption density signals that capture early churn precursors before cancellation occurs.

### 3. 🎯 Leak-Free Stratified Cross-Validation
Every transformation and out-of-fold prediction is strictly isolated inside stratified K-fold splits to guarantee production-grade generalization.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data ["1. Raw Telemetry & Feature Engineering"]
        D_Raw["Customer Demographics, Tenure & Billing Data"]
        D_FE["Interaction Ratios, Spend Velocity & Encoding"]
    end
    subgraph Base ["2. Stratified Out-of-Fold Base Learners (Level 0)"]
        M_LGB["LightGBM Classifier"]
        M_XGB["XGBoost Classifier"]
        M_HGB["HistGradientBoosting Classifier"]
    end
    subgraph Meta ["3. Meta-Learner & Calibration (Level 1)"]
        M_Meta["Logistic Regression Stacking Meta-Model"]
        M_Out["Calibrated Churn Probability & Retention Priority"]
    end
    D_Raw --> D_FE --> M_LGB & M_XGB & M_HGB
    M_LGB & M_XGB & M_HGB --> M_Meta --> M_Out
```

---

## Kaggle Playground Series – Season 6, Episode 3

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Playground%20S6E3-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/competitions/playground-series-s6e3)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

## 📝 Overview

This repository contains my solution for the **Kaggle Playground Series Season 6 Episode 3** competition. The goal is to predict the likelihood of customer churn based on a synthetically generated dataset that mimics real-world telecommunications customer behavior.

**Key details:**
- **Goal:** Predict probability of customer churn (binary classification)
- **Evaluation metric:** Area Under the ROC Curve (AUC)
- **Data source:** Synthetic dataset generated from real-world patterns
- **Competition status:** Active (deadline March 31, 2026)

## 📊 Dataset

The dataset includes typical customer information such as:
- **Demographics:** gender, senior citizen status, partner, dependents
- **Account details:** tenure, contract type, paperless billing, payment method
- **Services subscribed:** phone service, internet service, online security, backup, device protection, tech support, streaming TV/movies
- **Charges:** monthly charges, total charges

**Files:**
- `train.csv` – Training data with target column `Churn`
- `test.csv` – Test data (no target)
- `sample_submission.csv` – Example submission format

## 🎯 Goal

Build a model that predicts the probability of a customer churning (`Churn = 1`). Submissions are evaluated on **ROC‑AUC**.

## 🧠 Approach

The solution is implemented in a single Jupyter notebook: `churn_advanced.ipynb`. It covers:

- **Exploratory Data Analysis** – visualising churn rates and feature distributions
- **Feature Engineering** – creating interaction terms, ratios, flags, and transforms (40+ features)
- **Feature Selection** – using Random Forest importance to keep only the most relevant features
- **Base Models** – five diverse models:
  - HistGradientBoosting (two variants)
  - Logistic Regression
  - LightGBM
  - XGBoost
- **Hyperparameter Tuning** – with Optuna (optional)
- **Stacking Ensemble** – meta‑learner selection (Logistic, Ridge, Random Forest, LightGBM)
- **Ensemble Pruning** – removing highly correlated models
- **Visualization** – feature importance, model correlation, OOF AUC comparison

## 📈 Results

| Model | OOF AUC |
|-------|---------|
| HGB-A (depth=5) | ~0.915xx |
| HGB-B (depth=7) | ~0.915xx |
| Logistic Regression | ~0.898xx |
| LightGBM | ~0.916xx |
| XGBoost | ~0.916xx |
| **Stacked Ensemble** | **~0.917+** |

*Replace with your actual scores after running.*

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Required packages: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `lightgbm`, `xgboost`, `optuna`

### Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Predict-Customer-Churn.git
   cd Predict-Customer-Churn
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/competitions/playground-series-s6e3/data) and place `train.csv`, `test.csv`, and `sample_submission.csv` in the same folder as the notebook.

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (If you don't have a `requirements.txt`, you can manually install the packages listed above.)

4. Run the notebook:
   ```bash
   jupyter notebook churn_advanced.ipynb
   ```
   Or open it in your preferred IDE.

5. After executing all cells, the final submission file `submission.csv` will be created in the notebook directory.

## 📁 Repository Structure

```
Predict-Customer-Churn/
├── churn_advanced.ipynb   # Main solution notebook
├── README.md              # This file
├── requirements.txt       # Dependencies (optional)
└── LICENSE                # MIT License
```

## 📝 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Kaggle for hosting the competition and providing the synthetic dataset
- The Playground Series team for creating engaging challenges

---

*Happy modeling!* 🚀
```


---

### 🚀 Featured Google Cloud & Agentic AI Portfolio by Robina Mirbahar

| Repository | Architecture & Focus | Live Status |
| :--- | :--- | :--- |
| **[gemma4-multilora-adk-eval](https://github.com/RobinaMirbahar/gemma4-multilora-adk-eval)** | Dynamic Multi-LoRA Hot-Swapping & CI/CD Agent Evaluation (`adk eval`) on Cloud Run GPUs with Gemma 4 & vLLM | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/gemma4-multilora-adk-eval?style=flat-square&color=4285F4) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/gemma4-multilora-adk-eval?style=flat-square) |
| **[slothsre-napops](https://github.com/RobinaMirbahar/slothsre-napops)** | SlothSRE ("NapOps") — Overnight Gemma 4 & Google ADK Incident Digest Agent with OpenTelemetry & Sentry AI Monitoring | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/slothsre-napops?style=flat-square&color=34A853) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/slothsre-napops?style=flat-square) |
| **[streamforge-event-adk](https://github.com/RobinaMirbahar/streamforge-event-adk)** | Real-Time Event-Driven Multi-Agent Pipelines on Google Cloud with ADK (`ParallelAgent` & `LoopAgent`), Eventarc & Pub/Sub | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/streamforge-event-adk?style=flat-square&color=FBBC04) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/streamforge-event-adk?style=flat-square) |
| **[a2a-mcp-mesh-cloudrun](https://github.com/RobinaMirbahar/a2a-mcp-mesh-cloudrun)** | Agent-to-Agent (A2A) & Multi-MCP Mesh on Cloud Run with Gemma 4, Google ADK, Spanner Graph & AlloyDB | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/a2a-mcp-mesh-cloudrun?style=flat-square&color=EA4335) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/a2a-mcp-mesh-cloudrun?style=flat-square) |
| **[cloudguard-gemma-agent](https://github.com/RobinaMirbahar/cloudguard-gemma-agent)** | 6-Part Production Gemma Enterprise Agent Series (ADK, Cloud Run L4 GPU, Firestore, Model Armor, Vertex AI QLoRA) | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/cloudguard-gemma-agent?style=flat-square&color=4285F4) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/cloudguard-gemma-agent?style=flat-square) |
| **[Comic-Studio-Ai](https://github.com/RobinaMirbahar/Comic-Studio-Ai)** | Multi-Agent Comic Generator powered by Google Gemini API & FastAPI deployed on Google Cloud Run | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/Comic-Studio-Ai?style=flat-square&color=EA4335) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/Comic-Studio-Ai?style=flat-square) |
| **[vertex-ai-imagen3](https://github.com/RobinaMirbahar/vertex-ai-imagen3)** | Generate AI-powered product & editorial images using Google Vertex AI Imagen 3 in Google Colab | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/vertex-ai-imagen3?style=flat-square&color=34A853) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/vertex-ai-imagen3?style=flat-square) |
| **[Python-Learning-Assistant](https://github.com/RobinaMirbahar/Python-Learning-Assistant)** | AI-powered Python tutor covering 50+ interactive exercises with real-time feedback built with Streamlit + Gemini API | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/Python-Learning-Assistant?style=flat-square&color=FBBC04) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/Python-Learning-Assistant?style=flat-square) |
| **[cozy-ai-agent-cafe-game](https://github.com/RobinaMirbahar/cozy-ai-agent-cafe-game)** | Prompt Paws: 60 FPS Cozy AI Agent Bakery Simulation Game teaching RAG, Guardrails, Context Caching & Multi-Agent Loops | ![Stars](https://img.shields.io/github/stars/RobinaMirbahar/cozy-ai-agent-cafe-game?style=flat-square&color=7928CA) ![Last Commit](https://img.shields.io/github/last-commit/RobinaMirbahar/cozy-ai-agent-cafe-game?style=flat-square) |

<div align="center">
  <img src="https://raw.githubusercontent.com/RobinaMirbahar/RobinaMirbahar/main/ProfileImages/GithubBanner.svg" alt="Robina Mirbahar Banner" width="100%" />
</div>

<!-- AUTHOR_SIGNATURE_START -->
---

## 👩‍💻 About the Author

**Robina Mirbahar — Google Cloud & AI Architect**  
*Google Developer Expert (GDE) in Cloud & AI • Published Author • Multi-Agent Systems & Serverless GPU Builder*

[![GitHub](https://img.shields.io/badge/GitHub-RobinaMirbahar-181717?style=for-the-badge&logo=github)](https://github.com/RobinaMirbahar)
[![Medium](https://img.shields.io/badge/Medium-@robinamirbahar-000000?style=for-the-badge&logo=medium)](https://medium.com/@robinamirbahar)
[![DEV.to](https://img.shields.io/badge/DEV.to-robinamirbahar-0A0A0A?style=for-the-badge&logo=devdotto)](https://dev.to/robinamirbahar)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Robina_Mirbahar-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/robina-mirbahar)
[![GDE Profile](https://img.shields.io/badge/Google_Developer_Expert-Profile-4285F4?style=for-the-badge&logo=google)](https://g.dev/RobinaMirbahar)
[![YouTube](https://img.shields.io/badge/YouTube-@RobinaMirbahar-FF0000?style=for-the-badge&logo=youtube)](https://www.youtube.com/@RobinaMirbahar)
[![X](https://img.shields.io/badge/X-@RobinaMirbahar-000000?style=for-the-badge&logo=x)](https://x.com/RobinaMirbahar)
[![Amazon Author](https://img.shields.io/badge/Amazon_Book-Cloud_Computing_Voyage-FF9900?style=for-the-badge&logo=amazon)](https://www.amazon.com/dp/B0D146249R)

<!-- AUTHOR_SIGNATURE_END -->
