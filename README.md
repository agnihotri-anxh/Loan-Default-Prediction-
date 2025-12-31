# Loan Default Prediction 🚀

## Executive Summary
We developed a robust, production-ready machine learning system to predict loan repayment risk.  
Our approach prioritized not just high accuracy, but a deep understanding of the problem, the creation of meaningful behavioral features, and the development of a reliable, well-documented process.  
This project delivers a practical, explainable risk score.

---

## 1. Data Foundation & Feature Engineering

The project involved merging and analyzing various raw data sources to assess user stability and trustworthiness before a loan application.  
This step was the most critical, focused on turning raw logs into predictive signals while strictly preventing **data leakage** (only using information predating the application).

### Data Sources Utilized
- **loan_outcomes**: Repayment status (target variable)  
- **gps**: Real-world device movement logs  
- **events**: In-app user action logs  
- **features**: 10 masked, pre-built numerical attributes  

### Key Behavioral Features

| Data Source   | What We Extracted | Predictive Rationale |
|---------------|------------------|----------------------|
| **GPS**       | Number of signals, average accuracy, speed behavior, provider variety, "last seen" recency | Regular and recent activity is a proxy for stability and active life patterns |
| **Events**    | Total actions, unique screens, session counts, network types, last activity time | High engagement and consistent usage often indicate a more legitimate user |
| **Masked Features** | All 10 features merged | Integrated to provide additional, though unknown, meaningful signals |

**Data Cleaning & Preprocessing**
- Rigorous dataset alignment for training/prediction  
- Missing values handled via **median imputation** to preserve information  

---

## 2. Modeling & Performance Assessment

We experimented with several models using a stratified split and determined that **Gradient Boosting** offered the best predictive power.

### Evaluation Metrics
Given the noisy and imbalanced nature of lending data, we relied on sophisticated metrics beyond simple accuracy:
- **ROC-AUC**: Differentiates between risky and safe borrowers  
- **PR-AUC + F1**: Focuses on effectiveness in identifying the minority class (defaulters)  

### Core Performance Results
- ROC-AUC achieved **0.61+**  
- PR-AUC significantly outperformed a naive baseline  
- Model reliably distinguishes between good and risky borrowers, providing a meaningful risk layer  

### Key Learnings
- **Behavioral Recency** was the single most effective predictive factor  
- Performance is realistic and robust given data noise and masked features  
- Model creates a valuable **non-financial behavioral risk score** ready to integrate with traditional financial scoring  

---

## 3. Production Readiness

The project output is structured for immediate real-world use.

### Prediction CSV
A clean output file ready for downstream analysis:


user_id  application_at  prediction_probability

### Live Scoring API (FastAPI)
A simulation of a real-time credit engine:
- **Input**: `user_id`, `application_at`  
- **Process**: Pulls data, dynamically rebuilds features, preprocesses, and loads the trained model  
- **Output**: Returns repayment probability  
  ```json


