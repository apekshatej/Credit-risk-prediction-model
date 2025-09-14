# Credit Default Prediction  
*IMT 572: Introduction to Data Science – Final Project*  

## Overview  
This project applies data science techniques to the **UCI Credit Card Default Dataset**, exploring predictors of credit default and evaluating classification models. The workflow included:  

- **Exploratory Data Analysis (EDA)** in R and Tableau  
- **Regression-based exploratory modeling** (Logit & Probit)  
- **Classifier training & evaluation** (kNN, SVM, Random Forest)  
- **Visualization & dashboarding** with Tableau  

The goal was to understand key drivers of default, reduce dimensionality, and identify the most accurate predictive model.  

---

## Data  
- Source: Credit dataset (CSV)  
- Size: 30,000 records  
- Key variables:  
  - **LIMIT_BAL** – credit limit  
  - **SEX, EDUCATION, MARRIAGE, AGE** – demographic variables  
  - **BILL_AMT1–6** – historical bill amounts  
  - **PAY_AMT1–6** – repayment amounts  
  - **PAY_0–PAY_6** – repayment status (history of delays)  
  - **default.payment.next.month** – target (binary: default or not)  

---

## Exploratory Analysis  

### Correlation Insights  
- **BILL_AMT1–6** are highly correlated → risk of multicollinearity. Only **BILL_AMT1** retained.  
- **Payment amounts (PAY_AMT1, PAY_AMT3)** provide unique predictive power.  
- **LIMIT_BAL** shows low correlation with others but is a key protective factor against default
### Demographic Insights (Tableau)  
- **Females** show higher credit usage and default rates than males, especially ages 25–30.  
- **Singles** exhibit slightly higher defaults compared to married individuals.  
- **Younger individuals** (late 20s/early 30s) have higher bills, payments, and defaults.  
- **Higher education (university-level)** correlates with higher bills and greater default risk

---

## Regression Analysis  

### Logit & Probit Models  
- Significant predictors: **LIMIT_BAL, PAY_AMT1, PAY_AMT3, BILL_AMT1, PAY_0, PAY_2, PAY_3**.  
- **LIMIT_BAL** → negative effect (higher limits = lower default probability).  
- **PAY_AMT1, PAY_AMT3** → negative effect (larger payments reduce default).  
- **PAY_0, PAY_2, PAY_3** → positive effect (late payments increase default).  
- **Probit slightly outperforms Logit** (lower AIC)
---

## Classifier Models  

| Model            | Accuracy |
|------------------|----------|
| Random Forest    | **81.99%** |
| SVM (Linear)     | 81.74%   |
| SVM (Radial)     | 81.65%   |
| kNN (k=19)       | 79.63%   |  

- **Random Forest** is the best-performing model.  
- **SVM Linear** offers comparable performance with potential computational efficiency benefits.  
- **kNN** performs reasonably but lags behind ensemble and kernel methods 

---

## Visualization  
Interactive dashboards were built in **Tableau** (`Final project.twb`), highlighting:  
- Default rates by demographics  
- Credit usage patterns  
- Payment delays and amounts  
- Predictive variable comparisons  

---

## Key Takeaways  
- **Payment history** (PAY_0, PAY_2, PAY_3) is the strongest predictor of default.  
- **Credit limit (LIMIT_BAL)** is protective against default risk.  
- **Random Forest** achieves the highest classification accuracy.  
- **SVM (Linear)** is a strong alternative when runtime efficiency matters.  

---

## Files in Repository  
- `credit_dataset.csv` – Dataset used for analysis  
- `Final Project.pdf` – Detailed report with results and interpretations  
- `Final project.twb` – Tableau workbook with visualizations  

  

---
