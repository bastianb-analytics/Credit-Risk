# Credit Risk Prediction – HistGradientBoosting Model

## Overview

This project develops a credit risk model using **HistGradientBoosting** to prioritize clients with a high probability of default. The goal is to improve decision-making in credit approval and focus efforts on higher-risk applicants while maintaining operational efficiency.

**Dataset:** Home Credit Default Risk (application_train.csv, application_test.csv, bureau, previous_application, etc.)  

**Tools & Libraries:**
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- SHAP

---

## Data Exploration and Preprocessing

We started by analyzing the dataset, performing **Exploratory Data Analysis (EDA)** and selecting the top 11 features that most influence the target variable `TARGET` using permutation importance.

Missing values were inspected, notably in `EXT_SOURCE_1` (~56% missing). We tested imputing the median and adding missing flags, but the ROC-AUC showed minimal change (~0.0001 difference), so imputation was deemed optional.

**Feature selection and correlation:**  
- Selected top 11 features using permutation importance.
- Verified that removing features slightly reduces ROC-AUC, confirming importance of selected features.

**Image Placeholder:** Aquí va la imagen de (Feature Importance Permutation)

---

## Model Training

The model was trained using a **Pipeline** with a `ColumnTransformer` to process numerical and categorical features:

```python
pipe = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        random_state=42
    ))
])
```

## Model Evaluation Metrics

The trained model was evaluated using several key metrics:

- **ROC-AUC:** 0.75  
- **PR-AUC:** 0.23 (baseline: 0.08)  
- **Lift@10%:** 3.33  
- **KS:** 0.37  

These metrics indicate that the model can effectively rank clients according to default risk, capturing a significant proportion of defaulters while maintaining discriminative power over the minority class.

<img width="590" height="490" alt="roc curve" src="https://github.com/user-attachments/assets/8b5ad5a1-2265-450b-9089-fe88febda404" />
<img width="590" height="490" alt="pr curve" src="https://github.com/user-attachments/assets/f9c7ccc2-c719-4384-9c45-85a1016ae832" />

## Threshold Optimization

The model’s decision threshold was carefully analyzed to balance business objectives and operational constraints. Different strategies were considered:

- **Target Recall:** Capture at least 40% of defaults.  
  - Operationally infeasible due to reviewing a very large portion of clients.
- **Operational Capacity:** Only a fixed percentage of clients can be reviewed (e.g., 10%).  
  - High precision but low recall; useful if intervention costs are high.
- **Maximum F1:** Optimizes the balance between precision and recall.  
  - Provides a reasonable recall with acceptable precision and manageable intervention population.

A profit-based approach was also applied, defining:

- **Benefit per True Positive:** 100  
- **Cost per False Positive:** 10  

The threshold maximizing expected profit was identified at **0.089**, achieving a gain of 247,330 units.

At this operational point:

- **Recall:** 0.63 → captures the riskiest clients.  
- **Precision:** 0.17 → acceptable given the severe class imbalance.  
- **Predicted positive rate:** ≈ 29.5%.

<img width="656" height="470" alt="profit curve" src="https://github.com/user-attachments/assets/2e72982f-65d7-4ca4-a1dc-5240a0f6ec6f" />

## Explainability Analysis (SHAP)

To understand model predictions at both global and individual levels, SHAP (SHapley Additive exPlanations) was applied.

### Global Feature Importance
- Features such as `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_BIRTH`, `AMT_CREDIT`, and `DAYS_EMPLOYED` were identified as the most influential.  
- Contributions are moderate and no single variable dominates the predictions, indicating balanced signal across the dataset.

<img width="829" height="860" alt="shap summary" src="https://github.com/user-attachments/assets/9cb2cdde-c435-4fd5-931a-2f8ac6452dc2" />


### Individual Case Analysis
- **High-risk client:** Driven primarily by low values in `EXT_SOURCE_2` and `EXT_SOURCE_3`, showing high confidence for default prediction.  
- **Borderline client:** Risk is spread across multiple features, indicating lower model certainty and a good candidate for manual review.

<img width="1030" height="600" alt="shap high" src="https://github.com/user-attachments/assets/c3ec44ba-ec8b-4b57-ac02-021a85720150" />
<img width="1138" height="600" alt="shap borderline" src="https://github.com/user-attachments/assets/52f2cd8c-dd0b-4feb-a37c-ea21e14332fd" />

## High-Risk Population Analysis

The model was applied to the test set to identify the top 10% of clients by predicted risk.

### Key Observations
- **EXT_SOURCE_2 and EXT_SOURCE_3:** Top-risk clients show significantly lower values, indicating weaker credit history and financial reliability.  
- **AMT_CREDIT:** Moderate credit amounts requested, typical of mid-to-high risk clients.  
- **DAYS_BIRTH:** Average age of top-risk clients is 30–36 years, suggesting shorter credit history and potential income volatility.  
- **DAYS_EMPLOYED:** Shorter employment tenure, indicating lower income stability.

<img width="1152" height="389" alt="top-risk distribu" src="https://github.com/user-attachments/assets/7aec79cf-002e-41d0-9063-b8cb49dbfac8" />

### Statistical Summary
- Missing values in EXT_SOURCE features are more prevalent in the top-risk segment, which aligns with higher observed default rates.  
- Default rate in top 10% risk segment: 22.9%  
- Default rate in overall population: 19.8%  

<img width="1059" height="389" alt="top-risk describe" src="https://github.com/user-attachments/assets/0e8ccf5d-80df-4ec6-9de7-ee6dd74886aa" />
<img width="1048" height="389" alt="quartiles" src="https://github.com/user-attachments/assets/d72c6897-0ff7-4fd7-bd4f-7a045466bfce" />

## Executive Summary & Business Recommendations

A credit risk model was developed using HistGradientBoosting to prioritize clients with a high probability of default.

### Model Performance
- **ROC-AUC:** 0.75  
- **PR-AUC:** 0.23 (baseline: 0.08)  
- **Lift@10%:** 3.33  
- **KS:** 0.37  

**Business Impact:**  
- The model captures ~33% of defaults by reviewing only 10% of the population, significantly improving risk review efficiency.  
- Profit curve analysis identified an operational threshold of ~0.089, maximizing expected benefit.

### Business Recommendations
- High-risk clients (top 10% predicted risk) should be subject to stricter credit approval policies.  
- Borderline applicants may require additional verification or reduced credit limits.  
- Low-risk clients can benefit from faster approvals to enhance customer acquisition.  

### Key Risk Drivers
- **EXT_SOURCE_2** and **EXT_SOURCE_3** are the most influential predictors.  
- **DAYS_BIRTH, AMT_CREDIT, DAYS_EMPLOYED** also contribute significantly.  
- Lower EXT_SOURCE values strongly increase predicted default risk.

### Final Conclusions
- The model consistently ranks clients by default risk.  
- Accumulated gain analysis confirms the model concentrates defaults in the top-ranked population compared to random selection.  
- High-risk segment exhibits a default rate of 22.9% versus 19.8% in the overall training population.  

### Limitations & Future Work
- Probability calibration improvements  
- Advanced handling of class imbalance  
- Feature engineering from bureau and previous application tables  
- Model comparison with other gradient boosting methods

### Submission & Contact

#### Model Predictions on Test Data
The trained model was applied to the `app_test` dataset to generate risk probabilities (`risk_score`) for each client.  
Here goes the image of the risk_score distribution
<img width="704" height="470" alt="predicted" src="https://github.com/user-attachments/assets/a0e55f97-ca06-4b27-b67c-a5b7eff11d90" />)

#### Contact
- **Name:** Bastián Burgos / bastianb-analytics  
- **Email:** bastian.burgos.c@gmail.com
