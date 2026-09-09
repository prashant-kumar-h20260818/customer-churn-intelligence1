# Customer Churn Intelligence & Retention Prioritization

An end-to-end machine-learning decision-support system that identifies customers most likely to leave, explains their churn risk, and prioritizes retention actions using predicted churn probability and customer value.

## Business problem
Telecom teams cannot contact every customer with the same intensity. This project answers four practical questions:

1. Which customers are most likely to churn?
2. How likely are they to churn?
3. Which factors are associated with that risk?
4. Who should the retention team contact first?

## Validated results
Using the 7,043-customer IBM Telco churn sample and an 80/20 stratified split:

| Metric | Result |
|---|---:|
| Selected model | Logistic Regression |
| 5-fold CV ROC-AUC | 0.8469 |
| Held-out ROC-AUC | 0.8446 |
| Churn recall | 90.91% |
| PR-AUC | 0.6493 |
| F1 | 0.5975 |
| Decision threshold | 0.34 |

The recall-focused threshold detected 340 of 374 churners in the held-out test set.

## Models benchmarked
- Logistic Regression
- Random Forest
- XGBoost

The final model is selected using 5-fold cross-validation ROC-AUC. The operating threshold is selected using out-of-fold F2 score, so the test set remains untouched until final evaluation.

## Streamlit application
The app provides:
- Executive churn dashboard
- Individual churn probability prediction
- Low / Medium / High / Critical risk tiers
- Local SHAP-style explanation of churn drivers
- Batch CSV scoring
- Revenue-at-risk estimation
- Prioritized retention queue
- Model comparison and global feature importance

## Run locally
```bash
pip install -r requirements.txt
python train_deploy.py
streamlit run app.py
```

If the generated model or dataset is not yet present, the Streamlit app can build them automatically on first launch.

## Deployment
Use Streamlit Community Cloud with:
- Repository: this repository
- Branch: `main`
- Main file: `app.py`

## Business-value note
`RevenueAtRisk = ChurnProbability × MonthlyCharges × 12` is a prioritization estimate, not an audited revenue-loss forecast or full customer lifetime value.
