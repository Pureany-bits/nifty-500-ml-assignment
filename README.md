# ML Assignment 2: Nifty 500 High Profitability Classification

**a. Problem Statement**
The objective of this project is to implement multiple machine learning classification models to predict whether a company listed in the Nifty 500 index is "Highly Profitable".
"Highly Profitable" is defined as having a Net Profit in the top 25% of the dataset based on its quarterly operating and financial metrics.

**b. Dataset Description**
The dataset comprises quarterly financial results for companies in the Nifty 500 index.
After thorough data cleaning to remove empty columns and handle missing data, the final dataset features 501 instances and 13 numerical features, satisfying both the minimum instance size (500) and feature size (12) requirements.
The target variable is binary: 1 representing Highly Profitable (Top Quartile) and 0 otherwise.

**c. Github Repository Link**
[Insert your GitHub Repository URL here]

**d. Models used & Evaluation Metrics**

| ML Model Name       |   Accuracy |      AUC |   Precision |   Recall |       F1 |      MCC |
|:--------------------|-----------:|---------:|------------:|---------:|---------:|---------:|
| Logistic Regression |   0.960396 | 0.972308 |    1        | 0.846154 | 0.916667 | 0.896276 |
| Decision Tree       |   0.950495 | 0.941538 |    0.888889 | 0.923077 | 0.90566  | 0.872405 |
| KNN                 |   0.90099  | 0.939231 |    0.807692 | 0.807692 | 0.807692 | 0.741026 |
| Naive Bayes         |   0.881188 | 0.874872 |    0.791667 | 0.730769 | 0.76     | 0.682184 |
| Random Forest       |   0.970297 | 0.997179 |    0.925926 | 0.961538 | 0.943396 | 0.923574 |

**e. Observations**

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved a perfect Precision score (1.0), meaning every company it predicted as "Highly Profitable" actually was. However, its lower Recall (~0.846) indicates it missed several highly profitable companies. |
| **Decision Tree** | Demonstrated strong balance with an Accuracy of ~95%. It captured more true positives than Logistic Regression (Recall ~0.923) but had slightly more false positives. |
| **KNN** | Delivered moderate results (Accuracy ~90%). Notably, it achieved a perfectly balanced Precision and Recall score (~0.807), meaning its false positive and false negative rates were identical. |
| **Naive Bayes** | Yielded the lowest overall performance across all metrics (Accuracy ~88%, MCC ~0.682). This is expected, as Naive Bayes assumes feature independence, which is rarely true for financial data where metrics like revenue and profit are correlated. |
| **Random Forest (Ensemble)**| Successfully modeled the complex relationships and engineered features in the dataset, boasting a near-perfect AUC (~0.997) and excellent Recall (~0.961).|
| **Overall Winner** | Random Forest. It provided the highest Accuracy (~97%), AUC, F1 Score, and MCC out of all models tested, making it the most robust choice for this classification task. |