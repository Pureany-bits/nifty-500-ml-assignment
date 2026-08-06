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

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.90099 | 0.989506 | 0.916667 | 0.55 | 0.6875 | 0.662178 |
| Decision Tree | 0.940594 | 0.944136 | 0.791667 | 0.95 | 0.863636 | 0.831672 |
| KNN | 0.930693 | 0.940741 | 0.842105 | 0.8 | 0.820513 | 0.777996 |
| Naive Bayes | 0.841584 | 0.903086 | 0.75 | 0.3 | 0.428571 | 0.406248 |
| Random Forest | 0.980198 | 0.988272 | 0.909091 | 1 | 0.952381 | 0.941618 |

**e. Observations**

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Showed a very high AUC (~0.989) and Precision (~0.916), meaning its positive predictions are highly reliable. However, its low Recall (0.55) indicates it was too conservative and missed nearly half of the truly profitable companies. |
| **Decision Tree** | Achieved excellent Recall (0.95) and strong Accuracy (~0.94). While it successfully identified almost all profitable companies, a lower Precision (~0.79) indicates a higher rate of false positives.|
| **KNN** | Delivered a solid, well-rounded performance (Accuracy ~0.93). It balanced its predictions nicely with Precision at ~0.84 and Recall at 0.80. |
| **Naive Bayes** | Yielded the lowest overall performance across all metrics (Accuracy ~84%, MCC ~0.406). Its atrocious Recall of 0.30 means it missed 70% of the highly profitable companies, proving that assuming feature independence is a poor approach for this specific financial data. |
| **Random Forest (Ensemble)**| Flawless Recall (1.0), meaning it successfully identified every single highly profitable company in the test set without missing one.|
| **Overall Winner** | Random Forest. It provided the highest Accuracy (~98%), F1 Score (~0.952), and MCC (~0.941) out of all models tested, making it the undisputed best choice for this classification task. |




