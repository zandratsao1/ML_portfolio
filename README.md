# ML_portfolio
Machine Learning Projects
Project OverviewThis project aims to help Waze proactively identify users likely to stop using the app (churn). 
By predicting at-risk users, Waze can implement targeted retention strategies—such as fuel discounts or personalized notifications—to improve long-term user loyalty.


The Problem
Waze leadership wants to understand why users churn and build a model that can predict this behavior before it happens. 
Initial analysis showed that user activity (measured by activity_days) is a primary indicator of churn, but simple linear relationships are insufficient to capture complex behavioral shifts.


Solution & Methodology
I followed the PACE (Plan, Analyze, Construct, Execute) framework to develop and evaluate three distinct models:

1. Logistic Regression (Baseline)Goal: Establish a baseline understanding of feature significance.Key Finding: Every additional day a user is active significantly decreases the odds of churning.Limitation: The model had low recall, indicating it missed many potential churners due to its linear nature.

2. Random Forest & XGBoost (Champion Models)Approach: Implemented ensemble tree-based models to capture non-linear interactions.

Feature Engineering: Created new metrics like km_per_driving_day and percent_sessions_in_last_month to better represent user intensity.

Results: XGBoost was selected as the champion model due to its superior ability to handle imbalanced data and its higher F1-score.

Technical DetailsModel PerformanceThe models were evaluated primarily on Precision (avoiding false alarms) and Recall (catching as many churners as possible).


Key Features
According to the XGBoost feature importance, the top predictors were:

km_per_driving_day: The intensity of usage per active day.

percent_sessions_in_last_month: Recent velocity of engagement.

activity_days: Frequency of app usage.



Future Improvements
To move beyond the current performance, I recommend the following enhancements:
Recency-Weighted Features: Weighting recent activity more heavily than old data.
Feature Intersections: Analyzing interactions like Device Type + Driving Days.
External Context: Incorporating gas price fluctuations, competitor promotions, and holiday seasonality.
Advanced Optimization: Addressing class imbalance with SMOTE and building a stacking meta-model.
New Data Streams: Integrating app crash logs and clickstream data to identify technical friction points.


How to RunClone the repository.Ensure you have xgboost, scikit-learn, pandas, and matplotlib installed.
Open the notebook: Activity_Course 5 Waze project lab-Copy1.ipynb.
