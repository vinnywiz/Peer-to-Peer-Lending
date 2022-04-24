
# Capstone Project Team Falcon
***
![image](results/output/Prosper.png)

Members: Zagdbazar Davaadorj, Osei Bonsu , Supriya Patwardhan, Vineeth Yeluru
## Peer-to-Peer-Lending: Predicting Bad Loans with Machine Learning

****
## Table of Contents
<!-- no toc -->
- [1.	Introduction](#1introduction)
    - [a. Background](#a-background)
    - [b. Audience and Motivation](#b-audience-and-motivation)
- [2.	Data Acquisition](#2data-acquisition)
    - [a. Data source](#a-data-source)
- [3.	Methodology](#3methodology)
    - [a.	Data Preprocessing](#adata-preprocessing)
    - [b. Data Exploration](#b-data-exploration)
    - [c. SMOTE - Synthetic Minority Oversampling Technique](#c-smote---synthetic-minority-oversampling-technique)
    - [d.	Feature Engineering](#dfeature-engineering)
    - [c.	Machine Learning Models](#cmachine-learning-models)
- [4.	Results and Discussion](#4results-and-discussion)
    - [Dummy Classifier](#dummy-classifier)
    - [Logistic regression](#logistic-regression)
    - [Random Forest](#random-forest)
    - [XGBoost Classifier](#xgboost-classifier)
- [5.   Model Explainability (SHapley Additive exPlanations - SHAP)](#5---model-explainability-shapley-additive-explanations---shap)
  - [Local Explainations](#local-explainations)
  - [Global Explanations and Feature Importance](#global-explanations-and-feature-importance)
    - [Bar Plot](#bar-plot)
    - [Summary Plot](#summary-plot)
    - [Beeswarm Plot](#beeswarm-plot)
- [6.	Conclusion, Future Work and Ethical concern](#6conclusion-future-work-and-ethical-concern)
- [7.    Refrences](#7----refrences)

# 1.	Introduction 
### a. Background 
Peer-to-peer lending is an innovative FinTech product that disrupts the entire banking industry. Traditionally, banks play an intermediary role between borrowers and lenders. Banks collect money from lenders as deposits or savings at a lower rate, then issue loans to borrowers higher. In addition, banks execute professional due diligence to protect lenders' money, distinguishing good borrowers from bad ones. Recently, peer-to-peer lending fintech firms invented a platform that connects lenders with borrowers without these intermediary banks. It allows lenders to earn more returns and for borrowers to get loans more cheaply and quickly. 
### b. Audience and Motivation 
Even though peer-to-peer lending platforms give tremendous opportunities to both parties, the solution itself has its drawbacks. The professional due diligence work is on investors' shoulders. Not all investors have the professional knowledge like banks to distinguish good from bad borrowers. Investors are regular people who want to earn a high return on their loans. If they invest in a bad loan, they will lose their money.  

Prosper Lending enables investors to browse consumer loan applications containing the applicant's loan details, credit history, etc., to determine which loans to fund. For example, loans from applicants deemed at higher risk of default, such as those with lower FICO scores, will typically carry higher interest rates and, therefore, the potential to yield a higher return on investment for the investor. 

Alternatively, lower risk of default loans will usually carry lower interest rates. Therefore, a Machine Learning model that could accurately predict a bad loan (default risk of a loan) using the available data on Prosper Lending could help investors maximize their investment returns by identifying the loans worth investing in. 

 Additionally, it would diversify the portfolio of investors with high returns with high risks and low risks with low returns. 

# 2.	Data Acquisition 
### a. Data source
We use two separate datasets from [Prosper Marketplace, Inc](https://www.prosper.com). The first data is the Listing data. The listing data includes information about all applicants. It includes personal features that can estimate the credit quality of borrowers. The are 886 qualitative and quantitative features per customer with inconsistent data availalbility. 

The second data is the loan data. The loan data includes actual loan information about applicants whose listings are approved and who received a loan from the Prosper Lending platform. In addition, the loan data has information about loan characteristics such as loan interest rate, loan amount, loan maturities, principal payments, balance, etc. 

The loan data include our target feature, the Loan Status column. This column contains information that signifies if a loan is a bad loan (Defaulted, Charge-off & Cancelled) or not a bad loan (Current & Completed). We assumed that our current loans will end up being completed but as part of our future work we intend to relax this assumption. Some of the features in the listings and loan data are only relevant after loan issuance and are therefore not available to investors at the time of investing. We used the [Prosper Data Dictionary](https://www.prosper.com/Downloads/Services/Documentation/ProsperDataExport_Details.html) to better understand the features available to investors before a loan issuance.

# 3.	Methodology 

![image](results/Design/etl.jpg)

### a.	Data Preprocessing 
To begin, we executed a simple exploratory analysis using the data dictionary to find unique key identifiers to merge the listings and loans data. Both datasets have their unique key identifiers; however, there is no direct connection between the unique identifiers. We decided to merge the two datasets with 6 common variables **(loan_origination_date, amount_borrowed, borrower_rate, prosper_rating, term, and co_borrower_application)**. We kept only uniquely merged rows as there might be two same borrowers who requested the same amount of loan and received the same loan interest and maturity. We filtered the data to only include loans from 2015 to 2021. After merging, we had more than 80% unique matches between the two sources.

 Other features in the loans and listing data was dropped based on these conditions;
  - features with >50% of missing data 
  - features that would introduce data leakage (eg age_in_months)
  - features unavailable until loan issuance (eg days_past_due, late_fees_paid)
  - features that include only one value (eg scorex).


  *Missing Value Imputation*

We treated the missing values in all remaining features one by one;
- For numerical variables: imputation using mean or median
- For categorical variables: imputation using mode 

### b. Data Exploration
We created several exploratory data analysis plots using seaborn and matplotlib Python libraries in order to build high-level intuition of some of the relationships between different features.

A plot of distribution of borrowed amount by loan term

![image](results/EDA_Dist_Plot_amount_borrowed.png)

The distribution is right skewed and also consistent with the fact that more people tend to pay smaller loans quickly

A plot of listings with Bad Loan Status. The below scatterplot shows in Yellow the customers who have a Bad Loan in yellow. They represent a very small population size. 

![image](results/output/Scatter_Plot_Without_Smote.png)

We can see that a small percentage of customers are defaulting. This is a typical imbalanced dataset where the default rate is much lower than the successfully completed rate. Therefore, machine learning algorithms applied to imbalanced classification datasets can produce biased predictions with misleading accuracies. We will use the SMOTE library to try to mitigate this problem.

To avoid multicollinearity in the data, both numeric and categorical variables exhibiting high degrees of multicollinearity >0.85 were dropped from the dataset

![image](results/correlation_matrix_numbers.png)

### c. SMOTE - Synthetic Minority Oversampling Technique
We use the Synthetic Minority Oversampling Technique (SMOTE), which is a widely adopted approach, to address the class imbalance dataset. SMOTE uses bootstrapping and k-nearest neighbors to construct new minority class instances by transforming data based on feature space (rather than data space) similarities from minority samples. SMOTE performs a combination of oversampling and undersampling to construct a balanced dataset.


<img src="results/Design/smote.png" alt="drawing"/>


For our puposes we oversmapled the minority class to have 20% the number of examples of the majority class. We then used random undersampling to reduce the number of examples in the majority class to have 80% more than the minority class. Ratios are what works best for your data. We believe this ratio works best for us. As shown below we successfully generated a balanced dataset using smote.

<div align="center">
<img src="results/output/Bar_Plot_Without_Smote.png" alt="drawing" width="600" height="500"/>
</div>


After SMOTE our data is nearly balanced 


<div align="center">
<img src="results/output/Bar_Plot_With_Smote.png" alt="drawing" width="600" height="500"/>
</div>

### d.	Feature Engineering
We created two new features EMI (Equated Monthly Installment) and Balance_Income.

**EMI** - This is the monthly amount to be paid by the applicant to repay the loan. The idea behind creating this variable is that people who have high EMI’s might find it difficult to pay back the loan. We can calculate the EMI by taking the ratio of the loan amount with respect to the loan amount term.

*EMI = amount_borrowed / term*

We visualize the distribution of the newly created feature 'EMI'. The ditribution wasn't so much skewed.

<div align="center">
<img src="results/EDA_Dist_Plot_EMI.png" alt="drawing" width="600" height="500"/>
</div>

**Balance Income** - This is the income left after the EMI has been paid. The idea behind creating this variable is that if this value is high, the chances are high that a person will repay the loan and hence increasing the chances of loan approval. The distribution of this variable was highly skewed so we took the log transformation of it before feeding it to the machine learning model.

*balance_income = monthly_income - EMI*

<div align="center">
<img src="results/EDA_Hist_Plot_balance_log_balance.png" alt="drawing" width="600" height="500"/>
</div>



Above we can see the distribution of the balance income before and after log transformation. The distribution is normally distributed after log transformation.

### c.	Machine Learning Models 
To be able to understand and then improve our model’s performance truly, we established a baseline for the data that we have. We employed a **dummy classifier** as our baseline model. This classifier serves as a simple baseline to compare against other more complex classifiers. The other models we were interested in testing for this problem were **Logistic Regression, Random Forest, and XGBoost classifier.** For our Random Forest and XGBoost classifier, we used grid search to get the optimized values of hyperparameters. We observed the performance of our models through the confusion matrix plot and the model classification report (our interest was Accuracy, Precision, Recall, and F1-score). We primarily care about correctly identifying **bad loans**, so the **recall score** will be necessary. To show the performance of our model, we graphically depicted the Area under the ROC Curve (AUC). By analogy, the Higher the AUC, the better the model is at distinguishing between customers who will default or not.

<div align="center">
<img src="results/Design/buildModel.jpg" alt="drawing" />
</div>

# 4.	Results and Discussion 
### Dummy Classifier
As we stated earlier, the dummy classifier was just to have a baseline to compare our model with. When loans are predicted to be bad, they are wrong 86% of the time. Of all the actual bad loans, this model identified 50% of them. We observed an F1-score of 22% for bad loans.

<div align="center">
<img src="results/output/dummy_smote_classification_report.png" alt="drawing" width="600" height="500"/>
</div>

### Logistic regression 
Of all the true bad loans, the logistic regression model identified 63% of them, and when the loans are predicted to be bad, they are bad 24% of the time. By comparing with our baseline model, we saw a slight improvement in the model performance. An F1-score of 35% was observed for bad loans. The model had an accuracy of 68%.

<div align="center">
<img src="results/output/Logistic_smote_classification_report.png" alt="drawing" width="650" height="500"/>
</div>

### Random Forest
Our Random Forest model identified almost 61% of true bad-loans with a 24% precision. There wasn't much improvement in the model accuracy (68%). We also didn't observe an improvement in the F1-score (79%). The Random Forest model seems to be the best performing model so far.

<div align="center">
<img src="results/output/RandomForest_smote_classification_report.png" alt="drawing" width="650" height="500"/>
</div>

### XGBoost Classifier
The XGBoost model reported an accuracy of 70% (best performing model out of the lot). Of all the true bad-loans this model identified 62% of them and when the loans are predicted to be bad they are bad 26% of the time. The harmonic-mean of the Precision and Recall was 36% of bad-loans. We therefore concluded that the XGBoost is our best performing model to do a model explainability.

<div align="center">
<img src="results/output/XGBoost_smote_classification_report.png" alt="drawing" width="650" height="500"/>
</div>

 # 5.   Model Explainability (SHapley Additive exPlanations - SHAP)
We used the SHAP Python package to interpret our model. SHAP is an increasingly popular method used for interpretable machine learning. SHAP assigns each feature an importance value for a particular prediction. SHAP builds model explanations by asking the same question for every prediction and feature: "How does prediction *i* change when feature *j* is removed from the model?" So-called SHAP values are the answers. They quantify the magnitude and direction (positive or negative) of a feature’s effect on a prediction. It is important to point out that the SHAP values do not provide causality.

## Local Explainations
Each observation gets its own set of SHAP values. We randomly chose 5 individual observations (see the individual SHAP value plot below). This greatly increases the transparency of our model predictions. We can explain why a case receives its prediction and the contributions of the predictors. Traditional variable importance algorithms only show the results across the entire population but not on each individual case. The local interpretability enables us to pinpoint and contrast the impacts of the factors.

**Individual Observation 1**
<div align="center"> 

<img src="results/output/Bar_Plot_XGBoost_Local_Explanability_Listing_Id_0.png" alt="drawing" width="650" height="500"/>

**Individual Observation 2**
<img src="results/output/Bar_Plot_XGBoost_Local_Explanability_Listing_Id_1.png" alt="drawing" width="650" height="500"/>

**Individual Observation 3**
<img src="results/output/Bar_Plot_XGBoost_Local_Explanability_Listing_Id_2.png" alt="drawing" width="650" height="500"/>

**Individual Observation 4**
<img src="results/output/Bar_Plot_XGBoost_Local_Explanability_Listing_Id_3.png" alt="drawing" width="650" height="500"/>

**Individual Observation 5**
<img src="results/output/Bar_Plot_XGBoost_Local_Explanability_Listing_Id_4.png" alt="drawing" width="650" height="500"/>
</div>

Looking at **individual plot 1**, the amount_borrowed and balance_income indicates that both features are positively correlated with the target variable (bad loan) for individual 1. This means that they have a high (from color red) and positive impact (shown on the x-axis) on predicting that individual 1 loan is not a bad loan.

Alternatively, looking at **individual plot 5**, the model predicted that it is a bad loan when it is actually not a bad loan. The 6 variables in red had a high and positive impact on the prediction. We can see that balance_income had a negative and low impact. The understanding here is that the balance income of individual 5 may be less than his or her monthly payment. Therefore, there is a high chance that the loan will be a bad loan as the individual may default. 

## Global Explanations and Feature Importance
We put local explanations described above together to get a **global explanation**. And because of the axiomatic assumptions of SHAP, global SHAP explanations can be more reliable than other measures. The collective SHAP values can show how much each predictor contributes, either positively or negatively, to the target variable. This is like the variable importance plot but it is able to show the positive or negative relationship for each variable with the target.

### Bar Plot
Here the features are ordered from the highest to the lowest effect on the prediction.
It takes into account the absolute SHAP value, so it does not matter if the feature affects the prediction in a positive or negative way.

<div align="center">
<img src="results/output/global_bar_plot.png" alt="drawing" width="650" height="500"/>
</div>

Here, we notice that borrower_rate and FicoRange contributes more to the overall model and thus have high predictive power.

### Summary Plot
The SHAP value  Summary plot can further show the positive and negative relationships of the predictors with the target variable. This plot is made of all the dots in the train data. It demonstrates the following information:

 - Feature importance: Variables are ranked in descending order
 - Impact: The horizontal location shows whether the effect of that value is associated with a higher or lower prediction.
 - Original value: Color shows whether that variable is high (in red) or low (in blue) for that observation.
 - Correlation: A high level of the "borrower_rate" has a high and positive impact on the target. The "high" comes from the red color, and the "positive" impact is shown on the x-axis. Similarly, we will say the "funding_threshold" is negatively correlated with the target variable.

![image](results/output/global_summary_plot.png)

### Beeswarm Plot
The beeswarm plot is designed to display an information-dense summary of how the top features in the dataset impact the model’s output. All the little dots on the plot represent a single observation. The horizontal axis represents the SHAP value, while the color of the point shows us if that observation has a higher or a lower value, when compared to other observations.

We can see that "borrower_rate" is the most important feature on average, and that borrowers are less likely to get a rate of over 36 months.

![image](results/output/global_bee_plot.png)


# 6.	Conclusion, Future Work and Ethical concern 
Predicting the occurrences of bad loans in a peer-to-peer lending platform is crucial and challenging task. More accurate prediction models would be highly beneficial since the failure of a peer-to-peer lending platform could trigger a series of financial risks. Our project shows that machine learning methods have broad application prospects in the prediction of P2P loan default.

The performance has only improved slightly through the modeling process but we have removed attributes that could lead to leakage and getting performance that does capture over 60% of the bad loans. There is additional scope for hyperparameter tuning that would allow the model to better classify the loan. 

For future work, we would want to deploy our model and have a real-time machine learning predictions. Futhermore, we plan to include macro economic factors (Inflation, unemployment rate, GDP etc) that highly affect the loan status. This would increase the performance of the current machine learning model. 

Our target feature 'bad loan' has two flags (1-bad loan, 0-not a bad loan). The 'not a bad loan' flag is a combination of (Completed and Current loans). The problem here is that we have no knowledge of whether a current loan will be defaulted in the near future or not. So for our future work, in the training dataset, we would want to remove the current loan data and only include current loans in our test data. There are potential side effects to this as we would be excluding a large number of recently opened loans in our training process.  

Futhermore, we plan to develop a dashboard to help investors to examine listing applications more in-depth. With the dashboard, it would work as a support info along with current FICO score and other indicators. 

For our project analysis, we do not disclose any individual information about loan applicants in our visualization and outcomes. We carefully consider that our model would not discriminate loan applications based on racial, ethnicity group identifications. We have chosen features that can be generalized to produce our outcome. 

# 7.    Refrences 
Xu, J., Lu, Z. & Xie, Y. Loan default prediction of Chinese P2P market: a machine learning methodology. Sci Rep 11, 18759 (2021). https://doi.org/10.1038/s41598-021-98361-6

Molnar, C., 2022. Interpretable Machine Learning. [online] Christophm.github.io. Available at: <https://christophm.github.io/interpretable-ml-book/> [Accessed 15 March 2022].

Zlaoui, K., 2021. Interpretable Machine Learning using SHAP — theory and applications. [online] Medium. Available at: <https://towardsdatascience.com/interpretable-machine-learning-using-shap-theory-and-applications-26c12f7a7f1a> [Accessed 3 April 2022].

Yen, L., 2019. P2P Lending Platform Data Analysis: Exploratory Data Analysis in R —  Part 1. [online] Medium. Available at: <https://towardsdatascience.com/p2p-lending-platform-data-analysis-exploratory-data-analysis-in-r-part-1-32eb3f41ab16> [Accessed 4 March 2022].

Brownlee, J., 2022. SMOTE for Imbalanced Classification with Python. [online] Machine Learning Mastery. Available at: <https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/> [Accessed 3 April 2022].

Imbalanced-learn.org. 2022. SMOTE — Version 0.9.0. [online] Available at: <https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html> [Accessed 3 April 2022].

Shap.readthedocs.io. 2018. Welcome to the SHAP documentation — SHAP latest documentation. [online] Available at: <https://shap.readthedocs.io/en/latest/index.html> [Accessed 10 April 2022].
