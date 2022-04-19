# Capstone project Falcon Team 

## Table of Contents
# 1.	Introduction 
### a.Background 
Peer-to-peer lending is an innovative FinTech product that disrupts the entire banking industry. Traditionally, banks play an intermediary role between borrowers and lenders. Banks collect money from lenders as deposits or savings at a lower rate, then issue loans to borrowers at a higher rate. To protect lenders’ money, banks execute the professional due diligence that distinguishes good borrowers from bad ones. Recently, peer-to-peer lending fintech firms invented a platform that connects lenders with borrowers without these intermediary banks. It gives an opportunity for lenders to earn more returns and for borrowers to get loans in a cheaper and quicker way. 
### b.Audience and Motivation 
Even though peer-to-peer lending platforms give tremendous opportunities to both parties, the solution itself has its own drawbacks. The professional due diligence work is on lenders’ shoulders.  Not all lenders do not have the professional knowledge like banks to distinguish good borrowers from bad borrowers. Lenders are regular people who want to earn a high return on their loans.  If they invest in a bad loan, they would lose their money.  Therefore, we decided to develop a Machine Learning model to predict the probability of loan default and help investors to make better decisions. 

# 2.	Data Acquisition 
### a.Data source
For our initial datasets, we use two separate data from https://www.prosper.com. The first data is the Listing data.  The listing data includes information about all applicants. It includes personal features that can estimate the credit quality of borrowers. The row data has 811 features per customer. The second data is the loan data.  The loan data includes actual loan information about applicants whose listings are approved and who received a loan from the peer-to-peer platform. The loan data has information about loan characteristics such as loan interest rate, loan amount, loan maturities, principal payments, balance, etc. Additionally, the loan data include the loan status if the loan is defaulted or not. These two datasets provide crucial information for our machine learning model. 

# 3.	Methodology 
### a.	Data cleaning 
For the first stage, we execute the exploratory analysis to clean and keep usable features from both datasets. After carefully reading the dictionary of datasets, we decided to keep 12 variables from the listing and 8 variables from the loan dataset.  Both datasets have their unique key identifiers however, there is a direct connection between the listing and loan datasets. We decided to merge them with 4 common variables such as loan amount, interest rate, loan maturity, and loan location. I keep unique merged data as there might be two same borrowers who requested the same amount of loan and received the same loan interest and maturity.  

### b.	Feature engineering
We implemented various features engineering for our initial data. First, we scaled all our variables to take care of outliers and extreme values in our sample. Addtion to that, we converted some highly screwed variables by taking a log transformation.  Second,  we have created a new column named  'term_portion_finished' for each loan term. 

The portion_finished = age_in_months/term EMI - Equated Monthly Installment EMI 

EMI is the monthly amount to be paid by the applicant to repay the loan. The idea behind making this variable is that people who have high EMI’s might find it difficult to pay back the loan. We can calculate the EMI by taking the ratio of the loan amount with respect to the loan amount term Balance Income — This is the income left after the EMI has been paid. 

![image](https://user-images.githubusercontent.com/86815494/164076511-57593284-dbff-43e6-94b6-e8e8af208269.png)
![image](https://user-images.githubusercontent.com/86815494/164076538-d2ce5778-67f9-4bd6-beb6-f01507bf0893.png)
![image](https://user-images.githubusercontent.com/86815494/164076559-d2b8e42a-fda8-475d-a409-30aa8e1d6de5.png)


The idea behind creating this variable is that if this value is high, the chances are high that a person will repay the loan and hence increasing the chances of loan approval. As you can see from the distribution, the engineered variables can be a good explainer in our model.

### c.	Models. 

To reach our goal, we have adopted the logistic regression model for our baseline machine learning model. Our dependent variable in this model is a dummy variable which equals zero if the loan defaulted, otherwise equals one. Features to explain this probability are drawn from listing and loan datasets. To improve our model performance, we have implemented Synthetic Minority Oversampling Technique, the logistic regression with feature-engineered variables, the random forest model, and the XGboost method. 

# 4.	Result and discussion 
### Logistic regression 
We employ the logistic regression method to predict the loan status if the loan is likely to default or survive. We trained our model using 80% of the sample and tested the performance on the remaining 20% of the sample. The 61.40% of selected feature variations explain and predict the loan status. To show the performance of our model, we graphically depicted the Area under the ROC Curve (AUC).  Higher the AUC, the better the model is at predicting 0 classes as 0 and 1 classes as 1. By analogy, the Higher the AUC, the better the model is at distinguishing between customers who will default or not. The accuracy score of our model is 61.40.

![image](https://user-images.githubusercontent.com/86815494/164074491-f0c0aed7-2890-49ff-8eb9-3f5447ce7d13.png) 
![image](https://user-images.githubusercontent.com/86815494/164074511-845f4e4a-1243-4430-8d89-9d9371fa8ce0.png)

### Synthetic Minority Oversampling Technique
In our sample, we have 62,059 completed loans while just 614 defaulted loans. Our dataset is a typical imbalanced dataset where the default rate is much lower than the successful completed rate. Machine learning algorithms applied to imbalanced classification datasets can produce biased predictions with misleading accuracies. 

![image](https://user-images.githubusercontent.com/86815494/164077554-686aa158-50af-4e2d-a786-be5c4e3da62a.png) 
![image](https://user-images.githubusercontent.com/86815494/164077567-3fcfddd5-946a-421a-b243-3df151d6571b.png)

Even though our main model is based on the probability, we suspect there might be issues due to the unbalanced sample. Thus, we executed the Synthetic Minority Oversampling Technique (SMOT) for our analysis. The accuracy score of our model has improved to 75.00%.

![image](https://user-images.githubusercontent.com/86815494/164074561-9209b4a3-3f36-425c-960b-0b1d5d961bd7.png)
![image](https://user-images.githubusercontent.com/86815494/164074601-9727d233-19af-4140-9227-9930e767a6f5.png)

The motivation here is to see how the model will perform when we balance the data. Although, we did use the class weight property in the regression model the aim is to see if smote will improve on that splitting data into train and test before re-sampling.

### Logistic Regression with Feature Engineered Variables

![image](https://user-images.githubusercontent.com/86815494/164074647-c8cc1a54-3131-4d91-b648-3c1c15b62591.png)
![image](https://user-images.githubusercontent.com/86815494/164074679-85f01aef-8403-4098-812b-20b3e8566281.png)

### Random Forest

![image](https://user-images.githubusercontent.com/86815494/164074753-78550b0b-fbd2-4ee3-a654-8c396564bfe6.png)
![image](https://user-images.githubusercontent.com/86815494/164074774-40fc4eae-030f-420e-a854-da4c2557d63a.png)


### XGBoost

![image](https://user-images.githubusercontent.com/86815494/164074816-5db724bc-2f91-41a8-9fa0-f4cbc967133a.png)
![image](https://user-images.githubusercontent.com/86815494/164074843-0915d4de-9266-47f9-a5c5-fb6cf7e0b6b5.png)


 ## model explainability (SHAP)

# 5.	Conclusion and future work 

Refrences: 
https://towardsdatascience.com/p2p-lending-platform-data-analysis-exploratory-data-analysis-in-r-part-1-32eb3f41ab16



