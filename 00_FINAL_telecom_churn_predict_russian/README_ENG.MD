# Determining the probability of customers leaving the telecom company and maximizing revenue

Customer base management is of great importance for a company providing services in the field of telecommunications. It is obvious that the outflow of customers negatively affects the company's revenue. According to the data provided by the customer, while **customer churn for the last 4 months was 1869 customers** (this is the entire customer churn in the data provided or 26.5% of the total number of customers at the time of the data cut), **inflow of new amounted to only 464 clients.**

## Objective of the project

1. Conduct data analysis and develop a machine learning model to identify customers who are inclined to leave the company in the near future (classification task), which will allow them to offer special conditions and avoid a drop in the customer base, revenue and profit of the company.
2. Offer a mechanism for calculating the impact of offering discounts to specific customers, whose exit probabilities were predicted by the model, on the company's revenue in order to maximize it.
3. Make a portrait of a client prone to leaving for further marketing and product analysis.

**Work plan (stages):**
1. Data review and preliminary analysis.
2. Preparing data for exploratory analysis.
3. Exploratory data analysis to identify factors that can affect customer churn.
4. Statistical testing of hypotheses.
5. Model development, selection of hyperparameters on the training set using cross-validation.
6. Testing the model on a test set.
7. Interpretation of results.

## Applied tools
Matplotlib, NumPy, Pandas, Python, SciPy, Scikit-learn, CatBoost.

## Practiced Skills
Data processing, skipping, grouping, sorting, exploratory data analysis, descriptive statistics, statistical hypothesis testing, machine learning, classification threshold adjustment to maximize revenue.

## Conclusion

**Hyperparameter fitting and model selection:**

**Logistic regression** on a sample with a full set of features gave the ROC-AUC metric value slightly higher (0.86) than on a feature set from which monthly and total revenue were removed to avoid multicollinearity (0.85).
For **CatBoostClassifier**, the search for the best hyperparameters was carried out in several stages. For all stages, the loss function Logloss and a learning rate of 0.1 were chosen. The remaining parameters were in the following ranges (by stages):
1. Depth of trees from 1 to 5 in increments of 1; quantity from 900 to 1080 in increments of 20.
2. Depth of trees from 4 to 7 in increments of 1; number from 1000 to 1350 in increments of 50.
3. Depth of trees from 1 to 3 in increments of 1; number from 1800 to 2900 in increments of 100.
4. Depth of trees 2; number from 2900 to 4900 in increments of 100.
It is noted that the number of iterations improves the model more than the depth.

The best results were shown by the CatBoostClassifier model with Logloss loss function, learning rate 0.1, tree depth 2, number 4500 - **best ROC-AUC metric was 0.923**.

**Testing:**

On the test sample, the values ​​of the metrics of the best model were:
- ROC-AUC - 0.93
- Accuracy - 0.89
- Recall - 0.74

**Interpreting results to maximize revenue**

Classification according to the main metric ROC-AUC, on average, assigned customers to the first class with a probability above 27.4%, which approximately corresponds to the share of the target class in the sample (26.5%). This is how classification works with the selected sklearn tool roc_auc_score (classification probability scores roughly correspond to the probability of the class with the higher label, class 1).

At the same time, as noted above, the negative impact on revenue of a customer's departure is higher than the negative impact of offering discounts. And although the accuracy of calssification is also high, it concerns predictions for both classes of clients. However, the main thing for us is the completeness of identifying clients who tend to leave.

In the work, a selection of a different, manual threshold for the probability of assigning customers to the target class, above which customers should be considered inclined to leave, based on the maximization of the company's revenue, is made. The calculations are based on the error matrix for each threshold (true positive, false positive, false negative, and true negative predictions).

The calculation was carried out on the example of monthly revenue (leaving customers - an average check of $ 74, the remaining ones - $ 61 in accordance with historical data). **In order to additionally confirm this difference, statistical testing was carried out, which showed that the probability of a random difference in average receipts in the samples is extremely small.**

On the other hand, the calculation includes the offer of discounts of different sizes (5%, 10%, 20%, 50%) on the monthly payment to all customers classified as a target class:
1. The size of the discount is 5%, the highest revenue is 113123.45, the threshold of the highest revenue is 0.152602.
2. The size of the discount is 10%, the highest revenue is 108546.9, the threshold of the highest revenue is 0.152602.
3. The size of the discount is 20%, the highest revenue is 99395.8, the threshold of the highest revenue is 0.134607.
4. The size of the discount is 50%, the highest revenue is 72157.5, the threshold of the highest revenue is 0.087908.

It is proposed to use a 5% or 10% discount as a promo offer for these calculations with the proposed parameters. At the same time, discounts should not be offered too large - both 5% and 10% give the same threshold and the same number of potential customers who have left (detection completeness), and revenue with a 5% discount is higher. Of course, the question remains - what size of the discount is more reliable to keep the client.

At the same time, taking into account the costs of retention - discounts should not be offered to too many - the Recall value is not the maximum, although it is very high (0.92). At the same time, the Accuracy value (accuracy for both classes) has decreased - we have adjusted the threshold so that the model assigns more clients to the target class, including those that do not belong to it. However, from a business point of view, this value is optimal.

For reference, an illustration of the logic in terms of the meaning of additional interpreted metrics. Changes in metrics depending on the threshold value:
1. Threshold value 0.01, ROC-AUC 0.93, Accuracy 0.45, Recall 0.99. We identify 99% of the target class, however, the accuracy for both classes is very low, which means that discounts will be offered to a large number of loyal customers - the revenue value is not optimal.
2. Threshold value 0.152602, ROC-AUC 0.93, Accuracy 0.79, Recall 0.92. We identify 92% of the target class, the accuracy for both classes is acceptable, the revenue value is **optimal**.
3. Threshold value 0.9, ROC-AUC 0.93, Accuracy 0.84, Recall 0.42. We identify only 42% of the target class, however, the accuracy for both classes is high, which means that discounts will be offered to a smaller number of loyal customers. However, leaving customers will have more and more negative impact on revenue - the value of revenue is not optimal.

Similar calculations and setting the probability threshold for classifying customers can also be performed for other marketing promotions - offers of one-time promotional codes, discounts for a different period (for example, three months, in this calculation - a discount period of one month), free technical support or changes in the type of Internet connection ( when providing additional data on historical unit costs for the provision of these services), as well as adapted to a different business logic.

**According to the results of exploratory data analysis**, the following portrait of a client prone to leaving was obtained based on differences in the proportions of various indicators among the clients who left and those who remained.

1. According to the personal data of clients:
- there are more elderly people among those who left - 25%/13%;
- among those who left, there are fewer dependents - 17%/34%;
- among those who left, they have fewer partners - 36%/53%.
2. By connected services:
- there are more Internet users among those who left - 94%/73%;
- there are more Fiber optic users among those who left - 69%/35%;
- there are fewer DSL users among those who left - 25%/38%;
- among those who left, those who use technical support are fewer - 17% / 34%;
- among those who left less use the services of blocking unwanted content - 16%/33%;
- among those who have left, there are fewer users of backup services - 28%/37%;
- Among those who left, those who use anti-virus services are less - 29%/36%.
3. By types of contracts and payment:
- among those who left, those who pay more by electronic checks - 57% / 25%, among other types of payment - 12-16% compared to 25% for other types of payment for the remaining clients;
- among those who left, there are more with a monthly type of contract - 89% / 43%, an annual type of contract - 9% / 25%, a two-year one - 2.6% / 32%;
- among those who left, there are more people who receive electronic checks - 75%/54%.
4. By payment amounts:
- among those who left, the average monthly payment is slightly higher - 74/61 dollars;
- among those who left, the average total amount of payment is much less - 1532/2550 dollars.
5. By the term of cooperation with the company: among those who left, the average term is almost two times lower - 547/1144 days.

**According to the results of the influence of various features on the prediction result of the machine learning model**, the most important features that affect the propensity to leave were (in descending order):
- time spent in the company;
- the amount of the monthly payment;
- the amount of the total payment;
- type of contract (monthly, annual or two-year);
- type and availability of Internet connection;
- payment method;
- Availability of electronic invoicing.

The data of the exploratory data analysis and the analysis of the importance of the features of the model are consistent with each other.
**Thus, departed customers are characterized by the following main features** (in descending order of importance):
1. Less time were clients, the average term is almost two times lower - 547/1144 days.
2. Paid more (on average per month $74 versus $61).
3. They preferred the monthly type of contract - 89%/43%.
4. They used the Internet more - 94%/73%, mainly Fiber optic (69%/35%), while using technical support and other Internet services much less.
5. More paid by electronic checks - 57%/25%.
6. More received electronic checks - 75%/54%.

The above hypotheses about the inequality of average checks, total payments and the period of cooperation with the company are confirmed by statistical tests.