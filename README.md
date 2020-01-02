# Expedia-Hotel-Recommendations

- Expedia wanted to predict if a user will book a hotel given various features such as historical price, customer star ratings, geographical locations relative to city center, etc. 
- Expedia uses algorithm to form hotel clusters by grouping hotels based on the location of the source and destination, price, continent, type of location etc.
- These hotel clusters serve as a good source to which types of hotels people are going to book, while avoiding outliers such as new hotels that don't have historical data.

Evaluation Metric : Mean Average Precision @ 5 (MAP@5)

## Models: 

### Logistic Regression : 29.21%
### Decision Tree : 28.13%
### Random Forest : 22.14%
### Naive Bayes : 16.71%
### XGBoost : 29.12%
### KNN : 26.36%

