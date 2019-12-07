# Questions and Tasks
## (1) Data Usage.
### (a) What outside data have you appended to the original data set? Why did you choose this data?
We appended the number of nearby high schools based on zipcode to the original data. The reason why we chose this data is that families often move to areas with better schools, or areas that are closer to schools for the convenience of their children and themselves. This dataset contains the number of high schools at a specific zip code. With the original idea that families would want to live near schools, this leads us to believe that schools are one of the many factors that determine rent, and this helps us better train our model.
### (b) Does the inclusion of this additional data raise any ethical considerations?
Yes, because the families that have kids in high school, will be more likely looking for renting an apartment/house near these schools so that it will raise ethical considerations. The ethical considerations of this dataset are that rent prices might be raised considering that they’re closer to a school area. Most landlords know that being located next to a school increases the chance of their apartments being looked at/wanted thus they can raise the prices without batting an eye and the families would have to pay the price for the convenience of a nearby school.

## (2) Data Exploration.
### (a) What outliers present issues for your analysis? How have you chosen to handle them? Why?
We found outliers in size_sqft greater than 8000 when rent is less than 10000 and bathrooms greater than equal to 12 when rent is less than 10000.  We also found outliers when bathrooms are equal to 6 when rent is less than 20000 and when the number of bedrooms is greater than or equal to 8 when rent is less than 20000. We chose to drop these outliers as by removing them we got a smaller MSE. 
### (b) To what extent do missing values pose a challenge for your analysis? How have you chosen to handle them? Why?
If we didn’t handle the missing values it will result in higher MSE which is “11741567.0975” for test 3 but when we handle the missing value gave us lower MSE which is “11693457.4555 ”, so we have chosen to handle it by knowing which column has null values and handle using the mean and the median and also dropping as size_sqft>8000,  bathrooms>=12, 'bathrooms'==6, bedrooms >=8 comparing with the rent.
### (c) Are there any other aspects of the data your exploration shows might be problematic?
One aspect that might be problematic is when the person who posted the apartments didn't differentiate between studio apartments and not listing the amount of rooms.
### (d) Create at least one visualization that demonstrates the predictive power of your data.
train_df[(train_df['addr_zip'] > 0) ].plot(x='addr_lon', y='addr_lat', kind='scatter', alpha=0.3, c='rent', cmap=plt.get_cmap("jet"), colorbar=True, figsize=(14,8) )
plt.title('Rental Predictions')
plt.show()
![]()

## (3) Transformation and Modeling.
### (a) Describe 5 features you think play the biggest role in your model.
1- 'bedrooms', 
2-'year_built', 
3-'bathrooms', 
4-'min_to_subway', 
5-'size_sqft'
### • How did you create these features?
We created these features from the train data as we checked the most correlated ones to the rent using :
corr = data.corr()
### • How do you know these features are playing key roles?
We know that these features are playing a key role as they are the most ones correlated to the rent. Also, the predicted values seem so realistic.
### (b) Describe how you are implementing your model. Why do you think this works well?
As for implementing the model, we're training the model to understand what determines rent prices and then implementing towards places without listed rent so that we can predict based on the related features. This works well because by knowing how different features affect the rent of an apartment, we should be able to use this knowledge to predict the rent of another apartment with similar features.
### (c) Describe your methodology for selecting your model. Why do you think this type of model works well?
For the first task, we chose Random Forest because it creates decision trees on randomly selected data samples, gets a prediction from each tree and selects the best solution by means of voting. So on the first task, this model gave us the lowest MSE as it was lower than linear regression and decision tree.
But for the second task, we first choose the Gradient Boosting Regressor because it builds trees one at a time, where each new tree helps to correct errors made by the previously trained tree. So this time we have tried this new model and we figured out that this better than the Random Forest model and also it gave as the lowest MSE.
Then we chose the  K-Neighbour because we looked deeper for more models we found out the K-Neighbour give us MSE lower than the Gradient Boosting Regressor as K-Neighbour is used to predict values of any new data points that means that the new point is assigned a value based on how closely it resembles the points in the training set.
 
## (4) Metrics, Validation, and Evaluation.
### (a) How well do you think your model will perform on the holdout test set? How do you know?
We believe that our model will perform well on the holdout test set, because of the features we use as they lead to a lower MSE overall.
### (b) Is your model useful? Why or why not?
Yes, K-Neighbour is useful because it can be used for both classification and regression. The algorithm is simple and easy to implement.
### (c) Are there any special cases in which your model works particularly well or particularly poorly?
K-Neighbour Works particularly well on nonlinear data and also it is much faster compared to other algorithms
K-Neighbour Works particularly poorly as it is not suitable for large dimensional data and also It requires large memory for storing the entire training dataset for prediction.
### (d) Create at least one visualization that demonstrates the predictive power of your model.
data = pd.concat([train_df['rent'], train_df['addr_zip']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='addr_zip', y="rent", data=data)
fig.axis(ymin=0, ymax=60000);
![]()
 
## (5) Conclusion
### (a) How would you use this model?
The model was built to predict rent, so I would use it to do that just for the next apartment I look for.
### (b) If you could have additional modeling features, what would they be?
If we could have additional modeling features, they would be crime rates by zip code, nearby parks by zip code, rent history.
### (c) Would you rather have more data or more features?
Well both, but more data would be more important, having 1000 features and 10 observations isn’t going to be better than having 1000 observations and 10 features. With more data we can make a better and more accurate prediction.
 
 
 
 
 
 

