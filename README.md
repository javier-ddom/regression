Project Goals:
To make a more accurate estimate of property tax assessed values (‘taxvaluedollarcnt’)
Accomplish the above by finding drivers that have the highest effect on property value
Consider questions like: Why do some properties have a much higher value than others when they are located so close to each other? Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location? Is having 1 bathroom worse than having 2 bedrooms?
Deliver a report that can be replicated with analysis of what was made
Make recommendations for what works or doesn’t work for predicting home values

Planning
Write acquire function for zillow dataset
Strip out columns that are lacking most data, scale the remaining data, use feature engineering if needed
Begin to analyze correlations and whether or not certain characteristics influence property values
Adjust coefficients so that chosen variables have their effects within desired ranges
Explore different tests to find which provides the highest accuracy in predicting the tax assessed value

Start with creating the get_zillow_data function for the first regression exercise, the prep_zillow_data function, and combine them into the wrangle_zillow_data function to hopefully pull data and do some preparation in one step. 


