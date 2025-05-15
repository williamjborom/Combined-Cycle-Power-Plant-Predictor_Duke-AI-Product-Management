# Combined-Cycle-Power-Plant-Predictor_Duke-AI-Product-Management

In this short project, which is part of Duke's AI Product Management Course on Coursera, I will be trying some different modeling techniques on power plant data. The goal is to gather insights about what controibutes or takes away from creating
as much load as possible form a base load.

## Data

The dataset can be downloaded from this link: https://storage.googleapis.com/aipi_datasets/CCPP_data.csv

The variables included are:

AP - Ambient Pressure
T - Temperature
RH - Relative Humidity
EV - Exhaust Vacuum

And the output variable PE - Net Hourly Electrical Output.

## Models 

I will try out a total of 4 models. Two of these will be Linear Models built using SKLearn. The data is small enough that the linear models do not need any iterative training algorithm. The first linear model will be a basic linear combination of the features with an intercept or bias term. The second linear model will also have all degree polynomial terms including interaction effects. I used 10 fold CV to assess the accuracy of each model. Next I explored a random forest model with a maximum tree depth of 4. Lastly, I wanted to see how a neural network would do. 