# Disaster Response Pipeline Project

### Table of content

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Description](#files)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Licensing, Authors and Acknowledgements](#licensing)

## Installation<a name="Installation"></a>

The project is a Python 3.6+ project and its library dependencies will 
need to be installed before it can be run. 

It is recommended to install librabries in a dedicated virtual 
environment. The dependencies requirements can be found in
**requirements.txt** file. For example, the dependencies can be 
installed using the following commands after activation of the virtual
environment:

```
> cd /path/to/responsepipeline
> pip install -r requirements.txt
```

# Project Motivation<a name="motivation"></a>

Following a disaster event, large amounts of communications are sent 
either directly or via social media. Disaster response organisations 
would then filter through these messages in order to find the most 
relevant to their professionals: indeed, different aspects of the 
problem are taken care of by different organisations (e.g. access to 
water, medical supplies, etc.). It is safe to assume that such task 
represents a challenge for these organisations, especially right at the 
time when they have the least capacity. 

The objective of this project was therefore to aid disaster response 
organisations in completing this task more accurately and efficiently by
building a Machine Learning pipeline. 

The data used is a collection of real-life messages that were sent 
during disaster events. Each message is pre-labelled into several 
categories (e.g. "food", "water", etc.).

# File Description<a name="files"></a>

The project can be broken down into 4 main directories:

1. **data** contains the original datasets, the ETL pipeline used to 
preprocess the data, and the SQLite database the preprocessed data was 
stored in.
2. **models** contains the Machine Learning pipeline.
3. **results** contains the classification reports that were outputted 
for the test set. 
4. **app** contains the files required to run the web app where the user
can input a message and get classification results. 

# Instructions

The following commands can be run in the project's root directory to set
up the database and model:

1. To run the ETL pipeline - this will preprocess the data and store the
output in a SQLite database
```
> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

2. To run the ML pipeline - this will train the classifier and save it 
as a pickle file
```
> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

3. To run the web app - the web app will then be available at 
http://0.0.0.0:5000/
```
> python run.py
```

# Results<a name="results"></a>

The f1-score, accuracy, and recall for the test set are available for 
each category under the **results** directory.

We can see that the model struggles to predict certain categories: 
although the accuracy metric tends to be high for each category, the 
precision and/or recall is (are) low for some of them. This could be 
explained by the original dataset being imbalanced. For example, only 
0.02% of the messages in the original dataset were labelled under the 
*security* category while 0.1% of the messages were labelled under the 
*food* category - so 5 times more samples than for the *security* 
category. 

To remedy with this imbalance and improve the performance of the
classifier, further development of this project could include using 
Iterative Stratification in order to split our multi-label dataset into
new training and test datasets.

## Licensing, Authors and Acknowledgements<a name="licensing"></a>

The data comes from [Appen](https://www.appen.com/) (formerly Figure 8),
a data company that provides datasets for AI and Machine Learning 
solutions. The data was made available through Udacity's Data Scientist 
Nanodegree Program.
