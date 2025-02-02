# Disaster Response Pipeline Project

This is a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualisations of the data.

There is a data set containing real messages that were sent during disaster events. An ETL pipeline is created to clean the data for machine learning. A machine learning pipeline is created to categorise these events using the cleaned data. The classification model used here is RandomForestClassifier from sklearn. GridSearch is used to tune the hyper parameters and get the best model. The model will be deployed to a web app and use for classifying the new message.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
