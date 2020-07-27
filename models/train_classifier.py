import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from datetime import datetime
import pickle

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    conn = engine.connect()
    df = pd.read_sql("SELECT * FROM DisasterResponse", con=conn)
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    #tokenize and lemmatize raw text and return clean tokens
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [w for w in tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in filtered_tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens

def build_model():
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer(smooth_idf=False)),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #GridSearchCV parameters
    parameters = {
    'clf__estimator__n_estimators': [10,20]
    }

    #Train the model with GridSearchCV to find the best model
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_preds = model.predict(X_test)

    #transpose the values for each categories
    Y_preds_transpose = np.transpose(Y_preds)
    Y_test_transpose = np.transpose(Y_test)
    score = []

    #evaluate the model by categories
    for e in range(Y_preds_transpose.shape[0]):
        validation = classification_report(Y_test_transpose.iloc[e,:].values, Y_preds_transpose[e], output_dict=True)
        score.append((category_names[e], validation))
    print(score)


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        start = datetime.now()
        model.fit(X_train, Y_train)
        print("training finished. Time used:{}".format(datetime.now()-start))

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
