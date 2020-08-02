import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load data from CSV file and return a merged pandas dataframe

    Parameters:
    messages_filepath - the file path for the message
    categories_filepath - the file path for the categories

    Returns:
    df - pandas dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    """Clean data for a pandas dataframe.
    Clear format, drop invalid values and duplicated values in dataframe.

    Parameters:
    df - pandas dataframe

    Returns:
    df - pandas dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)

    # extract a list of new column names for categories
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x.split("-")[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(columns="categories", inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # drop values that is not 1 or 0
    df = df[df.related != 2]

    return df

def save_data(df, database_filename):
    """Save cleaned dataframe to SQL database file
    Parameters:
    df - pandas dataframe
    database_filename - name of the SQL database file

    Returns:
    None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
