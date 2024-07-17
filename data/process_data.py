import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load the data of interest from .CSV files.

    Args:
        messages_filepath (str): Filepath to the file to load message
            data from.
        categories_filepath (str): Filepath to the file to load
            categories data from.

    Returns:
        pd.DataFrame: DataFrame object with the loaded data.

    """
    # create DataFrame objects
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    # merge DataFrame objects
    df = df_messages.join(df_categories.set_index('id'), on='id', how='inner')
    return df


def clean_data(df):
    """Clean the data in DataFrame object `df`.

    :Processing:
        - Create separate category columns in `df`, rename the columns
          appropriately and the values from strings to integers.
        - Replace values of 2 by values of 0 in `related` column.
        - Remove duplicates

    Arg:
        df (pd.DataFrame): DataFrame object whose data is to be cleaned.

    Returns:
        df (pd.DataFrame): DataFrame object with clean data.

    """
    # create separate category columns in `df`
    category_cols = df['categories'].str.split(pat=';', expand=True)
    # use first row of `category_cols` DataFrame object to create column names for the categories data
    first_row = category_cols.loc[0]
    names = first_row.apply(lambda x: x[:-2]).to_list()
    category_cols.columns = names
    # convert category values to integers
    for col in category_cols:
        # set each value to be the last character of the string
        category_cols[col] = category_cols[col].apply(lambda x: x[-1])
        # convert column from string to integer
        category_cols[col] = category_cols[col].astype(int)
    # ADDITIONAL - Replace values of 2 by values of 0 in `related` column
    # Note: When `related` has a value of 2, we could see that all remaining categories have a
    # value of 0. We observe a similar behaviour when `related` has a value of 0. We therefore
    # concluded that all records with a value of 2 for `related` should be replaced by a value of 0
    # in that category.
    category_cols['related'] = category_cols['related'].apply(lambda x: 0 if x==2 else x)
    # update `df` DataFrame object with the new category columns in `category_cols` DataFrame object
    # -- drop the original column
    df.drop('categories', axis=1, inplace=True)
    # -- concatenate `df` and `category_columns` DataFrame objects
    df = pd.concat([df, category_cols], axis=1)
    # remove duplicates
    df.drop_duplicates(inplace=True)
    # Note: we notice that despite dropping all duplicates, we still have records with identical 'id' values.
    # Further analysis highlighted that they are records (from dataset categories.csv) with identical 'id' but with
    # different values in the category columns.
    # In order to address this, we chose to handle these duplicates by grouping them by 'id' value and taking the
    # maximum value for each category column.
    duplicates_to_handle = df.loc[df.duplicated(subset=['id'], keep=False)]
    handled_duplicates = duplicates_to_handle.groupby(by='id').max()
    handled_duplicates.reset_index(drop=False, inplace=True)
    # -- remove duplicates to handle from `df` DataFrame object, and replace them by the handled duplicates
    df.drop_duplicates(subset=['id'], keep=False, inplace=True)
    df = pd.concat([df, handled_duplicates], axis=0)
    return df


def save_data(df, database_filename):
    """Export the data to a database.

    Args:
        df (pd.DataFrame): DataFrame object whose data is to be
            exported.
        database_filename (str): Filepath to the exported data.

    """
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql(name=database_filename,
              con=engine,
              index=False)


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
