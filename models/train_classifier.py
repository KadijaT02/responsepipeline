import re
import sys

import cloudpickle
import nltk
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, hamming_loss, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """Load the data of interest from the database.

    Arg:
        database_filepath (str): Filepath to the database to load data
            from.

    Returns:
        X (pd.DataFrame): Input samples.
        Y (pd.Series): Target values.
        category_names (list): Names of all the categories.

    """
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(table_name="data/DisasterResponse.db", con=engine)
    X = df['message']
    Y = df.loc[:, pd.IndexSlice['related':'direct_report']]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Tokenize `text`.

    :Processing:
        - Detect any URLs and replace them by a placeholder string.
        - Tokenize the text
        - Lemmatize each token before transforming them to lowercase and
          removing any leading and trailing whitespaces.
        - Return the tokens.

    Arg:
        text (str): The text to be tokenized.

    Returns:
        clean_tokens (list): List of the tokens after processing of the
            text.

    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens


def build_model():
    """Build the Machine Learning (ML) pipeline.

    Returns:
        (Pipeline): The ML pipeline.

    """
    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('densify', FunctionTransformer(func=densify)),
            (
                'clf', MultiOutputClassifier(
                    estimator=HistGradientBoostingClassifier(random_state=0)
                )
            )
        ]
    )
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the performance of model `model` on the test data.

    :Args:
        model (Pipeline): The model to be evaluated.
        X_test (pandas.Series): Input samples for testing.
        Y_test (pandas.DataFrame): Target values for testing.
        category_names (list): Names of all the categories.

    :Processing:
        - Predict categories for `X_test`.
        - Loop through the categories and print their associated
          classification reports.
        - Calculate and print the Hamming Loss of the model.

    """
    y_pred = model.predict(X_test)
    # print classification report for each category
    for j in range(y_pred.shape[1]):
        report = classification_report(y_true=Y_test.iloc[:, j].to_list(),
                                       y_pred=[row[j] for row in y_pred],
                                       output_dict=False)
        print(f'Report Cat. "{category_names[j]}"', report)
    # calculate and print Hamming Loss
    hl = hamming_loss(y_true=Y_test, y_pred=y_pred)
    print(f'The Hamming Loss of our model is {hl}.')


def save_model(model, model_filepath):
    """Export built model `model` as pickle file.

    Args:
        model (Pipeline): Built model to be exported.
        model_filepath (str): Path to the filed where `model` will be
            exported.

    """
    with open(model_filepath, 'wb') as f:
        cloudpickle.dump(model, f)


# ADDITIONAL FUNCTION
def densify(X):
    """Return dense representation of a sparse matrix.

    Arg:
        X (scipy.csr_matrix): Sparse matrix.

    Returns:
        (ndarray) Dense representation of sparse matrix `X`.

    Note:
        This function will be used in the `build_pipeline`.
        It is needed because CountVectorizer returns a sparse matrix as output.
        However, the estimator HistGradientBoostingClassifier requires
        a dense representation as input.

        """
    return np.asarray(X.todense())


# ADDITIONAL FUNCTION (Optional): run gridsearch on the model
def gridsearch(model, param_grid, x_train, y_train):
    """Run exhaustive search over specified parameter values for the
    `model` estimator.

    Args:
        model (Pipeline): The Machine Learning pipeline.
        param_grid (dict): Parameter values to run the exhaustive search
            with.
        x_train (pandas.Series): Input samples for training.
        y_train (pandas.DataFrame): Target values for training.

    Returns:
        (sklearn.pipeline): The estimator chosen by the cross-validated
            grid search.

    """
    gs = GridSearchCV(estimator=model,
                      param_grid=param_grid,
                      scoring=make_scorer(score_func=hamming_loss,
                                          greater_is_better=False))
    gs.fit(x_train, y_train)
    return gs.best_estimator_


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

        # ADDITIONAL STEP (Optional): run gridsearch on the model
        param_grid = {'clf__estimator__learning_rate': [0.1, 0.2, 0.3],
                      'clf__estimator__max_iter': [85, 100, 115]}
        best_model = gridsearch(model=model,
                                param_grid=param_grid,
                                x_train=X_train,
                                y_train=Y_train)
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
