import logging
import os
import sys
import pandas as pd
from importlib import reload
from sklearn.preprocessing import LabelEncoder

data_dir = os.path.dirname(os.path.abspath(__file__)) + '/../../data/'
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    stream=sys.stdout,
                    level=logging.DEBUG)
logger = logging.getLogger('make_dataset')


def acquisition_read():
    """
    Read and return acquisition datasets.
    """
    logger.info('Reading datasets.')
    train = pd.read_csv(data_dir + 'raw/acquisition_train.csv')
    test = pd.read_csv(data_dir + 'raw/acquisition_test.csv')
    logger.info('Train shape {}, Test shape {}'
                ''.format(train.shape, test.shape))
    return train, test


def save_datasets(path, name, train, test=None):
    """
    Save datasets in given path/name
    """
    logger.info('Saving datasets in {}.'.format(path))
    train.to_pickle(path + name + '_train.pkl')
    if test:
        test.to_pickle(path + name + '_test.pkl')


def default_base(save=True):
    """
    Make an base-line dataset ready for the default model.
    """
    train, test = acquisition_read()
    datasets = [train, test]

    logger.info('Preprocessing columns.')

    # drop target_fraud
    train.drop('target_fraud', axis=1, inplace=True)
    # remove missing target_default rows
    train.target_default.dropna(inplace=True)
    # change target_default dtype to number
    train.target_default = train.target_default.astype(pd.np.number)

    # get all categorical columns names
    obj_cols = (train.select_dtypes(exclude=pd.np.number)).columns.values

    for df in datasets:
        # fill missing values with unique value
        df.fillna(-9999, inplace=True)
        # change dtypes of categorical columns to str (for encoding)
        df[obj_cols] = df[obj_cols].astype(str)

    # encode all categorical columns
    for col in obj_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]))
        for df in datasets:
            df[col] = le.transform(df[col])

    # save or return datasets
    if save:
        save_datasets(path=os.path.abspath(data_dir + '/processed/'))
    else:
        logger.info('Returning datasets')
        return (train, test)
