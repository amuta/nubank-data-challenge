import logging
import os
import sys
import pandas as pd
import numpy as np
from importlib import reload
from sklearn.preprocessing import LabelEncoder

data_dir = os.path.dirname(os.path.abspath(__file__)) + '/../../data/'
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    stream=sys.stdout,
                    level=logging.DEBUG)
logger = logging.getLogger('make_dataset')


def get_raw(name, test=True):
    """
    Read and return datasets
    """
    assert name in ['acquisition', 'spend']

    logger.info('Reading ' + name)

    if name == 'acquisition':
        train = pd.read_csv(data_dir + 'raw/acquisition_train.csv')
        logger.info(name + '_train shape: {}'.format(train.shape))
        if test:
            test = pd.read_csv(data_dir + 'raw/acquisition_test.csv')
            logger.info(name + '_test shape: {}'.format(test.shape))
            return train, test
    else:
        train = pd.read_csv(data_dir + 'raw/spend_train.csv')
        logger.info(name + '_train shape: {}'.format(train.shape))
    return train


def acquisition_prepare(task):
    """
    Read and return acquisition datasets.
    """
    assert task in ['default', 'fraud']

    train, test = get_raw('acquisition')

    logger.info('Preprocessing columns.')
    if task == 'default':
        # drop target_fraud
        train.drop('target_fraud', axis=1, inplace=True)

        # remove missing target_default rows
        train.target_default.dropna(inplace=True)

        # change target_default dtype to number
        train.target_default = train.target_default.astype(pd.np.number)
    else:
        # drop target_default
        train.drop('target_default', axis=1, inplace=True)

        # fill target_fraud missing values
        train.target_fraud.fillna('nodefault', inplace=True)

    return train, test


def data_save(path, name, train, test=None):
    """
    Save datasets in given path/name
    """
    logger.info('Saving datasets in {}.'.format(path))
    train.to_pickle(path + name + '_train.pkl')
    if test:
        test.to_pickle(path + name + '_test.pkl')


def encode_columns(columns, train, test=None):
    """
    Encode given columns of dataframes
    """        
    logger.info('Encoding columns.')
    for col in columns:
        le = LabelEncoder()
        if test:
            le.fit(pd.concat([train[col], test[col]]))
            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])
        else:
            train[col] = le.fit_transform(train[col])


def default_base(save=True):
    """
    Make an base-line dataset ready for the default model.
    """
    train, test = acquisition_prepare('default')
    datasets = [train, test]

    # get all categorical columns names
    cat_cols = (train.select_dtypes(exclude=pd.np.number)).columns.values

    for df in datasets:
        # fill missing values with unique value
        df.fillna(-9999, inplace=True)
        # change dtypes of categorical columns to str (for encoding)
        df[cat_cols] = df[cat_cols].astype(str)

    encode_columns(columns=cat_cols, train=train, test=test)

    # save or return datasets
    if save:
        data_save(
            path=os.path.abspath(data_dir + '/processed/'),
            name='default_base',
            train=train,
            test=test)
    else:
        logger.info('Returning datasets')
        return train, test


def fraud_base(save=True):
    """
    Make an base-line dataset ready for the default model.
    """
    train, test = acquisition_prepare('fraud')
    datasets = [train, test]

    # get all categorical columns names
    cat_cols = (train.select_dtypes(exclude=pd.np.number)).columns.values

    for df in datasets:
        # fill missing values with unique value
        df.fillna(-9999, inplace=True)
        # change dtypes of categorical columns to str (for encoding)
        df[cat_cols] = df[cat_cols].astype(str)

    encode_columns(columns=cat_cols, train=train, test=test)

    # save or return datasets
    if save:
        data_save(
            path=os.path.abspath(data_dir + '/processed/'),
            name='fraud_base',
            train=train,
            test=test)
    else:
        logger.info('Returning datasets')
        return train, test


def spend_base(save=True):
    """
    Extends the spends dataset using business metrics
    """
    # reading spends and acquisition data
    data = get_raw('spend')
    acq_data = get_raw('acquisition', False)

    # Business metrics
    interest_rate = 0.17
    interchange_rate = 0.05
    inflation_rate = 0.005
    unit_card_cost = 10
    cs_min_cost = 2.5

    logger.info('Creating new columns.')
    # flag for last month
    max_month = data.groupby('ids').month.transform(max)
    data['last_month'] = max_month == data.month

    # get fraud types ids
    fraud_family_ids = acq_data.ids[acq_data.target_fraud ==
                                    'fraud_friends_family'].values
    fraud_id_ids = acq_data.ids[acq_data.target_fraud == 'fraud_id'].values
    fraud_all_ids = np.concatenate((fraud_family_ids, fraud_id_ids))

    # create fraud type and fraud columns
    data['fraud_type'] = 0
    data.loc[(data.ids.isin(fraud_family_ids)), 'fraud_type'] = 1
    data.loc[(data.ids.isin(fraud_id_ids)), 'fraud_type'] = 2
    data['fraud'] = False
    data.loc[(data.ids.isin(fraud_all_ids)), 'fraud'] = True

    # create bool for ids with default
    default_ids = acq_data[acq_data.target_default == 1].ids.values
    data['default'] = False
    data.loc[data.ids.isin(default_ids), 'default'] = True

    # create business metrics columns
    data['last_revolving'] = (data.groupby('ids')['revolving_balance']
                              .shift(1).fillna(0))
    data['last_revolving_interest'] = data.last_revolving * interest_rate
    data['bill'] = (data.spends + data.last_revolving +
                    data.last_revolving_interest)
    data['interchange_renenue'] = data.spends * interchange_rate
    data['card_cost'] = data.card_request * unit_card_cost
    data['cs_cost'] = data.minutes_cs * cs_min_cost
    data['month_revenue'] = (data.last_revolving_interest +
                             data.interchange_renenue)
    data.loc[(data.default | data.fraud) &
             data.last_month, 'month_revenue'] = 0
    data['fraud_cost'] = data.credit_line * data.fraud * data.last_month
    data['default_cost'] = ((data.bill + data.revolving_balance) *
                            data.default * data.last_month * ~data.fraud)
    data['month_cost'] = (data.cs_cost + data.card_cost + data.fraud_cost +
                          data.default_cost)
    data['month_profit'] = data.month_revenue - data.month_cost
    data['limit_spent_pct'] = data.spends / data.credit_line

    # create discounted columns
    data['discount_factor'] = (1 + inflation_rate) ** data.month
    data['discounted_spends'] = data.spends / data.discount_factor
    data['discounted_revenue'] = data.month_revenue / data.discount_factor
    data['discounted_cost'] = data.month_cost / data.discount_factor
    data['discounted_profit'] = data.discounted_revenue - data.discounted_cost

    # encode all categorical columns
    # for col in cat_cols:
    #     le = LabelEncoder()
    #     le.fit(pd.concat([train[col], test[col]]))
    #     for df in datasets:
    #         df[col] = le.transform(df[col])

    # save or return datasets
    if save:
        data_save(
            path=os.path.abspath(data_dir + '/processed/'),
            name='spend_base',
            train=data)
    else:
        logger.info('Returning spend_base')
        return data


def spend_agg(save=True):
    """
    Create spend behavior data aggregated by id
    """

    # load spend dataset
    data = spend_base(False)

    # groupby by ids and peform aggregations
    grouped_ids = data.groupby('ids')
    behavior_data = pd.concat([
        grouped_ids.discounted_profit.agg('sum'),
        grouped_ids.discounted_spends.agg('sum'),
        grouped_ids.discounted_spends.agg('std'),
        grouped_ids.month.agg('last'),
        grouped_ids.fraud.agg('last'),
        grouped_ids.fraud_type.agg('last'),
        grouped_ids.default.agg('last')], axis=1)
    behavior_data.columns = [
        'profit',
        'spends',
        'spends_std',
        'months',
        'fraud',
        'fraud_type',
        'default']

    # save or return datasets
    if save:
        data_save(
            path=os.path.abspath(data_dir + '/processed/'),
            name='spend_agg',
            train=behavior_data)
    else:
        logger.info('Returning spend_agg')
        return behavior_data
