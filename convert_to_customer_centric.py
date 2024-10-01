from utils.global_variables import COMPANIES, CATEGORICAL_FEATURES
import pandas as pd
import numpy as np
from tqdm import tqdm 
import os

def load_data(company):

    if company not in COMPANIES:
        raise KeyError(f'''
        Company not available within top 20. Please pick one among the followings:
         {COMPANIES}''')
  
    full_dataset_path = 'extracted/transactions.csv'
    singular_company = f'segmented_data/{company}.csv'

    if os.path.isfile(singular_company):
        df = pd.read_csv(singular_company)
    else:
        df_list = []
        CHUNK_SIZE = 10**6
        for chunk in tqdm(pd.read_csv(full_dataset_path, chunksize=CHUNK_SIZE)):
            df_list.append(chunk.query(f'company == {company}'))
        df = pd.concat(df_list, axis=0)
        try:
            df.to_csv(singular_company, index=False)
        except:
            os.mkdir('segmented_data')
            df.to_csv(singular_company, index=False)
    return df

def turn_data_to_customer_centric(df):

    df = df[df['purchaseamount'] > 0] #Only entries with purchases, negative means returned goods!


    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['start_date'] = df.groupby('id')['date'].transform('min')


    calibration_value = df.query('date==start_date').groupby('id')['purchaseamount'].sum().reset_index(name='calibration_value')


    holdout_value = df[(df['date'] > df['start_date']) &
     (df['date'] <= df['start_date'] + np.timedelta64(365, 'D'))].groupby('id')['purchaseamount'].sum().reset_index(name='holdout_value')


    firstday_attributes = (
      df.query('date==start_date').sort_values(
          'purchaseamount', ascending=False).groupby('id')[[
              'chain', 'dept', 'category', 'brand', 'productmeasure'
          ]].first().reset_index())

    customer_data = firstday_attributes.merge(calibration_value, how='left', on='id').merge(holdout_value, how='left', on='id')
    customer_data['holdout_value'] = customer_data['holdout_value'].fillna(0.)
    customer_data[CATEGORICAL_FEATURES] = customer_data[CATEGORICAL_FEATURES].fillna('Not Given')


    # Final transformations

    customer_data['log_calibration_value'] = np.log(customer_data['calibration_value']).astype('float32')
    for col in CATEGORICAL_FEATURES:
        customer_data[col] = customer_data[col].astype('category')
    customer_data['label'] = customer_data['holdout_value'].astype('float32')


    return customer_data

def _saveCompany_asCSV(company):
    '''
    Utility function to load a company's data, convert it to customer-centric, and save it in CSV.
    '''
    if not isinstance(company, str):
        raise TypeError(f"Company must be 'str', but got {type(company)} instead.")
    else:
        singular_company = f'segmented_data/{company}.csv'
        
        if os.path.isfile(singular_company):
            df = pd.read_csv(singular_company)
            df = turn_data_to_customer_centric(df)
        else:
            df = turn_data_to_customer_centric(load_data(company))

        df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype('category')

        for col in [
          'log_calibration_value', 'calibration_value', 'holdout_value'
      ]:
            df[col] = df[col].astype('float32')


        df.drop(['id', 'holdout_value'], axis=1, inplace=True)
        
        df.to_csv(singular_company, index=False)
      
    return df


def load_customer_level_data(company=None):

    '''
    Utility function to convert transactions-based data into customer-level data through preprocessing steps.
    Arguments:

    - company: None or str: The company for transactions denoted by their ID. Default None.
    List of companies: ['10000', '101200010', '101410010', '101600010', '102100020', '102700020', '102840020', '103000030', '103338333', '103400030', '103600030', '103700030', '103800030', '104300040', '104400040', '104470040', 
    '104900040', '105100050', '105150050', '107800070']
    '''
    if not company:
        print("Defaulting to saving every of the top 20 companies. This will take 3 hours estimated.")
        for company in COMPANIES:
            _saveCompany_asCSV(company)
    else:
        if not isinstance(company, str):
            raise TypeError(f'Expected str input, got {type(company)} instead.')
        else:
            df = _saveCompany_asCSV(company)
            return df

        