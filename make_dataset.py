import os
import pandas as pd
import pycountry
import wbdata
import datetime
import statsmodels.formula.api as smf
import numpy as np
import pylab

def read_multi_csv_data(files_list):
    """Read multi csv files from data_files directory and
    create df_dict dictionary containing separate data frames
    :param files_list: list of csv files
    :returns df_data: dictionary with string keys and data frames values"""
    df_dict = {}
    for f in files_list:
        d = pd.read_csv('data_files/{0}'.format(f))
        df_dict[f.replace('.csv', '')] = d
    return df_dict

def show_dict_summary(df_dict):
    """Print name, head and tail for each data frame in a df_dict dictionary
    :param df_dict: dictionary with string keys and data frames values"""
    for k, v in df_dict.items():
        print('\n' + k + '\n')
        print(v.head())
        print(v.tail())

def drop_dict_columns(df_dict, del_col):
    """In each data frame in a df_dict dictionary delete del_col columns
    :param df_dict: dictionary with string keys and data frames values
    :param del_col: list"""
    for k, v in df_dict.items():
        for i in del_col:
            v.drop(i, axis=1, inplace=True)

def rename_dict_columns(df_dict, new_col):
    """In each data frame in a df_dict dictionary rename columns
    :param df_dict: dictionary with string keys and data frames values
    :param new_col: dictionary
    """
    for k, v in df_dict.items():
        v.rename(columns=new_col, inplace=True)

def filter_dict_by_year(df_dict, year):
    """Create a copy of df_dict and extract rows for a given year
    :param df_dict: dictionary with string keys and data frames values
    :param year: int
    :returns df_dict_year: dictionary with string keys and data frames values
    """
    df_dict_year = df_dict.copy()
    for k, v in df_dict_year.items():
        v = df_dict_year[k]
        v = v[v['Time'] == year]
        df_dict_year[k] = v
    return df_dict_year

def merge_dict_by_year(df_dict_year):
    """Take df_dict_year and merge each data frame in one based on Code,
    then drop columns with year number
    :param df_dict_year: dictionary with string keys and data frames values
    :returns df_data_joined: data frame"""
    df_data_joined = pd.DataFrame()
    for k, v in df_dict_year.items():
        if not df_data_joined.empty:
            df_data_joined = pd.merge(df_data_joined, v, on='Code')
        else:
            df_data_joined = df_data_joined.append(v)
    df_data_joined.drop(df_data_joined.columns[[1, 3, 5]], axis=1, inplace=True)
    return df_data_joined

def rename_columns(df_data, new_col):
    """Take df_data data frame and rename new_col columns
    :param df_data: data frame
    :param new_col: dictionary"""
    df_data.rename(columns=new_col, inplace=True)

def create_name_code_dict():
    """Use pycountry library to create a map for converting from country name to country code
    :returns name_code_dict: dictionary"""
    name_code_dict = {country.name: country.alpha_3 for country in pycountry.countries}
    dict_adjust = {'Czech Republic': 'CZE', 'Hong Kong SAR, China': 'HKG', 'Korea, Rep.': 'KOR',
                      'Macao SAR, China': 'MAC', 'OECD members': 'OED', 'Slovak Republic': 'SVK',
                  'China, Hong Kong Special Administrative Region': 'HKG', 'China, Macao Special Administrative Region': 'MAC',
                  'Republic of Korea': 'KOR', 'United Kingdom of Great Britain and Northern Ireland': 'GBR',
                  'United States of America': 'USA', 'OECD members': 'OAVG'}
    name_code_dict.update(dict_adjust)
    return name_code_dict

def reverse_dict(dictionary):
    """Reverse other map for converting from country code to country name
    :param dictionary: dictionary
    :returns reversed_dict: dictionary"""
    reversed_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return reversed_dict

def add_country_name(df_data, code_name_dict):
    """Take df_data, add a column with country name and fill it using code_name_dict
    :param df_data: data frame
    :param code_name_dict: dictionary"""
    mapper = lambda x: code_name_dict[x]
    df_data.insert(loc=0, column='Country', value=df_data.loc[:, 'Code'].copy())
    df_data['Country'] = df_data['Country'].apply(mapper)
    return df_data

def get_average(df_data):
    """Takes a copy of df_data and calculate average pisa result for a given country,
    with formula: (math+read+2*science)/4
    :param df_data: data frame
    :returns df_data_new: data frame
    """
    df_data_new = df_data.copy()
    df_data_new.insert(loc=2, column='ave_result', value=0)
    df_data_new['ave_result'] = round((df_data['math'] + df_data['read'] + df_data['science']) / 3, 0)
    df_data_new.drop(['math', 'read', 'science'], axis=1, inplace=True)
    return df_data_new

def get_codes_list(df_data):
    """Create a list of countries codes from column Code in df_data
    change code for OECD members from OAVG to OED
    :param df_data: data frame
    :returns codes_list: list"""
    codes_list = df_data['Code'].tolist()
    codes_list.remove('OAVG')
    codes_list.append('OED')
    return codes_list

def load_from_wbdata(countries, indicators, year_from, year_to):
    """Create data frame for given list of countries, indicators and dates using World Bank API
    :param countries: list of codes
    :param indicators: dict {ind_code : ind_name}
    :param year_from: starting year
    :param year_to: ending year
    :returns df_data: multi index data frame
    """
    data_date = (datetime.datetime(year_from, 1, 1), datetime.datetime(year_to, 1, 1))
    df_data = wbdata.get_dataframe(indicators, country=countries, data_date=data_date, convert_date=False)
    return df_data

def filter_by_year(df_data, year):
    """Create a copy of df_data and extract rows for a given year
    :param df_data: data frame
    :param year: string
    :returns df_data_year: data frame"""
    df_data_year = df_data.xs(year, level='date').copy()
    return df_data_year

def add_country_code(df_data, name_code_dict):
    """Take df_data, add a column with country code and fill it using name_code_dict
    :param df_data: data frame
    :param name_code_dict: dictionary"""
    mapper = lambda x: name_code_dict[x]
    df_data.insert(loc=1, column='Code', value=df_data.loc[:, 'Country'].copy())
    df_data['Code'] = df_data['Code'].apply(mapper)
    return df_data

def merge_df(df_data1, df_data2):
    """Merge two data frames on Code column, drop double country column
    :param df_data1: data frame
    :param df_data2: data frame
    :returns df_joined: data frame"""
    df_joined = pd.merge(df_data1, df_data2, on='Code')
    df_joined.drop(['Country_y'], axis=1, inplace=True)
    return df_joined

def take_log(df_data, columns):
    """Takes values from columns labels and transform them to log
    :param df_data: data frame
    :param columns: list of str"""
    log_data = df_data.copy()
    for i in columns:
        log_data[i] = np.log(log_data[i])
    return log_data

def read_in(file):
    df = pd.read_csv(file)
    return df

def read_csv_data_by_list(file, codes_list):
    """Read in a file from pisaprojectdatafiles directory and select rows with countries codes from codes_list
    :param file: str
    :param codes_list: list"""
    df = pd.read_csv('pisaprojectdatafiles/{0}'.format(file))
    df_data = pd.DataFrame()
    for i in codes_list:
        df_data = df_data.append(df.loc[df['LOCATION'] == i], ignore_index=True)
    return df_data

def filter_by_indicator(df_data, indicators):
    """Take df_data and select rows for given indicators, then append it to df_data_ind
    :param df_data: data frame
    :param indicators: list"""
    df_data_ind = pd.DataFrame()
    for i in indicators:
        df_data_ind = df_data_ind.append(df_data.loc[df_data['EDULIT_IND'] == i], ignore_index=True)
    return df_data_ind

def drop_columns(df_data, del_col):
    """Take df_data and drop del_col columns from it
    :param df_data: data frame
    :param del_col: list"""
    for i in del_col:
        df_data.drop(i, axis=1, inplace=True)
    return df_data

def divide_col_by_col(df_data, dividends, divisors):
    """Takes df_data and divides columns accordingly by rule dividends/divisors,
    returns df_data_div only with divided columns
    :param df_data: data frame
    :param dividends: list of column labels
    :param divisors: list of column labels
    :returns: df_data_div: data frame
    """
    df_data_div = df_data.copy()
    for n in range(len(dividends)):
        df_data_div[n] = df_data_div[dividends[n]] / df_data_div[divisors[n]] * 1000000
    drop_col(df_data_div, [dividends + divisors])
    df_data_div.round(0)
    return df_data_div

def estimate_total_cost(df_data):
    """Takes df_data and estimates total cost per student during 12 years of education
    :param df_data: data frame
    :returns total_cost: total_cost df with country Code and total cost"""
    total_cost = pd.DataFrame(columns=['Code', 'Total'])
    country_list = df_data.Code.unique()
    for country in country_list:
        country_df = df_data.loc[df_data['Code'] == country].copy()
        cost_pre_primary = sum_col(country_df, 0, 3, 'pre_primary_per_student')
        cost_primary = sum_col(country_df, 3, 9, 'primary_per_student')
        cost_second = sum_col(country_df, 9, 12, 'lower_sec_per_student')
        cost_temp = [cost_pre_primary, cost_primary, cost_second]
        for i in cost_temp:
            if i > 0:
                sum_temp = sum(cost_temp)
            else:
                sum_temp = 0
        total_cost = total_cost.append(pd.DataFrame([[country, sum_temp]], columns=['Code', 'Total']),
                                               ignore_index=True)
    return total_cost

def sum_col(country_df, index_start, index_end, label):
    """Helper function for estimate_total_cost, which calculate total cost for a single country
    :param country_df: data frame
    :param index_start: int
    :param index_end: int
    :param label: str
    :returns col_sum: int"""
    test_col = country_df[index_start:index_end][label]
    small_ratio = test_col.isnull().sum() / len(test_col.index)
    big_ratio = country_df[label].isnull().sum() / len(country_df.index)
    if small_ratio > 0:
        if big_ratio < 0.5:
            country_df.loc[:,label].interpolate(limit_direction='both', inplace=True)
            col_sum = sum(test_col)
        else:
            col_sum = 0
    else:
        col_sum = sum(test_col)
    return col_sum







