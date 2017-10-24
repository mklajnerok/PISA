import wbdata
import datetime
import pandas as pd
import pycountry
import statsmodels.formula.api as smf
import numpy as np
import pylab


def read_multi_csv_data(files_list):
    """Read multi csv files from pisaprojectdatafiles directory and
    create df_data dictionary containing separate data frames
    :param files_list: list of csv files
    :returns df_data: dict with name keys and data frames values"""
    df_data = {}
    for f in files_list:
        d = pd.read_csv('pisaprojectdatafiles/{0}'.format(f))
        df_data[f.replace('.csv', '')] = d
    return df_data

def show_dict_summary(dict):
    """Print name, head and tail for each df in a dict
    :param dict: dictionary with name keys and data frames values"""
    for k, v in dict.items():
        print('\n' + k + '\n')
        print(v.head())
        print(v.tail())

def drop_dict_col(dict, del_col):
    """In each df in a dict delete del_col columns
    :param dict: dictionary with name keys and data frames values
    :param del_col: list"""
    for k, v in dict.items():
        for i in del_col:
            v.drop(i, axis=1, inplace=True)

def rename_dict_col(dict, new_col):
    """Rename columns in each df in a dict
    :param dict: dictionary with name keys and data frames values
    :param new_col: list
    """
    for k, v in dict.items():
        v.rename(columns=new_col, inplace=True)

def filter_dict_by_year(dict, year):
    """Create a copy of dict and extract rows with a given year
    :param dict: dictionary with name keys and data frames values
    :param year: int
    :returns dict_year: dictionary with name keys and data frames values
    """
    dict_year = dict.copy()
    for k, v in dict_year.items():
        v = dict_year[k]
        v = v[v['Time'] == year]
        dict_year[k] = v
    return dict_year

def merge_dict_by_year(dict_year):
    """Take dict_year and merge each df in one based on Code, then drop columns with year number
    :param dict_year: dictionary with name keys and data frames values
    :returns df_data_joined: data frame"""
    df_data_joined = pd.DataFrame()
    for k, v in dict_year.items():
        if not df_data_joined.empty:
            df_data_joined = pd.merge(df_data_joined, v, on='Code')
        else:
            df_data_joined = df_data_joined.append(v)
    df_data_joined.drop(df_data_joined.columns[[1, 3, 5]], axis=1, inplace=True)
    return df_data_joined

def name_code_mapper():
    """Use pycountry library to create a map for converting from country name to country code
    :returns map: dict"""
    map = {country.name: country.alpha_3 for country in pycountry.countries}
    map_adjust = {'Czech Republic': 'CZE', 'Hong Kong SAR, China': 'HKG', 'Korea, Rep.': 'KOR',
                      'Macao SAR, China': 'MAC', 'OECD members': 'OED', 'Slovak Republic': 'SVK',
                  'China, Hong Kong Special Administrative Region': 'HKG', 'China, Macao Special Administrative Region': 'MAC',
                  'Republic of Korea': 'KOR', 'United Kingdom of Great Britain and Northern Ireland': 'GBR',
                  'United States of America': 'USA', 'OECD members': 'OAVG'}
    map.update(map_adjust)
    return map

def code_name_mapper(map):
    """Reverse other map for converting from country code to country name
    :param map: dict
    :returns reversed_map: dict"""
    reversed_map = dict(zip(map.values(), map.keys()))
    return reversed_map

def add_country_col(df_data, map):
    """Take df_data, add a column with country name and fill it using map
    :param df_data: data frame
    :param map: dict"""
    mapper = lambda x: map[x]
    df_data.insert(loc=0, column='Country', value=df_data.loc[:, 'Code'].copy())
    df_data['Country'] = df_data['Country'].apply(mapper)
    return df_data

def add_code_col(df_data, map):
    """Take df_data, add a column with country code and fill it using map
    :param df_data: data frame
    :param map: dict"""
    mapper = lambda x: map[x]
    df_data.insert(loc=1, column='Code', value=df_data.loc[:, 'Country'].copy())
    df_data['Code'] = df_data['Code'].apply(mapper)
    return df_data

def get_codes_list(df_data):
    """Create a list of values from given df_data, column Code
    change code for OECD members from OAVG to OED
    :param df_data: dataframe
    :returns codes_list: list"""
    codes_list = df_data['Code'].tolist()
    codes_list.remove('OAVG')
    codes_list.append('OED')
    return codes_list

def api_data(countries, indicators, year_from, year_to):
    """Create data frame for given list of countries ans indicators and dates using World Bank API
    :param countries: list of codes
    :param indicators: dict {ind_code : ind_name}
    :param year_from: starting year
    :param year_to: ending year
    :returns df_data: multiindex df
    """
    data_date = (datetime.datetime(year_from, 1, 1), datetime.datetime(year_to, 1, 1))
    df_data = wbdata.get_dataframe(indicators, country=countries, data_date=data_date, convert_date=False)
    return df_data

def filter_by_year(df_data, year):
    """Create a copy of df_data and extrac rows for a particular year
    :param df_data: data frame
    :param year: str
    :returns df_data_year: data frame"""
    df_data_year = df_data.xs(year, level='date').copy()
    return df_data_year

def merge_df(df_data1, df_data2):
    """Merge two data frames on Code column, drop double country column
    :type df_data1: data frame
    :type df_data2: data frame
    :returns df_joined: data frame"""
    df_joined = pd.merge(df_data1, df_data2, on='Code')
    df_joined.drop(df_joined.columns[[5]], axis=1, inplace=True)
    return df_joined

def take_log(df_data, columns):
    """Takes columns labels and transform values in the to log
    :type data: data frame
    :type columns: list of str"""
    log_data = df_data.copy()
    for i in columns:
        log_data[i] = np.log(log_data[i])
    return log_data

def read_in(file):
    df = pd.read_csv(file)
    return df

def csv_data_by_list(file, codes_list):
    """Read in a file and select rows with countries codes from codes_list
    :type file: str
    :type codes_list: list"""
    df = pd.read_csv(file)
    df_data = pd.DataFrame()
    for i in codes_list:
        df_data = df_data.append(df.loc[df['LOCATION'] == i], ignore_index=True)
    return df_data

def get_some_ind(df_data, indicators):
    """Take df_data and select rows for given indicators, then append it to basic_edu_data
    :type df_data: df
    :type indicators: list"""
    basic_edu_data = pd.DataFrame()
    for i in indicators:
        basic_edu_data = basic_edu_data.append(df_data.loc[df_data['EDULIT_IND'] == i], ignore_index=True)
    return basic_edu_data

def drop_col(df_data, del_col):
    """Take df_data and drop del_col columns from it
    :type df_data: df
    :type del_col: list"""
    for i in del_col:
        df_data.drop(i, axis=1, inplace=True)
    return df_data

def rename_col(df_data, new_col):
    """Take df_data and rename new_col columns
    :type df_data: df
    :type new_col: dict"""
    df_data.rename(columns=new_col, inplace=True)
    return df_data

def estimate_total_cost(df_data):
    """Takes df_data and estimates total cost per student during 12 years of education
    :param df_data: data frame
    :returns total_cost: total_cost df with country Code and total cost"""
    total_cost = pd.DataFrame(columns=['Code', 'Total'])
    country_list = df_data.Code.unique()
    for country in country_list:
        country_df = df_data.loc[df_data['Code'] == country]
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
            country_df[label].interpolate(limit_direction='both', inplace=True)
            col_sum = sum(test_col)
        else:
            col_sum = 0
    else:
        col_sum = sum(test_col)
    return col_sum


def label_plot():
    pylab.title('Measured displacement')
    pylab.xlabel('gdp_ppp')
    pylab.ylabel('pisa_results')

def fit_data(df_data1, df_data2, degree):
    """plot """
    pylab.plot(df_data1, df_data2, 'bo', label='measured points')
    label_plot()
    model = pylab.polyfit(df_data1, df_data2, degree)
    est_y_vals = pylab.polyval(model, df_data1)
    pylab.plot(df_data1, est_y_vals, 'r', label='Curve fit')
    pylab.legend(loc='best')




