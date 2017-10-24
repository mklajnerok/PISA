def clean_pisa(pisa_results):
    """In each df in a df_pisa dict delete columns:
    indicator, subject, measure, freq, flag codes
    and change column name from 'value' to math, read, science accordingly
    :type pisa_results: dict"""
    del_col = ['INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'Flag Codes']
    for k, v in pisa_results.items():
        for i in del_col:
            del v[i]
        v = v.rename(columns = {'LOCATION': 'COUNTRY CODE', 'Value': k[5:9], 'TIME': 'YEAR'}, inplace=True)


def column_to_index(df_data, columns):
    """takes data frame and converts its n first columns to indexes
    :type df_data: data frame
    :type columns: list o column's labels to convert"""
    df_data = df_data.set_index(columns, inplace=True)
    return df_data
#column_to_index(all_pisa_2015, ['COUNTRY CODE', 'COUNTRY NAME'])


"""WB API indicators summary:
    SE.XPD.SECO.PC.ZS	Government expenditure per student, secondary (% of GDP per capita) - drop for now
    SE.XPD.PRIM.PC.ZS	Government expenditure per student, primary (% of GDP per capita) - drop for now

    NY.GDP.PCAP.KD     GDP per capita (constant 2010 US$)
    unused:
    UIS.XUNIT.PPP.2.FSGOV   Government expenditure per lower secondary student (PPP$) -> okolo polowy danych to NaN
    UIS.E.2.PU              Enrolment in lower secondary education, public institutions, both sexes (number)
    UIS.NER.2              	Net enrolment rate, lower secondary, both sexes (%)
    SE.SEC.ENRL.LO.TC.ZS    Pupil-teacher ratio, lower secondary
    3.8_LOW.SEC.STUDENTS    Lower secondary education, pupils, national source"""


table_gdp_usd = merge_pisa_gdp(all_pisa_2015, countries_gdp_2015, ['COUNTRY CODE', 'GDP_USD'])
table_gdp_usd_log = take_log(table_gdp_usd,['GDP_USD'])


model_2m = smf.ols(formula='MATH ~ GDP_USD', data=table_gdp_usd_log).fit()
model_2r = smf.ols(formula='READ ~ GDP_USD', data=table_gdp_usd_log).fit()
model_2s = smf.ols(formula='SCIE ~ GDP_USD', data=table_gdp_usd_log).fit()


"""try to create a function for performing OLS
def perform_ols(data, variables):
    takes two variables x and y and perform regression, prints summary
    :type var_x, var_y: df columns
    :type data: dataframe
    for i in variables:
        model_str(i) = smf.ols(formula= i + ' ~ ' + )

    model = smf.ols(formula=variables, data=data).fit()
    return model.summary()

test_ols = perform_ols('MATH ~ GDP_PPP', table_gdp_ppp_log)"""


plot_usd_math = show_scatterplot(table_gdp_usd_log, ['GDP_USD', 'MATH'], 'r')
plot_usd_read = show_scatterplot(table_gdp_usd_log, ['GDP_USD', 'READ'], 'b')
plot_usd_scie = show_scatterplot(table_gdp_usd_log, ['GDP_USD', 'SCIE'], 'g')


"""try to create a function for plotting a few in one time with proper labels
def show_multi_scatterplot(df_data, indicators, colors):
    takes df_data and plots for chosen columns in variables list
    :type df_data: data frame
    :type variables: list of str
    :type color: str

    pd.table_gdp_ppp_log.plot(x=df_data['GDP_PPP'], y=df_data['MATH'], kind='scatter', subplots=True,
                              title='GDP PPP vs. PISA results')
    for i in range(len(indicators)):
        plt.scatter(x=df_data[i], y=df_data[i+1]], color=colors[0])
        plt.scatter(x=df_data[variables[2]], y=df_data[variables[3]], color=color[1])
        plt.title(variables[0] + ' vs. PISA ' + variables[1:] + ' results for 2015')
        plt.xlabel(variables[0])
        plt.ylabel(variables[1:])
        plt.show()
    return"""


#delete index name - nic z tego nie dziala (dalej jest EDULIT_IND)
#basic_edu_exp.index.name = None
#basic_edu_exp.drop('EDULIT_IND', axis=0, inplace=True)


def total_student_cost(df_data):
    """Takes df_data and estimates total cos per student during 12 years of education
    :returns total_cost: total_cost df with country Code and total cost"""
    codes_list = df_data.Code.unique()
    total_cost = pd.DataFrame(columns=['Code', 'Total'])
    for i in codes_list:
        df_temp = df_data.loc[df_data['Code'] == i]
        col_1 = df_temp[:3]['pre_primary_per_student']
        col_2 = df_temp[3:9]['primary_per_student']
        col_3 = df_temp[9:12]['lower_sec_per_student']
        col_list = [col_1, col_2, col_3]
        cost_temp = 0
        for col in col_list:
            sum_col = 0
            col_null_sum = col.isnull().sum()
            if col_null_sum > 0:
                ratio = col_null_sum/len(col.index)
                for j in range(len(col.index)):
                    if not np.isnan(col.iloc[j]):
                        sum_col += col.iloc[j]
                sum_col *= 1/ratio
            else:
                sum_col = sum(col)
            cost_temp += sum_col
        total_cost = total_cost.append(pd.DataFrame([[i, cost_temp]], columns=['Code', 'Total']), ignore_index=True)
    return total_cost

def total_student_cost_v2(df_data):
    """Takes df_data and estimates total cos per student during 12 years of education
    :returns total_cost: total_cost df with country Code and total cost"""
    codes_list = df_data.Code.unique()
    total_cost = pd.DataFrame(columns=['Code', 'Total'])
    for i in codes_list:
        df_temp = df_data.loc[df_data['Code'] == i]
        col_1 = df_temp[:3]['pre_primary_per_student']
        col_2 = df_temp[3:9]['primary_per_student']
        col_3 = df_temp[9:12]['lower_sec_per_student']
        col_list = [col_1, col_2, col_3]
        cost_temp = 0
        for col in col_list:
            sum_col = 0
            col_null_sum = col.isnull().sum()
            ratio = col_null_sum / len(col.index)
            if ratio >= 5/6:
                cost_temp = 0
                break
            elif ratio == 0:
                sum_col = sum(col)
            else:
                for j in range(len(col.index)):
                    if not np.isnan(col.iloc[j]):
                        sum_col += col.iloc[j]
                        sum_col *= 1/ratio
            cost_temp += sum_col
        total_cost = total_cost.append(pd.DataFrame([[i, cost_temp]], columns=['Code', 'Total']), ignore_index=True)
    return total_cost