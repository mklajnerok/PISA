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


### 3A/ perform OLS between PISA results (3 separate subjects) and GDP PPP data

#merge data
pisa_gdp_ppp = merge_df(all_pisa_2015, gdp_ppp_2015)

#rename column label
rename_col(pisa_gdp_ppp, {'Country_x': 'Country'})

#take log from GDP values
pisa_gdp_ppp_log = take_log(pisa_gdp_ppp,['gdp_ppp'])

#perform OLS
model_math_gdp_ppp_log = smf.ols(formula='math ~ gdp_ppp', data=pisa_gdp_ppp_log).fit()
model_read_gdp_ppp_log = smf.ols(formula='read ~ gdp_ppp', data=pisa_gdp_ppp_log).fit()
model_scie_gdp_ppp_log = smf.ols(formula='science ~ gdp_ppp', data=pisa_gdp_ppp_log).fit()

#show summary
model_math_gdp_ppp_log.summary()
model_read_gdp_ppp_log.summary()
model_scie_gdp_ppp_log.summary()

#plot
plot_math_gdp_ppp_log = show_scatterplot(pisa_gdp_ppp_log, ['gdp_ppp', 'math'], 'r')
plot_read_gdp_ppp = show_scatterplot(pisa_gdp_ppp_log, ['gdp_ppp', 'read'], 'b')
plot_scie_gdp_ppp_log = show_scatterplot(pisa_gdp_ppp_log, ['gdp_ppp', 'science'], 'g')

# try to create a function to perform ols automatically
# try to create a function to plot many things at once

import seaborn as sns

def fit_data_sea(df_data1, df_data2, order, x_label, y_label):
    """Plot data from df_data1 and df_data2 and try to fit a curve with a given degree using seaborn
    :param df_data1: data frame
    :param df_data2: data frame
    :param x_label: str
    :param y_label: str"""
    sns.regplot(df_data1, df_data2, order=order, line_kws={"color": "r", "alpha": 0.1, "lw": 5})
    plt.title('Measured displacement')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig('plots/with_curve.png')

lin_ave_gdp_ppp_log_pisa_2 = fit_data_sea(pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'], 1,
                                      'gdp per capita', 'average pisa result')

def show_bubbles(df_data, variables, color):
    plt.scatter(x=df_data[variables[0]],y=df_data[variables[1]], s=z*1000, alpha=0.5)
    plt.show()


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


#check number on Nan in edu_data_per_student
index1 = np.where(edu_data_per_student['pre_primary_per_student'].isnull())[0]
# pre_primary => CAN, GRC, MAC, OAVG, SGP, TUR
# primary =>  CAN, GRC, HKG, MAC, OAVG, SGP, TUR
# secondary => CAN, GRC, HKG, MAC, OAVG, SGP
# sum = [CAN, GRC, HKG, MAC, OAVG, SGP, TUR] [ISR, PER, SVN]
# maybe ISR, PER have different ed system, there's a huge lack for secondary, maybe extrapolation for SVN

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

#taking log
#https://stats.stackexchange.com/questions/298/in-linear-regression-when-is-it-appropriate-to-use-the-log-of-an-independent-va

### 7A/ make OLS for total education cost and PISA results form 2015 (3 separate subjects)

#add column with country name
student_total_expenses = add_country_col(student_total_expenses, code_name_map)

#merge with PISA results
pisa_expenses = merge_df(all_pisa_2015, student_total_expenses)

#rename column label
rename_col(pisa_expenses, {'Country_x': 'Country'})

#perform OLS
model_math_expenses = smf.ols(formula='math ~ Total', data=pisa_expenses).fit()
model_read_expenses = smf.ols(formula='read ~ Total', data=pisa_expenses).fit()
model_scie_expenses = smf.ols(formula='science ~ Total', data=pisa_expenses).fit()

#show summary
model_math_expenses.summary()
model_read_expenses.summary()
model_scie_expenses.summary()

#plot
plot_math_expenses = show_scatterplot(pisa_expenses, ['Total', 'math'], 'r')
plot_read_expenses = show_scatterplot(pisa_expenses, ['Total', 'read'], 'b')
plot_scie_expenses = show_scatterplot(pisa_expenses, ['Total', 'science'], 'g')



"""Getting GDP data:
- from World bank API (GDP PPP $, GDP US$ )
- for the second part of analysis from 
http://data.uis.unesco.org/
Government expenditure on education as a percentage of GDP (z podzialem na poziomy edukacji)
Government expenditure on education in PPP$ (z podzialem na poziomy edukacji)
Government expenditure on education in US$ (z podzialem na poziomy edukacji)
explaining levels of education http://www.unesco.org/education/information/nfsunesco/doc/isced_1997.html"""


import six
from pandas.plotting import table

def render_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for k, cell in six.iteritems(table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

render_table(df, header_columns=0, col_width=2.0)

# set fig size
fig, ax = plt.subplots(figsize=(40, 3))
# no axes
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
# no frame
ax.set_frame_on(False)
# plot table
tab = table(ax, all_pisa_2015_ave, loc='upper right')
# set font manually
#tab.auto_set_font_size(False)
#tab.set_fontsize(8)
# save the result
plt.savefig('test5_table.png')



# zrobic histogramy z rozkladem zmiennych wyniki pisa i gdp kraju
# poten scatterplot


# przykład z handbooka jak dopasowac polynomial 7 stopnia dodanych
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
poly_model = make_pipeline(PolynomialFeatures(7),LinearRegression())

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = np.sin(x) + 0.1 * rng.randn(50)

poly_model.fit(x[:, np.newaxis], y)
xfit = np.linspace(0, 10, 1000)
yfit = poly_model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit);