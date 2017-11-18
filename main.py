### ### ### I ETAP ### ### ###

### 1/ get PISA test results for 2015

#change directory
os.chdir('/Users/wikia/PycharmProjects/PISA')

#read in files with PISA results, separate for every subject (all years)
pisa_data = read_multi_csv_data(['pisa_math_2003_2015.csv', 'pisa_read_2000_2015.csv', 'pisa_science_2006_2015.csv'])

#show summary of PISA results
show_dict_summary(pisa_data)

#drop unnecessary columns
drop_dict_columns(pisa_data, ['INDICATOR', 'SUBJECT', 'MEASURE', 'FREQUENCY', 'Flag Codes'])

#rename columns
rename_dict_columns(pisa_data, {'LOCATION': 'Code', 'Value': 'test_score', 'TIME': 'Time'})

#extract PISA results for 2015
pisa_2015 = filter_dict_by_year(pisa_data, 2015)

#show summary for 2015
show_dict_summary(pisa_2015)

#join all tests' results from 2015
all_pisa_2015 = merge_dict_by_year(pisa_2015)

#rename column labels
rename_columns(all_pisa_2015, {'test_score_x': 'math', 'test_score_y': 'read', 'test_score': 'science'})

#add column with country name
name_code_dict = create_name_code_dict()
code_name_dict = reverse_dict(name_code_dict)
add_country_name(all_pisa_2015, code_name_dict)

#get average pisa result for every country
all_pisa_2015_ave = get_average(all_pisa_2015)

#show results in ascending order
all_pisa_2015_ave.sort_values(['ave_result'], ascending=False)



### 2/ get countries' GDP data for 2015

#get list of countries, who took PISA test
countries_codes = get_codes_list(all_pisa_2015)

#get GDP PPP data (NY.GDP.PCAP.PP.KD - GDP per capita, PPP (constant 2011 international $))
gdp_ppp = load_from_wbdata(countries_codes, {'NY.GDP.PCAP.PP.KD':'gdp_ppp'}, 2003, 2015)

#get GDP PPP for 2015
gdp_ppp_2015 = filter_by_year(gdp_ppp, '2015')

#reset index "country'
gdp_ppp_2015.reset_index(level=['country'], inplace=True)

#rename column label
rename_columns(gdp_ppp_2015, {'country': 'Country'})

#add column with country code
add_country_code(gdp_ppp_2015, name_code_dict)


### 3/ perform OLS between PISA results (3 separate subjects) and GDP PPP data

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


### 3A/ perform OLS between PISA average results and GDP PPP data

#merge data
pisa_ave_gdp_ppp = merge_df(all_pisa_2015_ave, gdp_ppp_2015)

#rename column label
rename_col(pisa_ave_gdp_ppp, {'Country_x': 'Country'})

#take log from GDP values
pisa_ave_gdp_ppp_log = take_log(pisa_ave_gdp_ppp, ['gdp_ppp'])
rename_col(pisa_ave_gdp_ppp_log, {'gdp_ppp': 'gdp_ppp_log'})

#leave LUX out as an outlier
pisa_ave_gdp_ppp_log_lux = pisa_ave_gdp_ppp_log[pisa_ave_gdp_ppp_log['Code'] != 'LUX']

#perform OLS
model_ave_gdp_ppp = smf.ols(formula='ave_result ~ gdp_ppp', data=pisa_ave_gdp_ppp).fit()
model_ave_gdp_ppp_log = smf.ols(formula='ave_result ~ gdp_ppp_log', data=pisa_ave_gdp_ppp_log).fit()
model_ave_gdp_ppp_log_lux = smf.ols(formula='ave_result ~ gdp_ppp_log', data=pisa_ave_gdp_ppp_log_lux).fit()

#show summary
model_ave_gdp_ppp.summary()
model_ave_gdp_ppp_log.summary()
model_ave_gdp_ppp_log_lux.summary()

#plot
plot_ave_gdp_ppp = show_scatterplot(pisa_ave_gdp_ppp, ['gdp_ppp', 'ave_result'], 'y')
plot_ave_gdp_ppp_log = show_scatterplot(pisa_ave_gdp_ppp_log, ['gdp_ppp_log', 'ave_result'], 'm')

#fit data
lin_ave_gdp_ppp_log_pisa = fit_data_mat(pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'], 1,
                                    'gdp per capita', 'average pisa result')

lin_ave_gdp_ppp_log_pisa_2 = fit_data_sea(pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'], 1,
                                      'gdp per capita', 'average pisa result')



### ### ### II ETAP ### ### ###

### 4/ get government expenses on education on pre-primary, primary and lower-secondary levels

#read in file and select given countries - source UNESCO database
gov_edu_expenses = csv_data_by_list('gov_exp_edu_ppp.csv', countries_codes)

#select indicators: pre-primary, primary and lower secondary
basic_edu_exp = get_some_ind(gov_edu_expenses, ['X_PPP_02_FSGOV', 'X_PPP_1_FSGOV', 'X_PPP_2_FSGOV'])

#drop unnecessary column
basic_edu_exp = drop_col(basic_edu_exp, ['TIME', 'Flag Codes', 'Flags'])

#pivot df to get similar structure like student_count_data
basic_edu_exp = basic_edu_exp.pivot_table('Value', ['Country','Time'], 'EDULIT_IND')

#reset multiindex "Country' and 'Time'
basic_edu_exp.reset_index(level=['Country', 'Time'], inplace=True)

#rename column labels
basic_edu_exp = rename_col(
    basic_edu_exp, { 'X_PPP_02_FSGOV': 'pre_primary_exp', 'X_PPP_1_FSGOV': 'primary_exp', 'X_PPP_2_FSGOV': 'lower_sec_exp'})

#add new column with country code
basic_edu_exp = add_code_col(basic_edu_exp, name_code_map)


### 5/ get population number from pre-primary, primary and lower-secondary levels

#read in file with student count for given countries
edu_indicators = {'SP.PRE.TOTL.IN':'pre_primary_pop', 'SP.PRM.TOTL.IN':'primary_pop', 'SP.SEC.LTOT.IN':'lower_sec_pop'}
"""indicators summary:
SP.PRE.TOTL.IN   	Population of the official age for pre-primary education, both sexes (number)
SP.PRM.TOTL.IN   	Population of the official age for primary education, both sexes (number)
SP.SEC.LTOT.IN   	Population of the official age for lower secondary education, both sexes (number)
"""
basic_student_pop = api_data(countries_codes, edu_indicators, 2003, 2014)

#reset multiindex "country' and 'date'
basic_student_pop.reset_index(level=['country', 'date'], inplace=True)

#rename column labels
basic_student_pop = rename_col(basic_student_pop, {'country': 'Country', 'date': 'Time'})

#change column order
basic_student_pop = basic_student_pop[['Country', 'Time', 'pre_primary_pop', 'primary_pop', 'lower_sec_pop']]

#add new column with country code
basic_student_pop = add_code_col(basic_student_pop, name_code_map)


### 6/ estimate average spending for education per student until he takes the test in 2015 (US$)

#convert 'Time' column type to int
basic_edu_exp['Time'] = basic_edu_exp['Time'].apply(np.int16)
basic_student_pop['Time'] = basic_student_pop['Time'].apply(np.int16)

#merge basic_edu_exp and basic_student_pop, right method to stay with 2003-2014 period
edu_data_joined = pd.merge(basic_edu_exp, basic_student_pop, how='right', on=['Code', 'Time'])

#sort by 'Code' and 'Time' and reset index
edu_data_joined.sort_values(['Code', 'Time'], ascending=[True, True], inplace=True)
edu_data_joined.reset_index(level=0, drop=True, inplace=True)

#divide total expenses by number of students
edu_data_per_student = divide_col_by_col(edu_data_joined, ['pre_primary_exp', 'primary_exp', 'lower_sec_exp'],
                                         ['pre_primary_pop', 'primary_pop', 'lower_sec_pop'])

#drop unnecessary column
edu_data_per_student = drop_col(edu_data_per_student, ['Country_y'])

#rename column label
edu_data_per_student = rename_col(edu_data_per_student, {'Country_x': 'Country', 0: 'pre_primary_per_student',
                                                         1: 'primary_per_student', 2: 'lower_sec_per_student'})

#check number on Nan in edu_data_per_student
index1 = np.where(edu_data_per_student['pre_primary_per_student'].isnull())[0]
# pre_primary => CAN, GRC, MAC, OAVG, SGP, TUR
# primary =>  CAN, GRC, HKG, MAC, OAVG, SGP, TUR
# secondary => CAN, GRC, HKG, MAC, OAVG, SGP
# sum = [CAN, GRC, HKG, MAC, OAVG, SGP, TUR] [ISR, PER, SVN]
# moze ISR, PER ma inny system edukacji bo brakuje danych tylko dla secondary, moze ekstrapolacje dla SVN

#estimate total expenses per student in a given country
student_total_expenses = estimate_total_cost(edu_data_per_student)

#delete countries with zero data ([CAN, GRC, HKG, MAC, OAVG, SGP, TUR] [ISR, PER, SVN])
student_total_expenses = student_total_expenses[student_total_expenses.Total != 0]

#reset index
student_total_expenses.reset_index(drop=True, inplace=True)


### 7/ make OLS for total education cost and PISA results form 2015 (3 separate subjects)

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


### 7A/ OLS for total education cost and average PISA results from 2015

#merge with average PISA results
pisa_ave_expenses = merge_df(all_pisa_2015_ave, student_total_expenses)

#rename column label
rename_col(pisa_ave_expenses, {'Country_x': 'Country'})

#leave LUX out as an outlier
pisa_ave_expenses_lux = pisa_ave_expenses[pisa_ave_expenses['Code'] != 'LUX']

#perform OLS
model_ave_expenses = smf.ols(formula='ave_result ~ Total', data=pisa_ave_expenses).fit()
model_ave_expenses_lux = smf.ols(formula='ave_result ~ Total', data=pisa_ave_expenses_lux).fit()

#show summary
model_ave_expenses.summary()
model_ave_expenses_lux.summary()

#plot
plot_ave_expenses = show_scatterplot(pisa_ave_expenses, ['Total', 'ave_result'], 'm')

#plot with curve
lin_pisa_ave_expenses = fit_data_mat(pisa_ave_expenses['Total'], pisa_ave_expenses['ave_result'], 1,
                                 'expenses on education per student', 'average pisa result')

lin_pisa_ave_expenses_2 = fit_data_sea(pisa_ave_expenses['Total'], pisa_ave_expenses['ave_result'], 1,
                                      'expenses on education per student', 'average pisa result')

quad_pisa_ave_expenses = fit_data_mat(pisa_ave_expenses['Total'], pisa_ave_expenses['ave_result'], 2,
                                 'expenses on education per student', 'average pisa result')

quad_pisa_ave_expenses_2 = fit_data_sea(pisa_ave_expenses['Total'], pisa_ave_expenses['ave_result'], 2,
                                    'expenses on education per student', 'average pisa result')



### ### ### III ETAP ### ### ### - Check goodness of fit

### 8/ Try to fit data to polynomial and test which degree fits best

#fit GDP_per capita to PISA results data (2015) manually
lin_ave_gdp_ppp_log_pisa = fit_data_mat(pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'], 1,
                                    'gdp per capita', 'average pisa result')

lin_ave_gdp_ppp_log_pisa_2 = fit_data_sea(pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'], 1,
                                      'gdp per capita', 'average pisa result')


quad_ave_gdp_ppp_log_pisa = fit_data_mat(pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'], 2,
                                    'gdp per capita', 'average pisa result')

quad_ave_gdp_ppp_log_pisa_2 = fit_data_sea(pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'], 2,
                                      'gdp per capita', 'average pisa result')

#leave LUX out
lin_ave_gdp_ppp_log_pisa_lux = fit_data_mat(pisa_ave_gdp_ppp_log_lux['gdp_ppp_log'], pisa_ave_gdp_ppp_log_lux['ave_result'], 1,
                                    'gdp per capita', 'average pisa result')

lin_ave_gdp_ppp_log_pisa_2_lux = fit_data_sea(pisa_ave_gdp_ppp_log_lux['gdp_ppp_log'], pisa_ave_gdp_ppp_log_lux['ave_result'], 1,
                                      'gdp per capita', 'average pisa result')

quad_ave_gdp_ppp_log_pisa_lux = fit_data_mat(pisa_ave_gdp_ppp_log_lux['gdp_ppp_log'], pisa_ave_gdp_ppp_log_lux['ave_result'], 2,
                                    'gdp per capita', 'average pisa result')

quad_ave_gdp_ppp_log_pisa_2_lux = fit_data_sea(pisa_ave_gdp_ppp_log_lux['gdp_ppp_log'], pisa_ave_gdp_ppp_log_lux['ave_result'], 2,
                                      'gdp per capita', 'average pisa result')


#fit data using list of many degrees for GDP_per capita to PISA results data (2015)
degrees = (1, 2, 3, 4)
gdp_models = gen_fits(pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'], degrees)
gdp_model_review = test_fits(gdp_models, degrees, pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'])

#leave LUX out
gdp_models_lux = gen_fits(pisa_ave_gdp_ppp_log_lux['gdp_ppp_log'], pisa_ave_gdp_ppp_log_lux['ave_result'], degrees)
gdp_model_review_lux = test_fits(gdp_models_lux, degrees, pisa_ave_gdp_ppp_log_lux['gdp_ppp_log'], pisa_ave_gdp_ppp_log_lux['ave_result'])


#fit data using list of many degrees for total_student_expenses to 2015 PISA results data
expenses_models = gen_fits(pisa_ave_expenses['Total'], pisa_ave_expenses['ave_result'], degrees)
expenses_model_review = test_fits(expenses_models, degrees, pisa_ave_expenses['Total'], pisa_ave_expenses['ave_result'])

#leave LUX out
expenses_models_lux = gen_fits(pisa_ave_expenses_lux['Total'], pisa_ave_expenses_lux['ave_result'], degrees)
expenses_model_review_lux = test_fits(expenses_models_lux, degrees, pisa_ave_expenses_lux['Total'], pisa_ave_expenses_lux['ave_result'])


### 8A/ Try to fit data into log or exponential

# WYCHODZA DZIWNE WARTOSCI gdy probuje sprawdzic R squared
np.polyfit(np.log(x), y, 1)
np.polyfit(x, np.log(y), 1)
np.polyfit(x, np.log(y), 1, w=np.sqrt(y))

from scipy.optimize import curve_fit
scipy.optimize.curve_fit(lambda t,a,b: a+b*np.log(t),  x,  y)
scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y)
scipy.optimize.curve_fit(lambda t,a,b: a*np.exp(b*t),  x,  y,  p0=(4, 0.1))

def gen_fits(x, y, degrees):
    """Compute coefficients for given degree polynomial
    :param x: data frame
    :param y: data frame
    :param degrees: list
    :returns array list"""
    models = []
    for d in degrees:
        model = pylab.polyfit(x, y, d)
        models.append(model)
    extra_models = [np.polyfit(np.log(x), y, 1), np.polyfit(x, np.log(y), 1), np.polyfit(x, np.log(y), 1, w=np.sqrt(y))]
    for j in extra_models:
        models.append(j)
    return models


### 9/ Repeated Random Sampling cross_validation of models - check predictive power

#GDP_per capita to PISA results data (2015)
compare_predictive_power(20, [1, 2, 3, 4], pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'])

#leave LUX out
compare_predictive_power(20, [1, 2, 3, 4], pisa_ave_gdp_ppp_log_lux['gdp_ppp_log'], pisa_ave_gdp_ppp_log_lux['ave_result'])

#total_student_expenses to 2015 PISA results data
compare_predictive_power(20, [1, 2, 3, 4], pisa_ave_expenses['Total'], pisa_ave_expenses['ave_result'])

#leave LUX out
compare_predictive_power(20, [1, 2, 3, 4], pisa_ave_expenses_lux['Total'], pisa_ave_expenses_lux['ave_result'])

# sa minusy w porownaniu dla expense, a przy analizie dopasowania nie bylo ich wcale
# nie dziala analiza dla danych bez luksemburga






### 10/ Clustering with k nearest neighbours





# test expenses per student for being exponentially distributed
# w krajach biednych matematyka jest ponizej innych wynikow,
    # a im bogatsi albo wiecej wydaja to bardziej sie to przeklada na wyniki z matematyki w stosunku do innych wynikow

### ### ### III ETAP ### ### ###
# moze LUX, FIN, POL, BRa bo ma najnizsze wyniki z maty
# wybrac kilka najciekawszych krajow: charakterystycznych albo odstajacych i przeanalizowac zmiennosc
    # relacji miedzy wydatkami na edukacje a wynikami pisa w czasie (albo gdp PPP a wynikami pisa)
    # sprobowac uwzgledni opoxnienie efektu wydatkow na edkacje na wyniki egzaminu - moze 9 lat ?

# moze jakis heat map gdzie najwyzsze wyniki albo gdzie najwyzsze korelacje

