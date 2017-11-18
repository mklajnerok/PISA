import random

def rSquared(observed, predicted):
    """Calculate r-squared value for given data"""
    error = ((predicted-observed)**2).sum()
    mean_error = error/len(observed)
    return 1 - (mean_error/np.var(observed))

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
    return models

def test_fits(models, degrees, x, y):
    """Compute R-squared value for given models with various polynomial degrees
    :param models: array list
    :param degrees: list
    :param x: data frame
    :param y: data frame
    :returns fit_goodness: dict"""
    fit_goodness = {}
    for i in range(len(models)):
        est_y = pylab.polyval(models[i], x)
        error = rSquared(y, est_y)
        fit_goodness[i] = error
    return fit_goodness

def test_fits_plot(models, degrees, x, y, title):
    """Compute R-squared value for given models with various polynomial degrees
    and display results within a plot
    :param models: array list
    :param degrees: list
    :param x: data frame
    :param y: data frame
    :param title: str"""
    pylab.plot(x, y, 'o', label='Data')
    for i in range(len(models)):
        est_y = pylab.polyval(models[i], x)
        error = rSquared(y, est_y)
        pylab.plot(x, est_y, label='Fit of degree' + str(degrees[i])+', R2 = ' + str(round(error, 5)))
    pylab.legend(loc='best')
    pylab.title(title)

def test_fits_sea(models, degrees, x, y, title,):
    sns.regplot(x, y, fit_reg=False, label='Data')
    for i in range(len(models)):
        sns.regplot(x, y, order=i, line_kws={"color": "r", "alpha": 0.4, "lw": 5},
                    label='Fit of degree' + str(degrees[i]))
    plt.title(title)

degrees = (1, 2, 3, 4)
models = gen_fits(pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'], degrees)
test_fits_sea(models, degrees, pisa_ave_gdp_ppp_log['gdp_ppp_log'], pisa_ave_gdp_ppp_log['ave_result'], 'Goodness of fit')




def split_data(x, y):
    """Split data set into subsets for training and testing
    :param x: data frame column
    :param y: data frame column
    :returns 4 lists"""
    to_train = random.sample(range(len(x)), len(x) // 2)
    train_x, train_y, test_x, test_y = [], [], [], []
    for i in range(len(x)):
        if i in to_train:
            train_x.append(x[i])
            train_y.append(y[i])
        else:
            test_x.append(x[i])
            test_y.append(y[i])
    return train_x, train_y, test_x, test_y

def compare_predictive_power(num_subsets, degrees, x, y):
    """Take data set and perform cross-validation using repeated random sampling,
    print R-squared and standard deviation for each dimension model
    :param num_subsets: int
    :param degrees: list
    :param x: data frame
    :param y: data frame"""
    r_squares = {}
    results = pd.DataFrame(index=range(1, 1+len(degrees)), columns=['mean', 'sd'])
    for d in degrees:
        r_squares[d] = []
    for f in range(num_subsets):
        train_x, train_y, test_x, test_y = split_data(x, y)
        for d in degrees:
            model = pylab.polyfit(train_x, train_y, d)
            est_y_vals = pylab.polyval(model, test_x)
            r_squares[d].append(rSquared(test_y, est_y_vals))
    print('Mean R-squares for test data')
    for d in degrees:
        mean = round(sum(r_squares[d]) / len(r_squares[d]), 4)
        sd = round(np.std(r_squares[d]), 4)
        results.loc[d] = [mean, sd]
    return results
