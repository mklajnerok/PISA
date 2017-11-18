import matplotlib.pyplot as plt
import seaborn as sns

# try to create a function to perform ols automatically
# try to create a function to plot many things at once

def fit_data_mat(df_data1, df_data2, degree, x_label, y_label):
    """Plot data from df_data1 and df_data2 and try to fit a curve with a given degree using matplotlib.pyplot
    :param df_data1: data frame
    :param df_data2: data frame
    :param degree: int
    :param x_label: str
    :param y_label: str"""
    pylab.plot(df_data1, df_data2, 'bo', label='Data')
    pylab.title('Measured displacement')
    pylab.xlabel(x_label)
    pylab.ylabel(y_label)
    model = pylab.polyfit(df_data1, df_data2, degree)
    est_y_vals = pylab.polyval(model, df_data1)
    pylab.plot(df_data1, est_y_vals, 'r', label='Curve fit')
    pylab.legend(loc='best')

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

def show_scatterplot(df_data, variables, color):
    """Take df_data and plot for chosen columns in variables list
    :param df_data: data frame
    :param variables: list of str
    :param color: str"""
    plt.scatter(x=df_data[variables[0]],y=df_data[variables[1]], color=color)
    plt.title(variables[0] + ' vs. PISA ' + variables[1] + ' results for 2015')
    plt.xlabel(variables[0])
    plt.ylabel(variables[1])
    plt.show()

def show_bubbles(df_data, variables, color):
    plt.scatter(x=df_data[variables[0]],y=df_data[variables[1]], s=z*1000, alpha=0.5)
    plt.show()

