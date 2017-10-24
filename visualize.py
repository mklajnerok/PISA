import matplotlib.pyplot as plt


# try to create a function to perform ols automatically

def show_scatterplot(df_data, variables, color):
    """takes df_data and plots for chosen columns in variables list
    :type df_data: data frame
    :type variables: list of str
    :type color: str"""
    plt.scatter(x=df_data[variables[0]],y=df_data[variables[1]],color=color)
    plt.title(variables[0] + ' vs. PISA ' + variables[1] + ' results for 2015')
    plt.xlabel(variables[0])
    plt.ylabel(variables[1])
    plt.show()

# try to create a function to plot may things at once