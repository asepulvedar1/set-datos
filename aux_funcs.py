#ancilliary_funcs.py
#%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, r2_score
import re
import matplotlib.pyplot as plt
from matplotlib import gridspec
import xgboost
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, accuracy_score, auc, mean_squared_error, r2_score, confusion_matrix

def rep_ed(obj, dna = True):
    '''
    Definición: La función rep_ed(), devuelve un reporte de estadísticas descriptivas (usando el método 'describe()', para una serie que contenga variables contínuas. Si la serie contiene una variable discreta, entonces devuelve la frecuencia usando 'value_counts()'
    Parámetros: 
    1. "obj", corresponde a un DataFrame de pandas. Por defecto contiene el dataframe subset 'df_ss'.
    2. "dna", booleano que indica si para el método value_counts() se desea dropear nulos.
    '''
    for i in obj.columns:
        if obj[i].dtype == 'object':
            print('variable: ', i)
            print(obj[i].value_counts(dropna = dna))
        else:
            print('variable: ', i)
            print(obj[i].describe())
        print('________________________________________________________________\n')
            

def lista_perdidas(df, var, print_list = False):
    '''
    Definición: La función lista_perdidas, retorna:
    1. Si el parámetro 'print_list' tiene el valor booleano 'Falso': retorna una tupla con la cantidad y porcentaje de casos perdidos, para la variable definida en el parámetro 'var', del dataframe definido en el parámetro 'df'.
    2. Si el parámetro 'print_list' tiene el valor booleano 'Verdadero': retorna una tripleta con la cantidad, porcentaje y un dataframe resultante con el subset de casos perdidos donde la variable definida en el parámetro 'var', contiene valores nulos.
    Parámetros:
    i- df: es el dataframe respecto del cual se quiere analizar las variables con valores perdidos.
    ii- var: es la variable que se quiere analizar respecto de valores faltantes.
    iii- print_list, por defecto toma el valor booleano Falso. Indica si se quiere retornar un subconjunto del dataframe original, donde la variable analizada 'var' contiene sólo valores perdidos.
    '''
    if len(df[var].isnull().value_counts().index) < 2:
        if True in df[var].isnull().value_counts():
            cant_cp = df[var].isnull().value_counts().values[0]
            porc_cp = df[var].isnull().value_counts('%').values[0].round(4)
    else:
        cant_cp = df[var].isnull().value_counts()[1]
        porc_cp = df[var].isnull().value_counts('%')[1].round(4)
        if print_list is True:
            df_ss_cp = df[df[var].isnull()]
            resultado = cant_cp, porc_cp, df_ss_cp
        else:
            resultado = cant_cp, porc_cp
        return resultado
    
def graf_hist(dataframe, var, sample_mean = False, true_mean = True):
    """
    La función graf_hist, recibe como parámetros:
    1. dataframe: Un DataFrame de pandas,
    2. var: variable a graficar.
    3. sample_mean: valor booleano (por defecto 'False'), que indica si se debe graficar la media de la variable analizada "var", definida, en la selección muestral de del dataframe subconjunto seleccionado. Si es 'True', graficará mediante una línea vertical '-.' roja, el valor de la media muestral.
    4. true_mean: valor booleano, (por defecto 'True'), que indica si se debe graficar una recta vertical que representa la media de la variable analizada "var", en la base de datos original.
    Devuelve: Un Histograma de frecuencias, graficando los valores que toma la variable analizada "var", y una representación de las medias muestrales y global de la variable analizada, mediante líneas rectas. Si es 'True', graficará mediante una línea vertical '--' color 'indigo', el valor de la media global.
    """
    plt.hist(dataframe[var].dropna(), color='royalblue', alpha=0.7);
    if sample_mean is True:
        plt.axvline(df_ss[var].dropna().mean(), color = 'red', lw=2, ls='-.')
    if true_mean is True:
        plt.axvline(df[var].dropna().mean(), color = 'indigo', lw=2, ls='--')
    plt.title('Distribución empírica de la variable {}'.format(var))
    plt.xlabel(var)
    plt.ylabel('Frecuencia')

    
def dot_plot(dataframe, plot_var, plot_by='ht_region', global_stat = False, statistic = 'mean'):
    """
    Definición: Devuelve un gráfico dotplot en que por cada valor de la variable agrupadora definida en el parámetro 'plot_by', muestra el valor agrupado de la variable definida en el parámetro 'plot_var', (según el criterio definido en la variable 'statistic').
    Parámetros:
    1. 'dataframe': Es el dataframe definido para la submuestra a analizar.
    2. 'plot_var': Es la variable numérica cuya agrupación se desea analizar.
    3. 'plot_by': Es la variable agrupadora (variable del tipo nominal), respecto de la cual se quiere visualizar el valor de la variabla 'plot_var', agrupada mediante un criterio definido en el parámetro 'statistic'..
    4. global state: Valor booleano, que indica si se desea graficar el valor de la media global de la variable analizada 'plot_var', en el DataFrame ingresado como parámetro.
    5. statistic: Indica el criterio de agrupación para la variable 'plot_var'. Se tiene dos opciones: 'mean' o 'median'. Por defecto, toma el valor 'mean'.
    Retorna: Un gráfico del tipo 'dot_plot'.
    """
    #plt.figure(figsize=(12,6))
    df_group_by = eval('dataframe.groupby(plot_by)[plot_var].' + statistic + '()')
    plt.plot(df_group_by.values, df_group_by.index, 'ro', lw=2);
    if global_stat is True:
        plt.axvline(dataframe[plot_var].mean(), color='yellowgreen', lw=2, ls='-.');
    plt.title('Posición de las distintas zonas geográficas en cuanto a variable {}'.format(plot_var))
    plt.xlabel(plot_var)
    plt.ylabel(plot_by)
    
    
def cdp(df, variable):
    """
    Definición: La función 'cdp()' retorma un gráfico de la función (curva) de densidad de probabilidad y el histograma con la distribución de la variable, definida en el parámetro 'variable'.
    Parámetros:
    1. 'df': Es el objeto dataframe de pandas, que contiene la variable a analizar.
    2. 'variable': Es la variable a analizar.
    """
    var_dropna = df[variable].dropna()
    log_var = np.log(var_dropna+1)
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.hist(var_dropna, color='grey', alpha=0.5, density=True);
    x_min, x_max = plt.xlim()
    x_axis = np.linspace(x_min, x_max,100)
    #gauss_kde = stats.gaussian_kde(df[variable].dropna())
    #Z = np.reshape(gauss_kde(x_axis).T, x_axis.shape)
    #plt.plot(x_axis, Z, color='deepskyblue', lw=3)
    plt.plot(x_axis, stats.norm.pdf(x_axis, var_dropna.mean(), var_dropna.std()), color='steelblue', lw=3)
    plt.axvline(var_dropna.mean(), color='orangered', ls='--', lw=3)
    plt.title("Histograma variable {}".format(variable))
    plt.subplot(1,2,2)
    plt.hist(log_var, color='grey', alpha=.5, density=True)
    x_min, x_max = plt.xlim()
    x_axis = np.linspace(x_min, x_max,100)
    plt.plot(x_axis, stats.norm.pdf(x_axis, log_var.mean(), log_var.std()), color='steelblue', lw=3)
    plt.axvline(log_var.mean(), color='orangered', ls='--', lw=3)
    plt.title("Histograma variable {} \n transformada mediante logaritmo".format(variable))
    

def funcion_plot(x, color):
    '''
    Definición: La función 'funcion_plot', recibe una variable llamada 'x', y genera un gráfico de distribución de la variabla, usando seaborn.
    Parámetro: "x", es la variable a graficar.
    Retorna: Un gráfico de distribución de la variable, junto con una línea vertical que representa la media de la variable.
    '''
    sns.set(font_scale=0.8)
    sns.distplot(x, kde=False)
    plt.axvline(x.mean(), color=color)

#funcion_plot(df['adfert'], color='tomato');

def sns_binarize_hist(dataframe, variable):
    '''
    Definición: La función 'sns_binarize_hist', recibe una dataframe de pandas, y una variable presente en éste, 
    filtrando los valores faltantes para esa variable, y genera una nueva variable binaria, denominada 'binarize' para esa 
    variable, en base a si los valores son menores a la media: 'binarize' = 1, y en caso contrario, 'binarize' = 0.
    Luego, usando FacetGrid, y la función 'funcion_plot' definida, grafica un histograma de la variable, en función a 'binarize'.
    Parámetros:
    1. 'dataframe': es el dataframe de pandas que contiene la información y variables en estudio.
    2. 'variable': es la variable contenida en el dataframe, que se desea analizar.
    Retorno: Un gráfico con la distribución de la variable.
    '''
    tmp = dataframe.copy()
    tmp[variable] = tmp[variable].dropna()
    tmp['binarize'] = np.where(tmp[variable] < np.mean(tmp[variable]), 1, 0)
    grid = sns.FacetGrid(tmp, col = 'binarize', col_wrap=2, size=5)
    grid = grid.map(funcion_plot, variable)
   
    
    
    
    
def fun_hist(df, variable, binarize):
    '''
    Definición: La función 'funcion_histograma', grafica histogramas para dos grupos de la 'variable' de interés, definidos en base al parámetro 'binarize'
    Parámetros:
    1. 'df': representa el dataframe de pandas que se está analizando.
    2. 'variable': Es la variable contenida en el dataframe, cuya distribución se desea graficar.
    3. 'binarize': Es una variable binaria, contenida en el dataframe, en base a la cual se separa la variable de interés en dos grupos, para caracterizar la distribución de la variable de interés.
    Retorna: Un histograma con la distribución de la variable de interés en base a los dos grupos en que se ha dividido.
    El gráfico se retorna con las distribuciones superpuestas, dos líneas que representan el promedio de la variable en ambos grupos, y una leyenda con el promedio.
    '''
    dat = df
    samp_g1 = dat[dat[binarize]==1][variable].dropna()
    samp_g2 = dat[dat[binarize]==0][variable].dropna()
    plt.hist(samp_g1, color='orangered', alpha = 0.6, label='Grupo 1: '+ variable + ' donde ' +  binarize + '= 1: tiene acceso a credito', bins=10)
    plt.hist(samp_g2, color='royalblue', alpha = 0.5, label='Grupo 2: '+ variable + ' donde ' + binarize + '= 0: no tiene acceso a credito', bins=10)
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.axvline(np.mean(samp_g1), lw=0.5, color='tomato', label='Promedio '+ variable + ' donde ' +  binarize + '= 1: tiene acceso a credito')
    plt.axvline(np.mean(samp_g2), lw=0.5, color='dodgerblue', label='Promedio '+ variable+ ' donde ' + binarize + '= 0: no tiene acceso a credito')
    plt.legend();
    #plt.show()
    
    
def sns_grouped_scatterplot(dataframe, x, y, gb):
    '''
    Definición: La función 'sns_grouped_scatterplot', genera gráficos de dispersión (scatterplot) entre las variables 'x' 
    e 'y' para cada valor de la variable definida en el parámetro gb.
    Parámetros:
    1. 'dataframe': es el dataframe de pandas que contiene la información y variables en estudio.
    2. 'x': variable contínua del dataframe para graficar en el eje de las abscisas.
    3. 'y': variable contínua del dataframe para graficar en el eje de las ordenadas.
    Retorno: La función devuelve un gráfico de dispersión de las variables x e y para cada valor de la variable definida en
    el parámetro gb.
    '''
    grid = sns.FacetGrid(dataframe, col = gb, col_wrap=2)
    grid = grid.map(sns.scatterplot, x, y, marker='o', s=50, color = 'darkblue')
    
    

def sns_grouped_box_plot(dataframe, variable, group_by):
    '''
    Definición: La función 'sns_grouped_box_plot', genera un gráfico boxplot (gráfico de cajas y bigotes), para la variable seleccionada en el parámetro 'variable', generándose boxplots comparativos de la variable para cada valor que tome la variable declarada en el parámetro 'group_by'.
    Parámetros:
    1. 'dataframe': es el dataframe de pandas que contiene la información y variables en estudio.
    2. 'variable': es la variable de interés, contenida en el dataframe, respecto a la cual se desea obtener el boxplot'.
    3. 'group_by': es la variable agrupadora, en función de la cual se graficarán los boxplot de la variable.
    '''
    font = {'family': 'arial',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
    plt.rcParams['figure.figsize']=(15,15)
    sns.boxplot(x = group_by, y = variable, data = dataframe);
    plt.xticks(fontsize = 15);
    plt.yticks(fontsize = 10);
    plt.xlabel(group_by.capitalize(), fontdict = font);
    plt.ylabel(variable.capitalize(), fontdict = font);


def funcion_histograma(df, variable, binarize):
    '''
    Definición: La función 'funcion_histograma', grafica histogramas para dos grupos de la 'variable' de interés, definidos 
    en base al parámetro 'binarize'
    Parámetros:
    1. 'df': representa el dataframe de pandas que se está analizando.
    2. 'variable': Es la variable contenida en el dataframe, cuya distribución se desea graficar.
    3. 'binarize': Es una variable binaria, contenida en el dataframe, en base a la cual se separa la variable de interés
    en dos grupos, para caracterizar la distribución de la variable de interés.
    Retorna: Un histograma con la distribución de la variable de interés en base a los dos grupos en que se ha dividido.
    El gráfico se retorna con las distribuciones superpuestas, dos líneas que representan el promedio de la variable en ambos
    grupos, y una leyenda con el promedio.
    '''
    dat = df
    samp_g1 = dat[dat[binarize]==1][variable].dropna()
    samp_g2 = dat[dat[binarize]==0][variable].dropna()
    plt.hist(samp_g1, color='orangered', alpha = 0.6, label='Grupo 1: '+ variable + ' ingresos > ' + binarize[-4:], bins=10)
    plt.hist(samp_g2, color='royalblue', alpha = 0.5, label='Grupo 2: '+ variable + ' ingresos <=' + binarize[-3:] , bins=10)
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.axvline(np.mean(samp_g1), lw=1, color='tomato', label='Promedio '+ variable + ' ingresos ' + binarize[-4:])
    plt.axvline(np.mean(samp_g2), lw=1, color='dodgerblue', label='Promedio '+ variable + ' ingresos <=' + binarize[-3:])
    plt.legend();
    #plt.show()
    
    
def f_test_hipotesis(df,variable,binarize):
    ''' Docstring
    Definición: Esta función realiza un test de hipótesis de diferencia de medias independientes por cada variable según 'binarize'.
    Por defecto, se asume igualdad de varianza.
    Parámetros:
    1. df_: dataframe con los datos.
    2. variable: variable a analizar
    3. binarize: variable binaroa para distinción de grupo por variable
    
    Retorna: Las medias de cada grupo, la diferencia entre las medias, el valor del estadístico de prueba y el p_value del test.
    '''
    dat = df
    
    #dat.query('binarize == 1')['variable'].dropna()
    samp_g1 = dat[dat[binarize]==1][variable].dropna()
    samp_g2 = dat[dat[binarize]==0][variable].dropna()
    t, p_val = stats.ttest_ind(samp_g1, samp_g2)
    media_sg1 = np.mean(samp_g1)
    media_sg2 = np.mean(samp_g2)
    print(f'La media del la variable {variable} para el grupo de ingresos {binarize[-4:]} es de {round(media_sg1,4)}')
    print(f'La media del la variable {variable} para el grupo de ingresos <= {binarize[-3:]} es de {round(media_sg2,4)}')
    print(f'La diferencia entre medias es de: {round(abs(media_sg1 - media_sg2),4)}')
    print(f'El estadístico t = {round(t,4)}, el p_value asociado = {round(p_val,5)}')
    if abs(t) > 1.96 or p_val <= 0.05:
        print(f'Considerando un nivel de significancia del 5%, existe evidencia estadística suficiente como para rechazar la hipótesis nula de que para la variable {variable} los promedios entre el grupo de ingresos {binarize[-4:]} y el grupo de ingresos {binarize[-3:]} son iguales.')
        print('\n')
        print(f'Por lo tanto, se concluye, con un nivel de confianza del 95% que para la variable {variable} los promedios entre el grupo de ingresos {binarize[-4:]} y el grupo de ingresos {binarize[-3:]} son distintos.')
    else:
        print(f'Considerando un nivel de significancia del 5%, no existe evidencia estadística suficiente como para rechazar la hipótesis nula de que para la variable {variable} los promedios entre el grupo de ingresos {binarize[-4:]} y el grupo de ingresos {binarize[-3:]} son iguales.')
        print('\n')
        print(f'Por lo tanto, se concluye, que con un nivel de confianza del 95% no es posible rechazar que para la variable {variable} los promedios entre el grupo de ingresos {binarize[-4:]} y el grupo de ingresos {binarize[-3:]} sean distintos.')
        
def inverse_logit(x):
    '''
    Definición: Calcula la inversa del logit de x.
    Parámetro: 'x': variable numérica.
    Retorno: devuelve la inversa del logit de x : 1/(1+e^(-x)).
    '''
    return 1/(1+np.exp(-x))
        
        
def fetch_features(dataframe, ov):
    '''
    Definición: La función retorna una listado con las correlaciones entre cada atributo del dataframe, y el vector objetivo.
    Parámetros:
    1. 'dataframe' es la base de datos original.
    2. 'ov', es el vector objetivo.
    Retorna: Un listado con las correlaciones entre cada atributo y el vector objetivo, así como también el nombre de cada atributo.
    '''
    
    lista = round(abs(dataframe.corr()[ov]),3).sort_values(ascending=False)
    ff = lista.drop(ov)
    return ff

def id_missing(df):
    '''
    Descripción: Identifica valores perdidos en el DataFrame
    Parámetros:
    1. 'df': DataFrame en análisis.
    Retorna: Imprime en pantalla la columna del dataframe, y el porcentaje de nulos que registra.
    '''
    for colname, serie in df.iteritems():
        if len(serie.isna().value_counts()) == 1 and serie.isna().value_counts().index == False:
            pass
        else:
            print(colname, serie.isna().value_counts('%')[1].round(4))
            
def plot_hist(dataframe, variable):
    '''
    Definición: La función recibe un dataframe y una variable de específica de éste, y grafica un histograma de la variable, entregando además los valores de la media y la mediana, junto con graficar líneas representando a ambos estadísticos en el gráfico.
    Parámetros:
    1. 'dataframe': Corresponde al conjunto de datos almacenados en el DataFrame de pandas.
    2. 'variable': Representa la variable cuyo histograma se desea graficar.
    Retorno: Grafica un histograma de la variable, con dos líneas verticales representando la media y la mediana, junto con una leyenda donde se entrega el valor de ambos estadísticos.
    '''
    plt.hist(dataframe[variable].dropna(), color = 'dodgerblue', alpha=0.7)
    plt.axvline(dataframe[variable].dropna().mean(), color = 'red', label = 'media='+ str(round(dataframe[variable].dropna().mean(),2)))
    plt.axvline(dataframe[variable].dropna().median(), color = 'forestgreen', label = 'mediana='+str(round(dataframe[variable].dropna().median(),2)))
    plt.legend()
    plt.title('Histograma de la variable '+ variable)
    plt.show()
    
def report_scores(pred, val):
    '''
    Definición: Imprime las métricas del Error Cuadrático Medio y R^2, en base a los vectores de datos predichos 'pred' y vector de datos a validar 'val'
    Parámetros:
    1. 'pred': vector de datos predichos.
    2. 'val': vector de datos por validar.
    Retorno: Imprime en pantalla los valores del Error Cuadrático Medio y R^2.
    '''
    print(f'Error cuadrático medio {round(mean_squared_error(val, pred),2)}')
    print(f'R^2 {round(r2_score(val, pred),2)}')
    
    
def plot_importance(fit_model, feat_names):
    """TODO: Docstring for plot_importance.

    :fit_model: TODO
    :feat_names: TODO
    :returns: TODO

    """
    tmp_importance = fit_model.feature_importances_
    sort_importances = np.argsort(tmp_importance)[::-1]
    names = [feat_names[i] for i in sort_importances]
    modelname = str(fit_model).split('(')[0]
    plt.title('Feature importance '+modelname, fontsize = 30)
    plt.barh(range(len(feat_names)), tmp_importance[sort_importances])
    plt.yticks(range(len(feat_names)), names, rotation=0, fontsize = 30)
   
    
    
    
def plot_classification_report(y_true, y_hat, dummy_class=False):
    """
    plot_classification_report: Genera una visualización de los puntajes reportados con la función `sklearn.metrics.classification_report`.

    Parámetros de ingreso:
        - y_true: Un vector objetivo de validación.
        - y_hat: Un vector objetivo estimado en función a la matriz de atributos de validación y un modelo entrenado.

    Retorno:
        - Un gráfico generado con matplotlib.pyplot

    """
    # process string and store in a list
    report = classification_report(y_true, y_hat).split()
    # keep values
    report = [i for i in report if i not in ['precision', 'recall', 'f1-score', 'support', 'avg', 'accuracy', 'macro', 'weighted']]
    # transfer to a DataFrame
    report = pd.DataFrame(np.array(report).reshape(len(report) // 5, 5))
    # asign columns labels
    report.columns = ['idx', 'prec', 'rec', 'f1', 'n']
    # preserve class labels
    class_labels = report.iloc[:np.unique(y_true).shape[0]].pop('idx').apply(float).apply(int)#.apply(int)
    # separate values
    class_report = report.iloc[:np.unique(y_true).shape[0], 1:4]
    # convert from str to float
    class_report = class_report.applymap(float)
    # convert to float average report
    average_report = report.iloc[-1, 1: 4].apply(float)

    colors = ['dodgerblue', 'tomato', 'purple', 'orange']

    for i in class_labels:
        plt.plot(class_report['prec'][i], [1], marker='X', color=colors[i])
        plt.plot(class_report['rec'][i], [2], marker='X', color=colors[i])
        plt.plot(class_report['f1'][i], [3], marker='X',color=colors[i], label=f'Class: {i}')

    plt.scatter(average_report, [1, 2, 3], marker='o', color='forestgreen', label='Avg', lw=5)
    plt.yticks([1.0, 2.0, 3.0], ['Precision', 'Recall', 'f1-Score'])
    
    if dummy_class is True:
        plt.axvline(.5, label = '.5 Boundary', linestyle='--')
        
def binarize_df(df):
    '''
    Objetivo: Binariza el conjunto de variables categóricas de un dataset
    Parámetros:
    df: corresponde al dataframe de pandas, cuyas variables categóricas se requiere binarizar
    '''
    # Se crean dos listas para contener los nombres de las columnas categóricas y numéricas
    lista_cat = []
    lista_num = []
    for i in df.columns:
        if df[i].dtype == 'O':
            lista_cat.append(i)
        else:
            lista_num.append(i)
    #Si la variable es categórica, se procede a binarizar, dicha variable usando pd.get_dummies()
    for i in lista_cat:
        dum = pd.get_dummies(df[i], prefix=i, drop_first = True)
        df = pd.concat([df, dum], axis=1)
    # Se dropean las variables originales
    for i in lista_cat:
        df = df.drop(columns=i)
    #Se retorna el dataframe binarizado.
    return df

def chi_sq(df, var1, ov):
    """
    Objetivo función chi_sq(): Permite correr el test chi-cuadrado para evaluar si dos variables categóricas (o dicotómicas) están relacionadas.
    Se ejecuta el test chi_sq para contrastar los test de hipótesis nula de independencia de variables y la alternativa de dependencia entre variables.
    Parámetros:
    1. 'df': corresponde al nombre de un dataframe de pandas definido.
    2. 'var1': variable dicotómica o categórica que se requiere contrastar con una segunda variable dicotómica.
    3. 'ov': corresponde al vector objetivo dicotómico respecto de cual se desea probar independencia con la variable dicotómica 'var1'.
    Retorma: 
    1. Imprime en pantalla el test realizado, incluyendo el estadísitco calculado, el p_value, los grados de libertad y las frecuencias esperadas.
    2. Retorna el nombre de la variable 'var1', en caso de que se detecte dependencia con 'ov'.
    """
    print('#'*125)
    print('\n')
    print(f'Test chi_cuadrado, variable {var1} v/s {ov}')
    cross_table = pd.crosstab(df[var1], df[ov], margins=True)
    print('chisq = %6.4f\n p-value = %6.4f\n dof = %i\n expected_freq = %s' %stats.chi2_contingency(cross_table))
    if stats.chi2_contingency(cross_table)[1] > 0.05:
        print(f'El valor de chi-cuadrado es {stats.chi2_contingency(cross_table)[0]}, el valor p es {stats.chi2_contingency(cross_table)[1]} > 0.05, lo que indica que no hay evidencia estadísitica para rechazar el que las variables {var1} y {ov} son independientes, por lo que se concluye que {var1} no varía en los distintos niveles de {ov}')
    else:
        print(f'El valor de chi-cuadrado es {stats.chi2_contingency(cross_table)[0]}, el valor p es {stats.chi2_contingency(cross_table)[1]} < 0.05, lo que indica que hay evidencia estadísitica para rechazar el que las variables {var1} y {ov} son independientes, por lo tanto se evidencia dependencia entre estas, por lo que se concluye que {var1} varía en los distintos niveles de {ov}')
        return var1
    
def IQR(dist):
    """
    devuelve el rango intercuantil para un arreglo o serie, definido en el parámetro 'dist'
    """
    return np.percentile(dist, 75) - np.percentile(dist, 25)


def plot_conf_matrix(ytest, yhat, df, col='VO', tipo_mc = 'p'):
    """
    Docstring para plot_conf_matrix()
    Retorna: Para un set de datos en un dataframe, devuelve un gráfico de mapa de calor, mostrando una matriz de confusión, con la cantidad de casos (o porcentaje de casos) por clase, vara valores reales y predichos del Vector Objetivo, definido en el parámetro 'col'
    Parámetros:
    1. 'ytest': Corresponde al vector de datos de validación.
    2. 'yhat': Corresponde al vector de datos predichos
    3. df: Objeto dataframe de pandas donde se encuentran los datos.
    4. col: Corresponde a la columna del vector objetivo en el dataframe
    5. tipo_mc: Es un parámetro que debe ingresar el usuario, indicando con valor 'p' si quiere un reporte de porcentaje de casos
    
    tipo_mc: Corresponde al tipo de matriz a generar: Seleccionar 'p' para porcentual, o 'n' para valores numéricos.
    """
    plt.rcParams['figure.figsize'] = (5, 5)
    cnf = confusion_matrix(ytest, yhat)
    cnf_p = confusion_matrix(ytest, yhat)/len(ytest)
    # Se guarda las etiquetas de las clases en un objeto:
    target_label = list(sorted(df[col].unique()))
    # Implementamos un mapa de calor definiendo las clases
    if tipo_mc == 'p':
        sns.heatmap(cnf_p, xticklabels=target_label, yticklabels=target_label,
                    # generamos las anotaciones en términos porcentuales
                    annot=True, fmt=".1%",
                    # evitamos la barra y cambiamos el colormap
                    cbar=False, cmap='Blues');
        plt.title('Matriz de confusión, porcentajes de casos', fontsize = 20)
        plt.xlabel('Valor Predicho', fontsize = 20)
        plt.ylabel('Valor Real ' + col, fontsize = 20)
    else:
        sns.heatmap(cnf, xticklabels=target_label, yticklabels=target_label,
                    # generamos las anotaciones en términos porcentuales
                    annot=True, fmt='d',
                    # evitamos la barra y cambiamos el colormap
                    cbar=False, cmap='Blues');
        plt.title('Matriz de confusión, cantidad de casos', fontsize = 20)
        plt.xlabel('Valor Predicho', fontsize = 20)
        plt.ylabel('Valor Real ' + col, fontsize = 20);
        

def plot_roc(modelo, X_test, y_test, model_name = 'Modelo Logit Depurado'):
    """
    Docstring plot_roc():
    Uso: Permite graficar la curva ROC, para un modelo guardado con el nombre definido en el parámetro 'modelo'
    Parámetros:
    1. 'modelo': es el nombre con que se guardó el modelo (no es texto)
    2. 'X_test': Nombre con el que se guardó la matriz de atributos del conjunto de validación
    3. 'model_name': Texto con el cual describe a su modelo.
    4. 'y_test': Nombre con el que se guardó el vector objetivo o variable dependiente en el set de validación.
    """
    plt.rcParams['figure.figsize']= (10,10)
    # re-estimamos los valores predichos de nuestro modelo para obtener la probabilidad entre 0 y 1.
    yhat_pr = modelo.predict_proba(X_test)[:, 1] #escojo la probabilidad de que sea 1.
    # generamos los objetos de roc_cruve
    false_positive, true_positive, threshold = roc_curve(y_test, yhat_pr)
    # Plot ROC curve
    f, ax = plt.subplots(1)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
    plt.title('Curva ROC' + model_name, fontsize = 16)
    plt.plot(false_positive, true_positive, lw=1)
    plt.plot([0, 1], ls="--", lw=1, color = 'tomato') # curva base
    plt.plot([0, 0], [1, 0] , c='limegreen', lw=3)
    plt.plot([1, 1], c='limegreen', lw=3)
    plt.ylabel('Verdaderos Positivos', fontsize = 15)
    plt.xlabel('Falsos Positivos', fontsize = 15);
    
def prep_df_validacion(df, var_dic_eliminar, lista_cat = ['Tamano', 'CIIU_final'], tipo_prep = 'df_completo'):
    """
    Docstring función prep_df_validacion()
    
    Utilidad: Permite preparar el dataframe de validación, para equipararlo al dataset de entrenamiento, para poder correr los modelos entrenados.
    Parámetros:
    1. 'df': Es un objeto pandas dataframe, que contiene la data del set de validación que se desea preprocesar.
    2. 'var_dic_eliminar': Es una lista de variables dicotómicas que se eliminarán, de acuerdo a lo arrojado por el análisis chi-cuadrado en el set de entrenamiento.
    3. 'lista_cat': Es una lista de variables categóricas, para recodificar como variables dummy. Por defecto son: ['Tamano', 'CIIU_final'].
    4. 'tipo_prep': Permite indicar si lo que se quiere obtener es el dataframe completo, previo a eliminación y transformación de columnas (opción 'df_completo'), o bien el dataset para validar modelos (opción 'df_validacion')
    Retorna: Un dataframe de pandas, con la data preprocesada, que se puede emplear para validar los modelos entrenados.
    """
    
    # Se obtienen las columnas del df de validación
    ele4_col = df.columns
    # Se realiza un remplazo de los nombres de las columnas, que vienen con sufijo identificatorio de la versión de la encuesta
    #, para equipararlas a las columnas del df de entrenamiento.
    ele4_col_mod = list(map(lambda x: x.replace('_ELE4',''), ele4_col))
    df_ele4 = df.set_axis(ele4_col_mod, axis=1)
    # Se redefine la variable 'B030', que se emplea en el Vector objetivo, de modo que no tenga valores nulos.
    df_ele4['B030'] = np.where((df_ele4['B030'].isnull() == True) & (df_ele4['B031'] == 1),0, df_ele4['B030'])
    # Se crea la columna del vector objetivo, para uso posterior.
    df_ele4['VO'] = df_ele4['B030'] + np.where(df_ele4['B034'].isna(),0,df_ele4['B034']) + np.where(df_ele4['B035'].isna(),0,df_ele4['B035'])
    # Se realiza reemplazo de nombres de columans, para equiparar al df de entrenamiento.
    dic_col = {'ID': 'rol_ficticio', 'DV': 'dv_rol_ficticio', 'Tamaño':'Tamano','CIIUfinal':'CIIU_final', 'GlosaCIIU':'Glosa_CIIU', 'Panel_Efectivo': 'Panel'}
    df_ele4 = df_ele4.rename(columns=dic_col)
    # Se recodifica columna 'Tamaño', para equiparar al dataset de entrenamiento.
    df_ele4['Tamano'] = df_ele4['Tamano'].replace('Grande', '1').replace('Mediana', '2').replace('Pequeña 2', '3').replace('Pequeña 1', '4').replace('Micro', '5')
    col_names = ['VO']
    for i in df_ele4.columns:
        if i not in col_names:
            col_names.append(i)
    df_ele4 = df_ele4.reindex(columns=col_names)
    # Filtrado de variables no atingentes al análisis:
    df_ele4_filt = df_ele4.loc[:,:'C072'].drop(columns=['B030','B031','B032','B034','B035','B036','B037','B038','B039','B040','B041','B042','B043','B044','B045','B046','B047','B048','B049','B050','B051','B052','B053','B054','B055','B056','B057','B058','B059','B060','B061','B062','B063','B064','B065','B066','B067','B068','B069','B070','B071','B072','B073','B074','B075','B076','B077','B078','B079','B080','B081','B082','B083','B084','B085','B086','B087',
                                                    'C001','C002','C003','C004','C005','C006','C007','C008','C009','C010','C011','C012','C013','C014','C015','C016','C017','C018','C019','C020','C021','C022','C023','C024','C025','C026','C027','C028','C029','C030','C031','C032','C033','C034','C035','C036','C037','C038','C039','C040','C042','C043','C044','C045','C046','C047','C048','C049','C050','C051','C052','C053','C054','C055','C057','C058','C059','C060',
                                                     'C061','C063','C065','C066','C069','C070','C071','rol_ficticio','dv_rol_ficticio','FE_Ventas','FE_Empresas','Panel','Glosa_CIIU', 'TipoPanel', 'VO'])
    # Filtrado de variables dicotómicas que según análisis chi-cuadrado (en set de entrenamiento), no aportan a explicar el VO
    df_ele4_filt = df_ele4_filt.drop(columns = var_dic_eliminar)
    # A continuación, se imputan los missing values de las variables categóricas a cero.
    # Lo anterior, responde a que las variables en el set de validación sólo vienen con valores '1', y no indican 
    # ausencia del fenómeno medido por cada una, por lo tanto se requiere equiparar al formato del dataset de entrenamiento.
    
    for i in df_ele4_filt.columns:
        df_ele4_filt[i] = np.where(df_ele4_filt[i].isna(),0,df_ele4_filt[i])
    # Finalmente, se realizan las transformaciones logarítmicas de las variables contínuas, y se dropean las originales:
    df_ele4_filt_trans = df_ele4_filt.copy()
    df_ele4_filt_trans['log_C041'] = np.log(df_ele4_filt_trans['C041']+1)
    df_ele4_filt_trans['log_C056'] = np.log(df_ele4_filt_trans['C056']+1)
    df_ele4_filt_trans['log_C062'] = np.log(np.where(df_ele4_filt_trans['C062'] < 0, 0, df_ele4_filt_trans['C062'])+1)
    df_ele4_filt_trans['log_C064'] = np.log(df_ele4_filt_trans['C064']+1)
    df_ele4_filt_trans['log_C067'] = np.log(df_ele4_filt_trans['C067']+1)
    df_ele4_filt_trans['log_C072'] = np.log(np.where(df_ele4_filt_trans['C072'] < 0, 0, df_ele4_filt_trans['C072'])+1)
    df_ele4_filt_trans['antiguedad'] = 2015 - df_ele4_filt_trans['A068']
    
    df_ele4_filt_trans = df_ele4_filt_trans.drop(columns=['C041', 'C056', 'C062', 'C064', 'C067', 'C072', 'A068'], axis = 1)
    df_ele4_filt_trans_c= df_ele4_filt_trans.copy()
    # Se realiza la binarización de las categóricas.
    for i in lista_cat:
        dum = pd.get_dummies(df_ele4_filt_trans_c[i], prefix=i, drop_first = True)
        df_ele4_filt_trans_c = pd.concat([df_ele4_filt_trans_c, dum], axis=1)

    for i in lista_cat:
        df_ele4_filt_trans_c = df_ele4_filt_trans_c.drop(columns=i)
        
    if tipo_prep == 'df_completo':
        return df_ele4
    elif tipo_prep == 'df_validacion':
        return df_ele4_filt_trans_c
    
# apply threshold to positive probabilities to create labels

def to_labels(pos_probs, threshold):
    """
    Permite mapear todos los valores mayores o iguales al umbral 'threshold' a 1, y los valores menores a 0.
    """
    return (pos_probs >= threshold).astype('int')

def tto_outliers(df, campo):
    '''
    Definición:
    Genera variables transformadas considerando tratamiento de outliers, de modo que reemplaza los valores /
    outliers por los valores de los percentiles 75 y 25, según la regla de  distancia de 1.5 veces el rango intercuartílico / 
    (Q3 - Q1).
    Parámetros:
    1. 'df': representa el dataframe de pandas que se está analizando.
    2. 'campo': corresponde a la variables que se requiere transformar.
    '''
    campo1 = campo+'_tr'
    df[campo1] = df[campo]
    rs = stats.iqr(df[campo])*1.5+df[campo].describe()[6]
    ri = df[campo].describe()[4]-stats.iqr(df[campo])*1.5
    mask1 = df[campo] > rs
    mask2 = df[campo] < ri
    df.loc[mask1,campo1] = rs
    df.loc[mask2,campo1] = ri
