import numpy as np
import pandas as pd

def describe_df(df):
    '''
    Esta función muestra información específica del DF original. 
    Esa información será: el tipo de objeto, el % de valores nulos o missings,
    los valores únicos y el % de cardinalidad de cada columna del DF original para tener 

    Argumentos:
    df: DF original sobre el que queremos recibir la información.

    Retorna:
    DF con la información específica.
    '''

    #Creo un diccionario con la columna que va a estar fija
    #Y después añadir las columnas del DF original
    dict_col = {'COL_N': ['DATA_TYPE', 'MISSINGS (%)', 'UNIQUE_VALUES', 'CARDIN (%)']}

    #Fórmula para calcular el porcentaje de nulos
    na_ratio = ((df.isnull().sum() / len(df))*100)

    #Añado al diccionario como clave el nombre de las columnas, y como valores
    #la información del describe
    for col in df:
        dict_col[col] = [df.dtypes[col], na_ratio[col], len(df[col].unique()), round(df[col].nunique()/len(df)*100,2)]

    # Creo el DF.describe
    df_describe = pd.DataFrame(dict_col)

    return df_describe



def tipifica_variables(df, umbral_categoria, umbral_continua):
    '''
    Esta función sirve para poder tipificar las variables de un DF dado en categórica, numerica continua o numerica discreta.

    Argumentos:
    df: DF original para adquirir las variables que se quiera tificar.
    umbral_categoria: un entero donde corresponda al umbral que queramos asignar a una variable categórica.
    umbral_continua: un float donde corresponda al umbral que queramos asignar a una variable numérica.

    Retorna:
    Un DF con dos columnas 'nombre_varibale' y 'tipo_sugerido', que tendrá tantas filas como columnas haya en el DF original.
    '''

    df_tipificacion = pd.DataFrame({
        'nombre_variable': df.columns
    })

    df_tipificacion['tipo_variable'] = ''

    for i, val in df_tipificacion['nombre_variable'].items():
        card = df[val].nunique()
        porcentaje = (df[val].nunique()/len(df)) * 100

        if card == 2:
            df_tipificacion.at[i,'tipo_variable'] = 'Binaria'
        
        elif card < umbral_categoria:
            df_tipificacion.at[i,'tipo_variable'] = 'Categórica'
    
        else:
            if porcentaje > umbral_continua:
                df_tipificacion.at[i, 'tipo_variable'] = 'Numérica Continua'
            else:
                df_tipificacion.at[i, 'tipo_variable'] = 'Numérica Discreta'
    
    return df_tipificacion
