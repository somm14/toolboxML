import numpy as np
import pandas as pd

def describe_df(df):
    '''
    Esta función muestra información específica del DF original. 
    Esa información será: el tipo de objeto, el % de valores nulos o missings,
    los valores únicos y el % de cardinalidad de cada columna del DF original para tener 

    Argumentos:
    df (pd.DataFrame): DF original sobre el que queremos recibir la información.

    Retorna:
    pd.DataFrame: Df con la información específica.
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
    df (pd.DataFrame): DF original para adquirir las variables que se quiera tificar.
    umbral_categoria (int): un entero donde corresponda al umbral que queramos asignar a una variable categórica.
    umbral_continua (float): un float donde corresponda al umbral que queramos asignar a una variable numérica.

    Retorna:
    pd.DataFrame: Un DF con dos columnas 'nombre_varibale' y 'tipo_sugerido', que tendrá tantas filas como columnas haya en el DF original.
    '''

    if type(umbral_categoria) != int:
        raise TypeError(f'El umbral_categoria debe ser de tipo {int}, pero recibió de tipo {type(umbral_categoria)}')

    elif type(umbral_continua) != float:
        raise TypeError(f'El umbral_continua debe ser de tipo {float}, pero recibió de tipo {type(umbral_continua)}')

    else:

        df_tipificacion = pd.DataFrame({
            'nombre_variable': df.columns
        })

        df_tipificacion['tipo_variable'] = ''

        for i, val in df_tipificacion['nombre_variable'].items():
            card = df[val].nunique()
            porcentaje = df[val].nunique()/len(df) * 100

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


def get_features_num_regression(df, target_col, umbral_corr, pvalue = None):
    '''
    Obtiene las columnas numéricas cuya correlación con target_col es significativa.
    
    Argumentos:
    df (pd.DataFrame): DataFrame que contiene los datos.
    target_col (str): Nombre de la columna objetivo (debe ser numérica).
    umbral_corr (float): Umbral de correlación para filtrar las columnas.
    pvalue (float, optional): Umbral de significancia para el valor p. Si es None, solo se considera el umbral de correlación.
    
    Returns:
    List[str]: Lista de nombres de columnas que cumplen con los criterios.
    '''
    if type(umbral_corr) != float:
        raise TypeError (f'El valor dado en umbral_corr debe ser de tipo {float}, pero recibió un valor de tipo {type(umbral_corr)}')
    elif umbral_corr < 0 or umbral_corr > 1:
        raise ValueError(f'El valor de umbral_corr no está entre 0 y 1 ya que el valor es {umbral_corr}')
        # Aqui falta validar si la target es una variable numerica continua, se puede hcaer con la fucnion tipifica_variables

    else:
        if pvalue == None:
            corr = np.abs(df.corr(numeric_only=True)[target_col])
            corr_list = [i for i,val in corr.items() if val > umbral_corr]
            corr_list.remove(target_col)
            return corr_list
        
        #else: si pvalue no es NONE
