import pandas as pd
from ..features.feature_engineering import feature_engineering
from app import cos, init_cols


def make_dataset(data, model_info, cols_to_remove, model_type='RandomForest'):

    """
        Función que permite crear el dataset usado para el entrenamiento
        del modelo.

        Args:
           data (List):  Lista con la observación llegada por request.
           model_info (dict):  Información del modelo en producción.

        Kwargs:
           model_type (str): tipo de modelo usado.

        Returns:
           DataFrame. Dataset a inferir.
    """

    print('---> Getting data')
    data_df = get_raw_data_from_request(data)

    return data_df.copy()


def get_raw_data_from_request(data):

    """
        Función para obtener nuevas observaciones desde request

        Args:
           data (List):  Lista con la observación llegada por request.

        Returns:
           DataFrame. Dataset con los datos de entrada.
    """
    return pd.DataFrame(data, columns=init_cols)