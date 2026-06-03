import pandas as pd
import os

ARCHIVO = "data/oferta_demanda.csv"

def cargar_oferta_demanda():

    if not os.path.exists(ARCHIVO):
        return pd.DataFrame(
            columns=[
                "id_nodo",
                "nombre",
                "tipo",
                "oferta",
                "demanda"
            ]
        )

    return pd.read_csv(ARCHIVO)


def guardar_oferta_demanda(df):

    df.to_csv(
        ARCHIVO,
        index=False
    )