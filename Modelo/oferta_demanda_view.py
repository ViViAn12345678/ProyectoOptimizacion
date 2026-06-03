import streamlit as st
import pandas as pd

from Modelo.oferta_demanda import (
    cargar_oferta_demanda,
    guardar_oferta_demanda
)

def mostrar_oferta_demanda(grafo):

    st.title("Oferta y Demanda")

    datos = []

    for nodo in grafo.nodes():

        tipo = grafo.nodes[nodo]["tipo"]

        datos.append({
            "id_nodo": nodo,
            "nombre": grafo.nodes[nodo]["nombre"],
            "tipo": tipo,
            "oferta": 0,
            "demanda": 0
        })

    df = pd.DataFrame(datos)

    for fila in df.index:

        tipo = df.loc[fila,"tipo"]

        if tipo == "origen":

            df.loc[fila,"oferta"] = st.number_input(
                f"Oferta {df.loc[fila,'nombre']}",
                min_value=0,
                key=f"o{fila}"
            )

        elif tipo == "destino":

            df.loc[fila,"demanda"] = st.number_input(
                f"Demanda {df.loc[fila,'nombre']}",
                min_value=0,
                key=f"d{fila}"
            )

    if st.button("Guardar"):

        guardar_oferta_demanda(df)

        st.success("Datos almacenados")