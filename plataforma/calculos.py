"""Cálculos pesados com cache, compartilhados entre as páginas e o modo apresentação."""
import pandas as pd
import streamlit as st

from plataforma import analises as an
from plataforma import dados


@st.cache_data(show_spinner=False)
def carregar_painel(versao):
    return dados.carregar()


@st.cache_data(show_spinner=False)
def garch(s, nomes, dist, horizonte, freq):
    r = an.variacao(s)
    ajustes, tabela = an.comparar_garch(r, list(nomes), dist)
    vols = pd.DataFrame({n: an.vol_condicional_anual(res, freq) for n, res in ajustes.items()})
    prev = pd.DataFrame({n: an.prever_vol(res, n, horizonte, freq) for n, res in ajustes.items()})
    params = {n: pd.DataFrame({"coef": res.params, "p-valor": res.pvalues}) for n, res in ajustes.items()}
    return tabela, vols, prev, params, an.efeito_arch(r)


@st.cache_data(show_spinner=False)
def selecao_arima(y, d):
    return an.selecionar_ordem(y, d)


@st.cache_data(show_spinner=False)
def arima(y, ordem, n_teste, horizonte, freq):
    metricas, teste = an.avaliar_arima(y, ordem, n_teste)
    prev, res = an.prever_arima(y, ordem, horizonte, freq)
    return metricas, teste, prev, res.aic, res.bic
