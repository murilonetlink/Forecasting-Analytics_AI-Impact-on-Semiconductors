"""Cálculos pesados com cache, compartilhados entre as páginas e o modo apresentação."""
import json

import pandas as pd
import streamlit as st

from plataforma import analises as an
from plataforma import dados, modelos


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


# ----------------------------------------------------------------- modelos de previsão
def versao_modelos():
    """Muda quando os dados, os eventos ou os resultados guardados mudam."""
    arqs = [dados.ARQ_PRECOS, dados.ARQ_EVENTOS, modelos.ARQ_RESULTADOS]
    return tuple(a.stat().st_mtime_ns if a.exists() else 0 for a in arqs)


def cfg_json(cfg):
    return json.dumps(cfg, sort_keys=True)


@st.cache_data(show_spinner=False)
def serie_modelo(nome, versao):
    painel, _ = dados.carregar()
    return modelos.preparar(painel, nome)


@st.cache_data(show_spinner="Calculando o modelo; sem os resultados guardados isso pode levar alguns minutos...")
def entrada_modelo(nome, familia, versao):
    """Melhor configuração, teste e previsão de uma família: do arquivo guardado, ou calculados na hora."""
    serie = serie_modelo(nome, versao)
    guardada = modelos.entrada_guardada(serie, familia)
    if guardada is not None:
        return guardada
    base = entrada_modelo(nome, "sarima", versao)["selecao"]["config"] if familia == "sarimax" else None
    return modelos.calcular_entrada(serie, familia, base=base)


@st.cache_data(show_spinner=False)
def previsao_manual(nome, familia, cfg, versao, taxa=None):
    serie = serie_modelo(nome, versao)
    mu, sd, _ = modelos.prever(serie, familia, json.loads(cfg), serie.n - 1, taxa)
    return {"mu": mu.tolist(), "sd": sd.tolist()}


@st.cache_data(show_spinner="Testando a configuração nos últimos 5 anos...")
def backtest_manual(nome, familia, cfg, versao, taxa=None):
    serie = serie_modelo(nome, versao)
    mu, sd, falhas = modelos.backtest(serie, familia, json.loads(cfg), taxa=taxa)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "falhas": falhas}


@st.cache_data(show_spinner=False)
def diagnostico_modelo(nome, familia, cfg, versao, taxa=None):
    return modelos.diagnostico(serie_modelo(nome, versao), familia, json.loads(cfg), taxa)


@st.cache_data(show_spinner=False)
def comparacao(nome, versao):
    serie = serie_modelo(nome, versao)
    return modelos.comparar(serie, {f: entrada_modelo(nome, f, versao) for f in modelos.FAMILIAS})
