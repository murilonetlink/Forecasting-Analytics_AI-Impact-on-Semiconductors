"""H2 · Novo patamar: o preço se acomodou mais alto depois de um marco?"""
from datetime import date

import pandas as pd
import streamlit as st

from plataforma import analises as an
from plataforma import graficos, leitura, visual
from plataforma.visual import COR

MARCO_PADRAO = date(2022, 11, 30)


def _periodos(ctx, nome, marco):
    tab = ctx.tabela([nome], "M")
    r = an.comparar_periodos(tab, pd.Timestamp(marco))
    a, d = r["antes"].loc[nome], r["depois"].loc[nome]
    visual.leitura(*leitura.periodos(nome, r["antes"], r["depois"], marco))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Retorno mensal antes", f"{a['retorno médio mensal (%)']:.2f}%".replace(".", ","),
              help=f"Retorno médio mensal até {marco:%d/%m/%Y}, em {int(a['meses'])} meses.")
    c2.metric("Retorno mensal depois", f"{d['retorno médio mensal (%)']:.2f}%".replace(".", ","),
              delta=f"{d['retorno médio mensal (%)'] - a['retorno médio mensal (%)']:+.2f} p.p.".replace(".", ","),
              help=f"Retorno médio mensal depois de {marco:%d/%m/%Y}, em {int(d['meses'])} meses.")
    c3.metric("Volatilidade antes", f"{a['volatilidade anualizada (%)']:.0f}% a.a.", help="Tamanho típico das oscilações antes do marco, ao ano.")
    c4.metric("Volatilidade depois", f"{d['volatilidade anualizada (%)']:.0f}% a.a.",
              delta=f"{d['volatilidade anualizada (%)'] - a['volatilidade anualizada (%)']:+.0f} p.p.", delta_color="off",
              help="Tamanho típico das oscilações depois do marco, ao ano.")


def _adf(ctx, nome):
    s = ctx.serie(nome, "M")
    if len(s) < 30:
        st.warning("Poucas observações neste período.")
        return
    n, d = an.teste_adf(an.nivel_log(s)), an.teste_adf(an.variacao(s))
    visual.leitura(*leitura.adf(nome, n, d))
    tab = pd.DataFrame([{"transformação": "nível (log)", **n}, {"transformação": "1ª diferença (retorno)", **d}])
    st.dataframe(tab, width="stretch", hide_index=True, column_config={
        "estatística": st.column_config.NumberColumn(format="%.3f", help="Valor do teste. Quanto mais negativo, mais evidência contra a raiz unitária."),
        "p-valor": st.column_config.NumberColumn(format="%.4f", help="Abaixo de 0,05 rejeita-se a raiz unitária (série estacionária)."),
        "defasagens": st.column_config.NumberColumn(help="Defasagens escolhidas automaticamente (AIC) para limpar a autocorrelação do teste."),
        "crítico 5%": st.column_config.NumberColumn(format="%.3f", help="Se a estatística for menor que este valor, a série é estacionária a 5%."),
        "estacionária (5%)": st.column_config.CheckboxColumn(help="Conclusão do teste ao nível de 5%."),
    })
    visual.dica("uma série estacionária 'volta' a uma média. Se o preço em nível não volta, os choques são permanentes: é o que a H2 chama de novo patamar.")


def render(ctx):
    visual.cabecalho("Novo patamar", "Depois de uma grande alta, o preço volta ao nível anterior ou se acomoda mais alto?", "H2 · NOVO PATAMAR", COR["mem"])
    c1, c2 = st.columns([2, 1])
    nome = c1.selectbox("Série", ctx.colunas, index=0, key="pat_serie", help="Série cujo patamar será analisado, em frequência mensal.")
    s = ctx.serie(nome, "M")
    if len(s) < 24:
        st.warning("Poucas observações neste período.")
        return
    lo, hi = s.index[6].date(), s.index[-6].date()
    marco = c2.date_input("Marco", value=min(max(MARCO_PADRAO, lo), hi), min_value=lo, max_value=hi, format="DD/MM/YYYY", key="pat_marco",
                          help="Data que separa 'antes' e 'depois'. O padrão é 30/11/2022, o lançamento do ChatGPT.")

    info = an.novo_patamar(s, pd.Timestamp(marco))
    visual.leitura(*leitura.patamar(nome, info))
    fig = graficos.patamar(
        s, pd.Timestamp(marco), info, nome, ctx.cores.get(nome, COR["mem"]), f"{nome} antes e depois do marco",
        "A linha pontilhada dourada é o pico anterior ao marco. Se a série fica acima dela, o patamar novo se sustentou.",
        ajuda="Cinza: antes do marco. Colorido: depois. A linha pontilhada marca o maior valor atingido antes do marco. "
              "Escala logarítmica, então a mesma inclinação significa o mesmo ritmo percentual.")
    visual.mostrar(fig, "patamar")

    st.markdown("#### Antes × depois", help="Compara o retorno médio mensal e a volatilidade nos dois lados do marco.")
    _periodos(ctx, nome, marco)
    with st.expander("Teste estatístico: a série volta a uma média? (ADF)"):
        _adf(ctx, nome)
