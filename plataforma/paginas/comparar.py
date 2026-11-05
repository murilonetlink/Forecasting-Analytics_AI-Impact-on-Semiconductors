"""Comparar séries: gráfico livre com base 100 e os números do período."""
import numpy as np
import pandas as pd
import streamlit as st

from plataforma import analises as an
from plataforma import dados, graficos, visual

PADRAO = ["Índice Memória", "Índice GPU", "Cobre (HG=F)", "Ouro (GC=F)"]


def _resumo(ctx, nomes):
    linhas = []
    for c in nomes:
        s = ctx.serie(c)
        if len(s) < 3:
            continue
        anos = (s.index[-1] - s.index[0]).days / 365.25
        positiva = bool((s > 0).all())
        linhas.append({
            "série": c,
            "retorno total (%)": (s.iloc[-1] / s.iloc[0] - 1) * 100 if positiva else np.nan,
            "retorno anual composto (%)": ((s.iloc[-1] / s.iloc[0]) ** (1 / anos) - 1) * 100 if positiva and anos > 0 else np.nan,
            "volatilidade anual (%)": an.variacao(s).std() * np.sqrt(an.PERIODOS_ANO[ctx.freq]) * 100,
            "maior queda (%)": an.drawdown(s).min() * 100 if positiva else np.nan,
        })
    return pd.DataFrame(linhas).set_index("série")


def render(ctx):
    visual.cabecalho("Comparar séries", "Essas séries andam juntas? Coloque quantas quiser no mesmo gráfico.")
    sel = st.multiselect("Séries", ctx.colunas, default=[c for c in PADRAO if c in ctx.painel], key="cmp_sel",
                         format_func=lambda c: f"{c}  ·  {ctx.categorias.get(c, '')}",
                         help="Escolha o que comparar. Depois do nome aparece a categoria. Séries que você enviou entram como 'Dados próprios'.")
    if not sel:
        st.info("Escolha ao menos uma série.")
        return

    c1, c2, c3 = st.columns(3)
    rebase = c1.checkbox("Base 100", True, key="cmp_base",
                         help="Faz todas as séries começarem em 100 na primeira data em que todas existem. Necessário para comparar ativos com preços e moedas diferentes.")
    log = c2.checkbox("Escala logarítmica", True, key="cmp_log",
                      help="Variações percentuais iguais ocupam o mesmo espaço no gráfico. Sem isso, séries que multiplicam de valor esmagam o começo.")
    mostrar_ev = c3.checkbox("Marcar lançamentos de IA", False, key="cmp_ev",
                             help="Pontos dourados sobre a primeira série escolhida, nas datas da tabela de eventos. Passe o mouse para ver cada evento.")

    d = ctx.tabela(sel)
    if d.dropna().empty:
        st.warning("As séries escolhidas não têm datas em comum neste período.")
        return
    if rebase:
        d = an.base100(d)
        st.caption(f"Base 100 em {d.index[0]:%d/%m/%Y}, a primeira data com todas as séries escolhidas.")
    fig = graficos.linhas(
        d, ctx.cores, "Séries selecionadas", "Passe o mouse para ver o valor de todas na mesma data.",
        base100=rebase, log=log and bool((d > 0).all().all()),
        eventos=dados.carregar_eventos() if mostrar_ev else None,
        ajuda="Cada linha é uma série. Em base 100, o valor mostra quanto ela acumulou desde o início: 200 = dobrou, 50 = caiu pela metade. "
              "O hover também traz a variação percentual desde a base.")
    visual.mostrar(fig, "comparar_series")
    visual.dica("linhas que sobem e descem juntas indicam relação; linhas que se afastam indicam desempenhos diferentes. "
                "Em escala logarítmica, a mesma inclinação significa o mesmo ritmo de crescimento percentual.")

    with st.expander("Números do período"):
        tab = _resumo(ctx, sel)
        st.dataframe(tab, width="stretch", column_config={
            "retorno total (%)": st.column_config.NumberColumn(format="%.0f%%", help="Quanto a série variou do primeiro ao último ponto do período."),
            "retorno anual composto (%)": st.column_config.NumberColumn(format="%.1f%%", help="Ritmo médio de variação por ano, com juros compostos."),
            "volatilidade anual (%)": st.column_config.NumberColumn(format="%.1f%%", help="Tamanho típico das oscilações, convertido para ao ano."),
            "maior queda (%)": st.column_config.NumberColumn(format="%.0f%%", help="Maior distância entre um pico e o fundo seguinte."),
        })
