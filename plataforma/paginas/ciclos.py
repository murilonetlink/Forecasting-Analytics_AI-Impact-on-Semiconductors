"""H1 · Ciclicidade: quedas e recuperações, calendário de retornos e decomposição."""
import streamlit as st

from plataforma import analises as an
from plataforma import graficos, leitura, visual
from plataforma.visual import COR


def _quedas(ctx, sel, limiar):
    series = {c: ctx.serie(c) for c in sel}
    fig = graficos.drawdowns(
        series, ctx.cores, "Quanto cada série está abaixo do seu último pico",
        "Quanto mais fundo o vale, maior a queda. Voltar a zero significa que o pico anterior foi recuperado.",
        ajuda="0% é o ponto em que a série está no seu máximo histórico até aquela data. -40% significa 40% abaixo do último pico. "
              "Vales fundos que se repetem são o desenho de um ciclo.")
    visual.mostrar(fig, "quedas")
    visual.dica("cada vale é uma queda a partir de um pico. Passe o mouse para ver de qual data era o pico e quanto a série estava abaixo dele.")

    principal = st.selectbox("Detalhar as quedas de", sel, key="ci_principal",
                             help="Lista as quedas maiores que o limite escolhido acima, com datas de pico, fundo e recuperação.")
    eps = an.episodios_queda(series[principal], limiar)
    visual.leitura(*leitura.quedas(principal, eps, limiar))
    if not eps.empty:
        tab = eps.copy()
        tab["queda"] = tab["queda"] * 100
        st.dataframe(tab, width="stretch", hide_index=True, column_config={
            "pico": st.column_config.DateColumn("pico", format="DD/MM/YYYY", help="Data do máximo antes da queda."),
            "fundo": st.column_config.DateColumn("fundo", format="DD/MM/YYYY", help="Data do ponto mais baixo da queda."),
            "recuperação": st.column_config.DateColumn("recuperação", format="DD/MM/YYYY", help="Primeira data em que a série voltou ao pico. Vazio: ainda não voltou."),
            "queda": st.column_config.NumberColumn("queda", format="%.0f%%", help="Distância entre o pico e o fundo."),
            "meses até o fundo": st.column_config.NumberColumn(format="%.0f", help="Tempo do pico até o fundo."),
            "meses para recuperar": st.column_config.NumberColumn(format="%.0f", help="Tempo do pico até a recuperação do pico."),
        })


def _calendario(ctx, sel):
    nome = st.selectbox("Série", sel, key="cal_serie", help="Retorno de cada mês do ano, ano a ano.")
    tab = an.tabela_sazonal(ctx.serie(nome, "M"))
    if tab.shape[0] < 3:
        st.warning("Poucos anos neste período.")
        return
    visual.leitura(*leitura.sazonal(tab, an.teste_sazonalidade(tab)))
    c1, c2 = st.columns([3, 2])
    with c1:
        visual.mostrar(graficos.mapa_sazonal(
            tab, "Retorno de cada mês", "Verde sobe, laranja cai. Cada linha é um ano.",
            ajuda="Cada célula é o retorno da série naquele mês. Se existisse um padrão sazonal forte, as mesmas colunas ficariam da mesma cor ano após ano."),
            "calendario")
    with c2:
        visual.mostrar(graficos.barras_mes(
            tab, "Média por mês", "Em quantos anos o mês foi positivo aparece no hover.",
            ajuda="Média dos retornos de cada mês do ano. Barras pequenas perto de zero indicam que não há mês 'bom' de forma consistente."),
            "media_mes")
    visual.dica("sazonalidade é um padrão que se repete todo ano no mesmo mês. Ciclos são mais longos e irregulares: por isso olhamos as duas coisas.")


def _decomposicao(ctx, sel):
    nome = st.selectbox("Série", sel, key="dec_serie", help="A série mensal é separada em tendência, sazonalidade e resíduo.")
    mensal = ctx.serie(nome, "M")
    if len(mensal) < 36:
        st.warning("São necessários ao menos 36 meses para decompor.")
        return
    dec = an.decomposicao(mensal)
    fig = graficos.decomposicao(
        dec, f"Decomposição de {nome}", "Série = tendência + sazonalidade + resíduo (em logaritmo).",
        ajuda="Tendência: para onde a série vai no longo prazo. Sazonalidade: o que se repete todo ano. Resíduo: o que sobra, ou seja, os choques que nenhum dos dois explica.")
    visual.mostrar(fig, "decomposicao")
    visual.dica("se o resíduo for grande perto da tendência, a maior parte do movimento é imprevisível por tendência ou calendário.")


def render(ctx):
    visual.cabecalho("Ciclos", "O setor passa por ciclos de alta e queda que se repetem?", "H1 · CICLICIDADE", COR["mem"])
    c1, c2 = st.columns([3, 1])
    sel = c1.multiselect("Séries", ctx.colunas, default=["Índice Memória", "Índice GPU"], key="ciclos_sel",
                         help="Compare quantas quiser. Por padrão, os dois índices do setor.")
    limiar = c2.slider("Queda mínima (%)", 10, 60, 30, step=5, key="ciclos_lim",
                       help="Uma queda só entra na lista se for maior que isso. Quedas pequenas são ruído normal de mercado.") / 100
    if not sel:
        st.info("Escolha ao menos uma série.")
        return
    abas = st.tabs(["Quedas e recuperações", "Calendário", "Tendência e sazonalidade"])
    with abas[0]:
        _quedas(ctx, sel, limiar)
    with abas[1]:
        _calendario(ctx, sel)
    with abas[2]:
        _decomposicao(ctx, sel)
