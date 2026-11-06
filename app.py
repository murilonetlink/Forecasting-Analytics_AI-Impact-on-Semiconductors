"""Plataforma de análise: IA × ações de memória e GPU.

Executar:  streamlit run app.py
"""
import warnings
from functools import partial

import streamlit as st

from plataforma import calculos, dados, visual
from plataforma.contexto import FREQ_ROTULOS, Contexto
from plataforma.paginas import ciclos, comparar, conclusao, dados_pg, eventos, inicio, modelo_base, palco, patamar, relacoes, risco

warnings.filterwarnings("ignore")
st.set_page_config(page_title="IA × Semicondutores", page_icon=":material/memory:", layout="wide")
visual.aplicar_tema()

painel, categorias = calculos.carregar_painel(dados.versao_cache())
if painel is None:
    st.title("IA e o preço das ações de memória e GPU")
    st.info("Ainda não há dados locais.")
    if st.button("Baixar dados do Yahoo Finance", type="primary"):
        with st.spinner("Baixando..."):
            dados.baixar()
        st.cache_data.clear()
        st.rerun()
    st.stop()

PAGINAS = [  # chave, função que desenha a página, título, ícone, seção do menu (None = fora do menu, só por link direto)
    ("inicio", inicio.render, "Início", ":material/home:", "Começar"),
    ("ciclos", ciclos.render, "H1 · Ciclos", ":material/waves:", "Hipóteses do projeto"),
    ("patamar", patamar.render, "H2 · Novo patamar", ":material/trending_up:", "Hipóteses do projeto"),
    ("eventos", eventos.render, "H3 · Lançamentos de IA", ":material/bolt:", "Hipóteses do projeto"),
    ("comparar", comparar.render, "Comparar séries", ":material/compare_arrows:", "Ferramentas"),
    ("relacoes", relacoes.render, "Quem anda junto", ":material/hub:", "Ferramentas"),
    ("risco", risco.render, "Risco e volatilidade", ":material/speed:", "Ferramentas"),
    ("arima", partial(modelo_base.render, familia="arima"), "ARIMA", ":material/timeline:", "Modelos de previsão"),
    ("sarima", partial(modelo_base.render, familia="sarima"), "SARIMA", ":material/calendar_month:", "Modelos de previsão"),
    ("sarimax", partial(modelo_base.render, familia="sarimax"), "SARIMAX", ":material/add_chart:", "Modelos de previsão"),
    ("arch", partial(modelo_base.render, familia="arch"), "ARCH", ":material/show_chart:", "Modelos de previsão"),
    ("garch", partial(modelo_base.render, familia="garch"), "GARCH", ":material/ssid_chart:", "Modelos de previsão"),
    ("melhor", conclusao.render, "Melhor modelo", ":material/emoji_events:", "Modelos de previsão"),
    ("dados", dados_pg.render, "Dados", ":material/database:", "Dados"),
    ("palco", palco.render, "Palco (apresentação)", ":material/slideshow:", None),
]


def _pagina(chave, desenhar):
    def executar():
        st.session_state.setdefault("visitadas", set()).add(chave)
        desenhar(ctx)
    executar.__name__ = chave
    return executar


paginas = {chave: st.Page(_pagina(chave, desenhar), title=titulo, icon=icone, url_path=chave, default=chave == "inicio")
           for chave, desenhar, titulo, icone, _ in PAGINAS}

with st.sidebar:
    # Menu próprio: o nativo não permite esconder uma página, e o Palco só deve abrir por link direto (/palco).
    secao_atual = None
    for chave, _, titulo, icone, secao in PAGINAS:
        if secao is None:
            continue
        if secao != secao_atual:
            st.markdown(f'<div class="menu-secao">{secao}</div>', unsafe_allow_html=True)
            secao_atual = secao
        st.page_link(paginas[chave], label=titulo, icon=icone)
    st.divider()
    st.markdown("### Filtros", help="Valem para as páginas de hipóteses e ferramentas. Os modelos usam sempre o histórico completo.")
    lo, hi = painel.index.min().date(), painel.index.max().date()
    periodo = st.date_input(
        "Período", value=(lo, hi), min_value=lo, max_value=hi, format="DD/MM/YYYY",
        help="Recorta as análises para este intervalo. Selecione a data inicial e a final. No estudo de eventos, "
             "o recorte filtra apenas quais eventos entram; o histórico anterior continua sendo usado para estimar o normal.")
    freq = st.radio(
        "Frequência", list(FREQ_ROTULOS), index=2, format_func=FREQ_ROTULOS.get,
        help="Um ponto por dia, semana ou mês (último valor do período). Mensal é mais limpo; diário tem mais dados e é melhor para "
             "volatilidade. Os modelos de previsão usam sempre a base diária. O estudo de eventos também é diário.")
    st.caption("Passe o mouse nos ícones de interrogação para ver o que cada controle faz.")
if len(periodo) != 2:
    st.stop()

ctx = Contexto(painel, categorias, str(periodo[0]), str(periodo[1]), freq)
ctx.paginas = paginas

st.navigation(list(paginas.values()), position="hidden").run()
