"""Plataforma de análise: IA × ações de memória e GPU.

Executar:  streamlit run app.py
"""
import warnings

import streamlit as st

from plataforma import calculos, dados, visual
from plataforma.contexto import FREQ_ROTULOS, Contexto
from plataforma.paginas import ciclos, comparar, dados_pg, eventos, inicio, palco, patamar, previsao, relacoes, risco

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

with st.sidebar:
    st.markdown("### Filtros", help="Valem para as páginas de hipóteses e ferramentas. O Palco usa sempre o histórico completo.")
    lo, hi = painel.index.min().date(), painel.index.max().date()
    periodo = st.date_input(
        "Período", value=(lo, hi), min_value=lo, max_value=hi, format="DD/MM/YYYY",
        help="Recorta as análises para este intervalo. Selecione a data inicial e a final. No estudo de eventos, "
             "o recorte filtra apenas quais eventos entram; o histórico anterior continua sendo usado para estimar o normal.")
    freq = st.radio(
        "Frequência", list(FREQ_ROTULOS), index=2, format_func=FREQ_ROTULOS.get,
        help="Um ponto por dia, semana ou mês (último valor do período). Mensal é mais limpo e é a base dos modelos do projeto; "
             "diário tem mais dados e é melhor para volatilidade. O estudo de eventos é sempre diário.")
    st.caption("Passe o mouse nos ícones de interrogação para ver o que cada controle faz.")
if len(periodo) != 2:
    st.stop()

ctx = Contexto(painel, categorias, str(periodo[0]), str(periodo[1]), freq)

PAGINAS = [  # chave, módulo, título, ícone, seção do menu
    ("inicio", inicio, "Início", ":material/home:", "Começar"),
    ("ciclos", ciclos, "H1 · Ciclos", ":material/waves:", "Hipóteses do projeto"),
    ("patamar", patamar, "H2 · Novo patamar", ":material/trending_up:", "Hipóteses do projeto"),
    ("eventos", eventos, "H3 · Lançamentos de IA", ":material/bolt:", "Hipóteses do projeto"),
    ("comparar", comparar, "Comparar séries", ":material/compare_arrows:", "Ferramentas"),
    ("relacoes", relacoes, "Quem anda junto", ":material/hub:", "Ferramentas"),
    ("risco", risco, "Risco e volatilidade", ":material/speed:", "Ferramentas"),
    ("previsao", previsao, "Previsão", ":material/query_stats:", "Ferramentas"),
    ("palco", palco, "Palco (apresentação)", ":material/slideshow:", "Apresentar"),
    ("dados", dados_pg, "Dados", ":material/database:", "Dados"),
]


def _pagina(chave, modulo):
    def executar():
        st.session_state.setdefault("visitadas", set()).add(chave)
        modulo.render(ctx)
    executar.__name__ = chave
    return executar


menu, ctx.paginas = {}, {}
for chave, modulo, titulo, icone, secao in PAGINAS:
    pagina = st.Page(_pagina(chave, modulo), title=titulo, icon=icone, url_path=chave, default=chave == "inicio")
    ctx.paginas[chave] = pagina
    menu.setdefault(secao, []).append(pagina)

st.navigation(menu).run()
