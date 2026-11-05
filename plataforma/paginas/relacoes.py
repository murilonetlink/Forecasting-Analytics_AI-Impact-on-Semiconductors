"""Quem anda junto: correlações, Granger e cointegração."""
import pandas as pd
import streamlit as st

from plataforma import analises as an
from plataforma import graficos, leitura, visual


def _categorias_padrao(alvo, todas):
    excluir = set()
    if alvo == "Índice Memória":
        excluir = {"Memória"}
    elif alvo.startswith("Índice GPU"):
        excluir = {"GPU / IA", "Semicondutores"}
    return [c for c in todas if c not in excluir]


def _ranking(ctx):
    c1, c2 = st.columns([2, 1])
    alvo = c1.selectbox("Comparar todas as séries com", ctx.colunas, index=0, key="rk_alvo",
                        help="A série de referência. O ranking mostra quanto cada outra série se move junto com ela.")
    atraso = c2.radio("Quando?", [0, 1], horizontal=True, key="rk_lag",
                      format_func={0: "Mesmo período", 1: "1 período antes"}.get,
                      help="'1 período antes' compara o alvo de hoje com a outra série do período anterior. "
                           "Se uma série costuma andar antes da outra, a correlação com atraso aparece aqui.")
    todas = sorted(set(ctx.categorias.values()))
    cats = st.multiselect("Categorias incluídas", todas, default=_categorias_padrao(alvo, todas), key=f"rk_cats_{alvo}",
                          help="Por padrão ficam de fora as empresas que compõem o próprio índice, para não comparar a série com ela mesma.")
    cand = [c for c in ctx.colunas if c != alvo and ctx.categorias.get(c) in cats
            and not (alvo.startswith("Índice GPU") and c.startswith("Índice GPU"))]
    if not cand:
        st.info("Escolha ao menos uma categoria.")
        return
    rank = an.ranking_correlacao(ctx.tabela([alvo] + cand), alvo, atraso)
    if rank.empty:
        st.info("Poucas observações em comum para calcular.")
        return
    visual.leitura(*leitura.ranking(alvo, rank, atraso))
    fig = graficos.barras_correlacao(
        rank, alvo, f"Correlação dos retornos com {alvo}", f"Frequência {ctx.rotulo_freq.lower()}. Barras douradas andam junto; azuis andam em sentido oposto.",
        ajuda="Correlação vai de -1 a +1. +1: sobem e descem juntas. 0: sem relação. -1: sentidos opostos. É calculada sobre os retornos, "
              "não sobre os preços, para não confundir tendência com relação real.")
    visual.mostrar(fig, "ranking_correlacao")
    visual.dica("quanto mais longa a barra, mais a série se move com o alvo. Correlação não prova causa: duas séries podem andar juntas "
                "porque ambas reagem a uma terceira coisa, como o humor do mercado.")


def _mapa(ctx):
    padrao = [c for c in ["Índice Memória", "Índice GPU", "Cobre (HG=F)", "Ouro (GC=F)", "S&P 500 (^GSPC)", "VIX (volatilidade) (^VIX)"]
              if c in ctx.painel]
    sel = st.multiselect("Séries", ctx.colunas, default=padrao, key="mp_sel", help="Mínimo de duas séries.")
    if len(sel) < 2:
        st.info("Escolha ao menos duas séries.")
        return
    d = ctx.tabela(sel)
    r = pd.concat([an.variacao(d[c]).rename(c) for c in sel], axis=1).dropna()
    if len(r) < 12:
        st.warning("Poucas observações em comum.")
        return
    fig = graficos.mapa_correlacao(
        r.corr(), "Mapa de correlações", f"Retornos {ctx.rotulo_freq.lower()}s, {len(r)} períodos em comum. Dourado = andam juntas, azul = sentidos opostos.",
        ajuda="Cada célula é a correlação entre duas séries. A diagonal é sempre 1 (a série com ela mesma).")
    visual.mostrar(fig, "mapa_correlacao")


def _no_tempo(ctx):
    c1, c2, c3 = st.columns(3)
    a = c1.selectbox("Série A", ctx.colunas, index=0, key="nt_a", help="Primeira série do par.")
    b = c2.selectbox("Série B", ctx.colunas, index=min(1, len(ctx.colunas) - 1), key="nt_b", help="Segunda série do par.")
    janela = c3.number_input("Janela (períodos)", 5, 500, {"D": 63, "W": 26, "M": 12}[ctx.freq], key=f"nt_j{ctx.freq}",
                             help="Quantos períodos entram em cada cálculo. Janela curta reage rápido, mas é mais ruidosa.")
    if a == b:
        st.info("Escolha séries diferentes.")
        return
    d = ctx.tabela([a, b])
    cm = an.corr_movel(d[a], d[b], int(janela)).dropna()
    if cm.empty:
        st.warning("Janela maior que o número de observações em comum.")
        return
    fig = graficos.correlacao_movel(
        cm, a, b, f"Correlação móvel: {a} × {b}", f"Janela de {int(janela)} períodos ({ctx.rotulo_freq.lower()}). Acima de zero andam juntas.",
        ajuda="A correlação recalculada a cada período, olhando só a janela mais recente. Mostra se a relação é estável ou muda com o tempo.")
    visual.mostrar(fig, "correlacao_movel")


def _granger(ctx):
    c1, c2, c3 = st.columns(3)
    padrao_x = ctx.colunas.index("Cobre (HG=F)") if "Cobre (HG=F)" in ctx.colunas else 0
    x = c1.selectbox("Série X (a possível causa)", ctx.colunas, index=padrao_x, key="gr_x", help="Ex.: cobre, ouro, juros.")
    y = c2.selectbox("Série Y (a possível consequência)", ctx.colunas, index=0, key="gr_y", help="Ex.: Índice Memória. A tabela mostra os dois sentidos.")
    lag = c3.slider("Defasagem máxima", 1, 12, 4, key="gr_lag",
                    help="Até quantos períodos atrás o teste olha. Na frequência mensal, 4 = até 4 meses.")
    if x == y:
        st.info("Escolha séries diferentes.")
        return
    d = ctx.tabela([x, y])
    try:
        xy = an.granger(d[x], d[y], lag).rename(f"{x} → {y}")
        yx = an.granger(d[y], d[x], lag).rename(f"{y} → {x}")
    except Exception as e:
        st.error(f"Não foi possível calcular: {e}")
        return
    visual.leitura(*leitura.granger(x, y, xy, yx))
    tab = pd.concat([xy, yx], axis=1)
    st.dataframe(tab.style.format("{:.4f}").map(lambda v: "background-color: rgba(217,164,65,.35)" if v < 0.05 else ""), width="stretch")
    visual.dica("cada linha é uma defasagem; a célula é o p-valor. Células destacadas (p < 0,05) indicam que o passado da primeira série ajuda a "
                "prever a segunda. 'Causalidade de Granger' é precedência estatística, não causa de verdade.")


def _cointegracao(ctx):
    c1, c2 = st.columns(2)
    a = c1.selectbox("Série A", ctx.colunas, index=0, key="ci_a", help="Primeira série do par.")
    b = c2.selectbox("Série B", ctx.colunas, index=min(1, len(ctx.colunas) - 1), key="ci_b",
                     help="Segunda série do par. O teste é sensível à ordem; se quiser, inverta A e B e compare.")
    if a == b:
        st.info("Escolha séries diferentes.")
        return
    d = ctx.tabela([a, b])
    try:
        r = an.cointegracao(d[a], d[b])
    except Exception as e:
        st.error(f"Não foi possível calcular: {e}")
        return
    visual.leitura(*leitura.cointegracao(a, b, r))
    m1, m2, m3 = st.columns(3)
    m1.metric("Estatística", f"{r['estatística']:.3f}", help=f"Se for menor que o valor crítico de 5% ({r['crítico 5%']:.3f}), há cointegração.")
    m2.metric("p-valor", f"{r['p-valor']:.4f}", help="Abaixo de 0,05: rejeita-se a ausência de cointegração.")
    m3.metric("Observações", r["observações"], help="Períodos em comum entre as duas séries.")
    visual.dica("duas séries cointegradas são como um cão e o dono na coleira: cada um anda para um lado, mas a distância entre eles não cresce sem limite.")


def render(ctx):
    visual.cabecalho("Quem anda junto", "Quais séries se movem com a memória e a GPU, e quem chega primeiro?")
    abas = st.tabs(["Ranking", "Mapa", "No tempo", "Quem prevê quem", "Ligação de longo prazo"])
    with abas[0]:
        _ranking(ctx)
    with abas[1]:
        _mapa(ctx)
    with abas[2]:
        _no_tempo(ctx)
    with abas[3]:
        st.caption("Teste de causalidade de Granger: o passado de X ajuda a prever Y?")
        _granger(ctx)
    with abas[4]:
        st.caption("Teste de cointegração de Engle-Granger: as duas séries têm uma relação estável de longo prazo?")
        _cointegracao(ctx)
