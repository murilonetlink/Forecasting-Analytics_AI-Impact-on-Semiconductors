"""Previsão com ARIMA, avaliada contra o passeio aleatório."""
import numpy as np
import streamlit as st

from plataforma import analises as an
from plataforma import calculos, graficos, leitura, visual


def render(ctx):
    visual.cabecalho("Previsão", "Dá para prever o nível da série? O ARIMA é testado contra o passeio aleatório.")
    nome = st.selectbox("Série", ctx.colunas, index=0, key="arima_serie", help="Série a ser prevista.")
    s = ctx.serie(nome)
    y = an.nivel_log(s)
    em_log = bool((s > 0).all())
    transf = np.exp if em_log else (lambda v: v)
    freq = ctx.freq

    padrao_teste = {"D": 126, "W": 26, "M": 12}[freq]
    padrao_h = {"D": 60, "W": 26, "M": 12}[freq]
    c1, c2, c3, c4, c5 = st.columns(5)
    p = c1.number_input("p", 0, 6, 1, key=f"p{freq}", help="Parte autorregressiva: quantos valores passados da série entram na previsão. 0 = nenhum.")
    d = c2.number_input("d", 0, 2, 1, key=f"d{freq}", help="Quantas vezes diferenciar a série para torná-la estacionária. Preços em log normalmente pedem d = 1.")
    q = c3.number_input("q", 0, 6, 1, key=f"q{freq}", help="Média móvel: quantos erros passados do modelo entram na previsão. 0 = nenhum.")
    n_teste = c4.number_input("Períodos de teste", 4, max(5, len(y) // 3), padrao_teste, key=f"nt{freq}",
                              help="Quantos períodos finais ficam escondidos do modelo para avaliar a previsão de verdade.")
    horizonte = c5.number_input("Horizonte", 1, 500, padrao_h, key=f"h{freq}", help="Quantos períodos à frente prever além do último dado.")

    if len(y) < 40:
        st.warning("Poucas observações neste período e frequência.")
        return

    ordem = (int(p), int(d), int(q))
    try:
        with st.spinner("Ajustando..."):
            metricas, teste, prev, aic, bic = calculos.arima(y, ordem, int(n_teste), int(horizonte), freq)
    except Exception as e:
        st.error(f"Falha ao ajustar ARIMA{ordem}: {e}")
        return

    razao = metricas.loc["RMSE (log)", "ARIMA"] / metricas.loc["RMSE (log)", "Passeio aleatório"]
    ultimo = transf(prev.iloc[-1])
    visual.leitura(*leitura.arima(ordem, razao, int(horizonte), float(ultimo["previsto"]), float(ultimo["inferior"]), float(ultimo["superior"])))

    hist = y.iloc[-max(int(horizonte) * 3, 60):]
    visual.mostrar(graficos.previsao(
        hist, prev, transf, nome, f"Previsão de {nome}, {horizonte} períodos à frente", "A faixa dourada é o intervalo de confiança de 95%.",
        ajuda="A linha dourada é a previsão central; a faixa é onde o valor deve cair 95% das vezes se o modelo estiver certo. "
              "A faixa se abre rápido: é o comportamento esperado de uma série com raiz unitária."), "previsao")

    visual.mostrar(graficos.teste_previsao(
        y.iloc[-int(n_teste) * 3:], teste, transf, "Teste: previsto × observado", "O modelo não viu o trecho tracejado ao ser ajustado.",
        ajuda="Escondemos os últimos períodos, ajustamos o modelo sem eles e comparamos a previsão com o que de fato aconteceu. "
              "A linha azul pontilhada é o passeio aleatório: simplesmente repetir o último valor conhecido."), "teste_previsao")

    m1, m2 = st.columns(2)
    m1.metric(f"AIC ARIMA{ordem}", f"{aic:.1f}", help="Nota de qualidade do ajuste, penalizando complexidade. Só serve para comparar ordens na mesma série. Menor é melhor.")
    m2.metric("Erro do ARIMA ÷ erro do passeio aleatório", f"{razao:.2f}×".replace(".", ","),
              help="Abaixo de 1: o ARIMA errou menos que repetir o último valor. Acima de 1: não compensa usar o modelo.")
    visual.dica("o passeio aleatório é o adversário mínimo. Em preços de ações, ganhar dele é difícil e, quando acontece, vale conferir se não foi sorte do trecho de teste.")

    with st.expander("Buscar a ordem por AIC (p, q até 3)"):
        tab = calculos.selecao_arima(y, int(d)).head(8)
        tab["ordem"] = tab["ordem"].astype(str)
        st.dataframe(tab, width="stretch", hide_index=True, column_config={
            "AIC": st.column_config.NumberColumn(format="%.1f", help="Menor é melhor."),
            "BIC": st.column_config.NumberColumn(format="%.1f", help="Menor é melhor; pune mais a complexidade que o AIC."),
        })
        st.caption("O AIC tende a escolher modelos grandes. Compare com o desempenho fora da amostra mostrado acima.")
    with st.expander("Como ler as métricas de erro"):
        st.dataframe(metricas.style.format("{:.4f}"), width="stretch")
        st.caption("RMSE: raiz do erro quadrático médio (penaliza erros grandes). MAE: erro absoluto médio. Ambos em log: 0,10 equivale a cerca de 10% de erro.")
