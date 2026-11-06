"""Conclusão: qual modelo prevê melhor, com ranking, erro por horizonte e previsões lado a lado."""
import numpy as np
import pandas as pd
import streamlit as st

from plataforma import calculos, graficos, modelos, visual
from plataforma import leitura as L
from plataforma.modelos import FAMILIAS, NOMES
from plataforma.visual import COR, COR_MODELO

CRITERIOS = {
    "rmse": "Erro médio nos 12 meses (RMSE)",
    "rmse_12": "Erro a 12 meses à frente",
    "mae": "Erro absoluto médio (MAE)",
    "crps": "Nota da previsão completa (CRPS)",
}
AJUDA_CRITERIOS = {
    "rmse": "Raiz do erro quadrático médio das previsões de 1 a 12 meses à frente, em log. Penaliza mais os erros grandes. É o critério padrão.",
    "rmse_12": "O mesmo erro, olhando só as previsões feitas 12 meses antes. É o horizonte do projeto, mas tem menos pontos independentes.",
    "mae": "Erro absoluto médio em log. Menos sensível a poucos erros muito grandes.",
    "crps": "Avalia a previsão inteira: o valor central e a largura do intervalo. Premia intervalos bem calibrados. Menor é melhor.",
}


def render(ctx):
    visual.cabecalho("Melhor modelo", "Qual modelo prevê melhor os próximos 12 meses?", "CONCLUSÃO", COR["gold"])
    c1, c2 = st.columns([2, 3])
    nome = c1.radio("Índice", modelos.SERIES, horizontal=True, key="concl_serie", help="O melhor modelo pode ser diferente para cada índice.")
    criterio = c2.selectbox("Critério para escolher o melhor", list(CRITERIOS), format_func=CRITERIOS.get, key="concl_crit",
                            help=" ".join(f"{CRITERIOS[k]}: {v}" for k, v in AJUDA_CRITERIOS.items()))

    versao = calculos.versao_modelos()
    serie = calculos.serie_modelo(nome, versao)
    tab, por_h, melhor_base, real, previsoes = calculos.comparacao(nome, versao)
    entradas = {f: calculos.entrada_modelo(nome, f, versao) for f in FAMILIAS}
    so = tab.loc[list(FAMILIAS)]
    melhor, melhor_crps = so[criterio].idxmin(), so["crps"].idxmin()
    linha = tab.loc[melhor]
    desc = modelos.descrever(melhor, entradas[melhor]["config"])
    base_nome = NOMES[melhor_base].lower()

    visual.leitura(*L.conclusao(nome, desc, CRITERIOS[criterio], linha, base_nome, linha["dm_p"],
                                None if melhor_crps == melhor else modelos.descrever(melhor_crps, entradas[melhor_crps]["config"])))

    atual = float(np.exp(serie.y.iloc[-1]))
    fin = entradas[melhor]["final"]
    central = float(np.exp(fin["mu"][-1]))
    k = st.columns(4)
    k[0].metric("Melhor modelo", NOMES[melhor], help=f"{desc}. Escolhido pelo critério: {CRITERIOS[criterio]}.")
    empate = abs(linha["vs_base"]) < .0005
    k[1].metric("Erro no teste", f"±{L.pct((np.exp(linha['rmse']) - 1) * 100, 0)}",
                delta="igual ao passeio" if empate else f"{L.pct(linha['vs_base'] * 100, 1, True)} vs passeio", delta_color="off" if empate else "inverse",
                help=f"Erro típico de 1 a 12 meses à frente nos últimos 5 anos. Referência: {base_nome}.")
    k[2].metric("IC 95% acertou", L.pct(linha["cobertura"] * 100, 0),
                help="Fração das previsões em que o valor real ficou dentro do intervalo de 95%. O ideal é perto de 95%.")
    k[3].metric("Em 12 meses", L.num(central, 0), delta=L.pct((central / atual - 1) * 100, 0, True),
                help=f"Valor central do melhor modelo para daqui a 12 meses; variação sobre o último valor ({L.num(atual, 0)}).")

    abas = st.tabs(["Ranking", "Erro por horizonte", "Previsões lado a lado", "Configurações"])

    with abas[0]:
        em_pct = criterio != "crps"
        valores = (np.exp(tab[criterio]) - 1) * 100 if em_pct else tab[criterio]
        visual.mostrar(graficos.barras_modelos(
            valores, f"{CRITERIOS[criterio]}: {nome}", "Menor é melhor. Contorno dourado: o melhor modelo. Cinza e branco: passeios aleatórios de referência.",
            melhor=melhor, formato="±{:.0f}%" if em_pct else "{:.3f}", x_titulo="Erro típico (%)" if em_pct else "CRPS (menor é melhor)",
            ajuda="Cada barra é a média dos erros de previsão no teste dos últimos 5 anos. Os passeios aleatórios mostram o que se consegue sem modelo nenhum."),
            "ranking_modelos", chave="concl_barras")
        ordem = tab[criterio].sort_values().index
        t = pd.DataFrame({
            "Posição": range(1, len(ordem) + 1),
            "Modelo": tab.loc[ordem, "nome"].values,
            "Configuração": [modelos.descrever(f, entradas[f]["config"]) if f in entradas else "referência" for f in ordem],
            "Erro típico (%)": ((np.exp(tab.loc[ordem, "rmse"]) - 1) * 100).values,
            "Erro a 12 meses (%)": ((np.exp(tab.loc[ordem, "rmse_12"]) - 1) * 100).values,
            "CRPS": tab.loc[ordem, "crps"].values,
            "IC 95% acertou (%)": (tab.loc[ordem, "cobertura"] * 100).values,
            "Viés (log)": tab.loc[ordem, "vies"].values,
            "vs referência (%)": (tab.loc[ordem, "vs_base"] * 100).values,
            "p-valor (DM)": tab.loc[ordem, "dm_p"].values,
        }, index=ordem)
        estilo = (t.style.format({"Erro típico (%)": "{:.0f}", "Erro a 12 meses (%)": "{:.0f}", "CRPS": "{:.3f}", "IC 95% acertou (%)": "{:.0f}",
                                  "Viés (log)": "{:+.3f}", "vs referência (%)": "{:+.1f}", "p-valor (DM)": "{:.3f}"}, na_rep="—")
                  .apply(lambda r: ["background-color: rgba(217,164,65,.20)" if r.name == melhor else "" for _ in r], axis=1))
        st.dataframe(estilo, hide_index=True, width="stretch", column_config={
            "Posição": st.column_config.Column(help="Posição pelo critério escolhido, incluindo os passeios aleatórios."),
            "Erro típico (%)": st.column_config.Column(help="Erro médio de 1 a 12 meses à frente (RMSE em log, mostrado em %)."),
            "Erro a 12 meses (%)": st.column_config.Column(help="O mesmo erro nas previsões feitas 12 meses antes."),
            "CRPS": st.column_config.Column(help="Nota da previsão completa (valor e intervalo). Menor é melhor."),
            "IC 95% acertou (%)": st.column_config.Column(help="Fração das previsões em que o valor real caiu dentro do intervalo de 95%. Ideal: perto de 95."),
            "Viés (log)": st.column_config.Column(help="Erro médio com sinal. Positivo: o modelo previu abaixo do que aconteceu (subestimou)."),
            "vs referência (%)": st.column_config.Column(help=f"Erro do modelo em relação ao {base_nome}. Negativo: errou menos."),
            "p-valor (DM)": st.column_config.Column(help=f"Teste de Diebold-Mariano contra o {base_nome}. Abaixo de 0,05: a diferença de erro é estatisticamente significativa. "
                                                         "Com poucas origens o teste tem pouco poder."),
        })
        visual.dica("a tabela mostra também os passeios aleatórios, para ver quantos modelos realmente ficam à frente de não fazer nada. "
                    "Diferenças pequenas entre modelos costumam ser ruído: olhe o p-valor.")

    with abas[1]:
        visual.mostrar(graficos.erro_por_horizonte(
            por_h, melhor, f"Erro típico por horizonte: {nome}", "Linha grossa: o melhor modelo. Pontilhadas: passeios aleatórios.",
            ajuda="Quanto mais meses à frente, maior o erro de todos. Um modelo só é útil se ficar abaixo das linhas pontilhadas."),
            "erro_horizonte_modelos", chave="concl_horizonte")
        visual.dica("passe o mouse para comparar os modelos em um mesmo horizonte. Se as linhas andam juntas, os modelos são equivalentes na prática.")

    with abas[2]:
        datas = modelos.datas_futuras(serie)
        outros = [(NOMES[f], entradas[f]["final"]["mu"], COR_MODELO[f]) for f in FAMILIAS if f != melhor]
        visual.mostrar(graficos.leque(
            serie.y.iloc[-modelos.HIST_DIAS:], datas, fin["mu"], fin["sd"], f"{nome}: previsão para os próximos 12 meses",
            f"Faixas e linha dourada: {NOMES[melhor]} (o melhor). Pontilhadas: os outros modelos.", outros=outros,
            ajuda="Compara as previsões centrais dos cinco modelos. Se estão próximas, a escolha do modelo importa pouco; se divergem, há incerteza sobre o modelo."),
            "previsoes_modelos", chave="concl_leque")
        linhas = []
        for f in FAMILIAS:
            mu, sd = np.asarray(entradas[f]["final"]["mu"]), np.asarray(entradas[f]["final"]["sd"])
            linhas.append({"Modelo": NOMES[f], "Valor central": float(np.exp(mu[-1])), "IC 95% inferior": float(np.exp(mu[-1] - 1.96 * sd[-1])),
                           "IC 95% superior": float(np.exp(mu[-1] + 1.96 * sd[-1])), "Variação vs hoje (%)": float((np.exp(mu[-1]) / atual - 1) * 100)})
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch", column_config={
            "Valor central": st.column_config.NumberColumn(format="%.0f", help="Previsão central para daqui a 12 meses."),
            "IC 95% inferior": st.column_config.NumberColumn(format="%.0f"), "IC 95% superior": st.column_config.NumberColumn(format="%.0f"),
            "Variação vs hoje (%)": st.column_config.NumberColumn(format="%+.0f%%", help=f"Em relação ao último valor ({L.num(atual, 0)}).")})

    with abas[3]:
        linhas = []
        for f in FAMILIAS:
            e = entradas[f]
            linhas.append({"Modelo": NOMES[f], "Configuração": modelos.descrever(f, e["config"]),
                           "Critério de escolha": f"{e['selecao']['criterio']} {e['selecao']['valor']:.1f}", "Candidatas": e["selecao"]["n_candidatos"],
                           "Escolhida com dados até": e["corte"], "Falhas no teste": e["falhas"]})
        st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch", column_config={
            "Critério de escolha": st.column_config.Column(help="AICc (ARIMA, SARIMA, SARIMAX) ou BIC (ARCH, GARCH): nota de ajuste que penaliza complexidade. Menor é melhor."),
            "Candidatas": st.column_config.Column(help="Quantas configurações foram testadas."),
            "Falhas no teste": st.column_config.Column(help="Origens em que o ajuste falhou e foi usado o passeio aleatório no lugar.")})
        st.caption("Cada página de modelo mostra as 8 melhores candidatas e permite testar outros parâmetros.")

    visual.leitura(
        "Leia com cuidado",
        "O teste cobre mês a mês os últimos 5 anos, que incluem a alta do setor com a IA, e prever 12 meses à frente é difícil: os erros são grandes para todos. "
        "O melhor modelo é escolhido no mesmo trecho em que é avaliado, o que favorece um pouco o vencedor. Nos modelos ARCH e GARCH, o valor central vem do "
        "retorno médio histórico, então a vantagem deles está mais no intervalo de confiança do que em acertar a direção. Previsão não é recomendação de investimento.",
        "info")
