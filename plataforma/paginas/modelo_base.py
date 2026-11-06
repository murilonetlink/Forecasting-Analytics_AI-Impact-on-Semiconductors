"""Estrutura comum das páginas de modelos: melhor configuração pronta, teste manual, previsão, backtest e diagnóstico."""
import re

import numpy as np
import pandas as pd
import streamlit as st

from plataforma import calculos, graficos, modelos, visual
from plataforma import leitura as L
from plataforma.modelos import NOMES
from plataforma.visual import COR, chip, esc, md

AJUDA_PROTOCOLO = (
    "A configuração é escolhida usando só os dados anteriores ao teste. Depois, a cada mês dos últimos 5 anos, o modelo é reajustado só com o que "
    "se sabia até aquele dia (base diária) e prevê os 252 pregões seguintes (12 meses); comparamos com o que de fato aconteceu. As métricas são médias dessas previsões.")

ESPEC = {
    "arima": dict(
        titulo="ARIMA", selo="MODELO · ARIMA", cor=COR["mem"], pergunta="O passado da própria série ajuda a prever o futuro?",
        resumo="usa só o histórico diário do índice: quanto os retornos e os erros passados pesam na previsão do dia seguinte.",
        explica=("**ARIMA(p, d, q)** olha apenas para a série, dia a dia.\n\n"
                 "- **d**: quantas vezes a série é diferenciada (trocamos o nível pelo retorno) para ficar estável. Preços costumam pedir d = 1.\n"
                 "- **p**: quantos retornos passados entram na previsão (parte AR).\n"
                 "- **q**: quantos erros passados entram na previsão (parte MA).\n"
                 "- **Deriva**: crescimento médio constante por dia. Sem deriva, a previsão central fica parada no último valor.\n\n"
                 "O ajuste é feito no logaritmo do índice diário e a previsão vai até 252 dias úteis à frente. Se o melhor modelo for ARIMA(0,1,0) com deriva, "
                 "ele coincide com o passeio aleatório com deriva: o passado dos retornos diários não ajuda a prever o próximo.")),
    "sarima": dict(
        titulo="SARIMA", selo="MODELO · SARIMA", cor="#A58BE0", pergunta="Padrões que se repetem toda semana melhoram a previsão?",
        resumo="é o ARIMA com uma parte sazonal, que olha para o mesmo dia da semana nas semanas anteriores.",
        explica=("**SARIMA(p, d, q)(P, D, Q, 5)** é o ARIMA mais uma parte sazonal. Na base diária, a sazonalidade é a semana de 5 dias úteis.\n\n"
                 "- **P, D, Q**: fazem o mesmo papel de p, d e q, mas olhando 5 dias para trás (o mesmo dia da semana anterior).\n"
                 "- **D** só vira 1 quando a sazonalidade é forte (força acima de 0,64). A força medida aparece na configuração abaixo.\n\n"
                 "Se não existe sazonalidade de verdade, o SARIMA não deve ganhar do ARIMA.")),
    "sarimax": dict(
        titulo="SARIMAX", selo="MODELO · SARIMAX", cor=COR["gold"], pergunta="Lançamentos de IA ajudam a prever o retorno das ações?",
        resumo="é o SARIMA com variáveis externas (o X): aqui, os lançamentos de IA da tabela de eventos.",
        explica=("**SARIMAX** é o SARIMA com variáveis externas. Aqui as variáveis são os lançamentos de IA da tabela de eventos "
                 "(tipo Modelo, por padrão), marcados no dia do lançamento e nos dias seguintes.\n\n"
                 "- O **coeficiente** de cada variável diz quanto o retorno do dia muda, em média, por lançamento (0,01 = 1 ponto percentual).\n"
                 "- **Lançamentos futuros são desconhecidos.** Para prever, usamos como cenário a taxa média histórica de lançamentos por mês, "
                 "que dá para ajustar na configuração manual.\n"
                 "- Este é o modelo que conversa com a hipótese H3 do projeto.")),
    "arch": dict(
        titulo="ARCH", selo="MODELO · ARCH", cor=COR["ok"], pergunta="A agitação de ontem ajuda a prever a agitação de amanhã?",
        resumo="modela a volatilidade (o tamanho das oscilações), não a direção do preço.",
        explica=("**ARCH(q)** modela a volatilidade: a variância de hoje depende dos q últimos choques ao quadrado. É ajustado nos retornos diários.\n\n"
                 "- Para entrar na comparação com os outros modelos, o **valor central** da previsão vem do retorno médio diário (deriva) e o "
                 "**intervalo** vem da variância prevista, somada ao longo dos 252 dias úteis.\n"
                 "- Por isso o que o ARCH acrescenta é principalmente a qualidade do intervalo de confiança.\n"
                 "- **Distribuição t de Student** dá mais peso a dias extremos, que são comuns em ações.")),
    "garch": dict(
        titulo="GARCH", selo="MODELO · GARCH", cor=COR["gpu"], pergunta="A volatilidade tem memória?",
        resumo="é o ARCH com memória da própria volatilidade: poucas defasagens captam as ondas de agitação.",
        explica=("**GARCH(p, q)** acrescenta ao ARCH a própria volatilidade passada.\n\n"
                 "- **p**: quantas defasagens da variância entram (a volatilidade tem memória).\n"
                 "- **q**: quantas defasagens dos choques ao quadrado entram.\n"
                 "- GARCH(1,1) costuma bastar e capta bem o fato de que dias agitados vêm em sequência.\n"
                 "- Como no ARCH, o valor central vem do retorno médio diário e o intervalo, da variância prevista.")),
}


def _rotulo_param(n):
    fixos = {"drift": "deriva (crescimento médio diário, em log)", "sigma2": "variância do erro", "mu": "retorno médio diário (%)",
             "omega": "ômega (nível base da variância)", "nu": "nu (graus de liberdade da t)"}
    if n in fixos:
        return fixos[n]
    if m := re.fullmatch(r"(ar|ma)\.L(\d+)", n):
        return f"{m[1].upper()}({m[2]})"
    if m := re.fullmatch(r"(ar|ma)\.S\.L(\d+)", n):
        return f"{m[1].upper()} sazonal (defasagem {m[2]})"
    if m := re.fullmatch(r"lanc_t-(\d+)", n):
        return "lançamento de IA no dia" if m[1] == "0" else f"lançamento de IA há {m[1]} dia(s)"
    if m := re.fullmatch(r"alpha\[(\d+)\]", n):
        return f"alfa {m[1]} (peso do choque)"
    if m := re.fullmatch(r"beta\[(\d+)\]", n):
        return f"beta {m[1]} (peso da volatilidade passada)"
    return n


def _controles(familia, melhor, serie, chave):
    """Widgets de configuração manual, já preenchidos com a melhor versão. Devolve (config, taxa de lançamentos)."""
    taxa = None
    with st.container(border=True):
        if familia in ("arima", "sarima", "sarimax"):
            o, s = melhor["order"], melhor["seasonal"]
            sazonal = familia != "arima"
            cols = st.columns(7 if sazonal else 4)
            p = cols[0].number_input("p", 0, 6, o[0], key=f"{chave}_p", help="Parte AR: quantos retornos passados entram na previsão.")
            d = cols[1].number_input("d", 0, 2, o[1], key=f"{chave}_d", help="Quantas vezes diferenciar a série. Preços em log costumam pedir d = 1.")
            q = cols[2].number_input("q", 0, 6, o[2], key=f"{chave}_q", help="Parte MA: quantos erros passados entram na previsão.")
            P = D = Q = 0
            if sazonal:
                P = cols[3].number_input("P", 0, 2, s[0], key=f"{chave}_P", help="AR sazonal: olha o mesmo dia da semana nas semanas anteriores.")
                D = cols[4].number_input("D", 0, 1, s[1], key=f"{chave}_D", help="Diferenciação sazonal. Use 1 só se houver sazonalidade forte.")
                Q = cols[5].number_input("Q", 0, 2, s[2], key=f"{chave}_Q", help="MA sazonal: erros do mesmo dia da semana nas semanas anteriores.")
            tr = cols[-1].selectbox("Deriva", ["n", "t"], index=["n", "t"].index(melhor["trend"]), key=f"{chave}_tr",
                                    format_func={"n": "Sem deriva", "t": "Com deriva"}.get,
                                    help="Deriva é um crescimento médio constante por dia. Só vale com uma diferenciação no total (d + D = 1).")
            if d + D != 1:
                tr = "n"
            cfg = {"order": [int(p), int(d), int(q)], "seasonal": [int(P), int(D), int(Q), modelos.SAZONAL if sazonal else 0], "trend": tr}
            if familia == "sarimax":
                c1, c2, c3 = st.columns(3)
                tipos = c1.multiselect("Tipos de evento", ["Modelo", "Hardware", "Mercado"], default=melhor["tipos"], key=f"{chave}_tipos",
                                       help="Quais eventos da tabela contam como lançamentos. Modelo: lançamento de IA. Hardware: chips. Mercado: fatos de mercado.")
                atrasos = c2.multiselect("Efeito em quais dias", list(modelos.ATRASOS), default=melhor["atrasos"], key=f"{chave}_atr",
                                         format_func=lambda k: "no dia do lançamento" if k == 0 else f"{k} dia(s) depois",
                                         help="O lançamento pode afetar o retorno no próprio dia ou nos dias seguintes. Escolha uma ou mais defasagens.")
                if not tipos or not atrasos:
                    st.warning("Escolha ao menos um tipo de evento e uma defasagem; usando a melhor versão.")
                    tipos, atrasos = melhor["tipos"], melhor["atrasos"]
                base = round(modelos.exog_matrizes(serie, tipos, sorted(atrasos), serie.n - 1)[2] * modelos.DIAS_MES, 3)
                t = c3.number_input("Lançamentos esperados por mês (cenário)", 0.0, 5.0, base, 0.05, key=f"{chave}_taxa",
                                    help="Lançamentos futuros são desconhecidos. O padrão é a média histórica mensal de lançamentos do tipo escolhido.")
                cfg.update(tipos=list(tipos), atrasos=sorted(atrasos))
                taxa = None if abs(t - base) < 1e-9 else float(t) / modelos.DIAS_MES
            return cfg, taxa
        c = st.columns(3)
        dist = c[-1].selectbox("Distribuição", list(modelos.DISTRIBUICOES), index=list(modelos.DISTRIBUICOES).index(melhor["dist"]), key=f"{chave}_dist",
                               format_func=modelos.DISTRIBUICOES.get, help="A t de Student dá mais peso a dias extremos, comuns em ações.")
        if familia == "arch":
            q = c[0].number_input("q (choques)", 1, 12, melhor["choques"], key=f"{chave}_qc", help="Quantos choques passados ao quadrado entram na variância.")
            return {"choques": int(q), "dist": dist}, None
        p = c[0].number_input("p (volatilidade passada)", 1, 3, melhor["vol"], key=f"{chave}_pv", help="Quantas defasagens da variância entram.")
        q = c[1].number_input("q (choques)", 1, 3, melhor["choques"], key=f"{chave}_qc", help="Quantos choques passados ao quadrado entram.")
        return {"vol": int(p), "choques": int(q), "dist": dist}, None


def render(ctx, familia):
    esp = ESPEC[familia]
    visual.cabecalho(esp["titulo"], esp["pergunta"], esp["selo"], esp["cor"])
    visual.dica(esp["resumo"], "O que é")
    with st.expander("Como este modelo funciona"):
        st.markdown(esp["explica"])

    nome = st.radio("Índice", modelos.SERIES, horizontal=True, key=f"{familia}_serie",
                    help="Cada modelo é escolhido e testado separadamente para o Índice Memória e para o Índice GPU.")
    versao = calculos.versao_modelos()
    serie = calculos.serie_modelo(nome, versao)
    entrada = calculos.entrada_modelo(nome, familia, versao)
    melhor, sel = entrada["config"], entrada["selecao"]
    chave = f"{familia}_{nome}"
    vol = familia in ("arch", "garch")

    st.markdown("#### Configuração", help="O modelo já abre na melhor versão encontrada. Dá para testar outros parâmetros e ver o efeito.")
    c1, c2 = st.columns([3, 2])
    with c1:
        md(f'{chip("MELHOR VERSÃO", COR["gold"])}<span style="font-size:19px;font-weight:600">{esc(modelos.descrever(familia, melhor))}</span>')
        dif = sel["diferenciacao"]
        extra = "" if vol else f" Diferenciação: d = {dif['d']}, D = {dif['D']} (força sazonal {L.num(dif['forca_sazonal'], 2)})."
        st.caption(f"Escolhida pelo menor {sel['criterio']} ({L.num(sel['valor'], 1)}) entre {sel['n_candidatos']} candidatas, "
                   f"usando só dados até {pd.Timestamp(entrada['corte']):%m/%Y}.{extra}")
    manual = c2.toggle("Testar outra configuração", key=f"{chave}_manual",
                       help="Mostra os parâmetros para você mudar. A previsão e o diagnóstico atualizam na hora; o teste nos últimos 5 anos é opcional porque é mais lento.")
    cfg, taxa = _controles(familia, melhor, serie, chave) if manual else (melhor, None)
    eh_melhor = cfg == melhor and taxa is None
    cj = calculos.cfg_json(cfg)
    desc = modelos.descrever(familia, cfg)

    tab, por_h_ref, melhor_base, real, previsoes = calculos.comparacao(nome, versao)
    pos = modelos.posicoes_origem(serie)
    bt = None
    if eh_melhor:
        prev, bt = entrada["final"], entrada
    else:
        try:
            prev = calculos.previsao_manual(nome, familia, cj, versao, taxa)
        except Exception as e:
            st.error(f"Não foi possível ajustar {desc}: {e}")
            return
        if st.toggle("Avaliar esta configuração no teste (mais lento)", key=f"{chave}_avaliar",
                     help="Refaz o teste dos últimos 5 anos com estes parâmetros, para comparar com a melhor versão e com o passeio aleatório."):
            bt = calculos.backtest_manual(nome, familia, cj, versao, taxa)

    fmu, fsd = np.asarray(prev["mu"]), np.asarray(prev["sd"])
    atual = float(np.exp(serie.y.iloc[-1]))
    final = (float(np.exp(fmu[-1])), float(np.exp(fmu[-1] - 1.96 * fsd[-1])), float(np.exp(fmu[-1] + 1.96 * fsd[-1])))
    geral = por_h = vs_base = dm_p = None
    if bt is not None:
        mu_bt, sd_bt = np.asarray(bt["mu"]), np.asarray(bt["sd"])
        geral, por_h = modelos.metricas(real, mu_bt, sd_bt)
        vs_base = geral["rmse"] / tab.loc[melhor_base, "rmse"] - 1
        dm_p = modelos.diebold_mariano(real, mu_bt, previsoes[melhor_base][0])[1]
        equivale = familia == "arima" and cfg["order"][0] == 0 and cfg["order"][2] == 0 and cfg["order"][1] == 1 and cfg["trend"] == "t"
        visual.leitura(*L.modelo(desc, geral, vs_base, NOMES[melhor_base].lower(), dm_p, final, atual, len(pos),
                                 serie.y.index[pos[0]], serie.y.index[pos[-1]], equivale))
    else:
        st.info("Ative **Avaliar esta configuração no teste** para ver o erro nos últimos 5 anos e comparar com o passeio aleatório.")

    k = st.columns(4)
    k[0].metric("Em 12 meses", L.num(final[0], 0), delta=L.pct((final[0] / atual - 1) * 100, 0, True),
                help=f"Valor central previsto para o {nome} daqui a 12 meses, e a variação em relação ao último valor ({L.num(atual, 0)}).")
    if geral:
        empate = abs(vs_base) < .0005
        k[1].metric("Erro no teste", f"±{L.pct((np.exp(geral['rmse']) - 1) * 100, 0)}",
                    delta="igual ao passeio" if empate else f"{L.pct(vs_base * 100, 1, True)} vs passeio", delta_color="off" if empate else "inverse",
                    help=f"Erro típico ao prever de 1 a 12 meses à frente nos últimos 5 anos (RMSE em log, mostrado em %). Referência: {NOMES[melhor_base].lower()}. {AJUDA_PROTOCOLO}")
        k[2].metric("Erro a 12 meses", f"±{L.pct((np.exp(geral['rmse_12']) - 1) * 100, 0)}", help="O mesmo erro, olhando só as previsões feitas 12 meses antes.")
        k[3].metric("IC 95% acertou", L.pct(geral["cobertura"] * 100, 0),
                    help="Em que fração das previsões o valor real caiu dentro do intervalo de 95%. O ideal é perto de 95%; bem abaixo significa intervalo estreito demais (excesso de confiança).")
    else:
        for c in k[1:]:
            c.metric("Erro no teste", "—", help="Ative a avaliação do teste para ver.")

    abas = st.tabs(["Previsão para 12 meses", "Teste nos últimos 5 anos", "Diagnóstico"] + (["Volatilidade"] if vol else []))

    with abas[0]:
        datas = modelos.datas_futuras(serie)
        visual.mostrar(graficos.leque(
            serie.y.iloc[-modelos.HIST_DIAS:], datas, fmu, fsd, f"{nome}: previsão para os próximos 12 meses", f"{desc}. Faixas: intervalos de confiança de 80% e 95%.",
            ajuda="A linha dourada é a previsão central; as faixas mostram onde o valor deve cair 80% e 95% das vezes se o modelo estiver certo. "
                  "A incerteza cresce com o horizonte."), f"previsao_{familia}", chave=f"{chave}_leque")
        visual.dica("passe o mouse sobre a linha dourada para ver o valor central, o intervalo de 95% e a variação em relação a hoje. "
                    "Faixas largas são normais: preços de ações são difíceis de prever a 12 meses.")
        with st.expander("Valores mês a mês"):
            m21 = np.arange(modelos.DIAS_MES - 1, modelos.HORIZONTE, modelos.DIAS_MES)
            t = pd.DataFrame({"data": datas[m21], "valor central": np.exp(fmu[m21]), "IC 95% inferior": np.exp(fmu[m21] - 1.96 * fsd[m21]),
                              "IC 95% superior": np.exp(fmu[m21] + 1.96 * fsd[m21]), "variação vs hoje (%)": (np.exp(fmu[m21]) / atual - 1) * 100})
            st.dataframe(t, hide_index=True, width="stretch", column_config={
                "data": st.column_config.DateColumn(format="DD/MM/YYYY"), "valor central": st.column_config.NumberColumn(format="%.0f"),
                "IC 95% inferior": st.column_config.NumberColumn(format="%.0f"), "IC 95% superior": st.column_config.NumberColumn(format="%.0f"),
                "variação vs hoje (%)": st.column_config.NumberColumn(format="%+.0f%%")})

    with abas[1]:
        if bt is None:
            st.info("Ative a avaliação do teste acima para ver estes gráficos com a configuração que você escolheu.")
        else:
            visual.mostrar(graficos.previsto_vs_realizado(
                serie.y, pos, mu_bt, sd_bt, "Previsto 12 meses antes × realizado", "Cada ponto dourado é uma previsão feita um ano antes da data em que aparece.",
                ajuda="Para cada mês, mostra o que o modelo previa 12 meses antes (dourado) e o que de fato aconteceu (branco). A faixa é o intervalo de 95%."),
                f"teste_{familia}", chave=f"{chave}_pvr")
            datas_o = [serie.y.index[p] for p in pos]
            i = st.select_slider("Data da previsão", options=list(range(len(pos))), value=len(pos) - 1, key=f"{chave}_origem",
                                 format_func=lambda j: f"{datas_o[j]:%d/%m/%Y}",
                                 help="Escolha uma data do teste (uma por mês): o gráfico mostra o que o modelo previa naquele dia para os 12 meses seguintes e o que aconteceu.")
            visual.mostrar(graficos.origem_explorer(
                serie.y, pos[i], mu_bt[i], sd_bt[i], f"Previsão feita em {datas_o[i]:%d/%m/%Y}", "A linha branca tracejada é o que aconteceu depois.",
                ajuda="Um caso concreto do teste: o histórico até a data escolhida, a previsão para os 12 meses seguintes e o realizado."),
                f"origem_{familia}", chave=f"{chave}_origem_fig")
            visual.mostrar(graficos.erro_por_horizonte(
                {familia: por_h, "rw": por_h_ref["rw"], "rw_deriva": por_h_ref["rw_deriva"]}, familia, "Erro típico por horizonte",
                "Quanto mais longe, maior o erro. Linhas pontilhadas: passeios aleatórios de referência.",
                ajuda="O erro típico (RMSE em log, mostrado em %) para cada número de meses à frente. O modelo é útil se ficar abaixo da referência."),
                f"erro_horizonte_{familia}", chave=f"{chave}_horizonte")
            visual.dica("o ideal é que a linha do modelo fique abaixo das pontilhadas. Se fica junto delas, ele não acrescenta nada além de repetir o último valor "
                        "(com ou sem crescimento médio).")

    with abas[2]:
        try:
            d = calculos.diagnostico_modelo(nome, familia, cj, versao, taxa)
        except Exception as e:
            st.error(f"Não foi possível calcular o diagnóstico: {e}")
            d = None
        if d is not None:
            tabp = d["params"].copy()
            tabp.index = [_rotulo_param(str(n)) for n in tabp.index]
            if vol:
                visual.leitura(*L.diagnostico_vol(desc, d))
                m = st.columns(3)
                m[0].metric("ARCH-LM (p-valor)", f"{d['archlm_p']:.3f}".replace(".", ","), help="H0: não sobrou agrupamento de volatilidade nos resíduos. Acima de 0,05 é bom.")
                m[1].metric("Ljung-Box dos quadrados", f"{d['lb2_p']:.3f}".replace(".", ","), help="H0: sem autocorrelação nos resíduos ao quadrado. Acima de 0,05 é bom.")
                m[2].metric("BIC", L.num(d["bic"], 1), help="Nota de qualidade que penaliza complexidade. Só compara modelos na mesma série. Menor é melhor.")
            else:
                visual.leitura(*L.diagnostico_arima(desc, d))
                if familia == "sarimax" and (l := L.lancamentos(d["params"])):
                    visual.leitura(*l)
                m = st.columns(3)
                m[0].metric("Ljung-Box (p-valor)", f"{d['lb_p']:.3f}".replace(".", ","), help="H0: resíduos sem autocorrelação. Acima de 0,05 é bom: o modelo captou o padrão.")
                m[1].metric("AICc", L.num(d["aicc"], 1), help="Nota de qualidade que penaliza complexidade, com correção para amostras pequenas. Menor é melhor.")
                m[2].metric("BIC", L.num(d["bic"], 1), help="Como o AICc, mas pune mais a complexidade. Menor é melhor.")
                visual.mostrar(graficos.acf_residuos(
                    d["acf"], len(d["resid"]), "Autocorrelação dos resíduos", "Barras dentro da faixa cinza: sem autocorrelação.",
                    ajuda="Se o modelo captou todo o padrão, os resíduos parecem ruído: nenhuma barra fora da faixa. Barras laranja saíram da faixa."),
                    f"acf_{familia}", chave=f"{chave}_acf")
            st.markdown("**Parâmetros estimados**")
            st.dataframe(tabp, width="stretch", column_config={
                "coef": st.column_config.NumberColumn("coeficiente", format="%.4f", help="Valor estimado do parâmetro."),
                "erro padrão": st.column_config.NumberColumn(format="%.4f", help="Incerteza da estimativa."),
                "p-valor": st.column_config.NumberColumn(format="%.3f", help="Abaixo de 0,05: o parâmetro é estatisticamente diferente de zero.")})
            if familia == "sarimax":
                st.caption("Coeficientes de lançamento: 0,01 equivale a 1 ponto percentual no retorno do mês, por lançamento.")

    if vol and d is not None:
        with abas[3]:
            visual.mostrar(graficos.volatilidade(
                pd.DataFrame({desc: d["vol_anual"]}), 252, "Volatilidade condicional diária", "Quanto maior, mais agitado. Picos marcam crises.",
                ajuda="Estimativa do tamanho das oscilações em cada dia, em % ao ano."), f"vol_{familia}", chave=f"{chave}_vol")
            visual.mostrar(graficos.previsao_volatilidade(
                pd.DataFrame({desc: d["vol_prevista"]}, index=range(1, 13)), "Volatilidade prevista para os próximos 12 meses",
                "Média da volatilidade diária prevista em cada mês, em % ao ano.",
                ajuda="Tende a voltar para o nível médio de longo prazo."), f"vol_prev_{familia}", chave=f"{chave}_volprev")
            visual.dica("o modelo usa essa volatilidade prevista para abrir o intervalo de confiança da previsão do índice.")

    with st.expander("Como a melhor configuração foi escolhida"):
        st.markdown(f"Todas as candidatas foram ajustadas só com dados até **{pd.Timestamp(entrada['corte']):%m/%Y}**, antes do trecho de teste, "
                    f"e comparadas pelo **{sel['criterio']}** (menor é melhor). As 8 melhores:")
        top = pd.DataFrame({"configuração": [modelos.descrever(familia, t["config"]) for t in sel["top"]], sel["criterio"]: [t["valor"] for t in sel["top"]]})
        st.dataframe(top, hide_index=True, width="stretch", column_config={sel["criterio"]: st.column_config.NumberColumn(format="%.1f")})
        st.caption(AJUDA_PROTOCOLO)
