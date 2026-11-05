"""H3 · Lançamentos de IA: estudo de eventos."""
import numpy as np
import pandas as pd
import streamlit as st

from plataforma import analises as an
from plataforma import dados, graficos, leitura, visual
from plataforma.visual import COR


def render(ctx):
    visual.cabecalho("Lançamentos de IA", "As ações se movem além do esperado quando sai um modelo de IA ou um chip novo?",
                     "H3 · LANÇAMENTOS DE IA", COR["gpu"])
    eventos = dados.carregar_eventos()
    colunas = ctx.colunas

    c1, c2 = st.columns(2)
    alvo = c1.selectbox("Série analisada", colunas, index=colunas.index("Índice Memória"), key="ev_alvo",
                        help="A série cuja reação aos eventos será medida.")
    opcoes = ["Nenhum (usar média do período)"] + [c for c in colunas if c != alvo]
    padrao = opcoes.index("S&P 500 (^GSPC)") if "S&P 500 (^GSPC)" in opcoes else 0
    bench = c2.selectbox("Referência de mercado", opcoes, index=padrao, key="ev_bench",
                         help="Usada para descontar o movimento geral do mercado: se a bolsa inteira subiu no dia, isso não conta como efeito do evento. "
                              "Com 'Nenhum', o esperado é só o retorno médio dos 250 pregões anteriores.")

    c3, c4, c5 = st.columns(3)
    tipos = sorted(eventos["categoria"].unique())
    cats = c3.multiselect("Tipos de evento", tipos, default=tipos, key="ev_cats",
                          help="Modelo: lançamento de modelo de IA. Hardware: anúncio de chip. Mercado: fato que mexeu com as ações (ex.: resultado da NVIDIA).")
    antes = c4.slider("Pregões antes", 0, 20, 5, key="ev_antes",
                      help="Dias úteis antes do evento que entram na análise. Serve para ver se o mercado se antecipou.")
    depois = c5.slider("Pregões depois", 1, 63, 10, key="ev_depois",
                       help="Dias úteis depois do evento que entram na análise. 21 pregões ≈ 1 mês; 63 ≈ 3 meses. "
                            "Janelas longas captam mais efeito, mas também mais ruído de outras notícias.")

    ev = eventos[eventos["categoria"].isin(cats) & eventos["data"].between(pd.Timestamp(ctx.ini), pd.Timestamp(ctx.fim))]
    if ev.empty:
        st.info("Nenhum evento no período e nos tipos escolhidos.")
        return

    s = ctx.serie(alvo, "D", completa=True)
    mercado = None if bench.startswith("Nenhum") else ctx.serie(bench, "D", completa=True)
    traj, meta, ignorados, resumo = an.estudo_eventos(s, mercado, ev, antes, depois)
    if resumo is None:
        st.warning("Eventos insuficientes com histórico para estimar o modelo (mínimo 2).")
        return

    visual.leitura(*leitura.eventos(resumo, depois, alvo))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Eventos usados", resumo["n"], help="Eventos com histórico suficiente (cerca de 280 pregões antes) para estimar o comportamento normal da série.")
    m2.metric("CAR médio", f"{resumo['car_final_medio'] * 100:+.2f}%".replace(".", ","),
              help=f"Média, entre os eventos, do retorno além do esperado acumulado até {depois} pregões depois. Positivo: a série tende a subir mais que o esperado.")
    m3.metric("Acima do esperado", f"{resumo['pct_positivos'] * 100:.0f}%", help="Fração dos eventos com CAR > 0, ou seja, em que a série ficou acima do esperado. Perto de 50% quer dizer sem direção clara.")
    m4.metric("p-valor", "n/d" if np.isnan(resumo["p"]) else f"{resumo['p']:.3f}".replace(".", ","),
              help="Testa se o CAR médio é diferente de zero. Abaixo de 0,05: o efeito médio é estatisticamente significativo. Acima: pode ser só acaso.")

    abas = st.tabs(["Efeito de cada evento", "Trajetória média", "Série com eventos"])
    with abas[0]:
        ordem = st.radio("Mostrar", ["data", "impacto"], horizontal=True, key="ev_ordem",
                         format_func={"data": "Linha do tempo", "impacto": "Ranking por impacto"}.get,
                         help="Linha do tempo: quando cada evento aconteceu. Ranking: do pior ao melhor resultado.")
        visual.mostrar(graficos.barras_eventos(
            traj, meta, depois, ordem, "Quanto a série andou além do esperado após cada evento",
            f"CAR em {depois} pregões. Barra para cima: subiu mais que o esperado. Cores: tipo de evento.",
            ajuda="Para cada evento, comparamos o retorno real com o que o mercado em geral faria. A barra é a diferença acumulada depois de "
                  f"{depois} pregões. Passe o mouse para ver o nome do evento."), "efeito_eventos")
        visual.dica("barras grandes pesam mais, mas um evento isolado pode ser coincidência com outras notícias. O que importa é o padrão geral, resumido no p-valor.")
    with abas[1]:
        todos = st.checkbox("Mostrar a trajetória de cada evento", False, key="ev_todos",
                            help="Desenha uma linha fraca para cada evento, para ver se a média esconde casos muito diferentes.")
        visual.mostrar(graficos.trajetoria_car(
            traj, resumo, depois, todos, "Retorno anormal acumulado médio", "A faixa é o intervalo de confiança de 95% da média.",
            ajuda="Dia 0 é o dia do evento. A linha mostra, em média, quanto a série andou além do esperado a cada dia. "
                  "Se a faixa inclui o zero, não dá para distinguir o efeito de acaso."), "car_medio")
        visual.dica("se a linha sobe depois do dia 0 e a faixa fica acima de zero, há efeito positivo. Se a faixa atravessa o zero, o resultado é inconclusivo.")
    with abas[2]:
        s_plot = ctx.serie(alvo, "D")
        visual.mostrar(graficos.linhas(
            s_plot.to_frame(alvo), ctx.cores, f"{alvo} com eventos de IA", "Pontos dourados: eventos catalogados. Passe o mouse para ver quais.",
            log=bool((s_plot > 0).all()), eventos=ev, ajuda="Série no período escolhido, com os eventos marcados sobre ela."), "serie_eventos")

    if ignorados:
        st.caption(f"{len(ignorados)} evento(s) sem histórico suficiente para a janela de estimação: " + "; ".join(ignorados))
    st.caption("Eventos próximos entre si compartilham janelas, então não são independentes. Para Samsung, SK Hynix e Nanya, "
               "o pregão reage ao evento só no dia seguinte por causa do fuso horário.")
