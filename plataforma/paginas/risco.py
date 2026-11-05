"""Risco: volatilidade com ARCH, GARCH, GJR-GARCH e EGARCH."""
import streamlit as st

from plataforma import analises as an
from plataforma import calculos, graficos, leitura, visual


def render(ctx):
    visual.cabecalho("Risco e volatilidade", "Quão agitada está a série e como essa agitação deve evoluir?")
    c1, c2, c3 = st.columns([2, 3, 1])
    nome = c1.selectbox("Série", ctx.colunas, index=0, key="vol_serie", help="Série cujos retornos serão modelados.")
    modelos = c2.multiselect("Modelos", list(an.ESPECIFICACOES), default=["GARCH(1,1)", "GJR-GARCH(1,1)", "EGARCH(1,1)"], key="vol_modelos",
                             help="ARCH(q): a volatilidade depende dos q últimos choques. GARCH(1,1): depende do último choque e da volatilidade anterior "
                                  "(o mais usado). GJR-GARCH e EGARCH: iguais ao GARCH, mas deixam as quedas pesarem mais que as altas.")
    dist = c3.radio("Distribuição", ["t", "normal"], key="vol_dist", format_func=lambda v: "t de Student" if v == "t" else "Normal",
                    help="Formato assumido para os erros. Retornos financeiros têm caudas pesadas (mais dias extremos do que a Normal prevê); a t costuma ajustar melhor.")
    horizonte = st.slider("Horizonte da previsão (períodos)", 1, 60, 20, key="vol_h",
                          help="Quantos períodos à frente prever a volatilidade, na frequência escolhida na barra lateral.")
    if not modelos:
        st.info("Escolha ao menos um modelo.")
        return

    s = ctx.serie(nome)
    if len(s) < 100:
        st.warning("Poucas observações neste período e frequência para ajustar o modelo. Tente a frequência diária ou um período maior.")
        return
    with st.spinner("Ajustando modelos..."):
        tabela, vols, prev, params, arch = calculos.garch(s, tuple(modelos), dist, horizonte, ctx.freq)

    melhor = tabela["AIC"].idxmin()
    atual = float(vols[melhor].iloc[-1])
    pct_atual = an.percentil(vols[melhor], atual)
    visual.leitura(*leitura.garch(nome, tabela, arch, atual, pct_atual))

    g1, g2 = st.columns([1, 2])
    with g1:
        visual.mostrar(graficos.termometro(
            atual, pct_atual, vols[melhor], f"Volatilidade de {nome}",
            ajuda=f"Volatilidade anualizada estimada pelo {melhor} no último período. As faixas dividem a história em calma (abaixo do 1º quartil), "
                  "normal e agitada (acima do 3º quartil)."), "termometro")
    with g2:
        visual.mostrar(graficos.volatilidade(
            vols, an.PERIODOS_ANO[ctx.freq], "Volatilidade condicional", "Quanto maior, mais agitado. Os picos mostram as crises e os vales, as calmarias.",
            ajuda="Estimativa do tamanho das oscilações em cada data, em % ao ano. A volatilidade vem em ondas: períodos agitados se agrupam."),
            "volatilidade")
    visual.dica("o termômetro compara a agitação de hoje com toda a história da série. Perto do vermelho, as oscilações atuais estão entre as maiores já vistas.")

    visual.mostrar(graficos.previsao_volatilidade(
        prev, f"Previsão da volatilidade, {horizonte} períodos à frente", "Tende a voltar ao nível médio de longo prazo.",
        ajuda="A volatilidade prevista não segue para sempre o nível de hoje: nos modelos GARCH ela volta aos poucos para a média histórica."),
        "previsao_vol")

    with st.expander("Comparar os modelos"):
        c_a, c_b = st.columns(2)
        c_a.metric("Teste ARCH-LM (p-valor)", f"{arch['p-valor']:.2e}",
                   help="H0: variância constante. p baixo indica agrupamento de volatilidade, o que justifica usar ARCH/GARCH.")
        c_b.metric("Melhor modelo pelo AIC", melhor,
                   help="Modelo com menor AIC entre os escolhidos. O AIC mede o ajuste penalizando complexidade; diferenças pequenas não são decisivas.")
        st.dataframe(tabela.style.format("{:.2f}").format({"Ljung-Box p (resíduos²)": "{:.3f}"}), width="stretch", column_config={
            "log-verossimilhança": st.column_config.Column(help="Quão bem o modelo explica os dados. Maior é melhor."),
            "AIC": st.column_config.Column(help="Critério de Akaike. Menor é melhor."),
            "BIC": st.column_config.Column(help="Critério de Bayes. Como o AIC, mas pune mais a complexidade. Menor é melhor."),
            "Ljung-Box p (resíduos²)": st.column_config.Column(help="Testa se sobrou padrão na volatilidade depois do modelo. p acima de 0,05 é bom."),
        })
        st.caption("omega: nível base da variância. alpha: peso do choque mais recente. beta: persistência da volatilidade anterior. "
                   "gamma: efeito extra de choques negativos (GJR/EGARCH). nu: graus de liberdade da t (menor = caudas mais pesadas). mu: retorno médio.")
        for n, p in params.items():
            st.markdown(f"**{n}**")
            st.dataframe(p.style.format({"coef": "{:.4f}", "p-valor": "{:.4f}"}), width="stretch")
