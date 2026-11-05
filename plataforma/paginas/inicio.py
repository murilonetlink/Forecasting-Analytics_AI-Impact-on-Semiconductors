"""Página inicial: pergunta do projeto, hipóteses, roteiro e glossário."""
import streamlit as st

from plataforma import dados, graficos, visual
from plataforma.visual import COR, chip, esc, md

HIPOTESES = [
    ("ciclos", "H1", "Ciclicidade", "O setor passa por ciclos de alta e queda que se repetem e podem ser mapeados pelo histórico.", COR["mem"]),
    ("patamar", "H2", "Novo patamar", "Depois de uma valorização forte, o preço não volta ao nível anterior, ele se acomoda mais alto.", COR["mem"]),
    ("eventos", "H3", "Lançamentos de IA", "Nos meses seguintes ao lançamento de grandes modelos, as ações valorizam acima da tendência.", COR["gpu"]),
]

FERRAMENTAS = [
    ("comparar", "Comparar séries", "Ações, cobre, ouro, câmbio e juros no mesmo gráfico, com base 100."),
    ("relacoes", "Quem anda junto", "Ranking de correlação, correlação no tempo, Granger e cointegração."),
    ("risco", "Risco e volatilidade", "Quão agitada está a série e como isso deve evoluir (ARCH/GARCH)."),
    ("previsao", "Previsão", "ARIMA com intervalo de confiança, testado contra o passeio aleatório."),
    ("palco", "Palco (apresentação)", "Os resultados em slides com números e roteiro de fala, direto dos dados."),
    ("dados", "Dados", "Atualizar o Yahoo Finance, enviar suas séries e editar a tabela de eventos."),
]

GLOSSARIO = [
    ("Base 100", "Todas as séries começam em 100 na mesma data; assim se compara variação percentual, não preço."),
    ("Retorno", "Variação percentual de um período para o outro."),
    ("Estacionária", "Série cujas média e variância não mudam com o tempo. Preços quase nunca são; retornos geralmente são."),
    ("Raiz unitária / ADF", "O teste ADF verifica se a série é não estacionária. Se não rejeita, choques têm efeito permanente."),
    ("p-valor", "Probabilidade de ver um resultado tão extremo se a hipótese nula fosse verdadeira. Abaixo de 0,05, rejeita-se a hipótese nula."),
    ("Granger", "X \"causa\" Y se o passado de X ajuda a prever Y. É precedência estatística, não causa de verdade."),
    ("Cointegração", "Duas séries que andam separadas, mas não se afastam indefinidamente uma da outra."),
    ("CAR", "Retorno anormal acumulado: quanto a ação andou além do esperado em torno de um evento."),
    ("Volatilidade", "O tamanho típico das oscilações. Aparece anualizada, em % ao ano."),
    ("ARCH / GARCH", "Modelos em que a volatilidade de hoje depende dos choques e da volatilidade de ontem (a volatilidade vem em ondas)."),
    ("Efeito alavancagem", "Quedas aumentam a volatilidade mais do que altas do mesmo tamanho. GJR e EGARCH medem isso."),
    ("AIC / BIC", "Notas de qualidade do modelo que penalizam complexidade. Quanto menor, melhor."),
    ("ARIMA(p,d,q)", "p = memória dos valores passados, d = diferenciações para estabilizar a série, q = memória dos erros passados."),
    ("Passeio aleatório", "Previsão ingênua que repete o último valor. É o adversário mínimo de qualquer modelo de preço."),
    ("IC 95%", "Faixa em que o valor real deve cair 95% das vezes, se o modelo estiver certo."),
    ("Drawdown", "Distância entre o valor atual e o pico anterior. Mede o tamanho das quedas."),
]


def _link(ctx, chave, rotulo, icone=":material/arrow_forward:"):
    if chave in ctx.paginas:
        st.page_link(ctx.paginas[chave], label=rotulo, icon=icone)


def _visto(chave):
    sim = chave in st.session_state.get("visitadas", set())
    return f'<span class="visto {"sim" if sim else ""}"><span class="led"></span>{"visitada" if sim else "ainda não visitada"}</span>'


def render(ctx):
    md('<div class="kicker">Projeto de pesquisa · FAE Centro Universitário</div>'
       '<div class="titulo-slide">A influência da IA sobre o preço das ações de memória e GPU</div><div class="regua"></div>')
    visual.leitura("Pergunta de pesquisa",
                   "Qual será o comportamento do preço das ações das empresas de memória e de GPU nos próximos doze meses, "
                   "e como os lançamentos de modelos de IA se relacionam com esse comportamento?")

    st.markdown("#### As três hipóteses", help="Cada hipótese tem uma página própria, com gráficos, testes e uma leitura em linguagem simples.")
    for col, (chave, ref, nome, texto, cor) in zip(st.columns(3), HIPOTESES):
        with col:
            md(f'<div class="hip" style="--c:{cor}"><div>{chip(ref, cor)}</div><div class="ht">{esc(nome)}</div><p>{esc(texto)}</p></div>')
            _link(ctx, chave, f"Testar a {ref}")

    st.markdown("&nbsp;")
    c1, c2, c3, c4 = st.columns(4)
    p = ctx.painel
    c1.metric("Séries", p.shape[1], help="Ações, commodities, câmbio, juros, índices do setor e as que você enviar.")
    c2.metric("Primeiro dia", f"{p.index.min():%d/%m/%Y}", help="Data mais antiga com algum dado.")
    c3.metric("Último dia", f"{p.index.max():%d/%m/%Y}", help="Data mais recente com algum dado. Atualize na página Dados.")
    c4.metric("Eventos de IA", len(dados.carregar_eventos()), help="Lançamentos de modelos, chips e fatos de mercado catalogados. Dá para editar em Dados.")

    idx = ctx.tabela(["Índice Memória", "Índice GPU"], "M", completa=True).dropna()
    fig = graficos.linhas(
        idx / idx.iloc[0] * 100, ctx.cores, "Memória × GPU desde 2012",
        "Base 100, escala logarítmica. Os pontos dourados são lançamentos e fatos de IA: passe o mouse para ver quais.",
        base100=True, log=True, eventos=dados.carregar_eventos(), marco=("2022-11-30", "ChatGPT"),
        ajuda="Cada índice é a média das empresas do setor, todas reescaladas para 100 na mesma data. "
              "O valor 1.000 significa que o setor multiplicou por dez desde o início. Escala log: variações percentuais iguais têm o mesmo tamanho.")
    visual.mostrar(fig, "indices_setor")

    st.markdown("#### Ferramentas", help="Páginas de apoio, na barra lateral. Podem ser usadas em qualquer ordem.")
    for linha in (FERRAMENTAS[:3], FERRAMENTAS[3:]):
        for col, (chave, nome, desc) in zip(st.columns(3), linha):
            with col.container(border=True):
                md(f'<div class="mapa-t">{esc(nome)}</div><div class="mapa-s">{esc(desc)}</div>{_visto(chave)}')
                _link(ctx, chave, "Abrir")

    with st.expander("Glossário rápido"):
        st.markdown("\n".join(f"- **{t}:** {d}" for t, d in GLOSSARIO))
