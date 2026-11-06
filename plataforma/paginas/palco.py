"""Palco: os resultados em slides calculados dos dados atuais, com números grandes e roteiro de fala."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from plataforma import analises as an
from plataforma import calculos, dados, graficos, modelos, visual
from plataforma import leitura as L
from plataforma.modelos import NOMES
from plataforma.visual import COR, COR_MODELO, chip, esc, md, numero

MEM, GPU, SP = "Índice Memória", "Índice GPU", "S&P 500 (^GSPC)"
MARCO = date(2022, 11, 30)
FONTE = 15

TECLAS = """
(function () {
  if (window.__palcoTeclas) return;
  window.__palcoTeclas = true;
  document.addEventListener('keydown', function (e) {
    if (e.target.closest && e.target.closest('input, textarea, select, [role="slider"], [contenteditable="true"]')) return;
    var alvo = e.key === 'ArrowRight' ? '.st-key-palco_prox button' : e.key === 'ArrowLeft' ? '.st-key-palco_ant button' : null;
    var b = alvo && document.querySelector(alvo);
    if (b && !b.disabled) b.click();
  });
})();
"""


# --------------------------------------------------------------------- utilitários
def _cabeca(kicker, titulo):
    md(f'<div class="kicker">{esc(kicker)}</div><div class="titulo-slide">{esc(titulo)}</div><div class="regua"></div>')


def _frase(texto_html):
    md(f'<div class="frase">{texto_html}</div>')


def _roteiro(linhas):
    with st.expander("Roteiro de fala (para quem apresenta)"):
        st.markdown("\n".join(f"- {t}" for t in linhas))


def _cartoes(itens):
    for col, (valor, rotulo, cor) in zip(st.columns(len(itens)), itens):
        col.markdown(numero(valor, rotulo, cor), unsafe_allow_html=True)


def _mensal(ctx):
    return ctx.tabela([MEM, GPU], "M", completa=True).dropna()


def _escolher_indice(chave):
    return st.radio("Índice", [MEM, GPU], horizontal=True, key=chave, label_visibility="collapsed")


# ------------------------------------------------------------------------- slides
def s_capa(ctx):
    md('<div class="kicker">Projeto de pesquisa · FAE Centro Universitário · Ciência de Dados para Negócios</div>'
       '<div class="titulo-slide" style="font-size:clamp(32px,4.4vw,58px);max-width:20ch">A influência da IA sobre o preço das ações de memória e GPU</div>'
       '<div class="regua"></div>')
    _frase("Qual será o comportamento do preço das ações de memória e de GPU nos próximos doze meses, "
           "e como os lançamentos de modelos de IA se relacionam com ele?")
    hips = [("H1", "Ciclicidade", "O setor passa por ciclos de alta e queda que se repetem.", COR["mem"]),
            ("H2", "Novo patamar", "Depois de uma valorização forte, o preço se acomoda mais alto.", COR["mem"]),
            ("H3", "Lançamentos de IA", "Depois de grandes lançamentos, as ações valorizam acima da tendência.", COR["gpu"])]
    for col, (ref, nome, texto, cor) in zip(st.columns(3), hips):
        col.markdown(f'<div class="hip" style="--c:{cor}"><div>{chip(ref, cor)}</div><div class="ht">{esc(nome)}</div><p>{esc(texto)}</p></div>', unsafe_allow_html=True)
    st.markdown("&nbsp;")
    p = ctx.painel
    n_acoes = sum(1 for c in ctx.categorias.values() if c in ("Memória", "GPU / IA", "Semicondutores"))
    _cartoes([(str(n_acoes), "empresas analisadas", "gold"),
              (L.num(p[MEM].dropna().shape[0], 0), "pregões em comum", "mem"),
              (f"{p.index.min().year}–{p.index.max().year}", "período dos dados", "gpu"),
              (f"{p.index.max():%d/%m/%Y}", "dado mais recente", "gold")])
    md('<div class="rot" style="margin-top:1rem">Felipe Niehues Garcia · Murilo Costa da Silva · Bruno Henrique da Silva Rossetim</div>')
    _roteiro(["Abra com a pergunta. O resto da apresentação responde a ela em três partes, uma por hipótese.",
              "H1 pergunta se há ciclos. H2, se o preço se acomoda mais alto depois de uma grande alta. H3, se lançamentos de IA mexem com as ações.",
              "Os números são recalculados a partir dos dados mais recentes, por isso a data no canto."])


def s_filme(ctx):
    m = _mensal(ctx)
    anos = round((m.index[-1] - m.index[0]).days / 365.25)
    _cabeca("Visão geral", f"{anos} anos em um gráfico")
    _frase(f"Desde {m.index[0].year}, o Índice Memória multiplicou por <b>{L.num(m[MEM].iloc[-1] / 100, 0)}</b> "
           f"e o Índice GPU por <b>{L.num(m[GPU].iloc[-1] / 100, 0)}</b>.")
    visual.mostrar(graficos.filme(m, dados.carregar_eventos(), ctx.cores, fonte=FONTE, altura=450), "filme", chave="palco_filme")
    _roteiro(["Cada linha é a média das empresas do setor, reescalada para começar em 100. Se a linha chega a 1.000, o setor multiplicou por dez.",
              "Aperte Reproduzir: os losangos dourados são lançamentos de IA aparecendo conforme aconteceram. Dá para arrastar a barra para voltar no tempo.",
              "A escala é logarítmica: a mesma inclinação significa o mesmo ritmo percentual. Sem isso, os primeiros anos pareceriam planos.",
              "Pergunta provável: por que a GPU sobe tanto mais? Resposta: o índice reúne empresas muito valorizadas pela demanda de IA, sobretudo a NVIDIA."])


def s_ciclos(ctx):
    series = {MEM: ctx.serie(MEM, "M", completa=True), GPU: ctx.serie(GPU, "M", completa=True)}
    eps = an.episodios_queda(series[MEM], .30)
    _cabeca("H1 · Ciclicidade", "O setor cai fundo e volta, mais de uma vez?")
    recuperadas = int(eps["recuperação"].notna().sum()) if len(eps) else 0
    _frase(f"Desde {series[MEM].index[0].year}, o Índice Memória caiu mais de 30% a partir de um pico <b>{len(eps)} vez{'es' if len(eps) != 1 else ''}</b>, "
           f"e em {recuperadas} delas já voltou ao pico anterior." + (" Quedas fundas que se repetem são o desenho de um ciclo." if len(eps) >= 2 else ""))
    atual = an.drawdown(series[MEM]).iloc[-1] * 100
    rec = eps["meses para recuperar"].dropna()
    _cartoes([(L.pct(eps["queda"].min() * 100, 0) if len(eps) else "–", "maior queda do Índice Memória", "gpu"),
              (str(len(eps)), "quedas de mais de 30% a partir de um pico", "gold"),
              (f"{L.num(rec.mean(), 0)} meses" if len(rec) else "–", "tempo médio para recuperar o pico", "mem"),
              (L.pct(atual, 0), "distância do pico hoje", "ok" if atual > -10 else "gold")])
    visual.mostrar(graficos.drawdowns(
        series, ctx.cores, "Quanto cada índice está abaixo do último pico", "Passe o mouse: o hover mostra de qual data era o pico.",
        fonte=FONTE, altura=360, ajuda="0% é o máximo histórico até a data. -40% é 40% abaixo do último pico."), "quedas", chave="palco_quedas")
    _roteiro(["Cada vale é uma queda a partir de um pico. Voltar a zero quer dizer que o pico anterior foi recuperado.",
              "Um ciclo é isso: cair forte e se recuperar, mais de uma vez. Contamos só quedas maiores que 30%, para não confundir ciclo com ruído.",
              "Pergunta provável: isso prova que existe ciclo? Resposta: é um sinal forte, mas ciclo pede alguma regularidade, e os intervalos aqui são irregulares."])


def s_calendario(ctx):
    tab = an.tabela_sazonal(ctx.serie(MEM, "M", completa=True))
    p = an.teste_sazonalidade(tab)
    _cabeca("H1 · Ciclicidade", "Existe um mês do ano que se repete?")
    _frase(L.sazonal(tab, p)[1])
    c1, c2 = st.columns([3, 2])
    with c1:
        visual.mostrar(graficos.mapa_sazonal(tab, "Retorno de cada mês do Índice Memória", "Verde subiu, laranja caiu.", fonte=FONTE, altura=500), "calendario", chave="palco_cal")
    with c2:
        visual.mostrar(graficos.barras_mes(tab, "Média por mês", "O hover mostra em quantos anos o mês subiu.", fonte=FONTE, altura=500), "media_mes", chave="palco_mes")
    _roteiro(["Cada linha é um ano e cada coluna, um mês. Verde: subiu. Laranja: caiu.",
              "Se houvesse sazonalidade forte, as mesmas colunas teriam a mesma cor em muitos anos. O teste estatístico (ANOVA) diz se os meses diferem de verdade.",
              "Sazonalidade é o que se repete todo ano; ciclo é mais longo e irregular. Por isso olhamos as duas coisas."])


def s_patamar(ctx):
    _cabeca("H2 · Novo patamar", "Depois da grande alta, o preço se acomodou mais alto?")
    nome = _escolher_indice("pal_pat_nome")
    s = ctx.serie(nome, "M", completa=True)
    lo, hi = s.index[12].date(), s.index[-12].date()
    marco = st.slider("Arraste o marco no tempo", lo, hi, min(max(MARCO, lo), hi), step=timedelta(days=30), format="MM/YYYY", key="pal_marco",
                      help="Por padrão, o lançamento do ChatGPT (30/11/2022). Mexa para ver como o 'antes' e o 'depois' mudam.")
    marco_ts = pd.Timestamp(marco)
    info = an.novo_patamar(s, marco_ts)
    _frase(L.patamar(nome, info)[1])
    r = an.comparar_periodos(ctx.tabela([nome], "M", completa=True), marco_ts)
    a, d = r["antes"].loc[nome], r["depois"].loc[nome]
    _cartoes([(L.pct(a["retorno médio mensal (%)"], 2), f"retorno médio mensal até {marco:%m/%Y}", "mem"),
              (L.pct(d["retorno médio mensal (%)"], 2), f"retorno médio mensal depois de {marco:%m/%Y}", "gpu"),
              (L.pct(a["volatilidade anualizada (%)"], 0), "volatilidade anual antes", "gold"),
              (L.pct(d["volatilidade anualizada (%)"], 0), "volatilidade anual depois", "gold")])
    visual.mostrar(graficos.patamar(
        s, marco_ts, info, nome, ctx.cores.get(nome, COR["mem"]), f"{nome} antes e depois do marco",
        "A linha pontilhada dourada é o pico anterior ao marco.", fonte=FONTE, altura=370,
        ajuda="Cinza: antes do marco. Colorido: depois. Se a série fica acima da linha dourada, o patamar novo se sustentou."), "patamar", chave="palco_patamar")
    _roteiro(["O padrão do marco é o lançamento do ChatGPT. Arraste o controle para mostrar que o resultado não depende de uma data escolhida a dedo.",
              "A linha dourada é o maior valor atingido antes do marco. Se, depois de superá-la, a série nunca mais voltou abaixo dela, houve um novo patamar.",
              "Pergunta provável: isso foi causado pela IA? Resposta: o gráfico mostra antes e depois, não a causa. Juros, pandemia e outros fatores também mexeram no período."])


def s_eventos(ctx):
    _cabeca("H3 · Lançamentos de IA", "As ações andam além do esperado depois de um lançamento?")
    c1, c2 = st.columns([1, 2])
    with c1:
        nome = _escolher_indice("pal_ev_nome")
    with c2:
        depois = st.radio("Janela", [10, 21, 63], horizontal=True, key="pal_ev_janela", label_visibility="collapsed",
                          format_func={10: "10 pregões (2 semanas)", 21: "21 pregões (1 mês)", 63: "63 pregões (3 meses)"}.get)
    ev = dados.carregar_eventos()
    mercado = ctx.serie(SP, "D", completa=True) if SP in ctx.painel else None
    traj, meta, _, res = an.estudo_eventos(ctx.serie(nome, "D", completa=True), mercado, ev, 5, depois)
    if res is None:
        st.warning("Eventos insuficientes para esta janela.")
        return
    _frase(L.eventos(res, depois, nome)[1])
    car = res["car_final_medio"] * 100
    _cartoes([(L.pct(car, 1, True), f"retorno além do esperado, em média, em {depois} pregões", "ok" if car > 0 else "gpu"),
              (L.pct(res["pct_positivos"] * 100, 0), f"dos {res['n']} eventos terminaram acima do esperado", "gold"),
              ("n/d" if np.isnan(res["p"]) else L.pnum(res["p"]), "p-valor (abaixo de 0,05 = efeito real)", "ok" if res["p"] < .05 else "mem")])
    visual.mostrar(graficos.barras_eventos(
        traj, meta, depois, "data", "Efeito de cada evento", f"Cada barra é um lançamento ou fato de IA. Passe o mouse para ver qual.",
        fonte=FONTE, altura=360, ajuda="Para cada evento: quanto a série andou além do que o mercado em geral faria nos pregões seguintes."), "efeito_eventos", chave="palco_ev")
    _roteiro(["Para cada lançamento, medimos quanto a série andou além do que o mercado inteiro (S&P 500) explicaria. Cada barra é um evento.",
              "O p-valor responde: esse efeito médio é diferente de zero ou pode ser acaso? Abaixo de 0,05, consideramos que é real.",
              "Troque a janela para ver se o efeito aparece em prazos mais longos, que é o que a H3 descreve (meses seguintes).",
              "Pergunta provável: por que tantos eventos juntos? Resposta: lançamentos próximos compartilham janelas e não são independentes; é uma limitação a declarar."])


def s_par(ctx):
    m = _mensal(ctx)
    r = m.pct_change().dropna() * 100
    reg = an.regressao_retornos(r[MEM], r[GPU])
    cm = an.corr_movel(m[MEM], m[GPU], 24).dropna()
    _cabeca("Memória × GPU", "Os dois segmentos andam juntos?")
    _frase(L.par(MEM, GPU, reg)[1])
    c1, c2 = st.columns(2)
    with c1:
        visual.mostrar(graficos.dispersao(reg, MEM, GPU, "Retornos mensais", "Cada ponto é um mês. A reta laranja é o ajuste.", fonte=FONTE, altura=420), "dispersao", chave="palco_disp")
    with c2:
        visual.mostrar(graficos.correlacao_movel(cm, MEM, GPU, "Correlação ao longo do tempo", "Janela de 24 meses.", fonte=FONTE, altura=420), "corr_movel", chave="palco_cm")
    _roteiro(["Cada ponto é um mês: o retorno da memória no eixo horizontal e o da GPU no vertical. A reta é o ajuste médio.",
              "Correlação perto de zero: cada setor segue a própria história. Perto de 1: andam juntos. É moderada: a demanda de IA os une, mas não por inteiro.",
              "O gráfico da direita mostra que essa relação muda com o tempo; não é um número fixo."])


def s_insumos(ctx):
    cats = {"Commodities", "Câmbio", "Juros / risco", "Mercado"}
    cand = [c for c in ctx.colunas if ctx.categorias.get(c) in cats]
    d = ctx.tabela([MEM] + cand, "M", completa=True)
    r0, r1 = an.ranking_correlacao(d, MEM, 0), an.ranking_correlacao(d, MEM, 1)
    _cabeca("Insumos e contexto", "O que anda junto com a memória?")
    if r0.empty:
        st.warning("Poucas observações em comum.")
        return
    _frase(L.ranking(MEM, r0, 0)[1])
    c1, c2 = st.columns(2)
    with c1:
        visual.mostrar(graficos.barras_correlacao(r0, MEM, "No mesmo mês", "Dourado anda junto; azul, em sentido oposto.", fonte=FONTE, altura=420), "corr_mesmo_mes", chave="palco_r0")
    with c2:
        if r1.empty:
            st.caption("Sem dados suficientes para a versão com atraso.")
        else:
            visual.mostrar(graficos.barras_correlacao(r1, MEM, "Com 1 mês de atraso", "A série de um mês antes contra a memória de hoje.", fonte=FONTE, altura=420), "corr_atraso", chave="palco_r1")
    _roteiro(["Cada barra é a correlação entre os retornos mensais da memória e de outra série: cobre, ouro, câmbio, juros, bolsa, VIX.",
              "Barras douradas andam junto com a memória. Azuis andam no sentido oposto. Perto de zero: sem relação.",
              "À direita, a mesma conta com a outra série um mês antes. Se alguma barra for grande ali, essa série pode antecipar a memória.",
              "Pergunta provável: cobre e ouro explicam a memória? Resposta: correlação não prova causa, e em geral aparecem como relações fracas em comparação com a bolsa de semicondutores."])


def s_risco(ctx):
    _cabeca("Risco", "O quão agitado está o mercado agora?")
    nome = _escolher_indice("pal_risco_nome")
    s = ctx.serie(nome, "D", completa=True)
    tabela, vols, _, _, arch = calculos.garch(s, ("GARCH(1,1)",), "t", 20, "D")
    vol = vols["GARCH(1,1)"]
    atual, pct = float(vol.iloc[-1]), an.percentil(vol, float(vol.iloc[-1]))
    _frase(L.garch(nome, tabela, arch, atual, pct)[1])
    c1, c2 = st.columns([1, 2])
    with c1:
        visual.mostrar(graficos.termometro(atual, pct, vol, f"Volatilidade de {nome}", fonte=FONTE, altura=400), "termometro", chave="palco_termo")
    with c2:
        visual.mostrar(graficos.volatilidade(vols, an.PERIODOS_ANO["D"], "Volatilidade ao longo do tempo", "GARCH(1,1) com distribuição t.", fonte=FONTE, altura=400), "volatilidade", chave="palco_vol")
    _roteiro(["Volatilidade é o tamanho típico das oscilações. Ela vem em ondas: dias agitados tendem a vir juntos. O modelo GARCH captura isso.",
              "O termômetro compara a agitação de hoje com toda a história: no vermelho, estamos entre os períodos mais agitados já vistos.",
              "Pergunta provável: isso serve para prever preço? Resposta: não diz a direção, só o tamanho esperado dos movimentos; é uma medida de risco."])


def _dados_modelos(nome):
    versao = calculos.versao_modelos()
    serie = calculos.serie_modelo(nome, versao)
    tab, _, melhor_base, _, _ = calculos.comparacao(nome, versao)
    entradas = {f: calculos.entrada_modelo(nome, f, versao) for f in modelos.FAMILIAS}
    melhor = tab.loc[list(modelos.FAMILIAS), "rmse"].idxmin()
    return serie, tab, melhor_base, entradas, melhor


def s_modelos(ctx):
    _cabeca("Modelos de previsão", "Qual modelo prevê melhor?")
    nome = _escolher_indice("pal_mod_nome")
    serie, tab, melhor_base, entradas, melhor = _dados_modelos(nome)
    linha = tab.loc[melhor]
    melhor_crps = tab.loc[list(modelos.FAMILIAS), "crps"].idxmin()
    desc = modelos.descrever(melhor, entradas[melhor]["config"])
    outro = None if melhor_crps == melhor else modelos.descrever(melhor_crps, entradas[melhor_crps]["config"])
    _frase(L.conclusao(nome, desc, "erro médio nos 12 meses (RMSE)", linha, NOMES[melhor_base].lower(), linha["dm_p"], outro)[1])
    _cartoes([(NOMES[melhor], "melhor modelo pelo menor erro", "gold"),
              (f"±{L.pct((np.exp(linha['rmse']) - 1) * 100, 0)}", "erro típico de 1 a 12 meses à frente", "mem"),
              ("igual" if abs(linha["vs_base"]) < .0005 else L.pct(linha["vs_base"] * 100, 1, True), f"erro em relação ao {NOMES[melhor_base].lower()}",
               "mem" if abs(linha["vs_base"]) < .0005 else "ok" if linha["vs_base"] < 0 else "gpu"),
              ("n/d" if np.isnan(linha["dm_p"]) else L.pnum(linha["dm_p"]), "p-valor (abaixo de 0,05 = diferença real)", "ok" if linha["dm_p"] < .05 else "mem")])
    visual.mostrar(graficos.barras_modelos(
        (np.exp(tab["rmse"]) - 1) * 100, "Erro típico de cada modelo no teste dos últimos 5 anos",
        "Menor é melhor. Contorno dourado: o melhor. Cinza e branco: passeios aleatórios de referência.", melhor=melhor, formato="±{:.0f}%",
        x_titulo="Erro típico (%)", fonte=FONTE, altura=340,
        ajuda="A cada mês dos últimos 5 anos, cada modelo previu os 12 meses seguintes só com o que se sabia então. A barra é o erro médio."),
        "modelos", chave="palco_modelos")
    _roteiro(["Comparamos cinco modelos: ARIMA, SARIMA, SARIMAX (com os lançamentos de IA), ARCH e GARCH. Cada um já foi ajustado na sua melhor versão.",
              "O teste é honesto: a cada mês dos últimos 5 anos o modelo prevê os 12 meses seguintes usando só o passado daquele mês. A barra é o erro médio.",
              "Os dois passeios aleatórios (cinza e branco) são o adversário mínimo: repetir o último valor, com ou sem crescimento médio.",
              "Olhe o p-valor: se estiver acima de 0,05, a vantagem do melhor modelo pode ser acaso. Nesse caso diga que os modelos empatam e que a escolha deve se apoiar em simplicidade e na qualidade do intervalo.",
              "Pergunta provável: por que ARCH e GARCH entram se eles modelam volatilidade? Resposta: o valor central vem do retorno médio e o intervalo, da volatilidade prevista; o ganho deles está sobretudo no intervalo."])


def s_previsao(ctx):
    _cabeca("Previsão", "E nos próximos doze meses?")
    nome = _escolher_indice("pal_prev_nome")
    serie, tab, melhor_base, entradas, melhor = _dados_modelos(nome)
    fin = entradas[melhor]["final"]
    mu, sd = np.asarray(fin["mu"]), np.asarray(fin["sd"])
    atual = float(np.exp(serie.y.iloc[-1]))
    central, lo, hi = float(np.exp(mu[-1])), float(np.exp(mu[-1] - 1.96 * sd[-1])), float(np.exp(mu[-1] + 1.96 * sd[-1]))
    var = (central / atual - 1) * 100
    _frase(f"Pelo melhor modelo ({NOMES[melhor]}), o <b>{esc(nome)}</b> vai de {L.num(atual, 0)} para cerca de <b>{L.num(central, 0)}</b> em 12 meses "
           f"({L.pct(var, 0, True)}), com intervalo de 95% entre {L.num(lo, 0)} e {L.num(hi, 0)}.")
    _cartoes([(L.num(central, 0), "valor central previsto em 12 meses", "gold"),
              (f"{L.num(lo, 0)} a {L.num(hi, 0)}", "intervalo de confiança de 95%", "mem"),
              (L.pct(var, 0, True), "variação sobre o último valor", "ok" if var >= 0 else "gpu")])
    outros = [(NOMES[f], entradas[f]["final"]["mu"], COR_MODELO[f]) for f in modelos.FAMILIAS if f != melhor]
    visual.mostrar(graficos.leque(
        serie.y.iloc[-modelos.HIST_DIAS:], modelos.datas_futuras(serie), mu, sd, f"Previsão de {nome}", f"Faixas e linha dourada: {NOMES[melhor]}. Pontilhadas: os outros modelos.",
        outros=outros, fonte=FONTE, altura=380,
        ajuda="Linha dourada: previsão central do melhor modelo. Faixas: onde o valor deve cair 80% e 95% das vezes se o modelo estiver certo."),
        "previsao", chave="palco_prev")
    _roteiro(["Esta é a previsão do melhor modelo do slide anterior. As faixas mostram a incerteza e se abrem rápido, o que é esperado.",
              "As linhas pontilhadas são os outros modelos: se ficam perto, a escolha do modelo importa pouco; se se afastam, há incerteza sobre o próprio modelo.",
              "Pergunta provável: dá para confiar na previsão? Resposta: como cenário sim, como aposta não. O intervalo é largo e o teste mostrou que os modelos mal superam o passeio aleatório."])


def _veredito(rotulo, cor):
    return chip(rotulo, cor)


def _linha_placar(nome, rotulo, cor, texto):
    return (f'<div style="margin:.5rem 0"><div class="rot" style="margin:0 0 .2rem">{esc(nome)}</div>'
            f'{_veredito(rotulo, cor)}<span class="rot" style="display:block;margin-top:.3rem">{texto}</span></div>')


def s_placar(ctx):
    _cabeca("Conclusões", "O que os dados dizem sobre as três hipóteses")
    ev = dados.carregar_eventos()
    mercado = ctx.serie(SP, "D", completa=True) if SP in ctx.painel else None
    h = {"H1": [], "H2": [], "H3": []}
    for nome in (MEM, GPU):
        sm = ctx.serie(nome, "M", completa=True)
        eps = an.episodios_queda(sm, .30)
        if len(eps) >= 2:
            h["H1"].append((nome, "Sustentada", COR["ok"], f"{len(eps)} quedas de mais de 30%; a maior, {L.pct(eps['queda'].min() * 100, 0)}"))
        else:
            h["H1"].append((nome, "Inconclusiva", COR["gold"], f"{len(eps)} queda(s) de mais de 30%"))
        info = an.novo_patamar(sm, pd.Timestamp(MARCO))
        if info is None or not info["superou"]:
            h["H2"].append((nome, "Não sustentada", COR["gpu"], "ainda não superou o pico anterior ao marco"))
        elif not info["voltou_abaixo"]:
            h["H2"].append((nome, "Sustentada", COR["ok"], "nunca voltou abaixo do pico anterior depois de superá-lo"))
        else:
            h["H2"].append((nome, "Parcial", COR["gold"], f"voltou abaixo do pico anterior em {L.mes_ano(info['data_piso'])}"))
        _, _, _, res = an.estudo_eventos(ctx.serie(nome, "D", completa=True), mercado, ev, 5, 21)
        if res is None or np.isnan(res["p"]):
            h["H3"].append((nome, "Inconclusiva", COR["gold"], "poucos eventos para testar"))
        else:
            car = res["car_final_medio"] * 100
            txt = f"CAR médio {L.pct(car, 1, True)} em 21 pregões, {L.pval(res['p'])}"
            if res["p"] < .05:
                h["H3"].append((nome, "Sustentada" if car > 0 else "Contrária", COR["ok"] if car > 0 else COR["gpu"], txt))
            else:
                h["H3"].append((nome, "Inconclusiva", COR["gold"], txt))

    titulos = {"H1": ("Ciclicidade", COR["mem"]), "H2": ("Novo patamar", COR["mem"]), "H3": ("Lançamentos de IA", COR["gpu"])}
    for col, ref in zip(st.columns(3), h):
        nome_h, cor = titulos[ref]
        linhas = "".join(_linha_placar(*x[:3], esc(x[3])) for x in h[ref])
        col.markdown(f'<div class="hip" style="--c:{cor}"><div>{chip(ref, cor)}</div><div class="ht">{esc(nome_h)}</div>{linhas}</div>', unsafe_allow_html=True)
    st.markdown("&nbsp;")
    visual.leitura("Leia com cuidado",
                   "Os rótulos acima são regras simples aplicadas aos dados atuais, um ponto de partida e não um veredito. Correlação não prova causa; "
                   "o marco de nov/2022 é um único recorte; lançamentos próximos compartilham janelas; e previsão de preço é incerta por natureza. "
                   "Revise cada afirmação antes de levá-la ao relatório.", "info")
    _roteiro(["Feche respondendo as três hipóteses, uma de cada vez, e sempre dizendo qual teste está por trás.",
              "Os rótulos são automáticos e seguem regras simples (descritas em cada página). Se o dado mudar, o rótulo muda.",
              "Próximos passos naturais: ampliar a base de eventos com datas conferidas e testar outros modelos e janelas de previsão."])


SLIDES = [("Abertura", s_capa), ("O filme", s_filme), ("H1 · Quedas", s_ciclos), ("H1 · Calendário", s_calendario),
          ("H2 · Novo patamar", s_patamar), ("H3 · Lançamentos de IA", s_eventos), ("Memória × GPU", s_par),
          ("Insumos", s_insumos), ("Risco", s_risco), ("Modelos", s_modelos), ("Previsão", s_previsao), ("Conclusões", s_placar)]


# --------------------------------------------------------------------------- página
def _ir(delta):
    st.session_state["palco_i"] = int(np.clip(st.session_state["palco_i"] + delta, 0, len(SLIDES) - 1))


def render(ctx):
    visual.aplicar_tema_palco()
    st.html(f"<script>{TECLAS}</script>", unsafe_allow_javascript=True)
    st.session_state.setdefault("palco_i", 0)
    i = st.session_state["palco_i"]

    topo, sair = st.columns([6, 1])
    with topo:
        pontos = "".join(f'<span class="ponto {"atual" if k == i else "passou" if k < i else ""}"></span>' for k in range(len(SLIDES)))
        md(f'<div class="pontos">{pontos}</div><div class="ref">PALCO · {i + 1} DE {len(SLIDES)} · {esc(SLIDES[i][0]).upper()}</div>')
    with sair:
        if "inicio" in ctx.paginas and st.button("Sair do palco", key="palco_sair", help="Volta para a página inicial."):
            st.switch_page(ctx.paginas["inicio"])

    SLIDES[i][1](ctx)

    st.markdown("&nbsp;")
    ant, meio, prox = st.columns([1, 4, 1])
    ant.button("◀  Anterior", key="palco_ant", disabled=i == 0, on_click=_ir, args=(-1,), width="stretch",
               help="Também funciona com a seta esquerda do teclado.")
    meio.markdown(f'<div class="rot" style="text-align:center">Use as setas ← → do teclado para trocar de slide · F11 para tela cheia</div>', unsafe_allow_html=True)
    prox.button("Próximo  ▶", key="palco_prox", disabled=i == len(SLIDES) - 1, on_click=_ir, args=(1,), type="primary", width="stretch",
                help="Também funciona com a seta direita do teclado.")
