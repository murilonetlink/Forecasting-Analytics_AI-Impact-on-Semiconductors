"""Leituras em linguagem simples dos resultados.

Cada função devolve (título, texto HTML, tom). Tom: "ok" = há evidência a favor, "info" = sem evidência
ou neutro, "alerta" = o resultado contraria a ideia ou pede cuidado. Nomes de séries passam por esc().
"""
import numpy as np

from plataforma.visual import esc

MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def num(x, casas=2, sinal=False):
    s = f"{x:+,.{casas}f}" if sinal else f"{x:,.{casas}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x, casas=1, sinal=False):
    """`x` já está em pontos percentuais."""
    return num(x, casas, sinal) + "%"


def pnum(p):
    return "< 0,001" if p < 0.001 else num(p, 3)


def pval(p):
    """Texto completo: 'p < 0,001' ou 'p = 0,797'."""
    return f"p {pnum(p)}" if p < 0.001 else f"p = {pnum(p)}"


def mes_ano(ts):
    return f"{MESES[ts.month - 1]}/{ts.year}"


def forca(r):
    a = abs(r)
    nivel = "muito fraca" if a < .2 else "fraca" if a < .4 else "moderada" if a < .6 else "forte" if a < .8 else "muito forte"
    return f"{nivel} e {'positiva' if r >= 0 else 'negativa'}"


# ----------------------------------------------------------------------- testes
def adf(nome, r_nivel, r_dif):
    n, d, nome = r_nivel, r_dif, esc(nome)
    if not n["estacionária (5%)"] and d["estacionária (5%)"]:
        return ("Leitura do teste",
                f"Em nível, <b>{nome}</b> <b>não é estacionária</b> ({pval(n['p-valor'])}): os choques tendem a ter efeito "
                f"permanente e a série não volta a uma média. Nos retornos (1ª diferença) ela é estacionária "
                f"({pval(d['p-valor'])}), então um ARIMA com d = 1 faz sentido. Isso é compatível com a H2 (novo patamar).", "ok")
    if n["estacionária (5%)"]:
        return ("Leitura do teste",
                f"<b>{nome}</b> já é estacionária em nível ({pval(n['p-valor'])}): os choques se dissipam e a série tende a "
                f"voltar a uma média. Isso vai contra a ideia de novo patamar.", "alerta")
    return ("Leitura do teste",
            f"<b>{nome}</b> não é estacionária nem na 1ª diferença ({pval(d['p-valor'])}). Pode precisar de outra transformação.", "alerta")


def granger(x, y, tab_xy, tab_yx):
    x, y = esc(x), esc(y)
    p1, p2 = tab_xy.min(), tab_yx.min()
    f1 = (f"<b>{x}</b> ajuda a prever <b>{y}</b> (menor {pval(p1)}, com {int(tab_xy.idxmin())} período(s) de defasagem)"
          if p1 < .05 else f"não há evidência de que <b>{x}</b> ajude a prever <b>{y}</b> (menor {pval(p1)})")
    f2 = (f"<b>{y}</b> ajuda a prever <b>{x}</b> (menor {pval(p2)})"
          if p2 < .05 else f"<b>{y}</b> também não ajuda a prever <b>{x}</b> (menor {pval(p2)})")
    return ("Leitura do teste",
            f"{f1[0].upper() + f1[1:]}; {f2}. É precedência estatística, não causa e efeito, e olhar várias defasagens "
            f"aumenta a chance de falso positivo.", "ok" if (p1 < .05 or p2 < .05) else "info")


def cointegracao(a, b, r):
    a, b = esc(a), esc(b)
    if r["p-valor"] < .05:
        return ("Leitura do teste",
                f"<b>{a}</b> e <b>{b}</b> parecem ligadas no longo prazo ({pval(r['p-valor'])}): podem se afastar, "
                f"mas tendem a voltar a uma relação estável.", "ok")
    return ("Leitura do teste",
            f"Sem evidência de ligação de longo prazo entre <b>{a}</b> e <b>{b}</b> ({pval(r['p-valor'])}): "
            f"as séries podem se afastar sem limite.", "info")


# ------------------------------------------------------------------ comparações
def ranking(alvo, rank, defasagem):
    alvo = esc(alvo)
    topo, fundo = rank.iloc[0], rank.iloc[-1]
    quando = "no mesmo período" if not defasagem else f"com {defasagem} período(s) de atraso"
    texto = (f"Quem mais anda junto com <b>{alvo}</b> ({quando}) é <b>{esc(topo.name)}</b> "
             f"(correlação {num(topo['correlação'], 2, True)}, {forca(topo['correlação'])}).")
    if fundo["correlação"] < -.1:
        texto += f" O que mais anda no sentido oposto é <b>{esc(fundo.name)}</b> ({num(fundo['correlação'], 2, True)})."
    return ("Quem anda junto", texto + " Correlação não prova causa.", "info")


def par(a, b, reg):
    a, b = esc(a), esc(b)
    return ("Leitura da dispersão",
            f"A correlação dos retornos mensais entre <b>{a}</b> e <b>{b}</b> é <b>{num(reg['corr'], 2, True)}</b> "
            f"({forca(reg['corr'])}). A cada 1 ponto percentual de retorno de {a}, {b} varia em média "
            f"{num(reg['b'], 2, True)} ponto percentual. {a} explica só {pct(reg['r2'] * 100, 0)} da variação de {b} (R²).", "info")


# ------------------------------------------------------------------ ciclos / H1
def quedas(nome, eps, limiar):
    nome = esc(nome)
    if eps.empty:
        return ("Leitura do gráfico", f"<b>{nome}</b> nunca caiu mais de {pct(limiar * 100, 0)} a partir de um pico neste período.", "info")
    pior = eps.loc[eps["queda"].idxmin()]
    texto = (f"<b>{nome}</b> caiu mais de {pct(limiar * 100, 0)} a partir de um pico <b>{len(eps)} vez{'es' if len(eps) != 1 else ''}</b>. A maior queda foi de "
             f"{pct(pior['queda'] * 100, 0)} (pico em {mes_ano(pior['pico'])}, fundo em {mes_ano(pior['fundo'])}).")
    rec = eps["meses para recuperar"].dropna()
    if len(rec):
        texto += f" Voltar ao pico levou em média {num(rec.mean(), 0)} meses."
    if len(eps) >= 2:
        return ("Leitura do gráfico", texto + " Quedas profundas que se repetem são um sinal de ciclos, o que sustenta a H1.", "ok")
    return ("Leitura do gráfico", texto + " Uma única queda grande não basta para falar em ciclo.", "info")


def sazonal(tab, p):
    media = tab.mean()
    pos = (tab > 0).sum() / tab.notna().sum() * 100
    mb, mp = media.idxmax(), media.idxmin()
    texto = (f"O melhor mês em média foi <b>{MESES[mb - 1]}</b> ({pct(media[mb], 1, True)}, positivo em {pct(pos[mb], 0)} dos anos) e o pior "
             f"foi <b>{MESES[mp - 1]}</b> ({pct(media[mp], 1, True)}, positivo em {pct(pos[mp], 0)}).")
    if np.isnan(p):
        return ("Leitura do calendário", texto, "info")
    if p < .05:
        return ("Leitura do calendário", texto + f" Os meses diferem de forma estatisticamente significativa (ANOVA, {pval(p)}): há sinal de sazonalidade.", "ok")
    return ("Leitura do calendário", texto + f" A diferença entre os meses não é estatisticamente significativa (ANOVA, {pval(p)}): "
            f"o padrão por mês é mais ruído do que regra.", "info")


# ----------------------------------------------------------------- patamar / H2
def patamar(nome, info):
    nome = esc(nome)
    if info is None:
        return ("Leitura do gráfico", "Escolha um marco com dados antes e depois.", "info")
    if not info["superou"]:
        return ("Leitura do gráfico",
                f"Depois do marco, <b>{nome}</b> ainda não superou o pico anterior ({mes_ano(info['data_pico'])}). Hoje está "
                f"{pct(info['atual_vs_pico'] * 100, 0, True)} em relação a ele. Sem evidência de novo patamar.", "info")
    if not info["voltou_abaixo"]:
        return ("Leitura do gráfico",
                f"Depois de superar o pico de {mes_ano(info['data_pico'])} (em {mes_ano(info['data_superou'])}), <b>{nome}</b> nunca "
                f"voltou a ficar abaixo dele: o menor valor desde então está {pct(info['folga'] * 100, 0, True)} do pico anterior. "
                f"É isso que a H2 descreve: um novo patamar.", "ok")
    return ("Leitura do gráfico",
            f"Depois de superar o pico de {mes_ano(info['data_pico'])}, <b>{nome}</b> voltou a cair abaixo dele em "
            f"{mes_ano(info['data_piso'])} (piso {pct(info['folga'] * 100, 0, True)} em relação ao pico anterior). O novo patamar "
            f"não se sustentou por completo.", "alerta")


def periodos(nome, antes, depois, marco):
    a, d = antes.loc[nome], depois.loc[nome]
    ra, rd = a["retorno médio mensal (%)"], d["retorno médio mensal (%)"]
    return ("Leitura da comparação",
            f"O retorno médio mensal de <b>{esc(nome)}</b> foi de {pct(ra, 2)} até {marco:%d/%m/%Y} e de {pct(rd, 2)} depois, "
            f"com volatilidade de {pct(a['volatilidade anualizada (%)'], 0)} para {pct(d['volatilidade anualizada (%)'], 0)} ao ano. "
            f"{'O retorno subiu' if rd > ra else 'O retorno não subiu'} depois do marco, mas um único recorte não prova causa.",
            "ok" if rd > ra else "info")


# ------------------------------------------------------------------- eventos / H3
def eventos(resumo, depois, alvo):
    alvo = esc(alvo)
    car, p = resumo["car_final_medio"] * 100, resumo["p"]
    pos = f"{pct(resumo['pct_positivos'] * 100, 0)} dos {resumo['n']} eventos tiveram CAR positivo."
    if np.isnan(p):
        return ("Leitura do estudo", f"Poucos eventos para testar (n = {resumo['n']}). CAR médio de {pct(car, 2, True)}. {pos}", "info")
    if p < .05:
        sentido = "acima" if car > 0 else "abaixo"
        return ("Leitura do estudo",
                f"Em média, <b>{alvo}</b> ficou {pct(abs(car), 1)} {sentido} do esperado {depois} pregões depois dos eventos "
                f"({pval(p)}), um efeito estatisticamente significativo. {pos}", "ok" if car > 0 else "alerta")
    return ("Leitura do estudo",
            f"Em média, <b>{alvo}</b> andou {pct(car, 1, True)} em relação ao esperado {depois} pregões depois dos eventos, mas com "
            f"{pval(p)} isso não se distingue de zero: pode ser acaso. Com esta amostra, a H3 <b>não se confirma</b>. {pos}", "info")


# -------------------------------------------------------------------- volatilidade
def garch(nome, tabela, arch, atual, pct_atual):
    ondas = arch["p-valor"] < .05
    melhor = tabela["AIC"].idxmin()
    return ("Leitura dos modelos",
            f"A volatilidade de <b>{esc(nome)}</b> {'vem em ondas: dias agitados tendem a se agrupar' if ondas else 'não mostra ondas claras'} "
            f"(ARCH-LM, {pval(arch['p-valor'])}). Pelo AIC, o melhor ajuste é <b>{melhor}</b>. Hoje a volatilidade estimada é de "
            f"{pct(atual, 0)} ao ano, maior que {pct(pct_atual, 0)} dos períodos da história.", "ok" if ondas else "info")


# ----------------------------------------------------------------------- previsão
def arima(ordem, razao, horizonte, previsto, inferior, superior):
    ordem = "(" + ",".join(str(o) for o in ordem) + ")"
    prev = f" Previsão para {horizonte} períodos: {num(previsto, 0)} (IC 95%: {num(inferior, 0)} a {num(superior, 0)})."
    if razao < 1:
        return ("Leitura da previsão",
                f"No trecho de teste, o ARIMA{ordem} errou {pct((1 - razao) * 100, 0)} menos que o passeio aleatório (repetir o último valor)."
                f"{prev} Vale como cenário, com a ressalva de ser um único trecho de teste.", "ok")
    return ("Leitura da previsão",
            f"No trecho de teste, o ARIMA{ordem} errou {pct((razao - 1) * 100, 0)} <b>mais</b> que o passeio aleatório: aqui o modelo não "
            f"compensa.{prev} Trate como cenário, não como aposta.", "alerta")
