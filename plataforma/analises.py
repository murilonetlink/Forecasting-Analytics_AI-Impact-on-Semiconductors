"""Análises estatísticas da plataforma. Todas as funções recebem pandas Series/DataFrame indexados por data."""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests

PERIODOS_ANO = {"D": 252, "W": 52, "M": 12}


def variacao(s):
    """Retorno logarítmico; diferença simples se a série tiver valores <= 0."""
    s = s.dropna()
    return np.log(s).diff().dropna() if (s > 0).all() else s.diff().dropna()


def nivel_log(s):
    s = s.dropna()
    return np.log(s) if (s > 0).all() else s


def base100(df, inicio=None, fim=None):
    d = df.loc[inicio:fim].dropna()
    return d / d.iloc[0] * 100


def corr_movel(a, b, janela):
    r = pd.concat([variacao(a), variacao(b)], axis=1, join="inner").dropna()
    return r.iloc[:, 0].rolling(janela).corr(r.iloc[:, 1])


# ------------------------------------------------------------ estudo de eventos
def estudo_eventos(preco, mercado, eventos, antes=5, depois=10, janela_est=250, folga=30):
    """CAR (retorno anormal acumulado) em torno de cada evento.

    Com `mercado`, usa modelo de mercado estimado em [-(janela_est+folga), -folga) pregões
    antes do evento. Sem `mercado`, o retorno anormal é o retorno menos a média da janela.
    `eventos`: DataFrame com colunas data, evento, categoria e empresa.
    Retorna (trajetórias, metadados dos eventos usados, ignorados, resumo).
    """
    r = variacao(preco)
    rm = None
    if mercado is not None:
        j = pd.concat([r, variacao(mercado)], axis=1, join="inner").dropna()
        r, rm = j.iloc[:, 0], j.iloc[:, 1]

    offsets = list(range(-antes, depois + 1))
    linhas, usados, ignorados = [], [], []
    for _, ev in eventos.iterrows():
        pos = r.index.searchsorted(ev["data"])
        ini_est, fim_est = pos - janela_est - folga, pos - folga
        if ini_est < 0 or pos - antes < 0 or pos + depois >= len(r):
            ignorados.append(f"{ev['data']:%Y-%m-%d} {ev['evento']}")
            continue
        if rm is None:
            esperado = np.full(len(offsets), r.iloc[ini_est:fim_est].mean())
        else:
            beta, alfa = np.polyfit(rm.iloc[ini_est:fim_est], r.iloc[ini_est:fim_est], 1)
            esperado = alfa + beta * rm.iloc[pos - antes: pos + depois + 1].values
        ar = r.iloc[pos - antes: pos + depois + 1].values - esperado
        linhas.append(np.cumsum(ar))
        usados.append(ev)

    meta = pd.DataFrame(usados).reset_index(drop=True)
    traj = pd.DataFrame(linhas, columns=offsets, index=meta.index if len(meta) else None)
    resumo = None
    if len(traj) >= 2:
        media = traj.mean()
        ep = traj.std(ddof=1) / np.sqrt(len(traj))
        final = traj.iloc[:, -1]
        t, p = stats.ttest_1samp(final, 0.0) if len(final) >= 3 else (np.nan, np.nan)
        resumo = {"n": len(traj), "media": media, "ic": 1.96 * ep,
                  "car_final_medio": final.mean(), "t": t, "p": p,
                  "pct_positivos": (final > 0).mean()}
    return traj, meta, ignorados, resumo


# --------------------------------------------------------------------- testes
def teste_adf(s):
    r = adfuller(s.dropna(), autolag="AIC")
    return {"estatística": r[0], "p-valor": r[1], "defasagens": r[2],
            "crítico 5%": r[4]["5%"], "estacionária (5%)": r[1] < 0.05}


def granger(x, y, max_lag=5):
    """p-valores do teste F de que x ajuda a prever y, por defasagem. Usa variações, não níveis."""
    d = pd.concat([variacao(y).rename("y"), variacao(x).rename("x")], axis=1, join="inner").dropna()
    res = grangercausalitytests(d[["y", "x"]], maxlag=max_lag)
    return pd.Series({lag: res[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)},
                     name="p-valor").rename_axis("defasagem")


def cointegracao(a, b):
    """Engle-Granger sobre níveis em log. H0: não há cointegração."""
    d = pd.concat([nivel_log(a), nivel_log(b)], axis=1, join="inner").dropna()
    t, p, crit = coint(d.iloc[:, 0], d.iloc[:, 1], trend="c", autolag="aic")
    return {"estatística": t, "p-valor": p, "crítico 5%": crit[1], "observações": len(d)}


# ----------------------------------------------------------------- ARCH / GARCH
ESPECIFICACOES = {
    "ARCH(1)":        dict(vol="ARCH", p=1),
    "ARCH(5)":        dict(vol="ARCH", p=5),
    "GARCH(1,1)":     dict(vol="GARCH", p=1, q=1),
    "GJR-GARCH(1,1)": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH(1,1)":    dict(vol="EGARCH", p=1, o=1, q=1),
}


def efeito_arch(r, lags=10):
    """Teste LM de Engle. H0: sem efeito ARCH (variância constante)."""
    lm, p, _, _ = het_arch(r - r.mean(), nlags=lags)
    return {"LM": lm, "p-valor": p}


def ajustar_garch(r, nome, dist="t"):
    """r em fração; o modelo trabalha em %. Retorna o resultado do arch."""
    spec = ESPECIFICACOES[nome]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return arch_model(r * 100, mean="Constant", dist=dist, **spec).fit(disp="off")


def comparar_garch(r, nomes, dist="t"):
    ajustes, linhas = {}, []
    for n in nomes:
        res = ajustar_garch(r, n, dist)
        ajustes[n] = res
        lb = acorr_ljungbox(res.std_resid.dropna() ** 2, lags=[10])["lb_pvalue"].iloc[0]
        linhas.append({"modelo": n, "log-verossimilhança": res.loglikelihood, "AIC": res.aic,
                       "BIC": res.bic, "Ljung-Box p (resíduos²)": lb})
    return ajustes, pd.DataFrame(linhas).set_index("modelo")


def prever_vol(res, nome, horizonte, freq):
    """Volatilidade prevista por período, em % anualizada."""
    metodo = "simulation" if nome.startswith("EGARCH") else "analytic"
    f = res.forecast(horizon=horizonte, method=metodo, simulations=500)
    var = f.variance.iloc[-1].values
    return pd.Series(np.sqrt(var * PERIODOS_ANO[freq]), index=range(1, horizonte + 1), name="vol anualizada (%)")


def vol_condicional_anual(res, freq):
    return res.conditional_volatility * np.sqrt(PERIODOS_ANO[freq])


# ------------------------------------------------------------- ciclos e patamar
def drawdown(s):
    """Distância (fração negativa) entre a série e o seu pico anterior."""
    s = s.dropna()
    return s / s.cummax() - 1


def data_do_pico(s):
    """Para cada data, a data do pico em que a queda atual começou."""
    s = s.dropna()
    datas = pd.Series(s.index, index=s.index)
    return datas.where(s == s.cummax()).ffill()


def episodios_queda(s, limiar=0.30):
    """Quedas de pico a fundo que passaram de `limiar`, com a data em que o pico foi recuperado."""
    s = s.dropna()
    dd = drawdown(s)
    linhas, i, n = [], 0, len(s)
    while i < n:
        if dd.iloc[i] >= 0:
            i += 1
            continue
        j = i
        while j < n and dd.iloc[j] < 0:
            j += 1
        trecho = dd.iloc[i:j]
        if trecho.min() <= -limiar:
            pico, fundo = s.index[i - 1], trecho.idxmin()
            rec = s.index[j] if j < n else pd.NaT
            linhas.append({"pico": pico, "fundo": fundo, "recuperação": rec, "queda": trecho.min(),
                           "meses até o fundo": (fundo - pico).days / 30.44,
                           "meses para recuperar": (rec - pico).days / 30.44 if pd.notna(rec) else np.nan})
        i = j
    return pd.DataFrame(linhas, columns=["pico", "fundo", "recuperação", "queda",
                                         "meses até o fundo", "meses para recuperar"])


def tabela_sazonal(mensal):
    """Retorno mensal (%) em matriz ano x mês. `mensal` já está em frequência mensal."""
    r = mensal.dropna().pct_change().dropna() * 100
    return r.groupby([r.index.year, r.index.month]).first().unstack()


def decomposicao(mensal, periodo=12):
    return seasonal_decompose(nivel_log(mensal.dropna()), model="additive", period=periodo)


def teste_sazonalidade(tab):
    """ANOVA de uma via sobre `tab` (ano x mês). H0: todos os meses têm o mesmo retorno médio. Devolve o p-valor."""
    grupos = [tab[c].dropna().values for c in tab.columns if tab[c].notna().sum() >= 2]
    return float(stats.f_oneway(*grupos).pvalue) if len(grupos) >= 3 else float("nan")


def comparar_periodos(mensal, corte):
    """Retorno médio e volatilidade antes e depois de `corte`. `mensal` já está em frequência mensal."""
    r = mensal.dropna().pct_change().dropna()
    saida = {}
    for nome, g in {"antes": r[r.index <= corte], "depois": r[r.index > corte]}.items():
        saida[nome] = pd.DataFrame({"retorno médio mensal (%)": g.mean() * 100,
                                    "volatilidade anualizada (%)": g.std() * np.sqrt(12) * 100,
                                    "meses": len(g)})
    return saida


def novo_patamar(s, corte):
    """Compara o pico anterior ao marco com o menor valor depois de a série superá-lo."""
    s = s.dropna()
    antes, depois = s[s.index <= corte], s[s.index > corte]
    if antes.empty or depois.empty:
        return None
    pico = antes.max()
    info = {"pico_anterior": pico, "data_pico": antes.idxmax(), "atual": s.iloc[-1],
            "atual_vs_pico": s.iloc[-1] / pico - 1, "superou": False}
    acima = depois[depois > pico]
    if acima.empty:
        return info
    cruzou = acima.index[0]
    apos = depois.loc[cruzou:]
    info.update(superou=True, data_superou=cruzou, piso=apos.min(), data_piso=apos.idxmin(),
                voltou_abaixo=bool(apos.min() < pico), folga=apos.min() / pico - 1)
    return info


def ranking_correlacao(d, alvo, defasagem=0, minimo=24):
    """Correlação dos retornos de cada coluna de `d` com os de `alvo`.

    Com defasagem > 0, compara o alvo de hoje com a outra série `defasagem` períodos atrás.
    """
    linhas = []
    y = variacao(d[alvo]).rename("y")
    for c in d.columns:
        if c == alvo:
            continue
        j = pd.concat([y, variacao(d[c]).rename("x")], axis=1, join="inner").dropna()
        if defasagem:
            j["x"] = j["x"].shift(defasagem)
            j = j.dropna()
        if len(j) >= minimo:
            linhas.append({"série": c, "correlação": j["y"].corr(j["x"]), "observações": len(j)})
    if not linhas:
        return pd.DataFrame(columns=["correlação", "observações"])
    return pd.DataFrame(linhas).set_index("série").sort_values("correlação", ascending=False)


def percentil(serie, valor):
    return float((serie.dropna() <= valor).mean() * 100)


def regressao_retornos(x, y):
    """Reta y = a + b x ajustada nos retornos (em %), com correlação e R²."""
    j = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    b, a = np.polyfit(j["x"], j["y"], 1)
    r = j["x"].corr(j["y"])
    return {"a": a, "b": b, "corr": r, "r2": r ** 2, "dados": j}
