"""Modelos de previsão: busca da melhor configuração, backtest com origem móvel e métricas.

Todos os modelos são ajustados na base diária e preveem o logaritmo do índice em h = 1..252 pregões (cerca de
12 meses), devolvendo média e desvio-padrão (distribuição Normal). Isso permite comparar erro de ponto (RMSE) e
qualidade da incerteza (CRPS) entre ARIMA, SARIMA, SARIMAX, ARCH e GARCH, e contra passeios aleatórios.

Protocolo: a configuração é escolhida só com dados até a primeira origem do teste (AICc para a família
ARIMA, BIC para ARCH/GARCH); depois é reestimada a cada origem (uma por mês), sem ver o futuro. Execução offline:
    python -m plataforma.modelos
"""
import hashlib
import itertools
import json
import time
import warnings
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller

from plataforma import dados

VERSAO = 2
DIAS_MES = 21           # pregões por mês
HORIZONTE = 12 * DIAS_MES  # 252 pregões
PASSO = DIAS_MES        # uma origem de teste por mês
N_ORIGENS = 60
HIST_DIAS = 3 * 252     # histórico mostrado nos gráficos de previsão
SAZONAL = 5             # período sazonal: 5 pregões = uma semana
SERIES = ("Índice Memória", "Índice GPU")
FAMILIAS = ("arima", "sarima", "sarimax", "arch", "garch")
BASELINES = ("rw", "rw_deriva")
NOMES = {"arima": "ARIMA", "sarima": "SARIMA", "sarimax": "SARIMAX", "arch": "ARCH", "garch": "GARCH",
         "rw": "Passeio aleatório", "rw_deriva": "Passeio aleatório com deriva"}
ATRASOS = (0, 1, 2, 3)
TIPOS_PADRAO = ("Modelo",)
DISTRIBUICOES = {"normal": "Normal", "t": "t de Student"}
ARQ_RESULTADOS = dados.DIR_DADOS / "modelos.json"


# ------------------------------------------------------------------------- dados
@dataclass
class Serie:
    nome: str
    y: pd.Series            # log do índice, base diária (pregões)
    eventos: pd.DataFrame

    @property
    def n(self):
        return len(self.y)

    def retornos_diarios(self, ate=None):
        r = self.y.diff().dropna() * 100
        return r if ate is None else r[r.index <= ate]


def preparar(painel, nome, eventos=None):
    """Calendário de dias úteis (252 por ano) para todos os índices; feriados repetem o último valor (retorno zero)."""
    nivel = painel[nome].dropna()
    y = np.log(nivel).asfreq("B").ffill()
    return Serie(nome, y, dados.carregar_eventos() if eventos is None else eventos)


def posicoes_origem(serie):
    """Posições das últimas N_ORIGENS origens (uma a cada PASSO pregões) que ainda têm 252 pregões realizados à frente."""
    ultimo = serie.n - 1 - HORIZONTE
    return list(range(ultimo - PASSO * (N_ORIGENS - 1), ultimo + 1, PASSO))


def datas_futuras(serie, passos=HORIZONTE):
    return pd.bdate_range(serie.y.index[-1] + pd.Timedelta(days=1), periods=passos)


def ordens_diferenciacao(y, s):
    """d pelo ADF e D pela força da sazonalidade de período s (D=1 se Fs > 0,64, regra de Wang, Smith e Hyndman)."""
    d, x = 0, y.copy()
    while adfuller(x.dropna(), result_object=False)[1] > 0.05 and d < 2:
        x, d = x.diff(), d + 1
    stl = STL(y.reset_index(drop=True), period=s, robust=True).fit()
    forca = max(0.0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
    return d, int(forca > 0.64), float(forca)


# -------------------------------------------------- regressores de lançamentos de IA
def exog_matrizes(serie, tipos, atrasos, pos, taxa=None):
    """Contagem acumulada de lançamentos por pregão, defasada. Com d=1, o coeficiente vira o efeito de 1 lançamento
    ocorrido k pregões antes sobre o retorno do dia. Lançamentos fora de pregão contam no pregão seguinte.
    Lançamentos futuros usam `taxa` (média por pregão até a origem)."""
    idx = serie.y.index[: pos + 1]
    ev = serie.eventos[serie.eventos["categoria"].isin(tipos) & (serie.eventos["data"] <= idx[-1])]
    lugar = idx.searchsorted(pd.DatetimeIndex(ev["data"]).normalize())
    cont = np.bincount(lugar, minlength=len(idx)).astype(float)
    taxa = float(cont.mean()) if taxa is None else float(taxa)
    acum = np.cumsum(np.r_[cont, np.full(HORIZONTE, taxa)])
    X = pd.DataFrame({f"lanc_t-{k}": np.r_[np.zeros(k), acum[: len(acum) - k]] for k in atrasos})
    return X.iloc[: pos + 1].reset_index(drop=True), X.iloc[pos + 1:].reset_index(drop=True), taxa


# -------------------------------------------------------------------- ajuste e previsão
def _ajustar_arima(y, cfg, X=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = ARIMA(np.asarray(y), exog=X, order=tuple(cfg["order"]), seasonal_order=tuple(cfg["seasonal"]), trend=cfg["trend"])
        return mod.fit()


def _ajustar_vol(r, familia, cfg):
    kw = dict(vol="ARCH", p=cfg["choques"]) if familia == "arch" else dict(vol="GARCH", p=cfg["choques"], q=cfg["vol"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return arch_model(r, mean="Constant", dist=cfg["dist"], **kw).fit(disp="off", show_warning=False)


def _passeio(y, deriva):
    d = np.diff(np.asarray(y))
    h = np.arange(1, HORIZONTE + 1)
    return y.iloc[-1] + (d.mean() if deriva else 0.0) * h, d.std(ddof=1) * np.sqrt(h)


def prever(serie, familia, cfg, pos, taxa=None):
    """Ajusta com dados até `pos` e prevê h = 1..252 pregões. Devolve (média, desvio-padrão, ajuste) do log do índice."""
    y = serie.y.iloc[: pos + 1]
    if familia in BASELINES:
        mu, sd = _passeio(y, familia == "rw_deriva")
        return mu, sd, None
    if familia in ("arima", "sarima", "sarimax"):
        X = Xf = None
        if familia == "sarimax":
            X, Xf, _ = exog_matrizes(serie, cfg["tipos"], cfg["atrasos"], pos, taxa)
            Xf = Xf.values
        res = _ajustar_arima(y.values, cfg, X)
        f = res.get_forecast(HORIZONTE, exog=Xf)
        return np.asarray(f.predicted_mean), np.asarray(f.se_mean), res
    ate = y.index[-1]
    res = _ajustar_vol(serie.retornos_diarios(ate), familia, cfg)
    var = res.forecast(horizon=HORIZONTE, reindex=False).variance.values[-1] / 1e4
    return y.iloc[-1] + res.params["mu"] / 100 * np.arange(1, HORIZONTE + 1), np.sqrt(np.cumsum(var)), res


def backtest(serie, familia, cfg=None, posicoes=None, progresso=None, taxa=None):
    """Previsões de cada origem. Se um ajuste falhar, usa o passeio aleatório naquela origem."""
    posicoes = posicoes or posicoes_origem(serie)
    mu, sd = np.zeros((len(posicoes), HORIZONTE)), np.zeros((len(posicoes), HORIZONTE))
    falhas = 0
    for i, p in enumerate(posicoes):
        try:
            mu[i], sd[i], _ = prever(serie, familia, cfg, p, taxa)
        except Exception:
            mu[i], sd[i] = _passeio(serie.y.iloc[: p + 1], False)
            falhas += 1
        if progresso:
            progresso((i + 1) / len(posicoes))
    return mu, sd, falhas


# ----------------------------------------------------------------- busca da melhor configuração
def _grade(familia, d, D):
    tendencias = ("n", "t") if d + D == 1 else ("n",)
    if familia == "arima":
        for p, q in itertools.product(range(4), range(4)):
            for tr in tendencias:
                yield {"order": [p, d, q], "seasonal": [0, 0, 0, 0], "trend": tr}
    else:
        for p, q in itertools.product(range(3), range(3)):
            for P, Q in ((1, 0), (0, 1), (1, 1)):
                for tr in tendencias:
                    yield {"order": [p, d, q], "seasonal": [P, D, Q, SAZONAL], "trend": tr}


def _grade_vol(familia):
    for dist in DISTRIBUICOES:
        if familia == "arch":
            for q in range(1, 11):
                yield {"choques": q, "dist": dist}
        else:
            for p, q in itertools.product((1, 2), (1, 2)):
                yield {"vol": p, "choques": q, "dist": dist}


def selecionar(serie, familia, pos, base=None, progresso=None):
    """Melhor configuração usando só dados até `pos`. AICc (ARIMA, SARIMA, SARIMAX) ou BIC (ARCH, GARCH)."""
    y = serie.y.iloc[: pos + 1]
    d, D, forca = ordens_diferenciacao(y, SAZONAL)
    if familia in ("arima", "sarima"):
        candidatos, criterio = [cfg for cfg in _grade(familia, d, D)], "AICc"
    elif familia == "sarimax":
        base = base or selecionar(serie, "sarima", pos)["config"]
        candidatos = [{**base, "atrasos": list(c), "tipos": list(TIPOS_PADRAO)}
                      for r in range(1, len(ATRASOS) + 1) for c in itertools.combinations(ATRASOS, r)]
        criterio = "AICc"
    else:
        candidatos, criterio = list(_grade_vol(familia)), "BIC"
    r = serie.retornos_diarios(y.index[-1]) if familia in ("arch", "garch") else None
    notas = []
    for i, cfg in enumerate(candidatos):
        try:
            if r is not None:
                nota = float(_ajustar_vol(r, familia, cfg).bic)
            else:
                X = exog_matrizes(serie, cfg["tipos"], cfg["atrasos"], pos)[0] if familia == "sarimax" else None
                nota = float(_ajustar_arima(y.values, cfg, X).aicc)
            if np.isfinite(nota):
                notas.append((nota, cfg))
        except Exception:
            pass
        if progresso:
            progresso((i + 1) / len(candidatos))
    notas.sort(key=lambda t: t[0])
    return {"config": notas[0][1], "criterio": criterio, "valor": notas[0][0], "n_candidatos": len(notas),
            "top": [{"config": c, "valor": v} for v, c in notas[:8]],
            "diferenciacao": {"d": d, "D": D, "forca_sazonal": forca}}


def descrever(familia, cfg):
    if familia in ("arima", "sarima", "sarimax"):
        p, d, q = cfg["order"]
        P, D, Q, s = cfg["seasonal"]
        txt = f"{NOMES[familia]}({p},{d},{q})" + (f"({P},{D},{Q},{s})" if familia != "arima" else "")
        if cfg["trend"] == "t":
            txt += " com deriva"
        if familia == "sarimax":
            txt += " + lançamentos de IA (" + ", ".join(f"t-{k}" if k else "t" for k in cfg["atrasos"]) + ")"
        return txt
    dist = DISTRIBUICOES[cfg["dist"]]
    if familia == "arch":
        return f"ARCH({cfg['choques']}) · {dist}"
    return f"GARCH({cfg['vol']},{cfg['choques']}) · {dist}"


# ------------------------------------------------------------------------- métricas
def reais(serie, posicoes):
    y = serie.y.values
    return np.array([[y[p + h] for h in range(1, HORIZONTE + 1)] for p in posicoes])


def metricas(real, mu, sd):
    e = real - mu
    z = e / sd
    crps = sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    por_h = pd.DataFrame({"rmse": np.sqrt((e ** 2).mean(axis=0)), "mae": np.abs(e).mean(axis=0),
                          "crps": crps.mean(axis=0), "cobertura": (np.abs(z) <= 1.96).mean(axis=0)},
                         index=range(1, HORIZONTE + 1))
    geral = {"rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()), "crps": float(crps.mean()),
             "cobertura": float((np.abs(z) <= 1.96).mean()), "rmse_12": float(por_h.loc[HORIZONTE, "rmse"]),
             "vies": float(e.mean())}
    return geral, por_h


def diebold_mariano(real, mu_a, mu_b):
    """Compara o erro quadrático (média dos 252 horizontes) por origem. H0: mesma precisão.
    As origens têm 12 meses de previsão e estão a 1 mês uma da outra, então o erro de origens vizinhas se sobrepõe:
    variância robusta (Newey-West, 11 defasagens) e correção de Harvey-Leybourne-Newbold; aproximado com poucas origens."""
    d = ((real - mu_a) ** 2).mean(axis=1) - ((real - mu_b) ** 2).mean(axis=1)
    h = HORIZONTE // PASSO
    n, lag = len(d), h - 1
    dc = d - d.mean()
    gama = [np.sum(dc[j:] * dc[: n - j]) / n for j in range(lag + 1)]
    var = gama[0] + 2 * sum((1 - j / (lag + 1)) * gama[j] for j in range(1, lag + 1))
    if var <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / np.sqrt(var / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), n - 1))


def comparar(serie, entradas):
    """Tabela de métricas dos modelos e dos passeios aleatórios no mesmo trecho de teste."""
    pos = posicoes_origem(serie)
    real = reais(serie, pos)
    previsoes = {f: (np.array(e["mu"]), np.array(e["sd"])) for f, e in entradas.items() if e}
    for b in BASELINES:
        previsoes[b] = backtest(serie, b, None, pos)[:2]
    linhas, por_h = {}, {}
    for fam, (mu, sd) in previsoes.items():
        geral, ph = metricas(real, mu, sd)
        linhas[fam], por_h[fam] = geral, ph
    tab = pd.DataFrame(linhas).T
    melhor_base = min(BASELINES, key=lambda b: tab.loc[b, "rmse"])
    tab["dm_p"] = [diebold_mariano(real, previsoes[f][0], previsoes[melhor_base][0])[1] if f != melhor_base else np.nan for f in tab.index]
    tab["vs_base"] = tab["rmse"] / tab.loc[melhor_base, "rmse"] - 1
    tab.insert(0, "nome", [NOMES[f] for f in tab.index])
    return tab, por_h, melhor_base, real, previsoes


# ----------------------------------------------------------------- resultados guardados
def assinatura(serie, familia):
    s = {"versao": VERSAO, "ate": str(serie.y.index[-1].date()), "n": int(serie.n), "horizonte": HORIZONTE, "origens": N_ORIGENS, "passo": PASSO}
    if familia == "sarimax":
        s["eventos"] = hashlib.md5(pd.util.hash_pandas_object(serie.eventos, index=False).values.tobytes()).hexdigest()[:12]
    return s


def calcular_entrada(serie, familia, progresso=None, cfg=None, base=None):
    """Seleciona (se `cfg` não vier), faz o backtest e a previsão final de uma família para uma série."""
    pos = posicoes_origem(serie)
    sel = None
    if cfg is None:
        sel = selecionar(serie, familia, pos[0], base, (lambda f: progresso("Escolhendo a configuração", f * 0.4)) if progresso else None)
        cfg = sel["config"]
    mu, sd, falhas = backtest(serie, familia, cfg, pos, (lambda f: progresso("Testando nos últimos 5 anos", 0.4 + f * 0.55)) if progresso else None)
    fmu, fsd, _ = prever(serie, familia, cfg, serie.n - 1)
    return {"config": cfg, "selecao": sel, "corte": str(serie.y.index[pos[0]].date()), "origens": [str(serie.y.index[p].date()) for p in pos],
            "mu": np.round(mu, 6).tolist(), "sd": np.round(sd, 6).tolist(), "falhas": falhas,
            "final": {"mu": np.round(fmu, 6).tolist(), "sd": np.round(fsd, 6).tolist()}, "assinatura": assinatura(serie, familia)}


def carregar_guardado():
    if not ARQ_RESULTADOS.exists():
        return {}
    return json.loads(ARQ_RESULTADOS.read_text(encoding="utf-8")).get("series", {})


def entrada_guardada(serie, familia):
    e = carregar_guardado().get(serie.nome, {}).get(familia)
    return e if e and e["assinatura"] == assinatura(serie, familia) else None


# ----------------------------------------------------------------------- diagnóstico
def diagnostico(serie, familia, cfg, taxa=None):
    """Ajuste com todos os dados: coeficientes, resíduos e testes. Para ARCH/GARCH, volatilidade condicional."""
    if familia in ("arima", "sarima", "sarimax"):
        X = exog_matrizes(serie, cfg["tipos"], cfg["atrasos"], serie.n - 1, taxa)[0] if familia == "sarimax" else None
        res = _ajustar_arima(serie.y.values, cfg, X)
        queima = cfg["order"][1] + cfg["seasonal"][1] * cfg["seasonal"][3]
        resid = pd.Series(np.asarray(res.resid), index=serie.y.index).iloc[max(queima, 1):]
        k = cfg["order"][0] + cfg["order"][2] + cfg["seasonal"][0] + cfg["seasonal"][2]
        lb = acorr_ljungbox(resid, lags=[max(DIAS_MES, k + 1)], model_df=k)["lb_pvalue"].iloc[0]
        nomes = list(res.param_names)
        tab = pd.DataFrame({"coef": np.asarray(res.params), "erro padrão": np.asarray(res.bse), "p-valor": np.asarray(res.pvalues)}, index=nomes)
        return {"params": tab, "resid": resid, "acf": acf(resid, nlags=30, fft=True), "lb_p": float(lb),
                "aicc": float(res.aicc), "aic": float(res.aic), "bic": float(res.bic)}
    r = serie.retornos_diarios()
    res = _ajustar_vol(r, familia, cfg)
    pad = res.std_resid.dropna()
    tab = pd.DataFrame({"coef": res.params, "erro padrão": res.std_err, "p-valor": res.pvalues})
    var = res.forecast(horizon=HORIZONTE, reindex=False).variance.values[-1]
    vol_mes = [float(np.sqrt(var[k * DIAS_MES:(k + 1) * DIAS_MES].mean() * 252)) for k in range(12)]
    return {"params": tab, "resid": pad, "lb_p": float(acorr_ljungbox(pad, lags=[10])["lb_pvalue"].iloc[0]),
            "lb2_p": float(acorr_ljungbox(pad ** 2, lags=[10])["lb_pvalue"].iloc[0]), "archlm_p": float(het_arch(pad, nlags=10)[1]),
            "vol_anual": res.conditional_volatility * np.sqrt(252), "vol_prevista": vol_mes,
            "aic": float(res.aic), "bic": float(res.bic)}


# ------------------------------------------------------------------------------- CLI
def main():
    painel, _ = dados.carregar()
    eventos = dados.carregar_eventos()
    saida = {"versao": VERSAO, "gerado_em": date.today().isoformat(), "series": {}}
    for nome in SERIES:
        serie = preparar(painel, nome, eventos)
        saida["series"][nome] = {}
        base = None
        for fam in FAMILIAS:
            t = time.time()
            e = calcular_entrada(serie, fam, base=base)
            if fam == "sarima":
                base = e["selecao"]["config"]
            saida["series"][nome][fam] = e
            print(f"{nome:16s} {fam:8s} {descrever(fam, e['config']):55s} {e['selecao']['criterio']} {e['selecao']['valor']:.1f}  falhas={e['falhas']}  {time.time() - t:.0f}s", flush=True)
    ARQ_RESULTADOS.write_text(json.dumps(saida, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("salvo em", ARQ_RESULTADOS)


if __name__ == "__main__":
    main()
