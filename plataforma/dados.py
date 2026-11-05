"""Catálogo, download, cache e carga das séries usadas na plataforma."""
from pathlib import Path
import io
import re

import numpy as np
import pandas as pd
import yfinance as yf

RAIZ = Path(__file__).resolve().parent.parent
DIR_DADOS = RAIZ / "data"
DIR_EXTRAS = DIR_DADOS / "extras"
ARQ_PRECOS = DIR_DADOS / "precos.csv"
ARQ_EVENTOS = DIR_DADOS / "eventos_ia.csv"

DATA_INICIO = "2012-01-02"
LIMITE_UPLOAD_BYTES = 5 * 1024 * 1024

# ticker -> (nome, categoria)
CATALOGO = {
    "MU":        ("Micron", "Memória"),
    "000660.KS": ("SK Hynix", "Memória"),
    "005930.KS": ("Samsung", "Memória"),
    "2408.TW":   ("Nanya", "Memória"),
    "NVDA":      ("NVIDIA", "GPU / IA"),
    "AMD":       ("AMD", "GPU / IA"),
    "TSM":       ("TSMC", "Semicondutores"),
    "INTC":      ("Intel", "Semicondutores"),
    "AVGO":      ("Broadcom", "Semicondutores"),
    "HG=F":      ("Cobre", "Commodities"),
    "GC=F":      ("Ouro", "Commodities"),
    "SI=F":      ("Prata", "Commodities"),
    "^SOX":      ("Índice SOX (semicondutores)", "Mercado"),
    "^IXIC":     ("Nasdaq Composite", "Mercado"),
    "^GSPC":     ("S&P 500", "Mercado"),
    "KRW=X":     ("USD/KRW", "Câmbio"),
    "TWD=X":     ("USD/TWD", "Câmbio"),
    "^TNX":      ("Treasury 10 anos (juros)", "Juros / risco"),
    "^VIX":      ("VIX (volatilidade)", "Juros / risco"),
}

GRUPOS_INDICE = {
    "Índice Memória": ["MU", "000660.KS", "005930.KS", "2408.TW"],
    "Índice GPU": ["NVDA", "AMD", "TSM", "INTC", "AVGO"],
    "Índice GPU (NVDA + AMD)": ["NVDA", "AMD"],
}

CAT_INDICES = "Índices do setor"
CAT_PROPRIAS = "Dados próprios"


def composicao(nome_indice):
    return ", ".join(CATALOGO[t][0] for t in GRUPOS_INDICE[nome_indice])


def rotulo(ticker):
    nome, _ = CATALOGO[ticker]
    return f"{nome} ({ticker})"


def baixar(inicio=DATA_INICIO):
    """Baixa preços ajustados do Yahoo Finance e grava o cache em CSV."""
    tickers = list(CATALOGO)
    bruto = yf.download(tickers, start=inicio, interval="1d", auto_adjust=False,
                        progress=False, threads=True)
    px = bruto["Adj Close"].fillna(bruto["Close"])
    px = px.dropna(how="all")
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px.index.name = "date"
    DIR_DADOS.mkdir(exist_ok=True)
    px.to_csv(ARQ_PRECOS)
    return px


def _ler_extras():
    series = {}
    if DIR_EXTRAS.exists():
        for arq in sorted(DIR_EXTRAS.glob("*.csv")):
            d = pd.read_csv(arq, parse_dates=["date"], index_col="date")
            series[arq.stem] = d["value"]
    return pd.DataFrame(series)


def indices_setor(px):
    """Média das séries reescaladas para 100 na primeira data em que todas existem."""
    saida = {}
    for nome, tickers in GRUPOS_INDICE.items():
        cols = [t for t in tickers if t in px.columns]
        if len(cols) != len(tickers):
            continue
        p = px[cols].dropna()
        if p.empty:
            continue
        saida[nome] = (p / p.iloc[0] * 100).mean(axis=1)
    return pd.DataFrame(saida)


def carregar():
    """Retorna (painel, categorias): painel tem uma coluna por série, rotulada; categorias mapeia coluna -> categoria."""
    if not ARQ_PRECOS.exists():
        return None, {}
    px = pd.read_csv(ARQ_PRECOS, index_col="date", parse_dates=True)
    idx = indices_setor(px)
    extras = _ler_extras()

    painel = px.rename(columns={t: rotulo(t) for t in px.columns if t in CATALOGO})
    categorias = {rotulo(t): CATALOGO[t][1] for t in px.columns if t in CATALOGO}
    for nome in idx.columns:
        categorias[nome] = CAT_INDICES
    for nome in extras.columns:
        categorias[nome] = CAT_PROPRIAS

    painel = pd.concat([idx, painel, extras], axis=1).sort_index()
    return painel, categorias


def versao_cache():
    """Muda quando os arquivos de dados mudam, para invalidar o cache do Streamlit."""
    arqs = [ARQ_PRECOS, *sorted(DIR_EXTRAS.glob("*.csv"))] if DIR_EXTRAS.exists() else [ARQ_PRECOS]
    return tuple((a.name, a.stat().st_mtime_ns) for a in arqs if a.exists())


def carregar_eventos():
    ev = pd.read_csv(ARQ_EVENTOS, parse_dates=["data"])
    return ev.sort_values("data").reset_index(drop=True)


def salvar_extra(nome, conteudo):
    """Valida e grava uma série enviada pelo usuário (2 colunas: data, valor)."""
    if len(conteudo) > LIMITE_UPLOAD_BYTES:
        raise ValueError("Arquivo maior que 5 MB.")
    nome = re.sub(r"[^\w\- ]", "", nome, flags=re.UNICODE).strip()[:40]
    if not nome:
        raise ValueError("Nome inválido.")
    if nome in {rotulo(t) for t in CATALOGO} | set(GRUPOS_INDICE):
        raise ValueError("Esse nome já é usado por uma série do catálogo.")

    d = pd.read_csv(io.BytesIO(conteudo), sep=None, engine="python")
    if d.shape[1] < 2:
        raise ValueError("O arquivo precisa ter ao menos duas colunas: data e valor.")
    datas = pd.to_datetime(d.iloc[:, 0], errors="coerce", dayfirst=False)
    valores = d.iloc[:, 1]
    if valores.dtype == object:
        valores = valores.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    valores = pd.to_numeric(valores, errors="coerce")

    serie = pd.Series(valores.values, index=datas, name="value").dropna()
    serie = serie[serie.index.notna()].sort_index()
    serie = serie[~serie.index.duplicated(keep="last")]
    if len(serie) < 24:
        raise ValueError("Poucos pontos válidos (mínimo 24). Confira o formato de data e número.")

    DIR_EXTRAS.mkdir(parents=True, exist_ok=True)
    serie.rename_axis("date").to_csv(DIR_EXTRAS / f"{nome}.csv")
    return nome, len(serie)


def reamostrar(df, freq):
    """freq: 'D', 'W' ou 'M'. Usa o último valor do período."""
    if freq == "D":
        return df
    regra = {"W": "W-FRI", "M": "ME"}[freq]
    out = df.resample(regra).last().dropna(how="all")
    return out[out.index <= df.index.max()]  # descarta o período ainda incompleto


if __name__ == "__main__":
    p = baixar()
    print(f"{p.shape[1]} séries | {p.index.min().date()} a {p.index.max().date()} | {ARQ_PRECOS}")
