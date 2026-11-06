"""Construtores de gráficos Plotly. Cada função devolve uma Figure já estilizada; o hover explica os valores."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plataforma import analises as an
from plataforma.leitura import pct
from plataforma.modelos import DIAS_MES, HORIZONTE, NOMES
from plataforma.visual import COR, COR_MODELO, estilizar

MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
COR_EVENTO = {"Modelo": "#D9A441", "Hardware": "#E2664B", "Mercado": "#A58BE0"}
ESCALA_CORR = [[0, COR["mem"]], [0.5, COR["mask2"]], [1, COR["gold"]]]
ESCALA_RETORNO = [[0, COR["gpu"]], [0.5, COR["mask2"]], [1, COR["ok"]]]


def mes_ano(ts):
    return f"{MESES[ts.month - 1]}/{ts.year}"


def rgba(hex_cor, alfa):
    h = hex_cor.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alfa})"


def eixo_log(fig, vmin, vmax):
    """Eixo y logarítmico com rótulos 1-2-5 completos (1.000, 2.000, 5.000...)."""
    k0, k1 = int(np.floor(np.log10(vmin))), int(np.ceil(np.log10(vmax)))
    ticks = [m * 10.0 ** k for k in range(k0, k1 + 1) for m in (1, 2, 5) if vmin * .8 <= m * 10.0 ** k <= vmax * 1.25]
    texto = [f"{v:,.0f}".replace(",", ".") if v >= 1 else f"{v:g}".replace(".", ",") for v in ticks]
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=texto)


def _linha_vertical(fig, x, texto, cor=None):
    x = pd.Timestamp(x).strftime("%Y-%m-%d")
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                  line=dict(color=cor or COR["gold"], width=1.5, dash="dash"))
    fig.add_annotation(x=x, y=1, yref="paper", text=texto, showarrow=False, xanchor="left", yanchor="top",
                       font=dict(size=11, color=cor or COR["gold"]), bgcolor="rgba(12,27,21,.7)")


def marcar_eventos(fig, s, eventos):
    """Pontos dourados sobre a série nas datas dos eventos; o hover mostra o que aconteceu."""
    ev = eventos[(eventos["data"] >= s.index[0]) & (eventos["data"] <= s.index[-1])]
    if ev.empty:
        return fig
    fig.add_scatter(
        x=ev["data"], y=[s.asof(d) for d in ev["data"]], mode="markers", name="Eventos de IA",
        marker=dict(size=9, color=COR["gold"], line=dict(width=1, color=COR["pcb"])),
        text=ev["evento"], customdata=ev[["empresa", "categoria"]].values,
        hovertemplate="<b>%{text}</b><br>%{customdata[0]} · %{customdata[1]}<extra></extra>")
    return fig


# ------------------------------------------------------------------ comparação
def linhas(df, cores, titulo, subtitulo=None, base100=False, log=False, eventos=None, marco=None, **est):
    fig = go.Figure()
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        destaque = c in ("Índice Memória", "Índice GPU")
        if base100:
            custom = (s / s.iloc[0] - 1).mul(100).values
            hover = "<b>%{fullData.name}</b>: %{y:,.0f} <i>(%{customdata:+,.0f}% desde a base)</i><extra></extra>"
        else:
            custom, hover = None, "<b>%{fullData.name}</b>: %{y:,.2f}<extra></extra>"
        fig.add_scatter(x=s.index, y=s.values, name=c, mode="lines", customdata=custom, hovertemplate=hover,
                        line=dict(color=cores.get(c), width=3 if destaque else 1.8))
    if eventos is not None and len(df.columns):
        marcar_eventos(fig, df[df.columns[0]].dropna(), eventos)
    if marco is not None:
        _linha_vertical(fig, *marco)
    if log:
        valores = pd.concat([df[c].dropna() for c in df.columns])
        eixo_log(fig, valores.min(), valores.max())
    est.setdefault("altura", 460)
    return estilizar(fig, titulo, subtitulo, **est)


def mapa_correlacao(corr, titulo, subtitulo=None, **est):
    z = corr.values
    fig = go.Figure(go.Heatmap(
        z=z, x=list(corr.columns), y=list(corr.index), zmin=-1, zmax=1, colorscale=ESCALA_CORR,
        text=np.round(z, 2), texttemplate="%{text:.2f}", colorbar=dict(title="corr.", thickness=12),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>correlação %{z:+.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    est.setdefault("altura", 480)
    return estilizar(fig, titulo, subtitulo, hover="closest", datas=False, legenda=False, **est)


def barras_correlacao(ranking, alvo, titulo, subtitulo=None, **est):
    r = ranking.sort_values("correlação")
    fig = go.Figure(go.Bar(
        x=r["correlação"], y=list(r.index), orientation="h",
        marker_color=[COR["gold"] if v >= 0 else COR["mem"] for v in r["correlação"]],
        customdata=r["observações"].values,
        hovertemplate="<b>%{y}</b><br>correlação %{x:+.2f} com " + alvo + "<br>%{customdata} períodos<extra></extra>"))
    fig.update_xaxes(range=[-1, 1], zeroline=True)
    est.setdefault("altura", max(320, 30 * len(r) + 130))
    return estilizar(fig, titulo, subtitulo, hover="closest", datas=False, legenda=False, **est)


def correlacao_movel(cm, a, b, titulo, subtitulo=None, **est):
    fig = go.Figure(go.Scatter(
        x=cm.index, y=cm.values, mode="lines", name="correlação", line=dict(color=COR["gold"], width=2.4),
        hovertemplate="correlação %{y:+.2f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dot", line_color=COR["dim"])
    fig.update_yaxes(range=[-1, 1])
    est.setdefault("altura", 340)
    return estilizar(fig, titulo, subtitulo, legenda=False, **est)


def dispersao(reg, nome_x, nome_y, titulo, subtitulo=None, **est):
    j = reg["dados"]
    rotulos = [mes_ano(d) for d in j.index]
    fig = go.Figure(go.Scatter(
        x=j["x"], y=j["y"], mode="markers", name="meses", text=rotulos,
        marker=dict(size=8, color=COR["mem"], opacity=.75, line=dict(width=.5, color=COR["pcb"])),
        hovertemplate="<b>%{text}</b><br>" + nome_x + ": %{x:+.1f}%<br>" + nome_y + ": %{y:+.1f}%<extra></extra>"))
    xs = np.array([j["x"].min(), j["x"].max()])
    fig.add_scatter(x=xs, y=reg["a"] + reg["b"] * xs, mode="lines", name="reta ajustada",
                    line=dict(color=COR["gpu"], width=2.5),
                    hovertemplate=f"reta: {nome_y} = {reg['a']:.2f} + {reg['b']:.2f} × {nome_x}<extra></extra>")
    fig.add_hline(y=0, line_color=COR["linha"], line_width=1)
    fig.add_vline(x=0, line_color=COR["linha"], line_width=1)
    est.setdefault("altura", 440)
    return estilizar(fig, titulo, subtitulo, x_titulo=f"Retorno mensal: {nome_x} (%)", y_titulo=f"{nome_y} (%)",
                     hover="closest", datas=False, legenda=False, **est)


# ----------------------------------------------------------------------- ciclos
def drawdowns(series, cores, titulo, subtitulo=None, **est):
    fig = go.Figure()
    for nome, s in series.items():
        s = s.dropna()
        dd = an.drawdown(s)
        pico = an.data_do_pico(s).dt.strftime("%d/%m/%Y")
        cor = cores.get(nome, COR["mem"])
        fig.add_scatter(x=dd.index, y=dd.values, name=nome, mode="lines", line=dict(color=cor, width=1.8),
                        fill="tozeroy", fillcolor=rgba(cor, .22), customdata=pico.values,
                        hovertemplate="<b>%{fullData.name}</b>: %{y:.1%} abaixo do pico de %{customdata}<extra></extra>")
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    est.setdefault("altura", 420)
    return estilizar(fig, titulo, subtitulo, **est)


def mapa_sazonal(tab, titulo, subtitulo=None, **est):
    fig = go.Figure(go.Heatmap(
        z=tab.values, x=MESES, y=[str(a) for a in tab.index], zmin=-25, zmax=25, colorscale=ESCALA_RETORNO,
        text=[["" if np.isnan(v) else f"{v:.0f}" for v in linha] for linha in tab.values], texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title="%", thickness=12), xgap=2, ygap=2,
        hovertemplate="<b>%{x}/%{y}</b><br>retorno do mês: %{z:+.1f}%<extra></extra>"))
    fig.update_yaxes(autorange="reversed", type="category")
    est.setdefault("altura", 520)
    return estilizar(fig, titulo, subtitulo, hover="closest", datas=False, legenda=False, **est)


def barras_mes(tab, titulo, subtitulo=None, **est):
    media = tab.mean()
    positivos = (tab > 0).sum() / tab.notna().sum() * 100
    cores = [COR["ok"] if v >= 0 else COR["gpu"] for v in media]
    fig = go.Figure(go.Bar(
        x=MESES, y=media.values, marker_color=cores, customdata=positivos.values,
        hovertemplate="<b>%{x}</b><br>retorno médio: %{y:+.1f}%<br>positivo em %{customdata:.0f}% dos anos<extra></extra>"))
    fig.add_hline(y=0, line_color=COR["dim"], line_width=1)
    est.setdefault("altura", 340)
    return estilizar(fig, titulo, subtitulo, hover="closest", datas=False, legenda=False, y_titulo="% médio", **est)


def decomposicao(dec, titulo, subtitulo=None, **est):
    comps = [("Série (log)", dec.observed, COR["silk"], "Valor em logaritmo: %{y:.2f}"),
             ("Tendência", dec.trend, COR["gold"], "Direção de longo prazo (log): %{y:.2f}"),
             ("Sazonalidade", dec.seasonal, COR["mem"], "Efeito do mês do ano: %{customdata:+.1f}%"),
             ("Resíduo", dec.resid, COR["dim"], "O que sobra: %{customdata:+.1f}%")]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.04)
    for i, (nome, s, cor, hover) in enumerate(comps, start=1):
        s = s.dropna()
        custom = (np.exp(s.values) - 1) * 100 if i >= 3 else None
        fig.add_scatter(x=s.index, y=s.values, name=nome, mode="lines", line=dict(color=cor, width=1.8),
                        customdata=custom, hovertemplate=f"<b>{nome}</b>: {hover}<extra></extra>", row=i, col=1)
        fig.update_yaxes(title_text=nome, row=i, col=1, title_font=dict(size=11))
    est.setdefault("altura", 640)
    return estilizar(fig, titulo, subtitulo, legenda=False, **est)


def patamar(s, corte, info, nome, cor, titulo, subtitulo=None, **est):
    antes = s[s.index <= corte]
    depois = s[s.index >= antes.index[-1]]
    fig = go.Figure()
    fig.add_scatter(x=antes.index, y=antes.values, name="Antes do marco", mode="lines",
                    line=dict(color=COR["dim"], width=2.2),
                    hovertemplate="<b>Antes do marco</b>: %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=depois.index, y=depois.values, name="Depois do marco", mode="lines",
                    line=dict(color=cor, width=3.2), hovertemplate="<b>Depois do marco</b>: %{y:,.0f}<extra></extra>")
    fig.add_vrect(x0=corte, x1=s.index[-1], fillcolor=cor, opacity=.06, line_width=0)
    if info:
        fig.add_hline(y=info["pico_anterior"], line_dash="dot", line_color=COR["gold"],
                      annotation_text=f"pico anterior ao marco ({mes_ano(info['data_pico'])})",
                      annotation_position="bottom right", annotation_font=dict(color=COR["gold"], size=11))
    _linha_vertical(fig, corte, f"marco: {pd.Timestamp(corte):%d/%m/%Y}")
    eixo_log(fig, s.min(), s.max())
    est.setdefault("altura", 460)
    return estilizar(fig, titulo, subtitulo, y_titulo=f"{nome} (base 100, escala log)", **est)


# ----------------------------------------------------------------------- eventos
def trajetoria_car(traj, resumo, depois, mostrar_todos=False, titulo="Retorno anormal acumulado médio", subtitulo=None, **est):
    media, ic = resumo["media"] * 100, resumo["ic"] * 100
    x = list(traj.columns)
    fig = go.Figure()
    if mostrar_todos:
        for _, linha in traj.iterrows():
            fig.add_scatter(x=x, y=linha.values * 100, mode="lines", showlegend=False, hoverinfo="skip",
                            line=dict(width=1, color=COR["dim"]), opacity=.35)
    fig.add_scatter(x=x, y=(media + ic).values, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=x, y=(media - ic).values, mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=rgba(COR["gold"], .25), name="IC 95%", hoverinfo="skip")
    fig.add_scatter(x=x, y=media.values, mode="lines+markers", name="CAR médio", customdata=np.c_[(media - ic).values, (media + ic).values],
                    line=dict(color=COR["gold"], width=3), marker=dict(size=6),
                    hovertemplate="<b>Dia %{x:+d}</b><br>CAR médio: %{y:+.2f}%<br>IC 95%: %{customdata[0]:+.2f}% a %{customdata[1]:+.2f}%<extra></extra>")
    fig.add_vline(x=0, line_dash="dot", line_color=COR["dim"])
    fig.add_hline(y=0, line_dash="dot", line_color=COR["dim"])
    est.setdefault("altura", 420)
    return estilizar(fig, titulo, subtitulo, x_titulo="Pregões em relação ao evento (0 = dia do evento)",
                     y_titulo="CAR (%)", datas=False, **est)


def barras_eventos(traj, meta, depois, ordem="data", titulo="Efeito de cada evento", subtitulo=None, **est):
    d = meta.copy()
    d["car"] = traj.iloc[:, -1].values * 100
    hover = ("<b>%{customdata[0]}</b><br>%{customdata[1]} · %{customdata[2]} · %{customdata[3]}"
             f"<br>CAR em +{depois} pregões: <b>%{{y:+.1f}}%</b><extra></extra>")
    fig = go.Figure()
    media = d["car"].mean()
    if ordem == "data":
        for cat, g in d.groupby("categoria"):
            fig.add_bar(x=g["data"], y=g["car"], name=cat, marker_color=COR_EVENTO.get(cat, COR["dim"]), width=1000 * 3600 * 24 * 14,
                        customdata=np.c_[g["evento"], g["empresa"], g["categoria"], g["data"].dt.strftime("%d/%m/%Y")],
                        hovertemplate=hover)
        fig.add_hline(y=media, line_dash="dot", line_color=COR["silk"],
                      annotation_text=f"média {pct(media, 1, True)}", annotation_position="top left",
                      annotation_font=dict(color=COR["silk"], size=11))
        fig.add_hline(y=0, line_color=COR["linha"], line_width=1)
        est.setdefault("altura", 420)
        return estilizar(fig, titulo, subtitulo, y_titulo=f"CAR em +{depois} pregões (%)", hover="closest", **est)
    d = d.sort_values("car")
    rot = d["evento"] + " (" + d["data"].dt.strftime("%m/%Y") + ")"
    cats = d["categoria"].tolist()
    fig.add_bar(x=d["car"], y=rot, orientation="h", marker_color=[COR_EVENTO.get(c, COR["dim"]) for c in cats],
                customdata=np.c_[d["evento"], d["empresa"], d["categoria"], d["data"].dt.strftime("%d/%m/%Y")],
                hovertemplate=hover.replace("%{y:+.1f}", "%{x:+.1f}"))
    fig.add_vline(x=0, line_color=COR["linha"], line_width=1)
    est.setdefault("altura", max(360, 24 * len(d) + 120))
    return estilizar(fig, titulo, subtitulo, x_titulo=f"CAR em +{depois} pregões (%)", hover="closest", datas=False,
                     legenda=False, **est)


# ----------------------------------------------------------------- volatilidade
def termometro(atual, percentil, serie_hist, titulo="Volatilidade agora", **est):
    q25, q75 = float(np.percentile(serie_hist, 25)), float(np.percentile(serie_hist, 75))
    lo, hi = float(serie_hist.min()), float(serie_hist.max())
    pct_txt = f"{percentil:.0f}%"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(atual), number=dict(suffix="% a.a.", valueformat=".0f", font=dict(size=40)),
        title=dict(text=f"<b>{titulo}</b><br><span style='font-size:13px;color:{COR['dim']}'>acima de {pct_txt} da história</span>"),
        gauge=dict(axis=dict(range=[lo, hi], tickcolor=COR["dim"]), bar=dict(color=COR["gold"], thickness=.28),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[lo, q25], color="#1B4A36"), dict(range=[q25, q75], color="#2A5B45"),
                          dict(range=[q75, hi], color="#6B3A2E")])))
    est.setdefault("altura", 320)
    estilizar(fig, hover="closest", datas=False, legenda=False, **est)
    fig.update_layout(margin=dict(l=30, r=30, t=70, b=10))
    return fig


def volatilidade(vols, periodos_ano, titulo, subtitulo=None, **est):
    fig = go.Figure()
    paleta = [COR["gold"], COR["mem"], COR["gpu"], "#A58BE0", COR["ok"]]
    for i, c in enumerate(vols.columns):
        v = vols[c].dropna()
        fig.add_scatter(x=v.index, y=v.values, name=c, mode="lines", line=dict(color=paleta[i % len(paleta)], width=1.8),
                        customdata=v.values / np.sqrt(periodos_ano),
                        hovertemplate="<b>%{fullData.name}</b>: %{y:.0f}% a.a. <i>(oscilação típica de ±%{customdata:.1f}% por período)</i><extra></extra>")
    est.setdefault("altura", 420)
    return estilizar(fig, titulo, subtitulo, y_titulo="% ao ano", **est)


def previsao_volatilidade(prev, titulo, subtitulo=None, **est):
    fig = go.Figure()
    paleta = [COR["gold"], COR["mem"], COR["gpu"], "#A58BE0", COR["ok"]]
    for i, c in enumerate(prev.columns):
        fig.add_scatter(x=prev.index, y=prev[c].values, name=c, mode="lines+markers", line=dict(color=paleta[i % len(paleta)], width=2.4),
                        hovertemplate="<b>%{fullData.name}</b>: %{y:.1f}% a.a.<extra></extra>")
    est.setdefault("altura", 340)
    return estilizar(fig, titulo, subtitulo, x_titulo="Períodos à frente", y_titulo="% ao ano", datas=False, **est)


# ------------------------------------------------------------ modelos de previsão
def leque(hist, datas, mu, sd, titulo, subtitulo=None, outros=None, **est):
    """Previsão em nível (exp do log) com faixas de 80% e 95%. `hist`, `mu` e `sd` estão em log.
    `outros`: lista de (nome, mu, cor) com a previsão central de outros modelos."""
    mu, sd = np.asarray(mu), np.asarray(sd)
    ultimo = float(hist.iloc[-1])
    x = [hist.index[-1], *datas]
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=np.exp(hist.values), name="Histórico", mode="lines", line=dict(color=COR["silk"], width=2.2),
                    hovertemplate="<b>Histórico</b>: %{y:,.0f}<extra></extra>")
    for z, alfa, nome in ((1.96, .16, "IC 95%"), (1.2816, .30, "IC 80%")):
        fig.add_scatter(x=x, y=np.exp(np.r_[ultimo, mu + z * sd]), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=x, y=np.exp(np.r_[ultimo, mu - z * sd]), mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor=rgba(COR["gold"], alfa), name=nome, hoverinfo="skip")
    for nome, m, cor in outros or []:
        fig.add_scatter(x=x, y=np.exp(np.r_[ultimo, np.asarray(m)]), name=nome, mode="lines", line=dict(color=cor, width=1.8, dash="dot"),
                        hovertemplate=f"<b>{nome}</b>: %{{y:,.0f}}<extra></extra>")
    centro = np.exp(np.r_[ultimo, mu])
    custom = np.c_[np.exp(np.r_[ultimo, mu - 1.96 * sd]), np.exp(np.r_[ultimo, mu + 1.96 * sd]), (centro / np.exp(ultimo) - 1) * 100]
    fig.add_scatter(x=x, y=centro, name="Previsão", mode="lines", line=dict(color=COR["gold"], width=3),
                    customdata=custom,
                    hovertemplate="<b>Previsão</b>: %{y:,.0f} (%{customdata[2]:+.0f}% em relação a hoje)<br>"
                                  "IC 95%: de %{customdata[0]:,.0f} a %{customdata[1]:,.0f}<extra></extra>")
    est.setdefault("altura", 440)
    estilizar(fig, titulo, subtitulo, **est)
    niveis = np.concatenate([np.exp(hist.values), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd)])
    eixo_log(fig, niveis.min(), niveis.max())
    return fig


def previsto_vs_realizado(y, pos, mu, sd, titulo, subtitulo=None, h=HORIZONTE, **est):
    """Para cada origem, a previsão feita h pregões antes colocada na data-alvo, contra o valor realizado."""
    meses = round(h / DIAS_MES)
    mu, sd = np.asarray(mu)[:, h - 1], np.asarray(sd)[:, h - 1]
    alvo = y.index[[p + h for p in pos]]
    real = np.exp(y.values[[p + h for p in pos]])
    origem = [f"{y.index[p]:%d/%m/%Y}" for p in pos]
    fig = go.Figure()
    fig.add_scatter(x=alvo, y=np.exp(mu + 1.96 * sd), mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=alvo, y=np.exp(mu - 1.96 * sd), mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor=rgba(COR["gold"], .18), name="IC 95%", hoverinfo="skip")
    fig.add_scatter(x=y.index[pos[0]:], y=np.exp(y.values[pos[0]:]), name="Realizado", mode="lines",
                    line=dict(color=COR["silk"], width=2.2), hovertemplate="<b>Realizado</b>: %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=alvo, y=np.exp(mu), name=f"Previsto {meses} meses antes", mode="lines", line=dict(color=COR["gold"], width=2.6),
                    customdata=np.c_[origem, real, (np.exp(mu) / real - 1) * 100],
                    hovertemplate=f"<b>Previsto há {meses} meses</b> (em %{{customdata[0]}}): %{{y:,.0f}}<br>"
                                  "Realizado: %{customdata[1]:,.0f} (erro de %{customdata[2]:+.0f}%)<extra></extra>")
    est.setdefault("altura", 400)
    estilizar(fig, titulo, subtitulo, **est)
    eixo_log(fig, min(real.min(), np.exp(mu - 1.96 * sd).min()), max(real.max(), np.exp(mu + 1.96 * sd).max()))
    return fig


def origem_explorer(y, p, mu, sd, titulo, subtitulo=None, **est):
    """Previsão feita na origem `p` (posição) com o que aconteceu depois, para ver um caso concreto."""
    hist = y.iloc[max(0, p - HORIZONTE): p + 1]
    datas = y.index[p + 1: p + 1 + len(mu)]
    fig = leque(hist, datas, mu, sd, titulo, subtitulo, **est)
    real = y.iloc[p: p + 1 + len(mu)]
    fig.add_scatter(x=real.index, y=np.exp(real.values), name="Realizado", mode="lines", line=dict(color=COR["silk"], width=2.2, dash="dash"),
                    hovertemplate="<b>Realizado</b>: %{y:,.0f}<extra></extra>")
    return fig


def erro_por_horizonte(por_h, destaque, titulo, subtitulo=None, **est):
    """Erro típico (%) por horizonte de cada modelo. `por_h`: {família: DataFrame com coluna rmse}."""
    fig = go.Figure()
    for fam, tab in por_h.items():
        base = fam in ("rw", "rw_deriva")
        grosso = fam == destaque
        fig.add_scatter(x=list(tab.index / DIAS_MES), y=(np.exp(tab["rmse"].values) - 1) * 100, name=NOMES[fam], mode="lines",
                        line=dict(color=COR_MODELO[fam], width=3.4 if grosso else 1.6, dash="dot" if base else "solid"), customdata=tab["rmse"].values,
                        hovertemplate=f"<b>{NOMES[fam]}</b>: erro típico de ±%{{y:.0f}}% (RMSE %{{customdata:.3f}} em log)<extra></extra>")
    est.setdefault("altura", 380)
    fig = estilizar(fig, titulo, subtitulo, x_titulo="Meses à frente", y_titulo="Erro típico (%)", datas=False, **est)
    fig.update_layout(legend=dict(y=-0.3))
    return fig.update_xaxes(dtick=1, hoverformat=".1f")


def barras_modelos(valores, titulo, subtitulo=None, melhor=None, formato="{:.3f}", x_titulo=None, **est):
    """Barras horizontais, com o menor valor (melhor) no topo. `valores`: Series indexada pela família."""
    d = valores.sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=d.values, y=[NOMES[f] for f in d.index], orientation="h", text=[formato.format(v) for v in d.values], textposition="outside",
        cliponaxis=False, marker=dict(color=[COR_MODELO[f] for f in d.index],
                                      line=dict(color=[COR["gold"] if f == melhor else "rgba(0,0,0,0)" for f in d.index], width=3)),
        hovertemplate="<b>%{y}</b>: %{text}<extra></extra>"))
    est.setdefault("altura", max(300, 46 * len(d) + 110))
    return estilizar(fig, titulo, subtitulo, x_titulo=x_titulo, hover="closest", datas=False, legenda=False, **est)


def acf_residuos(valores, n, titulo, subtitulo=None, **est):
    """Autocorrelação dos resíduos (sem a defasagem 0). Barras fora da faixa indicam padrão que o modelo não captou."""
    v = np.asarray(valores)[1:]
    lim = 1.96 / np.sqrt(n)
    fig = go.Figure(go.Bar(x=list(range(1, len(v) + 1)), y=v, marker_color=[COR["gpu"] if abs(a) > lim else COR["mem"] for a in v],
                           hovertemplate="defasagem %{x}: autocorrelação %{y:+.2f}<extra></extra>"))
    fig.add_hrect(y0=-lim, y1=lim, fillcolor=rgba(COR["dim"], .14), line_width=0)
    est.setdefault("altura", 300)
    return estilizar(fig, titulo, subtitulo, x_titulo="Defasagem (dias)", hover="closest", datas=False, legenda=False, **est)


# ------------------------------------------------------------------------ filme
def filme(df, eventos, cores, titulo="Memória × GPU, de 2012 até hoje", **est):
    """Animação mês a mês de índices base 100 (escala log); eventos de IA aparecem à medida que acontecem."""
    cols = list(df.columns)
    datas = df.index
    topo = df.max(axis=1)
    ev = eventos[(eventos["data"] >= datas[0]) & (eventos["data"] <= datas[-1])].reset_index(drop=True)
    ev_y = np.array([topo.asof(d) for d in ev["data"]]) * 1.25 if len(ev) else np.array([])

    def rotulo_evento(ate):
        rec = ev[(ev["data"] <= ate) & (ev["data"] > ate - pd.Timedelta(days=75))]
        return " · ".join(rec["evento"].tolist()[-2:])

    def anotacao(ate, texto):
        corpo = f"<b>{mes_ano(ate)}</b>" + (f"<br><span style='font-size:13px'>{texto}</span>" if texto else "")
        return [dict(x=0.02, y=0.97, xref="paper", yref="paper", text=corpo, showarrow=False, xanchor="left",
                     yanchor="top", align="left", font=dict(size=20, color=COR["silk"]))]

    fig = go.Figure()
    for c in cols:
        fig.add_scatter(x=datas, y=df[c].values, name=c, mode="lines", line=dict(color=cores.get(c), width=3.2),
                        hovertemplate="<b>%{fullData.name}</b>: %{y:,.0f} (base 100)<extra></extra>")
    fig.add_scatter(x=ev["data"], y=ev_y, name="Eventos de IA", mode="markers", text=ev["evento"],
                    customdata=ev[["empresa", "categoria"]].values if len(ev) else None,
                    marker=dict(size=10, color=COR["gold"], symbol="diamond", line=dict(width=1, color=COR["pcb"])),
                    hovertemplate="<b>%{text}</b><br>%{customdata[0]} · %{customdata[1]}<extra></extra>")

    frames = []
    for k in range(1, len(datas) + 1):
        ate = datas[k - 1]
        vis = ev[ev["data"] <= ate]
        dados = [go.Scatter(x=datas[:k], y=df[c].values[:k]) for c in cols]
        dados.append(go.Scatter(x=vis["data"], y=ev_y[:len(vis)], text=vis["evento"],
                                customdata=vis[["empresa", "categoria"]].values if len(vis) else None))
        frames.append(go.Frame(name=str(k), data=dados, traces=list(range(len(cols) + 1)),
                               layout=go.Layout(annotations=anotacao(ate, rotulo_evento(ate)))))
    fig.frames = frames

    passo = dict(mode="immediate", transition=dict(duration=0))
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.2, xanchor="left", yanchor="top", showactive=False,
                          bgcolor=COR["mask2"], bordercolor=COR["gold_dim"], font=dict(color=COR["gold"]),
                          buttons=[dict(label="▶  Reproduzir", method="animate",
                                        args=[None, dict(frame=dict(duration=70, redraw=True), fromcurrent=False, **passo)]),
                                   dict(label="❚❚  Pausar", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), **passo)])])],
        sliders=[dict(x=0.22, y=-0.2, len=0.78, xanchor="left", yanchor="top", pad=dict(t=0, b=0),
                      currentvalue=dict(visible=False), tickcolor="rgba(0,0,0,0)", bgcolor=COR["linha"],
                      activebgcolor=COR["gold"], bordercolor=COR["linha"], font=dict(color="rgba(0,0,0,0)"),
                      steps=[dict(method="animate", label=str(i + 1),
                                  args=[[str(i + 1)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                           transition=dict(duration=0))]) for i in range(len(datas))])])
    ymin, ymax = float(df.min().min()) * .9, float(df.max().max()) * 2.2
    eixo_log(fig, ymin, ymax)
    fig.update_yaxes(range=[np.log10(ymin), np.log10(ymax)])
    fig.update_xaxes(range=[datas[0], datas[-1] + pd.Timedelta(days=60)])
    fig.update_layout(annotations=anotacao(datas[-1], ""))
    est.setdefault("altura", 560)
    estilizar(fig, titulo, "Base 100 na primeira data em comum; escala logarítmica. Aperte ▶ ou arraste a barra para ver o tempo passar.",
              y_titulo="Índice (base 100, escala log)", **est)
    fig.update_layout(margin=dict(l=10, r=10, t=78, b=90))
    return fig
