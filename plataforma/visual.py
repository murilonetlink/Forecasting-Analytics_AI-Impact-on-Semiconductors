"""Identidade visual (herdada da apresentação), estilo dos gráficos e componentes HTML."""
import html
import textwrap

import streamlit as st

COR = {
    "pcb": "#0C1B15", "mask": "#123529", "mask2": "#0F2A20", "linha": "#2A5B45", "grid": "#1A392C",
    "gold": "#D9A441", "gold_dim": "#8C6A2A", "silk": "#E4EDE6", "dim": "#93A79A",
    "mem": "#5AA9DC", "gpu": "#E2664B", "ok": "#5BBE8A",
}
FONTE = "Archivo, system-ui, -apple-system, 'Segoe UI', sans-serif"

# Tons de cada categoria; memória e GPU não repetem a cor dos índices.
FAMILIAS = {
    "Memória": ["#3E86B8", "#84C1E8", "#2F6B94", "#A8D4F0"],
    "GPU / IA": ["#F08C6E", "#C24E36", "#F5AE95", "#A8402C", "#FFC2AD"],
    "Semicondutores": ["#A58BE0", "#8B6FD0", "#C3B1F0"],
    "Commodities": ["#D9A441", "#C98B4B", "#E8C374"],
    "Mercado": ["#E4EDE6", "#B9C7BE", "#93A79A"],
    "Câmbio": ["#4FC1B0", "#3AA797"],
    "Juros / risco": ["#E07AA5", "#C45F8B"],
    "Dados próprios": ["#B5D957", "#9CC23F", "#D4EA8C"],
}
CORES_INDICES = {"Índice Memória": COR["mem"], "Índice GPU": COR["gpu"], "Índice GPU (NVDA + AMD)": "#F5AE95"}
COR_MODELO = {"arima": COR["mem"], "sarima": "#A58BE0", "sarimax": COR["gold"], "arch": COR["ok"], "garch": COR["gpu"],
              "rw": "#93A79A", "rw_deriva": "#E4EDE6"}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
:root{--pcb:#0C1B15;--mask:#123529;--mask2:#0F2A20;--gold:#D9A441;--gold-dim:#8C6A2A;--silk:#E4EDE6;--silk-dim:#93A79A;
--mem:#5AA9DC;--gpu:#E2664B;--ok:#5BBE8A;--line:#1D4033;--fm:'IBM Plex Mono',ui-monospace,monospace}
.stApp{background:radial-gradient(circle at 18% 12%,rgba(90,169,220,.07),transparent 42%),
radial-gradient(circle at 84% 78%,rgba(226,102,75,.07),transparent 46%),var(--pcb)}
.stApp,.stApp p,.stApp li,.stApp label,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp button,.stApp input,
.stApp textarea,.stApp th,.stApp td,.stApp a,.stApp summary{font-family:'Archivo',system-ui,-apple-system,'Segoe UI',sans-serif}
.stMainBlockContainer{max-width:1280px;padding-top:2.4rem}
[data-testid="stHeader"]{background:transparent}
[data-testid="stAppDeployButton"]{display:none}
.titulo-slide,.frase,.cab-t,.cab-p,.big,.rot,.mapa-t,.mapa-s,.lb,.dica,.hip,.numcard,.leitura{font-family:'Archivo',system-ui,-apple-system,'Segoe UI',sans-serif}
[data-testid="stSidebar"]{background:var(--mask2);border-right:1px solid var(--line)}
.menu-secao{font-family:var(--fm);font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--silk-dim);margin:.9rem 0 .15rem .25rem}
[data-testid="stMetric"]{background:var(--mask2);border:1px solid var(--line);border-left:3px solid var(--gold);
border-radius:4px;padding:12px 16px}
[data-testid="stMetricLabel"] p{color:var(--silk-dim);font-size:13px}
[data-testid="stMetricValue"]{font-weight:700;letter-spacing:-.02em;font-size:1.35rem}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:4px;background:var(--mask2)}
[data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--line)!important;border-radius:4px}
.cab{display:flex;align-items:flex-start;margin:0 0 .6rem}
.cab-t{font-size:clamp(26px,3vw,36px);font-weight:600;letter-spacing:-.02em;line-height:1.15}
.cab-p{color:var(--silk-dim);font-size:16px;margin-top:.3rem;max-width:84ch}
.chip{display:inline-block;font-family:var(--fm);font-size:12px;letter-spacing:.08em;padding:3px 9px;border-radius:3px;
border:1px solid var(--c);color:var(--c);background:rgba(255,255,255,.03);margin-right:12px;white-space:nowrap;margin-top:.45rem}
.leitura{background:var(--mask2);border:1px solid var(--line);border-left:4px solid var(--gold);border-radius:4px;
padding:14px 18px;margin:.4rem 0 1rem}
.leitura.ok{border-left-color:var(--ok)}.leitura.alerta{border-left-color:var(--gpu)}
.leitura .lt{font-family:var(--fm);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--gold);margin-bottom:.35rem}
.leitura.ok .lt{color:var(--ok)}.leitura.alerta .lt{color:var(--gpu)}
.leitura .lb{font-size:16px;line-height:1.55;color:var(--silk)}
.led{display:inline-block;width:8px;height:8px;border-radius:50%;background:currentColor;margin-right:8px;
box-shadow:0 0 8px currentColor;vertical-align:baseline}
.dica{font-size:13.5px;color:var(--silk-dim);border-left:2px solid #21493A;padding:2px 0 2px 12px;margin:.2rem 0 1rem}
.ref{font-family:var(--fm);font-size:12.5px;color:var(--gold-dim);letter-spacing:.06em}
.hip{background:var(--mask2);border:1px solid var(--line);border-left:3px solid var(--c);border-radius:4px;padding:16px 18px;min-height:196px}
.hip .chip{margin:0 0 .6rem 0}
.hip .ht{margin:0 0 .35rem;font-size:18px;font-weight:600;line-height:1.3}
.hip p{margin:0;font-size:14.5px;color:var(--silk-dim);line-height:1.5}
.mapa-t{font-size:17px;font-weight:600;line-height:1.3}
.mapa-s{font-size:13.5px;color:var(--silk-dim);margin:.2rem 0 .4rem;line-height:1.45}
.visto{font-family:var(--fm);font-size:11.5px;color:var(--silk-dim);letter-spacing:.06em}
.visto.sim{color:var(--gold)}
.kicker{font-family:var(--fm);font-size:12.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--gold)}
.titulo-slide{font-size:clamp(28px,3.6vw,46px);font-weight:600;letter-spacing:-.022em;line-height:1.12;margin:.3rem 0 .2rem}
.regua{width:58px;height:2px;background:var(--gpu);margin:.7rem 0 1rem}
.big{font-size:clamp(30px,4vw,52px);font-weight:700;letter-spacing:-.025em;line-height:1.05}
.big.mem{color:var(--mem)}.big.gpu{color:var(--gpu)}.big.gold{color:var(--gold)}.big.ok{color:var(--ok)}
.rot{font-size:13.5px;color:var(--silk-dim);margin-top:.3rem;line-height:1.4}
.numcard{background:var(--mask2);border:1px solid var(--line);border-radius:4px;padding:14px 18px;min-height:124px}
.pontos{display:flex;gap:6px;flex-wrap:wrap;margin:.2rem 0 .8rem}
.ponto{width:28px;height:3px;background:#1F4335;border-radius:2px}
.ponto.passou{background:var(--gold-dim)}.ponto.atual{background:var(--gold)}
.frase{font-size:clamp(20px,2.3vw,28px);line-height:1.4;font-weight:500;letter-spacing:-.01em;max-width:60ch;margin:.2rem 0 1rem}
"""

CSS_PALCO = """
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="stExpandSidebarButton"],
[data-testid="stHeader"]{display:none}
.stMainBlockContainer{max-width:1320px;padding-top:.8rem}
.titulo-slide{font-size:clamp(26px,3.1vw,40px)}
.frase{font-size:clamp(18px,2vw,25px);margin:.1rem 0 .6rem}
.regua{margin:.45rem 0 .7rem}
.pontos{margin:0 0 .3rem}
"""


def md(corpo):
    st.markdown(corpo, unsafe_allow_html=True)


def aplicar_tema():
    md(f"<style>{CSS}</style>")


def aplicar_tema_palco():
    md(f"<style>{CSS_PALCO}</style>")


def esc(texto):
    return html.escape(str(texto))


# ------------------------------------------------------------------ componentes
def chip(texto, cor):
    return f'<span class="chip" style="--c:{cor}">{esc(texto)}</span>'


def cabecalho(titulo, pergunta, selo=None, cor=None):
    md(f'<div class="cab">{chip(selo, cor) if selo else ""}<div><div class="cab-t">{esc(titulo)}</div>'
       f'<div class="cab-p">{esc(pergunta)}</div></div></div><div class="regua"></div>')


def leitura(titulo, texto, tom="info"):
    """Resumo em linguagem simples. `texto` pode ter <b>; valores vindos de usuário devem passar por esc()."""
    md(f'<div class="leitura {tom}"><div class="lt"><span class="led"></span>{esc(titulo)}</div>'
       f'<div class="lb">{texto}</div></div>')


def dica(texto, rotulo="Como ler"):
    md(f'<div class="dica"><b>{rotulo}:</b> {texto}</div>')


def numero(valor, rotulo, cor="gold"):
    return f'<div class="numcard"><div class="big {cor}">{valor}</div><div class="rot">{rotulo}</div></div>'


def cores_series(categorias):
    """Cor estável por série, agrupada pela categoria."""
    cores, usados = {}, {}
    for nome, cat in categorias.items():
        if nome in CORES_INDICES:
            cores[nome] = CORES_INDICES[nome]
            continue
        familia = FAMILIAS.get(cat, [COR["dim"]])
        k = usados.get(cat, 0)
        usados[cat] = k + 1
        cores[nome] = familia[k % len(familia)]
    return cores


# ---------------------------------------------------------------------- gráficos
CONFIG_GRAFICO = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}


def quebrar(texto, largura=56):
    return "<br>".join(textwrap.wrap(texto, largura))


def estilizar(fig, titulo=None, subtitulo=None, altura=440, x_titulo=None, y_titulo=None,
              ajuda=None, legenda=True, hover="x unified", datas=True, fonte=13):
    """Aplica o visual da plataforma. `ajuda` vira um ⓘ no canto do gráfico com o texto ao passar o mouse."""
    topo = 78 if (titulo and subtitulo) else 52 if titulo else 24
    if titulo:
        t = dict(text=f"<b>{titulo}</b>", x=0, xanchor="left", font=dict(size=fonte + 3))
        if subtitulo:
            t["subtitle"] = dict(text=subtitulo, font=dict(size=fonte - 1, color=COR["dim"]))
        fig.update_layout(title=t)
    fig.update_layout(
        template="plotly_dark", height=altura, hovermode=hover, separators=",.",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONTE, color=COR["silk"], size=fonte),
        margin=dict(l=10, r=10, t=topo, b=10),
        hoverlabel=dict(bgcolor=COR["mask2"], bordercolor=COR["gold"], font=dict(family=FONTE, color=COR["silk"], size=fonte)),
        showlegend=legenda,
        legend=dict(orientation="h", yanchor="top", y=-0.1, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=COR["grid"], linecolor=COR["linha"], zeroline=False, title=x_titulo,
                     showspikes=datas, spikemode="across", spikecolor=COR["dim"], spikethickness=1, spikedash="dot")
    fig.update_yaxes(gridcolor=COR["grid"], linecolor=COR["linha"], zerolinecolor=COR["linha"], title=y_titulo)
    if datas:
        fig.update_xaxes(hoverformat="%d/%m/%Y")
    if ajuda:
        fig.add_annotation(
            text="ⓘ", xref="paper", yref="paper", x=1, y=1, xanchor="right", yanchor="bottom", showarrow=False,
            font=dict(size=20, color=COR["gold"]), hovertext=quebrar(ajuda), captureevents=True,
            hoverlabel=dict(bgcolor=COR["mask2"], bordercolor=COR["gold"], font=dict(color=COR["silk"], size=fonte)))
    return fig


def mostrar(fig, arquivo="grafico", chave=None):
    cfg = {**CONFIG_GRAFICO, "toImageButtonOptions": {"format": "png", "scale": 2, "filename": arquivo}}
    st.plotly_chart(fig, width="stretch", theme=None, config=cfg, key=chave)
