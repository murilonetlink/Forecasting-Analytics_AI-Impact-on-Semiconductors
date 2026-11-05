"""Estado compartilhado entre as páginas: dados, filtros da barra lateral e links entre páginas."""
from dataclasses import dataclass, field

import pandas as pd

from plataforma import dados, visual

FREQ_ROTULOS = {"D": "Diária", "W": "Semanal", "M": "Mensal"}


@dataclass
class Contexto:
    painel: pd.DataFrame
    categorias: dict
    ini: str
    fim: str
    freq: str
    paginas: dict = field(default_factory=dict)
    cores: dict = field(init=False)

    def __post_init__(self):
        self.cores = visual.cores_series(self.categorias)

    @property
    def colunas(self):
        return list(self.painel.columns)

    @property
    def rotulo_freq(self):
        return FREQ_ROTULOS[self.freq]

    def tabela(self, nomes, freq=None, completa=False):
        """Séries na frequência escolhida. `completa` ignora o recorte de período da barra lateral."""
        base = self.painel if completa else self.painel.loc[self.ini:self.fim]
        return dados.reamostrar(base[list(nomes)], freq or self.freq)

    def serie(self, nome, freq=None, completa=False):
        return self.tabela([nome], freq, completa)[nome].dropna()

    @classmethod
    def padrao(cls, freq="M"):
        painel, categorias = dados.carregar()
        return cls(painel, categorias, str(painel.index.min().date()), str(painel.index.max().date()), freq)
