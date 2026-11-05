"""Dados: cobertura, atualização, séries próprias e tabela de eventos."""
import pandas as pd
import streamlit as st

from plataforma import dados, visual


def render(ctx):
    visual.cabecalho("Dados", "De onde vêm os números: ver o que está carregado, atualizar, acrescentar suas séries e editar os eventos.")

    st.markdown("#### Cobertura", help="Para cada série: categoria, primeira e última data com dado e número de observações. "
                                       "Séries com início mais recente limitam a base 100 e os testes que usam várias séries juntas.")
    p = ctx.painel
    cob = pd.DataFrame({
        "categoria": pd.Series(ctx.categorias),
        "início": p.apply(lambda c: c.first_valid_index()).dt.date,
        "fim": p.apply(lambda c: c.last_valid_index()).dt.date,
        "observações": p.notna().sum(),
    }).sort_values("categoria")
    st.dataframe(cob, width="stretch", column_config={
        "início": st.column_config.DateColumn(format="DD/MM/YYYY"), "fim": st.column_config.DateColumn(format="DD/MM/YYYY")})

    st.markdown("#### Como os índices são construídos", help="Cada empresa é reescalada para 100 na primeira data em que todas as empresas do índice têm cotação; "
                                                               "o índice é a média dessas séries. Isso elimina o efeito das moedas diferentes (KRW, TWD, USD).")
    for nome in dados.GRUPOS_INDICE:
        st.markdown(f"- **{nome}:** {dados.composicao(nome)}")

    st.markdown("#### Atualizar do Yahoo Finance", help="Baixa de novo todas as séries do catálogo, incluindo os dias mais recentes. Pode levar alguns segundos.")
    if st.button("Baixar dados novamente", help="Substitui data/precos.csv pelos dados atuais do Yahoo Finance."):
        with st.spinner("Baixando..."):
            dados.baixar()
        st.cache_data.clear()
        st.rerun()

    st.markdown("#### Adicionar série própria", help="Traga qualquer dado que possa se relacionar com o tema. A série aparece em todas as páginas com a categoria 'Dados próprios'.")
    st.caption("CSV com duas colunas: data e valor. Aceita separador vírgula ou ponto e vírgula e decimal com vírgula. "
               "Exemplos: lançamentos por mês, preço de DRAM spot, capex de data centers.")
    nome = st.text_input("Nome da série", help="Como a série vai aparecer nas listas. Até 40 caracteres; só letras, números, espaço e hífen.")
    arq = st.file_uploader("Arquivo CSV", type=["csv"], help="Primeira coluna: data. Segunda coluna: valor numérico. Mínimo de 24 pontos, máximo de 5 MB.")
    if st.button("Adicionar", disabled=not (nome and arq), help="Valida e grava o arquivo em data/extras. Preencha o nome e escolha o arquivo antes."):
        try:
            n, linhas = dados.salvar_extra(nome, arq.getvalue())
            st.cache_data.clear()
            st.success(f"'{n}' adicionada com {linhas} pontos.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.markdown("#### Tabela de eventos de IA", help="Lista usada no estudo de eventos. Cada linha é uma data e o que aconteceu nela.")
    st.caption("Edite, adicione ou remova linhas e salve. Confira as datas antes de usar no relatório.")
    editado = st.data_editor(
        dados.carregar_eventos(), num_rows="dynamic", width="stretch", hide_index=True,
        column_config={
            "data": st.column_config.DateColumn("data", format="DD/MM/YYYY", required=True, help="Dia do anúncio ou lançamento."),
            "categoria": st.column_config.SelectboxColumn("categoria", options=["Modelo", "Hardware", "Mercado"], required=True,
                                                          help="Modelo: lançamento de IA. Hardware: anúncio de chip. Mercado: fato que mexeu com as ações."),
            "evento": st.column_config.TextColumn("evento", required=True, help="Descrição curta."),
            "empresa": st.column_config.TextColumn("empresa", help="Quem lançou."),
        })
    if st.button("Salvar eventos", help="Grava as alterações em data/eventos_ia.csv. Linhas sem data, evento ou categoria são descartadas."):
        out = editado.dropna(subset=["data", "evento", "categoria"]).copy()
        out["data"] = pd.to_datetime(out["data"])
        out.sort_values("data").to_csv(dados.ARQ_EVENTOS, index=False)
        st.success("Eventos salvos.")
        st.rerun()
