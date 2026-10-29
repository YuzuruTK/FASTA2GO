# FASTA2GO

Pipeline de extração de dados de proteínas: FASTA → DuckDB com características calculadas,
metadados do UniProt e anotações de Gene Ontology.

Esta é a **etapa 1** do projeto. Análise exploratória e modelos preditivos de função molecular
são etapas separadas e não estão implementadas aqui.

## Requisitos e instalação

```bash
uv sync          # cria .venv e instala as dependências
```

Dependências de execução: `duckdb`, `biopython` (ProtParam/pI), `requests`.
Dependências de desenvolvimento: `pytest` e `ipykernel` (kernel do notebook).

## Estrutura

```
data/                    arquivos FASTA de entrada (não versionados)
db/                      banco DuckDB gerado (não versionado)
sql/schema.sql           tabelas, chaves e índices
sql/views.sql            views derivadas
fasta2go/fasta.py        leitura streaming do FASTA e validação por registro
fasta2go/features.py     AAC, DPC e propriedades físico-químicas
fasta2go/uniprot.py      cliente REST do UniProt e parsing do JSON
fasta2go/db.py           conexão, criação do esquema e carga em massa
fasta2go/pipeline.py     ingestão, cálculo, enriquecimento e resumo
main.py                  CLI (run | status)
tests/                   12 testes de comportamento + fixtures
notebooks/pipeline.ipynb validação etapa por etapa com dados e API reais
```

Coloque o FASTA em `data/` (aceita `.gz`). O caminho padrão é `data/uniprot_sprot.fasta`.

## Dados de entrada

`uniprot_sprot.fasta` é o download FASTA do UniProtKB/Swiss-Prot (proteoma revisado):
**575.748 entradas**, 209.017.843 resíduos, comprimento médio 363 (mínimo 2, máximo 35.213).
Todos os cabeçalhos estão no formato `sp|ACC|NOME descrição OS=… OX=… GN=… PE=… SV=…`, sem
accession repetido; **2.685** sequências contêm resíduos não canônicos (`X`=8.181, `U`=331,
`B`=276, `Z`=249, `O`=29). Não há `*`, `-`, espaços nem minúsculas nas sequências, e nenhum
cabeçalho de isoforma (`sp|ACC-N|…`).

## Execução

```bash
# fluxo completo: ingestão -> características -> UniProt -> resumo
uv run python main.py run

# recorte pequeno, útil para testar (processa no máximo 2000 proteínas nesta execução)
uv run python main.py run --limit 2000

# sem rede
uv run python main.py run --no-uniprot

# lote maior de características (menos statements; mais memória)
uv run python main.py run --feature-batch-size 10000

# reconsultar anotações mais antigas que 30 dias
uv run python main.py run --max-age-days 30

# conferir o que já está no banco (não processa nada)
uv run python main.py status
```

Todos os comandos são seguros de repetir: nada é duplicado, nada já válido é recalculado.

## O que o fluxo faz

### 1. Ingestão (`pipeline.ingest`)

Leitura em streaming, um registro por vez. São preservados o cabeçalho original
(`proteins.fasta_header`), o primeiro token do cabeçalho (`fasta_id`) e, quando o formato é
`sp|ACC|NOME` ou `tr|ACC|NOME`, o accession do UniProt. A chave primária é
`(source_file, fasta_id)`: identifica o registro de origem sem depender da sequência.

Problemas vão para `ingestion_issues`, nunca são silenciados:

| issue | efeito |
|---|---|
| `malformed_header` | registro rejeitado |
| `empty_sequence` | registro rejeitado |
| `duplicate_id` | primeira ocorrência mantida, demais rejeitadas |
| `noncanonical_residues` | registro mantido; resíduos fora dos 20 canônicos excluídos dos denominadores |
| `header_without_accession` | registro mantido, não consultado no UniProt |

### 2. Características (`pipeline.compute_features`)

Calculadas apenas para proteínas sem linha de características ou com `feature_version`
diferente da atual. O lote é lido uma vez, cada sequência é analisada uma vez e as três tabelas
são gravadas em **uma transação por lote** — se qualquer gravação falhar, o lote inteiro é
revertido. Tamanho do lote: `--feature-batch-size` (padrão 5.000).

| tabela | conteúdo |
|---|---|
| `sequence_features` | comprimento canônico, resíduos não canônicos, pares adjacentes canônicos, massa molecular, pI, GRAVY, índice de instabilidade, índice alifático (Ikai), aromaticidade, entropia de Shannon |
| `amino_acid_composition` | 20 percentuais, denominador = resíduos canônicos (`canonical_length`) |
| `dipeptide_composition` | 400 percentuais de pares adjacentes, denominador = `canonical_pairs` |

Resíduos ambíguos (`X`, `B`, `Z`) e raros (`U`, `O`) são removidos antes da AAC e das fórmulas
físico-químicas — nenhuma delas é definida para eles — e o total removido fica em
`noncanonical_residues`. O **DPC é a exceção**: os pares são contados na sequência original e
somente quando os dois resíduos são canônicos, de modo que um resíduo não canônico **quebra** a
adjacência em vez de unir os vizinhos. Em `ACXDE` contam-se `AC` e `DE` (50% cada), nunca `CD`;
o denominador (`canonical_pairs` = 2) fica gravado ao lado das características. Sequências sem
nenhum par válido (`AXXA`) ficam com as 400 colunas `NULL` — indefinido, não zero. AAC e DPC
somam 100 por construção, cada um com o seu próprio denominador.

Uma proteína sem nenhum resíduo canônico mantém **uma linha nas três tabelas**, com valores
`NULL`, para que as três permaneçam alinhadas com `proteins`.

As classes químicas (hidrofóbico, polar, positivo, negativo, carregado, aromático, enxofre) são
**derivadas em view** a partir da AAC em `protein_features`; não são armazenadas de novo.

### 3. Enriquecimento (`pipeline.enrich`)

Uma requisição por lote de accessions (200 por padrão) ao endpoint oficial
`/uniprotkb/accessions`, com `fields` reduzido (a entrada completa é ~4× maior). Accessions são
validados antes do lote: um identificador malformado faria o lote inteiro falhar com HTTP 400.
Falhas 429/5xx/timeout são repetidas com backoff; falha persistente aborta a etapa com mensagem
clara e **nada é gravado como "não encontrado"**.

Campos gravados em `uniprot_annotations`: accession, entry name, nome da proteína, genes,
organismo, táxon, existência da proteína, accessions secundários, keywords, descrição de função,
localização subcelular e comprimento informado pela fonte.

Estado da anotação em `uniprot_annotations.status`:

- linha ausente → **nunca consultado**;
- `matched` → a fonte devolveu a entrada;
- `not_found` → consultado, a fonte não conhece o accession.

GO fica em `protein_go_annotations`, uma linha por `(proteína, termo, evidência)`:
`go_id`, `aspect` (`biological_process` / `cellular_component` / `molecular_function`),
`go_name`, `evidence_code`, `evidence_source` (quem atribuiu) e `provenance`
(referências `PubMed:...`, quando existem — cerca de 91% das anotações do UniProt não têm
referência). **Nenhuma propagação de ancestrais** é feita nesta etapa.

## Consultando o resultado

```bash
uv run python main.py status
```

```sql
-- visão pronta para a etapa de análise
SELECT * FROM protein_features LIMIT 10;

SELECT * FROM protein_annotation_summary
WHERE annotation_status = 'matched' AND go_mf_terms > 0 LIMIT 10;

-- anotações de função molecular de uma proteína
SELECT go_id, go_name, evidence_code, evidence_source, provenance
FROM protein_go_annotations
WHERE fasta_id = 'sp|P69905|HBA_HUMAN' AND aspect = 'molecular_function';
```

Para SQL livre, abra o banco com `duckdb db/fasta2go.duckdb` (CLI) ou em Python:

```bash
uv run python -c "import duckdb; print(duckdb.connect('db/fasta2go.duckdb').execute('SELECT count(*) FROM proteins').fetchone())"
```

## Tabelas

| tabela | finalidade |
|---|---|
| `proteins` | um registro por entrada FASTA, com cabeçalho, sequência e accession |
| `sequence_features` | propriedades físico-químicas e comprimentos |
| `amino_acid_composition` | 20 percentuais de aminoácidos |
| `dipeptide_composition` | 400 percentuais de dipeptídeos |
| `uniprot_annotations` | metadados do UniProt e estado da consulta |
| `protein_go_annotations` | anotações GO da fonte, com aspecto, evidência e proveniência |
| `ingestion_issues` | registros rejeitados ou sinalizados na última ingestão do arquivo |

Views: `protein_features` (sequência + características + AAC + classes químicas) e
`protein_annotation_summary` (status da anotação e contagem de termos GO por aspecto).

## Testes e validação

```bash
uv run pytest            # 12 testes, sem acesso à rede
```

Os testes usam `tests/data/sample.fasta` (proteínas válidas, sequência vazia, cabeçalho
malformado, identificador repetido, resíduos ambíguos) e uma resposta real do UniProt capturada
em `tests/data/uniprot_response.json`. Cobrem: leitura do FASTA, denominadores de AAC/DPC
(inclusive `ACXDE`, `AXXA` e o rollback do lote), proteína sem resíduo canônico com uma linha
por tabela, propriedades conhecidas da hemoglobina alfa, idempotência, retomada, parsing de GO
(aspecto/evidência/proveniência) e o caso de falha de rede sem gravação de dados.

`notebooks/pipeline.ipynb` percorre a etapa inteira com asserts, usando um banco descartável e a
API real.

## Limitações conhecidas

- **Custo do enriquecimento completo**: 575.748 proteínas exigem ~2.900 requisições, ~1,8 GB de
  download e cerca de 1 h (extrapolação da execução medida); o banco completo fica em torno de
  4 GB, dominado pela tabela de dipeptídeos (400 colunas `DOUBLE`).
- Sem propagação de ancestrais GO e sem GO Slim: pertencem à etapa de rotulagem.
- Sem comparação entre a sequência do FASTA e a do UniProt (`source_length` é gravado apenas
  como informação da fonte).
- Um FASTA sem cabeçalho na primeira linha interrompe a ingestão com erro explícito.
