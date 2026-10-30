# FASTA2GO

Pipeline de extração de dados de proteínas: FASTA → DuckDB com características calculadas,
metadados do UniProt e anotações de Gene Ontology.

## Etapas do projeto

| etapa | pasta | conteúdo | estado |
|---|---|---|---|
| **1. Preparação dos dados** | `etapa1/` | `main.py` (extração), `build_go_slim_labels.py` (rotulagem), `fasta2go.config`, `01_validacao_extracao.ipynb` | implementada |
| **2. Análise exploratória** | `etapa2/` | `02_analise_exploratoria.ipynb` | implementada |
| **3. Modelagem** | — | seleção experimental de atributos, treino e avaliação | não implementada |

`fasta2go/` e `sql/` ficam na raiz porque são **código compartilhado**: a etapa 1 escreve a base
com eles e a etapa 2 lê a base com eles. Nenhum script recebe argumentos — cada arquivo faz uma
coisa e os ajustes ficam em `etapa1/fasta2go.config`.

## Requisitos e instalação

```bash
uv sync          # cria .venv e instala as dependências
```

Dependências de execução: `duckdb`, `biopython` (ProtParam/pI), `requests`.
Dependências de desenvolvimento: `ipykernel` (kernel dos notebooks) e `seaborn` (que traz
`pandas` e `matplotlib`) para as visualizações da etapa 2.

## Estrutura

```
etapa1/main.py                 extração: FASTA -> features -> UniProt -> DuckDB (+ resumo)
etapa1/build_go_slim_labels.py rotulagem GO Slim da base -> exports/go_slim/
etapa1/fasta2go.config         ajustes da extração (entrada, banco, limite, lote, UniProt)
etapa1/01_validacao_extracao.ipynb   confere o banco real, somente leitura
etapa2/02_analise_exploratoria.ipynb análise exploratória e candidatos a atributos
fasta2go/fasta.py              leitura streaming do FASTA e validação por registro
fasta2go/features.py           AAC, DPC e propriedades físico-químicas
fasta2go/uniprot.py            cliente REST do UniProt e parsing do JSON
fasta2go/db.py                 conexão (inclusive somente leitura), esquema e carga em massa
fasta2go/pipeline.py           extração integrada em lotes + resumo do banco
fasta2go/go_slim.py            vocabulário GO Slim, propagação e mapeamento das anotações
sql/schema.sql                 tabelas, chaves e índices
sql/views.sql                  views derivadas
data/                          arquivos FASTA de entrada (não versionados)
data/go_slim/                  vocabulário de 41 rótulos + fontes GO fixadas por SHA-256
db/                            banco DuckDB gerado (não versionado)
exports/go_slim/               rótulos por proteína (gerado, versionado para a etapa 2)
cache/                         OBOs baixados e CSVs de staging (não versionados)
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

A extração começa de um banco vazio: se o arquivo já existir, o script apenas avisa e sai, sem
apagar nada. Apague o banco quando quiser extrair de zero.

```bash
# extração: FASTA -> features -> UniProt -> CSVs -> DuckDB -> resumo
uv run python etapa1/main.py

# rotulagem: 41 rótulos de função molecular -> exports/go_slim/protein_labels.csv
uv run python etapa1/build_go_slim_labels.py
```

Ajustes (recorte de 2.000 proteínas, extração sem rede, outro banco) ficam em
`etapa1/fasta2go.config` — nenhum script aceita argumentos.

## Rotulagem GO Slim (41 rótulos de função molecular)

O espaço de rótulos usado na modelagem é o GO Slim `goslim_generic` restrito a
**molecular_function**, mais uma extensão documentada, `GO:0005488 binding` (o slim genérico
omite `binding` e mantém `catalytic activity`), totalizando **41 rótulos**. Regras aplicadas por
`fasta2go/go_slim.py` e registradas em `data/go_slim/sources.json`:

- relações percorridas: `is_a` e `part_of`, apenas dentro do aspecto molecular_function;
- propagação de **todos** os ancestrais (não só o mais próximo);
- raiz `GO:0003674` excluída por regra, nunca por máscara de treino;
- `alt_id` resolvido para o id primário; termos obsoletos rejeitados;
- pares `(proteína, rótulo)` deduplicados (união de conjuntos).

As fontes são fixadas por SHA-256 em `data/go_slim/sources.json` (release `2026-07-26` do
`go-basic.obo` e do `goslim_generic.obo`) e verificadas antes de qualquer uso: um OBO divergente
aborta a rotulagem em vez de gerar rótulos de outro release. O resultado é
`exports/go_slim/protein_labels.csv` (uma linha por proteína, incluindo as que não têm termo de
função molecular) com o estado de cada anotação: `mapped`, `no_vocabulary_ancestor`, `obsolete`
ou `unknown`. A base é aberta em modo somente leitura e não é alterada.

```bash
# baixa o que faltar em cache/go_slim/, verifica os SHA-256 e rotula a base do config
uv run python etapa1/build_go_slim_labels.py
```

## Configuração

Os valores ficam em `etapa1/fasta2go.config` (INI, versionado), lido por `etapa1/main.py` e por
`etapa1/build_go_slim_labels.py`:

| chave | padrão | efeito |
|---|---|---|
| `fasta` | `data/uniprot_sprot.fasta` | arquivo de entrada (aceita `.gz`) |
| `database` | `db/fasta2go.duckdb` | banco DuckDB de saída (e de leitura na rotulagem) |
| `limit` | `2000` | proteínas a gravar; vazio grava o FASTA inteiro |
| `uniprot` | `true` | `false` grava apenas proteínas e características, sem rede |
| `batch_size` | `5000` | proteínas por lote (o lote inteiro fica em memória) |
| `accessions_per_request` | `200` | accessions por requisição ao UniProt |

## O que o fluxo faz

A extração é **um único laço em lotes** (`pipeline.extract`). Cada lote percorre o fluxo completo
antes de o próximo começar:

1. **Ler e preparar** até `batch_size` registros do FASTA (`etapa1/fasta2go.config`, 5.000), em streaming.
2. **Calcular as features** de cada proteína do lote (`fasta2go/features.py`).
3. **Consultar a UniProt** para os accessions do lote (`fasta2go/uniprot.py`), em requisições de
   `accessions_per_request` accessions (`etapa1/fasta2go.config`, 200).
4. **Consolidar o lote em CSV**, um arquivo por tabela (`db.write_rows`).
5. **Inserir o lote no DuckDB em uma transação**: as sete tabelas entram juntas, lidas pelo
   `read_csv` nativo, então uma falha não deixa lote pela metade.
6. Repetir até o fim do FASTA (ou até `limit` proteínas gravadas, em `etapa1/fasta2go.config`).

O lote de 5.000 proteínas existe porque o fluxo integrado mantém o lote inteiro em memória:
medindo leitura + features + respostas da UniProt + CSVs + `INSERT`, o pico é de ~0,75 GB de RSS
com 5.000 proteínas, 1,1 GB com 10.000 e 1,7 GB com 20.000 (~80 KB por proteína). O lote de
100.000 previsto inicialmente exigiria ~7-8 GB.

### Leitura e validação do FASTA

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

### Features da sequência

Cada sequência do lote é analisada uma vez e as três tabelas são gravadas na mesma transação do
lote. Uma proteína sem nenhum resíduo canônico mantém **uma linha nas três tabelas**, com valores
`NULL`, para que as três permaneçam alinhadas com `proteins`.

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

As classes químicas (hidrofóbico, polar, positivo, negativo, carregado, aromático, enxofre) são
**derivadas em view** a partir da AAC em `protein_features`; não são armazenadas de novo.

### Enriquecimento pela UniProt (no mesmo lote)

Uma requisição por bloco de `accessions_per_request` accessions (200 em `etapa1/fasta2go.config`) ao
endpoint oficial `/uniprotkb/accessions`, com `fields` reduzido (a entrada completa é ~4× maior).
Accessions são validados antes do bloco: um identificador malformado faria o lote inteiro falhar
com HTTP 400.
Falhas 429/5xx/timeout são repetidas com backoff; falha persistente aborta a execução antes de
o lote ser gravado, e **nada é gravado como "não encontrado"**.

Campos gravados em `uniprot_annotations`: accession, entry name, nome da proteína, genes,
organismo, táxon, existência da proteína, accessions secundários, keywords, descrição de função,
localização subcelular e comprimento informado pela fonte.

Estado da anotação em `uniprot_annotations.status`:

- `matched` → a fonte devolveu a entrada;
- `not_found` → consultado, a fonte não conhece o accession;
- linha ausente → a proteína não tem accession (`header_without_accession`) ou o lote ainda não
  foi gravado.

GO fica em `protein_go_annotations`, uma linha por `(proteína, termo, evidência)`:
`go_id`, `aspect` (`biological_process` / `cellular_component` / `molecular_function`),
`go_name`, `evidence_code`, `evidence_source` (quem atribuiu) e `provenance`
(referências `PubMed:...`, quando existem — cerca de 91% das anotações do UniProt não têm
referência). A associação é sempre feita pelo accession da própria proteína: um accession que a
fonte não devolveu fica `not_found` e **não recebe termo algum**. A propagação de ancestrais não
acontece aqui: ela é aplicada na rotulagem (`etapa1/build_go_slim_labels.py`), que lê esta tabela
e grava os 41 rótulos em `exports/go_slim/protein_labels.csv`.

## Consultando o resultado

O resumo da base sai no fim da extração e pode ser conferido a qualquer momento com
`etapa1/01_validacao_extracao.ipynb` (somente leitura) ou direto no DuckDB:

```bash
duckdb db/fasta2go.duckdb      # CLI oficial; ou `uv run python` com a conexão abaixo
```

Os rótulos de função molecular ficam em `exports/go_slim/protein_labels.csv` (uma linha por
proteína, rótulos separados por `;`). Para abrir só a base, em modo somente leitura:

```python
from fasta2go import db

conn = db.connect("db/fasta2go.duckdb", read_only=True)
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

## Análise exploratória (etapa 2)

`etapa2/02_analise_exploratoria.ipynb` investiga a base e o espaço de rótulos sem alterar
nada (conexão somente leitura): visão geral e integridade, qualidade e denominadores,
distribuições, correlação de Pearson/Spearman e redundância, prevalência dos 41 rótulos
(incluindo ausentes, raros e coocorrência) e associação atributo-rótulo por correlação
ponto-bisserial com p-valor de Fisher e correção de Benjamini-Hochberg. As conclusões listam
candidatos a remoção/seleção **sem remover nada** — a decisão pertence à etapa 3, tomada no
conjunto de treino e nunca no conjunto de teste externo.

## Validação

Não há suíte de testes: a verificação é feita em execução, onde ela importa.

| verificação | o que garante |
|---|---|
| `etapa1/01_validacao_extracao.ipynb` | contagens alinhadas, nenhuma linha órfã, AAC/DPC somando 100, DPC indefinido só sem par canônico, anotação sempre do accession da própria proteína, `assert` no resumo — e nada é gravado (conexão somente leitura) |
| `etapa1/build_go_slim_labels.py` | verifica o SHA-256 das fontes GO fixadas, aborta se o slim pinado não derivar os 41 rótulos publicados, se a raiz virar rótulo ou se um dos 3 casos de propagação registrados divergir |
| `etapa2/02_analise_exploratoria.ipynb` | falha alto se a base e o export de rótulos não tiverem o mesmo número de proteínas, se os 41 rótulos não estiverem no vocabulário publicado ou se os atributos e rótulos não alinharem |
| `etapa1/main.py` | recusa um banco existente em vez de sobrescrevê-lo |

## Limitações conhecidas

- **Custo do enriquecimento completo**: 575.748 proteínas exigem ~2.900 requisições, ~1,8 GB de
  download e cerca de 1 h (extrapolação da execução medida); o banco completo fica em torno de
  4 GB, dominado pela tabela de dipeptídeos (400 colunas `DOUBLE`).
- Cada extração completa começa de um banco vazio (apague `db/fasta2go.duckdb` para refazer):
  não há sincronização incremental entre execuções nem retomada de um banco parcialmente gravado.
- A rotulagem GO Slim depende de rede apenas para baixar as fontes fixadas; a análise
  exploratória e a etapa 3 usam o CSV já gerado.
- Os rótulos cobrem apenas molecular_function; processo biológico e componente celular ficam
  fora do espaço de rótulos.
- Sem comparação entre a sequência do FASTA e a do UniProt (`source_length` é gravado apenas
  como informação da fonte).
- Um FASTA sem cabeçalho na primeira linha interrompe a extração com erro explícito.
