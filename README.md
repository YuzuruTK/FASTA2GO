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
fasta2go.config          ajustes da extração (entrada, banco, lote, requisições)
fasta2go/fasta.py        leitura streaming do FASTA e validação por registro
fasta2go/features.py     AAC, DPC e propriedades físico-químicas
fasta2go/uniprot.py      cliente REST do UniProt e parsing do JSON
fasta2go/db.py           conexão, criação do esquema e carga em massa
fasta2go/pipeline.py     extração integrada em lotes + resumo do banco
main.py                  CLI (run | status)
tests/                   13 testes de comportamento + fixtures
notebooks/pipeline.ipynb validação do fluxo com dados e API reais
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

A extração sempre começa de um banco vazio: o `run` recusa um arquivo existente e só o
substitui quando `--recreate` é passado — nada é apagado sem que isso seja pedido.

```bash
# extração completa: FASTA -> features -> UniProt -> CSVs -> DuckDB -> resumo
uv run python main.py run

# recorte pequeno, útil para testar (grava no máximo 2000 proteínas)
uv run python main.py run --limit 2000

# refazer a extração sobre um banco já existente (o arquivo é substituído)
uv run python main.py run --recreate

# sem rede: grava as proteínas e as características, sem UniProt
uv run python main.py run --no-uniprot

# conferir o que já está no banco (não processa nada)
uv run python main.py status
```

## Configuração

Os valores fixos ficam em `fasta2go.config` (INI, versionado), lido pelo `main.py` na
inicialização:

| chave | padrão | efeito |
|---|---|---|
| `fasta` | `data/uniprot_sprot.fasta` | arquivo de entrada (aceita `.gz`) |
| `database` | `db/fasta2go.duckdb` | banco DuckDB de saída |
| `batch_size` | `5000` | proteínas por lote (o lote inteiro fica em memória) |
| `accessions_per_request` | `200` | accessions por requisição ao UniProt |

Precedência: `--fasta` e `--db` da linha de comando > `fasta2go.config` > valores padrão no
código (usados apenas se o arquivo não existir). `batch_size` e `accessions_per_request` não são
opções de linha de comando: são ajustes do arquivo. `--limit`, `--no-uniprot` e `--recreate` são
de execução e continuam apenas na linha de comando.

## O que o fluxo faz

A extração é **um único laço em lotes** (`pipeline.extract`). Cada lote percorre o fluxo completo
antes de o próximo começar:

1. **Ler e preparar** até `batch_size` registros do FASTA (`fasta2go.config`, 5.000), em streaming.
2. **Calcular as features** de cada proteína do lote (`fasta2go/features.py`).
3. **Consultar a UniProt** para os accessions do lote (`fasta2go/uniprot.py`), em requisições de
   `accessions_per_request` accessions (`fasta2go.config`, 200).
4. **Consolidar o lote em CSV**, um arquivo por tabela (`db.write_rows`).
5. **Inserir o lote no DuckDB em uma transação**: as sete tabelas entram juntas, lidas pelo
   `read_csv` nativo, então uma falha não deixa lote pela metade.
6. Repetir até o fim do FASTA (ou até `--limit` proteínas gravadas).

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

Uma requisição por bloco de `accessions_per_request` accessions (200 em `fasta2go.config`) ao
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
fonte não devolveu fica `not_found` e **não recebe termo algum**. **Nenhuma propagação de
ancestrais** é feita nesta etapa.

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
uv run pytest            # 13 testes, sem acesso à rede
```

Os testes usam `tests/data/sample.fasta` (proteínas válidas, sequência vazia, cabeçalho
malformado, identificador repetido, resíduos ambíguos) e uma resposta real do UniProt capturada
em `tests/data/uniprot_response.json`. Cobrem: leitura do FASTA, denominadores de AAC/DPC
(inclusive `ACXDE`, `AXXA`), proteína sem resíduo canônico com uma linha por tabela, propriedades
conhecidas da hemoglobina alfa, a extração integrada lote a lote (contagens, `ingestion_issues`,
`--limit`, transação revertida quando uma tabela falha, associação proteína → accession → GO,
`not_found` sem termo), parsing de GO (aspecto/evidência/proveniência), falha de rede sem
gravação de dados e a recusa do CLI em reaproveitar um banco existente sem `--recreate`.

`notebooks/pipeline.ipynb` percorre o fluxo integrado com asserts, usando um banco descartável e
a API real.

## Limitações conhecidas

- **Custo do enriquecimento completo**: 575.748 proteínas exigem ~2.900 requisições, ~1,8 GB de
  download e cerca de 1 h (extrapolação da execução medida); o banco completo fica em torno de
  4 GB, dominado pela tabela de dipeptídeos (400 colunas `DOUBLE`).
- Cada extração completa começa de um banco vazio (`--recreate`): não há sincronização
  incremental entre execuções nem retomada de um banco parcialmente gravado.
- Sem propagação de ancestrais GO e sem GO Slim: pertencem à etapa de rotulagem.
- Sem comparação entre a sequência do FASTA e a do UniProt (`source_length` é gravado apenas
  como informação da fonte).
- Um FASTA sem cabeçalho na primeira linha interrompe a extração com erro explícito.
