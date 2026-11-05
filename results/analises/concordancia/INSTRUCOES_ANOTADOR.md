# Instruções para o segundo anotador

Obrigado por aceitar. Este trabalho mede **o quanto dois especialistas concordam** ao marcar normas técnicas no formato RASE. A referência do *dataset* vem da conversão de Mendonça (2020, Poli-USP) da Seção 6 da NBR 9050. O anotador convidado é o segundo especialista, com experiência prévia de marcação RASE da NBR 9050.

Não existe resposta "certa" a adivinhar: o que interessa é a sua leitura independente, **como o anotador marcaria no seu próprio trabalho**. Se a sua convenção divergir das definições abaixo em algum ponto (por exemplo, onde termina a aplicabilidade, ou se uma oração condicional é seleção ou exceção), siga a sua convenção e anote a divergência num arquivo de texto à parte. Essa informação é útil para a dissertação.

Para que a medida seja válida:

- **Não consulte**, durante a anotação, o arquivo `dataset.json`, os prompts, os capítulos 4 e 5 da dissertação nem nenhuma outra anotação do projeto. Se o anotador já leu a dissertação, isso é declarado no texto; pedimos apenas que não a reabra enquanto anota.
- **Não converse** sobre os casos com o autor até terminar e devolver os arquivos.
- Pode consultar a NBR 9050 e os seus próprios quadros de marcação, se quiser.

São **duas tarefas**, cada uma num arquivo. Tempo estimado: Tarefa 1, cerca de 1 hora; Tarefa 2, cerca de 1h30. Podem ser feitas em dias diferentes.

---

## Como abrir e salvar os arquivos

- Os arquivos são CSV separados por ponto e vírgula (`;`), em UTF-8. Abrem diretamente no Excel ou no LibreOffice Calc (no LibreOffice, escolha "Separado por: ponto e vírgula" e "Conjunto de caracteres: UTF-8").
- **Preencha apenas as colunas indicadas.** Não apague, reordene nem edite as outras colunas, e não ordene as linhas.
- Ao salvar, mantenha o formato **CSV com separador `;`** e o **mesmo nome de arquivo**. No Excel: "Salvar como" > "CSV UTF-8 (delimitado por vírgulas)". Se o Excel trocar o separador para vírgula, avise o autor; não é problema, ele converte.
- Devolva os dois arquivos preenchidos ao autor.

---

## Tarefa 1: marcar os operadores RASE

**Arquivo:** `referencia_n2_segundo_anotador.csv` (40 linhas)

Cada linha traz uma **regra** já isolada (coluna `texto_n1`) e o texto completo do item da norma de onde ela saiu (coluna `texto_norma`, só para contexto). A tarefa é dividir a regra nos quatro operadores RASE e copiar cada trecho para a coluna correspondente:

| Coluna a preencher | Operador | O que é |
|---|---|---|
| `aplicabilidade` | **A**plicabilidade | A quem ou a que a regra se aplica: o elemento sobre o qual a regra incide. |
| `selecao` | **S**eleção | O subconjunto da aplicabilidade a que a regra se restringe: um recorte, um qualificador ou uma condição que limita o escopo. |
| `excecao` | **E**xceção | Os casos que **não** seguem a regra. |
| `requisito` | **R**equisito | A exigência propriamente dita: o que deve (ou não pode) ser feito ou atendido. |

### Regras de marcação

1. **Copie trechos literais da coluna `texto_n1`.** Não reescreva, não resuma, não corrija a grafia, não acrescente palavras. Cada operador é um recorte do texto da regra.
2. **O requisito é obrigatório** e, em geral, começa no verbo que expressa a obrigação ou a permissão: "deve", "devem", "não pode", "necessita", "recomenda-se", "pode" etc.
3. **Aplicabilidade e seleção não têm verbo de obrigação.** São sintagmas nominais ou expressões que delimitam o alvo da regra.
4. **Seleção, exceção e até a aplicabilidade podem não existir.** Nesse caso, deixe a célula **vazia**. Deixar vazio é uma resposta válida e frequente: na maior parte das regras não há exceção, e em boa parte não há seleção.
5. **Um trecho só pode ir para um operador.** Não repita o mesmo trecho em duas colunas.
6. Os trechos não precisam cobrir a regra inteira. Palavras que não pertencem a nenhum operador (artigos soltos, conectivos) podem ficar de fora.
7. Use o `texto_norma` apenas para entender o contexto. **Os trechos saem sempre de `texto_n1`.**

### Exemplo

(Regras inventadas para ilustrar; não estão no arquivo.)

Regra: *"Os balcões de atendimento de uso público devem ter altura máxima de 0,90 m do piso, exceto os balcões de uso exclusivo de funcionários."*

| aplicabilidade | selecao | excecao | requisito |
|---|---|---|---|
| Os balcões de atendimento | uso público | exceto os balcões de uso exclusivo de funcionários | devem ter altura máxima de 0,90 m do piso |

Regra: *"Os bebedouros devem ter bica com altura de 0,90 m do piso."*

| aplicabilidade | selecao | excecao | requisito |
|---|---|---|---|
| Os bebedouros | *(vazio)* | *(vazio)* | devem ter bica com altura de 0,90 m do piso |

### Quando ficar em dúvida

Decida pela a sua melhor leitura e siga em frente; não deixe linhas sem requisito. Se quiser registrar a dúvida, anote-a num arquivo de texto à parte, com o número da linha. **Não** escreva comentários dentro das colunas de operador.

---

## Tarefa 2: julgar se a saída de um modelo está correta

**Arquivo:** `pares_roc_segundo_anotador.csv` (264 linhas)

Cada linha traz um **trecho de referência** (coluna `referencia`) e um **trecho gerado por um modelo de linguagem** (coluna `previsto`) para o mesmo lugar da marcação. A coluna `nivel` diz o tipo:

- `n1`: os dois trechos são **regras inteiras**;
- `n2`: os dois trechos são **um operador RASE** (a coluna `operador` diz qual: `aplicability` = aplicabilidade, `selection` = seleção, `exception` = exceção, `requeriments` = requisito).

Preencha a coluna `correto` com:

- **`1`** se o trecho previsto **preserva o conteúdo normativo** da referência: a mesma entidade, os mesmos valores e as mesmas condições;
- **`0`** caso contrário.

### O que NÃO torna o par incorreto (pode marcar 1)

- Diferença de redação com o mesmo sentido (sinônimos, outra ordem de palavras, voz passiva × ativa).
- Diferença de pontuação, maiúsculas, acentos ausentes, artigos ("a rampa" × "rampa"), singular × plural.
- Ponto final ou aspas sobrando.

### O que torna o par incorreto (marque 0)

- **Omissão** de uma condição, de um valor, de uma unidade ou de uma remissão (por exemplo, "conforme 6.7").
- **Acréscimo** de condição, valor ou elemento que a referência não tem.
- **Escopo alterado**: a regra passa a valer para mais ou para menos casos (por exemplo, "pisos" × "pisos internos"; "superior a" × "igual ou superior a").
- **Trecho de outro lugar**: o previsto fala de outra coisa, ou repete um exemplo que nada tem a ver com a referência.
- **Mudança de obrigação**: "deve" × "pode", "recomenda-se" × "deve", "no máximo" × "no mínimo".

### Exemplos

(Exemplos inventados para ilustrar; não estão no arquivo.)

| referencia | previsto | correto | por quê |
|---|---|---|---|
| devem ter altura máxima de 0,90 m do piso | devem ter altura máxima de 0,90 m do piso. | 1 | só muda o ponto final |
| Todo bebedouro deve ter bica com altura de 0,90 m do piso. | Os bebedouros devem possuir bica a 0,90 m de altura do piso. | 1 | mesma regra, outra redação |
| Os balcões de uso público devem ter altura máxima de 0,90 m. | Os balcões devem ter altura máxima de 0,90 m. | 0 | perdeu a condição "de uso público" |
| devem atender ao descrito em 9.2 | devem atender aos requisitos | 0 | perdeu a remissão "em 9.2" |
| Os balcões devem ter altura máxima de 0,90 m. | Os bebedouros devem ter altura máxima de 0,90 m. | 0 | outro elemento (bebedouros × balcões) |
| devem ter altura máxima de 0,90 m | devem ter altura mínima de 0,90 m | 0 | obrigação invertida (máxima × mínima) |

Os pares foram sorteados de propósito com muitos casos ruins, então é normal que a maioria receba 0. Preencha **todas** as 264 linhas com 0 ou 1.

---

## Ao terminar

Por favor, devolva ao autor os dois arquivos preenchidos, com os mesmos nomes. Ele rodará:

```
python -m analyses.concordancia calcular
```

O script calcula o kappa de Cohen e grava o resultado em `results/analises/concordancia.json`.

Junto com os arquivos, se possível, envie:

- o arquivo de texto com as dúvidas e as divergências de convenção (se houver);
- uma indicação de quais regras ou pares o anotador **reconheceu** da leitura da dissertação (basta o número da linha), para que possam ser analisados à parte;
