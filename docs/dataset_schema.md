# `dataset.json` Schema

The dataset follows a nested structure with 3 levels (N1/N2/N3). Each item in `datas`
holds the original `text` and the reference RASE decomposition.

## Structure

```json
{
  "meta": {
    "versao": 3,
    "fonte": "union of dataset.json (v1) and dataset_v2.json, without duplicate standards",
    "duplicadas_removidas_do_v1": { "...": "..." },
    "n3_normalizacao": "ASCII comparators (>=, <=); multi-row operators keep every row in properties_n3_rows"
  },
  "datas": [
    {
      "item": 1,
      "origem": "v2 (full appendix B)",
      "text": "Standard text...",
      "texts_n1": [
        {
          "text_n1": "Segmented sentence (N1).",
          "operators_n2": {
            "aplicability": {
              "text_n2": "Operator excerpt (empty string if absent)",
              "properties_n3": {
                "type": "aplicabilidade",
                "object": "edificacao",
                "property": "uso",
                "comparation": "=",
                "target": "residencial",
                "unit": ""
              }
            },
            "selection":    {
              "text_n2": "public use; collective use",
              "properties_n3": { "...": "..." },
              "properties_n3_rows": [ { "...": "..." }, { "...": "..." } ]
            },
            "exception":    { "text_n2": "...", "properties_n3": { ... } },
            "requeriments": { "text_n2": "...", "properties_n3": { ... } }
          }
        }
      ]
    }
  ]
}
```

## JSON Schema (formal)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "MestradoRaseDataset",
  "type": "object",
  "required": ["datas"],
  "properties": {
    "counts": { "type": "integer", "minimum": 0 },
    "time":   { "type": "number", "minimum": 0 },
    "meta":   { "type": "object" },
    "datas":  {
      "type": "array",
      "items": { "$ref": "#/$defs/textItem" }
    }
  },
  "$defs": {
    "textItem": {
      "type": "object",
      "required": ["text", "texts_n1"],
      "properties": {
        "item":    { "type": "integer", "minimum": 1 },
        "origem":  { "type": "string" },
        "origem_idx_v1": { "type": "integer" },
        "text":    { "type": "string" },
        "texts_n1":{ "type": "array", "items": { "$ref": "#/$defs/n1Item" } }
      }
    },
    "n1Item": {
      "type": "object",
      "required": ["text_n1", "operators_n2"],
      "properties": {
        "text_n1":     { "type": "string" },
        "operators_n2":{
          "type": "object",
          "properties": {
            "aplicability": { "$ref": "#/$defs/n2Op" },
            "selection":    { "$ref": "#/$defs/n2Op" },
            "exception":    { "$ref": "#/$defs/n2Op" },
            "requeriments": { "$ref": "#/$defs/n2Op" }
          }
        }
      }
    },
    "n2Op": {
      "type": "object",
      "required": ["text_n2", "properties_n3"],
      "properties": {
        "text_n2":         { "type": "string" },
        "properties_n3":   { "$ref": "#/$defs/n3Props" },
        "properties_n3_rows": { "type": "array", "items": { "$ref": "#/$defs/n3Props" } }
      }
    },
    "n3Props": {
      "type": "object",
      "required": ["type", "object", "property", "comparation", "target", "unit"],
      "properties": {
        "type":        { "type": "string" },
        "object":      { "type": "string" },
        "property":    { "type": "string" },
        "comparation": { "type": "string" },
        "target":      { "type": "string" },
        "unit":        { "type": "string" }
      }
    }
  }
}
```

## Notes

- `operators_n2` holds exactly 4 keys: `aplicability`, `selection`, `exception`,
  `requeriments` (always with this English spelling in the code; only the `type`
  inside `properties_n3` uses Portuguese: `aplicabilidade`, `selecao`, `excecao`,
  `requisito`). An operator that does not apply is present with an empty
  `text_n2` and an empty `type`, so the four keys are always there.
- `properties_n3_rows` appears only on the 49 operators whose excerpt yields more
  than one N3 record (for example `"public use; collective use"`). In those cases
  `properties_n3` holds the first row, with the aggregated object joined by `"; "`,
  and `properties_n3_rows` holds every row. Validation reads `properties_n3`.
- The dataset version is in `meta.versao` (currently `3`): 76 standards, 150 atomic
  rules, from Section 6 of NBR 9050. `meta.fonte` records how v1 and v2 were merged
  and which duplicates were dropped.
- When the operator does not apply, `text_n2` is an empty string (`""`) and
  `properties_n3` keeps its structure but with empty strings.
- Files in `predicts/` follow the same format, with an extra `meta` block
  containing `model_id`, `temperature`, `seed`, `prompt_sha256`, etc.
