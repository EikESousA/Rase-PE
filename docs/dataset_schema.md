# `dataset.json` Schema

The dataset follows a nested structure with 3 levels (N1/N2/N3). Each item in `datas`
holds the original `text` and the reference RASE decomposition.

## Structure

```json
{
  "counts": 79,
  "datas": [
    {
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
            "selection":    { "text_n2": "...", "properties_n3": { ... } },
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
        "text_n2":      { "type": "string" },
        "properties_n3":{ "$ref": "#/$defs/n3Props" }
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

- `operators_n2` holds up to 4 keys: `aplicability`, `selection`, `exception`,
  `requeriments` (always with this English spelling in the code; only the `type`
  inside `properties_n3` uses Portuguese: `aplicabilidade`, `selecao`, `execcao`,
  `requisito`). Note: the exception value is spelled `execcao` (not `excecao`) in
  `dataset.json`; use exactly that spelling when comparing.
- When the operator does not apply, `text_n2` is an empty string (`""`) and
  `properties_n3` keeps its structure but with empty strings.
- Files in `predicts/` follow the same format, with an extra `meta` block
  containing `model_id`, `temperature`, `seed`, `prompt_sha256`, etc.
