"""Linhas de base nao-LLM para os experimentos isolados EN1, EN2 e EN3.

Cada linha de base gera predicoes no mesmo formato de
`predicts/generate_<nivel>_<modelo>.json` e passa pela MESMA rotina de
validacao dos LLMs (`utils.validates.run_validation`), com o mesmo
emparelhamento, as mesmas nove similaridades e o mesmo limiar de 0,7.

Linhas de base:

EN1 (entrada: texto bruto da norma)
    copia  - devolve o texto inteiro como uma unica regra (nao segmenta).
    regex  - divide o texto em sentencas pela pontuacao (tokenizador do NLTK).

EN2 (entrada: N1 de referencia)
    vazio  - deixa os quatro operadores vazios.
    copia  - devolve a regra inteira como requisito e deixa os demais vazios.
    regex  - divide a regra no primeiro verbo modal: o que vem antes e a
             aplicabilidade, o que vem depois e o requisito; a selecao e o
             primeiro modificador da aplicabilidade ("que", "com", "situado",
             "de uso", "em"); a excecao e a oracao iniciada por "exceto",
             "salvo" ou "com excecao de".

EN3 (entrada: N2 de referencia)
    vazio   - deixa todos os campos vazios.
    regex   - preenche type pelo operador, target e unit pelo primeiro numero
              do trecho, e comparation por palavras-chave ("maximo", "ate",
              "minimo", "inferior", "superior"); sem numero, usa "=" e
              VERDADEIRO/FALSO conforme a polaridade do trecho.
    maioria - para cada operador e campo, o valor nao vazio mais frequente na
              referencia, calculado deixando de fora a propria norma
              (leave-one-out). Usa a distribuicao da referencia e e, portanto,
              uma linha de base otimista.

Uso:
    python analyses/baselines.py
Saidas:
    predicts_baselines/generate_<nivel>_<baseline>.json
    metrics/baselines/validate_<nivel>.json
    results/analises/baselines.json (resumo usado nas tabelas)
"""

import json
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.n1.split_sentences import split_sentences  # noqa: E402
from utils.n2.empty_properties import empty_properties  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "dataset.json"
PREDICTS_DIR = PROJECT_ROOT / "predicts_baselines"
METRICS_DIR = PROJECT_ROOT / "metrics" / "baselines"
SUMMARY_PATH = PROJECT_ROOT / "results" / "analises" / "baselines.json"

OPERATORS = ["requeriments", "aplicability", "selection", "exception"]
FIELDS = ["object", "property", "comparation", "target", "unit"]
TYPE_LABEL = {
    "aplicability": "aplicabilidade",
    "selection": "selecao",
    "exception": "excecao",
    "requeriments": "requisito",
}

BASELINES = {
    "n1": ["copia", "regex"],
    "n2": ["vazio", "copia", "regex"],
    "n3": ["vazio", "regex", "maioria"],
}


def _empty_ops() -> Dict[str, Dict[str, Any]]:
    return {op: {"text_n2": "", "properties_n3": empty_properties()} for op in OPERATORS}


def _wrap(datas: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    return {"meta": {"model": f"baseline-{name}"}, "counts": len(datas), "datas": datas, "time": 0.0}


# ---------------------------------------------------------------- N1

def n1_predictions(dataset: Dict[str, Any], name: str) -> Dict[str, Any]:
    datas = []
    for item in dataset["datas"]:
        text = item.get("text", "").strip()
        rules = [text] if name == "copia" else split_sentences(text)
        datas.append({"text": text, "texts_n1": [{"text_n1": r, "operators_n2": _empty_ops()} for r in rules]})
    return _wrap(datas, name)


# ---------------------------------------------------------------- N2

_MODAL = re.compile(
    r"\b(n[ãa]o\s+)?(deve|devem|dever[áa]|dever[ãa]o|necessita|necessitam|precisa|precisam|"
    r"pode|podem|recomenda-se|[ée]\s+recomendad[oa]|[ée]\s+obrigat[óo]ri[oa])\b",
    flags=re.IGNORECASE,
)
_EXC = re.compile(r"\b(exceto|salvo|com\s+exce[çc][ãa]o\s+de|excetuando)\b[^,.;]*", flags=re.IGNORECASE)
_SEL = re.compile(r"\s(que|com|situad[oa]s?|de\s+uso|em)\s", flags=re.IGNORECASE)


def _regex_n2(rule: str) -> Dict[str, str]:
    out = {op: "" for op in OPERATORS}
    text = rule.strip().rstrip(".")
    exc = _EXC.search(text)
    if exc:
        out["exception"] = exc.group(0).strip(" ,")
        text = (text[: exc.start()] + text[exc.end():]).strip(" ,")
    modal = _MODAL.search(text)
    if not modal:
        out["requeriments"] = text
        return out
    subject = text[: modal.start()].strip(" ,")
    out["requeriments"] = text[modal.start():].strip(" ,")
    sel = _SEL.search(subject)
    if sel and sel.start() > 0:
        out["aplicability"] = subject[: sel.start()].strip(" ,")
        out["selection"] = subject[sel.start():].strip(" ,")
    else:
        out["aplicability"] = subject
    return out


def n2_predictions(dataset: Dict[str, Any], name: str) -> Dict[str, Any]:
    datas = []
    for item in dataset["datas"]:
        rules = []
        for ref_rule in item.get("texts_n1", []):
            rule = ref_rule.get("text_n1", "")
            ops = _empty_ops()
            if name == "copia":
                ops["requeriments"]["text_n2"] = rule.strip().rstrip(".")
            elif name == "regex":
                for op, value in _regex_n2(rule).items():
                    ops[op]["text_n2"] = value
            rules.append({"text_n1": rule, "operators_n2": ops})
        datas.append({"text": item.get("text", ""), "texts_n1": rules})
    return _wrap(datas, name)


# ---------------------------------------------------------------- N3

_NUM = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mm|cm|m|%|lux|kg)?(?![\w])", flags=re.IGNORECASE)
_LE = re.compile(r"m[áa]xim|\bat[ée]\b|n[ãa]o\s+(pode|deve)\s+exceder|no\s+m[áa]ximo", flags=re.IGNORECASE)
_GE = re.compile(r"m[íi]nim|igual\s+ou\s+superior|n[ãa]o\s+inferior", flags=re.IGNORECASE)
_LT = re.compile(r"inferior|menor", flags=re.IGNORECASE)
_GT = re.compile(r"superior|maior", flags=re.IGNORECASE)
_NEG = re.compile(r"\bn[ãa]o\b|\blivres?\b|\bsem\b|\bisent[oa]s?\b", flags=re.IGNORECASE)


def _format_number(raw: str) -> str:
    try:
        return f"{float(raw.replace(',', '.')):.2f}".replace(".", ",")
    except ValueError:
        return raw


def _regex_n3(text_n2: str, op: str) -> Dict[str, str]:
    props = empty_properties()
    if not text_n2.strip():
        return props
    props["type"] = TYPE_LABEL[op]
    num = _NUM.search(text_n2)
    if num:
        props["target"] = _format_number(num.group(1))
        props["unit"] = (num.group(2) or "").lower()
        if _GE.search(text_n2):
            props["comparation"] = ">="
        elif _LE.search(text_n2):
            props["comparation"] = "<="
        elif _LT.search(text_n2):
            props["comparation"] = "<"
        elif _GT.search(text_n2):
            props["comparation"] = ">"
        else:
            props["comparation"] = "="
    else:
        props["comparation"] = "="
        props["target"] = "FALSO" if _NEG.search(text_n2) else "VERDADEIRO"
    return props


def _majority_tables(dataset: Dict[str, Any]) -> List[Dict[tuple, str]]:
    """Valor nao vazio mais frequente por (operador, campo), excluindo cada norma."""
    per_item: List[Counter] = []
    total: Dict[tuple, Counter] = defaultdict(Counter)
    for item in dataset["datas"]:
        local: Dict[tuple, Counter] = defaultdict(Counter)
        for rule in item.get("texts_n1", []):
            for op in OPERATORS:
                data = rule["operators_n2"][op]
                if not data.get("text_n2", "").strip():
                    continue
                for field in FIELDS:
                    value = (data.get("properties_n3") or {}).get(field, "")
                    if value:
                        local[(op, field)][value] += 1
                        total[(op, field)][value] += 1
        per_item.append(local)
    tables = []
    for local in per_item:
        table = {}
        for key, counts in total.items():
            rest = counts - local.get(key, Counter())
            if rest:
                table[key] = rest.most_common(1)[0][0]
        tables.append(table)
    return tables


def n3_predictions(dataset: Dict[str, Any], name: str) -> Dict[str, Any]:
    tables = _majority_tables(dataset) if name == "maioria" else None
    datas = []
    for index, item in enumerate(dataset["datas"]):
        rules = []
        for ref_rule in item.get("texts_n1", []):
            ops = deepcopy(ref_rule["operators_n2"])
            for op in OPERATORS:
                text_n2 = ops[op].get("text_n2", "")
                if name == "vazio" or not text_n2.strip():
                    props = empty_properties()
                elif name == "regex":
                    props = _regex_n3(text_n2, op)
                else:
                    props = empty_properties()
                    props["type"] = TYPE_LABEL[op]
                    for field in FIELDS:
                        props[field] = tables[index].get((op, field), "")
                ops[op]["properties_n3"] = props
            rules.append({"text_n1": ref_rule.get("text_n1", ""), "operators_n2": ops})
        datas.append({"text": item.get("text", ""), "texts_n1": rules})
    return _wrap(datas, name)


# ---------------------------------------------------------------- validacao

BUILDERS = {"n1": n1_predictions, "n2": n2_predictions, "n3": n3_predictions}


def _summary(metrics_path: Path, level: str) -> Dict[str, Any]:
    with open(metrics_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)
    out: Dict[str, Any] = {}
    for name, payload in metrics.get("models", {}).items():
        cls = payload.get("classification", {})
        entry: Dict[str, Any] = {"similarity": metrics.get("averages", {}).get(name, {})}
        if level == "n1":
            entry["classification"] = cls.get("overall")
        elif level == "n2":
            entry["classification"] = cls.get("macro")
            entry["by_operator"] = {op: m.get("f1") for op, m in cls.get("by_operator", {}).items()}
        else:
            entry["classification"] = cls.get("macro")
            entry["by_field"] = {f: m.get("f1") for f, m in cls.get("by_field", {}).items()}
        out[name] = entry
    return out


TABLES_DIR = PROJECT_ROOT / "dissertacao" / "Conteudo" / "Tabelas"
BASELINE_LABELS = {
    "copia": "Linha de base: cópia",
    "regex": "Linha de base: regras",
    "vazio": "Linha de base: vazio",
    "maioria": "Linha de base: maioria",
}
LEVEL_EXPERIMENT = {"n1": "EN1", "n2": "EN2", "n3": "EN3"}


def _fmt(value: float | None) -> str:
    return "---" if value is None else f"{value:.3f}".replace(".", "{,}")


def _llm_metrics(level: str) -> Dict[str, Dict[str, Any]]:
    with open(PROJECT_ROOT / "metrics" / f"validate_{level}.json", "r", encoding="utf-8") as file:
        models = json.load(file)["models"]
    out = {}
    for name, payload in models.items():
        cls = payload["classification"]
        out[name] = {
            "classification": cls.get("overall") if level == "n1" else cls.get("macro"),
            "by_operator": {k: v["f1"] for k, v in cls.get("by_operator", {}).items()},
            "by_field": {k: v["f1"] for k, v in cls.get("by_field", {}).items()},
        }
    return out


def write_tables(summary: Dict[str, Any]) -> None:
    """Escreve as tabelas de linhas de base usadas no Capitulo 5."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Linhas de base sem LLM comparadas ao melhor e ao pior LLM em cada experimento isolado. "
        r"N1: métricas sobre as regras; N2: média macro sobre os quatro operadores; N3: média macro sobre os cinco campos. "
        r"Todas as linhas passam pela mesma rotina de validação, com o mesmo emparelhamento e o mesmo limiar de 0{,}7.}",
        r"\label{tab:baselines}",
        r"\footnotesize",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Experimento & Procedimento & Acurácia & Precisão & Revocação & F1 \\",
        r"\hline",
    ]
    for level in ("n1", "n2", "n3"):
        llms = _llm_metrics(level)
        ranked = sorted(llms.items(), key=lambda kv: kv[1]["classification"]["f1"], reverse=True)
        rows = [(BASELINE_LABELS[n], summary[level][n]["classification"]) for n in BASELINES[level]]
        best, worst = ranked[0], ranked[-1]
        rows.append((f"Melhor LLM ({best[0].capitalize()})", best[1]["classification"]))
        rows.append((f"Pior LLM ({worst[0].capitalize()})", worst[1]["classification"]))
        for index, (label, cls) in enumerate(rows):
            exp = LEVEL_EXPERIMENT[level] if index == 0 else ""
            values = " & ".join(_fmt(cls.get(k)) for k in ("accuracy", "precision", "recall", "f1"))
            lines.append(f"{exp} & {label} & {values} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    (TABLES_DIR / "baselines.tex").write_text("\n".join(lines), encoding="utf-8")

    ops = [("requeriments", "R"), ("aplicability", "A"), ("selection", "S"), ("exception", "E")]
    fields = [("object", "\\textit{object}"), ("property", "\\textit{property}"),
              ("comparation", "\\textit{comp.}"), ("target", "\\textit{target}"), ("unit", "\\textit{unit}")]
    n2_llm, n3_llm = _llm_metrics("n2"), _llm_metrics("n3")
    detail = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{F1 por operador no EN2 e por campo no EN3: linhas de base sem LLM e os seis LLMs. O traço indica linha de base não aplicável ao experimento.}",
        r"\label{tab:baselines_detalhe}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "r" * (len(ops) + len(fields)) + "}",
        r"\hline",
        r" & \multicolumn{4}{c}{EN2 (operador)} & \multicolumn{5}{c}{EN3 (campo)} \\",
        r"\cline{2-5}\cline{6-10}",
        "Procedimento & " + " & ".join(o[1] for o in ops) + " & " + " & ".join(f[1] for f in fields) + r" \\",
        r"\hline",
    ]
    names = ["vazio", "copia", "regex", "maioria"]
    for name in names:
        n2 = summary["n2"].get(name, {}).get("by_operator")
        n3 = summary["n3"].get(name, {}).get("by_field")
        cells = [_fmt(n2.get(o)) if n2 else "---" for o, _ in ops]
        cells += [_fmt(n3.get(f)) if n3 else "---" for f, _ in fields]
        detail.append(f"{BASELINE_LABELS[name]} & " + " & ".join(cells) + r" \\")
    detail.append(r"\hline")
    for model in sorted(n2_llm):
        cells = [_fmt(n2_llm[model]["by_operator"].get(o)) for o, _ in ops]
        cells += [_fmt(n3_llm[model]["by_field"].get(f)) for f, _ in fields]
        detail.append(f"{model.capitalize()} & " + " & ".join(cells) + r" \\")
    detail += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (TABLES_DIR / "baselines_detalhe.tex").write_text("\n".join(detail), encoding="utf-8")
    print("Tabelas de linhas de base escritas em", TABLES_DIR)


def main() -> None:
    if "--tables-only" in sys.argv:
        with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
            write_tables(json.load(file))
        return
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)
    PREDICTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for level, names in BASELINES.items():
        for name in names:
            path = PREDICTS_DIR / f"generate_{level}_{name}.json"
            with open(path, "w", encoding="utf-8") as file:
                json.dump(BUILDERS[level](dataset, name), file, ensure_ascii=False, indent=2)

    import utils.validates.run_validation as rv  # noqa: E402
    from utils.validates.build_pairs import build_pairs  # noqa: E402
    from utils.validates.build_pairs_n2 import build_pairs_n2  # noqa: E402
    from utils.validates.build_pairs_n3 import build_pairs_n3  # noqa: E402

    builders = {"n1": build_pairs, "n2": build_pairs_n2, "n3": build_pairs_n3}
    summary: Dict[str, Any] = {}
    for level, names in BASELINES.items():
        rv.MODEL_NAMES = names
        output = METRICS_DIR / f"validate_{level}.json"
        rv.run_validation(
            level=level,
            build_pairs_fn=builders[level],
            dataset_path=str(DATASET_PATH),
            predicts_dir=str(PREDICTS_DIR),
            output_path=str(output),
        )
        summary[level] = _summary(output, level)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(f"Resumo salvo em {SUMMARY_PATH}")
    write_tables(summary)


if __name__ == "__main__":
    main()
