"""Avaliacao complementar com bibliotecas de referencia da area.

- sacreBLEU (Post, 2018): chrF++, BLEU e TER no nivel de corpus, com intervalo de
  confianca de 95% por bootstrap sobre as sentencas.
- RapidFuzz: similaridade de Levenshtein normalizada e token-set ratio.
- Pydantic: validacao do esquema dos registros N3 (tipo, comparador e coerencia).

Os pares seguem o mesmo emparelhamento da validacao (regras alinhadas pelo texto
N1). Pares em que referencia ou saida estao vazias ficam fora das metricas de texto.

Uso:
    python -m analyses.library_metrics
"""

import argparse
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, ValidationError, field_validator, model_validator
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from sacrebleu.metrics import BLEU, CHRF, TER

from analyses.common import (
    EXPERIMENTS,
    OUTPUT_DIR,
    check_isolation,
    exclude_contaminated,
    iter_models,
    load_dataset,
    load_predictions,
    normalize,
    write_json,
)
from utils.validates.align_pairs import align_lists
from utils.validates.build_pairs_n2 import build_pairs_n2
from utils.validates.build_pairs_n3 import build_pairs_n3

BOOTSTRAP_SAMPLES = 1000
SEED = 42

TYPE_BY_OPERATOR = {
    "aplicability": {"aplicabilidade"},
    "selection": {"selecao"},
    "requeriments": {"requisito"},
    "exception": {"excecao", "execao", "execcao"},
}
ALLOWED_COMPARATORS = {"", "=", "contem", "inclui", "<=", ">=", "<", ">", "!="}
_SYMBOLS = {"≤": "<=", "≥": ">=", "≠": "!="}


def _norm_comparator(value: str) -> str:
    value = (value or "").strip()
    for symbol, ascii_ in _SYMBOLS.items():
        value = value.replace(symbol, ascii_)
    value = unicodedata.normalize("NFKD", value).encode("ASCII", "ignore").decode("ASCII")
    return value.lower().strip()


class RaseRecord(BaseModel):
    operator: str
    type: str
    object: str
    property: str
    comparation: str
    target: str
    unit: str

    @field_validator("comparation")
    @classmethod
    def comparator_in_vocabulary(cls, value: str) -> str:
        if _norm_comparator(value) not in ALLOWED_COMPARATORS:
            raise ValueError("comparador_fora_do_vocabulario")
        return value

    @model_validator(mode="after")
    def coherent(self) -> "RaseRecord":
        filled = [normalize(getattr(self, f)) for f in ("object", "property", "comparation", "target", "unit")]
        type_ = normalize(self.type)
        if not type_ and any(filled):
            raise ValueError("atributos_sem_tipo")
        if type_ and type_ not in TYPE_BY_OPERATOR.get(self.operator, set()):
            raise ValueError("tipo_incompativel_com_o_operador")
        has_comp, has_target = bool(_norm_comparator(self.comparation)), bool(normalize(self.target))
        if has_comp != has_target:
            raise ValueError("comparador_e_alvo_incoerentes")
        return self


def validate_record(operator: str, props: Dict[str, Any]) -> str | None:
    data = {k: props.get(k, "") if isinstance(props.get(k, ""), str) else str(props.get(k)) for k in ("type", "object", "property", "comparation", "target", "unit")}
    try:
        RaseRecord(operator=operator, **data)
        return None
    except ValidationError as exc:
        messages = [str(e.get("ctx", {}).get("error", e.get("msg", ""))) for e in exc.errors()]
        return messages[0].replace("Value error, ", "") if messages else "invalido"


def _text_metrics(pairs: List[Tuple[str, str]]) -> Dict[str, Any] | None:
    pairs = [(r, h) for r, h in pairs if r.strip() and h.strip()]
    if not pairs:
        return None
    refs = [r for r, _ in pairs]
    hyps = [h for _, h in pairs]
    chrf, bleu, ter = CHRF(word_order=2), BLEU(effective_order=True), TER()
    sentence_chrf = np.array([chrf.sentence_score(h, [r]).score for r, h in pairs])
    rng = np.random.default_rng(SEED)
    boots = [sentence_chrf[rng.integers(0, len(pairs), len(pairs))].mean() for _ in range(BOOTSTRAP_SAMPLES)]
    return {
        "n": len(pairs),
        "chrf++": chrf.corpus_score(hyps, [refs]).score / 100,
        "chrf++_sentenca_media": float(sentence_chrf.mean()) / 100,
        "chrf++_ic95": [float(np.percentile(boots, 2.5)) / 100, float(np.percentile(boots, 97.5)) / 100],
        "bleu": BLEU().corpus_score(hyps, [refs]).score / 100,
        "bleu_sentenca_media": float(np.mean([bleu.sentence_score(h, [r]).score for r, h in pairs])) / 100,
        "ter": ter.corpus_score(hyps, [refs]).score / 100,
        "levenshtein_norm": float(np.mean([Levenshtein.normalized_similarity(r, h) for r, h in pairs])),
        "token_set_ratio": float(np.mean([fuzz.token_set_ratio(r, h) for r, h in pairs])) / 100,
    }


def _pairs_n1(dataset: Dict[str, Any], predictions: Dict[str, Any]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    items = predictions.get("datas", [])
    for idx, item in enumerate(dataset.get("datas", [])):
        ref = [r.get("text_n1", "") for r in item.get("texts_n1", [])]
        pred = [r.get("text_n1", "") for r in (items[idx] if idx < len(items) else {}).get("texts_n1", [])]
        out.extend((ref[t], pred[p]) for t, p in align_lists(ref, pred) if t is not None and p is not None)
    return out


def _schema_report(predictions: Dict[str, Any]) -> Dict[str, Any]:
    errors: Counter = Counter()
    total = 0
    for item in predictions.get("datas", []):
        for rule in item.get("texts_n1", []):
            for operator, data in (rule.get("operators_n2") or {}).items():
                props = (data or {}).get("properties_n3") or {}
                total += 1
                problem = validate_record(operator, props)
                errors[problem or "valido"] += 1
    return {"registros": total, "validos": errors.pop("valido", 0), "erros": dict(errors)}


def run(experiment_keys: List[str], models: List[str], suffix: str = "") -> Dict[str, Any]:
    dataset = load_dataset()
    output: Dict[str, Any] = {"experimentos": {}, "avisos": [], "referencia_n3": _schema_report(dataset)}
    for key in experiment_keys:
        experiment = EXPERIMENTS[key]
        for model in models:
            predictions = load_predictions(experiment, model)
            if predictions is None:
                continue
            problem = check_isolation(experiment, dataset, predictions)
            if problem:
                output["avisos"].append(f"{problem} (modelo {model})")
                continue
            print(f"Bibliotecas {experiment.label} / {model}...", flush=True)
            res: Dict[str, Any] = {}
            if "n1" in experiment.levels:
                res["n1"] = _text_metrics(_pairs_n1(dataset, predictions))
            if "n2" in experiment.levels:
                res["n2"] = _text_metrics([(p["target"], p["predicted"]) for p in build_pairs_n2(dataset, predictions)])
            if "n3" in experiment.levels:
                pairs3 = build_pairs_n3(dataset, predictions)
                res["n3"] = _text_metrics([(p["target"], p["predicted"]) for p in pairs3])
                res["n3_esquema"] = _schema_report(predictions)
            output["experimentos"].setdefault(key, {})[model] = res
    for w in output["avisos"]:
        print("AVISO:", w)
    path = OUTPUT_DIR / f"metricas_bibliotecas{suffix}.json"
    write_json(path, output)
    print(f"Resultado salvo em {path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--sem-contaminacao", action="store_true", help="Exclui as normas usadas como exemplo nos prompts.")
    args = parser.parse_args()
    suffix = exclude_contaminated(args.sem_contaminacao)
    run(args.experiments, iter_models(args.models), suffix)


if __name__ == "__main__":
    main()
