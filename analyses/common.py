import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.models import MODEL_NAMES  # noqa: E402

DATASET_PATH = PROJECT_ROOT / "dataset.json"
PREDICTS_DIR = PROJECT_ROOT / "predicts"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analises"
THRESHOLD = 0.7
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

OPERATORS: List[str] = ["aplicability", "selection", "exception", "requeriments"]
OPERATOR_LABELS: Dict[str, str] = {
    "aplicability": "Aplicabilidade",
    "selection": "Seleção",
    "exception": "Exceção",
    "requeriments": "Requisito",
}
FIELDS: List[str] = ["object", "property", "comparation", "target", "unit"]
MODEL_LABELS: Dict[str, str] = {m: m.capitalize() for m in MODEL_NAMES}


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    prefix: str
    levels: Tuple[str, ...]
    isolated_from: str | None


EXPERIMENTS: Dict[str, Experiment] = {
    "en1": Experiment("en1", "EN1", "generate_n1", ("n1",), None),
    "en2": Experiment("en2", "EN2", "generate_n2", ("n2",), "n1"),
    "en1n2": Experiment("en1n2", "EN1N2", "generate_n1n2", ("n1", "n2"), None),
    "en3": Experiment("en3", "EN3", "generate_n3", ("n3",), "n2"),
    "en1n2n3": Experiment("en1n2n3", "EN1N2N3", "generate_n1n2n3", ("n1", "n2", "n3"), None),
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


_EXCLUDED: List[int] = []


def exclude_contaminated(enabled: bool) -> str:
    """Ativa a exclusao das normas usadas como exemplo nos prompts; devolve o sufixo dos arquivos."""
    _EXCLUDED.clear()
    if enabled:
        _EXCLUDED.extend(contaminated_items(load_json(DATASET_PATH)))
        print(f"Excluindo {len(_EXCLUDED)} normas usadas como exemplo nos prompts: {_EXCLUDED}")
        return "_sem_contaminacao"
    return ""


def load_dataset() -> Dict[str, Any]:
    return drop_items(load_json(DATASET_PATH), _EXCLUDED)


def load_predictions(experiment: Experiment, model: str) -> Dict[str, Any] | None:
    path = PREDICTS_DIR / f"{experiment.prefix}_{model}.json"
    if not path.exists():
        return None
    return drop_items(load_json(path), _EXCLUDED)


def _n1_signature(data: Dict[str, Any]) -> List[List[str]]:
    return [[r.get("text_n1", "") for r in it.get("texts_n1", [])] for it in data.get("datas", [])]


def _n2_signature(data: Dict[str, Any]) -> List[List[Dict[str, str]]]:
    return [
        [
            {op: (r.get("operators_n2", {}).get(op, {}) or {}).get("text_n2", "") for op in OPERATORS}
            for r in it.get("texts_n1", [])
        ]
        for it in data.get("datas", [])
    ]


def check_isolation(experiment: Experiment, dataset: Dict[str, Any], predictions: Dict[str, Any]) -> str | None:
    """Retorna uma mensagem de erro se um experimento isolado nao recebeu a referencia."""
    if experiment.isolated_from is None:
        return None
    if _n1_signature(predictions) != _n1_signature(dataset):
        return f"{experiment.label}: o N1 das predicoes difere do N1 de referencia; regenere com dataset.json como entrada."
    if experiment.isolated_from == "n2" and _n2_signature(predictions) != _n2_signature(dataset):
        return f"{experiment.label}: o N2 das predicoes difere do N2 de referencia; regenere com dataset.json como entrada."
    return None


PROMPTS_DIR = PROJECT_ROOT / "prompts"
_PROMPT_FILES = ["n1.txt", "n2.txt", "n3_aplicabilidade.txt", "n3_requisito.txt", "n3_selecao.txt", "n3_execao.txt"]


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", normalize(text))


_EXAMPLE_HEADERS = re.compile(r"^(entrada|texto n1|operador n2 \(.*\)):\s*$", re.IGNORECASE)
_EXAMPLE_ANSWER = re.compile(r"^(aplicabilidade|selecao|execao|requisito):\s*(.+)$", re.IGNORECASE)


def prompt_examples() -> List[str]:
    """Textos usados como exemplo dentro dos prompts (entradas e respostas de demonstracao)."""
    examples: List[str] = []
    for name in _PROMPT_FILES:
        path = PROMPTS_DIR / name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            nxt = lines[i + 1].strip().strip('"') if i + 1 < len(lines) else ""
            if _EXAMPLE_HEADERS.match(line.strip()) and nxt and "{" not in nxt and "..." not in nxt:
                examples.append(nxt)
            answer = _EXAMPLE_ANSWER.match(line.strip())
            if answer and answer.group(2).strip() not in {'""', ""}:
                examples.append(answer.group(2).strip())
    return examples


def contaminated_items(dataset: Dict[str, Any]) -> List[int]:
    """Normas cujo texto, regra ou operador aparece como exemplo dentro dos prompts."""
    examples = {_squash(e) for e in prompt_examples() if _squash(e)}
    found: List[int] = []
    for idx, item in enumerate(dataset.get("datas", [])):
        text = _squash(item.get("text", ""))
        hit = any(len(e) > 25 and e in text for e in examples)
        for rule in item.get("texts_n1", []):
            if _squash(rule.get("text_n1", "")) in examples:
                hit = True
            for op in OPERATORS:
                if _squash((rule.get("operators_n2", {}).get(op, {}) or {}).get("text_n2", "")) in examples:
                    hit = True
        if hit:
            found.append(idx)
    return found


def drop_items(data: Dict[str, Any], indices: Iterable[int]) -> Dict[str, Any]:
    skip = set(indices)
    return {**data, "datas": [it for i, it in enumerate(data.get("datas", [])) if i not in skip]}


def iter_models(models: Iterable[str] | None = None) -> List[str]:
    return list(models) if models else list(MODEL_NAMES)


def normalize(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("≥", ">=").replace("≤", "<=").replace("≠", "!=")
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
    return text.lower().strip().rstrip(".,;:")


def tokens(text: str) -> List[str]:
    return re.findall(r"\w+", normalize(text))


def token_f1(reference: str, predicted: str) -> float:
    ref, pred = tokens(reference), tokens(predicted)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    remaining = list(ref)
    common = 0
    for tok in pred:
        if tok in remaining:
            remaining.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def overlap_coefficient(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def support_in_source(predicted: str, source: str) -> float:
    """Fracao dos tokens previstos que aparecem no texto de origem."""
    pred = tokens(predicted)
    if not pred:
        return 1.0
    src = set(tokens(source))
    return sum(1 for t in pred if t in src) / len(pred)


class Similarity:
    """Similaridade de cosseno com o mesmo modelo usado na validacao, com cache."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._cache: Dict[str, Any] = {}

    def _encode(self, texts: Iterable[str]) -> None:
        missing = sorted({t for t in texts if t and t not in self._cache})
        if not missing:
            return
        vectors = self._model.encode(
            missing, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
        for text, vec in zip(missing, vectors):
            self._cache[text] = vec

    def pairs(self, pairs: List[Tuple[str, str]]) -> List[float | None]:
        self._encode([p for pair in pairs for p in pair])
        scores: List[float | None] = []
        for a, b in pairs:
            if not (a and a.strip() and b and b.strip()):
                scores.append(None)
                continue
            scores.append(float((self._cache[a] * self._cache[b]).sum()))
        return scores

    def one(self, a: str, b: str) -> float | None:
        return self.pairs([(a, b)])[0]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "---"
    return f"{value:.{digits}f}".replace(".", "{,}")
