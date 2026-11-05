"""Concordância entre anotadores (instrumento para um segundo anotador).

Dois instrumentos, ambos cegos em relação à anotação existente:

  1. Referência N2: 40 regras sorteadas do dataset (semente 42), com o texto da
     norma e a regra N1. O segundo anotador preenche os quatro operadores RASE
     seguindo a convenção de Mendonça (2020). Mede a confiabilidade da referência.
  2. Rótulos da curva ROC: os 264 pares de results/analises/rotulos_manuais.csv,
     sem a coluna "correto" e em ordem embaralhada. O segundo anotador marca 1
     (preserva o conteúdo normativo) ou 0, pelo critério da Seção 4.9.2.

Uso:
    python -m analyses.concordancia preparar
    (o segundo anotador preenche os CSV em results/analises/concordancia/)
    python -m analyses.concordancia calcular

Medidas: kappa de Cohen sobre a presença de cada operador e sobre os rótulos dos
pares; F1 de tokens entre os trechos quando os dois anotadores preenchem o operador.
"""

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List

from analyses.common import OPERATORS, OUTPUT_DIR, load_dataset, token_f1, write_json

DIR = OUTPUT_DIR / "concordancia"
N2_FILE = DIR / "referencia_n2_segundo_anotador.csv"
PAIRS_FILE = DIR / "pares_roc_segundo_anotador.csv"
SAMPLE_RULES = 40
SEED = 42


def preparar() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()["datas"]
    rules = [(ti, si, item["text"], rule["text_n1"])
             for ti, item in enumerate(data) for si, rule in enumerate(item["texts_n1"])]
    random.Random(SEED).shuffle(rules)
    if N2_FILE.exists() or PAIRS_FILE.exists():
        sys.exit("Arquivos já existem; apague-os para sortear de novo.")
    with open(N2_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["norma", "regra", "texto_norma", "texto_n1", "aplicabilidade", "selecao", "excecao", "requisito"])
        for ti, si, text, n1 in sorted(rules[:SAMPLE_RULES]):
            w.writerow([ti, si, text, n1, "", "", "", ""])
    with open(OUTPUT_DIR / "rotulos_manuais.csv", encoding="utf-8-sig") as f:
        pairs = list(csv.DictReader(f, delimiter=";"))
    random.Random(SEED).shuffle(pairs)
    with open(PAIRS_FILE, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "nivel", "operador", "referencia", "previsto", "correto"])
        for p in pairs:
            w.writerow([p["id"], p["nivel"], p["operador"], p["referencia"], p["previsto"], ""])
    print(f"Instrumentos gravados em {DIR}")


def _kappa(a: List[int], b: List[int]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else None


def calcular() -> None:
    data = load_dataset()["datas"]
    cols = dict(zip(OPERATORS, ["aplicabilidade", "selecao", "excecao", "requisito"]))
    result: Dict[str, object] = {}

    with open(N2_FILE, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f, delimiter=";") if any(r[c].strip() for c in cols.values())]
    if rows:
        per_op = {}
        for op, col in cols.items():
            ref, ann, f1s = [], [], []
            for r in rows:
                t = data[int(r["norma"])]["texts_n1"][int(r["regra"])]["operators_n2"][op]["text_n2"].strip()
                q = r[col].strip()
                ref.append(int(bool(t)))
                ann.append(int(bool(q)))
                if t and q:
                    f1s.append(token_f1(t, q))
            per_op[op] = {"kappa_presenca": _kappa(ref, ann), "concordancia_presenca": sum(x == y for x, y in zip(ref, ann)) / len(ref),
                          "f1_tokens_medio": sum(f1s) / len(f1s) if f1s else None, "ambos_preenchidos": len(f1s)}
        result["referencia_n2"] = {"regras": len(rows), "por_operador": per_op}

    with open(PAIRS_FILE, encoding="utf-8-sig") as f:
        second = {r["id"]: r["correto"].strip() for r in csv.DictReader(f, delimiter=";") if r["correto"].strip()}
    if second:
        with open(OUTPUT_DIR / "rotulos_manuais.csv", encoding="utf-8-sig") as f:
            first = {r["id"]: r["correto"].strip() for r in csv.DictReader(f, delimiter=";")}
        ids = [i for i in second if i in first]
        a, b = [int(first[i]) for i in ids], [int(second[i]) for i in ids]
        result["rotulos_roc"] = {"pares": len(ids), "kappa": _kappa(a, b), "concordancia": sum(x == y for x, y in zip(a, b)) / len(ids)}

    if not result:
        sys.exit("Nenhum instrumento preenchido ainda.")
    write_json(OUTPUT_DIR / "concordancia.json", result)
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("comando", choices=["preparar", "calcular"])
    args = parser.parse_args()
    preparar() if args.comando == "preparar" else calcular()


if __name__ == "__main__":
    main()
