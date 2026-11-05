"""Aplica a conferencia dos rotulos da ROC e registra quem respondeu por ela.

Fluxo:
  1. `python -m tools.preparar_revisao` gera results/analises/rotulos_para_revisao.csv,
     ordenado por quanto o julgamento pesou e ja com a coluna `seu_rotulo` preenchida
     com o rotulo proposto automaticamente.
  2. Voce le o arquivo e corrige em `seu_rotulo` o que discordar.
  3. `python -m tools.aplicar_revisao --conferido "Seu Nome"` reescreve
     results/analises/rotulos_manuais.csv e grava a procedencia em
     results/analises/rotulos_procedencia.json.

O flag `--conferido` e obrigatorio: como a coluna vem pre-preenchida, o arquivo
sozinho nao distingue o que foi conferido do que nao foi, e e essa atestacao que
autoriza a dissertacao a chamar o gabarito de humano.
"""

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALISES = PROJECT_ROOT / "results" / "analises"
REVISAO = ANALISES / "rotulos_para_revisao.csv"
DESTINO = ANALISES / "rotulos_manuais.csv"
PROCEDENCIA = ANALISES / "rotulos_procedencia.json"
COLUNAS = ["id", "nivel", "operador", "referencia", "previsto", "correto"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conferido", metavar="NOME", required=True,
                        help="Nome de quem conferiu os rotulos par a par. Fica registrado "
                             "em rotulos_procedencia.json e e o que sustenta a afirmacao "
                             "de gabarito humano na dissertacao.")
    parser.add_argument("--criterio", default="adequacao semantica",
                        help="Criterio aplicado (padrao: adequacao semantica).")
    args = parser.parse_args()

    if not REVISAO.exists():
        raise SystemExit(f"Arquivo nao encontrado: {REVISAO}\n"
                         "Rode antes: python -m tools.preparar_revisao")

    linhas, mudancas, invalidos = [], 0, 0
    with REVISAO.open(encoding="utf-8-sig") as arquivo:
        for row in csv.DictReader(arquivo, delimiter=";"):
            seu = (row.get("seu_rotulo") or "").strip()
            if seu not in {"0", "1"}:
                invalidos += 1
                continue
            if seu != (row.get("correto") or "").strip():
                mudancas += 1
            linhas.append({**{c: row[c] for c in COLUNAS[:-1]}, "correto": seu})

    if not linhas:
        raise SystemExit("Nenhum rotulo valido em `seu_rotulo`.")

    with DESTINO.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, delimiter=";", fieldnames=COLUNAS)
        escritor.writeheader()
        escritor.writerows(linhas)

    positivos = sum(1 for l in linhas if l["correto"] == "1")
    niveis = Counter(l["nivel"] for l in linhas)
    PROCEDENCIA.write_text(json.dumps({
        "conferido_por": args.conferido,
        "data": date.today().isoformat(),
        "criterio": args.criterio,
        "pares": len(linhas),
        "corretos": positivos,
        "por_nivel": dict(niveis),
        "alterados_na_conferencia": mudancas,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{DESTINO}: {len(linhas)} pares ({positivos} corretos, "
          f"{len(linhas) - positivos} incorretos)")
    print(f"  por nivel: {dict(niveis)}")
    print(f"  alterados na conferencia: {mudancas}")
    if invalidos:
        print(f"  ignorados (sem rotulo valido): {invalidos}")
    print(f"  procedencia registrada em {PROCEDENCIA.name} (conferido por {args.conferido})")
    print("\nAgora rode:  python -m analyses.threshold_analysis && python -m tools.chapter_tables")


if __name__ == "__main__":
    main()
