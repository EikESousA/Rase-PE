"""Emparelha as regras atomicas previstas com as de referencia.

Nos experimentos encadeados o N1 vem do proprio modelo e pode ter outra
quantidade e outra ordem de regras. Comparar operadores pela posicao mistura
regras diferentes; por isso as regras sao emparelhadas pelo texto N1 antes da
comparacao. Quando os textos coincidem (experimentos isolados), o resultado e
o emparelhamento identidade.
"""

from typing import Any, Dict, List, Tuple

from utils.validates.align_pairs import align_lists


RulePair = Tuple[Dict[str, Any] | None, Dict[str, Any] | None]


def align_rules(
    target_rules: List[Dict[str, Any]],
    predicted_rules: List[Dict[str, Any]],
) -> List[RulePair]:
    target_texts = [r.get("text_n1", "") for r in target_rules]
    predicted_texts = [r.get("text_n1", "") for r in predicted_rules]

    if target_texts == predicted_texts:
        return list(zip(target_rules, predicted_rules))

    pairs: List[RulePair] = []
    for ti, pi in align_lists(target_texts, predicted_texts):
        target = target_rules[ti] if ti is not None else None
        predicted = predicted_rules[pi] if pi is not None else None
        pairs.append((target, predicted))
    return pairs
