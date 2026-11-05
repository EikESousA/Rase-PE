_EMPTY_MARKERS = {'""', "''", "nenhum", "nenhuma", "n/a", "nao existe", "não existe", "-"}


def normalize_empty(value: str | None) -> str:
    """O prompt pede "" para operador ausente; o modelo as vezes devolve as aspas ou 'Nenhum'."""
    if not isinstance(value, str):
        return ""
    stripped = value.strip()
    return "" if stripped.rstrip(".").strip().lower() in _EMPTY_MARKERS else stripped
