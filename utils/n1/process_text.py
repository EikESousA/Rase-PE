import re
from typing import List

from utils.n1.clean_output import clean_output
from utils.n1.split_sentences import split_sentences


MIN_WORDS = 3


def process_text(text: str) -> List[str]:
    cleaned: str = clean_output(text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Restos de numeracao ("2.") e fragmentos sem verbo nao sao regras.
    return [s for s in split_sentences(cleaned) if len(re.findall(r"\w+", s)) >= MIN_WORDS]
