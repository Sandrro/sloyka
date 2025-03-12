import warnings
warnings.filterwarnings("ignore")

from natasha.extractors import Match
from natasha.extractors import Extractor

from ..utils.data_processing.rule_for_natasha import ADDR_PART


class AddrExtractorError(Exception):
    """Custom exception for address extractor errors"""

    pass


class AddressExtractorExtra(Extractor):
    def __init__(self, morph):
        Extractor.__init__(self, ADDR_PART, morph)

    def find(self, text):
        matches = self(text)
        if not matches:
            return

        matches = sorted(matches, key=lambda _: _.start)
        if not matches:
            return
        start = matches[0].start
        stop = matches[-1].stop
        parts = [_.fact for _ in matches]
        return Match(start, stop, obj.Addr(parts))