from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
from natasha.extractors import Extractor
import natasha.obj as obj
import pandas as pd
from natasha.extractors import Match
import pdb

class AddrNEWExtractor(Extractor):
    def __init__(self, morph):
        from factfinder.src.addrNEW import ADDR_PART
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

class NER_parklike:
    @staticmethod
    def extract_parklike(text):
        """
        The function extracts parklike entities in the text, using the Natasha library.
        """
        morph = MorphVocab()
        extractor = AddrNEWExtractor(morph)

        data = {'start': "", 'stop': "", 'toponim': "", 'toponim_type': ""}

        match = extractor.find(text)
        if not match:
            return pd.Series([data['start'], data['stop'], data['toponim'], data['toponim_type']])

        data['start'] = match.start
        data['stop'] = match.stop
        data['toponim'] = match.fact.parts[0].value
        data['toponim_type'] = match.fact.parts[0].type
        return pd.Series([data['start'], data['stop'], data['toponim'], data['toponim_type']])
        

    @staticmethod
    def natasha_normalization(text, start_ind, stop_ind):
        """
        This function extracts entities and normalize them. For entities that match the 
        extraction of "extract_parklike", it returns a normalized form.
        """
        normalized_form = ""
        if start_ind != "":
            return normalized_form
        
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        doc.tag_ner(ner_tagger)
        for span in doc.spans:
            if span.start == start_ind and span.stop == stop_ind:
                span.normalize(morph_vocab)
                normalized_form = span.normal
        return normalized_form
    
    def run(self,df, text_column):
        df[["Start", "Stop", "Toponim", "Toponim_type"]] = df[text_column].apply(lambda text: NER_parklike.extract_parklike(text) if NER_parklike.extract_parklike(text) is not None else pd.Series(["", "", "", ""]))
        df["Normal"] = df.apply(lambda row: NER_parklike.natasha_normalization(row[text_column], row["Start"], row["Stop"]),axis=1)
        

        
                     
