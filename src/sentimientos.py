from pysentimiento import create_analyzer
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer

class AnalizadorSentimientosPipeline:
    def __init__(self, idioma):
        self.idioma = idioma
        if idioma in ['es', 'en']:
            self.analyzer = create_analyzer(task="sentiment", lang=idioma)

    def analizar(self, texto):
        if not isinstance(texto, str) or texto.strip() == "":
            return "NEU"
            
        if self.idioma in ['es', 'en']:
            try:
                res = self.analyzer.predict(texto)
                return res.output # devuelve 'POS', 'NEU', 'NEG'
            except Exception:
                return "NEU"
                
        elif self.idioma == 'fr':
            # polaridad de -1 a 1
            blob = TextBlob(texto, pos_tagger=PatternTagger(), analyzer=PatternAnalyzer())
            polaridad = blob.sentiment[0]
            if polaridad > 0.05: return "POS"
            elif polaridad < -0.05: return "NEG"
            else: return "NEU"
        return "NEU"