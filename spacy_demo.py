import spacy
import coref_model_wrapped
import coref_pipe

nlp = spacy.load("en_core_web_sm")

nlp.add_pipe("coref")

text = "John called from London, he says it's raining in the city."
doc = nlp(text)

for key, val in doc.spans:
    print(key, val)
