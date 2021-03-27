from typing import List, Tuple, Callable

import spacy
from spacy.tokens import Doc, Span
from spacy.util import registry
from thinc.types import Floats2d, Ints2d, cast
from thinc.api import Model, PyTorchWrapper, ArgsKwargs, require_gpu, fix_random_seed, chain
fix_random_seed(23)
require_gpu()
import thinc
import torch
import numpy as np


from model import CorefModel
from util import initialize_config, flatten
from tensorize import CorefDataProcessor
from preprocess import get_document

# from model.py, refactored to be non-member
def get_predicted_antecedents(antecedent_idx, antecedent_scores):
    """Get the ID of the antecedent for each span. -1 if no antecedent."""
    predicted_antecedents = []
    for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
        if idx < 0:
            predicted_antecedents.append(-1)
        else:
            predicted_antecedents.append(antecedent_idx[i][idx])
    return predicted_antecedents

# from model.py, refactored to be non-member
def get_predicted_clusters(span_starts, span_ends, antecedent_idx, antecedent_scores):
    """Convert predictions to usable cluster data.

    return values:

    clusters: a list of spans (i, j) that are a cluster

    Note that not all spans will be in the final output; spans with no
    antecedent or referrent are omitted from clusters and mention2cluster.
    """
    # Get predicted antecedents
    predicted_antecedents = get_predicted_antecedents(antecedent_idx, antecedent_scores)

    # Get predicted clusters
    mention_to_cluster_id = {}
    predicted_clusters = []
    for i, predicted_idx in enumerate(predicted_antecedents):
        if predicted_idx < 0:
            continue
        assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
        # Check antecedent's cluster
        antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
        antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
        if antecedent_cluster_id == -1:
            antecedent_cluster_id = len(predicted_clusters)
            predicted_clusters.append([antecedent])
            mention_to_cluster_id[antecedent] = antecedent_cluster_id
        # Add mention to cluster
        mention = (int(span_starts[i]), int(span_ends[i]))
        predicted_clusters[antecedent_cluster_id].append(mention)
        mention_to_cluster_id[mention] = antecedent_cluster_id

    predicted_clusters = [tuple(c) for c in predicted_clusters]
    return predicted_clusters

def make_conll_lines(doc, genre="nw"):
    """Convert a doc into the conll line-based format.

    This is weird but it's used internally by the coref model. Factored out of
    `get_document_from_string`.
    """
    doc_lines = []

    for token in doc:
        cols = [genre] + ['-'] * 11
        cols[3] = token.text
        doc_lines.append('\t'.join(cols))
        if token.is_sent_end:
            doc_lines.append('\n')
    return doc_lines

@thinc.registry.layers("coref_preprocessor.v1")
def CorefPreprocessor(name: str) -> Model[str, List]:
    #XXX A limitation of coref-hoi is that it calls forward once per doc
    config = initialize_config(name)
    seg_len = config["max_segment_len"]
    # this includes a bert tokenizer
    data_processor = CorefDataProcessor(config)

    def forward(model, inputs: Doc, is_train: bool):
        data_processor = model.attrs["data_processor"]
        bert_tokenizer = model.attrs["bert_tokenizer"]

        conll_lines = make_conll_lines(inputs)

        # training data has genres so we have to provide something
        # genres aren't relevant to input data though so just use default
        genre = "nw" # default genre
        doc = get_document("nw", conll_lines, 'english', seg_len, bert_tokenizer)

        tensor_examples, _ = data_processor.get_tensor_examples_from_custom_input([doc])
       
        out = tensor_examples[0][1:8]
        return (out[0], doc), lambda x: []

    return Model("coref_preprocessor", forward, attrs={
        "data_processor": data_processor,
        "bert_tokenizer": data_processor.tokenizer,
        })


def convert_coref_inputs(model, inputs, is_train):
    kwargs = {}
    # doc is not used here, but later for token alignment
    out, doc = inputs
    return ArgsKwargs(args=out, kwargs=kwargs), lambda dX: []

def convert_coref_outputs(
        model: Model,
        # TODO make this spans
        inputs_outputs,
        is_train: bool
        ) -> Tuple[List[List[Tuple[int,int]]], Callable]:
    inputs, outputs = inputs_outputs
    _, doc = inputs
    # XXX not sure why this is necessary, where is the extra thing coming from?
    outputs = outputs[0]
    _, _, _, span_starts, span_ends, ant_idx, ant_scores = outputs

    # put everything on cpu
    span_starts = span_starts.tolist()
    span_ends = span_ends.tolist()
    ant_idx = ant_idx.tolist()
    ant_scores = ant_scores.tolist()

    clusters = get_predicted_clusters(span_starts, span_ends, ant_idx, ant_scores)
    #TODO actually implement backprop

    # clusters here are actually wordpiece token indexes, we should convert
    # those to spaCy token indexes / spans
    out = []
    stm = doc['subtoken_map']
    tokens = doc['tokens']
    for cluster in clusters:
        cmap = []
        for start, finish in cluster:
            # these are actual token indices
            # XXX check how the actual token indices are generated
            cmap.append( (stm[start], stm[finish]+1) )
        out.append(cmap)

    return out, lambda x: []

def initialize_model(config_name, device, starter=None):
    config = initialize_config(config_name)
    coref = CorefModel(config, device)
    if starter:
        sdict = torch.load(starter)
        coref.load_state_dict(sdict, strict=False)
    return coref

@thinc.registry.layers("coref_model.v1")
def CorefCore(name, saved_model=None) -> Model[List, List[List[Tuple]]]:
    return PyTorchWrapper(
            initialize_model(name, "cuda:0", saved_model),
            convert_inputs=convert_coref_inputs,
            convert_outputs=convert_coref_outputs)

@registry.architectures("spacy.Coref.v0")
def WrappedCoref(config_name, model_path=None) -> Model:
    return chain(
            CorefPreprocessor(config_name),
            CorefCore(config_name, model_path)
            )

if __name__ == "__main__":
    from icecream import ic
    config_name = "bert_small"
    model_path = "./data/bert_small/model_Mar21_19-13-39_65000.bin"

    text = "John is in London, he says it's raining in the city."

    model = chain(
            CorefPreprocessor(config_name),
            CorefCore(config_name, model_path)
            )
    out, backprop = model(text, is_train = False)
    ic(out)
