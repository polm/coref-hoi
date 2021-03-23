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

from icecream import ic

from model import CorefModel
from util import initialize_config, flatten
from tensorize import CorefDataProcessor
from predict import get_document_from_string

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
    mention2cluster: a mapping of spans (i, j) to cluster ids
    ants: a list of antecedent ids for each span.

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
    return predicted_clusters, mention_to_cluster_id, predicted_antecedents


@thinc.registry.layers("coref_preprocessor.v1")
def CorefPreprocessor(name: str) -> Model[str, List]:
    #XXX A limitation of coref-hoi is that it calls forward once per doc
    config = initialize_config(name)
    seg_len = config["max_segment_len"]
    data_processor = CorefDataProcessor(config)

    sentencizer = spacy.blank("en")
    sentencizer.add_pipe("sentencizer")

    def forward(model, inputs: str, is_train: bool):
        data_processor = model.attrs["data_processor"]
        bert_tokenizer = model.attrs["bert_tokenizer"]
        sentencizer = model.attrs["sentencizer"]

        doc = get_document_from_string(
                        inputs,
                        seg_len,
                        bert_tokenizer,
                        sentencizer)
        tensor_examples, _ = data_processor.get_tensor_examples_from_custom_input([doc])
       
        out = []
        for (doc_key, ex) in tensor_examples:
            out.append(ex[:7])
        return out, lambda x: []

    return Model("coref_preprocessor", forward, attrs={
        "data_processor": data_processor,
        "bert_tokenizer": data_processor.tokenizer,
        "sentencizer": sentencizer})


def convert_coref_inputs(model, inputs, is_train):
    kwargs = {}
    # XXX hack because this can't actually be batched
    inputs = inputs[0]
    return ArgsKwargs(args=inputs, kwargs=kwargs), lambda dX: []

def convert_coref_outputs(
        model: Model,
        # TODO make this spans
        inputs_outputs,
        is_train: bool
        ) -> Tuple[List[List[Tuple[int,int]]], Callable]:
    inputs, outputs = inputs_outputs
    _, _, _, span_starts, span_ends, ant_idx, ant_scores = outputs

    # put everything on cpu
    span_starts = span_starts.tolist()
    span_ends = span_ends.tolist()
    ant_idx = ant_idx.tolist()
    ant_scores = ant_scores.tolist()

    clusters, mention2cluster, ants = get_predicted_clusters(span_starts, span_ends, ant_idx, ant_scores)
    #TODO actually implement backprop

    # clusters here are actually wordpiece token indexes, we should convert
    # those to spaCy token indexes

    return clusters, lambda x: []

def initialize_model(config_name, device, starter=None):
    config = initialize_config(config_name)
    coref = CorefModel(config, device)
    if starter:
        #XXX cpu may not be necessary here?
        sdict = torch.load(starter, map_location=torch.device("cpu"))
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
    #segment_length = 128
    config_name = "bert_small"
    model_path = "./data/bert_small/model_Mar21_19-13-39_65000.bin"

    text = "John is in London, he says it's raining in the city."

    model = chain(
            CorefPreprocessor(config_name),
            CorefCore(config_name, model_path)
            )
    out, backprop = model(text, is_train = False)
    ic(out)
