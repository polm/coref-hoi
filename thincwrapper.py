from thinc.api import PyTorchWrapper, require_gpu, fix_random_seed
fix_random_seed(23)
require_gpu()

import numpy as np
import spacy
import torch

from model import CorefModel
from util import initialize_config, flatten
from tensorize import CorefDataProcessor
from predict import get_document_from_string

from icecream import ic

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

if __name__ == "__main__":
    text = "John Smith reported the news from London today, he said it was raining in the city."

    seg_len = 128 # length of segments for bert
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    config_name = "bert_small"
    # load config as a dict
    config = initialize_config(config_name) 
    data_processor = CorefDataProcessor(config)
    bert_tokenizer = data_processor.tokenizer

    device = "cuda:0"
    model_path = "./data/bert_small/model_Mar21_19-13-39_65000.bin"
    # XXX not sure why this is cpu
    sdict = torch.load(model_path, map_location=torch.device("cpu"))

    coref = CorefModel(config, device)
    coref.load_state_dict(sdict, strict=False)
    model = PyTorchWrapper(coref)

    # this uses spaCy and BERT tokenizers to create artificial conll training data
    doc = get_document_from_string(text, seg_len, bert_tokenizer, nlp)
    # this creates the basic tensor repr of the input, including all metadata
    # for raw text input metadata is mostly empty
    tensor_examples, _ = data_processor.get_tensor_examples_from_custom_input([doc])


    # why is the genre doc_key here? It's not used anywhere
    for (doc_key, ex) in tensor_examples:
        # first do the actual prediction
        ex = ex[:7]
        predictions, backprop = model(ex, is_train=False)
        _, _, _, span_starts, span_ends, ant_idx, ant_scores = predictions

        # make everything into lists
        span_starts = span_starts.tolist()
        span_ends = span_ends.tolist()
        ant_idx = ant_idx.tolist()
        ant_scores = ant_scores.tolist()
        ic(ant_idx)
        ic(ant_scores)

        # convert the predictions into usable data
        ic(span_starts, span_ends)
        clusters, mention2cluster, ants = get_predicted_clusters(span_starts, span_ends, ant_idx, ant_scores)
        ic(clusters)
        ic(mention2cluster)
        ic(ants)

        # render the predictions, each cluster a list.
        subtokens = flatten(doc["sentences"])
        for cluster in clusters:
            mentions_str = [' '.join(subtokens[m[0]:m[1]+1]) for m in cluster]
            mentions_str = [m.replace(' ##', '') for m in mentions_str]
            mentions_str = [m.replace('##', '') for m in mentions_str]
            print(mentions_str)  # Print out strings

