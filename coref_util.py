# This is a new file for the spaCy model
# it holds functions not directly part of the model
from thinc.types import Pairs
from spacy.tokens import Doc

# from model.py, refactored to be non-member
def get_predicted_antecedents(xp, antecedent_idx, antecedent_scores):
    """Get the ID of the antecedent for each span. -1 if no antecedent."""
    #ic(antecedent_scores)
    predicted_antecedents = []
    for i, idx in enumerate(xp.argmax(antecedent_scores, axis=1) - 1):
        if idx < 0:
            predicted_antecedents.append(-1)
        else:
            predicted_antecedents.append(antecedent_idx[i][idx])
    return predicted_antecedents

# from model.py, refactored to be non-member
def get_predicted_clusters(xp, span_starts, span_ends, antecedent_idx, antecedent_scores):
    """Convert predictions to usable cluster data.

    return values:

    clusters: a list of spans (i, j) that are a cluster

    Note that not all spans will be in the final output; spans with no
    antecedent or referrent are omitted from clusters and mention2cluster.
    """
    # Get predicted antecedents
    predicted_antecedents = get_predicted_antecedents(xp, antecedent_idx, antecedent_scores)

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

def get_sentence_map(doc: Doc):
    """For the given span, return a list of sentence indexes."""

    si = 0
    out = []
    for sent in doc.sents:
        for tok in sent:
            out.append(si)
        si += 1
    return out

def get_candidate_mentions(doc: Doc, max_span_width: int = 20) -> Pairs[int]:
    """Given a Doc, return candidate mentions.

    This isn't a trainable layer, it just returns raw candidates.
    """
    # XXX Note that in coref-hoi the indexes are designed so you actually want [i:j+1], but here
    # we're using [i:j], which is more natural.

    sentence_map = get_sentence_map(doc)

    begins = []
    ends = []
    for tok in doc:
        si = sentence_map[tok.i] # sentence index
        for ii in range(1, max_span_width):
            ei = tok.i + ii # end index
            if ei < len(doc) and sentence_map[ei] == si:
                begins.append(tok.i)
                ends.append(ei)

    return Pairs(begins, ends)


