# This is a new file for the spaCy model
# it holds functions not directly part of the model
from thinc.types import Pairs, Ints2d
from spacy.tokens import Doc
from typing import List, Tuple

from icecream import ic

# type alias to make writing this less tedious
MentionClusters = List[List[Tuple[int, int]]]

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

def select_non_crossing_spans(idxs, starts, ends, limit) -> List[int]:
    """Given a list of spans sorted in descending order, return the indexes of
    spans to keep, discarding spans that cross.

    Nested spans are allowed.
    """
    # ported from Model._extract_top_spans
    selected = []
    start_to_max_end = {}
    end_to_min_start = {}

    for idx in idxs:
        if len(selected) >= limit or idx > len(starts):
            break

        start, end = starts[idx], ends[idx]
        cross = False

        for ti in range(start, end + 1):
            max_end = start_to_max_end.get(ti, -1)
            if ti > start and max_end > end:
                cross = True
                break

            min_start = end_to_min_start.get(ti, -1)
            if ti < end and 0 <= min_start < start:
                cross = True
                break

        if not cross:
            # this index will be kept
            # record it so we can exclude anything that crosses it
            selected.append(idx)
            max_end = start_to_max_end.get(start, -1)
            if end > max_end:
                start_to_max_end[start] = end
            min_start = end_to_min_start.get(end, -1)
            if start == -1 or start < min_start:
                end_to_min_start[end] = start

    # sort idxs by order in doc
    selected = sorted(selected, key=lambda idx: (starts[idx], ends[idx]))
    while len(selected) < limit:
        selected.append(selected[0]) # this seems a bit weird?
    return selected

def get_clusters_from_doc(doc) -> List[List[Tuple[int, int]]]:
    """Given a Doc, convert the cluster spans to simple int tuple lists.
    """
    out = []
    for key, val in doc.spans.items():
        cluster = []
        for span in val:
            # TODO check that there isn't an off-by-one error here
            cluster.append( (span.start, span.end) )
        out.append(cluster)
    return out

def make_clean_doc(nlp, doc):
    """Return a doc with raw data but not span annotations."""
    # Surely there is a better way to do this?

    sents = [tok.is_sent_start for tok in doc]
    words = [tok.text for tok in doc]
    out = Doc(nlp.vocab, words=words, sent_starts=sents)
    return out

def create_gold_scores(ments: Ints2d, clusters: List[List[Tuple[int, int]]]) -> List[List[bool]]:
    """Given mentions considered for antecedents and gold clusters,
    construct a gold score matrix."""
    # make a mapping of mentions to cluster id
    # id is not important but equality will be
    ment2cid = {}
    for cid, cluster in enumerate(clusters):
        for ment in cluster:
            ment2cid[ment] = cid

    out = []
    mentuples = [tuple(mm) for mm in ments]
    for ii, ment in enumerate(mentuples):
        if ment not in ment2cid:
            # this is not in a cluster so it's a dummy
            out.append([True] + ([False] * len(ments)))
            continue

        # this might change if no real antecedent is a candidate
        row = [False] 
        cid = ment2cid[ment]
        for jj, ante in enumerate(mentuples):
            # antecedents must come first
            if jj >= ii: 
                row.append(False)
                continue

            row.append(cid == ment2cid.get(ante, -1))

        if not any(row):
            row[0] = True # dummy
        out.append(row)

    # caller needs to convert to array
    return out



