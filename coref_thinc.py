from dataclasses import dataclass

from thinc.api import Model, Linear, Relu, Dropout, chain, concatenate
from thinc.api import list2ragged, reduce_mean, ragged2list
from thinc.types import Pairs, Floats2d, Floats1d, DTypesFloat
import spacy
from spacy.tokens import Doc, Span
from typing import cast, List

from collections import namedtuple

from icecream import ic

@dataclass
class Mentions:
    """A collection of mentions. Each mention is a span of text with token
    indices and a vector representation.
    """
    vecs: Floats2d
    idxs: Pairs[int]

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

# model converting a Doc/Mention to span embeddings
def build_span_embedder(mention_generator, tok2vec, span2vec) -> Model[Doc, Mentions]:
    
    return Model("SpanEmbedding", 
            forward=span_embeddings_forward, 
            attrs={
                "mention_generator": mention_generator,
                },
            refs={
                "tok2vec": tok2vec,
                "span2vec": span2vec,
                })

def span_embeddings_forward(model, inputs: Doc, is_train):
    mgen = model.attrs["mention_generator"]
    tok2vec = model.get_ref("tok2vec")
    span2vec = model.get_ref("span2vec")
    xp = model.ops.xp

    mentions = mgen(inputs)
    tokvecs, _ = tok2vec([inputs], False)
    tokvecs = tokvecs[0]

    spans = [tokvecs[ii:jj] for ii, jj in zip(mentions.one, mentions.two)]
    spans, _ = span2vec(spans, False)
    
    # first and last token embeds
    starts = [tokvecs[ii] for ii in mentions.one]
    ends   = [tokvecs[jj] for jj in mentions.two]

    starts = xp.asarray(starts)
    ends   = xp.asarray(ends)
    out = xp.concatenate( (starts, ends, spans), 1 )
    # TODO backprop
    return Mentions(out, mentions), lambda x: []


#XXX would it make sense to have the other model just be inside this one?
def build_antecedent_selector(
        mention_scorer: Model[Floats2d, Floats1d],
        dim: int,
        antecedent_limit: int, 
        mention_limit: int,  # maybe this and the next one should be a union?
        mention_ratio: float = 1.0, 
        dropout: float = 0.3,
        ) -> Model[Mentions, Pairs]:
    # TODO take size as a param or something
    linear = Linear(dim, dim)
    linear.initialize()
    dropout = Dropout(dropout)
    cb = chain(linear, dropout)
    return Model("AntecedentSelector", 
            forward=antecedent_forward,
            attrs={
                "mention_limit": mention_limit,
                "mention_ratio": mention_ratio,
                "antecedent_limit": antecedent_limit,
                },
            refs={
                "mention_scorer": mention_scorer,
                "dropout": dropout,
                "coarse_bilinear": cb,
                }
            )

def antecedent_forward(model, inputs: Mentions, is_train):
    xp = model.ops.xp
    mention_scorer = model.get_ref("mention_scorer")
    mention_scores, _ = mention_scorer(inputs.vecs, is_train)

    # pick top spans
    top_mentions = xp.argsort(-1 * mention_scores).flatten()

    # num_top_spans in old code
    top_span_limit = model.attrs["mention_limit"]

    #XXX this is kind of a weird step. They drop from GPU to CPU in the
    # coref-hoi code, and I'm not sure how this would look correctly in
    # thinc.

    # Get only valid spans; (spans [like) this] are not allowed
    selected = select_non_crossing_spans(top_mentions, 
            inputs.idxs.one,
            inputs.idxs.two,
            top_span_limit)
    top_scores = mention_scores[selected]
    top_vectors = inputs.vecs[selected]

    # max_top_antecedents in old code
    ant_limit = min(
            inputs.vecs.shape[0] * model.attrs["mention_ratio"],
            model.attrs["antecedent_limit"],
            # can't have a higher ant limit than spans we check
            top_span_limit)

    # create a mask so that antecedents must come before referrents
    top_span_range = xp.arange(top_span_limit) 
    offsets = xp.expand_dims(top_span_range, 1) - xp.expand_dims(top_span_range, 0)
    mask = xp.float64(offsets >= 1) 

    pairwise_score_sum = xp.expand_dims(top_scores, 1) + xp.expand_dims(top_scores, 0)
    dropout = model.get_ref("dropout")
    coarse_bilinear = model.get_ref("coarse_bilinear")
    #XXX make this a chain
    source_span_emb, source_backprop = coarse_bilinear(top_vectors, is_train)
    target_span_emb, target_backprop = dropout(xp.transpose(top_vectors), is_train)
    pairwise_coref_scores = xp.matmul(source_span_emb, target_span_emb)
    pairwise_fast_scores = pairwise_score_sum + pairwise_coref_scores
    # TODO this gives a runtime warning but it's not a problem; silence it
    pairwise_fast_scores += xp.log(mask)
    # these scores can be used for final output

    # This used topk in coref-hoi
    top_ant_idx = xp.argsort(xp.argpartition(pairwise_fast_scores, ant_limit))
    top_ant_scores = pairwise_fast_scores[top_ant_idx]
    top_ant_mask = batch_select(xp, mask, top_ant_idx)
    top_ant_offset = batch_select(xp, offsets, top_ant_idx)

    return top_ant_idx, lambda x: []

def batch_select(xp, tensor, idx):
    # this is basically used to apply 2d indices to Floats2d
    # original comment:
    # do selection per row (first axis)
    n0, n1 = tensor.shape[0], tensor.shape[1]

    offset = xp.expand_dims(xp.arange(0, n0) * n1, 1)
    nidx = idx + offset
    #ic(nidx, idx, offset)
    tt = xp.reshape(tensor, [n0 * n1, -1])
    selected = tt[nidx]

    if tt.shape[-1] == 1:
        selected = xp.squeeze(selected, -1)

    return selected


def select_non_crossing_spans(idxs, starts, ends, limit):
    """Given a list of spans sorted in descending order, select the top spans,
    discarding spans that cross.

    Nested spans are allowed.
    """
    # ported from Model._extract_top_spans
    selected = []
    start_to_max_end = {}
    end_to_min_start = {}

    ic(idxs)
    for idx in idxs:
        if len(selected) >= limit or idx > len(starts):
            break

        ic(len(starts), len(ends), len(idxs), idx)

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

def logsumexp(xp, arg):
    return xp.log(xp.sum( xp.exp(arg), axis=1))

def get_gold_matrix(xp, top_mentions: Pairs, top_antecedents: Pairs, gold_clusters: List[List[Span]]):
    # this creates an equivalent to the top_antecedent_gold_labels

    gold_starts = []
    gold_ends = []
    gold_ids = []
    # XXX technically the original data has ids, but we only care about equality
    # first just get the data in lists
    for ii, cluster in enumerate(gold_clusters, start=1):
        for span in cluster:
            gold_starts.append(span.start)
            gold_ends.append(span.end)
            gold_ids.append(ii)

    # make it arrays
    gold_starts = xp.asarray(gold_starts)
    gold_ends = xp.asarray(gold_ends)
    gold_ids = xp.asarray(gold_ids)

    # the pairs are start:end indices
    same_start = xp.expand_dims(gold_starts, 1) == xp.expand_dims(top_mentions.one, 0)
    same_end = xp.expand_dims(gold_ends, 1) == xp.expand_dims(top_mentions.two, 0)
    same_span = (same_start & same_end).astype(float)
    gids = xp.expand_dims(gold_ids, 0).astype(float)
    labels = xp.matmul( gids, same_span ).astype(int)
    # this is equivalent to candidate_labels in coref-hoi
    # 1d array of integers where:
    # idx: index in mention list
    # value: cluster id (start at 1, 0 means no cluster)
    labels = xp.squeeze(labels, 0)
    return labels

def get_antecedent_gold(xp, mention_gold, selected_mentions, top_antes, ante_mask):
    # mention_gold: 1d, idx = mention id, val = cluster id (0 for none)
    # selected_mentions: non-pruned idxs after crossing eliminated
    # top_ants: idx of top antecedents from topk
    top_ment_clusters = mention_gold[selected_mentions]

    top_ante_clusters = top_ment_clusters[top_antes]
    # this is magic designed to mask invalid ids
    top_ante_clusters += (ante_mask.to(int) - 1) * 100000
    same_gold = (top_ante_clusters == xp.expand_dims(top_ment_clusters, 1))
    non_dummy = xp.expand_dims(top_ment_clusters > 0, 1)
    pairwise_labels = same_gold & non_dummy

    dummy_ante = xp.logical_not(pairwise_labels.any(axis=1, keepdims=True))
    ante_gold = xp.concatenate([dummy_ante, pairwise_labels], axis=1)
    return ante_gold

def gold_data_test(xp):
    mentions = Pairs([1, 2, 3], [2, 4, 7])
    Span = namedtuple("DummySpan", ('start', 'end'))
    gold_clusters = [
            ( Span(1, 2), Span(2, 4) ),
            ( Span(5, 7), Span(8, 9) ),
            ]

    res = get_gold_matrix(xp, mentions, None, gold_clusters)
    ic(res)


def loss_demo(xp):
    preds = xp.asarray([
        [1.0, 0, 0],
        [0, 100.0, 1.0],
        [0, 1.0, 0],
        [1.0, 0, 0],
        ])

    # given:
    #   top mentions
    #   top antecedents
    #   gold mention list
    # generate:
    #   binary truth matrix
    # the original coref-hoi uses numpy magic for this





    truth = xp.asarray([
        [True, False, False],
        [False, True, True],
        [False, True, False],
        [True, False, False],
        ])
   
    ic(preds)
    ic(truth)
    logmarg = logsumexp(xp, ( preds + xp.log(truth.astype(float))))
    ic(logmarg)
    lognorm = logsumexp(xp, preds )
    loss = xp.sum(lognorm - logmarg)
    return loss


def test_run():
    nlp = spacy.load("en_core_web_sm")
    text = "John called from London, he says it's raining in the city. He's all wet."
    doc = nlp(text)
    tok2vec = nlp.pipeline[0][1].model
    dim = 96 # TODO get this from the model or something
    
    from thinc.util import get_array_module
    xp = get_array_module(doc.tensor)
    span2vec = chain(list2ragged(), reduce_mean())
    spanembed = build_span_embedder(get_candidate_mentions, tok2vec, span2vec)
    hidden = 1000 # from coref-hoi config
    # XXX I shouldn't have to supply the dimensions here, right?
    #mention_scorer = lambda x, y: (xp.average(x, 1), [])
    mention_scorer = chain(Linear(nI=dim, nO=hidden), Relu(nI=hidden, nO=hidden), Dropout(0.3), Linear(nI=hidden, nO=1))
    mention_scorer.initialize()
    #TODO use config
    antsel = build_antecedent_selector(
            mention_limit=20, 
            antecedent_limit=10, 
            mention_scorer=mention_scorer, 
            dim=(dim * 3), 
            dropout=0.3)

    coref = chain(spanembed, antsel)

    out, backprop = coref(doc, False)
    print(out)
    #print(get_candidate_mentions(doc))


if __name__ == "__main__":
    from thinc.api import get_current_ops
    ops = get_current_ops()
    gold_data_test(ops.xp)
