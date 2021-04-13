from dataclasses import dataclass

from thinc.api import Model, Linear, Relu, Dropout, chain, concatenate
from thinc.api import list2ragged, reduce_mean, ragged2list
from thinc.types import Pairs, Floats2d, Floats1d, DTypesFloat
import spacy
from spacy.tokens import Doc, Span
from typing import cast

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
        mention_limit: int,  # maybe this and the next one should be a union?
        mention_ratio: float, 
        antecedent_limit: int, 
        mention_scorer: Model[Floats2d, Floats1d],
        dim: int,
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
    # each row of input is a span


    # calculate span scores
    xp = model.ops.xp
    # this should have backprop
    mention_scorer = model.get_ref("mention_scorer")

    mention_scores, _ = mention_scorer(inputs.vecs, is_train)
    #ic(inputs.vecs.shape, inputs.vecs)
    #ic(mention_scores.shape, mention_scores)
    # pick top spans
    top_mentions = xp.argsort(-1 * mention_scores).flatten()
    ic(top_mentions)
    # num_top_spans in old code
    top_span_limit = model.attrs["mention_limit"]
    # XXX need to have token indexes here still...
    starts = inputs.idxs.one
    ends = inputs.idxs.two
    selected = select_non_crossing_spans(top_mentions, starts, ends, top_span_limit)
    top_scores = mention_scores[selected]
    top_vectors = inputs.vecs[selected]

    # TODO this is the smaller of a hyperparameter or (hyperparameter * word in doc)
    #
    #ic(inputs.vecs)
    # max_top_antecedents in old code
    ant_limit = min(inputs.vecs.shape[0], model.attrs["antecedent_limit"])
    ant_limit = min(ant_limit, top_span_limit) # can't have a higher ant limit than spans we check

    # create a mask so that antecedents must come before referrents
    # XXX following manipulations should probably use ops
    top_span_range = xp.arange(top_span_limit) 
    offsets = xp.expand_dims(top_span_range, 1) - xp.expand_dims(top_span_range, 0)
    mask = (offsets >= 1) 
    mask = xp.float64(mask)
    #ic(mask)

    pairwise_score_sum = xp.expand_dims(top_scores, 1) + xp.expand_dims(top_scores, 0)
    dropout = model.get_ref("dropout")
    coarse_bilinear = model.get_ref("coarse_bilinear")
    #XXX make this a chain
    source_span_emb, source_backprop = coarse_bilinear(top_vectors, is_train)
    target_span_emb, target_backprop = dropout(xp.transpose(top_vectors), False)
    pairwise_coref_scores = xp.matmul(source_span_emb, target_span_emb)
    pairwise_fast_scores = pairwise_score_sum + pairwise_coref_scores
    # this gives a runtime warning but it's not a problem; silence it
    pairwise_fast_scores += xp.log(mask)
    # these scores can be used for final output

    #TODO figure out topk
    top_ant_idx = xp.argsort(xp.argpartition(pairwise_fast_scores, ant_limit))
    top_ant_scores = pairwise_fast_scores[top_ant_idx]
    top_ant_mask = batch_select(xp, mask, top_ant_idx)
    top_ant_offset = batch_select(xp, offsets, top_ant_idx)


    return top_ant_idx, lambda x: []

def batch_select(xp, tensor, idx):
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


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    text = "John called from London, he says it's raining in the city. He's all wet."
    doc = nlp(text)
    tok2vec = nlp.pipeline[0][1].model
    dim = 96 # TODO get this from the model or something
    
    from thinc.util import get_array_module
    xp = get_array_module(doc.tensor)
    span2vec = chain(list2ragged(), reduce_mean())
    model1 = build_span_embedder(get_candidate_mentions, tok2vec, span2vec)
    hidden = 1000 # from coref-hoi config
    # XXX I shouldn't have to supply the dimensions here, right?
    #mention_scorer = lambda x, y: (xp.average(x, 1), [])
    mention_scorer = chain(Linear(nI=dim, nO=hidden), Relu(nI=hidden, nO=hidden), Dropout(0.3), Linear(nI=hidden, nO=1))
    mention_scorer.initialize()
    model2 = build_antecedent_selector(20, 0.4, 10, mention_scorer, dim * 3, 0.3)

    coref = chain(model1, model2)

    out, backprop = coref(doc, False)
    print(out)
    #print(get_candidate_mentions(doc))


