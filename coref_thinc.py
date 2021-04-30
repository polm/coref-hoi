from dataclasses import dataclass

from thinc.api import Model, Linear, Relu, Dropout, chain, concatenate
from thinc.api import list2ragged, reduce_mean, ragged2list, noop
from thinc.types import Pairs, Floats2d, Floats1d, DTypesFloat, Ints2d, Ragged, Ints1d
import spacy
from spacy.tokens import Doc, Span
from typing import cast, List, Callable, Any, Tuple

from collections import namedtuple
from coref_model_wrapped import get_predicted_clusters

from icecream import ic

def tuplify(layer1: Model, layer2: Model, *layers) -> Model:
    layers = (layer1, layer2) + layers
    names = [layer.name for layer in layers]
    return Model(
            "tuple(" + ", ".join(names) + ")", 
            tuplify_forward, 
            layers=layers,
    )


def tuplify_forward(model, X, is_train):
    Ys = []
    backprops = []
    for layer in model.layers:
        Y, backprop = layer(X, is_train)
        Ys.append(Y)
        backprops.append(backprop)

    def backprop_tuplify(dYs):
        dXs = [bp(dY) for bp, dY in zip(backprops, dYs)]
        dX = dXs[0]
        for dx in dXs[1:]:
            dX += dx
        return dX

    return tuple(Ys), backprop_tuplify


def tuplify_span_embeds(layer1: Model, layer2: Model, *layers) -> Model:
    layers = (layer1, layer2) + layers
    names = [layer.name for layer in layers]
    return Model(
            "tuple(" + ", ".join(names) + ")", 
            tuplify_span_embeds_forward, 
            layers=layers,
    )


def tuplify_span_embeds_forward(model, X, is_train):
    Ys = []
    backprops = []
    for layer in model.layers:
        Y, backprop = layer(X, is_train)
        Ys.append(Y)
        backprops.append(backprop)

    def backprop_tuplify(dYs):
        dXs = [bp(dY) for bp, dY in zip(backprops, dYs)]
        dX = dXs[0]
        # indices / lengths are the same
        for dx in dXs[1:]:
            dX.vectors.data += dx.vectors.data
        return dX

    return tuple(Ys), backprop_tuplify


@dataclass
class SpanEmbeddings:
    indices: Ints2d # Array with 2 columns (for start and end index)
    vectors: Ragged # Ragged[Floats2d] # One vector per span
    # NB: We assume that the indices refer to a concatenated Floats2d that
    # has one row per token in the *batch* of documents. This makes it unambiguous
    # which row is in which document, because if the lengths are e.g. [10, 5],
    # a span starting at 11 must be starting at token 2 of doc 1. A bug could
    # potentially cause you to have a span which crosses a doc boundary though,
    # which would be bad.
    # The lengths in the Ragged are not the tokens per doc, but the number of 
    # mentions per doc.


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
# mention_generator: Callable[Doc, Pairs[int]]
def build_span_embedder(mention_generator) -> Model[Tuple[List[Floats2d], List[Doc]], SpanEmbeddings]:
    
    return Model("SpanEmbedding", 
            forward=span_embeddings_forward, 
            attrs={ "mention_generator": mention_generator, },
    )

def span_embeddings_forward(model, inputs: Tuple[List[Floats2d], List[Doc]], is_train) -> SpanEmbeddings:
    ops = model.ops
    xp = ops.xp

    tokvecs, docs = inputs

    mgen = model.attrs["mention_generator"]
    mentions = ops.alloc2i(0, 2)
    total_length = 0
    docmenlens = [] # number of mentions per doc
    for doc in docs:
        mm = mgen(doc)
        docmenlens.append(len(mm))
        cments = ops.asarray2i([mm.one, mm.two]).transpose()

        mentions = xp.concatenate( (mentions, cments + total_length) )
        total_length += len(doc)

    # TODO support attention here
    tokvecs = xp.concatenate(tokvecs)
    spans = [tokvecs[ii:jj] for ii, jj in mentions.tolist()]
    idx = 10
    avgs = [xp.mean(ss, axis=0) for ss in spans]
    spanvecs = ops.asarray2f(avgs)
    
    # first and last token embeds
    starts = [tokvecs[ii] for ii in mentions[:,0]]
    ends   = [tokvecs[jj] for jj in mentions[:,1]]


    starts = ops.asarray2f(starts)
    ends   = ops.asarray2f(ends)
    concat = xp.concatenate( (starts, ends, spanvecs), 1 )
    embeds = Ragged(concat, docmenlens)

    def backprop_span_embed(dY: SpanEmbeddings) -> Tuple[List[Floats2d], List[Doc]]:
        # how does this work?
        tokvecs[0].shape[0] # TODO get this properly
        dX = [ops.alloc2f(len(doc), dim) for doc in docs]

        docidx = 0
        offset = len(docs[docidx])
        for mi in range(0, len(dY.indices)):
            start, end = dY.indices[mi, :]
            if end > offset + len(docs[docidx]):
                docidx += 1
                offset += len(docs[docidx])

            # get the tokvec
            embed = dY.vectors.data[mi, :]
            #XXX probably better to divide by number of tokens
            dX[start:end] += embed
        return dX

    return SpanEmbeddings(mentions, embeds), backprop_span_embed


# SpanEmbeddings -> SpanEmbeddings
def build_coarse_pruner(
        mention_limit: int,
        ) -> Model[SpanEmbeddings, SpanEmbeddings]:
    model = Model("CoarsePruner",
            forward=coarse_prune,
            attrs={
                "mention_limit": mention_limit,
                },
            )
    return model

def coarse_prune(model, inputs: Tuple[Floats1d, SpanEmbeddings], is_train) -> SpanEmbeddings:
    # input spanembeddings are *all* candidate mentions
    # output spanembeddings are *pruned* mentions
    # do scoring
    rawscores, spanembeds = inputs
    scores = rawscores.squeeze()
    # do pruning
    mention_limit = model.attrs["mention_limit"]
    #XXX: Issue here. Don't need docs to find crossing spans, but might for the limits.
    # In old code the limit can be:
    # - hard number per doc
    # - ratio of tokens in the doc

    offset = 0
    selected = []
    sellens = []
    for menlen in spanembeds.vectors.lengths:
        hi = offset + menlen
        cscores = scores[offset:hi]

        # negate it so highest numbers come first
        tops = (model.ops.xp.argsort(-1 * cscores)).tolist()
        # TODO this is wrong
        starts = spanembeds.indices[offset:hi, 0].tolist()
        ends = spanembeds.indices[offset:hi:, 1].tolist()

        # selected is a 1d integer list
        csel = select_non_crossing_spans(
                tops, starts, ends, mention_limit)
        # add the offset so these indices are absolute
        csel = [ii + offset for ii in csel]
        # this should be constant because short choices are padded
        sellens.append(len(csel))
        selected += csel
        offset += menlen
  
    selected = model.ops.asarray1i(selected)
    top_spans = spanembeds.indices[selected]
    top_vecs = spanembeds.vectors.data[selected]

    out = SpanEmbeddings(top_spans, Ragged(top_vecs, sellens))

    def coarse_prune_backprop(dY: Tuple[Floats1d, SpanEmbeddings]) -> Tuple[Floats1d, SpanEmbeddings]:
        ll = spanembeds.indices.shape[0]

        dYscores, dYembeds = dY

        dXscores = model.ops.alloc1f(ll) 
        #ic(dXscores.shape, selected.shape, dYscores.shape)
        # I think we only backprop the selected ones here?
        # TODO check this
        #dXscores[selected] = dYscores[selected]
        # actually, maybe this is pass through?
        dXscores[selected] = dYscores.squeeze()

        dXvecs = model.ops.alloc2f(*spanembeds.vectors.data.shape)
        ic(dXscores.shape, dYscores.shape, dXvecs.shape, dYembeds.vectors.data.shape)
        ic(spanembeds.indices.shape)
        dXvecs[selected] = dYembeds.vectors.data
        dXembeds = SpanEmbeddings(spanembeds.indices, dXvecs)

        # inflate for mention scorer
        dXscores = model.ops.xp.expand_dims(dXscores, 1)

        return (dXscores, dXembeds)

    return (scores[selected], out), coarse_prune_backprop

def batch_select(xp, tensor, idx):
    # this is basically used to apply 2d indices to Floats2d
    # original comment:
    # do selection per row (first axis)
    n0, n1 = tensor.shape[0], tensor.shape[1]

    offset = xp.expand_dims(xp.arange(0, n0) * n1, 1)
    nidx = idx + offset
    tt = xp.reshape(tensor, [n0 * n1, -1])
    selected = tt[nidx]

    if tt.shape[-1] == 1:
        selected = xp.squeeze(selected, -1)

    return selected


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

def get_gold_clusters_array(xp, docs: List[Doc]) -> List[Ints2d]:
    # this assumes the only spans on the doc are coref clusters
    # XXX this is copied from coref-hoi but probably not necessary
    out = []
    for doc in docs:
        starts = []
        ends = []
        ids = []
        # XXX technically the original data has ids, but we only care about equality
        # first just get the data in lists
        for ii, cluster in enumerate(doc.spans.values(), start=1):
            for span in cluster:
                starts.append(span.start)
                ends.append(span.end)
                ids.append(ii)

        # make it arrays
        starts = xp.asarray(starts)
        ends = xp.asarray(ends)
        ids = xp.asarray(ids)

        out.append( xp.column_stack( (starts, ends, ids) ) )
    return out

def get_gold_mention_labels(xp, mentions: List[Ints2d], gold_clusters: List[Ints2d]):
    """Given mentions and gold clusters, find the gold cluster ID for each mention.
    If the mention is not a gold mention the label will be 0.

    The inputs are each a list with one entry per doc."""
    # this creates an equivalent to the top_antecedent_gold_labels

    out = []

    for ments, gold in zip(mentions, gold_clusters):
        gold_starts = gold[:, 0]
        gold_ends = gold[:, 1]
        gold_ids = gold[:, 2]
        ment_starts = ments[:, 0]
        ment_ends = ments[:, 1]


        same_start = xp.expand_dims(gold_starts, 1) == xp.expand_dims(ment_starts, 0)
        same_end = xp.expand_dims(gold_ends, 1) == xp.expand_dims(ment_ends, 0)
        same_span = (same_start & same_end).astype(float)
        gids = xp.expand_dims(gold_ids, 0).astype(float)
        labels = xp.matmul( gids, same_span ).astype(int)
        # this is equivalent to candidate_labels in coref-hoi
        # 1d array of integers where:
        # idx: index in mention list
        # value: cluster id (start at 1, 0 means no cluster/wrong)
        labels = xp.squeeze(labels, 0)
        out.append(labels)
    return out

def get_antecedent_gold(xp, mention_gold: Ints1d, selected_mentions: Ints2d, top_antes, ante_mask) -> Floats2d:
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


def build_take_vecs() -> Model[SpanEmbeddings, Floats2d]:
    # this just gets vectors out of spanembeddings
    return Model("TakeVecs", forward=take_vecs_forward)

def take_vecs_forward(model, inputs: SpanEmbeddings, is_train):
    def backprop(dY: Floats2d) -> SpanEmbeddings:
        ic("TAKE VECS BACK")
        vecs = Ragged(dY, inputs.vectors.lengths)
        ic(dY.shape, inputs.vectors.data.shape)
        return SpanEmbeddings(inputs.indices, vecs)

    return inputs.vectors.data, backprop

def build_ant_scorer(bilinear, dropout) -> Model[SpanEmbeddings, List[Floats2d]]: 
    return Model("AntScorer", 
            forward=ant_scorer_forward,
            layers=[bilinear, dropout])

def ant_scorer_forward(model, inputs: Tuple[Floats1d, SpanEmbeddings], is_train) -> Tuple[List[Floats2d], Ints2d]:
    ops = model.ops
    xp = ops.xp
    # this contains the coarse bilinear in coref-hoi
    # coarse bilinear is a single layer linear network
    #TODO make these proper refs
    bilinear = model.layers[0]
    dropout = model.layers[1]

    #XXX Note on dimensions: This won't work as a ragged because the floats2ds
    # are not all the same dimentions. Each floats2d is a square in the size of
    # the number of antecedents in the document. Actually, that will have the
    # same size if antecedents are padded... Needs checking.

    mscores, sembeds = inputs
    vecs = sembeds.vectors # ragged

    offset = 0
    backprops = []
    out = []
    for ll in vecs.lengths:
        hi = offset+ll
        # each iteration is one doc

        # first calculate the pairwise product scores
        cvecs = vecs.data[offset:hi]
        source, source_b = bilinear(cvecs, is_train)
        target, target_b = dropout(cvecs, is_train)
        pw_prod = xp.matmul(source, target.T)

        # now calculate the pairwise mention scores
        ms = mscores[offset:hi].squeeze()
        pw_sum = xp.expand_dims(ms, 1) + xp.expand_dims(ms, 0)

        # make a mask so antecedents precede referrents
        ant_range = xp.arange(0, cvecs.shape[0])
        with xp.errstate(divide="ignore"):
            mask = xp.log((xp.expand_dims(ant_range, 1) - xp.expand_dims(ant_range, 0)) >= 1).astype(float)

        scores = pw_prod + pw_sum + mask
        out.append(scores)

        offset += ll
        backprops.append( (source_b, target_b, source, target) )

    def backprop(dYs: Tuple[List[Floats2d], Ints2d]) -> Tuple[Floats2d, SpanEmbeddings]:
        dYscores, dYembeds = dYs
        dXembeds = Ragged(ops.alloc2f(*vecs.data.shape), vecs.lengths)
        # TODO actually backprop to these scores
        ic(mscores.shape, pw_sum.shape, sum(vecs.lengths))
        dXscores = ops.alloc1f(*mscores.shape)

        offset = 0
        for dy, (source_b, target_b, source, target), ll in zip(dYscores, backprops, vecs.lengths):
            ic(dy.shape, source.shape, target.shape)
            ic(dy.dtype, source.dtype, target.dtype)
            dS = source_b(dy @ target)
            dT = target_b(dy @ source)
            dXembeds.data[offset:offset+ll] = dS + dT
            # TODO really unsure about this, check it
            dXscores[offset:offset+ll] = xp.diag(dy)
            ic(dS.shape, dT.shape, ms.shape, pw_sum.shape, sum(vecs.lengths))
            offset += ll
        # make it fit back into the linear
        dXscores = xp.expand_dims(dXscores, 1)
        return (dXscores, SpanEmbeddings(sembeds.indices, dXembeds))

    return (out, sembeds.indices), backprop

def build_cluster_maker() -> Model[List[Floats2d], Ints2d]:
    return Model("ClusterMaker", forward=make_clusters)


def make_clusters(model, inputs: Tuple[List[Floats2d], Ints2d], is_train) -> List[List[List[Tuple[int, int]]]]:
    # one item in scores for each doc
    # output: per doc, one list of clusters, which are a list of int spans
    xp = model.ops.xp
    scores, idxs = inputs

    out = []
    offset = 0
    for cscores in scores:
        ll = cscores.shape[0]
        hi = offset + ll

        starts = idxs[offset:hi, 0].tolist()
        ends = idxs[offset:hi, 1].tolist()
        score_idx = xp.argsort(-1 * cscores, 1)

        # need to add the dummy
        dummy = model.ops.alloc2f(cscores.shape[0], 1)
        cscores = xp.concatenate( (dummy, cscores), 1)
        #ic(cscores.shape)

        predicted = get_predicted_clusters(
                starts, ends, score_idx, cscores)
        #ic(predicted)
        out.append(predicted)

    def backward(dY: List[List[List[Tuple[int, int]]]]) -> Tuple[List[Floats2d], Ints2d]:
        offset = 0
        dXs = []
        for docgold, cscores in zip(dY, scores):

            ll = cscores.shape[0]
            hi = offset + ll
            gscores = create_gold_scores(idxs[offset:hi], docgold)
            #ic(gscores)
            gscores = model.ops.asarray2f(gscores)
            # remove the dummy
            gscores = gscores[:,1:]

            # can probably save this somewhere
            #dummy = model.ops.alloc2f(cscores.shape[0], 1)
            #cscores = xp.concatenate( (dummy, cscores), 1)

            # this shouldn't be necessary but for some reason one is a double and
            # one is a float.
            diff = model.ops.asarray2f(cscores - gscores)
            dXs.append(diff)
            #ic(dXs[-1])
            #ic(cscores.dtype, gscores.dtype, dXs[-1].dtype)
        return dXs, idxs

    return out, backward

def prep_model():
    nlp = spacy.load("en_core_web_sm")
    tok2vec = nlp.pipeline[0][1].model
    dim = 96 * 3 # TODO get this from the model or something

    span_embedder = build_span_embedder(get_candidate_mentions)

    hidden = 1000
    mention_scorer = chain(Linear(nI=dim, nO=hidden), Relu(nI=hidden, nO=hidden), Dropout(0.3), Linear(nI=hidden, nO=1))
    mention_scorer.initialize()

    bilinear = chain(Linear(nI=dim, nO=dim), Dropout(0.3))
    bilinear.initialize()
 
    model = chain(
            tuplify(tok2vec, noop()),
            span_embedder,
            #XXX this doesn't work
            tuplify_span_embeds(
                # output here is technically floats2d...
                # maybe needs with_flatten.
                chain(build_take_vecs(), mention_scorer),  
                noop()
            ),
            build_coarse_pruner(20), # [Floats1d, SpanEmbeds] -> [Floats1d, SpanEmbeds]
            build_ant_scorer(bilinear, Dropout(0.3)), # [Floats1d, SpanEmbeds] -> [List[Floats2d] (scores), Ints2d (mentions)]
            build_cluster_maker(), # [Floats2d, Ints2d] -> List[List[Tuple[Int,Int]]]
            )
    return model

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

def create_gold_scores(ments: Ints2d, clusters: List[List[Tuple[int, int]]]):
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


def load_training_data(nlp, path):
    from spacy.tokens import DocBin

    db = DocBin().from_disk(path)
    docs = list(db.get_docs(nlp.vocab))

    #TODO check if this has issues with gold tokenization
    raw = [make_clean_doc(nlp, doc) for doc in docs]
    return raw, docs

def train_loop(nlp):
    model = prep_model()
    train_X, train_Y = load_training_data(nlp, "stuff.spacy")

    print(f"Loaded {len(train_X)} examples to train on")

    from thinc.api import Adam, fix_random_seed
    from tqdm import tqdm

    fix_random_seed(23)
    optimizer = Adam(0.001)
    batch_size = 32
    epochs = 10

    for ii in range(epochs):
        batches = model.ops.multibatch(batch_size, train_X, train_Y, shuffle=True)
        for X, Y in tqdm(batches):
            Yh, backprop = model.begin_update(X)
            # Yh is List[List[List[Tuple[int, int]]]]

            clusters = [get_clusters_from_doc(yy) for yy in Y]
            backprop(clusters)

            model.finish_update(optimizer)
        # TODO evaluate
        """
        correct = 0
        total = 0
        for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
            Yh = model.predict(X)
            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]

        score = correct / total
        print(f" {i} accuracy: {float(score):.3f}")
        """


if __name__ == "__main__":
    #from thinc.api import get_current_ops
    #ops = get_current_ops()
    #gold_data_test(ops.xp)

    import spacy
    nlp = spacy.load("en_core_web_sm")

    texts = [
            "John called from London, he says it's raining in the city. He's all wet.",
            "Tarou went to Tokyo Tower. It was sunny there.",
            ]
    #docs = [nlp(text) for text in texts]

    #model = prep_model()

    #out, backprop = model(docs, False)
    #ic(out)

    train_loop(nlp)


