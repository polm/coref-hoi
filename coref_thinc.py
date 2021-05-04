from dataclasses import dataclass

from thinc.api import Model, Linear, Relu, Dropout, chain, concatenate
from thinc.api import list2ragged, reduce_mean, ragged2list, noop
from thinc.types import Pairs, Floats2d, Floats1d, DTypesFloat, Ints2d, Ragged, Ints1d
import spacy
from spacy.tokens import Doc, Span
from typing import cast, List, Callable, Any, Tuple

from collections import namedtuple
from coref_util import (
    get_predicted_clusters,
    get_candidate_mentions,
    select_non_crossing_spans,
    get_clusters_from_doc,
    make_clean_doc,
    create_gold_scores,
)

from icecream import ic

# type alias to make writing this less tedious
MentionClusters = List[List[Tuple[int, int]]]


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


@dataclass
class SpanEmbeddings:
    indices: Ints2d  # Array with 2 columns (for start and end index)
    vectors: Ragged  # Ragged[Floats2d] # One vector per span
    # NB: We assume that the indices refer to a concatenated Floats2d that
    # has one row per token in the *batch* of documents. This makes it unambiguous
    # which row is in which document, because if the lengths are e.g. [10, 5],
    # a span starting at 11 must be starting at token 2 of doc 1. A bug could
    # potentially cause you to have a span which crosses a doc boundary though,
    # which would be bad.
    # The lengths in the Ragged are not the tokens per doc, but the number of
    # mentions per doc.

    def __add__(self, right):
        # XXX is something like this necessary?
        # assert self.indices == right.indices, "Can only add with equal indices"
        # assert self.vectors.lengths == right.vectors.lengths, "Can only add with equal lengths"

        out = self.vectors.data + right.vectors.data
        return SpanEmbeddings(self.indices, Ragged(out, self.vectors.lengths))

    def __iadd__(self, right):
        self.vectors.data += right.vectors.data
        return self


# model converting a Doc/Mention to span embeddings
# mention_generator: Callable[Doc, Pairs[int]]
def build_span_embedder(
    mention_generator,
) -> Model[Tuple[List[Floats2d], List[Doc]], SpanEmbeddings]:

    return Model(
        "SpanEmbedding",
        forward=span_embeddings_forward,
        attrs={
            "mention_generator": mention_generator,
        },
    )


def span_embeddings_forward(
    model, inputs: Tuple[List[Floats2d], List[Doc]], is_train
) -> SpanEmbeddings:
    ops = model.ops
    xp = ops.xp

    tokvecs, docs = inputs

    dim = tokvecs[0].shape[1]

    mgen = model.attrs["mention_generator"]
    mentions = ops.alloc2i(0, 2)
    total_length = 0
    docmenlens = []  # number of mentions per doc
    for doc in docs:
        mm = mgen(doc)
        docmenlens.append(len(mm))
        cments = ops.asarray2i([mm.one, mm.two]).transpose()

        mentions = xp.concatenate((mentions, cments + total_length))
        total_length += len(doc)

    # TODO support attention here
    tokvecs = xp.concatenate(tokvecs)
    spans = [tokvecs[ii:jj] for ii, jj in mentions.tolist()]
    idx = 10
    avgs = [xp.mean(ss, axis=0) for ss in spans]
    spanvecs = ops.asarray2f(avgs)

    # first and last token embeds
    starts = [tokvecs[ii] for ii in mentions[:, 0]]
    ends = [tokvecs[jj] for jj in mentions[:, 1]]

    starts = ops.asarray2f(starts)
    ends = ops.asarray2f(ends)
    concat = xp.concatenate((starts, ends, spanvecs), 1)
    embeds = Ragged(concat, docmenlens)

    def backprop_span_embed(dY: SpanEmbeddings) -> Tuple[List[Floats2d], List[Doc]]:

        oweights = []
        odocs = []
        offset = 0
        tokoffset = 0
        for indoc, mlen in zip(docs, dY.vectors.lengths):
            hi = offset + mlen
            hitok = tokoffset + len(indoc)
            odocs.append(indoc)  # no change
            vecs = dY.vectors.data[offset:hi]

            starts = vecs[:, :dim]
            ends = vecs[:, dim : 2 * dim]
            spanvecs = vecs[:, 2 * dim :]

            out = model.ops.alloc2f(len(indoc), dim)

            for ii, (start, end) in enumerate(dY.indices[offset:hi]):
                # adjust indexes to align with doc
                start -= tokoffset
                end -= tokoffset

                out[start] += starts[ii]
                out[end] += ends[ii]
                out[start:end] += spanvecs[ii]
            oweights.append(out)

            offset = hi
            tokoffset = hitok
        return oweights, odocs

    return SpanEmbeddings(mentions, embeds), backprop_span_embed


# SpanEmbeddings -> SpanEmbeddings
def build_coarse_pruner(
    mention_limit: int,
) -> Model[SpanEmbeddings, SpanEmbeddings]:
    model = Model(
        "CoarsePruner",
        forward=coarse_prune,
        attrs={
            "mention_limit": mention_limit,
        },
    )
    return model


def coarse_prune(
    model, inputs: Tuple[Floats1d, SpanEmbeddings], is_train
) -> SpanEmbeddings:
    # input spanembeddings are *all* candidate mentions
    # output spanembeddings are *pruned* mentions
    # do scoring
    rawscores, spanembeds = inputs
    scores = rawscores.squeeze()
    # do pruning
    mention_limit = model.attrs["mention_limit"]
    # XXX: Issue here. Don't need docs to find crossing spans, but might for the limits.
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
        csel = select_non_crossing_spans(tops, starts, ends, mention_limit)
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

    def coarse_prune_backprop(
        dY: Tuple[Floats1d, SpanEmbeddings]
    ) -> Tuple[Floats1d, SpanEmbeddings]:
        ll = spanembeds.indices.shape[0]

        dYscores, dYembeds = dY

        dXscores = model.ops.alloc1f(ll)
        # ic(dXscores.shape, selected.shape, dYscores.shape)
        # I think we only backprop the selected ones here?
        # TODO check this
        # dXscores[selected] = dYscores[selected]
        # actually, maybe this is pass through?
        dXscores[selected] = dYscores.squeeze()

        dXvecs = model.ops.alloc2f(*spanembeds.vectors.data.shape)
        # ic(dXscores.shape, dYscores.shape, dXvecs.shape, dYembeds.vectors.data.shape)
        # ic(spanembeds.indices.shape)
        dXvecs[selected] = dYembeds.vectors.data
        dXembeds = SpanEmbeddings(spanembeds.indices, dXvecs)

        # inflate for mention scorer
        dXscores = model.ops.xp.expand_dims(dXscores, 1)

        return (dXscores, dXembeds)

    return (scores[selected], out), coarse_prune_backprop


def build_take_vecs() -> Model[SpanEmbeddings, Floats2d]:
    # this just gets vectors out of spanembeddings
    return Model("TakeVecs", forward=take_vecs_forward)


def take_vecs_forward(model, inputs: SpanEmbeddings, is_train):
    def backprop(dY: Floats2d) -> SpanEmbeddings:
        vecs = Ragged(dY, inputs.vectors.lengths)
        return SpanEmbeddings(inputs.indices, vecs)

    return inputs.vectors.data, backprop


def build_ant_scorer(
    bilinear, dropout, ant_limit=50
) -> Model[SpanEmbeddings, List[Floats2d]]:
    return Model(
        "AntScorer",
        forward=ant_scorer_forward,
        layers=[bilinear, dropout],
        attrs={
            "ant_limit": ant_limit,
        },
    )


def ant_scorer_forward(
    model, inputs: Tuple[Floats1d, SpanEmbeddings], is_train
) -> Tuple[List[Floats2d], Ints2d]:
    ops = model.ops
    xp = ops.xp
    # this contains the coarse bilinear in coref-hoi
    # coarse bilinear is a single layer linear network
    # TODO make these proper refs
    bilinear = model.layers[0]
    dropout = model.layers[1]

    # XXX Note on dimensions: This won't work as a ragged because the floats2ds
    # are not all the same dimentions. Each floats2d is a square in the size of
    # the number of antecedents in the document. Actually, that will have the
    # same size if antecedents are padded... Needs checking.

    mscores, sembeds = inputs
    vecs = sembeds.vectors  # ragged

    offset = 0
    backprops = []
    out = []
    for ll in vecs.lengths:
        hi = offset + ll
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
            mask = xp.log(
                (xp.expand_dims(ant_range, 1) - xp.expand_dims(ant_range, 0)) >= 1
            ).astype(float)

        scores = pw_prod + pw_sum + mask
        out.append(scores)
        # TODO this should be topk'd
        # this is the index in the original scores, which is also the mention index
        # top_scores_idx = xp.argsort(-1 * scores, 1)[:,:ant_limit]
        # These are the actual scores
        # top_scores = scores[top_scores_idx]
        # out.append( (top_scores, top_scores_idx) )

        # In the full model these scores can be further refined. In the current
        # state of this model we're done here, so this pruning is less important.

        offset += ll
        backprops.append((source_b, target_b, source, target))

    def backprop(dYs: Tuple[List[Floats2d], Ints2d]) -> Tuple[Floats2d, SpanEmbeddings]:
        dYscores, dYembeds = dYs
        dXembeds = Ragged(ops.alloc2f(*vecs.data.shape), vecs.lengths)
        dXscores = ops.alloc1f(*mscores.shape)

        offset = 0
        for dy, (source_b, target_b, source, target), ll in zip(
            dYscores, backprops, vecs.lengths
        ):
            # ic(dy.shape, source.shape, target.shape)
            # ic(dy.dtype, source.dtype, target.dtype)

            # first undo the mask so there are no infinite values
            dy[dy == -xp.inf] = 0
            # ic(target)
            # ic(dy)
            # ic(xp.isnan(target).any(), xp.isnan(source).any(), xp.isnan(dy).any())
            dS = source_b(dy @ target)
            dT = target_b(dy @ source)
            dXembeds.data[offset : offset + ll] = dS + dT
            # TODO really unsure about this, check it
            dXscores[offset : offset + ll] = xp.diag(dy)
            # ic(dS.shape, dT.shape, ms.shape, pw_sum.shape, sum(vecs.lengths))
            offset += ll
        # make it fit back into the linear
        dXscores = xp.expand_dims(dXscores, 1)
        return (dXscores, SpanEmbeddings(sembeds.indices, dXembeds))

    return (out, sembeds.indices), backprop


def build_cluster_maker() -> Model[List[Floats2d], Ints2d]:
    return Model("ClusterMaker", forward=make_clusters)


def make_clusters(
    model, inputs: Tuple[List[Floats2d], Ints2d], is_train
) -> Tuple[List[List[List[Tuple[int, int]]]], Callable]:
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
        cscores = xp.concatenate((dummy, cscores), 1)
        # ic(cscores.shape)

        predicted = get_predicted_clusters(xp, starts, ends, score_idx, cscores)
        # ic(predicted)
        out.append(predicted)

    def backward(
        dY: List[List[List[Tuple[int, int]]]]
    ) -> Tuple[List[Floats2d], Ints2d]:
        offset = 0
        dXs = []
        loss = 0
        for docgold, cscores in zip(dY, scores):

            ll = cscores.shape[0]
            hi = offset + ll
            gscores = create_gold_scores(idxs[offset:hi], docgold)
            # ic(gscores)
            # boolean to float
            gscores = model.ops.asarray2f(gscores)
            # remove the dummy
            # gscores = gscores[:,1:]
            # add the dummy to cscores
            dummy = model.ops.alloc2f(ll, 1)
            cscores = xp.concatenate((dummy, cscores), 1)
            with xp.errstate(divide="ignore"):
                log_marg = xp.logaddexp.reduce(cscores + xp.log(gscores), 1)
            log_norm = xp.logaddexp.reduce(cscores, 1)

            # can probably save this somewhere
            # dummy = model.ops.alloc2f(cscores.shape[0], 1)
            # cscores = xp.concatenate( (dummy, cscores), 1)

            # this shouldn't be necessary but for some reason one is a double and
            # one is a float.
            diff = model.ops.asarray2f(cscores - gscores)
            # remove the dummy, which doesn't backprop
            diff = diff[:, 1:]
            dXs.append(diff)

            # do loss calcs
            loss += xp.sum(log_norm - log_marg)
            # ic(dXs[-1])
            # ic(cscores.dtype, gscores.dtype, dXs[-1].dtype)
        print("Cluster loss: ", loss)
        return dXs, idxs

    return out, backward


def build_coref(
    tok2vec: Model,
    mention_getter: Callable = get_candidate_mentions,
    hidden: int = 1000,
    dropout: float = 0.3,
    mention_limit: int = 3900,
):
    dim = tok2vec.get_dim("nO") * 3

    span_embedder = build_span_embedder(mention_getter)

    with Model.define_operators({">>": chain, "&": tuplify}):

        mention_scorer = (
            Linear(nI=dim, nO=hidden)
            >> Relu(nI=hidden, nO=hidden)
            >> Dropout(dropout)
            >> Linear(nI=hidden, nO=1)
        )
        mention_scorer.initialize()

        bilinear = Linear(nI=dim, nO=dim) >> Dropout(dropout)
        bilinear.initialize()

        ms = build_take_vecs() >> mention_scorer

        model = (
            (tok2vec & noop())
            >> span_embedder
            >> (ms & noop())
            >> build_coarse_pruner(mention_limit)
            >> build_ant_scorer(bilinear, Dropout(dropout))
        )
    return model


def load_training_data(nlp, path):
    from spacy.tokens import DocBin

    db = DocBin().from_disk(path)
    docs = list(db.get_docs(nlp.vocab))

    # TODO check if this has issues with gold tokenization
    raw = [make_clean_doc(nlp, doc) for doc in docs]
    return raw, docs


def train_loop(nlp):
    nlp = spacy.load("en_core_web_sm")
    tok2vec = nlp.pipeline[0][1].model

    model = build_coref(tok2vec)
    train_X, train_Y = load_training_data(nlp, "stuff.spacy")[:100]

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
            print("Example:")
            print(X[0])
            for cluster in Yh[0]:
                spans = [X[0][ss:ee].text for ss, ee in cluster]
                print("::", *spans, sep=" | ")
        # TODO evaluate on dev data
        dev_X = train_X
        dev_Y = train_Y
        correct = 0
        total = 0
        for X, Y in model.ops.multibatch(batch_size, dev_X, dev_Y):
            Yh = model.predict(X)
            correct += (Yh.argmax(axis=1) == Y.argmax(axis=1)).sum()
            total += Yh.shape[0]

        score = correct / total
        print(f" {i} accuracy: {float(score):.3f}")


if __name__ == "__main__":

    import spacy

    nlp = spacy.load("en_core_web_sm")

    texts = [
        "John called from London, he says it's raining in the city. He's all wet.",
        "Tarou went to Tokyo Tower. It was sunny there.",
    ]
    train_loop(nlp)
