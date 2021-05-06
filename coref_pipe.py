from typing import List, Iterable, Optional, Dict, Tuple

from thinc.types import Floats2d, Ints2d
from thinc.api import Model, Config, Optimizer, CategoricalCrossentropy

from spacy.tokens.doc import Doc
from spacy.training import Example
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.language import Language
from spacy.vocab import Vocab

from coref_util import (
    create_gold_scores,
    MentionClusters,
    get_clusters_from_doc,
    logsumexp,
    get_predicted_clusters,
)


default_config = """
[model]
@architectures = "spacy.Coref.v0"
max_span_width = 20
mention_limit = 3900
dropout = 0.3
hidden = 1000
@get_mentions = "spacy.CorefCandidateGenerator.v0"
"""

DEFAULT_MODEL = Config().from_str(default_config)["model"]

DEFAULT_CLUSTER_PREFIX = "coref_clusters"


@Language.factory(
    "coref",
    assigns=["doc.spans"],
    requires=["doc.spans"],
    default_config={
        "model": DEFAULT_MODEL,
        "span_cluster_prefix": DEFAULT_CLUSTER_PREFIX,
    },
    default_score_weights={"coref_f": 1.0, "coref_p": None, "coref_r": None},
)
def make_coref(
    nlp: Language,
    name: str,
    model,
    span_cluster_prefix: str = "coref",
) -> "CoreferenceResolver":
    """Create a CoreferenceResolver component."""

    return CoreferenceResolver(nlp.vocab, model, name, span_cluster_prefix)


class CoreferenceResolver(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "coref",
        span_cluster_prefix: str = "coref",
    ) -> None:
        """Initialize a coreference resolution component."""

        self.vocab = vocab
        self.model = model
        self.name = name
        self.span_cluster_prefix = span_cluster_prefix
        self.loss = CategoricalCrossentropy()

        self.cfg = {}

    def set_annotations(self, docs: Iterable[Doc], clusters_by_doc) -> None:
        """Set the clusters on docs using predictions."""

        for doc, clusters in zip(docs, clusters_by_doc):
            clusters, _ = clusters  # pop off backpropr (shouldn't be done here??)
            for ii, cluster in enumerate(clusters):
                key = self.span_cluster_prefix + "_" + str(ii)
                if key in doc.spans:
                    raise ValueError(
                        "Found coref clusters incompatible with the "
                        "documents provided to the 'coref' component. "
                        "This is likely a bug in spaCy."
                    )

                doc.spans[key] = []
                for mention in cluster:
                    doc.spans[key].append(doc[mention[0] : mention[1]])

    def predict(self, docs: Iterable[Doc]) -> List[MentionClusters]:
        preds, _ = self.model.predict(docs)

        xp = self.model.ops.xp
        scores, idxs = preds

        out = []
        offset = 0
        for cscores in scores:
            ll = cscores.shape[0]
            hi = offset + ll

            starts = idxs[offset:hi, 0].tolist()
            ends = idxs[offset:hi, 1].tolist()
            score_idx = xp.argsort(-1 * cscores, 1)

            # need to add the placeholder
            placeholder = self.model.ops.alloc2f(cscores.shape[0], 1)
            cscores = xp.concatenate((placeholder, cscores), 1)

            predicted = get_predicted_clusters(xp, starts, ends, score_idx, cscores)
            out.append(predicted)
        return out

    def get_loss(
        self,
        examples: Iterable[Example],
        # TODO convert next to ragged?
        score_matrix: List[Tuple[Floats2d, Ints2d]],
        mention_idx: Ints2d,
    ):

        ops = self.model.ops
        xp = ops.xp

        offset = 0
        gradients = []
        loss = 0
        for example, (cscores, cidx) in zip(examples, score_matrix):
            # assume cids has absolute mention ids

            ll = cscores.shape[0]
            hi = offset + ll

            clusters = get_clusters_from_doc(example.reference)
            gscores = create_gold_scores(mention_idx[offset:hi], clusters)
            gscores = xp.asarray(gscores)
            top_gscores = xp.take_along_axis(gscores, cidx, axis=1)
            # now add the placeholder
            gold_placeholder = ~top_gscores.any(axis=1).T
            gold_placeholder = xp.expand_dims(gold_placeholder, 1)
            top_gscores = xp.concatenate((gold_placeholder, top_gscores), 1)

            # boolean to float
            top_gscores = ops.asarray2f(top_gscores)

            # add the placeholder to cscores
            placeholder = self.model.ops.alloc2f(ll, 1)
            cscores = xp.concatenate((placeholder, cscores), 1)
            # with xp.errstate(divide="ignore"):
            #    log_marg = xp.logaddexp.reduce(cscores + xp.log(gscores), 1)
            # log_norm = logsumexp(xp, cscores, 1)
            # log_marg = logsumexp(xp, cscores + xp.log(top_gscores), 1)

            # why isn't this just equivalent to xp.log(top_gscores) + error?

            # TODO check the math here
            # diff = log_norm - log_marg
            # diff = self.model.ops.asarray2f(cscores - top_gscores)
            # remove the placeholder, which doesn't backprop

            # do softmax to cscores
            cscores = ops.softmax(cscores, axis=1)

            diff = self.loss.get_grad(cscores, top_gscores)
            diff = diff[:, 1:]
            gradients.append((diff, cidx))

            # scalar loss
            # loss += xp.sum(log_norm - log_marg)
            loss += self.loss.get_loss(cscores, top_gscores)
            offset += ll
        return loss, gradients

    def update(
        self,
        examples: Iterable[Example],
        *,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        if losses is None:
            losses = {}

        losses.setdefault(self.name, 0.0)
        if not examples:
            return losses

        inputs = (example.predicted for example in examples)
        preds, backprop = self.model.begin_update(inputs)
        score_matrix, mention_idx = preds
        loss, d_scores = self.get_loss(examples, score_matrix, mention_idx)
        backprop(d_scores)

        if sgd is not None:
            self.finish_update(sgd)

        losses[self.name] += loss

        return losses
