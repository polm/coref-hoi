from typing import List, Iterable, Optional, Dict

from thinc.types import Floats2d, Ints2d
from thinc.api import Model, Config, Optimizer

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
config_name = "bert_small"
model_path = "data/bert_small/model_Mar21_19-13-39_65000.bin"
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

            # need to add the dummy
            dummy = self.model.ops.alloc2f(cscores.shape[0], 1)
            cscores = xp.concatenate((dummy, cscores), 1)

            predicted = get_predicted_clusters(xp, starts, ends, score_idx, cscores)
            out.append(predicted)
        return out

    def get_loss(
        self,
        examples: Iterable[Example],
        # TODO convert next to ragged?
        score_matrix: List[Floats2d],
        mention_idx: Ints2d,
    ):

        ops = self.model.ops
        xp = ops.xp

        offset = 0
        gradients = []
        loss = 0
        for example, cscores in zip(examples, score_matrix):

            ll = cscores.shape[0]
            hi = offset + ll
            clusters = get_clusters_from_doc(example.reference)
            gscores = create_gold_scores(mention_idx[offset:hi], clusters)
            # boolean to float
            gscores = ops.asarray2f(gscores)
            # add the dummy to cscores
            dummy = self.model.ops.alloc2f(ll, 1)
            cscores = xp.concatenate((dummy, cscores), 1)
            # with xp.errstate(divide="ignore"):
            #    log_marg = xp.logaddexp.reduce(cscores + xp.log(gscores), 1)
            log_marg = logsumexp(xp, cscores + xp.log(gscores), 1)
            log_norm = logsumexp(xp, cscores, 1)

            diff = self.model.ops.asarray2f(cscores - gscores)
            # remove the dummy, which doesn't backprop
            diff = diff[:, 1:]
            gradients.append(diff)

            # scalar loss
            loss += xp.sum(log_norm - log_marg)
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
