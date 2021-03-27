from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any

from thinc.types import Floats2d
from thinc.api import Model, Config

from spacy.tokens.doc import Doc
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.language import Language
from spacy.vocab import Vocab
from wasabi import Printer

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

    return CoreferenceResolver(
            nlp.vocab,
            model,
            name,
            span_cluster_prefix)


class CoreferenceResolver(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "coref",
        span_cluster_prefix: str = "coref",
    ) -> None:
        """Initialize a coreference resolution component.
        """

        self.vocab = vocab
        self.model = model
        self.name = name
        self.span_cluster_prefix = span_cluster_prefix

        self.cfg = {}


    def predict(self, docs: Iterable[Doc]):
        """Apply the model without modifying the docs."""

        clusters_by_doc = []
        for i, doc in enumerate(docs):
            clusters = self.model(doc, is_train=False)
            clusters_by_doc.append(clusters)
        return clusters_by_doc

    def set_annotations(self, docs: Iterable[Doc], clusters_by_doc) -> None:
        """Set the clusters on docs using predictions."""


        for doc, clusters in zip(docs, clusters_by_doc):
            clusters, _ = clusters # pop off backpropr (shouldn't be done here??)
            for ii, cluster in enumerate(clusters):
                key = self.span_cluster_prefix + "_" + str(ii)
                if key in doc.spans:
                    raise ValueError("Found coref clusters incompatible with the "
                                     "documents provided to the 'coref' component. "
                                     "This is likely a bug in spaCy.")

                doc.spans[key] = []
                for mention in cluster:
                    doc.spans[key].append(doc[mention[0]:mention[1]])

