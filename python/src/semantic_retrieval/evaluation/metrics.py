import logging
from functools import partial
from typing import TypeVar

from semantic_retrieval.common.core import LOGGER_FMT
from semantic_retrieval.evaluation.lib import (
    EvaluationMetric,
    IDSet,
    IDSetPairEvalDataset,
    NumericalEvalDataset,
    SampleEvaluationFunction,
    SampleEvaluationParams,
    SampleEvaluationResult,
)
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)

T = TypeVar("T")


def accuracy_metric(dataset: NumericalEvalDataset) -> float:
    return float(accuracy_score(dataset.ground_truth, dataset.output))


def jaccard_similarity(s1: IDSet, s2: IDSet):
    logger.debug(f"{s1=}, {s2=}")
    intersection = s1 & s2
    union = s1 | s2
    logger.debug(f"{intersection=}, {union=}")

    diff = s1 - s2, s2 - s1
    logger.debug(f"missing={diff}")
    logger.debug(f"{len(intersection)=}, {len(union)=}")
    return len(intersection) / len(union)


InterpretationJaccardSimilarity = EvaluationMetric(
    name="Jaccard similarity with ground truth",
    best_value=1.0,
    worst_value=0.0,
)


def idset_jaccard_with_gt(
    output_datum: IDSet, gt: IDSet, name: str
) -> SampleEvaluationResult[IDSet]:
    value = jaccard_similarity(gt, output_datum)
    return SampleEvaluationResult(
        name=name,
        value=value,
        interpretation=InterpretationJaccardSimilarity,
    )


def id_set_pair_to_jaccard_params(
    idset_pair: IDSetPairEvalDataset, name: str
) -> SampleEvaluationParams[IDSet]:
    """
    Take an ID set set pair and prepare it
    for evaluation using Jaccard similarity.
    Input: Pair of ID sets (output and expected).

    This function prepares the data for `evaluate()`.
    """
    gt: IDSet = IDSet(idset_pair.input_set)

    eval_fn_jaccard: SampleEvaluationFunction[IDSet] = partial(
        idset_jaccard_with_gt, gt=gt, name=name
    )

    output_set: IDSet = IDSet(idset_pair.output_set)

    return SampleEvaluationParams(
        output_sample=output_set,
        evaluation_fn=eval_fn_jaccard,
    )
