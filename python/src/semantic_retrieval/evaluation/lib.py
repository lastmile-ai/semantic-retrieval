import json
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    NewType,
    Protocol,
    Sequence,
    Set,
    TypeVar,
)

import pandas as pd
from pydantic import root_validator
from result import Ok, Result
from semantic_retrieval.common import types
from semantic_retrieval.common.core import LOGGER_FMT

logger = logging.getLogger(__name__)
logging.basicConfig(format=LOGGER_FMT)


T = TypeVar("T")


class IDSetPairEvalDataset(types.Record):
    input_set: Set[str]
    output_set: Set[str]

    def __repr__(self):
        return json.dumps(
            {
                "input_set": list(self.input_set),
                "output_set": list(self.output_set),
            },
            indent=2,
        )

    def __str__(self) -> str:
        return self.__repr__()


class NumericalEvalDataset(types.Record):
    output: Sequence[float | int]
    ground_truth: Sequence[float | int]


IDSetPairEvalDataPathMuncher = Callable[
    [str, str], Awaitable[IDSetPairEvalDataset]
]
NumericalEvalDataPathMuncher = Callable[
    [str, str], Awaitable[NumericalEvalDataset]
]


class EvaluationMetric(types.Record):
    name: str
    best_value: float
    worst_value: float


IDSet = NewType("IDSet", Set[str])

T_OutputDatum = TypeVar("T_OutputDatum", contravariant=True)


class SampleEvaluationResult(types.Record, Generic[T_OutputDatum]):
    name: str
    value: float
    interpretation: EvaluationMetric

    @root_validator(pre=True)
    def check_value_range(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        wv, bv = (
            values["interpretation"].worst_value,
            values["interpretation"].best_value,
        )
        value = values["value"]
        if wv == bv:
            raise ValueError("best_value and worst_value cannot be equal")
        if wv < bv and not wv <= value <= bv:
            raise ValueError(
                f"value {value} is not in range [{wv}, {bv}] (inclusive)"
            )
        if wv > bv and not wv >= value >= bv:
            raise ValueError(
                f"value {value} is not in range [{bv}, {wv}] (inclusive)"
            )

        return values


class SampleEvaluationFunction(Protocol, Generic[T_OutputDatum]):
    def __call__(
        self, output_datum: T_OutputDatum
    ) -> SampleEvaluationResult[T_OutputDatum]:
        return SampleEvaluationResult(
            name="example",
            value=0.0,
            interpretation=EvaluationMetric(
                name="example", best_value=1.0, worst_value=0.0
            ),
        )


class DatasetEvaluationResult(Generic[T_OutputDatum], types.Record):
    results: Sequence[SampleEvaluationResult[T_OutputDatum]]


@dataclass
class SampleEvaluationParams(Generic[T_OutputDatum]):
    output_sample: T_OutputDatum
    evaluation_fn: SampleEvaluationFunction[T_OutputDatum]

    def __str__(self) -> str:
        return f"\nSampleEvaluationParams:\n\t{self.output_sample=}\n\t{self.evaluation_fn=}"


def evaluate(
    evaluation_params_list: Sequence[SampleEvaluationParams[T_OutputDatum]],
) -> Result[DatasetEvaluationResult[T_OutputDatum], str]:
    results = []

    for eval_params in evaluation_params_list:
        sample, evaluation_fn = (
            eval_params.output_sample,
            eval_params.evaluation_fn,
        )
        res_ = evaluation_fn(sample)
        logger.debug(f"{res_=}")
        results.append(res_)

    return Ok(DatasetEvaluationResult(results=results))


def eval_res_to_df(
    eval_res: DatasetEvaluationResult[
        T_OutputDatum  # pyright: ignore[reportInvalidTypeVarUse]
    ],
):
    records = []
    for sample_res in eval_res.results:
        records.append(
            {
                "name": sample_res.name,
                "value": sample_res.value,
                "interpretation": sample_res.interpretation.name,
            }
        )
    return pd.DataFrame.from_records(records)
