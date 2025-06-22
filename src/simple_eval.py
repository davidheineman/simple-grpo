from math_extract import extract_answer, is_equiv
from nanovllm import LLM, SamplingParams
import huggingface_hub
from typing import List, Optional
from datasets import load_dataset
from dataclasses import dataclass, field


@dataclass
class Instance:
    request: str
    gold_completion: Optional[str] = None 
    solution: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Response:
    input: Instance
    output: str


class MathMetric:
    def __init__(self, responses: List[Response]):
        self.responses = responses

    def grade_responses(self):
        self.scores = list(map(self._grade_response, self.responses))

    def compute_metric(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def _grade_response(self, response: Response) -> bool:
        generated = response.output
        correct = response.input.solution

        assert correct is not None

        gen_answers: List[str] = extract_answer(
            generated
        )  # extracts many possible answers

        # "math flex" will allow any extracted answer to be correct
        for gen in gen_answers:
            if is_equiv(gen, correct):
                return True

        return False


def main():
    model_name = "Qwen/Qwen3-0.6B"
    # model_name = "Qwen/Qwen3-32B"
    model_path = huggingface_hub.snapshot_download(model_name)

    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.6, max_tokens=1024)

    for subset in MinervaMath.SUBSETS:
        print(f"Evaluating {subset}...")

        dataset = MinervaMath(subset)

        instances: List[Instance] = dataset.requests
        queries: List[str] = [instance.request for instance in instances]

        outputs = llm.generate(queries, sampling_params)

        responses = []
        for input, output in zip(instances, outputs):
            generated = output["text"]
            responses += [Response(input=input, output=generated)]

        metric = MathMetric(responses)
        metric.grade_responses()
        score = metric.compute_metric()

        print(f"\n{subset} Accuracy: {score:.2%}")


if __name__ == '__main__':
    main()

# TODO: "stop_sequences": ["Problem:", "\n\n"],
# TODO: SelfC