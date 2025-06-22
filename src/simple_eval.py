from requests import Response
from nanovllm import LLM, SamplingParams
import huggingface_hub
from typing import List
from simple_data import MinervaMath
from simple_metric import Instance, MathMetric


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