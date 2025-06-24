from datasets import load_dataset
from simple_grpo.simple_metric import Instance
from simple_grpo.math_extract import extract_answer

class MinervaMath:
    SUBSETS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    def __init__(self, subset):
        self.dataset = load_dataset(
            path="EleutherAI/hendrycks_math", name=subset, split="test"
        )
        self.build_requests()

    def build_requests(self):
        self.requests = list(map(self._process_instance, self.dataset))

    def _process_instance(self, doc) -> Instance:
        solution = extract_answer(doc["solution"])[0]  # get primary extracted answer

        query = "Problem:\n" + doc["problem"] + "\n\nSolution:"

        return Instance(
            request=query,
            gold_completion=doc["solution"],
            solution=solution,
            metadata={"level": doc.get("level"), "type": doc.get("type")},
        )
    

class HamishMathORZ:
    def __init__(self):
        self.dataset = load_dataset(
            path="hamishivi/rlvr_orz_math_57k_collected", split="train"
        )
        self.build_requests()

    def build_requests(self):
        self.requests = list(map(self._process_instance, self.dataset))
    
    def _process_instance(self, doc) -> Instance:
        # TODO: Lots: (1) better template formatting. (2) dataset mixer
        messages = doc["messages"]
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += "Problem:\n" + msg["content"] + "\n\nSolution:"
                break

        assert len(doc["ground_truth"]) == 1
        solution = doc["ground_truth"][0] # only 1 solution str
        solution = extract_answer(solution)[0]  # get primary extracted answer
        
        return Instance(
            request=prompt,
            solution=solution,
            metadata={
                "dataset": doc.get("dataset")
            }
        )