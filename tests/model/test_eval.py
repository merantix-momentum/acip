import time

from accelerate import init_on_device

from acip.core.acip_model import ACIPModel
from acip.eval.eval_efficiency import EfficiencyModelEvaluator
from acip.eval.eval_generation import GenerationModelEvaluator
from acip.eval.eval_lm_eval import LMEvalModelEvaluator
from acip.eval.eval_model_size import SizeModelEvaluator
from acip.eval.eval_ppl import PPLModelEvaluator
from acip.eval.evaluator import ComposedModelEvaluator
from acip.model.tokenizer_factory import AutoTokenizerFactory


def test_eval(acip_model_config, test_device):
    """Test if evaluation of an ACIP model runs through."""
    with init_on_device(test_device):
        config = acip_model_config
        acip_model = ACIPModel(config).to(test_device)

    tokenizer_factory = AutoTokenizerFactory(
        model_name_or_path=acip_model.base_model_name_or_path,
        tokenizer_kwargs={"use_fast": True, "padding_side": "left"},
    )

    evaluator = ComposedModelEvaluator(
        evaluators={
            "model_size": SizeModelEvaluator(eval_parametrized_modules=True),
            "ppl": PPLModelEvaluator(
                tokenizer_factory=tokenizer_factory,
                dataset_names=["c4", "wikitext"],
                ctx_length=2048,
            ),
            "lm_eval": LMEvalModelEvaluator(
                tokenizer_factory=tokenizer_factory,
                task_names=["piqa", "hellaswag"],
                batch_size=64,
                num_fewshots=0,
            ),
            "efficiency": EfficiencyModelEvaluator(
                tokenizer_factory=tokenizer_factory,
                eval_model_size=True,
                eval_inference=True,
            ),
            "generation": GenerationModelEvaluator(
                tokenizer_factory=tokenizer_factory,
                generation_prompts=["The capital of France is", "Berlin is the capital of"],
            ),
        }
    )
    time_start = time.time()
    results = evaluator(model=acip_model)
    print(results)
    time_stop = time.time()
    print(f"Evaluation completed after {round((time_stop - time_start) / 60, 2)}mins.")
