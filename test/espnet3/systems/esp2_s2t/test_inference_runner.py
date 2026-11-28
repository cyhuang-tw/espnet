"""Tests for espnet3/systems/esp2_s2t/src/inference_runner.py."""

import numpy as np

from espnet3.contract.inference import load_inference_contract
from espnet3.systems.esp2_s2t.src.inference_runner import ESP2S2TInferenceRunner

# The contract of egs3/TEMPLATE/esp2_s2t/conf/inference.yaml.
CONTRACT = {
    "inputs": {"speech": {"kind": "audio", "sample_rate": 16000, "channels": 1}},
    "outputs": {"hyp": {"kind": "text"}},
}


def _result(text):
    """Return an n-best list in the format of espnet2's S2T Speech2Text."""
    return [(text, text.split(), [], text, None)]


def test_the_hypothesis_is_the_text_of_the_best_entry():
    """format_output returns the text of the first n-best entry as hyp."""
    nbest = _result("hello world") + _result("hello word")

    assert ESP2S2TInferenceRunner.format_output(nbest, batched=False) == {
        "hyp": "hello world"
    }


def test_no_hypothesis_or_no_text_is_an_empty_hyp():
    """An empty n-best list and a missing text both give an empty hyp."""
    no_text = [(None, [], [], None, None)]

    assert ESP2S2TInferenceRunner.format_output([], batched=False) == {"hyp": ""}
    assert ESP2S2TInferenceRunner.format_output(no_text, batched=False) == {"hyp": ""}


def test_a_batch_gives_one_hyp_per_item():
    """For a batch, format_output formats each n-best list separately."""
    batch = [_result("hello"), _result("hi")]

    assert ESP2S2TInferenceRunner.format_output(batch, batched=True) == [
        {"hyp": "hello"},
        {"hyp": "hi"},
    ]


def test_forward_passes_mono_samples_and_keeps_the_items_id():
    """The forward call resamples the speech, passes a 1-D array and keeps the id."""
    seen = {}

    def model(speech):
        seen["speech"] = speech
        return _result("hello")

    dataset = [{"utt_id": "u1", "speech": (8000, np.zeros(8000))}]
    output = ESP2S2TInferenceRunner.forward(
        0,
        dataset=dataset,
        model=model,
        input_key="speech",
        inference_contract=load_inference_contract(CONTRACT),
    )

    assert output == {"hyp": "hello", "utt_id": "u1"}
    assert seen["speech"].shape == (16000,)
