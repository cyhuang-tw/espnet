"""Inference runner for ESPnet2 S2T models."""

from __future__ import annotations

from espnet3.contract.kinds.audio import Audio
from espnet3.systems.base.inference_runner import InferenceRunner


class ESP2S2TInferenceRunner(InferenceRunner):
    """Convert the result of an ESPnet2 S2T model to the contracted ``hyp`` field.

    The runner works with ``espnet2.bin.s2t_inference.Speech2Text`` and returns
    the text of the best hypothesis. Serialized output training needs its own
    output format, which
    :class:`espnet3.systems.esp2_s2t.src.sot.inference_runner.SotInferenceRunner`
    provides.
    """

    @staticmethod
    def prepare_model_inputs(inputs):
        """Pass ESPnet2 the samples of the checked mono ``Audio``.

        Args:
            inputs: Keyword arguments for the backend. ``speech`` is an
                ``Audio`` object that the contract has already resampled and
                checked to have one channel.

        Returns:
            A copy of ``inputs`` whose ``speech`` is a 1-D array.

        Examples:
            >>> import numpy as np
            >>> audio = Audio(np.zeros(8000, dtype=np.float32), 16000)
            >>> out = ESP2S2TInferenceRunner.prepare_model_inputs({"speech": audio})
            >>> out["speech"].shape
            (8000,)
        """
        inputs = dict(inputs)
        if isinstance(inputs["speech"], Audio):
            inputs["speech"] = inputs["speech"].array
        return inputs

    @staticmethod
    def format_output(result, *, batched):
        """Return the text of the best hypothesis as ``hyp``.

        Args:
            result: The n-best list of ``Speech2Text``, or a list of n-best
                lists for a batch. Each entry starts with the text.
            batched: Whether the call was a batch.

        Returns:
            ``{"hyp": ...}`` for one call, or a list of them for a batch. The
            value is ``""`` when the model returned no hypothesis or no text.

        Examples:
            >>> result = [("hello", ["hello"], [7], "hello", None)]
            >>> ESP2S2TInferenceRunner.format_output(result, batched=False)
            {'hyp': 'hello'}
        """
        if batched:
            return [ESP2S2TInferenceRunner._format_one(nbest) for nbest in result]
        return ESP2S2TInferenceRunner._format_one(result)

    @staticmethod
    def _format_one(nbest):
        """Return the text of the first entry of one n-best list."""
        if not nbest or nbest[0][0] is None:
            return {"hyp": ""}
        return {"hyp": nbest[0][0]}
