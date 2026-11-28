"""Inference runner for serialized output training (SOT)."""

from __future__ import annotations

from espnet3.systems.esp2_s2t.src.inference_runner import ESP2S2TInferenceRunner


class SotInferenceRunner(ESP2S2TInferenceRunner):
    """Keep the timestamps and speaker-change symbols in the ``hyp`` field.

    Use this runner with
    :class:`espnet3.systems.esp2_s2t.src.sot.s2t_inference.Speech2Text`. The
    input handling is the same as in :class:`ESP2S2TInferenceRunner`.
    """

    @staticmethod
    def format_output(result, *, batched):
        """Render the best hypothesis, timestamps and separators intact.

        Args:
            result: The n-best list of ``Speech2Text``.
            batched: Whether the call was a batch.

        The text comes from the converter of the decoder. The SOT
        ``Speech2Text`` builds that converter so that it keeps special tokens.
        A separator that the checkpoint added to its vocabulary therefore stays
        in the text. Only the prompt is removed, which is the language symbol
        and the task symbol at the start.

        Returns:
            ``{"hyp": ...}``. The value is ``""`` when the decoder returned no
            hypothesis.

        Raises:
            ValueError: If the call was a batch. The timestamp rules track
                one prompt length for the whole beam, so SOT decoding takes
                one utterance group at a time (``batch_size: null``).
            ValueError: If the text does not start with the two prompt
                tokens, which means a converter dropped special tokens.

        Examples:
            >>> tokens = ["<|en|>", "<|transcribe|>", "<|0.00|>", " hi", "<|0.40|>"]
            >>> result = [("".join(tokens), tokens, [], " hi", None)]
            >>> SotInferenceRunner.format_output(result, batched=False)
            {'hyp': '<|0.00|> hi<|0.40|>'}
        """
        if batched:
            raise ValueError(
                "SOT decoding takes one utterance group at a time. "
                "Set batch_size: null in the inference config."
            )
        if not result:
            return {"hyp": ""}
        text, tokens, *_ = result[0]
        prompt = "".join(tokens[:2])
        # A converter that drops special tokens leaves words in these two slots.
        special = [t for t in tokens[:2] if t.startswith("<|") and t.endswith("|>")]
        if len(special) != 2 or not text.startswith(prompt):
            raise ValueError(
                f"hypothesis {text!r} does not start with its prompt {prompt!r}. "
                "Decode with espnet3.systems.esp2_s2t.src.sot.s2t_inference."
                "Speech2Text, and set speaker_change_symbol or adaptive_timestamp."
            )
        return {"hyp": text[len(prompt) :]}
