"""Beam-search caption generation.

Greedy decoding (the only Phase 1 option) routinely produces generic captions
because the model's most-likely-next-token at every step rarely lines up with
the most-likely-*sequence*. Beam search explores multiple partial captions in
parallel and ranks them by total log-probability, lifting BLEU-4 by 2-5
points on most transformer captioners without retraining.

Algorithm (standard beam search with length and repetition controls):
    * Maintain ``beam_width`` active beams, each a (token-id sequence, score).
    * At each step, batch every active beam through the decoder once, take the
      log-softmax at the current position, apply the repetition penalty and
      the optional no-repeat-ngram block, and pick the global top-K
      candidates across (beam, vocab) pairs.
    * Beams that emit ``[end]`` move into the finished list (their score is
      already final at that point); the search ends when ``beam_width`` beams
      have finished or we hit the max-length budget.
    * Final ranking divides each finished beam's score by
      ``len(seq) ** length_penalty`` so the search isn't biased toward very
      short sequences (the classic length problem in beam search).

This implementation is intentionally kept *callable* — the same predictor
class dispatches between :func:`generate_caption_greedy` and this one based
on ``decode_strategy``. Phase 3 model wrappers (BLIP, ViT-GPT2) can reuse
the same dispatcher.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from captioning.preprocessing.caption import END_TOKEN, START_TOKEN
from captioning.preprocessing.tokenizer import CaptionTokenizer

_LOG_EPSILON = 1e-12


@dataclass
class _Beam:
    """One partial caption under exploration."""

    token_ids: list[int]
    score: float
    finished: bool = False
    history: set[int] = field(default_factory=set)

    def length(self) -> int:
        """Number of generated tokens (excludes the seed [start] token)."""
        return max(len(self.token_ids) - 1, 1)


def _apply_repetition_penalty(
    log_probs,
    history_ids: set[int],
    penalty: float,
):
    """Subtract ``log(penalty)`` from already-seen tokens' log-probabilities.

    HuggingFace's repetition_penalty (Keskar et al. 2019) divides logits by
    ``penalty`` (>1) for tokens already in the context. We work with log-
    probabilities here, so the equivalent operation is to *subtract*
    ``log(penalty)`` for positive log-probabilities and add it for negative
    ones — but log-probabilities are always non-positive, so we always make
    seen tokens less likely. That is the correct direction (we want to
    discourage repetition).
    """
    if penalty <= 1.0 or not history_ids:
        return log_probs
    log_pen = math.log(penalty)
    for tid in history_ids:
        if 0 <= tid < log_probs.shape[-1]:
            log_probs[tid] -= log_pen
    return log_probs


def _blocks_repeat_ngram(seq: list[int], candidate: int, n: int) -> bool:
    """Return True if appending ``candidate`` would repeat an n-gram in ``seq``."""
    if n <= 0 or len(seq) < n - 1:
        return False
    tail = tuple(seq[-(n - 1) :] + [candidate]) if n > 1 else (candidate,)
    return any(tuple(seq[i : i + n]) == tail for i in range(len(seq) - n + 1))


def generate_caption_beam(  # — beam search has many knobs by nature
    model,
    tokenizer: CaptionTokenizer,
    image_tensor,
    max_length: int,
    *,
    beam_width: int = 3,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    """Generate a caption using beam search with optional length / repetition control.

    Args:
        model: An ``ImageCaptioningModel`` whose weights have been loaded.
        tokenizer: Fitted :class:`CaptionTokenizer`.
        image_tensor: ``[299, 299, 3]`` float tensor as produced by
            ``inference.load_image_from_path``.
        max_length: Same budget as greedy (``config.model.max_length``); the
            search stops at the first of (all beams finished, length exhausted).
        beam_width: Number of parallel hypotheses. ``1`` reduces to greedy.
        length_penalty: GNMT-style penalty exponent. ``score / len ** alpha``.
            ``0.0`` disables it; ``0.6-1.0`` is the common range. Higher values
            favour longer captions.
        repetition_penalty: HuggingFace's CTRL-style penalty. ``1.0`` disables
            it; ``>1.0`` penalises tokens already in the partial caption.
        no_repeat_ngram_size: If ``> 0``, forbids emitting any token that
            would complete an n-gram already present in the partial caption.
            ``3`` is a common choice for captioning.

    Returns:
        The best-scoring caption (sentinels stripped, same convention as
        :func:`generate_caption_greedy`).
    """
    import numpy as np
    import tensorflow as tf

    # 1. Encode the image once. Beams share the encoded features.
    img = tf.expand_dims(image_tensor, axis=0)
    img_embed = model.cnn_model(img)
    img_encoded = model.encoder(img_embed, training=False)

    start_id = tokenizer.word_to_id(START_TOKEN)
    end_id = tokenizer.word_to_id(END_TOKEN)

    # 2. Initialise a single seed beam containing only the [start] token.
    beams: list[_Beam] = [_Beam(token_ids=[start_id], score=0.0, history={start_id})]
    finished: list[_Beam] = []

    decode_steps = max_length - 1  # decoder is fed sequences of length max_length-1

    for step in range(decode_steps):
        if not beams:
            break

        # 3. Batch every active beam into a single decoder forward pass.
        token_batch = np.zeros((len(beams), decode_steps), dtype=np.int64)
        for i, beam in enumerate(beams):
            seq = beam.token_ids[:decode_steps]
            token_batch[i, : len(seq)] = seq

        token_tensor = tf.convert_to_tensor(token_batch)
        mask = tf.cast(token_tensor != 0, tf.int32)
        # Encoded features must be broadcast to match the beam batch dimension.
        encoded_batch = tf.repeat(img_encoded, repeats=len(beams), axis=0)
        preds = model.decoder(token_tensor, encoded_batch, training=False, mask=mask)
        # preds is [B, T, V]; we read position `step` for each beam.
        step_probs = preds.numpy()[:, step, :]
        step_log_probs = np.log(step_probs + _LOG_EPSILON)

        # 4. Expand every beam, then keep the global top-K.
        candidates: list[_Beam] = []
        vocab_size = step_log_probs.shape[-1]
        for i, beam in enumerate(beams):
            lp = step_log_probs[i].copy()
            lp = _apply_repetition_penalty(lp, beam.history, repetition_penalty)

            # Pick a wider candidate pool than beam_width per beam — when most
            # beams want the same token, expansion needs slack to remain diverse.
            pool = min(beam_width * 2, vocab_size)
            top_ids = np.argpartition(-lp, pool - 1)[:pool]
            top_ids = top_ids[np.argsort(-lp[top_ids])]

            for tid in top_ids:
                tid_int = int(tid)
                if no_repeat_ngram_size > 0 and _blocks_repeat_ngram(
                    beam.token_ids, tid_int, no_repeat_ngram_size
                ):
                    continue
                new_seq = [*beam.token_ids, tid_int]
                new_score = beam.score + float(lp[tid_int])
                new_history = beam.history | {tid_int}
                candidates.append(
                    _Beam(
                        token_ids=new_seq,
                        score=new_score,
                        finished=(tid_int == end_id),
                        history=new_history,
                    )
                )

        # 5. Sort candidates by score and keep the top ``beam_width`` actives.
        candidates.sort(key=lambda b: b.score, reverse=True)
        next_beams: list[_Beam] = []
        for cand in candidates:
            if cand.finished:
                finished.append(cand)
                continue
            next_beams.append(cand)
            if len(next_beams) >= beam_width:
                break
        beams = next_beams

        # 6. Early termination — we already have enough finished beams and
        # none of the active ones can beat the best finished score (their
        # best-case future log-prob is 0, so length-normalised score won't
        # beat the current top).
        if len(finished) >= beam_width and beams:
            best_finished = max(_length_normalised(b, length_penalty) for b in finished)
            best_active_upper_bound = max(_length_normalised(b, length_penalty) for b in beams)
            if best_active_upper_bound <= best_finished:
                break

    # 7. Anything still active at the budget cap counts as finished.
    finished.extend(beams)
    if not finished:
        return ""

    finished.sort(key=lambda b: _length_normalised(b, length_penalty), reverse=True)
    best = finished[0]
    return _detokenize(best.token_ids, tokenizer, end_id)


def _length_normalised(beam: _Beam, alpha: float) -> float:
    """Apply length penalty to a beam score (higher == better)."""
    if alpha == 0.0:
        return beam.score
    return beam.score / (beam.length() ** alpha)


def _detokenize(
    token_ids: list[int],
    tokenizer: CaptionTokenizer,
    end_id: int,
) -> str:
    """Convert beam token ids back to a clean caption string."""
    words: list[str] = []
    for tid in token_ids:
        if tid == end_id:
            break
        word = tokenizer.decode_id(tid)
        # Skip [start], padding, and OOV ids that decode to empty strings.
        if word in {"", START_TOKEN, END_TOKEN, "[UNK]"}:
            continue
        words.append(word)
    return " ".join(words)
