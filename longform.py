"""Long-form synthesis in one consistent voice.

The voice comes from one of two places:
  - Voice design (default): paragraph 1 is spoken in a voice designed from
    --instruction, and its audio becomes the clone reference for the rest.
  - A reference clip (--ref-audio and --ref-text, its exact transcript): every
    paragraph clones that clip. --instruction is then optional and acts as a
    direction (tone, emotion, pace) on every paragraph, with CFG 4.

Paragraphs in TEXT_FILE are separated by blank lines. Each paragraph is one
request (capped at 1500 frames, i.e. 120 s of audio). In clip mode, the clip,
its transcript, the direction, the paragraph and its audio share one 2048-frame
context, so keep the clip short (5-20 s) and paragraphs under about a minute.

Usage:
  uv run longform.py MODEL_DIR TEXT_FILE OUT_WAV [--instruction TEXT] [--seed N]
                     [--ref-audio WAV --ref-text TEXT] [--fast] [--keep-chunks]
"""

from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from breeze_infer.runtime import (
    load_runtime,
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig
from models.warmup_profile import load_warmup_profile

REPO_ROOT = Path(__file__).resolve().parent
FAST_CONFIG = REPO_ROOT / "configs" / "fast.json"

DEFAULT_INSTRUCTION = (
    "A warm, thoughtful young woman with a clear voice and a calm, reflective delivery."
)
MAX_NEW_TOKENS = 1500
FRAME_RATE = 12.5
GAP_S = 0.4
SEED = 42
DIRECTION_CFG = 4.0  # clip mode with --instruction (the README's Voice Direction setting)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("text_file", type=Path)
    p.add_argument("out_wav", type=Path)
    p.add_argument(
        "--instruction",
        help="voice description used to design the voice in paragraph 1 "
        "(default: a warm young woman); with --ref-audio, an optional direction "
        "for every paragraph",
    )
    p.add_argument(
        "--ref-audio",
        type=Path,
        help="clone this clip for every paragraph instead of designing a voice",
    )
    p.add_argument(
        "--ref-text", help="exact transcript of --ref-audio (required with it)"
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="random seed, reset before every paragraph; with the same instruction, "
        "paragraph 1 and mode it gives the same voice (default %(default)s)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="enable CUDA-graph fast stages (all except backbone prefill); "
        "adds ~15 s warmup",
    )
    p.add_argument(
        "--keep-chunks",
        action="store_true",
        help="also save each paragraph's WAV to OUT_WAV's <name>_chunks/ directory",
    )
    args = p.parse_args()
    clip = args.ref_audio is not None
    ref_text = " ".join((args.ref_text or "").split())
    if clip != bool(ref_text):
        p.error("--ref-audio and --ref-text must be given together")
    if clip and not args.ref_audio.is_file():
        p.error(f"reference audio not found: {args.ref_audio}")
    if clip:
        instruction = (args.instruction or "").strip()  # empty: plain clone
    else:
        instruction = (
            DEFAULT_INSTRUCTION if args.instruction is None else args.instruction
        ).strip()
        if not instruction:
            p.error("--instruction must not be empty")
    if not 0 <= args.seed < 2**32:
        p.error("--seed must be between 0 and 4294967295")

    paragraphs = [
        " ".join(block.split())
        for block in args.text_file.read_text().split("\n\n")
        if block.strip()
    ]
    chunk_dir = None
    if args.keep_chunks:
        chunk_dir = args.out_wav.parent / f"{args.out_wav.stem}_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model, audio_tokenizer = load_runtime(
        args.model, device=resolve_device(), attn_implementation="eager"
    )
    update_generation_config_for_breeze(model)
    stages = (
        {
            "fast_text_encoder": True,
            "fast_backbone_decode": True,
            "fast_depth_decoder": True,
            "fast_codec": True,
        }
        if args.fast
        else {}
    )
    runtime = FastBreezeStreamingRuntime(
        model,
        audio_tokenizer,
        FastStreamingConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            max_seq_len=2048,
            repetition_penalty=1.1,
            **stages,
        ),
        tokenizer=tokenizer,
    )
    warm_s = 0.0
    if runtime.fast_enabled:
        t0 = time.perf_counter()
        profile = load_warmup_profile(FAST_CONFIG)
        profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
        runtime.warmup_from_profile(profile)
        torch.cuda.synchronize()
        warm_s = time.perf_counter() - t0
    mode = "fast (all stages except prefill)" if args.fast else "eager"
    if not clip:
        voice = "designed"
    elif instruction:
        voice = f"clip {args.ref_audio.name} with direction"
    else:
        voice = f"clip {args.ref_audio.name}"
    print(
        f"mode: {mode}, warmup {warm_s:.1f}s, {len(paragraphs)} paragraphs, "
        f"seed {args.seed}, voice {voice}"
    )

    sr = runtime.sample_rate
    pieces: list[np.ndarray] = []
    rows = []
    # Paragraph 1's audio is the clone reference for the rest; it only needs to
    # live for the duration of the run, so keep it in a self-deleting temp dir.
    with tempfile.TemporaryDirectory(prefix="longform-") as tmp:
        ref_path = Path(tmp) / "ref.wav"
        for i, text in enumerate(paragraphs):
            if clip:
                request = {
                    "id": f"p{i}",
                    "text": text,
                    "speaker": "S0",
                    "ref_audio_path": str(args.ref_audio),
                    "ref_text": ref_text,
                }
                if instruction:
                    request["instruction"] = instruction
                cfg = DIRECTION_CFG if instruction else 1.0
            elif i == 0:
                request = {
                    "id": f"p{i}",
                    "text": text,
                    "speaker": "S0",
                    "instruction": instruction,
                }
                cfg = 4.0
            else:
                request = {
                    "id": f"p{i}",
                    "text": text,
                    "speaker": "S0",
                    "ref_audio_path": str(ref_path),
                    "ref_text": paragraphs[0],
                }
                cfg = 1.0
            template = select_template_name(request)
            n_tokens = len(tokenizer(text)["input_ids"])

            set_all_seeds(args.seed)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            inputs = prepare_inputs(
                tokenizer,
                audio_tokenizer,
                model,
                [request],
                get_template(template),
                guidance_scale=cfg,
                guidance_scale_ref=None,
                guidance_scale_ins=None,
            )
            ttfa = None
            audio = []
            for chunk in runtime.iter_audio_chunks(
                inputs, request_id=f"p{i}", seed=args.seed
            ):
                if ttfa is None:
                    torch.cuda.synchronize()
                    ttfa = time.perf_counter() - t0
                audio.append(chunk.audio)
            torch.cuda.synchronize()
            gen = time.perf_counter() - t0

            wav = np.concatenate(audio)
            if i == 0 and not clip:
                sf.write(ref_path, wav, sr, subtype="PCM_16")
            if chunk_dir is not None:
                sf.write(chunk_dir / f"{i:02d}.wav", wav, sr, subtype="PCM_16")
            dur = len(wav) / sr
            frames = round(dur * FRAME_RATE)
            words = len(text.split())
            rows.append((i, template, cfg, words, n_tokens, dur, frames, ttfa, gen))
            flag = "  <-- HIT FRAME CAP" if frames >= MAX_NEW_TOKENS - 1 else ""
            print(
                f"p{i}: {template:16s} cfg={cfg:.0f} words={words:3d} "
                f"tok={n_tokens:3d} audio={dur:6.2f}s ({words / dur * 60:4.0f} wpm) "
                f"frames={frames:4d} ttfa={ttfa * 1000:4.0f}ms gen={gen:5.2f}s "
                f"RTF={gen / dur:.3f}{flag}"
            )
            pieces.append(wav)
            pieces.append(np.zeros(int(GAP_S * sr), dtype=wav.dtype))

    full = np.concatenate(pieces[:-1])
    sf.write(args.out_wav, full, sr, subtype="PCM_16")
    total_gen = sum(r[8] for r in rows)
    total_audio = sum(r[5] for r in rows)
    print(
        f"TOTAL: {total_audio:.1f}s speech in {total_gen:.1f}s generation "
        f"(RTF {total_gen / total_audio:.3f}, {total_audio / total_gen:.1f}x real time); "
        f"file {len(full) / sr:.1f}s -> {args.out_wav}"
    )
    if chunk_dir is not None:
        print(f"per-paragraph WAVs kept in {chunk_dir}")
    print(f"peak GPU reserved: {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
