#!/usr/bin/env python3
"""Stream the right channel of an asr-data DB and report realtime CER.

Each Audio opens an independent realtime WebSocket session. Its URL source is
decoded through asr-data, channel index 1 is selected, resampled to 16 kHz,
converted to PCM16, and sent as ``input_audio_buffer.append`` events. Completed
transcriptions are concatenated and compared with the Audio's right timeline.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import websockets
from asr_data import Audio, AudioDB


FASR_ROOT = Path(__file__).resolve().parents[2] / "fasr"
if str(FASR_ROOT) not in sys.path:
    sys.path.insert(0, str(FASR_ROOT))
from fasr.utils.metric import compute_char_error_rate  # noqa: E402


DEFAULT_DB = Path("/workspace/code/vASR/房产客服电话.vasr")
DEFAULT_WS_URL = "wss://lbggateway.58corp.com/asr/v1/realtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an asr-data right timeline through realtime ASR."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_DB)
    parser.add_argument("--url", default=DEFAULT_WS_URL)
    parser.add_argument("--output", type=Path, default=Path("results/realtime_right.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of independent realtime sessions. Default: 4",
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=1_000,
        help="PCM duration per WebSocket append; the service still frames it internally.",
    )
    parser.add_argument(
        "--send-interval-ms",
        type=float,
        default=0.0,
        help="Delay between appends. Use chunk-ms for real-time pacing; default sends fast.",
    )
    parser.add_argument("--finish-timeout", type=float, default=120.0)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--hotwords-id", default="fangchan")
    parser.add_argument("--rule-id", default="fangchan")
    parser.add_argument("--vad-threshold", type=float, default=0.25)
    parser.add_argument("--vad-silence-duration-ms", type=int, default=800)
    parser.add_argument("--vad-prefix-padding-ms", type=int, default=400)
    parser.add_argument("--enhancement-strength", type=float, default=1.0)
    parser.add_argument("--show-events", action="store_true")
    return parser.parse_args()


def session_update(args: argparse.Namespace) -> dict[str, Any]:
    # The top-level fields are understood by vASR. The nested session object is
    # compatible with the fasr gateway used by the reference client.
    session: dict[str, Any] = {
        "audio": {"sample_rate": args.sample_rate, "format": "pcm"},
        "vad": {
            "silence_duration_ms": args.vad_silence_duration_ms,
            "threshold": args.vad_threshold,
            "prefix_padding_ms": args.vad_prefix_padding_ms,
        },
        "asr": {"language": args.language},
        "enhance": {"enhancement_strength": args.enhancement_strength},
    }
    if args.hotwords_id:
        session["asr"]["hotwords_id"] = args.hotwords_id
    if args.rule_id:
        session["correct"] = {"rule_id": args.rule_id}
    return {
        "type": "session.update",
        "sample_rate": args.sample_rate,
        "input_audio_format": "pcm16",
        "session": session,
    }


async def send_json(ws: Any, event: dict[str, Any]) -> None:
    await ws.send(json.dumps(event, ensure_ascii=False))


def pcm16_right(audio: Audio, sample_rate: int) -> tuple[bytes, float]:
    waveform = audio.load()
    if waveform.channels < 2:
        raise ValueError(f"expected stereo source, got {waveform.channels} channel(s)")
    right = waveform.channel(1)
    if right.sample_rate != sample_rate:
        right = right.resample(sample_rate)
    samples = np.asarray(right.numpy(), dtype=np.float32)
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    return pcm.tobytes(), len(pcm) / sample_rate


def reference_right(audio: Audio) -> str:
    timeline = audio.timeline("right")
    if timeline is None:
        raise ValueError("audio has no right timeline")
    return timeline.transcript().text


async def receive_events(
    ws: Any,
    completed: list[str],
    errors: list[str],
    finished: asyncio.Event,
    show_events: bool,
) -> None:
    try:
        async for raw in ws:
            event = json.loads(raw)
            if show_events:
                printable = dict(event)
                if isinstance(printable.get("audio"), str):
                    printable["audio"] = f"<base64 {len(printable['audio'])} chars>"
                print("<<", json.dumps(printable, ensure_ascii=False))
            event_type = event.get("type")
            if event_type == "conversation.item.input_audio_transcription.completed":
                text = str(event.get("text") or "")
                if text:
                    completed.append(text)
            elif event_type == "error":
                errors.append(str(event.get("message") or event))
            elif event_type == "session.finished":
                finished.set()
                return
    except websockets.ConnectionClosed as error:
        if not finished.is_set():
            errors.append(f"connection closed before session.finished: {error}")
        finished.set()


async def transcribe_one(
    audio: Audio, args: argparse.Namespace, index: int, total: int
) -> dict[str, Any]:
    started = time.perf_counter()
    reference = reference_right(audio)
    pcm, duration_s = await asyncio.to_thread(pcm16_right, audio, args.sample_rate)
    completed: list[str] = []
    errors: list[str] = []
    finished = asyncio.Event()
    chunk_bytes = max(2, args.sample_rate * 2 * args.chunk_ms // 1_000)
    chunk_bytes -= chunk_bytes % 2

    async with websockets.connect(
        args.url,
        max_size=None,
        ping_interval=20,
        ping_timeout=60,
        open_timeout=30,
    ) as ws:
        first = json.loads(await ws.recv())
        if first.get("type") == "error":
            raise RuntimeError(first.get("message") or first)
        if args.show_events:
            print("<<", json.dumps(first, ensure_ascii=False))
        await send_json(ws, session_update(args))
        receiver = asyncio.create_task(
            receive_events(ws, completed, errors, finished, args.show_events)
        )
        for offset in range(0, len(pcm), chunk_bytes):
            chunk = pcm[offset : offset + chunk_bytes]
            await send_json(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                },
            )
            if args.send_interval_ms > 0:
                await asyncio.sleep(args.send_interval_ms / 1_000)
        await send_json(ws, {"type": "session.finish"})
        try:
            await asyncio.wait_for(finished.wait(), timeout=args.finish_timeout)
        except TimeoutError:
            errors.append(f"session.finished timeout after {args.finish_timeout}s")
        if not receiver.done():
            receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)

    hypothesis = "".join(completed)
    # fasr's Chinese Normalizer asserts on an empty input. Empty text needs no
    # normalization; compute_char_error_rate itself defines empty-vs-empty as
    # 0 and empty-vs-nonempty as 1.
    cer = compute_char_error_rate(
        x=hypothesis,
        y=reference,
        ignore_space=True,
        ignore_punc=True,
        normalize=bool(hypothesis and reference),
    )
    reference_cer_text = cer_text(reference)
    hypothesis_cer_text = cer_text(hypothesis)
    ref_chars = len(reference_cer_text)
    elapsed_s = time.perf_counter() - started
    print(
        f"[{index}/{total}] {audio.id} duration={duration_s:.1f}s "
        f"elapsed={elapsed_s:.1f}s cer={cer:.4f} segments={len(completed)}"
    )
    return {
        "audio_id": audio.id,
        "duration_s": round(duration_s, 4),
        "elapsed_s": round(elapsed_s, 4),
        "reference": reference,
        "hypothesis": hypothesis,
        "reference_cer_text": reference_cer_text,
        "hypothesis_cer_text": hypothesis_cer_text,
        "reference_chars": ref_chars,
        "cer": round(cer, 6),
        "completed_segments": completed,
        "errors": errors,
    }


def cer_text(text: str) -> str:
    """Mirror fasr's ignore-space and ignore-punctuation denominator text."""
    return re.sub(r"[^\w\s]", "", text.replace(" ", ""))


def summary(results: list[dict[str, Any]], failures: list[dict[str, str]]) -> dict[str, Any]:
    total_reference = sum(item["reference_chars"] for item in results)
    total_audio = sum(item["duration_s"] for item in results)
    total_elapsed = sum(item["elapsed_s"] for item in results)
    weighted_cer = (
        sum(item["cer"] * item["reference_chars"] for item in results)
        / total_reference
        if total_reference
        else 0.0
    )
    return {
        "items": len(results),
        "failed": len(failures),
        "audio_seconds": round(total_audio, 4),
        "elapsed_seconds": round(total_elapsed, 4),
        "speedup": round(total_audio / total_elapsed, 4) if total_elapsed else 0.0,
        "cer_weighted": round(weighted_cer, 6),
        "cer_average": round(
            sum(item["cer"] for item in results) / len(results), 6
        )
        if results
        else 0.0,
        "reference_chars": total_reference,
    }


def save_report(
    path: Path,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    failures: list[dict[str, str]],
    wall_seconds: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    report_summary = summary(results, failures)
    if wall_seconds is not None:
        report_summary["wall_seconds"] = round(wall_seconds, 4)
    report = {
        "input": str(args.input),
        "service_url": args.url,
        "timeline": "right",
        "sample_rate": args.sample_rate,
        "chunk_ms": args.chunk_ms,
        "cer_implementation": "fasr.utils.metric.compute_char_error_rate",
        "normalize": True,
        "ignore_space": True,
        "ignore_punc": True,
        "summary": report_summary,
        "items": results,
        "failures": failures,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


async def run(args: argparse.Namespace) -> None:
    if (
        args.chunk_ms <= 0
        or args.sample_rate <= 0
        or args.concurrency <= 0
        or args.retries <= 0
    ):
        raise ValueError(
            "--chunk-ms, --sample-rate, --concurrency, and --retries must be positive"
        )
    db = AudioDB(str(args.input), read_only=True)
    audios = list(db)
    if args.limit is not None:
        audios = audios[: args.limit]
    result_slots: list[dict[str, Any] | None] = [None] * len(audios)
    failures: list[dict[str, str]] = []
    print(
        f"realtime eval START input={args.input} items={len(audios)} "
        f"timeline=right service={args.url}"
    )
    started = time.perf_counter()
    semaphore = asyncio.Semaphore(args.concurrency)
    report_lock = asyncio.Lock()

    async def evaluate(index: int, audio: Audio) -> None:
        async with semaphore:
            last_error: Exception | None = None
            for attempt in range(1, args.retries + 1):
                try:
                    result_slots[index - 1] = await transcribe_one(
                        audio, args, index, len(audios)
                    )
                    break
                except Exception as error:
                    last_error = error
                    message = str(error) or type(error).__name__
                    if attempt < args.retries:
                        delay = min(20, attempt * 2)
                        print(
                            f"[{index}/{len(audios)}] {audio.id} retry "
                            f"{attempt}/{args.retries} in {delay}s: {message}"
                        )
                        await asyncio.sleep(delay)
            if result_slots[index - 1] is None:
                message = str(last_error) or type(last_error).__name__
                failures.append({"audio_id": audio.id, "error": message})
                print(f"[{index}/{len(audios)}] {audio.id} FAILED: {message}")
        async with report_lock:
            completed_results = [item for item in result_slots if item is not None]
            save_report(args.output, args, completed_results, failures)

    await asyncio.gather(
        *(evaluate(index, audio) for index, audio in enumerate(audios, 1))
    )

    wall_seconds = time.perf_counter() - started
    results = [item for item in result_slots if item is not None]
    final = summary(results, failures)
    final["wall_seconds"] = round(wall_seconds, 4)
    save_report(args.output, args, results, failures, wall_seconds)
    print("\nrealtime eval DONE")
    print(json.dumps(final, ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
