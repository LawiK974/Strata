"""Benchmark a Strata server that is already running; never starts or stops anything.

    python bench/toks-tokenizer/http_bench.py --url http://127.0.0.1:5000 \
        --api-key-file /data/strata-data/state/api-key --label baseline \
        --out bench/toks-tokenizer/baseline.json

Sends the same fixed prompts, temperature 0, fixed seed and fixed max_tokens, and keeps the server's own
timings (llama.cpp's prompt/decode rates) next to the client's wall time. The wall time minus the server's
timings is the part the server does not measure: HTTP, the template and tokenizing -- the part this
experiment is about. Prompt cache makes a repeated prompt faster, so each prompt is sent twice and both
passes are kept apart.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

WORDS = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
         "ledger quartz meadow anchor violet thunder").split()


def prompt(words: int) -> str:
    import random
    rng = random.Random(7)
    body = " ".join(rng.choice(WORDS) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


def ask(url: str, key: str, messages: list[dict], max_tokens: int, timeout: float) -> tuple[dict, float, float]:
    """Streamed request: returns the final chunk (usage + timings), wall time and time to first token."""
    body = {"model": "strata", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "top_k": 1, "top_p": 1, "min_p": 0, "seed": 42, "reasoning_effort": "none", "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    start = time.monotonic()
    ttft = None
    last = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[len(b"data: "):].strip()
            if payload == b"[DONE]":
                continue
            chunk = json.loads(payload)
            if chunk.get("usage") or chunk.get("timings"):
                last = chunk
            delta = [c.get("delta") or {} for c in chunk.get("choices", [])]
            if ttft is None and any(d.get("content") or d.get("reasoning_content") for d in delta):
                ttft = (time.monotonic() - start) * 1000.0
    return last, (time.monotonic() - start) * 1000.0, ttft or float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lengths", default="40,400,3000", help="words per prompt")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=600.0)
    a = ap.parse_args()
    key = Path(a.api_key_file).read_text().strip() if a.api_key_file else ""

    req = urllib.request.Request(a.url.rstrip("/") + "/v1/models",
                                 headers={"Authorization": "Bearer " + key} if key else {})
    with urllib.request.urlopen(req, timeout=10) as r:
        model = json.loads(r.read())["data"][0]["id"]
    runs = []
    for words in [int(x) for x in a.lengths.split(",")]:
        text = prompt(words)
        for i in range(a.repeat):
            data, wall, ttft = ask(a.url, key, [{"role": "user", "content": text}], a.max_tokens, a.timeout)
            t = data.get("timings", {})
            u = data.get("usage", {})
            runs.append({"prompt_words": words, "pass": i + 1, "wall_ms": round(wall, 1),
                         "ttft_ms": round(ttft, 1),
                         "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": u.get("prompt_tokens_details", {}).get("cached_tokens"),
                         "completion_tokens": u.get("completion_tokens"),
                         "prompt_ms": t.get("prompt_ms"), "prefill_tps": t.get("prompt_per_second"),
                         "decode_ms": t.get("predicted_ms"), "decode_tps": t.get("predicted_per_second"),
                         "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted"),
                         "finish_reason": data["choices"][0].get("finish_reason") if data.get("choices") else None})
            print(json.dumps(runs[-1]))

    med = lambda key: statistics.median(r[key] for r in runs if r.get(key))  # noqa: E731
    summary = {"label": a.label, "model": model, "url": a.url, "n_ctx": None, "runs": runs,
               "median_ttft_ms": round(med("ttft_ms"), 1),
               "median_prefill_tps": round(med("prefill_tps"), 1),
               "median_decode_tps": round(med("decode_tps"), 1),
               "median_wall_ms": round(med("wall_ms"), 1)}
    a.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
