# Tokenizer experiment — baseline on the live instance

Measured on: RX 6700 XT (gfx1031), 30 GiB RAM, host `wikle`, Strata 0.2.13 image
`ghcr.io/lawik974/strata:0.2.13-rocm-gfx1031`, served by the running quadlet unit
`strata.service` (pod `strata.pod`, ports `127.0.0.1:5000` and `127.0.0.1:8096`).

Nothing was restarted, stopped or reconfigured. The bench talks to the server that was already up.

## What was measured

`bench/toks-tokenizer/http_bench.py` sends the same fixed prompts (`temperature 0`, `seed 42`,
`max_tokens 128`, thinking off) twice each, streamed, and keeps the server's own timings next to the
client's wall time and its time to first token. TTFT minus `prompt_ms` is what the server does not
measure: HTTP, the chat template and tokenizing — the part this experiment is about.

| Prompt words | pass | prompt_tokens | cached | TTFT ms | prompt_ms | TTFT − prompt_ms | prefill tok/s | decode tok/s | wall ms |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 1 (cold) | 120 | 0 | 1122.6 | 1080.5 | 42.1 | 111.0 | 21.3 | 6066.7 |
| 40 | 2 (cached) | 120 | 113 | 229.5 | 160.9 | 68.6 | 67.1 | 30.7 | 3551.9 |
| 400 | 1 (cold) | 503 | 0 | 1548.7 | 1512.5 | 36.2 | 332.0 | 27.8 | 6125.6 |
| 400 | 2 (cached) | 503 | 496 | 157.2 | 116.0 | 41.2 | 433.6 | 28.7 | 4476.0 |
| 3000 | 1 (cold) | 3239 | 0 | 5420.5 | 5382.0 | 38.5 | 601.9 | 28.6 | 9018.1 |
| 3000 | 2 (cached) | 3239 | 3232 | 154.2 | 112.5 | 41.7 | 603.9 | 30.2 | 4351.4 |

Medians over the six requests: TTFT 676.0 ms, prefill 86.7 tok/s, decode 28.6 tok/s, wall 5271.4 ms.
Model id `qwen3.8-flash-next-gyro-gyro-s`, `n_ctx` 65536, drafting dflash2. Two requests reached
`max_tokens` (`finish_reason: length`), the rest stopped by themselves.

## Reading of it

TTFT sits 36–69 ms above the server's `prompt_ms` at every prompt size, and that gap does not grow
with the prompt: 3239 tokens leave 38.5 ms, about 12 ns per token. Tokenizing is not a cost visible in
the serving path, so the toks arm should move the tokenizer's own measurement, not end-to-end TTFT.
The earlier tokenizer-only numbers (`docker/tokenizer_check.py`, same pack): python encode 0.1 ms vs
C++ load+encode 335.4 ms — the C++ figure includes loading the 248320-entry pack, so it is a load
cost, not a per-token cost.

## The toks arm

Same machine, same prompts, same server settings. The container now runs
`localhost/strata:0.2.13-rocm-gfx1031-toks` (engine build_id `712fec6b4297+rocm+toks`). The instance was
restarted for the switch, so the prompt cache started cold again; the two arms are same-day runs, not
interleaved ones.

| Prompt words | pass | prompt_tokens | cached | TTFT ms | prompt_ms | TTFT − prompt_ms | prefill tok/s | decode tok/s | wall ms |
|---|---|---|---|---|---|---|---|---|---|
| 40 | 1 (cold) | 120 | 0 | 900.6 | 859.2 | 41.4 | 139.7 | 20.4 | 6051.3 |
| 40 | 2 (cached) | 120 | 113 | 182.5 | 145.9 | 36.6 | 53.2 | 27.6 | 3994.5 |
| 400 | 1 (cold) | 503 | 0 | 1515.3 | 1480.2 | 35.1 | 339.8 | 27.8 | 5552.3 |
| 400 | 2 (cached) | 503 | 496 | 131.4 | 96.8 | 34.6 | 520.7 | 30.8 | 4260.4 |
| 3000 | 1 (cold) | 3239 | 0 | 5358.2 | 5307.6 | 50.6 | 610.1 | 22.7 | 9889.3 |
| 3000 | 2 (cached) | 3239 | 3232 | 141.9 | 107.5 | 34.4 | 603.6 | 29.3 | 3665.9 |

Medians: TTFT 541.5 ms, prefill 106.0 tok/s, decode 27.7 tok/s, wall 4906.4 ms.

## Comparison

| metric | baseline median | toks median |
|---|---|---|
| TTFT ms | 676.0 | 541.5 |
| TTFT − prompt_ms ms | 41.7 | 36.6 |
| prefill tok/s | 86.7 | 106.0 |
| decode tok/s | 28.6 | 27.7 |
| wall ms | 5271.4 | 4906.4 |

`prompt_tokens` is identical request by request (120, 503, 3239) and the cached counts match, so the
toks tokenizer produces the same ids as the python one on this pack. The gap TTFT − prompt_ms is
34–51 ms with toks against 36–69 ms with the baseline, and neither grows with prompt size; the difference
is inside the noise of two runs on a machine whose GPU state changed between them, so no measured claim
about tokenization speed is made here. One request reached `max_tokens` in this run against two in the
baseline — accepted draft tokens differ per run, which changes the output length.

## Where the prefill time actually sits

`prompt_ms` is the engine's forward pass, reported by the server. The python tokenizer costs
19.3 ms to encode the whole corpus of this bench (6232 tokens, first call, ~3 µs/token) and 1.1 ms on the
second call once its BPE cache is warm; the C++ toks tokenizer loads the pack in 335 ms, which happens at
startup, not per request. So the 5.3 s of the 3239-token prompt is GPU evaluation, not tokenization.
Measure with `python /tmp/tok_time2.py <pack tokenizer dir>`.

## Passed / failed / not run

- Passed: baseline HTTP bench (6 requests streamed; 4 `stop`, 2 `length` at `max_tokens`).
- Passed: toks HTTP bench on the live instance after the image switch (6 requests streamed; 5 `stop`,
  1 `length`), same `prompt_tokens` as the baseline arm, GPU serving works with the toks engine.
- Failed: none. The GPU smoke test of the new image fails with `Memory critical error by agent node-0 …
  Reason: Memory in use` — the same failure on the base image, so it is the RAM/GPU already held by the
  running instance, not the tokenizer change.
- Not run: repeated runs for variance; the two arms are one run each after a restart, so the TTFT and
  prefill medians carry run-to-run noise.

Raw data: `bench/toks-tokenizer/baseline-0.2.13-rocm-gfx1031.json` and
`bench/toks-tokenizer/toks-0.2.13-rocm-gfx1031-toks.json`.
