"""CPU dry run of every GPU code path the Nebius runner uses, on a tiny model.

Builds a randomly initialised Qwen2.5-Omni with 2-layer, 64-wide towers but
the REAL tokenizer and processor, then pushes the exact commands the runner
issues through it: bf16 loading, LoRA on the language model and the audio
encoder, the transcription target through the real collator, adapter save,
reload through the inference backend, generation, the timeline grounder, and
both time-symbol configurations (full rows and delta rows with BPE-initialised
head rows and the none weight). Takes a few minutes on a laptop; needs network
once for the processor files (~10 MB).

    python scripts/dry_run_gpu_paths.py [--keep DIR]

Anything that fails here would have failed on the paid machine after the data
generation. It does not check numerical quality: the model is random.

Found so far by running it: the Omni processor needs torchvision; qwen-omni-utils
needs audioread; transformers 5.2 removed warmup_ratio; without CUDA the Trainer
and the loader pick Apple's MPS backend, where bf16 crashes (all now guarded).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REAL = "Qwen/Qwen2.5-Omni-7B"


def sh(label, cmd, cwd):
    print(f"\n=== {label}\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        raise SystemExit(f"*** {label} FAILED (rc={r.returncode})")


def tiny_model(out: Path):
    """The REAL config shrunk in place, not a config built from the constructor:
    the checkpoint's config.json carries token-id fields (vision_start_token_id
    and friends) that the constructor does not list and the forward pass reads.
    Shrinking keeps every such field and changes only the sizes."""
    import torch
    from transformers import Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    proc = Qwen2_5OmniProcessor.from_pretrained(REAL)
    cfg = Qwen2_5OmniConfig.from_pretrained(REAL)
    h = 64
    th = cfg.thinker_config
    for k, v in dict(d_model=h, encoder_layers=2, encoder_attention_heads=4, encoder_ffn_dim=128, output_dim=h).items():
        setattr(th.audio_config, k, v)
    for k, v in dict(depth=2, hidden_size=h, intermediate_size=128, num_heads=4, out_hidden_size=h,
                     fullatt_block_indexes=[1]).items():
        setattr(th.vision_config, k, v)
    for k, v in dict(hidden_size=h, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, max_window_layers=2, layer_types=["full_attention"] * 2).items():
        setattr(th.text_config, k, v)
    # multimodal RoPE splits the rotary dims into three sections that must sum
    # to head_dim / 2: the real [16, 24, 24] is for head_dim 128; ours is 16
    rp = dict(getattr(th.text_config, "rope_parameters", None) or getattr(th.text_config, "rope_scaling", None) or {})
    rp["mrope_section"] = [2, 3, 3]
    th.text_config.rope_parameters = rp
    cfg.enable_audio_output = False
    torch.manual_seed(0)
    model = Qwen2_5OmniForConditionalGeneration(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"[tiny] {n/1e6:.1f}M parameters (the real one is 8,400M)")
    model.save_pretrained(out)
    proc.save_pretrained(out)
    # the full-model loader looks for the talker's speaker dictionary even with
    # audio output disabled; the real checkpoint ships one, ours is empty
    torch.save({}, out / "spk_dict.pt")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=None, help="work directory to keep (default: temp, deleted)")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="ctag_dry_"))
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model_dir = tiny_model(work / "model")
    py = [sys.executable, "-m"]
    bench = work / "proc"

    # data: the same commands the runner issues, on the procedural bank
    sh("benchmark", py + ["ctag.build_benchmark", "--source", "procedural", "--n-clips", "6", "--out", str(bench)], root)
    sh("split", py + ["ctag.split", "--bench", f"{bench}/benchmark.jsonl", "--out", str(bench), "--frac-train", "0.5", "--frac-val", "0.25"], root)
    sh("gen (hard, queries)", py + ["ctag.gen_train", "--source", "procedural", "--n-clips", "4", "--hard", "--queries",
                                    "--workers", "2", "--out", f"{work}/gen"], root)
    sh("sft transcribe (gen)", py + ["ctag.sft_data", "--task", "transcribe", "--timelines", f"{work}/gen/timelines.jsonl",
                                     "--out", f"{work}/gen/sft_transcribe.jsonl"], root)
    sh("sft transcribe (val)", py + ["ctag.sft_data", "--task", "transcribe", "--timelines", f"{bench}/timelines.jsonl",
                                     "--bench", f"{bench}/benchmark_val.jsonl", "--out", f"{bench}/sft_transcribe_val.jsonl"], root)
    for tag, extra in (("text", []), ("tt", ["--time-tokens"])):
        sh(f"sft queries ({tag})", py + ["ctag.sft_data", "--task", "queries", "--timelines", f"{work}/gen/timelines.jsonl",
                                          "--bench", f"{work}/gen/benchmark.jsonl", "--plain-ratio", "0.6", "--max-empty-share", "0.15",
                                          "--max-examples", "12"] + extra + ["--out", f"{work}/gen/sft_q_{tag}.jsonl"], root)

    train = py + ["ctag.train_lora", "--model-id", str(model_dir), "--precision", "bf16", "--amp", "none",
                  "--batch-size", "1", "--grad-accum", "1", "--epochs", "1", "--max-steps", "2", "--optim", "adamw_torch",
                  "--lora-r", "4", "--lora-alpha", "8", "--max-seq-len", "4096"]

    # 1. transcription adapter, encoder trained, bf16 -> reload through the backend -> generate -> grounder
    sh("train transcription (bf16, encoder)", train + ["--train-encoder", "--data", f"{work}/gen/sft_transcribe.jsonl",
                                                       "--val", f"{bench}/sft_transcribe_val.jsonl", "--out", f"{work}/lora_transcribe"], root)
    sh("transcribe with the adapter", py + ["ctag.run_transcribe", "--model", "qwen2.5-omni", "--model-id", str(model_dir),
                                            "--adapter", f"{work}/lora_transcribe", "--precision", "bf16", "--max-new-tokens", "24",
                                            "--stop-probs",
                                            "--bench", f"{bench}/benchmark_test.jsonl", "--timelines", f"{bench}/timelines.jsonl",
                                            "--out", f"{work}/runs/transcribe_test"], root)
    first = json.loads(open(f"{work}/runs/transcribe_test/pred_timelines.jsonl").readline())
    assert "stop_probs" in first and isinstance(first["stop_probs"], list), first.keys()
    assert all(p is None or 0.0 <= p <= 1.0 for p in first["stop_probs"]), first["stop_probs"]
    sh("trailing check on the stop probabilities", py + ["ctag.trailing", "--pred-timelines", f"{work}/runs/transcribe_test/pred_timelines.jsonl",
                                                          "--timelines", f"{bench}/timelines.jsonl", "--rule", "stop", "--sweep", "0.1", "0.5"], root)
    sh("queries from the timelines", py + ["ctag.run_agent", "--grounder", "timeline", "--pred-timelines", f"{work}/runs/transcribe_test/pred_timelines.jsonl",
                                           "--bench", f"{bench}/benchmark_test.jsonl", "--out", f"{work}/runs/test_from_timeline"], root)
    sh("direct prompting with the adapter", py + ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--model-id", str(model_dir),
                                                  "--adapter", f"{work}/lora_transcribe", "--precision", "bf16", "--max-new-tokens", "16",
                                                  "--n", "4", "--bench", f"{bench}/benchmark_test.jsonl", "--out", f"{work}/runs/test_direct"], root)

    # 2. arm C at scale (text digits) -- same path as 1 without the encoder
    sh("train text digits", train + ["--data", f"{work}/gen/sft_q_text.jsonl", "--out", f"{work}/lora_q_text"], root)

    # 3. arm E, both row modes, with the retest flags
    for rows in ("full", "delta"):
        sh(f"train time symbols ({rows} rows)", train + ["--data", f"{work}/gen/sft_q_tt.jsonl", "--time-tokens", "--head-init", "bpe",
                                                          "--none-weight", "0.3", "--time-rows", rows, "--out", f"{work}/lora_q_tt_{rows}"], root)
        for f in ("adapter_config.json", "time_rows.json", "tokenizer_config.json"):
            assert (work / f"lora_q_tt_{rows}" / f).exists(), f"{rows}: {f} missing"
        assert (work / f"lora_q_tt_{rows}" / "time_deltas.pt").exists() == (rows == "delta"), rows
        sh(f"eval time symbols ({rows} rows)", py + ["ctag.run_zeroshot", "--model", "qwen2.5-omni", "--model-id", str(model_dir),
                                                     "--adapter", f"{work}/lora_q_tt_{rows}", "--precision", "bf16", "--max-new-tokens", "8",
                                                     "--n", "4", "--bench", f"{bench}/benchmark_test.jsonl", "--out", f"{work}/runs/test_q_tt_{rows}"], root)
        s = json.load(open(work / f"runs/test_q_tt_{rows}/summary.json"))
        assert "ALL" in s["by_type"], s

    # 4. preemption: a checkpoint every step, the run cut after 2 of 4 steps
    #    (the done marker removed as a kill would leave it), then the same
    #    command resumes from checkpoint-2. Plain LoRA and delta rows, whose
    #    deltas live outside the adapter and must travel with the checkpoint.
    for tag, extra in (("lora", ["--data", f"{work}/gen/sft_transcribe.jsonl", "--train-encoder"]),
                       ("delta", ["--data", f"{work}/gen/sft_q_tt.jsonl", "--time-tokens", "--head-init", "bpe",
                                  "--none-weight", "0.3", "--time-rows", "delta"])):
        out = work / f"lora_resume_{tag}"
        sh(f"resume: first 2 steps ({tag})", train + extra + ["--save-steps", "1", "--max-steps", "2", "--out", str(out)], root)
        assert (out / "checkpoint-2").is_dir(), "no checkpoint-2"
        assert len([c for c in out.glob("checkpoint-*")]) <= 2, "save_total_limit not applied"
        if tag == "delta":
            assert (out / "checkpoint-2" / "time_deltas.pt").exists(), "deltas missing from the checkpoint"
        (out / "train_done.json").unlink()
        sh(f"resume: continue to 4 steps ({tag})", train + extra + ["--save-steps", "1", "--max-steps", "4", "--out", str(out)], root)
        done = json.load(open(out / "train_done.json"))
        assert done["resumed_from"] == "checkpoint-2" and done["steps"] == 4, done

    print("\n*** every GPU code path ran end to end on the tiny model ***")
    print(f"work dir: {work}" + ("" if a.keep else "  (temporary)"))


if __name__ == "__main__":
    main()
