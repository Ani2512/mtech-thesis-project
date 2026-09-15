"""Model backends. Interface: ground(audio_path, query_text) -> raw text.

mock:oracle            returns the ground truth (harness sanity)
mock:ignore_condition  returns every occurrence of X, ignoring the condition
mock:first_only        returns the first occurrence of X only (TAG-Bench's dominant failure)
mock:jitter            oracle with +-0.4 s boundary noise
qwen2.5-omni           Qwen/Qwen2.5-Omni-7B thinker (text out)
qwen2-audio            Qwen/Qwen2-Audio-7B-Instruct
gemini                 Gemini via google-genai (GEMINI_API_KEY)
"""
from __future__ import annotations

import json
import os
import random

SYSTEM = (
    "You are an audio analysis assistant. You will hear one recording and receive a query "
    "describing which sound events to locate. Reply with ONLY a JSON list of [start, end] pairs "
    "in seconds, e.g. [[1.2, 2.0], [7.5, 8.1]]. If nothing in the recording satisfies the query, reply []."
)


def prompt_for(query_text: str, duration: float | None = None) -> str:
    d = f" The recording is {duration:.1f} seconds long." if duration else ""
    return f"Locate: {query_text}.{d} Reply with only the JSON list."


class MockBackend:
    def __init__(self, mode: str = "oracle", seed: int = 0):
        self.mode, self.rng = mode, random.Random(seed)
        self.name = f"mock:{mode}"

    def ground(self, audio_path, query_text, query=None, duration=None, temperature=None):
        assert query is not None, "mock backends need the Query object"
        if self.mode == "oracle":
            iv = query.answer
        elif self.mode == "ignore_condition":
            iv = query.plain_intervals
        elif self.mode == "first_only":
            iv = query.plain_intervals[:1]
        elif self.mode == "jitter":
            iv = [(max(0, a + self.rng.uniform(-0.4, 0.4)), b + self.rng.uniform(-0.4, 0.4)) for a, b in query.answer]
        else:
            raise KeyError(self.mode)
        return json.dumps([[round(a, 2), round(b, 2)] for a, b in iv])


def choose_precision(total_gib: float | None, param_billions: float = 8.4) -> str:
    """fp16 when the weights plus working headroom genuinely fit, else 4-bit.

    Qwen2-Audio-7B-Instruct and Qwen2.5-Omni-7B are ~8.4B parameters, so fp16
    weights alone are ~17 GB. A Colab T4 has ~15 GB: fp16 there silently spills
    to CPU through device_map and inference becomes unusably slow. We skip 8-bit
    in the automatic path because bitsandbytes LLM.int8() is slower than NF4
    4-bit on Turing and Ampere while using twice the memory; 8bit stays available
    as an explicit override.
    """
    if total_gib is None:
        return "cpu"
    return "fp16" if total_gib >= param_billions * 2 * 1.25 else "4bit"


def _fit_plan(param_billions: float = 8.4, override: str | None = None) -> tuple[str, dict]:
    """Returns (label, kwargs for from_pretrained)."""
    import torch

    if override in ("fp16", "bf16", "8bit", "4bit"):
        choice = override
    else:
        total = (torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                 if torch.cuda.is_available() else None)
        choice = choose_precision(total, param_billions)

    if choice == "cpu":
        return "cpu (no GPU visible - this will be very slow)", {"dtype": torch.float32}
    if choice == "fp16":
        return "fp16", {"dtype": torch.float16, "device_map": "auto"}
    if choice == "bf16":
        # Full-precision weights on a card with native bf16 (Ampere+, e.g. L40S):
        # no quantisation error, no dequantisation in every forward, bf16's
        # fp32 range so no overflow. The 7B thinker is ~17 GB this way.
        return "bf16", {"dtype": torch.bfloat16, "device_map": "auto"}

    from transformers import BitsAndBytesConfig

    if choice == "8bit":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16,
                                 bnb_4bit_use_double_quant=True)
    return choice, {"quantization_config": cfg, "device_map": "auto"}


class Qwen25OmniBackend:
    name = "qwen2.5-omni"

    def __init__(self, model_id="Qwen/Qwen2.5-Omni-7B", precision=None, max_new_tokens=96,
                 adapter=None):
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        label, kw = _fit_plan(8.4, precision)
        print(f"[qwen2.5-omni] loading in {label}" + (f" + adapter {adapter}" if adapter else ""))
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, enable_audio_output=False, **kw
        ).eval()
        if adapter:
            import os

            from peft import PeftModel

            # The adapter's own processor carries any timestamp tokens that were
            # added during training. Load it rather than the base one, or the
            # model emits tokens the tokenizer cannot decode.
            proc_src = adapter if os.path.exists(os.path.join(adapter, "tokenizer_config.json")) else model_id
            self.processor = Qwen2_5OmniProcessor.from_pretrained(proc_src)
            from .timetokens import DELTA_FILE, load_deltas

            n_tok = len(self.processor.tokenizer)
            cur = self.model.thinker.get_input_embeddings().weight.shape[0]
            has_deltas = os.path.exists(os.path.join(adapter, DELTA_FILE))
            # Qwen pads the matrix (152,064 rows for 151,665 tokens), so a plain
            # text adapter's tokenizer is *smaller* than the matrix. Shrinking to
            # it is pointless and untested; only a tokenizer that added tokens,
            # or a delta file whose row bookkeeping needs the exact size, gets a
            # resize.
            if has_deltas or n_tok > cur:
                print(f"[qwen2.5-omni] resizing embeddings {cur} -> {n_tok} for the adapter")
                self.model.thinker.resize_token_embeddings(n_tok)

            # Timestamp tokens are trained as a small delta on the new rows only
            # (see timetokens.wrap_new_rows), so they live beside the adapter
            # rather than inside it. Without this they stay at initialisation and
            # arm E silently measures nothing.
            had_deltas = load_deltas(self.model.thinker, adapter, self.processor.tokenizer)
            # --time-rows full trains embed_tokens and lm_head inside the adapter
            # (modules_to_save), so no delta file is the correct state there.
            full_rows = False
            marker = os.path.join(adapter, "time_rows.json")
            if os.path.exists(marker):
                import json as _json
                full_rows = _json.load(open(marker)).get("time_rows") == "full"
            if n_tok > cur and not had_deltas and not full_rows:
                raise RuntimeError(
                    "the adapter's tokenizer added tokens but no time_deltas.pt sits "
                    "next to it; the new embeddings would be random and every "
                    "prediction using them meaningless")

            # train_lora tunes the thinker, so the adapter attaches there, not to
            # the top-level wrapper whose module names it would not match.
            self.model.thinker = PeftModel.from_pretrained(self.model.thinker, adapter).eval()
            self.name = "qwen2.5-omni+lora"
        else:
            self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    def ground(self, audio_path, query_text, query=None, duration=None, temperature=None):
        from qwen_omni_utils import process_mm_info

        conv = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "audio", "audio": audio_path}, {"type": "text", "text": prompt_for(query_text, duration)}]},
        ]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=False)
        inputs = self.processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            # Call the thinker directly rather than the top-level generate().
            # Qwen2_5OmniForConditionalGeneration.generate builds the talker's
            # kwargs dict unconditionally -- it reads self.talker.codec_pad_token
            # before checking whether audio output was requested -- so with
            # enable_audio_output=False it raises
            #   AttributeError: 'Qwen2_5OmniForConditionalGeneration' object has no attribute 'talker'
            # The text path inside that method is exactly self.thinker.generate(...),
            # which is what we want and which skips the broken branch.
            gen = ({"do_sample": True, "temperature": temperature} if temperature
                   else {"do_sample": False})
            ids = self.model.thinker.generate(**inputs, max_new_tokens=self.max_new_tokens, **gen)
        ids = ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class Qwen2AudioBackend:
    name = "qwen2-audio"

    def __init__(self, model_id="Qwen/Qwen2-Audio-7B-Instruct", precision=None, max_new_tokens=96):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        label, kw = _fit_plan(8.4, precision)
        print(f"[qwen2-audio] loading in {label}")
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, **kw).eval()
        self.sr = self.processor.feature_extractor.sampling_rate

    def ground(self, audio_path, query_text, query=None, duration=None, temperature=None):
        import librosa

        audio, _ = librosa.load(audio_path, sr=self.sr)
        conv = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "audio", "audio_url": audio_path}, {"type": "text", "text": prompt_for(query_text, duration)}]},
        ]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, audio=[audio], sampling_rate=self.sr, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.no_grad():
            gen = ({"do_sample": True, "temperature": temperature} if temperature
                   else {"do_sample": False})
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, **gen)
        ids = ids[:, inputs.input_ids.size(1):]
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def af3_conversation(audio_path: str, query_text: str, duration: float | None = None) -> list[dict]:
    """Audio Flamingo 3's chat template takes one user turn with an audio part and a
    text part. There is no separate system role in the published template, so the
    system instruction is folded into the user text -- the same words the other
    backends see, just in one message."""
    return [{"role": "user", "content": [
        {"type": "audio", "path": audio_path},
        {"type": "text", "text": SYSTEM + "\n\n" + prompt_for(query_text, duration)},
    ]}]


class AudioFlamingo3Backend:
    """NVIDIA Audio Flamingo 3 (arXiv:2507.08128): AF-Whisper encoder, MLP adaptor,
    Qwen2.5-7B LLM. One of TAG-Bench's 21 evaluated systems, so scores here are
    directly comparable to theirs. Audio longer than 30 s is processed in 30 s
    windows; every clip in this benchmark is 20 s. Licence: NVIDIA OneWay
    Noncommercial (plus the Qwen Research License) -- research use only.
    """
    name = "audio-flamingo-3"

    def __init__(self, model_id="nvidia/audio-flamingo-3-hf", precision=None, max_new_tokens=96):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AudioFlamingo3ForConditionalGeneration
        except ImportError as e:  # pragma: no cover - runtime environment
            raise ImportError(
                "AudioFlamingo3ForConditionalGeneration is not in this transformers build; "
                "the native integration is recent. Try: pip install -U transformers") from e

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id)
        label, kw = _fit_plan(8.4, precision)          # same footprint class as the Qwen models
        print(f"[audio-flamingo-3] loading in {label}")
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, **kw).eval()

    def ground(self, audio_path, query_text, query=None, duration=None, temperature=None):
        conv = af3_conversation(audio_path, query_text, duration)
        inputs = self.processor.apply_chat_template(
            conv, tokenize=True, add_generation_prompt=True, return_dict=True).to(self.model.device)
        with self.torch.no_grad():
            gen = ({"do_sample": True, "temperature": temperature} if temperature
                   else {"do_sample": False})
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, **gen)
        ids = ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


class GeminiBackend:
    name = "gemini"

    def __init__(self, model_id="gemini-2.5-pro"):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_id = model_id

    def ground(self, audio_path, query_text, query=None, duration=None, temperature=None):
        from google.genai import types

        data = open(audio_path, "rb").read()
        r = self.client.models.generate_content(
            model=self.model_id,
            contents=[types.Part.from_bytes(data=data, mime_type="audio/wav"), SYSTEM + "\n" + prompt_for(query_text, duration)],
        )
        return (r.text or "").strip()


def get_backend(name: str, **kw):
    if name.startswith("mock:"):
        return MockBackend(name.split(":", 1)[1])
    if name == "gemini":
        return GeminiBackend()
    return {"qwen2.5-omni": Qwen25OmniBackend, "qwen2-audio": Qwen2AudioBackend,
            "audio-flamingo-3": AudioFlamingo3Backend}[name](**kw)
