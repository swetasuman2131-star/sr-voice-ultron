#!/usr/bin/env python3
"""
SR Voice Ultron
===============
A voice-sentence lock. To open the lock you must say the right SENTENCE
and say it in the right VOICE. It is two factors in one action:

    factor 1 (something you know)  -> the passphrase sentence, matched by speech-to-text
    factor 2 (something you are)   -> your voiceprint, matched by speaker embeddings

You enrol once (say the sentence a few times). After that the lock stays
shut unless a recording matches BOTH the words and the voice.

Author: The new developer SR

This is a personal prototype, not a security product. See README.md for the
honest limits of voice biometrics.

Commands
--------
    python sr_voice_ultron.py enroll          # set up your voice + sentence
    python sr_voice_ultron.py unlock          # try to open the lock
    python sr_voice_ultron.py status          # show what is stored
    python sr_voice_ultron.py enroll --from-file samples.wav
    python sr_voice_ultron.py unlock --from-file test.wav
    python sr_voice_ultron.py gui             # simple on-screen lock (Tkinter)

Test without a microphone by passing --from-file <wav> (16 kHz mono WAV).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SAMPLE_RATE = 16000          # resemblyzer + faster-whisper both expect 16 kHz
DEFAULT_PASSPHRASE = "the silver fox jumps over the quiet river"
VAULT_DIR = Path(os.environ.get("SRVU_DIR", Path.home() / ".sr_voice_ultron"))
VAULT_EMB = VAULT_DIR / "voiceprint.npz"
VAULT_META = VAULT_DIR / "meta.json"

# How similar (cosine, 0..1) a voice must be to the enrolled voiceprint.
# Same speaker is usually 0.75-0.95; a different speaker usually 0.5-0.70.
DEFAULT_VOICE_THRESHOLD = 0.75

# How similar the spoken words must be to the passphrase (0..1).
DEFAULT_TEXT_THRESHOLD = 0.80

# How long to record for (seconds).
DEFAULT_RECORD_SECONDS = 4.0


# --------------------------------------------------------------------------
# Small helpers (pure logic, no heavy deps -> easy to unit test)
# --------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_similarity(said: str, expected: str) -> float:
    """Blend of whole-string similarity and word overlap, in 0..1."""
    a, b = normalize_text(said), normalize_text(expected)
    if not a or not b:
        return 0.0
    seq = SequenceMatcher(None, a, b).ratio()
    wa, wb = set(a.split()), set(b.split())
    if not wa or not wb:
        return seq
    overlap = len(wa & wb) / max(len(wb), 1)   # fraction of expected words present
    return 0.5 * seq + 0.5 * overlap


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# --------------------------------------------------------------------------
# Audio capture
# --------------------------------------------------------------------------

def record(duration: float = DEFAULT_RECORD_SECONDS, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record mono float32 audio from the default microphone."""
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Microphone support needs the 'sounddevice' package.\n"
            "Install it with:  pip install sounddevice"
        ) from exc

    print(f"  Recording for {duration:.1f}s ... speak now.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    audio = np.squeeze(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-3:
        print("  (warning: the recording is almost silent - check your mic)")
    return audio


def load_wav(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read a WAV file as mono float32 at the target sample rate."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        audio = resample(audio, sr, sample_rate)
    return audio.astype(np.float32)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear resample - good enough for a prototype."""
    if sr_in == sr_out or audio.size == 0:
        return audio
    n_out = int(round(audio.size * sr_out / sr_in))
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def save_wav(path: str, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    import soundfile as sf
    sf.write(path, audio, sample_rate)


def get_audio(args) -> np.ndarray:
    """Either read a file (--from-file) or record from the mic."""
    if getattr(args, "from_file", None):
        print(f"  Reading audio from {args.from_file}")
        return load_wav(args.from_file)
    return record(getattr(args, "seconds", DEFAULT_RECORD_SECONDS))


# --------------------------------------------------------------------------
# Voiceprint (speaker embedding) - lazy import so the module loads without it
# --------------------------------------------------------------------------

_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        try:
            from resemblyzer import VoiceEncoder
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Voiceprint support needs 'resemblyzer' (and torch).\n"
                "Install with:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "  pip install resemblyzer"
            ) from exc
        _ENCODER = VoiceEncoder()
    return _ENCODER


def embed_audio(audio: np.ndarray) -> np.ndarray:
    """Turn audio into a 256-dim voiceprint vector."""
    from resemblyzer import preprocess_wav
    encoder = _get_encoder()
    wav = preprocess_wav(np.asarray(audio, dtype=np.float32), source_sr=SAMPLE_RATE)
    return np.asarray(encoder.embed_utterance(wav), dtype=np.float64)


# --------------------------------------------------------------------------
# Speech-to-text - lazy import, with a fallback
# --------------------------------------------------------------------------

_WHISPER = None


def transcribe_audio(audio: np.ndarray) -> str:
    """Transcribe speech to text (offline via faster-whisper if available)."""
    global _WHISPER
    # Preferred: faster-whisper (offline, private, accurate)
    try:
        from faster_whisper import WhisperModel
        if _WHISPER is None:
            _WHISPER = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = _WHISPER.transcribe(
            np.asarray(audio, dtype=np.float32), language="en", beam_size=1)
        return " ".join(s.text for s in segments).strip()
    except ImportError:
        pass

    # Fallback: SpeechRecognition + Google Web Speech (needs internet)
    try:
        import speech_recognition as sr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Speech-to-text needs 'faster-whisper' (offline) or 'speech_recognition'.\n"
            "Install with:  pip install faster-whisper"
        ) from exc

    recognizer = sr.Recognizer()
    # SpeechRecognition wants 16-bit PCM bytes
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    clip = sr.AudioData(pcm, SAMPLE_RATE, 2)
    try:
        return recognizer.recognize_google(clip)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as exc:  # pragma: no cover
        raise RuntimeError(f"Google speech service unreachable: {exc}") from exc


# --------------------------------------------------------------------------
# Vault (what we store after enrolment)
# --------------------------------------------------------------------------

def save_vault(embeddings: list[np.ndarray], passphrase: str) -> None:
    VAULT_DIR.mkdir(parents=True, exist_ok=True)
    mat = np.vstack(embeddings)                 # shape (n_samples, 256)
    mean = mat.mean(axis=0)
    mean = mean / (np.linalg.norm(mean) + 1e-9)  # unit-normalised voiceprint
    np.savez(VAULT_EMB, samples=mat, mean=mean)
    VAULT_META.write_text(json.dumps({
        "passphrase": passphrase,
        "n_samples": len(embeddings),
        "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_rate": SAMPLE_RATE,
    }, indent=2))


def load_vault():
    if not VAULT_EMB.exists() or not VAULT_META.exists():
        return None
    data = np.load(VAULT_EMB)
    meta = json.loads(VAULT_META.read_text())
    return {"mean": data["mean"], "samples": data["samples"], "meta": meta}


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_enroll(args) -> int:
    passphrase = args.passphrase or DEFAULT_PASSPHRASE
    n = args.samples
    print("Enrolling your voice for the SR Voice Ultron lock.")
    print(f'  Passphrase: "{passphrase}"')
    print(f"  You will speak it {n} times.\n")

    embeddings: list[np.ndarray] = []
    if args.from_file:
        # one file = one sample (repeat the flag to give more)
        audio = load_wav(args.from_file)
        embeddings.append(embed_audio(audio))
    else:
        for i in range(n):
            print(f"Sample {i + 1}/{n}")
            audio = get_audio(args)
            embeddings.append(embed_audio(audio))
            print("  voiceprint captured.\n")
            if i < n - 1:
                time.sleep(0.4)

    if len(embeddings) < 1:
        print("No samples captured - aborting.")
        return 1

    # Sanity check: samples of the same person should agree with each other.
    if len(embeddings) >= 2:
        sims = [cosine_similarity(embeddings[i], embeddings[j])
                for i in range(len(embeddings)) for j in range(i + 1, len(embeddings))]
        avg = float(np.mean(sims))
        print(f"Self-consistency across your samples: {avg:.3f}")
        if avg < 0.70:
            print("  (low - try again somewhere quieter, closer to the mic)")

    save_vault(embeddings, passphrase)
    print(f"\nLock enrolled. Vault written to {VAULT_DIR}")
    return 0


def cmd_unlock(args) -> int:
    vault = load_vault()
    if vault is None:
        print("No enrolment found. Run:  python sr_voice_ultron.py enroll")
        return 1

    voice_th = args.voice_threshold
    text_th = args.text_threshold

    print("Lock is LOCKED. Speak your passphrase to open it.")
    audio = get_audio(args)

    print("  Checking your voice ...")
    probe = embed_audio(audio)
    voice_score = cosine_similarity(probe, vault["mean"])

    print("  Checking your words ...")
    said = transcribe_audio(audio)
    text_score = text_similarity(said, vault["meta"]["passphrase"])

    voice_ok = voice_score >= voice_th
    text_ok = text_score >= text_th

    print("\n--- result ---")
    print(f'  heard          : "{said}"')
    print(f'  expected       : "{vault["meta"]["passphrase"]}"')
    print(f"  voice match    : {voice_score:.3f}  (need >= {voice_th:.2f})  {'OK' if voice_ok else 'no'}")
    print(f"  sentence match : {text_score:.3f}  (need >= {text_th:.2f})  {'OK' if text_ok else 'no'}")

    if voice_ok and text_ok:
        print("\n  UNLOCKED - sentence and voice both match.")
        return 0
    if text_ok and not voice_ok:
        print("\n  STILL LOCKED - right words, but that is not the enrolled voice.")
    elif voice_ok and not text_ok:
        print("\n  STILL LOCKED - right voice, wrong sentence.")
    else:
        print("\n  STILL LOCKED - neither the sentence nor the voice matched.")
    return 2


def cmd_status(args) -> int:
    vault = load_vault()
    if vault is None:
        print("No enrolment found.")
        print(f"Vault location: {VAULT_DIR}")
        return 1
    m = vault["meta"]
    print("SR Voice Ultron vault")
    print(f"  location    : {VAULT_DIR}")
    print(f'  passphrase  : "{m["passphrase"]}"')
    print(f"  samples     : {m['n_samples']}")
    print(f"  enrolled at : {m['enrolled_at']}")
    return 0


# --------------------------------------------------------------------------
# Optional simple GUI
# --------------------------------------------------------------------------

def cmd_gui(args) -> int:
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        print("Tkinter is not available on this Python build. Use the CLI instead.")
        return 1

    import threading

    root = tk.Tk()
    root.title("SR Voice Ultron")
    root.geometry("440x320")
    root.configure(bg="#12161c")

    state = {"unlocked": False}
    status_var = tk.StringVar(value="LOCKED")
    detail_var = tk.StringVar(value="Press 'Unlock' and speak your passphrase.")

    tk.Label(root, text="SR Voice Ultron", fg="#7fd1ff", bg="#12161c",
             font=("Helvetica", 20, "bold")).pack(pady=(18, 0))
    tk.Label(root, textvariable=status_var, fg="#ff6b6b", bg="#12161c",
             font=("Helvetica", 26, "bold")).pack(pady=10)
    tk.Label(root, textvariable=detail_var, fg="#9aa4b2", bg="#12161c",
             font=("Helvetica", 11), wraplength=380).pack(pady=4)

    def set_state(unlocked, detail):
        state["unlocked"] = unlocked
        status_var.set("UNLOCKED" if unlocked else "LOCKED")
        detail_var.set(detail)
        color = "#4ade80" if unlocked else "#ff6b6b"
        for w in root.winfo_children():
            if isinstance(w, tk.Label) and w.cget("font") == ("Helvetica", 26, "bold"):
                w.configure(fg=color)

    def do_unlock():
        if not load_vault():
            messagebox.showinfo("SR Voice Ultron", "No enrolment yet. Please enrol first.")
            return
        btn.config(state="disabled")
        detail_var.set("Listening ...")

        def work():
            try:
                audio = record(args.seconds)
                probe = embed_audio(audio)
                vault = load_vault()
                vs = cosine_similarity(probe, vault["mean"])
                said = transcribe_audio(audio)
                ts = text_similarity(said, vault["meta"]["passphrase"])
                ok = vs >= args.voice_threshold and ts >= args.text_threshold
                root.after(0, lambda: set_state(
                    ok, f'heard "{said}"\nvoice {vs:.2f} . sentence {ts:.2f}'))
            except Exception as exc:  # keep the GUI alive
                root.after(0, lambda: set_state(False, f"Error: {exc}"))
            finally:
                root.after(0, lambda: btn.config(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    def do_enroll():
        if not messagebox.askyesno("SR Voice Ultron", "Enrol (or re-enrol) your voice now?"):
            return
        args_local = argparse.Namespace(passphrase=None, samples=3,
                                        seconds=args.seconds, from_file=None)
        btn.config(state="disabled")
        detail_var.set("Enrolling - speak 3 times ...")

        def work():
            try:
                cmd_enroll(args_local)
                root.after(0, lambda: set_state(False, "Enrolled. Press 'Unlock' to test."))
            except Exception as exc:
                root.after(0, lambda: set_state(False, f"Error: {exc}"))
            finally:
                root.after(0, lambda: btn.config(state="normal"))

        threading.Thread(target=work, daemon=True).start()

    btn = tk.Button(root, text="Unlock", command=do_unlock,
                    bg="#2563eb", fg="white", font=("Helvetica", 13, "bold"),
                    relief="flat", padx=24, pady=8)
    btn.pack(pady=14)
    tk.Button(root, text="Enrol voice", command=do_enroll,
              bg="#1f2937", fg="#cbd5e1", font=("Helvetica", 11),
              relief="flat", padx=14, pady=4).pack()

    root.mainloop()
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sr_voice_ultron",
        description="SR Voice Ultron - open a lock by speaking the right sentence in the right voice.")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--from-file", help="use a 16 kHz mono WAV instead of the mic")
    common.add_argument("--seconds", type=float, default=DEFAULT_RECORD_SECONDS,
                        help="record duration in seconds (default 4)")
    common.add_argument("--voice-threshold", type=float, default=DEFAULT_VOICE_THRESHOLD,
                        help="min voiceprint similarity to unlock (default 0.75)")
    common.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD,
                        help="min sentence similarity to unlock (default 0.80)")

    e = sub.add_parser("enroll", parents=[common], help="set up your voice + sentence")
    e.add_argument("--passphrase", help=f'your sentence (default: "{DEFAULT_PASSPHRASE}")')
    e.add_argument("--samples", type=int, default=3, help="how many times to speak it")
    e.set_defaults(func=cmd_enroll)

    u = sub.add_parser("unlock", parents=[common], help="try to open the lock")
    u.set_defaults(func=cmd_unlock)

    s = sub.add_parser("status", help="show what is stored")
    s.set_defaults(func=cmd_status)

    g = sub.add_parser("gui", parents=[common], help="on-screen lock (Tkinter)")
    g.set_defaults(func=cmd_gui)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
