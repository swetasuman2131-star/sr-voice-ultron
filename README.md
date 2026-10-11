# SR Voice Ultron

A voice-sentence lock for your computer. To open it you must say the right
**sentence** *and* say it in the right **voice**. Two factors in a single
action:

| Factor | What it checks | How |
|---|---|---|
| Something you know | the passphrase sentence | speech-to-text, then text match |
| Something you are | your voiceprint | speaker embeddings, then cosine match |

You enrol once (speak your sentence a few times). From then on the lock stays
shut unless a recording matches **both** the words and the voice.

**Author:** The new developer SR

> This is a personal prototype for trying at home. It is **not** a security
> product - see *Honest limits* at the bottom.

---

## 1. Install

You need Python 3.10+ and a working microphone.

```bash
# 1. get a CPU build of PyTorch (smaller than the default CUDA one)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. install the rest
pip install -r requirements.txt
```

If `resemblyzer` fails to build `webrtcvad`, install a C compiler first
(`build-essential` on Debian/Ubuntu, Xcode command-line tools on macOS).

---

## 2. Use it

### Enrol your voice (do this once)

```bash
python sr_voice_ultron.py enroll
```

It prints the passphrase, records you **3 times**, and stores a voiceprint in
`~/.sr_voice_ultron/`. Choose your own sentence with:

```bash
python sr_voice_ultron.py enroll --passphrase "my cat plays piano at midnight"
```

Tips: sit in a quiet room, keep a steady distance from the mic, and say the
sentence the same way each time. After enrolment it prints a *self-consistency*
score - if that is below ~0.70, re-enrol somewhere quieter.

### Open the lock

```bash
python sr_voice_ultron.py unlock
```

You get a clear verdict:

```
  heard          : "the silver fox jumps over the quiet river"
  expected       : "the silver fox jumps over the quiet river"
  voice match    : 0.912  (need >= 0.75)  OK
  sentence match : 1.000  (need >= 0.80)  OK

  UNLOCKED - sentence and voice both match.
```

If the words are right but the voice is someone else's, it stays **LOCKED** and
tells you why - that is the whole point of the design.

### On-screen lock (optional)

```bash
python sr_voice_ultron.py gui
```

A small window with a big LOCKED/UNLOCKED indicator and *Unlock* / *Enrol*
buttons.

### Check what is stored

```bash
python sr_voice_ultron.py status
```

---

## 3. Testing without a microphone

Every command accepts `--from-file` with a **16 kHz mono WAV**, so you can try
the whole flow using pre-recorded clips:

```bash
python sr_voice_ultron.py enroll --from-file my_voice.wav
python sr_voice_ultron.py unlock --from-file my_voice.wav      # -> UNLOCKED
python sr_voice_ultron.py unlock --from-file a_friend.wav      # -> LOCKED (wrong voice)
```

(Need a WAV? Record one with
`python -c "import sounddevice as sd, soundfile as sf; a=sd.rec(64000,16000,1); sd.wait(); sf.write('my_voice.wav', a, 16000)"`.)

---

## 4. Tuning the sensitivity

Two thresholds control how strict the lock is. Raise them for a stricter lock,
lower them if you are being wrongly rejected.

| Flag | Default | Meaning |
|---|---|---|
| `--voice-threshold` | `0.75` | min voiceprint similarity to open. Same speaker ~0.75-0.95; a different speaker ~0.50-0.70 |
| `--text-threshold` | `0.80` | min sentence similarity. `0.80` tolerates small slips of the tongue |
| `--seconds` | `4` | recording length |

Example - a very strict lock:

```bash
python sr_voice_ultron.py unlock --voice-threshold 0.85 --text-threshold 0.90
```

---

## 5. How it works (short version)

1. **Capture** - 16 kHz mono audio from the mic (or a WAV file).
2. **Voiceprint** - [resemblyzer](https://github.com/resemble-ai/Resemblyzer)
   turns the clip into a 256-dimension "voice fingerprint". Enrolment averages
   several of these into one reference vector.
3. **Words** - [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
   transcribes the clip offline.
4. **Decision** - the lock opens only if `cosine(voiceprint, reference)` clears
   the voice threshold **and** the transcript is close enough to the
   passphrase. Both must pass.

The voiceprint itself never leaves your machine; everything runs offline
(unless you use the optional Google fallback for speech-to-text).

---

## 6. Tests

The matching and decision logic has a small test suite that runs without a
microphone or the heavy models:

```bash
python test_logic.py
```

---

## 7. Honest limits

Voice biometrics are convenient but weaker than a good password, and you should
know why before trusting one:

- **It can be spoofed.** A recording of the enrolled person, or a cloned voice,
  can fool a speaker-verification model. Real systems add *liveness* checks
  (ask for a random phrase, detect replay) - this prototype does not.
- **Your voice changes.** Colds, stress, and background noise all shift the
  voiceprint, so you may occasionally get a false rejection.
- **The stored voiceprint is not encrypted.** It sits in `~/.sr_voice_ultron/`
  in plain form. Fine for a home prototype; not for anything sensitive.
- **This is a demo, not access control.** Do not put it in front of anything
  that actually matters (banking, your OS login, a server).

Use it to learn, to demo, and to play with - and if you ever want to build on
it, the honest next steps are liveness detection, encrypted storage, and
pairing the voice factor with a real secret.
