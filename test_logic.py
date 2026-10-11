import os, sys, tempfile, argparse
import numpy as np

# point the vault at a temp dir BEFORE importing the module
tmp = tempfile.mkdtemp(prefix="srvu_test_")
os.environ["SRVU_DIR"] = tmp

sys.path.insert(0, os.path.dirname(__file__))
import sr_voice_ultron as vl

fails = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)

# --- text normalisation / similarity ---
check("normalize strips punctuation",
      vl.normalize_text("The Silver Fox, jumps!") == "the silver fox jumps")
check("exact match = 1.0",
      vl.text_similarity("the silver fox jumps over the quiet river",
                         "the silver fox jumps over the quiet river") == 1.0)
check("close match high",
      vl.text_similarity("the silver fox jumped over a quiet river",
                         "the silver fox jumps over the quiet river") > 0.8)
check("wrong words low",
      vl.text_similarity("hello there how are you today",
                         "the silver fox jumps over the quiet river") < 0.3)
check("empty gives 0", vl.text_similarity("", "anything") == 0.0)

# --- cosine ---
check("cosine identical = 1", abs(vl.cosine_similarity([1,2,3],[1,2,3]) - 1) < 1e-9)
check("cosine orthogonal = 0", abs(vl.cosine_similarity([1,0],[0,1])) < 1e-9)
check("cosine zero-safe", vl.cosine_similarity([0,0],[1,1]) == 0.0)

# --- resample ---
x = np.sin(np.linspace(0, 20, 8000)).astype("float32")
check("resample doubles length", len(vl.resample(x, 8000, 16000)) == 16000)

# --- liveness helpers ---
code = vl.random_code(4)
check("random_code length", len(code) == 4 and code.isdigit())
check("random_code varies", len({vl.random_code(6) for _ in range(20)}) > 1)
check("extract_digits from words",
      vl.extract_digits("say four seven one now") == "471")
check("extract_digits from numerals",
      vl.extract_digits("the code is 471") == "471")
check("extract_digits oh -> 0", vl.extract_digits("one oh two") == "102")
check("strip_digits removes code words",
      vl.strip_digits("the silver fox jumps four seven one")
      == "the silver fox jumps")
check("code_matches in order", vl.code_matches("four seven one", "471"))
check("code_matches rejects wrong order",
      not vl.code_matches("seven four one", "471"))
check("code_matches tolerates extra digit",
      vl.code_matches("four nine seven one", "471"))
check("code_matches rejects missing digit",
      not vl.code_matches("four seven", "471"))

# --- vault save/load with fake embeddings ---
rng = np.random.default_rng(0)
base = rng.normal(size=256)
embs = [base + rng.normal(scale=0.05, size=256) for _ in range(3)]
vl.save_vault(embs, "the silver fox jumps over the quiet river")
v = vl.load_vault()
check("vault loads", v is not None)
check("vault stores passphrase",
      v["meta"]["passphrase"] == "the silver fox jumps over the quiet river")
check("vault mean is unit norm", abs(np.linalg.norm(v["mean"]) - 1) < 1e-6)
check("mean close to true base", vl.cosine_similarity(v["mean"], base) > 0.99)

# --- full unlock decision flow, monkeypatched embed/transcribe ---
def make_args(**kw):
    d = dict(from_file="dummy", seconds=4.0, challenge=False,
             voice_threshold=0.75, text_threshold=0.80)
    d.update(kw); return argparse.Namespace(**d)

vl.load_wav = lambda path, sample_rate=vl.SAMPLE_RATE: np.zeros(16000, dtype="float32")
PHRASE = "the silver fox jumps over the quiet river"

# same voice + right sentence -> unlock (rc 0)
vl.embed_audio = lambda audio: v["mean"] + rng.normal(scale=0.02, size=256)
vl.transcribe_audio = lambda audio: PHRASE
check("unlock: same voice + right words -> OPEN", vl.cmd_unlock(make_args()) == 0)

# same voice + wrong sentence -> locked (rc 2)
vl.transcribe_audio = lambda audio: "open sesame please let me in"
check("unlock: same voice + wrong words -> LOCKED", vl.cmd_unlock(make_args()) == 2)

# different voice + right sentence -> locked (rc 2)
vl.embed_audio = lambda audio: rng.normal(size=256)
vl.transcribe_audio = lambda audio: PHRASE
check("unlock: wrong voice + right words -> LOCKED", vl.cmd_unlock(make_args()) == 2)

# --- challenge (liveness) flow ---
vl.random_code = lambda length=3: "471"
vl.embed_audio = lambda audio: v["mean"] + rng.normal(scale=0.02, size=256)

# right voice + right phrase + right code -> OPEN
vl.transcribe_audio = lambda audio: PHRASE + " four seven one"
check("challenge: correct code -> OPEN", vl.cmd_unlock(make_args(challenge=True)) == 0)

# right voice + right phrase + WRONG code (replayed clip) -> LOCKED
vl.transcribe_audio = lambda audio: PHRASE + " nine nine nine"
check("challenge: wrong code (replay) -> LOCKED",
      vl.cmd_unlock(make_args(challenge=True)) == 2)

# right voice + right code but missing phrase -> LOCKED
vl.transcribe_audio = lambda audio: "four seven one"
check("challenge: code only, no phrase -> LOCKED",
      vl.cmd_unlock(make_args(challenge=True)) == 2)

# --- status ---
check("status returns 0 when enrolled", vl.cmd_status(argparse.Namespace()) == 0)

print("\n" + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
