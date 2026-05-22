"""QA-only: cross-check our OCR-parsed headwords for a volume against a reference transcript's
headwords (Humphrey Southall's HTML transcripts, used solely to VALIDATE our parser — counts and
headword lists are facts; the transcript prose is NOT ingested into our pipeline or output).

  his-only  = headwords the transcript has but we don't  -> entries we MERGED AWAY (segmentation gaps)
  ours-only = headwords we have but the transcript doesn't -> our spurious headwords (stamps/fragments)
            (both lists carry OCR-spelling noise; the large/real toponyms in his-only are the signal.)

  srun ... python3 process/crosscheck_headwords.py <reference.html> <our-ocr.txt>
"""
import sys, re, types, random

# stub tiktoken (token-count column only; not needed here)
_fake = types.ModuleType("tiktoken")
_fake.get_encoding = lambda name: types.SimpleNamespace(encode=lambda s: s.split())
sys.modules["tiktoken"] = _fake
sys.path.insert(0, "process")
import parse_ocr as P


def norm(h):
    return re.sub(r"[^A-Z]", "", (h or "").upper())


html = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
his = set()
for m in re.finditer(r"<p>\s*([A-ZÀ-Þ][A-ZÀ-Þ0-9'’ .&-]{1,60}?)\s*[,.]", html):
    k = norm(m.group(1))
    if len(k) >= 3:
        his.add(k)

entries = P.parse(open(sys.argv[2], encoding="utf-8").read())
ours = {norm(e["headword"]) for e in entries if norm(e["headword"])}

his_only, ours_only = his - ours, ours - his
print(f"reference={len(his)}  ours={len(ours)}  common={len(his & ours)} "
      f"({100*len(his & ours)/max(len(his),1):.1f}% of reference matched)")
print(f"his-only (we may be MISSING): {len(his_only)}")
print(f"ours-only (possible spurious): {len(ours_only)}")
print("\nsample his-only:", sorted(random.sample(sorted(his_only), min(30, len(his_only)))))
print("\nsample ours-only:", sorted(random.sample(sorted(ours_only), min(30, len(ours_only)))))
