"""QA-only: cross-check our OCR-parsed headwords for a volume against a reference transcript's
headwords (Humphrey Southall's HTML transcripts, used solely to VALIDATE our parser — counts and
headword lists are facts; the transcript prose is NOT ingested into our pipeline or output).

  his-only  = headwords the transcript has but we don't  -> entries we MERGED AWAY (segmentation gaps)
  ours-only = headwords we have but the transcript doesn't -> our spurious headwords (stamps/fragments)
            (both lists carry OCR-spelling noise; the large/real toponyms in his-only are the signal.)

  srun ... python3 process/crosscheck_headwords.py <reference.html> <our-ocr.txt>
"""
import sys, re, types, random, difflib
from collections import defaultdict

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
# headword = leading ALL-CAPS run + the " (PARENTHETICAL)" (e.g. ACCRINGTON (NEW)) CAPTURED as part
# of the toponym (it distinguishes New/Old, Grande/Piccolo, Mount, …), then , or .
for m in re.finditer(r"<p>\s*([A-ZÀ-Þ][A-ZÀ-Þ0-9'’ .&-]{1,60}?(?: \([^)]*\))?)\s*[,.]", html):
    k = norm(m.group(1))
    if len(k) >= 3:
        his.add(k)

entries = P.parse(open(sys.argv[2], encoding="utf-8").read())
ours_kind = {}                                   # full headword (incl. parenthetical) -> kind
for e in entries:
    k = norm(e["headword_raw"])                  # headword_raw keeps the (PARENTHETICAL)
    if k:
        ours_kind.setdefault(k, e["kind"])
ours = set(ours_kind)

his_only, ours_only = his - ours, ours - his
print(f"reference={len(his)}  ours={len(ours)}  common={len(his & ours)} "
      f"({100*len(his & ours)/max(len(his),1):.1f}% of reference matched)")
print(f"his-only (we may be MISSING): {len(his_only)}")

# --- decompose ours-only: crossref | spelling-variant of a his headword | short fragment | OTHER ---
his_bucket = defaultdict(list)
for h in his:
    his_bucket[h[:2]].append(h)
oo_crossref = oo_variant = oo_fragment = 0
oo_other = []
for h in ours_only:
    if ours_kind.get(h) == "crossref":
        oo_crossref += 1
    elif len(h) <= 4:
        oo_fragment += 1
    elif difflib.get_close_matches(h, his_bucket.get(h[:2], []), n=1, cutoff=0.84):
        oo_variant += 1                          # close to a his headword -> OCR spelling variant
    else:
        oo_other.append(h)
print(f"ours-only total={len(ours_only)}: crossref={oo_crossref} "
      f"spelling-variant={oo_variant} fragment(<=4)={oo_fragment} OTHER={len(oo_other)}")
print("sample OTHER (genuine spurious candidates):",
      sorted(random.sample(oo_other, min(30, len(oo_other)))))
