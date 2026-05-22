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

ocr_text = open(sys.argv[2], encoding="utf-8").read()
normtext = re.sub(r"[^A-Z]", "", ocr_text.upper())   # whole volume, alpha-only, for substring tests
entries = P.parse(ocr_text)
ours_kind, ours_text = {}, {}                    # full headword (incl. parenthetical) -> kind / text
for e in entries:
    k = norm(e["headword_raw"])                  # headword_raw keeps the (PARENTHETICAL)
    if k:
        ours_kind.setdefault(k, e["kind"])
        ours_text.setdefault(k, e["text"])
ours = set(ours_kind)

common = his & ours
rh, ro = his - ours, ours - his                  # residual his-only / ours-only after EXACT match

# --- tighten: fuzzy bipartite match (OCR spelling variants pair up & cancel from both residuals) ---
rh_bucket = defaultdict(list)
for h in rh:
    rh_bucket[h[0]].append(h)                    # bucket by FIRST char (tolerate 2nd-char OCR diffs)
paired_h, var_o = set(), set()
for o in sorted(ro):
    cands = [c for c in rh_bucket.get(o[0], []) if c not in paired_h]
    m = difflib.get_close_matches(o, cands, n=1, cutoff=0.82)
    hit = m[0] if m else None
    if not hit:                                  # leading-junk / truncation: one contains the other
        for c in cands:
            if min(len(o), len(c)) >= 6 and (o in c or c in o):
                hit = c
                break
    if hit:
        var_o.add(o)
        paired_h.add(hit)
# 2nd pass for first-char OCR errors: containment against ALL remaining his-only
leftover_h = [h for h in rh if h not in paired_h]
for o in sorted(ro - var_o):
    for c in leftover_h:
        if c not in paired_h and min(len(o), len(c)) >= 7 and (o in c or c in o):
            var_o.add(o)
            paired_h.add(c)
            break
true_ro = ro - var_o                             # ours-only with NO fuzzy partner in his
true_rh = rh - paired_h                          # his-only with NO fuzzy partner in ours
match_pct = 100 * (len(common) + len(var_o)) / max(len(his), 1)

# classify the TRUE ours-only by our entry-text quality (a real place has a descriptor; junk doesn't)
DESCR = re.compile(r"\b(a|an|the|see|or|is|in|of|on|near|chief|town|vil|river|prov)\b", re.I)
def junk(h):
    t = ours_text.get(h, "")
    return len(h) <= 4 or len(t) < 35 or not DESCR.search(t[:90])
ro_crossref = {h for h in true_ro if ours_kind.get(h) == "crossref"}
ro_junk = {h for h in true_ro - ro_crossref if junk(h)}          # over-split / fragment / garbled
ro_realplace = true_ro - ro_crossref - ro_junk                   # real places his transcript omits
# his-only that LOOK like real toponyms (vowel, length) = candidate entries we may have merged away
rh_real = {h for h in true_rh if re.search(r"[AEIOU]", h) and 4 < len(h) < 22}
# ENTRY-START context test: collect every "NAME, <lowercase descriptor>" in our OCR — a real
# (possibly un-split) entry start, ALL-CAPS as the gazetteer prints headwords — vs a prose mention.
starts = set()
for m in re.finditer(r"(?:^|\n|(?<=\. )|(?<=\.\n))\s*"
                     r"([A-ZÀ-Þ][A-ZÀ-Þ '’.\-]{1,40}?)(?: \([^)]*\))?,\s+[a-z]", ocr_text):
    k = norm(m.group(1))
    if len(k) >= 4:
        starts.add(k)
rh_merge = {h for h in rh_real if h in starts}                     # un-split entry-start => MERGE (check)
rh_prose = {h for h in rh_real if h not in starts and h in normtext}  # only a prose mention (not ours)
rh_absent = rh_real - rh_merge - rh_prose

print(f"reference={len(his)} ours={len(ours)} exact-common={len(common)} fuzzy-variant={len(var_o)} "
      f"=> {match_pct:.1f}% matched")
print(f"true_ours_only={len(true_ro)} [crossref={len(ro_crossref)} junk/oversplit={len(ro_junk)} "
      f"real-place-he-omits={len(ro_realplace)}]")
print(f"true_his_only={len(true_rh)} [MERGE(un-split start in our text)={len(rh_merge)} "
      f"prose-mention-only={len(rh_prose)} absent-from-our-edition={len(rh_absent)}]")
print(f"MANUAL-CHECK ~= junk({len(ro_junk)}) + merge({len(rh_merge)}) = {len(ro_junk)+len(rh_merge)}")
print("sample MERGE:", sorted(random.sample(sorted(rh_merge), min(30, len(rh_merge)))))
