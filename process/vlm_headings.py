"""Reference-free OCR/segmentation validator: ask a self-hosted vision-LLM (Qwen2.5-VL) to list the
place-name ENTRY HEADINGS on a page image, independently of our Surya OCR + parser, then diff against
our parsed headwords. Surfaces MERGES (a heading the VLM sees but we lack) and SPURIOUS entries (we
have, the VLM doesn't). No transcript needed -> generalises to any gazetteer.

Serve Qwen2.5-VL first (vLLM OpenAI server), then:
  TABLE_VL_BASE=http://127.0.0.1:8000/v1 python3 process/vlm_headings.py \
      --db data/gotw_seg.sqlite --vol v3 --img-dir /vast/ishi/gotw/img/v3 --idx 100 237
"""
import argparse, base64, io, json, os, re, sqlite3, urllib.request
from pathlib import Path
from PIL import Image

MAXDIM = int(os.environ.get("VLM_MAXDIM", "1600"))   # downscale: 600dpi scans are too big for the VL


def _jpeg(im, maxdim=MAXDIM):
    if max(im.size) > maxdim:
        s = maxdim / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=90)
    return out.getvalue()


def column_crops(jpeg):
    """Split a two-column page into left/right column crops (with gutter overlap), each downscaled —
    one full dense page overwhelms the VL; a single column is half as dense and stays legible."""
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    w, h = im.size
    return [_jpeg(im.crop((0, 0, int(w * 0.54), h))),       # left + a little past the gutter
            _jpeg(im.crop((int(w * 0.46), 0, w, h)))]        # right + a little before the gutter

VL_MODEL = os.environ.get("TABLE_VL_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct-AWQ")
VL_BASE = os.environ.get("TABLE_VL_BASE", "http://127.0.0.1:8000/v1")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

PROMPT = (
    "This is one page of *A Gazetteer of the World* (1856), a two-column dictionary of places. "
    "List every ENTRY HEADING on the page — the bold capitalised place-names that begin each entry — "
    "in reading order (left column top-to-bottom, then the right column). For each heading give:\n"
    "  • name: the headword as printed;\n"
    "  • variants: any alternative forms given right after it ('X, or Y' -> [\"Y\"]); else [];\n"
    "  • see: if the entry is only a cross-reference ('... See Z.'), the target Z; else null.\n"
    "EXCLUDE the running head (the repeated page-title at the very top), page numbers, column headers, "
    "and anything inside statistical tables. Return JSON: {\"headings\":[{\"name\",\"variants\",\"see\"}]}."
)
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["headings"],
    "properties": {"headings": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["name", "variants", "see"],
        "properties": {"name": {"type": "string"},
                       "variants": {"type": "array", "items": {"type": "string"}},
                       "see": {"type": ["string", "null"]}}}}}}


def alpha(s):
    return re.sub(r"[^A-Z]", "", (s or "").upper())


def vlm_headings(jpeg):
    b64 = base64.b64encode(jpeg).decode()
    body = json.dumps({
        "model": VL_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
        "max_tokens": 8192, "temperature": 0,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "Headings", "schema": SCHEMA, "strict": True}},
    }).encode()
    req = urllib.request.Request(VL_BASE.rstrip("/") + "/chat/completions", data=body, method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=900))
        return json.loads(r["choices"][0]["message"]["content"]).get("headings", [])
    except Exception:
        return []                                    # truncated/garbage VLM output -> skip this column, don't crash


def vol_headwords(con, vol):
    """Our headwords in seq (≈alphabetical) order for a volume — for sequence-window alignment."""
    return [alpha(hw) for (hw,) in con.execute(
        "SELECT e.headword_disp FROM entry e JOIN source s ON e.source_id=s.source_id "
        "WHERE s.filename=? AND e.kind IN ('entry','crossref') ORDER BY e.seq", (f"gotw-{vol}-ocr.txt",))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/gotw_seg.sqlite")
    ap.add_argument("--vol", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--idx", type=int, nargs="+", required=True, help="image indices to validate")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    seqhw = vol_headwords(con, args.vol)
    files = sorted(p for p in Path(args.img_dir).iterdir() if p.suffix.lower() in IMG_EXTS)
    for idx in args.idx:
        try:
            heads = []
            for crop in column_crops(files[idx].read_bytes()):   # left then right column
                heads += vlm_headings(crop)
        except Exception as e:
            print(f"idx {idx}: VLM error {e}")
            continue
        # dedup by normalised name, preserve order
        seen, uniq = set(), []
        for h in heads:
            k = alpha(h.get("name", ""))
            if k and k not in seen:
                seen.add(k)
                uniq.append(h)
        heads = uniq
        vlm = {alpha(h["name"]) for h in heads if alpha(h.get("name", ""))}
        # sequence-window alignment: the contiguous run of our headwords where the VLM list lands
        pos = [i for i, h in enumerate(seqhw) if h in vlm]
        if not pos:
            print(f"\n=== idx {idx} | VLM headings={len(vlm)} — no overlap with our headwords ===")
            print("  VLM:", sorted(vlm)[:20]); continue
        a, b = min(pos), max(pos)
        window = seqhw[a:b + 1]
        wset = set(window)
        vlm_only = sorted(vlm - wset)
        ours_only = [h for h in window if h not in vlm]
        print(f"\n=== idx {idx} -> our seq window [{a}..{b}] ({len(window)} entries) | "
              f"VLM headings={len(vlm)} common={len(vlm & wset)} ===")
        print("  VLM sees, we LACK (merge?):", vlm_only[:20])
        print("  we have in-range, VLM lacks (spurious / VLM-missed?):", ours_only[:20])
        xref = [h["name"] for h in heads if h.get("see")]
        var = [(h["name"], h["variants"]) for h in heads if h.get("variants")]
        if xref:
            print("  VLM cross-refs:", xref[:8])
        if var:
            print("  VLM variants:", var[:6])


if __name__ == "__main__":
    main()
