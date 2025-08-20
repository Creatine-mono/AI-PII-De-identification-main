import re
import pandas as pd

PH_TOKEN = re.compile(r"^\{?([A-Za-z0-9_]{1,64})\}?$")

def _norm_ph(name: str) -> str:
    try:
        return PH_MAP_TO_CSV.get(name, name)
    except NameError:
        return name

def inject_pii_inline(gen_explode: pd.DataFrame, pii_row: pd.Series, pii_placeholders: list) -> pd.DataFrame:
    rows = []
    ph_set = set(pii_placeholders)
    for t, ws, fname in zip(gen_explode["tokens"], gen_explode["trailing_whitespace"], gen_explode["file_name"]):
        lab = "O"
        m = PH_TOKEN.match(str(t))
        if m:
            raw = m.group(1)
            ph = _norm_ph(raw)
            if ph in ph_set and ph in pii_row.index:
                value = "" if pd.isna(pii_row[ph]) else str(pii_row[ph])
                if value:
                    for i, ch in enumerate(value):
                        rows.append({
                            "file_name": fname,
                            "tokens": ch,
                            "trailing_whitespace": False if i < len(value) - 1 else ws,
                            "label": f"B-{ph}" if i == 0 else f"I-{ph}",
                        })
                    continue
        rows.append({
            "file_name": fname,
            "tokens": t,
            "trailing_whitespace": ws,
            "label": lab,
        })
    out = pd.DataFrame(rows)
    assert len(out["tokens"]) == len(out["trailing_whitespace"]) == len(out["label"])
    return out
