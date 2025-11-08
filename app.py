pip install sentence-transformers plotly
import math
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import streamlit as st

import inspect
import re
import io
import zipfile
import json

# Neu: Plotly für Treemap
try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


def bordered_container():
    try:
        if "border" in inspect.signature(st.container).parameters:
            return st.container(border=True)
    except Exception:
        pass
    return st.container()


# ===============================
# Page config & Branding
# ===============================
st.set_page_config(page_title="ONE Link Intelligence", layout="wide")

# Session-State init
if "ready" not in st.session_state:
    st.session_state.ready = False
if "res1_df" not in st.session_state:
    st.session_state.res1_df = None
if "out_df" not in st.session_state:
    st.session_state.out_df = None

# Analyse-3 Loader Flags
if "__gems_loading__" not in st.session_state:
    st.session_state["__gems_loading__"] = False
if "__ready_gems__" not in st.session_state:
    st.session_state["__ready_gems__"] = False
if "__gems_ph__" not in st.session_state:
    st.session_state["__gems_ph__"] = st.empty()

# Logo
try:
    st.image(
        "https://onebeyondsearch.com/img/ONE_beyond_search%C3%94%C3%87%C3%B4gradient%20%282%29.png",
        width=250,
    )
except Exception:
    pass

st.title("ONE Link Intelligence")

st.markdown(
    """
<div style="background-color: #f2f2f2; color: #000000; padding: 15px 20px; border-radius: 6px; font-size: 0.9em; max-width: 850px; margin-bottom: 1.5em; line-height: 1.5;">
  Entwickelt von <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">Daniel Kremer</a> von <a href="https://onebeyondsearch.com/" target="_blank">ONE Beyond Search</a> &nbsp;|&nbsp;
  Folge mir auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a> für mehr SEO-Insights und Tool-Updates
</div>
<hr>
""",
    unsafe_allow_html=True,
)

# ===============================
# Helpers
# ===============================
POSSIBLE_SOURCE = ["quelle", "source", "from", "origin", "linkgeber", "quell-url"]
POSSIBLE_TARGET = ["ziel", "destination", "to", "target", "ziel-url", "ziel url"]
POSSIBLE_POSITION = ["linkposition", "link position", "position"]

# Neu: Anchor/ALT-Erkennung
POSSIBLE_ANCHOR = [
    "anchor", "anchor text", "anchor-text", "anker", "ankertext", "linktext", "text",
    "link anchor", "link anchor text", "link text"
]
POSSIBLE_ALT = ["alt", "alt text", "alt-text", "alttext", "image alt", "alt attribute", "alt attribut"]

# Navigative/generische Anchors ausschließen (für Konflikte)
NAVIGATIONAL_ANCHORS = {
    "hier", "zum artikel", "mehr", "mehr erfahren", "klicken sie hier", "click here",
    "here", "read more", "weiterlesen", "zum beitrag", "zum blog", "zum shop"
}

def _num(x, default: float = 0.0) -> float:
    v = pd.to_numeric(x, errors="coerce")
    return default if pd.isna(v) else float(v)

def _safe_minmax(lo, hi) -> Tuple[float, float]:
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else (0.0, 1.0)

def robust_range(values, lo_q: float = 0.05, hi_q: float = 0.95) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0, 1.0)
    lo = float(np.quantile(arr, lo_q))
    hi = float(np.quantile(arr, hi_q))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return (0.0, 1.0)
    return (lo, hi)

def robust_norm(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x) or hi <= lo:
        return 0.0
    v = (float(x) - lo) / (hi - lo)
    return float(np.clip(v, 0.0, 1.0))

def _norm_header(s: str) -> str:
    s = str(s or "")
    s = s.replace("\ufeff", "").replace("\u200b", "").replace("\xa0", " ")
    s = s.strip().lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def find_column_index(header: List[str], possible_names: List[str]) -> int:
    hdr = [_norm_header(h) for h in header]
    cand = [_norm_header(c) for c in possible_names]
    for i, h in enumerate(hdr):
        if h in cand:
            return i
    for i, h in enumerate(hdr):
        for c in cand:
            tokens = c.split()
            if all(t in h for t in tokens):
                return i
    return -1

def normalize_url(u: str) -> str:
    try:
        s = str(u or "").strip()
        if not s:
            return ""
        if not s.lower().startswith(("http://", "https://")):
            s = "https://" + s
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
        p = urlparse(s)
        fragment = ""
        qs = [
            (k, v)
            for (k, v) in parse_qsl(p.query, keep_blank_values=True)
            if k.lower()
            not in {
                "utm_source","utm_medium","utm_campaign","utm_term","utm_content",
                "gclid","fbclid","mc_cid","mc_eid","pk_campaign","pk_kwd",
            }
        ]
        qs.sort(key=lambda kv: kv[0])
        query = urlencode(qs)
        hostname = (p.hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        path = p.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        rebuilt = urlunparse(
            (p.scheme, hostname + (":" + str(p.port) if p.port else ""), path, "", query, fragment)
        )
        return rebuilt
    except Exception:
        return str(u or "").strip()

def is_content_position(position_raw) -> bool:
    pos_norm = str(position_raw or "").strip().lower().replace("\xa0", " ")
    return any(
        token in pos_norm for token in ["inhalt", "content", "body", "main", "artikel", "article", "copy", "text", "editorial"]
    )

# Anzeige-Originale
if "_ORIG_MAP" not in st.session_state:
    st.session_state["_ORIG_MAP"] = {}

def remember_original(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    key = normalize_url(s)
    if not key:
        return ""
    prev = st.session_state["_ORIG_MAP"].get(key)
    if prev is None or (not prev.endswith("/") and s.endswith("/")):
        st.session_state["_ORIG_MAP"][key] = s
    return key

def disp(key_or_url: str) -> str:
    return st.session_state["_ORIG_MAP"].get(str(key_or_url), str(key_or_url))

# Embedding-Parser
def parse_vec(x) -> Optional[np.ndarray]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)
    s = str(x).strip()
    if not s:
        return None
    if s.startswith("[") and s.endswith("]"):
        try:
            return np.asarray(json.loads(s), dtype=float)
        except Exception:
            pass
    s_clean = re.sub(r"[\[\]]", "", s)
    parts = [p for p in re.split(r"[,\s;|]+", s_clean) if p]
    try:
        vec = np.asarray([float(p) for p in parts], dtype=float)
        return vec if vec.size > 0 else None
    except Exception:
        return None

# Datei-Leser (CSV/XLSX)
def read_any_file(f) -> Optional[pd.DataFrame]:
    if f is None:
        return None
    name = (getattr(f, "name", "") or "").lower()
    try:
        if name.endswith(".csv"):
            for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
                try:
                    f.seek(0)
                    return pd.read_csv(f, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
                except UnicodeDecodeError:
                    continue
                except Exception:
                    for sep_try in [";", ",", "\t"]:
                        try:
                            f.seek(0)
                            return pd.read_csv(f, sep=sep_try, engine="python", encoding=enc, on_bad_lines="skip")
                        except Exception:
                            continue
            raise ValueError("Kein passendes Encoding/Trennzeichen gefunden.")
        else:
            f.seek(0)
            return pd.read_excel(f)
    except Exception as e:
        st.error(f"Fehler beim Lesen von {getattr(f, 'name', 'Datei')}: {e}")
        return None

# Related aus Embeddings berechnen (FAISS/NumPy)
def build_related_from_embeddings(urls: List[str], V: np.ndarray, top_k: int, sim_threshold: float, backend: str) -> pd.DataFrame:
    n = V.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["Ziel", "Quelle", "Similarity"])
    K = int(top_k)
    pairs: List[List[object]] = []

    if backend == "Schnell (FAISS)":
        try:
            import faiss  # type: ignore
            dim = V.shape[1]
            index = faiss.IndexFlatIP(dim)
            Vf = V.astype("float32", copy=False)
            index.add(Vf)
            topk = min(K + 1, n)
            D, I = index.search(Vf, topk)
            for i in range(n):
                taken = 0
                for rank, j in enumerate(I[i]):
                    if j == -1 or j == i:
                        continue
                    s = float(D[i][rank])
                    if s < sim_threshold:
                        continue
                    pairs.append([urls[i], urls[j], s])
                    taken += 1
                    if taken >= K:
                        break
        except Exception:
            backend = "Exakt (NumPy)"

    if backend == "Exakt (NumPy)":
        Vf = V.astype(np.float32, copy=False)
        row_chunk = 512
        col_chunk = 2048
        for r0 in range(0, n, row_chunk):
            r1 = min(r0 + row_chunk, n)
            block = Vf[r0:r1]
            b = r1 - r0
            top_vals = np.full((b, K), -np.inf, dtype=np.float32)
            top_idx  = np.full((b, K), -1,      dtype=np.int32)
            for c0 in range(0, n, col_chunk):
                c1 = min(c0 + col_chunk, n)
                part = Vf[c0:c1]
                sims = block @ part.T
                if (c0 < r1) and (c1 > r0):
                    o0 = max(r0, c0); o1 = min(r1, c1)
                    br = np.arange(o0 - r0, o1 - r0)
                    bc = np.arange(o0 - c0, o1 - c0)
                    sims[br, bc] = -1.0
                if K < sims.shape[1]:
                    part_idx = np.argpartition(sims, -K, axis=1)[:, -K:]
                else:
                    part_idx = np.argsort(sims, axis=1)
                rows = np.arange(b)[:, None]
                cand_vals = sims[rows, part_idx]
                cand_idx  = part_idx + c0
                all_vals = np.concatenate([top_vals, cand_vals], axis=1)
                all_idx  = np.concatenate([top_idx,  cand_idx],  axis=1)
                sel = np.argpartition(all_vals, -K, axis=1)[:, -K:]
                top_vals = all_vals[rows, sel]
                top_idx  = all_idx[rows,  sel]
                order = np.argsort(top_vals, axis=1)[:, ::-1]
                top_vals = top_vals[rows, order]
                top_idx  = top_idx[rows,  order]
            for bi, i in enumerate(range(r0, r1)):
                taken = 0
                for v, j in zip(top_vals[bi], top_idx[bi]):
                    s = float(v)
                    if j == -1 or s < sim_threshold:
                        continue
                    pairs.append([urls[i], urls[int(j)], s])
                    taken += 1
                    if taken >= K:
                        break
    if not pairs:
        return pd.DataFrame(columns=["Ziel", "Quelle", "Similarity"])
    return pd.DataFrame(pairs, columns=["Ziel", "Quelle", "Similarity"])

@st.cache_data(show_spinner=False)
def read_any_file_cached(filename: str, raw: bytes) -> Optional[pd.DataFrame]:
    from io import BytesIO
    if not raw:
        return None
    name = (filename or "").lower()
    try:
        if name.endswith(".csv"):
            for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
                try:
                    return pd.read_csv(BytesIO(raw), sep=None, engine="python", encoding=enc, on_bad_lines="skip")
                except UnicodeDecodeError:
                    continue
                except Exception:
                    for sep_try in [";", ",", "\t"]:
                        try:
                            return pd.read_csv(BytesIO(raw), sep=sep_try, engine="python", encoding=enc, on_bad_lines="skip")
                        except Exception:
                            continue
            raise ValueError("Kein passendes Encoding/Trennzeichen gefunden.")
        else:
            return pd.read_excel(BytesIO(raw))
    except Exception as e:
        st.error(f"Fehler beim Lesen von {filename or 'Datei'}: {e}")
        return None

def _faiss_available() -> bool:
    try:
        import faiss  # type: ignore
        return True
    except Exception:
        return False

def _numpy_footprint_gb(n: int) -> float:
    return (n * n * 8) / (1024**3)

def choose_backend(prefer: str, n_items: int, mem_budget_gb: float = 1.5) -> Tuple[str, str]:
    faiss_ok = _faiss_available()
    if prefer == "Schnell (FAISS)":
        if faiss_ok:
            return "Schnell (FAISS)", "FAISS gewählt"
        else:
            return "Exakt (NumPy)", "FAISS nicht installiert → NumPy"
    est = _numpy_footprint_gb(max(0, int(n_items)))
    if est > mem_budget_gb and faiss_ok:
        return "Schnell (FAISS)", f"NumPy-Schätzung {est:.2f} GB > Budget {mem_budget_gb:.2f} GB → FAISS"
    return "Exakt (NumPy)", "NumPy innerhalb Budget"

@st.cache_data(show_spinner=False)
def build_related_cached(
    urls: tuple, V: np.ndarray, top_k: int, sim_threshold: float, backend: str, _v: int = 1,
) -> pd.DataFrame:
    Vf = V.astype("float32", copy=False)
    return build_related_from_embeddings(list(urls), Vf, top_k, sim_threshold, backend)

def build_related_auto(urls: List[str], V: np.ndarray, top_k: int, sim_threshold: float, prefer_backend: str, mem_budget_gb: float = 1.5) -> pd.DataFrame:
    n = int(V.shape[0])
    eff_backend, reason = choose_backend(prefer_backend, n, mem_budget_gb)
    if eff_backend != prefer_backend:
        st.info(f"Backend auf **{eff_backend}** umgestellt ({reason}).")
    try:
        return build_related_cached(tuple(urls), V.astype("float32", copy=False), int(top_k), float(sim_threshold), eff_backend, _v=1)
    except MemoryError:
        if eff_backend == "Exakt (NumPy)" and _faiss_available():
            st.warning("NumPy ist am Speicherlimit – Wechsel auf **FAISS**.")
            return build_related_cached(tuple(urls), V, int(top_k), float(sim_threshold), "Schnell (FAISS)", _v=1)
        raise
    except Exception as e:
        if eff_backend == "Schnell (FAISS)":
            st.warning(f"FAISS-Indexierung fehlgeschlagen ({e}). Fallback auf **NumPy**.")
            return build_related_cached(tuple(urls), V, int(top_k), float(sim_threshold), "Exakt (NumPy)", _v=1)
        else:
            raise

# =============================
# Hilfe / Tool-Dokumentation (Expander) – aktualisiert
# =============================
with st.expander("❓ Hilfe / Tool-Dokumentation", expanded=False):
    st.markdown("""
## Was macht ONE Link Intelligence?

**ONE Link Intelligence** bietet vier Analysen, die deine interne Verlinkung datengetrieben verbessern:

1) **Interne Verlinkungsmöglichkeiten (Analyse 1)**  
   - Findet semantisch passende interne Links (auf Basis von Embeddings oder bereitgestellter „Related URLs“).
   - Zeigt bestehende (Content-)Links und bewertet Linkgeber nach **Linkpotenzial**.

2) **Potenziell zu entfernende Links (Analyse 2)**  
   - Identifiziert schwache/unpassende Links (niedrige semantische Ähnlichkeit, Waster-Score).
   - Optional: Offpage-Dämpfung (Backlinks/Ref. Domains) und Schwellwerte.

3) **Gems & SEO-Potenziallinks (Analyse 3)**  
   - Ermittelt starke Linkgeber („Gems“) anhand des Linkpotenzials.
   - Priorisiert Ziele nach **Linkbedarf (PRIO)**: Hidden Champions, Semantische Linklücke, Sprungbrett-URLs, Mauerblümchen.
   - Ergebnis: „Cheat-Sheet“ mit wertvollen, noch nicht gesetzten Content-Links.

4) **Anchor & Query Intelligence (Analyse 4)** *(optional)*  
   - **Over-Anchor-Check:** Listet **alle Anchors ≥ 200** Vorkommen je Ziel-URL (inkl. Bild-Links via ALT).
   - **GSC-Query-Coverage (Top-20 % je URL):** Prüft, ob Top-Suchanfragen (nach Klicks/Impr) als Anker vorkommen — **Exact** und/oder **Embedding** (Cosine-Schwelle einstellbar).  
     – **Brand-Handling:** Brand-Begriffe via Text/CSV; Modus „nur Non-Brand“, „nur Brand“ oder „beides“. Optional **Auto-Varianten** (z. B. „bora kochfeld“) über editierbare Nomenliste.  
   - **Keyword-Zielvorgaben:** Upload `URL + Keyword-Spalten` (Spaltenname enthält `keyword|suchanfrage|suchbegriff|query`). Prüft pro URL, ob ihre Ziel-Keywords als Anchor vorkommen (**Exact**/**Embedding**).  
   - **Leader-Konflikte:** Für jede Query die **Leader-URL** (höchste Klicks/Impr) ermitteln. Flag, wenn semantisch passende Anchors **auf eine andere URL** verlinken (nicht navigative Anchors werden bewertet).  
   - **UI & Export:** Oberfläche zeigt **nur Problemfälle**. **Ein Download** liefert alle Reports als **Excel mit Tabs** (bei großen Daten automatisch **ZIP mit CSVs**).

### Inputs & Formate
- **URLs + Embeddings** *oder* **Related URLs**
- **All Inlinks** (*Anchor Text*, optional *ALT*, *Linkposition*)
- **Linkmetriken** (URL, Score, Inlinks, Outlinks)
- **Backlinks** (URL, Backlinks, Ref. Domains)
- **Search Console** (optional für Analysen 3 & 4): URL/Page, Query, Clicks, Impressions
- **Keyword-Zielvorgaben** (optional, Analyse 4): `URL` + beliebige `Keyword`-Spalten
""")

# ===============================
# Sidebar Controls
# ===============================
with st.sidebar:
    st.header("Einstellungen")

    backend = st.radio(
        "Matching-Backend (Auto-Switch bei Bedarf)",
        ["Exakt (NumPy)", "Schnell (FAISS)"],
        index=0,
        horizontal=True,
        help=("Bestimmt, wie semantische Nachbarn ermittelt werden (Cosine Similarity). "
              "Mit **Auto-Switch**: Wenn NumPy voraussichtlich zu viel RAM braucht, "
              "oder FAISS nicht verfügbar ist, wird automatisch umgeschaltet.")
    )
    if not _faiss_available():
        st.caption("FAISS ist hier nicht installiert – Auto-Switch nutzt ggf. NumPy.")

    st.subheader("Gewichtung (Linkpotenzial)")
    st.caption("Das Linkpotenzial gibt Aufschluss über die Lukrativität einer **URL** als Linkgeber.")
    w_ils = st.slider("Interner Link Score", 0.0, 1.0, 0.30, 0.01)
    w_pr  = st.slider("PageRank-Horder-Score", 0.0, 1.0, 0.35, 0.01)
    w_rd  = st.slider("Referring Domains", 0.0, 1.0, 0.20, 0.01)
    w_bl  = st.slider("Backlinks", 0.0, 1.0, 0.15, 0.01)
    w_sum = w_ils + w_pr + w_rd + w_bl
    if not math.isclose(w_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        st.warning(f"Gewichtungs-Summe = {w_sum:.2f} (sollte 1.0 sein)")

    st.subheader("Schwellenwerte & Limits (Related URLs Ermittlung)")
    sim_threshold = st.slider("Ähnlichkeitsschwelle", 0.0, 1.0, 0.80, 0.01)
    max_related   = st.number_input("Anzahl Related URLs", min_value=1, max_value=50, value=10, step=1)

    st.subheader("Entfernung von Links")
    not_similar_threshold = st.slider("Unähnlichkeitsschwelle (schwache Links)", 0.0, 1.0, 0.60, 0.01)
    backlink_weight_2x = st.checkbox("Backlinks/Ref. Domains doppelt gewichten", value=False)

# Styling
CSS_ACTION_BUTTONS = """
<style>
:root { --one-red: #e02424; --one-red-hover: #c81e1e; }
div.stButton > button[kind="secondary"] {
  background-color: var(--one-red) !important; color: #fff !important; border: 1px solid var(--one-red) !important; border-radius: 6px !important;
}
div.stButton > button[kind="secondary"]:hover {
  background-color: var(--one-red-hover) !important; border-color: var(--one-red-hover) !important;
}
div.stDownloadButton > button, div.stDownloadButton > a {
  background-color: var(--one-red) !important; color: #fff !important; border: 1px solid var(--one-red) !important; border-radius: 6px !important; text-decoration: none !important;
}
div.stDownloadButton > button:hover, div.stDownloadButton > a:hover {
  background-color: var(--one-red-hover) !important; border-color: var(--one-red-hover) !important; color: #fff !important;
}
</style>
"""
st.markdown(CSS_ACTION_BUTTONS, unsafe_allow_html=True)
st.markdown("<style>div[data-testid='stContainer'] > div:has(> .stSlider) { padding-bottom: .25rem; }</style>", unsafe_allow_html=True)

# ===============================
# Data ingestion
# ===============================
st.markdown("---")
st.subheader("Daten laden")

mode = st.radio(
    "Eingabemodus",
    ["URLs + Embeddings", "Related URLs"],
    horizontal=True,
    help="Entweder Embeddings hochladen (App berechnet 'Related URLs') oder bereits vorliegende 'Related URLs' nutzen.",
)

related_df = inlinks_df = metrics_df = backlinks_df = None
emb_df = None

if mode == "URLs + Embeddings":
    st.write("Lade **URL + Embedding** sowie **All Inlinks**, **Linkmetriken**, **Backlinks**.")
    up_emb = st.file_uploader("URLs + Embeddings (CSV/Excel)", type=["csv","xlsx","xlsm","xls"], key="embs")
    col1, col2 = st.columns(2)
    with col1:
        inlinks_up = st.file_uploader("All Inlinks (CSV/Excel)", type=["csv","xlsx","xlsm","xls"], key="inl2")
        metrics_up = st.file_uploader("Linkmetriken (CSV/Excel)", type=["csv","xlsx","xlsm","xls"], key="met2")
    with col2:
        backlinks_up = st.file_uploader("Backlinks (CSV/Excel)", type=["csv","xlsx","xlsm","xls"], key="bl2")
    emb_df       = read_any_file_cached(getattr(up_emb,       "name", ""), up_emb.getvalue())         if up_emb       else None
    inlinks_df   = read_any_file_cached(getattr(inlinks_up,   "name", ""), inlinks_up.getvalue())     if inlinks_up   else None
    metrics_df   = read_any_file_cached(getattr(metrics_up,   "name", ""), metrics_up.getvalue())     if metrics_up   else None
    backlinks_df = read_any_file_cached(getattr(backlinks_up, "name", ""), backlinks_up.getvalue())   if backlinks_up else None

elif mode == "Related URLs":
    st.write("Lade **Related URLs**, **All Inlinks**, **Linkmetriken**, **Backlinks**.")
    col1, col2 = st.columns(2)
    with col1:
        related_up = st.file_uploader("Related URLs (CSV/Excel)", type=["csv","xlsx","xlsm","xls"], key="rel")
        metrics_up = st.file_uploader("Linkmetriken (CSV/Excel)", type=["csv","xlsx","xlsm","xls"], key="met")
    with col2:
        inlinks_up = st.file_uploader("All Inlinks (CSV/Excel)", type=["csv","xlsx","xlsm","xls"], key="inl")
        backlinks_up = st.file_uploader("Backlinks (CSV/Excel)", type=["csv","xlsx","xlsm","xls"], key="bl")
    related_df   = read_any_file_cached(getattr(related_up,  "name", ""), related_up.getvalue())    if related_up  else None
    inlinks_df   = read_any_file_cached(getattr(inlinks_up,  "name", ""), inlinks_up.getvalue())    if inlinks_up  else None
    metrics_df   = read_any_file_cached(getattr(metrics_up,  "name", ""), metrics_up.getvalue())    if metrics_up  else None
    backlinks_df = read_any_file_cached(getattr(backlinks_up,"name", ""), backlinks_up.getvalue())  if backlinks_up else None

# Let's Go
run_clicked = st.button("Let's Go", type="secondary")
if not run_clicked and not st.session_state.ready:
    st.info("Bitte Dateien hochladen und auf **Let's Go** klicken, um die Analysen zu starten.")
    st.stop()

if run_clicked:
    placeholder = st.empty()
    with placeholder.container():
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.image("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDJweGExcHhhOWZneTZwcnAxZ211OWJienY5cWQ1YmpwaHR0MzlydiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/dBRaPog8yxFWU/giphy.gif", width=280)
            st.caption("Die Berechnungen laufen – Zeit für eine kleine Stärkung …")

# Validierung & ggf. Related aus Embeddings
if run_clicked or st.session_state.ready:
    if mode == "URLs + Embeddings":
        if emb_df is None or any(df is None for df in [inlinks_df, metrics_df, backlinks_df]):
            st.error("Bitte alle benötigten Dateien hochladen (Embeddings, All Inlinks, Linkmetriken, Backlinks).")
            st.stop()
        if emb_df is not None and not emb_df.empty:
            hdr = [_norm_header(c) for c in emb_df.columns]
            def pick_col(candidates: list, default=None):
                cand_norm = [_norm_header(c) for c in candidates]
                for cand in cand_norm:
                    for i, h in enumerate(hdr):
                        if h == cand:
                            return i
                for cand in cand_norm:
                    toks = cand.split()
                    for i, h in enumerate(hdr):
                        if all(t in h for t in toks):
                            return i
                for cand in cand_norm:
                    for i, h in enumerate(hdr):
                        if cand in h:
                            return i
                for i, h in enumerate(hdr):
                    if re.search(r"\bembed(ding)?s?\b", h):
                        return i
                for i, h in enumerate(hdr):
                    if re.search(r"\bvec(tor)?\b", h) and "url" not in h:
                        return i
                return default
            url_i = pick_col(["url","urls","page","seite","address","adresse","landingpage","landing page"], 0)
            emb_i = pick_col(["embedding","embeddings","embedding json","embedding_json","text embedding","openai embedding","sentence embedding","vector","vec","content embedding","sf embedding","embedding 1536","embedding_1536"], 1 if emb_df.shape[1] >= 2 else None)
            url_col = emb_df.columns[url_i] if url_i is not None else emb_df.columns[0]
            emb_col = emb_df.columns[emb_i] if emb_i is not None else (emb_df.columns[1] if emb_df.shape[1] >= 2 else emb_df.columns[0])

            urls: List[str] = []
            vecs: List[np.ndarray] = []
            for _, r in emb_df.iterrows():
                nkey = remember_original(r[url_col])
                v = parse_vec(r[emb_col])
                if not nkey or v is None:
                    continue
                urls.append(nkey)
                vecs.append(v)
            if len(vecs) < 2:
                st.error("Zu wenige gültige Embeddings erkannt (mindestens 2 benötigt).")
                st.stop()

            dims = [v.size for v in vecs]
            max_dim = max(dims)
            V = np.zeros((len(vecs), max_dim), dtype=np.float32)
            shorter = sum(1 for d in dims if d < max_dim)
            for i, v in enumerate(vecs):
                d = min(max_dim, v.size)
                V[i, :d] = v[:d]
            norms = np.linalg.norm(V, axis=1, keepdims=True).astype(np.float32, copy=False)
            norms[norms == 0] = 1.0
            V = V / norms
            V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
            if shorter > 0:
                st.caption(f"⚠️ {shorter} Embeddings hatten geringere Dimensionen und wurden auf {max_dim} gepaddet.")
                with st.expander("Was bedeutet das ‘Padden’ der Embeddings?"):
                    st.markdown(f"""
- Einige Embeddings sind kürzer als die Ziel-Dimension (**{max_dim}**). Diese werden mit `0` aufgefüllt.
- Nach L2-Normierung funktionieren Cosine-Ähnlichkeiten wie gewohnt, sofern alle Embeddings aus **demselben Modell** stammen.
- Empfehlung: alle Embeddings mit demselben Modell erzeugen.
""")
            related_df = build_related_auto(list(urls), V, int(max_related), float(sim_threshold), backend, mem_budget_gb=1.5)
            try:
                if isinstance(urls, list) and isinstance(V, np.ndarray) and V.size > 0:
                    st.session_state["_emb_urls"] = list(urls)
                    st.session_state["_emb_matrix"] = V.astype("float32", copy=False)
                    st.session_state["_emb_index_by_url"] = {u: i for i, u in enumerate(urls)}
                else:
                    st.session_state.pop("_emb_urls", None)
                    st.session_state.pop("_emb_matrix", None)
                    st.session_state.pop("_emb_index_by_url", None)
            except Exception:
                pass

# Prüfen, ob alles da ist
have_all = all(df is not None for df in [related_df, inlinks_df, metrics_df, backlinks_df])
if not have_all:
    st.error("Bitte alle benötigten Tabellen bereitstellen.")
    st.stop()

# ===============================
# Normalization maps / data prep
# ===============================
# Linkmetriken
metrics_df = metrics_df.copy()
metrics_df.columns = [str(c).strip() for c in metrics_df.columns]
m_header = [str(c).strip() for c in metrics_df.columns]
m_url_idx   = find_column_index(m_header, ["url","urls","page","seite","address","adresse"])
m_score_idx = find_column_index(m_header, ["score","interner link score","internal link score","ils","link score","linkscore"])
m_in_idx    = find_column_index(m_header, ["inlinks","in links","interne inlinks","eingehende links","einzigartige inlinks","unique inlinks","inbound internal links"])
m_out_idx   = find_column_index(m_header, ["outlinks","out links","ausgehende links","interne outlinks","unique outlinks","einzigartige outlinks","outbound links"])
if -1 in (m_url_idx, m_score_idx, m_in_idx, m_out_idx):
    if metrics_df.shape[1] >= 4:
        if m_url_idx   == -1: m_url_idx   = 0
        if m_score_idx == -1: m_score_idx = 1
        if m_in_idx    == -1: m_in_idx    = 2
        if m_out_idx   == -1: m_out_idx   = 3
        st.warning("Linkmetriken: Header nicht vollständig erkannt – Fallback auf Spaltenpositionen (1–4).")
    else:
        st.error("'Linkmetriken' braucht mindestens 4 Spalten (URL, Score, Inlinks, Outlinks).")
        st.stop()
metrics_df.iloc[:, m_url_idx] = metrics_df.iloc[:, m_url_idx].astype(str)
metrics_map: Dict[str, Dict[str, float]] = {}
for _, r in metrics_df.iterrows():
    u = remember_original(r.iloc[m_url_idx])
    if not u:
        continue
    score    = _num(r.iloc[m_score_idx])
    inlinks  = _num(r.iloc[m_in_idx])
    outlinks = _num(r.iloc[m_out_idx])
    prdiff   = inlinks - outlinks
    metrics_map[u] = {"score": score, "prDiff": prdiff}

# Backlinks
backlinks_df = backlinks_df.copy()
backlinks_df.columns = [str(c).strip() for c in backlinks_df.columns]
b_header = [str(c).strip() for c in backlinks_df.columns]
b_url_idx = find_column_index(b_header, ["url","urls","page","seite","address","adresse"])
b_bl_idx  = find_column_index(b_header, ["backlinks","backlink","external backlinks","back links","anzahl backlinks","backlinks total"])
b_rd_idx  = find_column_index(b_header, ["referring domains","ref domains","verweisende domains","anzahl referring domains","anzahl verweisende domains","domains","rd"])
if -1 in (b_url_idx, b_bl_idx, b_rd_idx):
    if backlinks_df.shape[1] >= 3:
        if b_url_idx == -1: b_url_idx = 0
        if b_bl_idx  == -1: b_bl_idx  = 1
        if b_rd_idx  == -1: b_rd_idx  = 2
        st.warning("Backlinks: Header nicht vollständig erkannt – Fallback auf Spaltenpositionen (1–3).")
    else:
        st.error("'Backlinks' braucht mindestens 3 Spalten (URL, Backlinks, Referring Domains).")
        st.stop()
backlink_map: Dict[str, Dict[str, float]] = {}
for _, r in backlinks_df.iterrows():
    u = remember_original(r.iloc[b_url_idx])
    if not u:
        continue
    bl = _num(r.iloc[b_bl_idx])
    rd = _num(r.iloc[b_rd_idx])
    backlink_map[u] = {"backlinks": bl, "referringDomains": rd}

# Ranges
ils_vals = [m["score"] for m in metrics_map.values()]
prd_vals = [m["prDiff"] for m in metrics_map.values()]
bl_vals  = [b["backlinks"] for b in backlink_map.values()]
rd_vals  = [b["referringDomains"] for b in backlink_map.values()]
min_ils, max_ils = robust_range(ils_vals, 0.05, 0.95)
min_prd, max_prd = robust_range(prd_vals, 0.05, 0.95)
min_bl,  max_bl  = robust_range(bl_vals,  0.05, 0.95)
min_rd,  max_rd  = robust_range(rd_vals,  0.05, 0.95)
bl_log_vals = [float(np.log1p(max(0.0, v))) for v in bl_vals]
rd_log_vals = [float(np.log1p(max(0.0, v))) for v in rd_vals]
lo_bl_log, hi_bl_log = robust_range(bl_log_vals, 0.05, 0.95)
lo_rd_log, hi_rd_log = robust_range(rd_log_vals, 0.05, 0.95)

# Inlinks lesen (Quelle/Ziel/Position)
inlinks_df = inlinks_df.copy()
header = [str(c).strip() for c in inlinks_df.columns]
src_idx = find_column_index(header, POSSIBLE_SOURCE)
dst_idx = find_column_index(header, POSSIBLE_TARGET)
pos_idx = find_column_index(header, POSSIBLE_POSITION)
if src_idx == -1 or dst_idx == -1:
    st.error("In 'All Inlinks' wurden die Spalten 'Quelle/Source' oder 'Ziel/Destination' nicht gefunden.")
    st.stop()

# Anchor/ALT Spalten (für Analyse 4)
anchor_idx = find_column_index(header, POSSIBLE_ANCHOR)
alt_idx    = find_column_index(header, POSSIBLE_ALT)

all_links: set[Tuple[str, str]] = set()
content_links: set[Tuple[str, str]] = set()
for row in inlinks_df.itertuples(index=False, name=None):
    source = remember_original(row[src_idx])
    target = remember_original(row[dst_idx])
    if not source or not target:
        continue
    key = (source, target)
    all_links.add(key)
    if pos_idx != -1 and is_content_position(row[pos_idx]):
        content_links.add(key)
st.session_state["_all_links"] = all_links
st.session_state["_content_links"] = content_links

# Related map (beidseitig, thresholded)
related_df = related_df.copy()
if related_df.shape[1] < 3:
    st.error("'Related URLs' braucht mindestens 3 Spalten.")
    st.stop()
rel_header = [str(c).strip() for c in related_df.columns]
rel_dst_idx = find_column_index(rel_header, POSSIBLE_TARGET)
rel_src_idx = find_column_index(rel_header, POSSIBLE_SOURCE)
rel_sim_idx = find_column_index(rel_header, ["similarity","similarität","ähnlichkeit","cosine","cosine similarity","semantische ähnlichkeit","sim"])
if -1 in (rel_dst_idx, rel_src_idx, rel_sim_idx):
    if related_df.shape[1] >= 3:
        if rel_dst_idx == -1: rel_dst_idx = 0
        if rel_src_idx == -1: rel_src_idx = 1
        if rel_sim_idx == -1: rel_sim_idx = 2
        st.warning("Related URLs: Header nicht vollständig erkannt – Fallback auf Spaltenpositionen (1–3).")
    else:
        st.error("'Related URLs' braucht mindestens 3 Spalten (Ziel, Quelle, Similarity).")
        st.stop()

related_map: Dict[str, List[Tuple[str, float]]] = {}
processed_pairs = set()
for _, r in related_df.iterrows():
    urlA = remember_original(r.iloc[rel_dst_idx])   # Ziel
    urlB = remember_original(r.iloc[rel_src_idx])   # Quelle
    try:
        sim = float(str(r.iloc[rel_sim_idx]).replace(",", "."))
    except Exception:
        sim = np.nan
    if not urlA or not urlB or np.isnan(sim):
        continue
    if sim < sim_threshold:
        continue
    pair_key = "↔".join(sorted([urlA, urlB]))
    if pair_key in processed_pairs:
        continue
    related_map.setdefault(urlA, []).append((urlB, sim))
    related_map.setdefault(urlB, []).append((urlA, sim))
    processed_pairs.add(pair_key)

# Linkpotenzial (Quelle)
source_potential_map: Dict[str, float] = {}
for u, m in metrics_map.items():
    ils_raw = _num(m.get("score"))
    pr_raw  = _num(m.get("prDiff"))
    bl      = backlink_map.get(u, {"backlinks": 0.0, "referringDomains": 0.0})
    bl_raw  = _num(bl.get("backlinks"))
    rd_raw  = _num(bl.get("referringDomains"))
    norm_ils = robust_norm(ils_raw, min_ils, max_ils)
    norm_pr  = robust_norm(pr_raw,  min_prd, max_prd)
    norm_bl  = robust_norm(bl_raw,  min_bl,  max_bl)
    norm_rd  = robust_norm(rd_raw,  min_rd,  max_rd)
    final_score = (w_ils * norm_ils) + (w_pr * norm_pr) + (w_bl * norm_bl) + (w_rd * norm_rd)
    source_potential_map[u] = round(final_score, 4)

st.session_state["_source_potential_map"] = source_potential_map
st.session_state["_metrics_map"] = metrics_map
st.session_state["_backlink_map"] = backlink_map
st.session_state["_norm_ranges"] = {
    "ils": (min_ils, max_ils),
    "prd": (min_prd, max_prd),
    "bl":  (min_bl,  max_bl),
    "rd":  (min_rd,  max_rd),
    "bl_log": (lo_bl_log, hi_bl_log),
    "rd_log": (lo_rd_log, hi_rd_log),
}

# ===============================
# Analyse 1
# ===============================
st.markdown("## Analyse 1: Interne Verlinkungsmöglichkeiten")
st.caption("Diese Analyse schlägt thematisch passende interne Verlinkungen vor, zeigt bestehende (Content-)Links und bewertet das Linkpotenzial der Linkgeber.")
if not st.session_state.get("__gems_loading__", False):
    cols = ["Ziel-URL"]
    for i in range(1, int(max_related) + 1):
        cols += [
            f"Related URL {i}",
            f"Ähnlichkeit {i}",
            f"Link von Related URL {i} auf Ziel-URL bereits vorhanden?",
            f"Link von Related URL {i} auf Ziel-URL aus Inhalt heraus vorhanden?",
            f"Linkpotenzial Related URL {i}",
        ]
    rows_norm, rows_view = [], []
    for target, related_list in sorted(related_map.items()):
        related_sorted = sorted(related_list, key=lambda x: x[1], reverse=True)[: int(max_related)]
        row_norm = [target]
        row_view = [disp(target)]
        for source, sim in related_sorted:
            anywhere = "ja" if (source, target) in all_links else "nein"
            from_content = "ja" if (source, target) in content_links else "nein"
            final_score = source_potential_map.get(source, 0.0)
            row_norm.extend([source, round(float(sim), 3), anywhere, from_content, final_score])
            row_view.extend([disp(source), round(float(sim), 3), anywhere, from_content, final_score])
        while len(row_norm) < len(cols):
            row_norm.append(np.nan)
        while len(row_view) < len(cols):
            row_view.append(np.nan)
        rows_norm.append(row_norm)
        rows_view.append(row_view)
    res1_df = pd.DataFrame(rows_norm, columns=cols)
    st.session_state.res1_df = res1_df

    long_rows = []
    max_i = int(max_related)
    for _, r in res1_df.iterrows():
        ziel = r["Ziel-URL"]
        for i in range(1, max_i + 1):
            col_src = f"Related URL {i}"
            col_sim = f"Ähnlichkeit {i}"
            col_any = f"Link von Related URL {i} auf Ziel-URL bereits vorhanden?"
            col_con = f"Link von Related URL {i} auf Ziel-URL aus Inhalt heraus vorhanden?"
            col_pot = f"Linkpotenzial Related URL {i}"
            if col_src not in res1_df.columns:
                break
            src = r.get(col_src, "")
            if not isinstance(src, str) or not src:
                continue
            sim = r.get(col_sim, np.nan)
            anywhere = r.get(col_any, "nein")
            from_content = r.get(col_con, "nein")
            pot = r.get(col_pot, 0.0)
            long_rows.append([
                disp(ziel),
                disp(src),
                round(float(sim), 3) if pd.notna(sim) else np.nan,
                anywhere,
                from_content,
                float(pot),
            ])
    res1_view_long = pd.DataFrame(long_rows, columns=[
        "Ziel-URL","Related URL","Ähnlichkeit (Cosinus Ähnlichkeit)",
        "Link von Related URL auf Ziel-URL vorhanden?","Link von Related URL auf Ziel-URL aus Inhalt heraus vorhanden?","Linkpotenzial",
    ])
    if not res1_view_long.empty:
        res1_view_long = res1_view_long.sort_values(
            by=["Ziel-URL", "Ähnlichkeit (Cosinus Ähnlichkeit)"],
            ascending=[True, False],
            kind="mergesort"
        ).reset_index(drop=True)
    st.dataframe(res1_view_long, use_container_width=True, hide_index=True)
    csv1 = res1_view_long.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download 'Interne Verlinkungsmöglichkeiten (Long-Format)' (CSV)",
        data=csv1, file_name="interne_verlinkungsmoeglichkeiten_long.csv", mime="text/csv", key="dl_interne_verlinkung_long",
    )

    # ===============================
    # Analyse 2
    # ===============================
    st.markdown("## Analyse 2: Potenziell zu entfernende Links")
    st.caption("Diese Analyse legt bestehende Links zwischen semantisch nicht stark verwandten URLs offen.")
    # Similarity-Map
    sim_map: Dict[Tuple[str, str], float] = {}
    processed_pairs2 = set()
    for _, r in related_df.iterrows():
        a = remember_original(r.iloc[rel_src_idx])
        b = remember_original(r.iloc[rel_dst_idx])
        try:
            sim = float(str(r.iloc[rel_sim_idx]).replace(",", "."))
        except Exception:
            continue
        if not a or not b:
            continue
        pair_key = "↔".join(sorted([a, b]))
        if pair_key in processed_pairs2:
            continue
        sim_map[(a, b)] = sim
        sim_map[(b, a)] = sim
        processed_pairs2.add(pair_key)

    # fehlende Similarities aus Embeddings
    _idx_map = st.session_state.get("_emb_index_by_url")
    _Vmat    = st.session_state.get("_emb_matrix")
    _has_emb = isinstance(_idx_map, dict) and isinstance(_Vmat, np.ndarray)

    if _has_emb:
        missing = [
            (src, dst) for (src, dst) in all_links
            if (src, dst) not in sim_map and (dst, src) not in sim_map
            and (_idx_map.get(src) is not None) and (_idx_map.get(dst) is not None)
        ]
        if missing:
            I = np.fromiter((_idx_map[src] for src, _ in missing), dtype=np.int32)
            J = np.fromiter((_idx_map[dst] for _, dst in missing), dtype=np.int32)
            Vf = _Vmat
            sims = np.einsum('ij,ij->i', Vf[I], Vf[J]).astype(float)
            for (src, dst), s in zip(missing, sims):
                val = float(s)
                sim_map[(src, dst)] = val
                sim_map[(dst, src)] = val

    # Waster
    raw_score_map: Dict[str, float] = {}
    for _, r in metrics_df.iterrows():
        u   = remember_original(r.iloc[m_url_idx])
        inl = _num(r.iloc[m_in_idx])
        outl= _num(r.iloc[m_out_idx])
        raw_score_map[u] = outl - inl

    adjusted_score_map: Dict[str, float] = {}
    for u, raw in raw_score_map.items():
        bl = backlink_map.get(u, {"backlinks": 0.0, "referringDomains": 0.0})
        impact = 0.5 * _num(bl.get("backlinks")) + 0.5 * _num(bl.get("referringDomains"))
        factor = 2.0 if backlink_weight_2x else 1.0
        malus = 5.0 * factor if impact == 0 else 0.0
        adjusted_score_map[u] = (raw or 0.0) - (factor * impact) + malus

    w_vals = np.asarray([v for v in adjusted_score_map.values() if np.isfinite(v)], dtype=float)
    if w_vals.size == 0:
        q70 = q90 = 0.0
    else:
        q70 = float(np.quantile(w_vals, 0.70))
        q90 = float(np.quantile(w_vals, 0.90))

    def waster_class_for(u: str) -> Tuple[str, float]:
        score = float(adjusted_score_map.get(u, 0.0))
        if score >= q90:
            return "hoch", score
        elif score >= q70:
            return "mittel", score
        else:
            return "niedrig", score

    out_rows = []
    rest_cols = [c for i, c in enumerate(header) if i not in (src_idx, dst_idx)]
    out_header = [
        "Quelle","Ziel","Waster-Klasse (Quelle)","Waster-Score (Quelle)","Semantische Ähnlichkeit",*rest_cols,
    ]
    for row in inlinks_df.itertuples(index=False, name=None):
        quelle = remember_original(row[src_idx])
        ziel   = remember_original(row[dst_idx])
        if not quelle or not ziel:
            continue
        w_class, w_score = waster_class_for(normalize_url(quelle))
        sim = sim_map.get((quelle, ziel), sim_map.get((ziel, quelle), np.nan))
        if not (isinstance(sim, (int, float)) and np.isfinite(sim)) and _has_emb:
            i = _idx_map.get(normalize_url(quelle)); j = _idx_map.get(normalize_url(ziel))
            if i is not None and j is not None:
                sim = float(np.dot(_Vmat[i], _Vmat[j]))
        if isinstance(sim, (int, float)) and np.isfinite(sim):
            sim_display = round(float(sim), 3)
            if float(sim) > float(not_similar_threshold):
                continue
        else:
            sim_display = "Cosine Similarity nicht erfasst"
        rest = [row[i] for i in range(len(header)) if i not in (src_idx, dst_idx)]
        out_rows.append([disp(quelle), disp(ziel), w_class, round(float(w_score), 3), sim_display, *rest])

    out_df = pd.DataFrame(out_rows, columns=out_header)
    st.session_state.out_df = out_df
    st.dataframe(out_df, use_container_width=True, hide_index=True)
    csv2 = out_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download 'Potenziell zu entfernende Links' (CSV)",
        data=csv2, file_name="potenziell_zu_entfernende_links.csv", mime="text/csv", key="dl_remove_candidates",
    )

    if run_clicked:
        try:
            placeholder.empty()
        except Exception:
            pass
        st.success("✅ Berechnung abgeschlossen!")
        st.session_state.ready = True

# =========================================================
# Analyse 3 (unverändert inhaltlich, lediglich hier belassen)
# =========================================================
st.markdown("---")
st.subheader("Analyse 3: Was sind starke Linkgeber („Gems“) & welche URLs diese verlinken sollten (⇒ SEO-Potenziallinks)")
st.caption("Diese Analyse identifiziert die aus SEO-Gesichtspunkten wertvollsten, aber noch nicht gesetzten, Content-Links.")

if st.session_state.get("__gems_loading__", False):
    ph3 = st.session_state.get("__gems_ph__")
    if ph3 is None:
        ph3 = st.empty()
        st.session_state["__gems_ph__"] = ph3
    with ph3.container():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            try:
                st.image("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExcnY0amo3NThxZnpnb3I4dDB6NWF2a2RkZm9uaXJ0bml1bG5lYm1mciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6HypNJJjcfnZ1bzWDs/giphy.gif", width=280)
            except Exception:
                st.write("⏳")
            st.caption("Analyse 3 läuft … Wir geben Gas – versprochen!")

with st.expander("Erklärung: Wie werden Gems & Zielseiten bestimmt?", expanded=False):
    st.markdown('''
**Kurz & klar:** Gems = Top-Linkgeber nach Linkpotenzial. Ziele = thematisch nahe URLs ohne Content-Link vom Gem.
PRIO setzt sich aus Hidden Champions (Impressions), Semantische Linklücke, Sprungbrett-URLs (Position) und Mauerblümchen zusammen.
''')

# Eingänge
res1_df: Optional[pd.DataFrame] = st.session_state.get("res1_df")
source_potential_map: Dict[str, float] = st.session_state.get("_source_potential_map", {})
metrics_map: Dict[str, Dict[str, float]] = st.session_state.get("_metrics_map", {})
norm_ranges: Dict[str, Tuple[float, float]] = st.session_state.get("_norm_ranges", {})
all_links: set = st.session_state.get("_all_links", set())

# Gems UI
gem_pct = st.slider("Anteil starker Linkgeber (Top-X %)", 1, 30, 10, step=1)
max_targets_per_gem = st.number_input("Top-Ziele je Gem", min_value=1, max_value=50, value=10, step=1)

# GSC Upload/Maps
gsc_up = st.file_uploader("Search Console Daten (CSV/Excel)", type=["csv","xlsx","xlsm","xls"], key="gsc_up_merged_no_opp")
gsc_df_loaded = None
demand_map: Dict[str, float] = {}
pos_map: Dict[str, float] = {}
has_gsc = False
has_pos = False
if gsc_up is not None:
    gsc_df_loaded = read_any_file_cached(getattr(gsc_up, "name", ""), gsc_up.getvalue())

if gsc_df_loaded is not None and not gsc_df_loaded.empty:
    has_gsc = True
    df = gsc_df_loaded.copy()
    df.columns = [str(c).strip() for c in df.columns]
    hdr = [_norm_header(c) for c in df.columns]
    def _find_idx(candidates: Iterable[str], default=None):
        cand_norm = {_norm_header(c) for c in candidates}
        for i, h in enumerate(hdr):
            if h in cand_norm:
                return i
        for i, h in enumerate(hdr):
            if any(c in h for c in cand_norm):
                return i
        return default
    url_idx  = _find_idx({"url","page","seite","address","adresse","landingpage","landing page"}, 0)
    impr_idx = _find_idx({"impressions","impr","search impressions","impressions_total",
                          "impressionen","impression","suchimpressionen"}, 1)
    click_idx = _find_idx({"clicks","klicks","click","klick"}, 2 if df.shape[1] >= 3 else None)
    pos_idx   = _find_idx({"position","avg position","average position","durchschnittliche position","durchschn. position"}, 3 if df.shape[1] >= 4 else None)
    df.iloc[:, url_idx] = df.iloc[:, url_idx].astype(str).map(normalize_url)
    urls_series = df.iloc[:, url_idx]
    impr = pd.to_numeric(df.iloc[:, impr_idx], errors="coerce").fillna(0)
    log_impr = np.log1p(impr)
    if (log_impr.max() - log_impr.min()) > 0:
        demand_norm = (log_impr - log_impr.min()) / (log_impr.max() - log_impr.min())
    else:
        demand_norm = np.zeros_like(log_impr)
    demand_map = {str(u): float(d) for u, d in zip(urls_series, demand_norm)}
    has_pos = False
    pos_map = {}
    if pos_idx is not None and pos_idx < df.shape[1]:
        pos_series = pd.to_numeric(df.iloc[:, pos_idx], errors="coerce")
        for u, p in zip(urls_series, pos_series):
            if pd.notna(p) and str(u):
                pos_map[str(u)] = float(p)
        has_pos = len(pos_map) > 0
    st.session_state["__gsc_df_raw__"] = df.copy()

st.markdown("#### Linkbedarf-Gewichtung für Zielseiten")
col1, col2, col3, col4 = st.columns(4)
with col1:
    with bordered_container():
        st.markdown("Gewicht: **Hidden Champions**")
        w_lihd = st.slider("", 0.0, 1.0, 0.30, 0.05, disabled=not has_gsc)
with col2:
    with bordered_container():
        st.markdown("Gewicht: **Semantische Linklücke**")
        w_def  = st.slider("", 0.0, 1.0, 0.30, 0.05)
with col3:
    with bordered_container():
        st.markdown("Gewicht: **Sprungbrett-URLs**")
        w_rank = st.slider("", 0.0, 1.0, 0.30, 0.05, disabled=not has_pos)
        st.caption("Sprungbrett-URLs – Feineinstellung")
        rank_minmax = st.slider("Ranking Sprungbrett-URL (Positionsbereich)", 1, 50, (8, 20), 1, disabled=not has_pos)
with col4:
    with bordered_container():
        st.markdown("Gewicht: **Mauerblümchen**")
        w_orph = st.slider("", 0.0, 1.0, 0.10, 0.05)
        st.caption("Mauerblümchen – Feineinstellung")
        thin_k = st.slider("Thin-Schwelle (Inlinks ≤ K)", 0, 10, 2, 1)

eff_sum = (0 if not has_gsc else w_lihd) + w_def + (0 if not has_pos else w_rank) + w_orph
if not math.isclose(eff_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
    st.caption(f"ℹ️ Aktuelle PRIO-Gewichtungs-Summe: {eff_sum:.2f}. (wird intern normalisiert)")

with st.expander("Offpage-Einfluss (Backlinks & Ref. Domains)", expanded=False):
    st.caption("Seiten mit vielen Ref. Domains werden bei Hidden Champions & Linklücke gedämpft.")
    offpage_damp_enabled = st.checkbox("Offpage-Dämpfung anwenden", value=True)
    beta_offpage = st.slider("Stärke der Dämpfung", 0.0, 1.0, 0.5, 0.05, disabled=not offpage_damp_enabled)

with st.expander("Reihenfolge der Empfehlungen - *OPTIONAL*", expanded=False):
    st.caption("Mix aus Nähe & Linkbedarf oder nur eines von beidem.")
    sort_labels = {"rank_mix":"Mix (Nähe & Linkbedarf kombiniert)","prio_only":"Nur Linkbedarf","sim_only":"Nur inhaltliche Nähe"}
    sort_options = ["rank_mix","prio_only","sim_only"]
    sort_choice = st.radio("Sortierung", options=sort_options, index=0, format_func=lambda k: sort_labels.get(k, k), horizontal=True, key="sort_choice_radio")
    alpha_mix = st.slider("Gewichtung: inhaltliche Nähe vs. Linkbedarf", 0.0, 1.0, 0.5, 0.05, key="alpha_mix_slider")

if "__gems_loading__" not in st.session_state:
    st.session_state["__gems_loading__"] = False
if "__ready_gems__" not in st.session_state:
    st.session_state["__ready_gems__"] = False
run_gems = st.button("Let's Go (Analyse 3)", type="secondary")
if run_gems:
    st.session_state["__gems_loading__"] = True
    st.session_state["__ready_gems__"] = False
    st.rerun()

if not (st.session_state["__gems_loading__"] or st.session_state.get("__ready_gems__", False)):
    st.info("Stell die Regler ein und lade ggf. **Search Console Daten**. Dann klicke auf **Let's Go (Analyse 3)**.")
    # ---- Wir fallen hier NICHT aus dem Skript. Analyse 4 darf unten weiterlaufen ----

# ---- PRIO-Helfer für Analyse 3 ----
from collections import defaultdict
inbound_count = defaultdict(int)
for s, t in st.session_state.get("_content_links", set()):
    inbound_count[t] += 1
min_ils, max_ils = norm_ranges.get("ils", (0.0, 1.0))
lo_bl_log, hi_bl_log = norm_ranges.get("bl_log", (0.0, 1.0))
lo_rd_log, hi_rd_log = norm_ranges.get("rd_log", (0.0, 1.0))
backlink_map: Dict[str, Dict[str, float]] = st.session_state.get("_backlink_map", {})

def _safe_norm(x: float, lo: float, hi: float) -> float:
    if hi > lo:
        v = (float(x) - lo) / (hi - lo)
        return float(np.clip(v, 0.0, 1.0))
    return 0.0

def ext_auth_norm_for(u: str) -> float:
    bl = backlink_map.get(u, {})
    bl_raw = float(bl.get("backlinks", 0.0) or 0.0)
    rd_raw = float(bl.get("referringDomains", 0.0) or 0.0)
    bl_log = float(np.log1p(max(0.0, bl_raw)))
    rd_log = float(np.log1p(max(0.0, rd_raw)))
    bl_n = robust_norm(bl_log, lo_bl_log, hi_bl_log)
    rd_n = robust_norm(rd_log, lo_rd_log, hi_rd_log)
    return 0.5 * (bl_n + rd_n)

def damp_factor(u: str) -> float:
    if not offpage_damp_enabled:
        return 1.0
    x = float(np.clip(ext_auth_norm_for(u), 0.0, 1.0))
    k = 6.0; m = 0.5
    s  = 1.0 / (1.0 + np.exp(-k * (x - m)))
    s0 = 1.0 / (1.0 + np.exp(-k * (0.0 - m)))
    s1 = 1.0 / (1.0 + np.exp(-k * (1.0 - m)))
    s_norm = (s - s0) / (s1 - s0)
    s_norm = float(np.clip(s_norm, 0.0, 1.0))
    return float(np.clip(1.0 - beta_offpage * s_norm, 0.0, 1.0))

def ils_norm_for(u: str) -> float:
    m = metrics_map.get(u)
    if not m:
        return 0.0
    x = float(m.get("score", 0.0))
    return float(np.clip((x - min_ils) / (max_ils - min_ils), 0.0, 1.0)) if max_ils > min_ils else 0.0

def lihd_for(u: str) -> float:
    if not has_gsc:
        return 0.0
    d = float(demand_map.get(u, 0.0))
    base = float((1.0 - ils_norm_for(u)) * d)
    return base * damp_factor(u)

def rank_sweetspot_for(u: str, lo: int, hi: int) -> float:
    p = pos_map.get(u)
    if p is None:
        return 0.0
    return 1.0 if lo <= p <= hi else 0.0

def orphan_score_for(u: str, k: int) -> float:
    inl = int(inbound_count.get(u, 0))
    orphan = 1.0 if inl == 0 else 0.0
    thin   = 1.0 if inl <= k else 0.0
    return float(max(orphan, 0.5 * thin))

def deficit_weighted_for(target: str) -> float:
    if not isinstance(res1_df, pd.DataFrame):
        return 0.0
    row = res1_df.loc[res1_df["Ziel-URL"] == target]
    if row.empty:
        return 0.0
    r = row.iloc[0]
    sum_all = 0.0
    sum_missing = 0.0
    i = 1
    while True:
        col_sim = f"Ähnlichkeit {i}"
        col_src = f"Related URL {i}"
        if col_src not in res1_df.columns or col_sim not in res1_df.columns:
            break
        sim_val = r.get(col_sim, np.nan)
        src_val = normalize_url(r.get(col_src, ""))
        if pd.isna(sim_val) or not src_val:
            i += 1; continue
        simf = float(sim_val) if pd.notna(sim_val) else 0.0
        simf = max(0.0, simf)
        sum_all += simf
        cont_new = f"Link von Related URL {i} auf Ziel-URL aus Inhalt heraus vorhanden?"
        cont_old = f"aus Inhalt heraus verlinkt {i}?"
        from_content = str(r.get(cont_new, r.get(cont_old, "nein"))).strip().lower()
        if from_content != "ja":
            sum_missing += simf
        i += 1
    ratio = float(np.clip(sum_missing / sum_all, 0.0, 1.0)) if sum_all > 0 else 0.0
    return ratio * damp_factor(target)

# Gems bestimmen
if source_potential_map:
    sorted_sources = sorted(source_potential_map.items(), key=lambda x: x[1], reverse=True)
    cutoff_idx = max(1, int(len(sorted_sources) * gem_pct / 100))
    gems = [u for u, _ in sorted_sources[:cutoff_idx]]
else:
    gems = []

# PRIO je Ziel
target_priority_map: Dict[str, float] = {}
if isinstance(res1_df, pd.DataFrame) and not res1_df.empty:
    for _, row in res1_df.iterrows():
        u = normalize_url(row["Ziel-URL"])
        if not u:
            continue
        li   = lihd_for(u)
        ddef = deficit_weighted_for(u)
        rnk  = rank_sweetspot_for(u, lo=rank_minmax[0], hi=rank_minmax[1]) if has_pos else 0.0
        oph  = orphan_score_for(u, thin_k)
        weights = np.array([
            (0.0 if not has_gsc else w_lihd),
            w_def,
            (0.0 if not has_pos else w_rank),
            w_orph
        ], dtype=float)
        comps   = np.array([li, ddef, rnk, oph], dtype=float)
        denom = weights.sum()
        prio = float((weights @ comps) / denom) if denom > 0 else 0.0
        target_priority_map[u] = prio

# Empfehlungen pro Gem
try:
    from collections import defaultdict
    related_by_source: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    if not isinstance(res1_df, pd.DataFrame) or res1_df.empty or not gems:
        pass
    else:
        def content_link_flag(row_obj, i: int) -> bool:
            cont_new = f"Link von Related URL {i} auf Ziel-URL aus Inhalt heraus vorhanden?"
            cont_old = f"aus Inhalt heraus verlinkt {i}?"
            return str(row_obj.get(cont_new, row_obj.get(cont_old, "nein"))).strip().lower() == "ja"
        for _, row in res1_df.iterrows():
            target = normalize_url(row["Ziel-URL"])
            if not target:
                continue
            i = 1
            while f"Related URL {i}" in res1_df.columns:
                src_raw = row.get(f"Related URL {i}", "")
                src = normalize_url(src_raw)
                if src and not content_link_flag(row, i):
                    try:
                        simf = float(row.get(f"Ähnlichkeit {i}", 0.0) or 0.0)
                    except Exception:
                        simf = 0.0
                    related_by_source[src].append((target, float(simf)))
                i += 1

        gem_rows: List[List] = []
        N_TOP = int(max_targets_per_gem)
        TOTAL_CAP = 3000
        for gem in gems:
            candidates = related_by_source.get(gem, [])
            if not candidates:
                continue
            rows = []
            for target, simf in candidates:
                prio_t = float(target_priority_map.get(target, 0.0))
                if sort_choice == "prio_only":
                    sort_score = prio_t; sort_key = lambda r: (r[3], r[2])
                elif sort_choice == "sim_only":
                    sort_score = simf;   sort_key = lambda r: (r[2], r[3])
                else:
                    sort_score = float(alpha_mix) * float(simf) + (1.0 - float(alpha_mix)) * float(prio_t)
                    sort_key = lambda r: (r[4], r[2], r[3])
                rows.append([gem, target, float(simf), float(prio_t), float(sort_score)])
            rows.sort(key=sort_key, reverse=True)
            gem_rows.extend(rows[:N_TOP])
            if len(gem_rows) >= TOTAL_CAP:
                st.info(f"Ausgabe auf {TOTAL_CAP} Empfehlungen gekappt (Performance-Schutz).")
                break

        if gem_rows:
            def pot_for(g: str) -> float:
                return float(st.session_state.get("_source_potential_map", {}).get(normalize_url(g), 0.0))
            long_rows = []
            for gem, target, simv, prio_t, sortv in gem_rows:
                long_rows.append([disp(gem), round(pot_for(gem), 3), disp(target), round(float(simv), 3), round(float(prio_t), 3), round(float(sortv), 3)])
            cheat_long_df = pd.DataFrame(long_rows, columns=[
                "Gem (Quell-URL)","Linkpotenzial (Quell-URL)","Ziel-URL","Similarity (inhaltliche Nähe)","Linkbedarf","Score für Sortierung",
            ])
            if not cheat_long_df.empty:
                cheat_long_df = cheat_long_df.sort_values(
                    by=["Gem (Quell-URL)", "Score für Sortierung", "Similarity (inhaltliche Nähe)"],
                    ascending=[True, False, False],
                    kind="mergesort",
                ).reset_index(drop=True)
            st.dataframe(cheat_long_df, use_container_width=True, hide_index=True)
            csv_cheat_long = cheat_long_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download »Cheat-Sheet der internen Verlinkung (Long-Format)« (CSV)",
                data=csv_cheat_long, file_name="Cheat-Sheet_der_internen_Verlinkung_long.csv",
                mime="text/csv", key="cheat_sheet_long_download",
            )
            st.session_state["__gems_loading__"] = False
            ph3 = st.session_state.get("__gems_ph__")
            if ph3: ph3.empty()
            st.success("✅ Analyse abgeschlossen!")
            st.session_state["__ready_gems__"] = True
        else:
            st.session_state["__gems_loading__"] = False
            ph3 = st.session_state.get("__gems_ph__")
            if ph3: ph3.empty()
            st.caption("Keine Gem-Empfehlungen gefunden.")
except Exception as e:
    st.session_state["__gems_loading__"] = False
    ph3 = st.session_state.get("__gems_ph__")
    if ph3: ph3.empty()
    st.exception(e)
    # KEIN stop() – damit Analyse 4 unten verfügbar ist


# =========================================================
# Analyse 4: Anchor & Query Intelligence (Embeddings)
# =========================================================
st.markdown("---")
st.subheader("🔎 Analyse 4: Anchor & Query Intelligence (Embeddings)")

with st.sidebar:
    st.markdown("### 🔎 Analyse 4: Optionen")
    a4_enable = st.toggle("Analyse 4 aktivieren", value=False, help="Aktiviere die Anchor/Query-Checks & Reports.")
    st.caption("— nur Problemfälle werden angezeigt & exportiert —")

if not a4_enable:
    st.info("Aktiviere **Analyse 4** in der Sidebar, um die Anchor/Query-Checks auszuführen.")
    st.stop()

# ---- Eingaben für Analyse 4 (Sidebar) ----
with st.sidebar:
    st.markdown("#### Brand-Handling")
    brand_text = st.text_area("Brand-Schreibweisen (eine pro Zeile oder komma-getrennt)", value="")
    brand_file = st.file_uploader("Optional: Brand-Liste (CSV/Excel, 1 Spalte)", type=["csv","xlsx","xlsm","xls"], key="brand_file")
    auto_variants = st.checkbox("Automatisch Varianten erzeugen (z. B. „bora kochfeld“, „bora-kochfeld“)", value=True)
    head_nouns_text = st.text_input("Head-Nomen (kommagetrennt, editierbar)", value="kochfeld, kochfeldabzug, system, kochfelder")
    brand_mode = st.radio("Welche Queries berücksichtigen?", ["Nur Non-Brand", "Nur Brand", "Beides"], index=0, horizontal=True)

    st.markdown("#### Matching")
    metric_choice = st.radio("GSC-Bewertung nach …", ["Impressions", "Clicks"], index=0, horizontal=True)
    check_exact = st.checkbox("Exact Match prüfen", value=True)
    check_embed = st.checkbox("Embedding Match prüfen", value=True)
    embed_model_name = st.selectbox("Embedding-Modell", ["sentence-transformers/all-MiniLM-L6-v2","sentence-transformers/all-MiniLM-L12-v2","sentence-transformers/all-mpnet-base-v2"], index=0, help="Standard: all-MiniLM-L6-v2")
    embed_thresh = st.slider("Cosine-Schwelle (Embedding)", 0.50, 0.95, 0.75, 0.01)

    st.markdown("#### Schwellen")
    min_clicks = st.number_input("Mindest-Klicks/Query", min_value=0, value=50, step=10)
    min_impr   = st.number_input("Mindest-Impressions/Query", min_value=0, value=500, step=50)
    topN_default = st.number_input("Top-N Queries pro URL (als zusätzliche Bedingung)", min_value=1, value=10, step=1)
    top_anchor_abs = st.number_input("Schwelle identischer Anker (absolut)", min_value=1, value=200, step=10)
    top_anchor_share = st.slider("Schwelle TopAnchorShare (%)", 0, 100, 60, 1)

    st.markdown("#### Zusätzliche Uploads")
    gsc_up_a4 = st.file_uploader("Search Console: URL | Query | Clicks | Impressions", type=["csv","xlsx","xlsm","xls"], key="gsc_a4")
    kwmap_up  = st.file_uploader("Keyword-Zielvorgaben: URL + Keyword-Spalten", type=["csv","xlsx","xlsm","xls"], key="kwmap")
    show_treemap = st.checkbox("Treemap-Visualisierung aktivieren", value=True)
    treemap_topK = st.number_input("Treemap: Top-K Anchors anzeigen", min_value=3, max_value=50, value=12, step=1)

# ---- Helper: Brand-Liste bauen ----
def split_list_text(s: str) -> List[str]:
    if not s:
        return []
    arr = []
    for line in s.replace(";", ",").splitlines():
        for tok in line.split(","):
            v = tok.strip()
            if v:
                arr.append(v)
    return arr

def read_single_col_file(up) -> List[str]:
    if up is None:
        return []
    df = read_any_file_cached(getattr(up, "name", ""), up.getvalue())
    if df is None or df.empty:
        return []
    col = df.columns[0]
    vals = [str(x).strip() for x in df[col].tolist() if str(x).strip()]
    return vals

def make_brand_variants(brands: List[str], nouns: List[str], auto: bool) -> List[str]:
    base = set()
    for b in brands:
        b = b.strip()
        if not b:
            continue
        base.add(b)
    if auto:
        out = set(base)
        for b in base:
            for n in nouns:
                n = n.strip()
                if not n:
                    continue
                out.add(f"{b} {n}")
                out.add(f"{b}-{n}")
        return sorted(out)
    return sorted(base)

brand_list = split_list_text(brand_text)
brand_list += read_single_col_file(brand_file)
brand_list = sorted({b.strip() for b in brand_list if b.strip()})
head_nouns = [x.strip() for x in head_nouns_text.split(",") if x.strip()]
brand_all_terms = make_brand_variants(brand_list, head_nouns, auto_variants)

def is_brand_query(q: str) -> bool:
    s = (q or "").lower().strip()
    if not s:
        return False
    # Wortgrenzen grob
    tokens = [re.escape(t.lower()) for t in brand_all_terms]
    if not tokens:
        return False
    pat = r"(?:^|[^a-z0-9äöüß])(" + "|".join(tokens) + r")(?:$|[^a-z0-9äöüß])"
    return re.search(pat, s, flags=re.IGNORECASE) is not None

def is_navigational(anchor: str) -> bool:
    return (anchor or "").strip().lower() in NAVIGATIONAL_ANCHORS

# ---- Anchor-Inventar aus All Inlinks (inkl. ALT als Fallback) ----
def extract_anchor_inventory(df: pd.DataFrame) -> pd.DataFrame:
    # Ziel, Anchor, Count
    rows = []
    hdr = [str(c).strip() for c in df.columns]
    a_idx = find_column_index(hdr, POSSIBLE_ANCHOR)
    alt_i = find_column_index(hdr, POSSIBLE_ALT)
    for row in df.itertuples(index=False, name=None):
        src = remember_original(row[src_idx]); dst = remember_original(row[dst_idx])
        if not dst:
            continue
        anchor_val = None
        if a_idx != -1:
            anchor_val = row[a_idx]
        if (anchor_val is None or (str(anchor_val).strip() == "")) and alt_i != -1:
            anchor_val = row[alt_i]
        anchor = str(anchor_val or "").strip()
        if not anchor:
            continue
        rows.append([normalize_url(dst), anchor])
    if not rows:
        return pd.DataFrame(columns=["target","anchor","count"])
    tmp = pd.DataFrame(rows, columns=["target","anchor"])
    agg = tmp.groupby(["target","anchor"], as_index=False).size().rename(columns={"size":"count"})
    return agg

anchor_inv = extract_anchor_inventory(inlinks_df)

# ---- Over-Anchor ≥ 200 (alle Anchors ≥ Schwelle; TopAnchorShare optional)
over_anchor_df = pd.DataFrame(columns=["Ziel-URL","Anchor","Count","TopAnchorShare(%)"])
if not anchor_inv.empty:
    # Anteil je Ziel berechnen
    totals = anchor_inv.groupby("target")["count"].sum().rename("total")
    tmp = anchor_inv.merge(totals, on="target", how="left")
    tmp["share"] = (100.0 * tmp["count"] / tmp["total"]).round(2)
    filt = (tmp["count"] >= int(top_anchor_abs)) | (tmp["share"] >= float(top_anchor_share))
    over_anchor_df = tmp.loc[filt, ["target","anchor","count","share"]].copy()
    over_anchor_df.columns = ["Ziel-URL","Anchor","Count","TopAnchorShare(%)"]

# ---- GSC laden & Top-20% je URL bestimmen (Non-Brand/Brand-Modus + Mindestschwellen + Top-N)
gsc_df = None
if "gsc_df_cache_for_a4" not in st.session_state:
    st.session_state["gsc_df_cache_for_a4"] = None
if gsc_up_a4 is not None:
    gsc_df = read_any_file_cached(getattr(gsc_up_a4, "name", ""), gsc_up_a4.getvalue())
else:
    # Wenn bereits für Analyse 3 geladen, verwenden
    gsc_df = st.session_state.get("__gsc_df_raw__", None)

gsc_issues_df = pd.DataFrame(columns=["Ziel-URL","Query","Match-Typ","Anker gefunden?","Fund-Count","Hinweis"])
leader_conflicts_df = pd.DataFrame(columns=["Query","Verlinkte URL (aktueller Link)","Leader-URL","Leader-Wert","Hinweis (navigativ ausgeschlossen?)"])

if isinstance(gsc_df, pd.DataFrame) and not gsc_df.empty:
    df = gsc_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    hdr = [_norm_header(c) for c in df.columns]
    def _find_idx(candidates: Iterable[str], default=None):
        cand_norm = {_norm_header(c) for c in candidates}
        for i, h in enumerate(hdr):
            if h in cand_norm: return i
        for i, h in enumerate(hdr):
            if any(c in h for c in cand_norm): return i
        return default
    url_i = _find_idx({"url","page","seite","address","adresse","landingpage","landing page"}, 0)
    q_i   = _find_idx({"query","suchanfrage","suchbegriff"}, 1)
    c_i   = _find_idx({"clicks","klicks"}, None)
    im_i  = _find_idx({"impressions","impr","impressionen","suchimpressionen","search impressions"}, None)
    if url_i is None or q_i is None or (c_i is None and im_i is None):
        st.warning("GSC-Datei: Spalten URL + Query + (Clicks/Impressions) wurden nicht vollständig erkannt.")
    else:
        df.iloc[:, url_i] = df.iloc[:, url_i].astype(str).map(normalize_url)
        df.iloc[:, q_i]   = df.iloc[:, q_i].astype(str).fillna("").str.strip()
        if c_i is not None:
            df.iloc[:, c_i] = pd.to_numeric(df.iloc[:, c_i], errors="coerce").fillna(0)
        if im_i is not None:
            df.iloc[:, im_i] = pd.to_numeric(df.iloc[:, im_i], errors="coerce").fillna(0)

        # Brand-Filter anwenden
        def brand_filter(row) -> bool:
            q = str(row.iloc[q_i])
            is_brand = is_brand_query(q)
            if brand_mode == "Nur Non-Brand":
                return (not is_brand)
            elif brand_mode == "Nur Brand":
                return is_brand
            else:
                return True

        df = df[df.apply(brand_filter, axis=1)]
        # Mindestschwellen
        if c_i is not None:
            df = df[df.iloc[:, c_i] >= int(min_clicks)] if metric_choice == "Clicks" else df
        if im_i is not None:
            df = df[df.iloc[:, im_i] >= int(min_impr)] if metric_choice == "Impressions" else df

        # Pro URL sortieren & Top 20% (mind. 1), zusätzlich max Top-N
        metric_col = c_i if metric_choice == "Clicks" else im_i
        df = df.sort_values(by=[df.columns[url_i], df.columns[metric_col]], ascending=[True, False])
        top_rows = []
        for u, grp in df.groupby(df.columns[url_i], sort=False):
            n = max(1, int(math.ceil(0.2 * len(grp))))
            n = max(1, min(n, int(topN_default)))  # zusätzliche Top-N-Grenze
            top_rows.append(grp.head(n))
        df_top = pd.concat(top_rows) if top_rows else pd.DataFrame(columns=df.columns)

        # ---- Matching gegen Anchor-Inventar (Exact/Embeddings) ----
        # Anchor-Multiset: target -> {anchor: count}
        inv_map: Dict[str, Dict[str, int]] = {}
        for _, r in anchor_inv.iterrows():
            inv_map.setdefault(str(r["target"]), {})[str(r["anchor"])] = int(r["count"])

        # Embedding-Vorbereitung
        model = None
        if check_embed:
            try:
                from sentence_transformers import SentenceTransformer
                if "_A4_EMB_MODEL_NAME" not in st.session_state or st.session_state.get("_A4_EMB_MODEL_NAME") != embed_model_name:
                    st.session_state["_A4_EMB_MODEL"] = SentenceTransformer(embed_model_name)
                    st.session_state["_A4_EMB_MODEL_NAME"] = embed_model_name
                model = st.session_state["_A4_EMB_MODEL"]
            except Exception as e:
                st.warning(f"Embedding-Modell konnte nicht geladen werden ({e}). Embedding-Abgleich wird übersprungen.")
                check_embed = False

        def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            # L2-Normalisierung + dot
            A = A.astype(np.float32, copy=False)
            B = B.astype(np.float32, copy=False)
            A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
            return A @ B.T

        # Cache Anchor-Embeddings je target
        anchor_emb_cache: Dict[str, Tuple[List[str], Optional[np.ndarray]]] = {}
        if check_embed and model is not None:
            # alle (target, anchor) bündeln je target
            for target, sub in anchor_inv.groupby("target"):
                anchors = sub["anchor"].astype(str).tolist()
                if anchors:
                    try:
                        emb = model.encode(anchors, batch_size=64, show_progress_bar=False)
                        anchor_emb_cache[target] = (anchors, np.asarray(emb))
                    except Exception:
                        anchor_emb_cache[target] = (anchors, None)
                else:
                    anchor_emb_cache[target] = ([], None)

        # 1) Coverage der Top-Queries als Anchors (pro Ziel-URL)
        issues_rows = []
        for u, grp in df_top.groupby(df_top.columns[url_i], sort=False):
            inv = inv_map.get(u, {})
            anchors_list = list(inv.keys())
            anchors_lower = [a.lower() for a in anchors_list]
            # Embedding vorbereiten
            a_names, a_emb = anchor_emb_cache.get(u, ([], None))
            for _, rr in grp.iterrows():
                q = str(rr.iloc[q_i]).strip()
                if not q:
                    continue
                found = False
                found_cnt = 0
                match_type = []
                # Exact
                if check_exact:
                    cnt = 0
                    for a, c in inv.items():
                        if a.lower() == q.lower():
                            cnt += int(c)
                    if cnt > 0:
                        found = True
                        found_cnt = max(found_cnt, cnt)
                        match_type.append("Exact")
                # Embedding
                if (not found or check_embed) and check_embed and model is not None and a_emb is not None and len(a_names) > 0:
                    try:
                        q_emb = model.encode([q], show_progress_bar=False)
                        S = cosine_sim_matrix(np.asarray(q_emb), a_emb)[0]
                        # alle Anchor >= threshold zählen
                        idxs = np.where(S >= float(embed_thresh))[0]
                        if idxs.size > 0:
                            found = True
                            # Summe der Counts ähnlicher Anchors
                            cnt = int(sum(inv.get(a_names[i], 0) for i in idxs))
                            found_cnt = max(found_cnt, cnt)
                            match_type.append("Embedding")
                    except Exception:
                        pass
                if not found:
                    issues_rows.append([disp(u), q, "+".join(match_type) if match_type else "—", "nein", 0, "Top-Query kommt nicht als Anchor vor"])
        if issues_rows:
            gsc_issues_df = pd.DataFrame(issues_rows, columns=["Ziel-URL","Query","Match-Typ","Anker gefunden?","Fund-Count","Hinweis"])

        # 2) Leader-Konflikte (Query verlinkt via Anchor auf andere URL als Leader)
        # Leader bestimmen
        leader_rows = []
        # Aggregation: pro Query -> Leader-URL
        if metric_choice == "Clicks":
            lead_ser = df.groupby(df.columns[q_i]).apply(lambda x: x.loc[x.iloc[:, c_i].idxmax(), df.columns[url_i]] if c_i is not None else None)
            lead_val = df.groupby(df.columns[q_i]).apply(lambda x: float(x.iloc[:, c_i].max()) if c_i is not None else 0.0)
        else:
            lead_ser = df.groupby(df.columns[q_i]).apply(lambda x: x.loc[x.iloc[:, im_i].idxmax(), df.columns[url_i]] if im_i is not None else None)
            lead_val = df.groupby(df.columns[q_i]).apply(lambda x: float(x.iloc[:, im_i].max()) if im_i is not None else 0.0)
        leader_map = {q: normalize_url(u) for q, u in lead_ser.to_dict().items() if isinstance(u, str)}
        leader_val_map = {q: float(v) for q, v in lead_val.to_dict().items()}
        # Für jeden Link prüfen: Anchor ~ Query, Ziel != Leader
        # Wir nutzen Anchor-Inventar (zielbezogen). Wir brauchen aber „links“, also Quelle→Ziel + Anchor. Wir haben counts je Ziel+Anchor.
        # Näherungsweise: Wenn irgendein Anchor (zu einer Ziel-URL) die Query matcht und Ziel != Leader → Konflikt.
        # Optional: navigative Anchors ausschließen.
        # Exact:
        if check_exact or check_embed:
            for q, lead_u in leader_map.items():
                # alle Ziele, die einen passenden Anchor zu q haben
                # Exact
                exact_targets = set()
                if check_exact:
                    sub = anchor_inv[anchor_inv["anchor"].str.lower() == q.lower()]
                    exact_targets.update(normalize_url(t) for t in sub["target"].tolist())
                # Embedding
                embed_targets = set()
                if check_embed and model is not None:
                    # precompute q embedding
                    try:
                        q_emb = model.encode([q], show_progress_bar=False)
                        for tgt, (a_names, a_emb) in anchor_emb_cache.items():
                            if a_emb is None or len(a_names) == 0:
                                continue
                            S = cosine_sim_matrix(np.asarray(q_emb), a_emb)[0]
                            if (S >= float(embed_thresh)).any():
                                embed_targets.add(tgt)
                    except Exception:
                        pass
                cand_targets = exact_targets.union(embed_targets)
                for tgt in cand_targets:
                    if not lead_u or not tgt:
                        continue
                    if tgt != lead_u:
                        # navigative?
                        nav_flag = False
                        # prüfen, ob alle passenden Anchors navigativ sind; wenn ja -> "navigativ ausgeschlossen"
                        # (vereinfachend: wenn irgendein passender Anchor NICHT navigativ ist -> Konflikt)
                        nav_only = True
                        anchors_for_tgt = anchor_inv[anchor_inv["target"] == tgt]["anchor"].astype(str).tolist()
                        # simple heuristic: wenn Query exakt/näherungsweise vorkommt, prüfe Navigational-Liste
                        for a in anchors_for_tgt:
                            is_match = (check_exact and a.lower() == q.lower())
                            if not is_match and check_embed and model is not None:
                                try:
                                    a_emb = model.encode([a], show_progress_bar=False)
                                    s = float(cosine_sim_matrix(np.asarray(q_emb), np.asarray(a_emb))[0,0])
                                    is_match = (s >= float(embed_thresh))
                                except Exception:
                                    is_match = False
                            if is_match:
                                if not is_navigational(a):
                                    nav_only = False
                                    break
                        nav_flag = nav_only
                        if not nav_flag:
                            leader_rows.append([
                                q, disp(tgt), disp(lead_u),
                                int(leader_val_map.get(q, 0)),
                                "Anchor navigativ? (ausgeschlossen): nein"
                            ])
                        else:
                            # Falls ausschließlich navigative Anchors matchen, aus Konflikt-Report ausschließen
                            pass
        if leader_rows:
            leader_conflicts_df = pd.DataFrame(leader_rows, columns=["Query","Verlinkte URL (aktueller Link)","Leader-URL","Leader-Wert","Hinweis (navigativ ausgeschlossen?)"])

# ---- Keyword-Zielvorgaben prüfen (Exact &/oder Embedding) ----
kw_missing_df = pd.DataFrame(columns=["Ziel-URL","Keyword","Hinweis","Match-Typ"])
if kwmap_up is not None:
    kw_df = read_any_file_cached(getattr(kwmap_up, "name", ""), kwmap_up.getvalue())
    if isinstance(kw_df, pd.DataFrame) and not kw_df.empty:
        dff = kw_df.copy()
        dff.columns = [str(c).strip() for c in dff.columns]
        hdr = [_norm_header(c) for c in dff.columns]
        url_idx_map = None
        url_i = find_column_index(dff.columns.tolist(), ["url","urls","page","seite","address","adresse","ziel url","ziel-url"])
        if url_i == -1:
            # Fallback: erste Spalte
            url_i = 0
        # Keywords = alle Spalten, deren Name Regex enthält
        kw_cols = []
        for i, col in enumerate(dff.columns):
            if i == url_i:
                continue
            h = _norm_header(col)
            if re.search(r"(keyword|suchanfrage|suchbegriff|query)", h):
                kw_cols.append(col)
        if not kw_cols:
            # Wenn nichts erkannt: alle Nicht-URL-Spalten als Keywords
            kw_cols = [c for i, c in enumerate(dff.columns) if i != url_i]
        # Embedding-Cache für Keywords global
        model_kw = None
        if check_embed:
            model_kw = st.session_state.get("_A4_EMB_MODEL", None)
        for _, r in dff.iterrows():
            url = remember_original(r.iloc[url_i])
            if not url:
                continue
            inv = anchor_inv[anchor_inv["target"] == normalize_url(url)]
            inv_map_u = {str(a): int(c) for a, c in zip(inv["anchor"].astype(str), inv["count"].astype(int))}
            a_names_u = list(inv_map_u.keys())
            a_emb_u = None
            if check_embed and model_kw is not None and len(a_names_u) > 0:
                try:
                    a_emb_u = model_kw.encode(a_names_u, batch_size=64, show_progress_bar=False)
                    a_emb_u = np.asarray(a_emb_u)
                except Exception:
                    a_emb_u = None
            for col in kw_cols:
                kw = str(r[col]).strip()
                if not kw:
                    continue
                found = False
                match_t = []
                if check_exact and any(kw.lower() == a.lower() for a in a_names_u):
                    found = True; match_t.append("Exact")
                if (not found or check_embed) and check_embed and a_emb_u is not None and model_kw is not None:
                    try:
                        q_emb = model_kw.encode([kw], show_progress_bar=False)
                        S = cosine_sim_matrix(np.asarray(q_emb), a_emb_u)[0]
                        if (S >= float(embed_thresh)).any():
                            found = True; match_t.append("Embedding")
                    except Exception:
                        pass
                if not found:
                    kw_missing_df.loc[len(kw_missing_df)] = [disp(url), kw, "Ziel-Keyword nicht als Anchor verlinkt", "+".join(match_t) if match_t else "—"]

# ---- Treemap-Visualisierung (optional) ----
if show_treemap and _HAS_PLOTLY and not anchor_inv.empty:
    st.markdown("### Treemap: Ankerverteilung pro Ziel-URL")
    targets_sorted = sorted(anchor_inv["target"].unique())
    sel = st.selectbox("Ziel-URL wählen", [disp(t) for t in targets_sorted], index=0)
    t_norm = None
    # map back to normalized key
    for k, v in st.session_state["_ORIG_MAP"].items():
        if v == sel:
            t_norm = k; break
    if t_norm is None:
        # fallback
        t_norm = normalize_url(sel)
    sub = anchor_inv[anchor_inv["target"] == t_norm].sort_values("count", ascending=False)
    if not sub.empty:
        top = sub.head(int(treemap_topK)).copy()
        rest_sum = int(sub["count"].iloc[int(treemap_topK):].sum()) if len(sub) > treemap_topK else 0
        if rest_sum > 0:
            top.loc[len(top)] = [t_norm, "— Sonstige —", rest_sum]
        fig = px.treemap(top, path=["anchor"], values="count", title=f"Anchor-Verteilung: {sel}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Keine Anchors für die gewählte URL gefunden.")

# ---- Ergebnisse anzeigen (nur Problemfälle) & Download bündeln ----
st.markdown("### Ergebnisse (nur Problemfälle)")
cols = st.columns(3)
with cols[0]:
    st.markdown("**Over-Anchor (≥ Schwelle / Share ≥ Schwelle)**")
    if over_anchor_df.empty:
        st.caption("Keine Problemfälle.")
    else:
        st.dataframe(over_anchor_df, use_container_width=True, hide_index=True)
with cols[1]:
    st.markdown("**GSC-Top-Queries → nicht als Anchor**")
    if gsc_issues_df.empty:
        st.caption("Keine Problemfälle oder keine GSC-Daten.")
    else:
        st.dataframe(gsc_issues_df, use_container_width=True, hide_index=True)
with cols[2]:
    st.markdown("**Leader-Konflikte**")
    if leader_conflicts_df.empty:
        st.caption("Keine Konflikte.")
    else:
        st.dataframe(leader_conflicts_df, use_container_width=True, hide_index=True)

st.markdown("**Fehlende Ziel-Keywords (URL-Mapping)**")
if kw_missing_df.empty:
    st.caption("Keine fehlenden Ziel-Keywords oder keine Mapping-Datei.")
else:
    st.dataframe(kw_missing_df, use_container_width=True, hide_index=True)

# Download als XLSX (mit ZIP-Fallback)
def export_excel_or_zip(dfs: Dict[str, pd.DataFrame]) -> Tuple[bytes, str, str]:
    # Versuch XLSX
    try:
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
            for name, df in dfs.items():
                # nur Problemfälle exportieren
                df_use = df.copy()
                if df_use.empty:
                    # leeres Blatt trotzdem anlegen (mit Header)
                    df_use = df_use.reindex(columns=["—"])
                df_use.to_excel(xw, index=False, sheet_name=name[:31])
        bio.seek(0)
        return bio.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "analyse4_anchor_query_intelligence.xlsx"
    except Exception:
        pass
    # ZIP (CSVs)
    zbio = io.BytesIO()
    with zipfile.ZipFile(zbio, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in dfs.items():
            csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
            zf.writestr(f"{name}.csv", csv_bytes)
    zbio.seek(0)
    return zbio.getvalue(), "application/zip", "analyse4_anchor_query_intelligence.zip"

export_tabs = {
    "over_anchor_200+": over_anchor_df,
    "gsc_top_queries_missing_anchor": gsc_issues_df,
    "leader_conflicts": leader_conflicts_df,
    "target_keywords_missing": kw_missing_df,
}
data_bytes, mime, fname = export_excel_or_zip(export_tabs)
st.download_button("Download Analyse 4 (XLSX/ZIP)", data=data_bytes, file_name=fname, mime=mime)
