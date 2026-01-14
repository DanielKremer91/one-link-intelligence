import math
from typing import List, Tuple, Dict, Optional, Iterable

import numpy as np
import pandas as pd
import streamlit as st

import inspect
import re
import io
import zipfile
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

inlinks_df = None
metrics_df = None
emb_df = None
related_df = None
backlinks_df = None
offpage_anchors_df = None
gsc_df_loaded = None
kw_df_a4 = None
crawl_df_a1 = None
gsc_df_a1 = None
kw_df_a1 = None


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

st.markdown("""
<style>
/* Download-Buttons rot einfärben */
.stDownloadButton > button {
    background-color: #e60000 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
}

/* Hover-Effekt optional */
.stDownloadButton > button:hover {
    background-color: #cc0000 !important;
}

/* Disabled-State sauber */
.stDownloadButton > button:disabled {
    background-color: #ff9999 !important;
    color: #ffffff !important;
    opacity: 0.6 !important;
}
</style>
""", unsafe_allow_html=True)


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

# Analyse-4 Loader Flags
if "__a4_loading__" not in st.session_state:
    st.session_state["__a4_loading__"] = False
if "__ready_a4__" not in st.session_state:
    st.session_state["__ready_a4__"] = False
if "__a4_ph__" not in st.session_state:
    st.session_state["__a4_ph__"] = None


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
POSSIBLE_SOURCE = ["quelle", "source", "from", "origin", "linkgeber", "quell-url", "referring page url", "referring page", "referring url", "referring page address"]
POSSIBLE_TARGET = ["ziel", "destination", "to", "target", "ziel-url", "ziel url", "target url", "target-url", "target page"]
POSSIBLE_POSITION = ["linkposition", "link position", "position", "link pos","position link","link_position"]

# Neu: Anchor/ALT-Erkennung (inkl. "anchor")
POSSIBLE_ANCHOR = [
    "anchor", "anchor text", "anchor-text", "anker", "ankertext", "linktext", "text",
    "link anchor", "link anchor text", "link text"
]
POSSIBLE_ALT = ["alt", "alt text", "alt-text", "alttext", "image alt", "alt attribute", "alt attribut"]

# Navigative/generische Anchors ausschließen (für Konflikte)
NAVIGATIONAL_ANCHORS = {
    "hier", "zum artikel", "mehr", "mehr erfahren", "mehr lesen", "klicken sie hier", "click here",
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

def _path_segments(url: str) -> list[str]:
    """
    Zerlegt den Pfad einer URL in Segmente.
    Beispiel: https://kaipara.de/wolle-guide/produktberater/x/
    -> ["wolle-guide", "produktberater", "x"]
    """
    from urllib.parse import urlparse
    p = urlparse(str(url or "").strip())
    path = p.path or "/"
    segs = [s for s in path.split("/") if s]
    return segs

def dir_key(url: str, depth: int) -> str:
    """
    Liefert einen Key für die Verzeichnisebene.

    depth = 1 -> erste Verzeichnisebene (kaipara.de/wolle-guide)
    depth = 2 -> zwei Ebenen        (kaipara.de/wolle-guide/produktberater)
    depth = 3 -> drei Ebenen        (maximal, falls vorhanden)

    Host wird immer mit ausgegeben (ohne www.).
    """
    from urllib.parse import urlparse
    p = urlparse(str(url or "").strip())
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    segs = _path_segments(url)
    if not segs or depth <= 0:
        return f"{host}/"

    # depth auf 1–3 begrenzen und nicht tiefer als vorhandene Segmente
    depth = min(max(depth, 1), 3, len(segs))
    return f"{host}/" + "/".join(segs[:depth])


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

def l2_normalize(M: np.ndarray) -> np.ndarray:
    M = M.astype(np.float32, copy=False)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return M / norms


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

def build_related_auto(
    emb_df: pd.DataFrame,
    key_suffix: str,
    label: str,
    pos_idx: Optional[int],
    rel_mult_default: float = 1.0
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Automatische Related-Berechnung:
    - Filtert Inlinks nach Positionen
    - Erzeugt Related-URLs über Cosine-Similarity (aus Embeddings)
    """

    df = emb_df.copy()

    # Spalten normalisieren
    df.columns = [str(c).strip() for c in df.columns]

    # URL-Spalte finden
    url_col = None
    for c in df.columns:
        if c.lower() in ("url", "page", "adresse", "landingpage"):
            url_col = c
            break

    if url_col is None:
        return pd.DataFrame(columns=["Ziel-URL", "Related-URL", "Score"]), {}

    # Embedding-Spalte finden
    emb_col = None
    for c in df.columns:
        if "embedding" in c.lower():
            emb_col = c
            break

    if emb_col is None:
        return pd.DataFrame(columns=["Ziel-URL", "Related-URL", "Score"]), {}

    urls = df[url_col].astype(str).tolist()
    emb = df[emb_col].tolist()

    try:
        V = np.vstack([np.asarray(v, dtype=np.float32) for v in emb])
    except Exception:
        return pd.DataFrame(columns=["Ziel-URL", "Related-URL", "Score"]), {}

    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    V = V / norms

    S = V @ V.T

    np.fill_diagonal(S, 0.0)

    thresh = float(st.session_state.get("a4_related_thresh", 0.50))

    rows = []
    related_map = {}

    for i, u in enumerate(urls):
        sims = S[i]
        idxs = np.where(sims >= thresh)[0]

        if len(idxs) == 0:
            continue

        order = idxs[np.argsort(sims[idxs])[::-1]]

        rels = []
        for j in order:
            rows.append([u, urls[j], float(sims[j])])
            rels.append(urls[j])

        related_map[u] = rels

    df_rel = pd.DataFrame(rows, columns=["Ziel-URL", "Related-URL", "Score"])
    return df_rel, related_map

# =========================
# Zusatz-Helper für A1-KI
# =========================

def _find_crawl_columns(df: pd.DataFrame) -> dict:
    """
    Ermittelt Spaltennamen für URL, Title, H1, Meta Description, Main Content
    nach den definierten Varianten. Erkennt auch Spalten mit Brand-Präfix
    (z. B. "Fressnapf Main Content Extraction") oder anderen Präfixen
    (z. B. "Snippet 1: Fressnapf Main Content Extraction 1").
    Überzählige Spalten werden ignoriert.
    """
    lower = {str(c).strip().lower(): c for c in df.columns}
    
    def pick(candidates):
        # 1. Exakte Matches (wie bisher)
        for cand in candidates:
            key = cand.lower()
            if key in lower:
                return lower[key]
        
        # 2. Teilstring-Matches (für Brand-Präfixe und andere Präfixe)
        # Suche nach Spalten, die die Kandidaten als Teilstring enthalten
        for cand in candidates:
            key = cand.lower()
            # Normalisiere den Key (entferne Sonderzeichen, normalisiere Whitespace)
            key_normalized = re.sub(r'[^\w\s]', ' ', key)
            key_normalized = re.sub(r'\s+', ' ', key_normalized).strip()
            
            for col_lower, col_original in lower.items():
                # Normalisiere auch den Spaltennamen
                col_normalized = re.sub(r'[^\w\s]', ' ', col_lower)
                col_normalized = re.sub(r'\s+', ' ', col_normalized).strip()
                
                # Prüfe, ob der normalisierte Key im normalisierten Spaltennamen enthalten ist
                if key_normalized in col_normalized:
                    return col_original
        
        return None
    
    url_col = pick(["URL", "Url", "Address", "address"])
    title_col = pick(["Title", "Title 1", "Title Tag", "Title-Tag"])
    h1_col = pick(["H1", "H1-1", "H1 - 1", "h1"])
    meta_col = pick(["Meta Description", "Meta Description 1", "MD", "MD 1"])
    main_col = pick(["Main Content Extraction 1", "Main Content Extraction", "Main Content Extraction 2"])
    
    return {
        "url": url_col,
        "title": title_col,
        "h1": h1_col,
        "meta": meta_col,
        "main": main_col,
    }

def build_page_text_maps_for_a1(crawl_df: pd.DataFrame) -> dict:
    """
    Baut Dictionaries:
    - url_norm -> title / h1 / meta / main_content (als Strings)
    """
    if crawl_df is None or crawl_df.empty:
        return {"title": {}, "h1": {}, "meta": {}, "main": {}}

    cols = _find_crawl_columns(crawl_df)
    url_c = cols["url"]
    if url_c is None:
        st.warning("Crawl-Datei (A1) enthält keine erkennbare URL-Spalte – Seitendaten können nicht genutzt werden.")
        return {"title": {}, "h1": {}, "meta": {}, "main": {}}

    title_c = cols["title"]
    h1_c = cols["h1"]
    meta_c = cols["meta"]
    main_c = cols["main"]

    title_map, h1_map, meta_map, main_map = {}, {}, {}, {}

    for _, r in crawl_df.iterrows():
        url_raw = r[url_c]
        key = remember_original(url_raw)
        if not key:
            continue
        if title_c is not None:
            title_map[key] = str(r[title_c]) if pd.notna(r[title_c]) else ""
        if h1_c is not None:
            h1_map[key] = str(r[h1_c]) if pd.notna(r[h1_c]) else ""
        if meta_c is not None:
            meta_map[key] = str(r[meta_c]) if pd.notna(r[meta_c]) else ""
        if main_c is not None:
            main_map[key] = str(r[main_c]) if pd.notna(r[main_c]) else ""

    return {"title": title_map, "h1": h1_map, "meta": meta_map, "main": main_map}


def build_top_gsc_keyword_map_for_a1(gsc_df: pd.DataFrame, metric: str) -> dict:
    """
    Liefert pro URL die Top-10-Keywords (nach Klicks oder Impressions).
    metric: 'Clicks' oder 'Impressions'
    Rückgabe: url_norm -> [keyword1, keyword2, ..., keyword10] (Liste, max. 10)
    """
    if gsc_df is None or gsc_df.empty:
        return {}

    df = gsc_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    hdr = [_norm_header(c) for c in df.columns]

    def find_idx(cands, default=None):
        cands = {_norm_header(x) for x in cands}
        for i, h in enumerate(hdr):
            if h in cands:
                return i
        for i, h in enumerate(hdr):
            if any(c in h for c in cands):
                return i
        return default

    url_i = find_idx(["url", "page", "adresse", "address"], 0)
    q_i   = find_idx(["query", "suchanfrage", "suchbegriff"], 1)
    c_i   = find_idx(["clicks", "klicks"], None)
    im_i  = find_idx(["impressions", "impr", "impressionen"], None)

    if url_i is None or q_i is None or (metric == "Clicks" and c_i is None) or (metric == "Impressions" and im_i is None):
        st.warning("GSC-Datei (A1) konnte nicht eindeutig interpretiert werden (URL/Query/Metric fehlen).")
        return {}

    df.iloc[:, url_i] = df.iloc[:, url_i].astype(str).map(remember_original)
    df.iloc[:, q_i]   = df.iloc[:, q_i].astype(str)

    if c_i is not None:
        df.iloc[:, c_i] = pd.to_numeric(df.iloc[:, c_i], errors="coerce").fillna(0)
    if im_i is not None:
        df.iloc[:, im_i] = pd.to_numeric(df.iloc[:, im_i], errors="coerce").fillna(0)

    metric_col = c_i if metric == "Clicks" else im_i
    df = df[df.iloc[:, metric_col] > 0]

    if df.empty:
        return {}

    top_map = {}
    for url, grp in df.groupby(df.columns[url_i], sort=False):
        # Top 10 statt Top 1
        best = grp.sort_values(df.columns[metric_col], ascending=False).head(10)
        if not best.empty:
            keywords = [str(best.iloc[i, q_i]) for i in range(len(best))]
            top_map[url] = keywords

    return top_map


def build_manual_keyword_map_for_a1(kw_df: pd.DataFrame) -> dict:
    """
    Erwartet: erste Spalte URL, weitere Spalten Keywords
    oder long-Format. Liefert: url_norm -> [keyword1, keyword2, ...]
    """
    if kw_df is None or kw_df.empty:
        return {}

    df = kw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] < 2:
        st.warning("Keyword-Liste (A1) braucht mindestens 2 Spalten (URL + Keywords).")
        return {}

    url_col = df.columns[0]
    kw_cols = list(df.columns[1:])

    pairs = []
    for _, r in df.iterrows():
        url_raw = str(r[url_col]).strip()
        if not url_raw:
            continue
        for c in kw_cols:
            val = str(r[c]).strip()
            if not val:
                continue
            pairs.append((url_raw, val))

    if not pairs:
        return {}

    from collections import defaultdict
    res = defaultdict(list)
    for u_raw, kw in pairs:
        key = remember_original(u_raw)
        if not key:
            continue
        res[key].append(kw)

    return {u: sorted(set(kws)) for u, kws in res.items()}

def is_brand_query(query: str, brand_list: list, auto_variants: bool = True) -> bool:
    """
    Prüft, ob eine Query als Brand-Query gilt.
    query: Die zu prüfende Query
    brand_list: Liste von Brand-Strings (lowercase)
    auto_variants: Wenn True, werden auch Kombinationen wie "marke keyword" erkannt
    """
    if not brand_list:
        return False
    
    q_lower = str(query).strip().lower()
    
    # Direkter Match
    if q_lower in brand_list:
        return True
    
    # Auto-Variants: Kombinationen wie "marke keyword", "keyword marke", "marke-keyword"
    if auto_variants:
        q_tokens = set(q_lower.split())
        for brand in brand_list:
            brand_tokens = set(brand.split())
            if brand_tokens.issubset(q_tokens):
                return True
            # Auch mit Bindestrich/Hybrid
            if brand.replace("-", " ") in q_lower or brand.replace(" ", "-") in q_lower:
                return True
    
    return False


def remove_brand_from_text(text: str) -> str:
    """Entfernt Markennamen und Pipe-Zeichen aus Text."""
    if not text:
        return text
    
    # Entferne Pipe-Zeichen und alles danach (typisch: "| MARKENNAME")
    text = re.sub(r'\s*\|\s*[^|]*$', '', text)
    text = re.sub(r'\s*\|\s*', ' ', text)  # Entferne alle Pipe-Zeichen
    
    return text.strip()
def generate_anchor_variants_for_url(
    url: str,
    page_maps: dict,
    top_kw_map: dict,
    manual_kw_map: dict,
    cfg: dict,
) -> Tuple[str, str, str]:
    """
    Erzeugt 3 Ankertext-Varianten (Kurz, Beschreibend, Handlungsorientiert) für eine Ziel-URL.
    Nutzt abhängig von cfg:
    - Seitendaten (Title/H1/Meta/Main Content)
    - GSC-Top-Keyword
    - manuelle Keywords
    und ruft entweder OpenAI oder Gemini auf.
    """
    title_map = page_maps.get("title", {})
    h1_map    = page_maps.get("h1", {})
    meta_map  = page_maps.get("meta", {})
    main_map  = page_maps.get("main", {})

    fields = cfg.get("fields", [])
    title = title_map.get(url, "")
    h1    = h1_map.get(url, "")
    meta  = meta_map.get(url, "")
    main  = main_map.get(url, "")
    
    # Keine Keywords mehr
    top_kws = []
    manual_kws = []

    blocks = []
    blocks.append(f"Ziel-URL: {url}")

    # Seitendaten (unterstützend, optional)

    # Priorisierung: 3) Seitendaten (unterstützend, optional)
    if "Title" in fields and title:
        blocks.append(f"Title: {title}")
    if "H1" in fields and h1:
        blocks.append(f"H1: {h1}")
    if "Meta Description" in fields and meta:
        blocks.append(f"Meta Description: {meta}")
    if "Main Content" in fields and main:
        blocks.append(
            "Main Content (zur inhaltlichen Orientierung, nicht 1:1 als Ankertext verwenden):"
        )
        blocks.append(main[:3000])

    page_info = "\n".join(blocks)

    system_prompt = (
        "Du bist ein erfahrener SEO-Experte "
        "Du erstellst präzise, natürliche und SEO-sinnvolle Ankertexte für interne Verlinkungen "
        "und hältst dich strikt an das geforderte Ausgabeformat "
    )

    user_prompt = f"""
    Erzeuge 3 unterschiedliche Ankertexte für eine interne Verlinkung zu der folgenden Zielseite.

    

    Seitendaten:

    {page_info}

    

    WICHTIG – Priorisierung für die Ankertext-Generierung:

    1. PRIORITÄT: Seitendaten (Title, H1, Meta Description)
    
    2. UNTERSTÜTZEND: Der extrahierte Main Content dient dem thematischen Verständnis der Zielseite.

    

    Ziel:

    - Der Ankertext soll das Hauptthema der Zielseite klar widerspiegeln.
    
    - Nutze nach Möglichkeit die Seitendaten (Title, H1, Meta Description).
    
    - Vermeide harte Überoptimierung (keine unnatürlich gehäuften Keywords).

    

    Erzeuge GENAU diese 3 Varianten:

    1. Kurz & präzise – 2–3 Wörter, fokussiert auf das Hauptthema.

    2. Beschreibend – 3–5 Wörter, mit etwas mehr Kontext.

    3. Handlungsorientiert – 3–5 Wörter, mit klarer Nutzen- oder Handlungsorientierung, aber ohne generische Floskeln wie „hier klicken". Es soll zum Klicken anregen ohne Clickbait zu erzeugen.

    

    Strikte Regeln:

    - Sprache: ausschließlich natürliches, korrektes Deutsch.

    - Keine generischen Phrasen wie "hier klicken", "mehr erfahren", "weiterlesen", "klicke hier" oder ähnlich.

    - Keine Emojis, keine Anführungszeichen, keine Nummerierung, keine Aufzählungszeichen.

    - Keine HTML-Tags und keine Sonderzeichen wie <, >, #, *, /.

    - Keine reinen Brand-Ankertexte; falls eine Marke vorkommt, immer mit beschreibendem Keyword kombinieren.

    - Jede der 3 Varianten muss sich in Bedeutung und Wortlaut klar von den anderen unterscheiden.

    

    

    """.strip()

    provider = cfg.get("provider", "OpenAI")
    api_key = cfg.get("openai_key") if provider == "OpenAI" else cfg.get("gemini_key")
    
    # Prüfe API-Key - KEIN FALLBACK
    if not api_key or not api_key.strip():
        raise ValueError(f"Kein API-Key für {provider} angegeben. Bitte in den Einstellungen eingeben.")
    
    text = ""
    try:
        if provider == "OpenAI":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
            except Exception:
                import openai
                client = openai.OpenAI(api_key=api_key)
    
            model_name = cfg.get("openai_model", "gpt-5.1")
            
            # Neuere Modelle (z.B. gpt-5.1) verwenden max_completion_tokens statt max_tokens
            # Versuche zuerst max_completion_tokens (für neuere Modelle)
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=200,  # ✅ NEU
                    temperature=0.5,
                )
            except Exception as e:
                # Falls max_completion_tokens nicht unterstützt wird, verwende max_tokens (für ältere Modelle)
                if "max_completion_tokens" in str(e) or "unsupported" in str(e).lower():
                    resp = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=200,  # ✅ Fallback für ältere Modelle
                        temperature=0.5,
                    )
                else:
                    raise
            
            text = resp.choices[0].message.content
            
            if not text or not text.strip():
                raise ValueError(f"OpenAI API hat leeren Text zurückgegeben. Response: {resp}")

        else:  # ✅ RICHTIG - auf gleicher Ebene wie "if provider == "OpenAI":"
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            # Versuche zuerst, verfügbare Modelle automatisch zu ermitteln
            available_models = []
            try:
                all_models = genai.list_models()
                # Filtere Modelle, die generateContent unterstützen
                for m in all_models:
                    if 'generateContent' in m.supported_generation_methods:
                        model_name = m.name.split('/')[-1]  # Extrahiere Modellname (z.B. "gemini-1.5-flash")
                        available_models.append(model_name)
            except Exception as list_error:
                # Falls list_models() fehlschlägt, verwende Fallback-Liste
                available_models = []
            
            # Priorität: 1) User-konfiguriertes Modell, 2) Automatisch gefundene Modelle, 3) Fallback-Liste
            model_candidates = []
            
            # 1. User-konfiguriertes Modell (höchste Priorität)
            user_model = cfg.get("gemini_model")
            if user_model:
                model_candidates.append(user_model)
            
            # 2. Automatisch gefundene Modelle (sortiert nach Priorität)
            priority_order = [
                "gemini-1.5-flash-latest",
                "gemini-1.5-pro-latest",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-pro",
            ]
            
            # Füge verfügbare Modelle in Prioritätsreihenfolge hinzu
            for priority_model in priority_order:
                if priority_model in available_models and priority_model not in model_candidates:
                    model_candidates.append(priority_model)
            
            # Füge alle anderen verfügbaren Modelle hinzu
            for model_name in available_models:
                if model_name not in model_candidates:
                    model_candidates.append(model_name)
            
            # 3. Fallback-Liste (falls keine Modelle gefunden wurden)
            if not model_candidates:
                model_candidates = [
                    "gemini-1.5-flash-latest",
                    "gemini-1.5-pro-latest",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "gemini-pro",
                ]
            
            model = None
            last_error = None
            tried_models = []
            
            for model_name in model_candidates:
                if not model_name:
                    continue
                tried_models.append(model_name)
                try:
                    model = genai.GenerativeModel(
                        model_name,
                        system_instruction=system_prompt
                    )
                    # Teste, ob das Modell funktioniert
                    resp = model.generate_content(user_prompt)
                    text = resp.text
                    
                    if not text or not text.strip():
                        raise ValueError(f"Gemini API hat leeren Text zurückgegeben. Response: {resp}")
                    
                    # Erfolg - verwende dieses Modell
                    break
                except Exception as e:
                    last_error = e
                    model = None
                    continue
            
            if model is None:
                # Erstelle detaillierte Fehlermeldung
                error_msg = f"Kein verfügbares Gemini-Modell gefunden.\n\n"
                error_msg += f"Versuchte Modelle: {tried_models}\n\n"
                
                if available_models:
                    error_msg += f"Verfügbare Modelle (laut API): {available_models}\n\n"
                else:
                    error_msg += "Konnte verfügbare Modelle nicht automatisch ermitteln.\n\n"
                
                error_msg += f"Letzter Fehler: {str(last_error)}\n\n"
                error_msg += "Bitte prüfe:\n"
                error_msg += "- API-Key ist korrekt eingegeben\n"
                error_msg += "- API-Key hat die nötigen Berechtigungen\n"
                error_msg += "- Internetverbindung ist vorhanden"
                
                raise RuntimeError(error_msg) from last_error
    
    except Exception as e:
        # Fehler weiterwerfen mit detaillierter Info
        error_details = {
            "provider": provider,
            "url": url,
            "error": str(e),
            "error_type": type(e).__name__,
        }
        # Zeige auch die API-Response, falls vorhanden
        if 'resp' in locals():
            try:
                error_details["api_response"] = str(resp)[:500]
            except:
                pass
        
        # Logge für Debugging
        import sys
        print(f"ERROR Ankertext-Generierung: {error_details}", file=sys.stderr)
        
        # Werfe Fehler weiter (kein Fallback)
        raise RuntimeError(
            f"Fehler bei Ankertext-Generierung für {url} ({provider}): {str(e)}\n"
            f"Bitte prüfe:\n"
            f"- API-Key ist korrekt eingegeben\n"
            f"- API-Key hat die nötigen Berechtigungen\n"
            f"- Internetverbindung ist vorhanden\n"
            f"- Modell-Name ist korrekt (z.B. 'gpt-5.1' für OpenAI)"
        ) from e
    
    # Parse die Antwort
    parts = [p.strip() for p in str(text).split("|||") if p.strip()]
    
    # Wenn weniger als 3 Teile, versuche alternative Trennzeichen
    if len(parts) < 3:
        # Versuche andere Trennzeichen
        for sep in [" | ", " |", "| ", "\n", " - ", " – "]:
            parts_alt = [p.strip() for p in str(text).split(sep) if p.strip()]
            if len(parts_alt) >= 3:
                parts = parts_alt[:3]
                break
    
    if len(parts) < 3:
        # Fehler: Antwort hat nicht das erwartete Format
        raise ValueError(
            f"API-Response hat nicht das erwartete Format (3 Varianten mit ||| getrennt).\n"
            f"Erhaltene Response: {text[:500]}\n"
            f"Gefundene Teile: {len(parts)} ({parts})"
        )
    
    # Entferne mögliche Marken und Pipe-Zeichen aus den Ankertexten
    cleaned_parts = []
    for p in parts[:3]:
        # Entferne Marken und Pipe-Zeichen
        cleaned = remove_brand_from_text(p)
        # Entferne auch andere unerwünschte Zeichen (aber nicht |||)
        cleaned = re.sub(r'[|]', '', cleaned).strip()
        if cleaned:
            cleaned_parts.append(cleaned)
    
    if len(cleaned_parts) < 3:
        raise ValueError(
            f"Nach Bereinigung weniger als 3 gültige Ankertexte erhalten.\n"
            f"Original Response: {text[:500]}\n"
            f"Bereinigte Teile: {cleaned_parts}"
        )
    
    return cleaned_parts[0], cleaned_parts[1], cleaned_parts[2]



# =============================
# Hilfe / Tool-Dokumentation (Expander) – aktualisiert
# =============================
with st.expander("ℹ️ Was du mit dem Tool machen kannst und wie du es nutzt", expanded=False):
    st.markdown("""
## Was macht ONE Link Intelligence?

**ONE Link Intelligence** bietet vier Analysen, die deine interne Verlinkung datengetrieben verbessern:

1) **Interne Verlinkungsmöglichkeiten (Analyse 1)**  
   - Findet semantisch passende interne Links (auf Basis von Embeddings oder bereitgestellter „Related URLs“).
   - Zeigt bestehende (Content-)Links und bewertet linkgebende URLs nach **Linkpotenzial**.

2) **Potenziell zu entfernende Links (Analyse 2)**  
   - Identifiziert schwache/unpassende Links (niedrige semantische Ähnlichkeit).

3) **Gems & SEO-Potenziallinks (Analyse 3)**  
   - Ermittelt starke linkgebende URLs („Gems“) anhand des Linkpotenzials.
   - Priorisiert Link-Ziele nach deren **Linkbedarf**
   - Ergebnis: „Cheat-Sheet“ mit wertvollen, noch nicht gesetzten Content-Links.

4) **Ankertext-Analyse (Analyse 4)** 
   - **Over-Anchor-Check:** Listet **alle Anchors ≥ 200** Vorkommen je Ziel-URL (inkl. Bild-Links via ALT).
   - **GSC-Query-Coverage (Top-20 % je URL):** Prüft, ob Top-Suchanfragen (nach Klicks/Impr) als Anker für URL vorkommen   

### Wie kann ich das Tool konkret nutzen?
- Analysen auswählen (*Dropdown*)
- Es öffnen sich automatisch die Masken für die benötigten **Datei-Uploads im Hauptbereich** und die **Sidebar mit Detail-Einstellungen** und Feinjustierungen (Gewichtungen, Schwellen, Visualisierung etc.)
""")

# =====================================================================
# NEU: Analyse-Auswahl im Hauptbereich + Upload-Center (nach Analysen getrennt)
# =====================================================================
A1_NAME = "Interne Verlinkungsmöglichkeiten finden (A1)"
A2_NAME = "Unpassende interne Links entfernen (A2)"
A3_NAME = "SEO-Potenziallinks finden (A3)"
A4_NAME = "Ankertexte analysieren (A4)"



st.markdown("---")
st.header("Welche Analysen möchtest du durchführen?")
selected_analyses = st.multiselect(
    "Mehrfachauswahl möglich",
    options=[A1_NAME, A2_NAME, A3_NAME, A4_NAME],
    default=[],
)

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    if selected_analyses:
        st.header("Einstellungen")

        # Gemeinsame Settings (A1/A2/A3): Backend & Linkpotenzial-Gewichte
        if any(a in selected_analyses for a in [A1_NAME, A2_NAME, A3_NAME]):
            backend = st.radio(
                "Matching-Backend (Auto-Switch bei Bedarf)",
                ["Exakt (NumPy)", "Schnell (FAISS)"],
                index=0,
                horizontal=True,
                help=("Bestimmt, wie semantische Nachbarn ermittelt werden (Cosine Similarity). "
                      "Auto-Switch: Wenn NumPy zu viel RAM braucht oder FAISS fehlt, wird automatisch umgeschaltet.")
            )
            if '_faiss_available' in globals() and callable(_faiss_available) and not _faiss_available():
                st.caption("FAISS ist hier nicht installiert – Auto-Switch nutzt ggf. NumPy.")

            st.subheader("Gewichtung (Linkpotenzial)")
            st.caption("Das Linkpotenzial gibt Aufschluss über die Lukrativität einer URL als Linkgeber.")
            w_ils = st.slider(
                "Interner Link Score", 0.0, 1.0, 0.30, 0.01,
                help="URLs, die selbst bereits intern stark verlinkt sind, werden priorisiert."
            )
            w_pr  = st.slider(
                "PageRank-Horder-Score", 0.0, 1.0, 0.35, 0.01,
                help="URLs, die vergleichsweise viele eingehende Links haben, aber nur wenige ausgehende, wird eine höhere Lukrativität zugeschrieben."
            )
            w_rd  = st.slider(
                "Referring Domains", 0.0, 1.0, 0.20, 0.01,
                help="URLs, die von vielen verschiedenen externen Domains verlinkt werden, soll eine höhere Gewichtung zugeschrieben werden."
            )
            w_bl  = st.slider(
                "Backlinks", 0.0, 1.0, 0.15, 0.01,
                help="Gewichtung für die Anzahl der Backlinks einer URL in der Linkpotenzial-Berechnung."
            )
            w_sum = w_ils + w_pr + w_rd + w_bl
            if not math.isclose(w_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3) and w_sum > 0:
                w_ils, w_pr, w_rd, w_bl = (w_ils/w_sum, w_pr/w_sum, w_rd/w_sum, w_bl/w_sum)
                st.caption(f"Gewichtungen wurden auf 1.0 normalisiert (Summe war {w_sum:.2f}).")
                st.caption(f"Aktuelle Gewichtungen – ILS: {w_ils:.2f}, PR: {w_pr:.2f}, RD: {w_rd:.2f}, BL: {w_bl:.2f}")

            st.subheader("Schwellen & Limits (Related URLs)")
            sim_threshold = st.slider(
                "Ähnlichkeitsschwelle", 0.0, 1.0, 0.80, 0.01,
                help="Nur Linkkandidaten oberhalb dieser semantischen Ähnlichkeit (0–1) werden berücksichtigt."
            )
            max_related   = st.number_input(
                "Anzahl Related URLs", min_value=1, max_value=50, value=10, step=1,
                help="Begrenze, wie viele semantisch verwandte URLs je Linkziel in der Analyse berücksichtigt werden sollen."
            )

            st.subheader("Scope der URL-Auswahl")
            scope_mode = st.radio(
                "Welche URLs sollen in die Analysen einbezogen werden?",
                [
                    "Alle URLs (Standard)",
                    "Regex-basiert (nur passende URLs)",
                ],
                index=0,
                key="scope_mode",
                help=(
                    "Alle URLs: Verhalten wie bisher.\n"
                    "Regex-basiert: Ziel- und Linkgeber-URLs jeweils per Regex definieren "
                    "(mehrere Regeln möglich, eine pro Zeile)."
                ),
            )
            if scope_mode == "Regex-basiert (nur passende URLs)":
                st.text_area(
                    "Regex für Ziel-URLs (eine pro Zeile, top-down)",
                    value="",
                    key="scope_regex_target_text",
                    help=(
                        "Nur URLs, die mindestens auf eine dieser Regeln matchen, "
                        "können als Ziel-URL in den Analysen auftreten.\n"
                        "Beispiele:\n"
                        r"^https://kaipara\.de/wolle-guide/.*"
                    ),
                )
                st.text_area(
                    "Regex für Related-URLs / Linkgeber (eine pro Zeile, top-down)",
                    value="",
                    key="scope_regex_source_text",
                    help=(
                        "Nur URLs, die mindestens auf eine dieser Regeln matchen, "
                        "können als Linkgeber / Related-URL auftreten.\n"
                        "Beispiele:\n"
                        r"^https://kaipara\.de/herren/.*\n"
                        r"^https://kaipara\.de/.*/shirts/.*"
                    ),
                )

        # A1 – KI-Ankertexte (optional)
        if A1_NAME in selected_analyses:
            st.markdown("---")
            st.subheader("Ankertext-Vorschläge (optional, A1)")

            enable_anchor_ai = st.checkbox(
                "KI-Ankertext-Vorschläge für Ziel-URLs generieren",
                value=False,
                key="a1_enable_anchor_ai",
                help=(
                    "Wenn aktiviert, werden für jede Ziel-URL in Analyse 1 drei "
                    "Ankertext-Vorschläge erzeugt und "
                    "als zusätzliche Spalten ausgegeben."
                ),
            )

            if enable_anchor_ai:
                anchor_provider = st.radio(
                    "Welchen Anbieter möchtest du für die KI-Ankertexte nutzen?",
                    ["OpenAI", "Gemini"],
                    index=0,
                    key="a1_anchor_provider",
                    help="Wähle, ob die Ankertexte über OpenAI oder Google Gemini generiert werden sollen.",
                )

                if anchor_provider == "OpenAI":
                    st.text_input(
                        "OpenAI API Key",
                        type="password",
                        key="a1_openai_api_key",
                        help="Dein OpenAI API Key (wird nur in dieser Session im Speicher gehalten).",
                    )
                else:
                    st.text_input(
                        "Gemini API Key",
                        type="password",
                        key="a1_gemini_api_key",
                        help="Dein Gemini API Key (wird nur in dieser Session im Speicher gehalten).",
                    )

                st.markdown("**Welche Seitendaten sollen in den KI-Prompt zur Ankertext-Generierung einfließen?**")
                st.multiselect(
                    "Seitendaten für die Ankertext-Generierung",
                    options=["Title", "H1", "Meta Description", "Main Content"],
                    default=[],
                    key="a1_prompt_fields",
                    help=(
                        "Diese Felder werden (falls in der Crawl-Datei vorhanden) in den Prompt aufgenommen. "
                        "Main Content dient primär dem thematischen Verständnis der Ziel-URL – "
                        "der Text wird nicht 1:1 als Anker verwendet."
                    ),
                )

                


        # ----------------
        # A2 – eigene Sektion
        # ----------------
        if A2_NAME in selected_analyses:
            if len(selected_analyses) > 1:
                st.markdown("---")
            st.subheader("Einstellungen – Unpassende interne Links entfernen A2")
            st.caption("Schwellen & Filter für potenziell unpassende Links")
            not_similar_threshold = st.slider(
                "Unähnlichkeitsschwelle (schwache Links)", 0.0, 1.0, 0.60, 0.01,
                key="a2_not_sim",
                help="Links mit Similarity ≤ Schwelle gelten als unpassend."
            )
            only_content_links = st.checkbox(
                "Nur Contentlinks berücksichtigen", value=False, key="a2_only_content",
                help="Blendet Navigations-/Footerlinks aus. Es werden nur Links aus dem Content berücksichtigt."
            )
            backlink_weight_2x = st.checkbox(
                "Backlinks/Ref. Domains doppelt gewichten", value=False, key="a2_weight2x",
                help="Erhöht den negativen Einfluss externer Signale bei der Waster-Bewertung (stärkere Priorisierung schwacher Quellen)."
            )
        else:
            st.session_state.setdefault("a2_not_sim", 0.60)
            st.session_state.setdefault("a2_weight2x", False)
            st.session_state.setdefault("a2_only_content", False)

        # ----------------
        # A3 – komplette Steuerung in die Sidebar verlegt
        # ----------------
        if A3_NAME in selected_analyses:
            if len(selected_analyses) > 1:
                st.markdown("---")
            st.subheader("Einstellungen – SEO-Potenziallinks finden A3")
            st.caption("Gems bestimmen, Linkbedarf gewichten, Sortierung steuern.")

            gem_pct = st.slider(
                "Anteil starker Linkgeber (Top-X %)", 1, 30, 10, 1, key="a3_gem_pct",
                help="Definiert, welcher Anteil der URLs mit dem höchsten Linkpotenzial als Gems (starke Linkgeber) gilt."
            )
            max_targets_per_gem = st.number_input(
                "Top-Ziele je Gem", min_value=1, max_value=50, value=10, step=1, key="a3_max_targets",
                help="Begrenzt die Anzahl empfohlener Ziel-URLs je Gem."
            )

            st.markdown("**Linkbedarf-Gewichtung für Zielseiten**")
            col1, col2 = st.columns(2)
            with col1:
                w_lihd = st.slider(
                    "Gewicht: Hidden Champions", 0.0, 1.0, 0.30, 0.05, key="a3_w_lihd",
                    help="Mehr Nachfrage (Impressions) + schwacher interner Link-Score ⇒ höherer Linkbedarf (GSC-Daten erforderlich)."
                )
                w_orph = st.slider(
                    "Gewicht: Mauerblümchen", 0.0, 1.0, 0.10, 0.05, key="a3_w_orph",
                    help="Orphan/Thin-URLs werden höher priorisiert (geringe interne Inlinks)."
                )
                thin_k = st.slider(
                    "Thin-Schwelle (Inlinks ≤ K)", 0, 10, 2, 1, key="a3_thin_k",
                    help="Grenzwert, ab dem eine Seite als 'Thin' gilt."
                )
            with col2:
                w_def = st.slider(
                    "Gewicht: Semantische Linklücke", 0.0, 1.0, 0.30, 0.05, key="a3_w_def",
                    help="Fehlende Content-Links zwischen semantisch ähnlichen Seiten erhöhen den Linkbedarf."
                )
                w_rank = st.slider(
                    "Gewicht: Sprungbrett-URLs", 0.0, 1.0, 0.30, 0.05, key="a3_w_rank",
                    help="URLs im Ranking-Sweet-Spot (z. B. Position 8–20) werden als Hebel priorisiert (GSC-Position nötig)."
                )
                rank_minmax = st.slider(
                    "Ranking-Sweet-Spot (Position)", 1, 50, (8, 20), 1, key="a3_rank_minmax",
                    help="Positionsbereich, in dem sich Sprungbrett-URLs befinden sollen."
                )

            with st.expander("Offpage-Einfluss (Backlinks & Ref. Domains)", expanded=False):
                offpage_damp_enabled = st.checkbox(
                    "Offpage-Dämpfung anwenden", value=True, key="a3_offpage_enable",
                    help="Fakturiert Backlinks und Referring Domains mit ein (starke Offpage-Signale reduzieren den Linkbedarf einer Ziel-URL)."
                )
                beta_offpage = st.slider(
                    "Stärke der Dämpfung", 0.0, 1.0, 0.5, 0.05,
                    disabled=not st.session_state.get("a3_offpage_enable", False),
                    key="a3_offpage_beta",
                    help="Wie stark Offpage-Signale den Linkbedarf dämpfen (0 = aus, 1 = stark)."
                )

            with st.expander("Reihenfolge der Empfehlungen", expanded=False):
                sort_labels = {
                    "rank_mix":"Mix (inhaltliche Nähe + Linkbedarf)",
                    "prio_only":"Nur Linkbedarf",
                    "sim_only":"Nur inhaltliche Nähe"
                }
                sort_choice = st.radio(
                    "Sortierung", options=["rank_mix","prio_only","sim_only"], index=0,
                    format_func=lambda k: sort_labels.get(k, k), horizontal=False, key="a3_sort_choice",
                    help="Steuert die Reihung der Empfehlungen."
                )
                alpha_mix = st.slider(
                    "Gewichtung: inhaltliche Nähe vs. Linkbedarf", 0.0, 1.0, 0.5, 0.05, key="a3_alpha_mix",
                    help="α = Anteil inhaltliche Nähe; (1−α) = Anteil Linkbedarf."
                )

        # ----------------
        # A4 – Sidebar (umstrukturiert mit Switches)
        # ----------------
        if A4_NAME in selected_analyses:
            if len(selected_analyses) > 1:
                st.markdown("---")
            st.subheader("Einstellungen – Ankertexte analysieren A4")
            st.caption("Hier sind mehrere Detail-Analysen möglich – diese können nachfolgend aktiviert oder deaktiviert werden.")
            # NEU: Link-Scope – alle Links vs. nur Content-Links
            a4_link_scope = st.radio(
                "Welche Links sollen bei den Ankertext-Analysen berücksichtigt werden?",
                ["Alle Links (Standard)", "Nur Content-Links"],
                index=0,
                key="a4_link_scope",
                help=(
                    "Basis ist die Spalte 'Link position'"
                    "in der All-Inlinks-Datei. "
                    "'Nur Content-Links' filtert auf Linkpositionen, die z. B. 'Inhalt', 'Content', 'Body' enthalten. Navi- und Footerlinks z. B. werden rausgefiltert."
                ),
            )

            
            # Switch für Over-Anchor-Check
            enable_over_anchor = st.checkbox(
                "Over-Anchor-Check aktivieren", 
                value=True, 
                key="a4_enable_over_anchor",
                help="Im Google Leak gibt es Hinweise darauf, dass ein und derselbe Ankertext pro URL nur 200-mal gezählt wird. Ab dem 201. Link mit diesem Ankertext wird der Ankertext ignoriert. Hier analysieren wir, welche Ankertexte mehr als 200-mal für die gleiche URL vorkommen."
            )
            
            if enable_over_anchor:
                st.markdown("**Over-Anchor-Check**")
                st.caption("Identifiziert URLs, die mehr als 200 mal mit demselben Ankertext verlinkt werden.")
            
                col_o1, _ = st.columns(2)
                with col_o1:
                    top_anchor_abs = st.number_input(
                        "Schwelle identischer Anker (absolut)",
                        min_value=1,
                        value=200,
                        step=10,
                        key="a4_top_anchor_abs",
                        help="Ab wie vielen identischen Ankern eine URL als Over-Anchor-Fall gilt."
                    )
            
            else:
                # Default setzen, falls disabled
                st.session_state.setdefault("a4_top_anchor_abs", 200)

                

            # Abstand / Trennlinie zur nächsten Unteranalyse
            st.markdown(
                "<div style='margin:18px 0; border-bottom:1px solid #eee;'></div>",
                unsafe_allow_html=True,
            )
            
            # Switch für GSC-Query-Coverage
            enable_gsc_coverage = st.checkbox(
                "Search Console Query Coverage bei Ankertexten aktivieren",
                value=True,
                key="a4_enable_gsc_coverage",
                help="Prüft, ob je URL die Top-Queries aus Google Search Console (nach Impressionen oder Klicks) als Ankertexte vorhanden sind."
            )
            
            if enable_gsc_coverage:
                st.markdown("**Search Console Query Coverage bei Ankertexten**")
                st.caption("Gleicht die Top 20 % der Suchanfragen einer URL mit den vorhandenen Ankertexten ab.")

                # Brand / Non-Brand
                brand_mode = st.radio(
                    "Sollen auch Brand-Suchanfragen bei dieser Analyse berücksichtigt werden?",
                    ["Nur Non-Brand", "Nur Brand", "Beides"],
                    index=0,
                    horizontal=True,
                    key="a4_brand_mode",
                )

                brand_text = st.text_area(
                    "Brand-Schreibweisen (eine pro Zeile oder komma-getrennt)",
                    value=st.session_state.get("a4_brand_text", ""),
                    key="a4_brand_text",
                )
                brand_file = st.file_uploader(
                    "Optional: Brand-Liste (1 Spalte)",
                    type=["csv", "xlsx", "xlsm", "xls"],
                    key="a4_brand_file",
                )
                auto_variants = st.checkbox(
                    "Branded Keywords auch als Brand-Keywords behandeln? (Kombis wie 'marke keyword', 'keyword marke', 'marke-keyword' usw.)",
                    value=st.session_state.get("a4_auto_variants", True),
                    key="a4_auto_variants",
                )

                # Relevanzgrundlage
                metric_choice = st.radio(
                    "Sollen die Top 20 % Suchanfragen auf Basis der Klicks oder Impressionen analysiert werden?",
                    ["Impressions", "Clicks"],
                    index=0,
                    horizontal=True,
                    key="a4_metric_choice",
                )

                st.caption(
                    "Soll der Abgleich der Search Console Queries mit den Ankertexten als Exact Match "
                    "oder auf Basis semantischer Ähnlichkeit erfolgen?"
                )

                check_exact = st.checkbox("Exact Match prüfen", value=True, key="a4_cov_exact")
                check_embed = st.checkbox("Embedding Match prüfen", value=True, key="a4_cov_embed")

                embed_model_name = st.selectbox(
                    "Sentence Transformer Modell (Embedding-Modell)",
                    [
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "sentence-transformers/all-MiniLM-L12-v2",
                        "sentence-transformers/all-mpnet-base-v2",
                    ],
                    index=0,
                    key="a4_embed_model",
                )
                embed_thresh = st.slider(
                    "Cosine-Schwelle (Embedding)",
                    0.50, 0.95, 0.75, 0.01,
                    key="a4_embed_thresh",
                )

                help_text_schwellen = (
                    "Mit den Schwellen & Filtern reduzierst du Rauschen und fokussierst die Analyse auf wirklich relevante Suchanfragen:\n\n"
                    "• Mindest-Klicks/Query – wird nur angewendet, wenn oben Clicks ausgewählt ist.\n"
                    "• Mindest-Impressions/Query – wird nur angewendet, wenn oben Impressions ausgewählt ist.\n"
                    "• Top-N Queries pro URL – zusätzlicher Deckel nach der Top-20-%-Auswahl.\n\n"
                    "Hinweise:\n"
                    "– Die Auswahl Impressions vs. Clicks steuert, welche Schwelle greift.\n"
                    "– Erst werden Marke/Non-Brand und Mindestwerte gefiltert, dann die Top-20-% berechnet."
                )

                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    min_clicks = st.number_input(
                        "Mindest-Klicks/Query",
                        min_value=0,
                        value=50,
                        step=10,
                        key="a4_min_clicks",
                        help=help_text_schwellen,
                    )
                with col_s2:
                    min_impr = st.number_input(
                        "Mindest-Impressions/Query",
                        min_value=0,
                        value=500,
                        step=50,
                        key="a4_min_impr",
                        help=help_text_schwellen,
                    )
                with col_s3:
                    topN_default = st.number_input(
                        "Top-N Queries pro URL (zusätzliche Bedingung)",
                        min_value=0,  # 0 = kein zusätzlicher Deckel
                        value=st.session_state.get("a4_topN", 0),
                        step=1,
                        key="a4_topN",
                    )

            else:
                # Defaults setzen, wenn GSC-Coverage deaktiviert ist
                st.session_state.setdefault("a4_brand_mode", "Nur Non-Brand")
                st.session_state.setdefault("a4_brand_text", "")
                st.session_state.setdefault("a4_auto_variants", True)
                st.session_state.setdefault("a4_metric_choice", "Impressions")
                st.session_state.setdefault("a4_check_exact", True)
                st.session_state.setdefault("a4_check_embed", True)
                st.session_state.setdefault("a4_embed_model", "sentence-transformers/all-MiniLM-L6-v2")
                st.session_state.setdefault("a4_embed_thresh", 0.75)
                st.session_state.setdefault("a4_min_clicks", 50)
                st.session_state.setdefault("a4_min_impr", 500)
                st.session_state.setdefault("a4_topN", 0)
                st.session_state.setdefault("a4_over_anchor_mode", "Absolut")


            
            # Abstand / Trennlinie zur nächsten Unteranalyse
            st.markdown(
                "<div style='margin:18px 0; border-bottom:1px solid #eee;'></div>",
                unsafe_allow_html=True,
            )
            
            # Switch für Keyword-Coverage mit eigener Keyword-Liste (UNABHÄNGIG von GSC)
            enable_kw_coverage = st.checkbox(
                "Keyword-Ankertext-Coverage mit eigener Keyword-Liste aktivieren",
                value=True,
                key="a4_enable_kw_coverage",
                help=(
                    "Prüft für eine manuell hochgeladene Keyword-Liste, "
                    "ob die Keywords als Ankertexte vorkommen und welchen Anteil sie am gesamten Anchor-Inventar haben."
                )
            )
            
            if enable_kw_coverage:
                st.markdown("**Keyword-Ankertext-Coverage mit eigener Keyword-Liste**")
                st.caption(
                    "Prüft für jede URL, ob deine definierten Keywords als Ankertexte vorkommen "
                    "und wie hoch ihr Anteil am gesamten Anchor-Inventar ist."
                )



            

            # Abstand / Trennlinie zur nächsten Unteranalyse
            st.markdown(
                "<div style='margin:18px 0; border-bottom:1px solid #eee;'></div>",
                unsafe_allow_html=True,
            )

           

            # --- Ankertext-Matrix (A4) ---
            st.markdown("**Ankertext-Matrix / Anchor-Inventar**")
            st.caption(
                "Lasse dir je Ziel-URL die Ankertexte als Tabelle (Wide/Long) ausgeben. "
                "Optional kannst du externe Offpage-Ankertexte mit einbeziehen."
            )


            
            # NEU: Offpage-Anker einbeziehen
            include_offpage_anchors = st.checkbox(
                "Auch Ankertexte aus externen Backlinks berücksichtigen (Offpage-Datei)",
                value=False,
                key="a4_include_offpage_anchors",
                help=(
                    "Wenn aktiviert, werden Ankertexte aus einer separaten Offpage-Datei "
                    "(z. B. Backlink-Export mit Anchor-Text) zusätzlich zu den internen Ankertexten gezählt."
                )
            )
            
           
            # ✅ NEU: Switch für URL-Ankertext-Matrix (Wide)
            enable_anchor_matrix = st.checkbox(
                "URL-Ankertext-Matrix (Wide) anzeigen",
                value=True,
                key="a4_enable_anchor_matrix",
                help="Zeigt je Ziel-URL die Ankertexte in breitem Format (inkl. CSV-Export)."
            )
            
            # ✅ NEU: Switch für URL-Ankertext-Matrix (Long)
            enable_anchor_matrix_long = st.checkbox(
                "URL-Ankertext-Matrix (Long-Format) anzeigen",
                value=False,
                key="a4_enable_anchor_matrix_long",
                help="Zeigt je Ziel-URL alle Ankertexte untereinander (Long-Format, inkl. CSV-Export)."
            )

            st.markdown(
                "<div style='margin:18px 0; border-bottom:1px solid #eee;'></div>",
                unsafe_allow_html=True,
            )
            
            # ✅ NEU: Switch + Settings für Shared-Ankertexte
            enable_shared_sidebar = st.checkbox(
                "Shared-Ankertexte anzeigen",
                value=True,
                key="a4_shared_enable",
                help="Zeigt Ankertexte, die auf mehreren unterschiedlichen Ziel-URLs verwendet werden."
            )

            col_shs1, col_shs2 = st.columns(2)
            with col_shs1:
                min_urls_per_anchor_sidebar = st.number_input(
                    "Shared: mind. N Ziel-URLs",
                    min_value=2, value=2, step=1, key="a4_shared_min_urls",
                    help="Nur Anker zeigen, die auf mindestens N Ziel-URLs vorkommen."
                )
            with col_shs2:
                ignore_nav_sidebar = st.checkbox(
                    "Shared: Navigative Anker ausschließen",
                    value=True, key="a4_shared_ignore_nav",
                    help="Blendet generische/navigative Anker wie 'hier', 'mehr' etc. aus."
                )

            st.markdown(
                "<div style='margin:18px 0; border-bottom:1px solid #eee;'></div>",
                unsafe_allow_html=True,
            )
            
            st.markdown(
                "<div style='margin:18px 0; border-bottom:1px solid #eee;'></div>",
                unsafe_allow_html=True,
            )
            
            # --- Visualisierung (Treemap) ---
            st.markdown("**Visualisierung (Treemap)**")
            st.caption(
                "Steuere die Treemap-Visualisierung der Ankertexte je Ziel-URL. "
                "Die Treemap nutzt dieselben Anchor-Daten wie die Ankertext-Matrix."
            )
            
            show_treemap = st.checkbox(
                "Treemap-Visualisierung aktivieren",
                value=True,
                key="a4_show_treemap",
                help=(
                    "Schaltet die Treemap ein/aus. Die Treemap zeigt je Ziel-URL die häufigsten Ankertexte. "
                    "Grundlage sind die Anker aus All Inlinks und – falls aktiviert – aus der Offpage-Ankerdatei."
                ),
            )

            treemap_topK = st.number_input(
                "Treemap: Top-K Anchors anzeigen",
                min_value=3,
                max_value=50,
                value=12,
                step=1,
                key="a4_treemap_topk",
                help=(
                    "Begrenzt **nur die Treemap** auf die K häufigsten Anker pro Ziel-URL. "
                    "Die nachfolgende Analyse/Exports enthalten **immer alle** Anker."
                )
            )
            
            # Auswahl, für welche URLs Treemaps erzeugt werden sollen
            treemap_url_mode = st.radio(
                "Für welche URLs sollen Treemaps erzeugt werden?",
                ["Alle URLs", "Ausgewählte URLs"],
                index=0,
                key="a4_treemap_url_mode",
                help="Bestimmt, ob für alle Ziel-URLs oder nur für eine Auswahl Treemaps gebaut werden."
            )

            st.markdown(
                "<div style='margin:18px 0; border-bottom:1px solid #eee;'></div>",
                unsafe_allow_html=True,
            )
            
            if treemap_url_mode == "Ausgewählte URLs":
                # Versuche, bekannte URLs aus dem Anchor-Inventar zu laden (falls A4 schon einmal lief)
                anchor_inv_check_sidebar = st.session_state.get("_anchor_inv_vis", pd.DataFrame())
                if not anchor_inv_check_sidebar.empty and "target" in anchor_inv_check_sidebar.columns:
                    url_options = sorted(anchor_inv_check_sidebar["target"].astype(str).unique())
                    selected_urls_for_treemap = st.multiselect(
                        "URLs für Treemap auswählen",
                        options=url_options,
                        format_func=lambda u: disp(u),
                        key="a4_treemap_selected_urls",
                        help="Wähle eine oder mehrere Ziel-URLs, für die im Hauptbereich je eine Treemap erzeugt wird."
                    )
                else:
                    st.caption("Noch keine Ziel-URLs bekannt – die Auswahl wird nutzbar, nachdem A4 einmal gelaufen ist.")
                    st.session_state["a4_treemap_selected_urls"] = []
            else:
                # Modus 'Alle URLs' → Leere Liste = bedeutet später: nimm alle
                st.session_state["a4_treemap_selected_urls"] = []


    else:
        st.caption("Wähle oben mindestens eine Analyse aus, um Einstellungen zu sehen.")


# ===============================
# Bedarf je Analyse für Uploads
# ===============================
# Embeddings / Related werden jetzt auch für A5 & A6 genutzt
needs_embeddings_or_related = any(
    a in selected_analyses for a in [A1_NAME, A2_NAME, A3_NAME]
)

needs_inlinks_a1 = A1_NAME in selected_analyses
needs_inlinks_a2 = A2_NAME in selected_analyses
needs_inlinks_a3 = A3_NAME in selected_analyses
needs_inlinks_a4 = A4_NAME in selected_analyses

needs_inlinks = any([
    needs_inlinks_a1,
    needs_inlinks_a2,
    needs_inlinks_a3,
    needs_inlinks_a4,
])

needs_metrics_a1 = A1_NAME in selected_analyses
needs_metrics_a2 = A2_NAME in selected_analyses
needs_metrics_a3 = A3_NAME in selected_analyses
needs_metrics = needs_metrics_a1 or needs_metrics_a2 or needs_metrics_a3

needs_backlinks_a1 = A1_NAME in selected_analyses
needs_backlinks_a2 = A2_NAME in selected_analyses
needs_backlinks_a3 = A3_NAME in selected_analyses
needs_backlinks = needs_backlinks_a1 or needs_backlinks_a2 or needs_backlinks_a3

needs_gsc_a3 = A3_NAME in selected_analyses  # optional
needs_gsc_a4 = A4_NAME in selected_analyses  # benötigt in A4-Teil für Coverage



# ===============================
# Upload-Center: nach Analysen getrennt + "Für mehrere Analysen benötigt"
# ===============================
if selected_analyses:
    st.markdown("---")
    st.subheader("Benötigte Dateien hochladen")

    # Sammle Bedarfe je Upload-Typ
    required_sets = {
        "URLs + Embeddings": {
            "analyses": [
                a for a in [A1_NAME, A2_NAME, A3_NAME]
                if a in selected_analyses and needs_embeddings_or_related
            ]
        },
        "Related URLs": {
            "analyses": [
                a for a in [A1_NAME, A2_NAME, A3_NAME]
                if a in selected_analyses and needs_embeddings_or_related
            ]
        },
        "All Inlinks": {
            "analyses": [
                a for a in [A1_NAME, A2_NAME, A3_NAME, A4_NAME]
                if a in selected_analyses and needs_inlinks
            ]
        },
        "Linkmetriken": {
            "analyses": [
                a for a in [A1_NAME, A2_NAME, A3_NAME]
                if a in selected_analyses and needs_metrics
            ]
        },
        "Backlinks": {
            "analyses": [
                a for a in [A1_NAME, A2_NAME, A3_NAME]
                if a in selected_analyses and needs_backlinks
            ]
        },
        "Search Console": {"analyses": [A3_NAME] if needs_gsc_a3 else []},
    }

    # Ermitteln, welche Uploads in ≥ 2 Analysen identisch gebraucht werden
    shared_uploads = [k for k, v in required_sets.items() if len(v["analyses"]) >= 2]

    emb_df = related_df = inlinks_df = metrics_df = backlinks_df = None
    offpage_anchors_df = None  # <– NEU
    gsc_df_loaded = None
    kw_df_a4 = None

    def _read_up(label: str, uploader, required: bool):
        df = None
        if uploader is not None:
            try:
                df = read_any_file_cached(getattr(uploader, "name", ""), uploader.getvalue())
            except Exception as e:
                st.error(f"Fehler beim Lesen von {getattr(uploader, 'name', 'Datei')}: {e}")
        if required and df is None:
            st.info(f"Bitte lade die Datei für **{label}**.")
        return df

    # Hilfstexte je Upload
    HELP_EMB = ("Struktur: mindestens **URL** + **Embedding**-Spalte. Embeddings als JSON-Array "
                "(z. B. `[0.12, 0.03, …]`) oder Zahlenliste (Komma/Whitespace/; / | getrennt). "
                "Zusätzliche Spalten werden ignoriert. Spaltenerkennung erfolgt automatisch.")
    HELP_REL = ("Struktur: genau **3 Spalten** – **Ziel-URL**, **Quell-URL**, **Similarity** (0–1). "
                "Zusätzliche Spalten werden ignoriert. Spaltenerkennung erfolgt automatisch.")
    HELP_INL = ("Export aus Screaming Frog: **Massenexport → Links → Alle Inlinks**. "
                "Spalten: Quelle/Source, Ziel/Destination, optional Position und Anchor/ALT. "
                "Spaltenerkennung erfolgt automatisch; zusätzliche Spalten werden ignoriert.")
    HELP_MET = ("Struktur: mindestens **4 Spalten** – **URL**, **Score (Interner Link Score)**, **Inlinks**, **Outlinks**. "
                "Spaltenerkennung erfolgt automatisch; zusätzliche Spalten werden ignoriert.")
    HELP_BL = ( "Du kannst hier zwei Arten von Backlink-Dateien hochladen:\n\n"
                "1) Aggregierte Backlink-Metriken (klassisch)\n"
                "   – Struktur: mindestens **URL**, **Backlinks**, **Referring Domains**.\n"
                "   – Typisch: Export aus einem SEO-Tool (z. B. pro URL bereits aggregierte Metriken).\n\n"
                "2) Offpage-Linkliste (eine Zeile = ein Backlink)\n"
                "   – Struktur: mindestens **Ziel-URL** und eine Spalte mit verweisender Domain oder Quell-URL\n"
                "     (z. B. 'Referring Domain', 'Source URL', 'Domain').\n"
                "   – Das Tool aggregiert daraus automatisch **Backlinks** und **Referring Domains** je Ziel-URL.\n\n"
                "Spaltennamen werden automatisch erkannt; zusätzliche Spalten werden ignoriert.")
    HELP_GSC_A3 = ("Struktur: **URL**, **Impressions** · optional **Clicks**, **Position**. "
                   "Spaltenerkennung erfolgt automatisch; zusätzliche Spalten werden ignoriert.")
    HELP_GSC_A4 = ("Struktur: **URL**, **Query**, **Impressions** oder **Clicks** (mind. eine der beiden). "
                   "Optional **Position**. Spaltenerkennung erfolgt automatisch.")
    HELP_A4_KW = (
    "Struktur: erste Spalte = URL. Die zugehörigen Keywords können entweder\n"
    "- in den folgenden Spalten derselben Zeile stehen (Wide-Format: URL, Keyword1, Keyword2, …)\n"
    "- oder zeilenweise untereinander mit derselben URL in Spalte 1 (Long-Format: jede Zeile = URL, Keyword).\n\n"
    "Leere Zellen werden ignoriert. Zusätzliche Spalten werden nicht ausgewertet."
)

    

    # =====================================================
    # Gemeinsame Sektion (falls mehrfach benötigt)
    # =====================================================

    # 1) Globaler Eingabemodus: URLs + Embeddings vs. Related URLs
    if needs_embeddings_or_related:
        mode = st.radio(
            "Eingabemodus",
            ["URLs + Embeddings", "Related URLs"],
            horizontal=True,
            key="emb_rel_mode_global",
            help=(
                "URLs + Embeddings: Das Tool berechnet die 'Related URLs' intern "
                "(NumPy oder FAISS, je nach Einstellung).\n"
                "Related URLs: Fertige Similarity-Tabelle (Quelle/Ziel/Score) hochladen."
            ),
        )
    else:
        mode = st.session_state.get("emb_rel_mode_global", "Related URLs")

    if shared_uploads:
        st.markdown("### Für mehrere Analysen benötigt")

    colA, colB = st.columns(2)
    mode = st.session_state.get("emb_rel_mode_global", "Related URLs")

    with colA:
        if "URLs + Embeddings" in shared_uploads and mode == "URLs + Embeddings" and needs_embeddings_or_related:
            up_emb = st.file_uploader(
                "URLs + Embeddings (CSV/Excel)",
                type=["csv", "xlsx", "xlsm", "xls"],
                key="up_emb_shared",
                help=HELP_EMB,
            )
            emb_df = _read_up("URLs + Embeddings", up_emb, required=True)

        if "Related URLs" in shared_uploads and mode == "Related URLs" and needs_embeddings_or_related:
            up_related = st.file_uploader(
                "Related URLs (CSV/Excel)",
                type=["csv", "xlsx", "xlsm", "xls"],
                key="up_related_shared",
                help=HELP_REL,
            )
            related_df = _read_up("Related URLs", up_related, required=True)

        if "All Inlinks" in shared_uploads and needs_inlinks:
            up_inlinks = st.file_uploader(
                "All Inlinks (CSV/Excel)",
                type=["csv", "xlsx", "xlsm", "xls"],
                key="up_inlinks_shared",
                help=HELP_INL,
            )
            inlinks_df = _read_up("All Inlinks", up_inlinks, required=True)

    with colB:
        if "Linkmetriken" in shared_uploads and needs_metrics:
            up_metrics = st.file_uploader(
                "Linkmetriken (CSV/Excel)",
                type=["csv", "xlsx", "xlsm", "xls"],
                key="up_metrics_shared",
                help=HELP_MET,
            )
            metrics_df = _read_up("Linkmetriken", up_metrics, required=True)

        if "Backlinks" in shared_uploads and needs_backlinks:
            up_backlinks = st.file_uploader(
                "Backlinks (CSV/Excel)",
                type=["csv", "xlsx", "xlsm", "xls"],
                key="up_backlinks_shared",
                help=HELP_BL,
            )
            backlinks_df = _read_up("Backlinks", up_backlinks, required=True)

        if "Search Console" in shared_uploads and needs_gsc_a3:
            up_gsc = st.file_uploader(
                "Search Console Daten (optional, CSV/Excel)",
                type=["csv", "xlsx", "xlsm", "xls"],
                key="up_gsc_shared",
                help=HELP_GSC_A3,
            )
            if up_gsc is not None:
                gsc_df_loaded = _read_up("Search Console", up_gsc, required=False)

    # Individuelle Sektionen pro Analyse (nur Uploads, die NICHT in shared gelandet sind)
    def upload_for_analysis(title: str, needs: List[Tuple[str, str, str]]):
        st.markdown(f"### {title}")
        cols = st.columns(2)
        bucketA, bucketB = [], []
        for i, (label, key, help_txt) in enumerate(needs):
            (bucketA if i % 2 == 0 else bucketB).append((label, key, help_txt))
        for (col, bucket) in zip(cols, [bucketA, bucketB]):
            with col:
                for label, key, help_txt in bucket:
                    up = st.file_uploader(label, type=["csv","xlsx","xlsm","xls"], key=key, help=help_txt)
                    df = _read_up(label, up, required=True)
                    yield (label, df)

else:
    # Optional: Wenn du GAR NICHTS anzeigen willst, lass den Block leer (pass)
    # oder kurze Info direkt unter der Analyse-Auswahl:
    # st.info("Bitte wähle mindestens eine Analyse, um die Upload-Sektion zu sehen.")
    pass


# A1
if A1_NAME in selected_analyses:

    needs = []
    if needs_embeddings_or_related and "URLs + Embeddings" not in shared_uploads and "Related URLs" not in shared_uploads:
        if 'mode' not in locals():
            mode = "Related URLs"
        if mode == "URLs + Embeddings":
            needs.append(("URLs + Embeddings (CSV/Excel)", "up_emb_a1", HELP_EMB))
        else:
            needs.append(("Related URLs (CSV/Excel)", "up_rel_a1", HELP_REL))
    if needs_inlinks_a1 and "All Inlinks" not in shared_uploads:
        needs.append(("All Inlinks (CSV/Excel)", "up_inlinks_a1", HELP_INL))
    if needs_metrics_a1 and "Linkmetriken" not in shared_uploads:
        needs.append(("Linkmetriken (CSV/Excel)", "up_metrics_a1", HELP_MET))
    if needs_backlinks_a1 and "Backlinks" not in shared_uploads:
        needs.append(("Backlinks (CSV/Excel)", "up_backlinks_a1", HELP_BL))

    # KI-spezifische Zusatz-Uploads nur anhängen, wenn aktiviert
    if st.session_state.get("a1_enable_anchor_ai", False):
        needs.append((
            "Crawl / Page-Details (Title, H1, Meta, Main Content) (CSV/Excel)",
            "up_crawl_a1",
            (
                "Erwartete Spaltennamen (mindestens die benötigten):\n"
                "- URL-Spalte: z. B. 'URL', 'Address'\n"
                "- Title: 'Title', 'Title 1', 'Title Tag', 'Title-Tag'\n"
                "- H1: 'H1', 'H1-1', 'H1 - 1', 'h1'\n"
                "- Meta Description: 'Meta Description', 'Meta Description 1', 'MD', 'MD 1'\n"
                "- Main Content: 'Main Content Extraction 1', 'Main Content Extraction', 'Main Content Extraction 2'\n\n"
                "Weitere Spalten werden ignoriert."
            )
        ))

        

    # Diese Schleife MUSS außerhalb des if a1_enable_anchor_ai stehen
    for (label, df) in upload_for_analysis(
        "Analyse 1 interne Verlinkungsmöglichkeiten finden – erforderliche Dateien",
        needs
    ):
        if "Embeddings" in label:
            emb_df = df
        if "Related URLs" in label:
            related_df = df
        if "Inlinks" in label:
            inlinks_df = df
        if "Linkmetriken" in label:
            metrics_df = df
        if "Backlinks" in label:
            backlinks_df = df
        if "Crawl / Page-Details" in label:
            crawl_df_a1 = df
        

# A2
if A2_NAME in selected_analyses:
    needs = []
    if needs_embeddings_or_related and "URLs + Embeddings" not in shared_uploads and "Related URLs" not in shared_uploads:
        if 'mode' not in locals():
            mode = "Related URLs"
        if mode == "URLs + Embeddings":
            needs.append(("URLs + Embeddings (CSV/Excel)", "up_emb_a2", HELP_EMB))
        else:
            needs.append(("Related URLs (CSV/Excel)", "up_rel_a2", HELP_REL))
    if needs_inlinks_a2 and "All Inlinks" not in shared_uploads:
        needs.append(("All Inlinks (CSV/Excel)", "up_inlinks_a2", HELP_INL))
    if needs_metrics_a2 and "Linkmetriken" not in shared_uploads:
        needs.append(("Linkmetriken (CSV/Excel)", "up_metrics_a2", HELP_MET))
    if needs_backlinks_a2 and "Backlinks" not in shared_uploads:
        needs.append(("Backlinks (CSV/Excel)", "up_backlinks_a2", HELP_BL))
    for (label, df) in upload_for_analysis("Analyse 2 unpassende interne Links entfernen – erforderliche Dateien", needs):
        if "Embeddings" in label: emb_df = df
        if "Related URLs" in label: related_df = df
        if "Inlinks" in label: inlinks_df = df
        if "Linkmetriken" in label: metrics_df = df
        if "Backlinks" in label: backlinks_df = df

# A3 (GSC optional)
if A3_NAME in selected_analyses:
    needs = []
    if needs_embeddings_or_related and "URLs + Embeddings" not in shared_uploads and "Related URLs" not in shared_uploads:
        if 'mode' not in locals():
            mode = "Related URLs"
        if mode == "URLs + Embeddings":
            needs.append(("URLs + Embeddings (CSV/Excel)", "up_emb_a3", HELP_EMB))
        else:
            needs.append(("Related URLs (CSV/Excel)", "up_rel_a3", HELP_REL))

    if needs_inlinks_a3 and "All Inlinks" not in shared_uploads:
        needs.append(("All Inlinks (CSV/Excel)", "up_inlinks_a3", HELP_INL))

    if needs_metrics_a3 and "Linkmetriken" not in shared_uploads:
        needs.append(("Linkmetriken (CSV/Excel)", "up_metrics_a3", HELP_MET))
    if needs_backlinks_a3 and "Backlinks" not in shared_uploads:
        needs.append(("Backlinks (CSV/Excel)", "up_backlinks_a3", HELP_BL))
    # GSC optional: ...
    if needs_gsc_a3 and "Search Console" not in shared_uploads:
        needs.append(("Search Console Daten (optional, CSV/Excel)", "up_gsc_a3", HELP_GSC_A3))

    for (label, df) in upload_for_analysis("Analyse 3 SEO-Potenziallinks finden – erforderliche Dateien", needs):
        if "Embeddings" in label: emb_df = df
        if "Related URLs" in label: related_df = df
        if "Inlinks" in label: inlinks_df = df          # 👈 NEU
        if "Linkmetriken" in label: metrics_df = df
        if "Backlinks" in label: backlinks_df = df
        if "Search Console" in label: gsc_df_loaded = df


# A4 – separate Uploads
if A4_NAME in selected_analyses:
    needs = []
    if needs_inlinks_a4 and "All Inlinks" not in shared_uploads:
        needs.append(("All Inlinks (CSV/Excel)", "up_inlinks_a4", HELP_INL))
    # GSC Upload nur wenn GSC-Coverage aktiviert ist
    if st.session_state.get("a4_enable_gsc_coverage", True):
        needs.append(("Search Console (CSV/Excel)", "up_gsc_a4", HELP_GSC_A4))

    # Eigene Keyword-Liste nur anzeigen, wenn Keyword-Coverage aktiv ist
    if st.session_state.get("a4_enable_kw_coverage", True):
        needs.append((
            "Eigene Keyword-Liste (CSV/Excel)",
            "up_kw_a4",
            HELP_A4_KW
        ))

    # Offpage-Ankerdatei nur anbieten, wenn Option aktiviert
    if st.session_state.get("a4_include_offpage_anchors", False):
        needs.append((
            "Offpage-Ankertexte (CSV/Excel)",
            "up_offpage_anchors_a4",
            "Struktur: mindestens Ziel-URL + Ankertext. "
            "Ziel-URL wird ähnlich wie in 'All Inlinks' erkannt, Ankertext über Spaltennamen wie "
            "'Anchor', 'Anchor Text', 'Anker', 'Ankertext' etc."
        ))


    


    if needs:
        for (label, df) in upload_for_analysis(
            "Analyse 4 Ankertexte analysieren – erforderliche Dateien", needs
        ):
            if "Inlinks" in label:
                inlinks_df = df
            if "Search Console" in label:
                gsc_df_loaded = df
            if "Keyword-Liste" in label:
                kw_df_a4 = df
            if "Offpage-Ankertexte" in label:
                offpage_anchors_df = df
            if "Crawl" in label:
                crawl_df_a4 = df
                # ✅ Spalten für Dropdown merken
                if df is not None and not df.empty:
                    st.session_state["a4_crawl_columns"] = [str(c).strip() for c in df.columns]
                else:
                    st.session_state["a4_crawl_columns"] = []
            if "Embeddings für semantische" in label:
                emb_df_a4 = df




# =========================================================
# Zentrale Vorverarbeitung für "All Inlinks"
# - baut _all_links und _content_links
# - kann von A2/A3/A5/A6 verwendet werden
# =========================================================
if inlinks_df is not None and "_all_links" not in st.session_state:
    inlinks_df = inlinks_df.copy()
    header = [str(c).strip() for c in inlinks_df.columns]

    src_idx = find_column_index(header, POSSIBLE_SOURCE)
    dst_idx = find_column_index(header, POSSIBLE_TARGET)
    pos_idx = find_column_index(header, POSSIBLE_POSITION)

    if src_idx == -1 or dst_idx == -1:
        st.error("In 'All Inlinks' konnten Quelle/Ziel nicht erkannt werden.")
        st.stop()

    all_links = set()
    content_links = set()

    for row in inlinks_df.itertuples(index=False, name=None):
        src_raw = row[src_idx]
        dst_raw = row[dst_idx]

        src = remember_original(src_raw)
        dst = remember_original(dst_raw)
        if not src or not dst:
            continue

        # Alle Links
        all_links.add((src, dst))

        # Content-Links (falls Positionsspalte existiert)
        if pos_idx != -1:
            pos_val = row[pos_idx]
            if is_content_position(pos_val):
                content_links.add((src, dst))

    st.session_state["_all_links"] = all_links
    st.session_state["_content_links"] = content_links


# Separate Start-Buttons für jede Analyse (rot eingefärbt)
st.markdown("---")
start_cols = st.columns(4)
run_clicked_a1 = run_clicked_a2 = run_clicked_a3 = run_clicked_a4 = False

if A1_NAME in selected_analyses:
    with start_cols[0]:
        run_clicked_a1 = st.button("Let's Go (Analyse 1)", type="primary", key="btn_a1", use_container_width=True)

if A2_NAME in selected_analyses:
    with start_cols[1]:
        run_clicked_a2 = st.button("Let's Go (Analyse 2)", type="primary", key="btn_a2", use_container_width=True)

if A3_NAME in selected_analyses:
    with start_cols[2]:
        run_clicked_a3 = st.button("Let's Go (Analyse 3)", type="primary", key="btn_a3", use_container_width=True)

if A4_NAME in selected_analyses:
    with start_cols[3]:
        run_clicked_a4 = st.button("Let's Go (Analyse 4)", type="primary", key="btn_a4", use_container_width=True)


# Merker für Sichtbarkeit
if run_clicked_a1:
    st.session_state["__show_a1__"] = True
if run_clicked_a2:
    st.session_state["__show_a2__"] = True

# NEU: Gesamt-Flag, ob irgendeine der relevanten Analysen gestartet wurde
run_clicked = run_clicked_a1 or run_clicked_a2 or run_clicked_a3



# Gate nur anwenden, wenn A1 oder A2 überhaupt gewählt wurden
if (A1_NAME in selected_analyses or A2_NAME in selected_analyses) and (not run_clicked) and (not st.session_state.get("ready", False)):
    st.info("Bitte Dateien für die gewählten Analysen hochladen und auf **Let's Go** klicken.")
    st.stop()

# Vorverarbeitung & Validierung für A1/A2/A3 (und ggf. Embeddings/Related)
if any(a in selected_analyses for a in [A1_NAME, A2_NAME, A3_NAME]) and (run_clicked or st.session_state.ready):

    # Validierung & ggf. Related aus Embeddings bauen
    if needs_embeddings_or_related and (emb_df is not None or related_df is not None):
        if (emb_df is None and related_df is None) or any(df is None for df in [inlinks_df, metrics_df, backlinks_df]):
            st.error("Bitte alle benötigten Dateien hochladen (Embeddings/Related, All Inlinks, Linkmetriken, Backlinks).")
            st.stop()

        # Wenn wir bis hier ohne st.stop() gekommen sind, können wir den Loader sicher zeigen
        if run_clicked:
            placeholder = st.empty()
            with placeholder.container():
                c1, c2, c3 = st.columns([1,2,1])
                with c2:
                    st.image("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDJweGExcHhhOWZneTZwcnAxZ211OWJienY5cWQ1YmpwaHR0MzlydiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/dBRaPog8yxFWU/giphy.gif", width=280)
                    st.caption("Die Berechnungen laufen – Zeit für eine kleine Stärkung …")


        if emb_df is not None and not emb_df.empty and (related_df is None or related_df.empty):
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
        # Backend aus Sidebar
        backend_eff = locals().get("backend", "Exakt (NumPy)")
        top_k = int(locals().get("max_related", 10))
        sim_thr = float(locals().get("sim_threshold", 0.80))

        # Related-URLs aus Embeddings berechnen
        related_df = build_related_from_embeddings(urls, V, top_k, sim_thr, backend_eff)

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


    # Prüfen, ob alles da ist (nur A1/A2 relevant)
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
    m_score_idx = find_column_index(m_header, ["link score","interner link score","internal link score","ils","score","linkscore"])
    m_in_idx    = find_column_index(m_header, ["unique inlinks","einzigartige inlinks","inlinks","interne inlinks","eingehende links","in links","inbound internal links"])
    m_out_idx   = find_column_index(m_header, ["unique outlinks","einzigartige outlinks","outlinks","ausgehende links","interne outlinks","out links","outbound links"])
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
    ils_vals: List[float] = []
    prd_vals: List[float] = []

    for _, r in metrics_df.iterrows():
        u = remember_original(r.iloc[m_url_idx])
        if not u:
            continue
        score    = _num(r.iloc[m_score_idx])
        inlinks  = _num(r.iloc[m_in_idx])
        outlinks = _num(r.iloc[m_out_idx])
        prdiff   = inlinks - outlinks
        metrics_map[u] = {"score": score, "prDiff": prdiff}
        ils_vals.append(score)
        prd_vals.append(prdiff)

    # Backlinks
    backlinks_df = backlinks_df.copy()
    backlinks_df.columns = [str(c).strip() for c in backlinks_df.columns]
    b_header = [str(c).strip() for c in backlinks_df.columns]

    # Kandidaten für Offpage-Backlinkliste (Referring page URL plus Target-URL)
    src_idx_file = find_column_index(b_header, POSSIBLE_SOURCE)
    tgt_idx_file = find_column_index(b_header, POSSIBLE_TARGET)

    backlink_map: Dict[str, Dict[str, float]] = {}
    bl_vals: List[float] = []
    rd_vals: List[float] = []

    # Fall A: Offpage Linkliste eine Zeile ist ein Backlink
    if src_idx_file != -1 and tgt_idx_file != -1:
        from urllib.parse import urlparse

        rows = []
        for row in backlinks_df.itertuples(index=False, name=None):
            tgt_raw = row[tgt_idx_file]
            src_raw = row[src_idx_file]

            tgt = remember_original(tgt_raw)
            if not tgt:
                continue

            src_val = str(src_raw or "").strip()
            if not src_val:
                continue

            try:
                dom = urlparse(src_val).netloc.lower()
            except Exception:
                dom = ""
            if not dom:
                continue

            rows.append([normalize_url(tgt), dom])

        if not rows:
            st.error(
                "Backlinks Datei konnte nicht als Offpage Linkliste interpretiert werden "
                "keine gültigen Kombinationen aus Referring page URL und Target URL gefunden."
            )
            st.stop()

        tmp = pd.DataFrame(rows, columns=["URL", "Domain"])
        agg = tmp.groupby("URL").agg(
            Backlinks=("Domain", "size"),
            ReferringDomains=("Domain", "nunique")
        ).reset_index()

        backlinks_df = agg
        backlinks_df.columns = ["URL", "Backlinks", "Referring Domains"]
        b_header = [str(c).strip() for c in backlinks_df.columns]
        b_url_idx = 0
        b_bl_idx  = 1
        b_rd_idx  = 2

        st.info(
            "Backlink Metriken wurden automatisch aus einer Offpage Linkliste berechnet "
            "Referring page URL zu Domain, aggregiert je Target URL."
        )

    # Fall B: Aggregierte Metriken eine Zeile ist eine URL
    else:
        b_url_idx = find_column_index(b_header, ["url","urls","page","seite","address","adresse"])
        if b_url_idx == -1:
            if backlinks_df.shape[1] >= 1:
                b_url_idx = 0
                st.warning("Backlinks aggregiert URL Spalte nicht eindeutig erkannt nehme erste Spalte als URL.")
            else:
                st.error("'Backlinks' braucht mindestens eine URL Spalte.")
                st.stop()

        b_bl_idx = find_column_index(
            b_header,
            ["backlinks","backlink","external backlinks","back links","anzahl backlinks","backlinks total"]
        )
        b_rd_idx = find_column_index(
            b_header,
            ["referring domains","ref domains","verweisende domains",
             "anzahl referring domains","anzahl verweisende domains","domains","rd"]
        )

        if -1 in (b_bl_idx, b_rd_idx):
            st.error(
                "Backlinks aggregiert Spalten für Backlinks und Referring Domains "
                "konnten nicht eindeutig erkannt werden."
            )
            st.stop()

    # Backlink Map und Werte Listen aufbauen
    for row in backlinks_df.itertuples(index=False, name=None):
        url = remember_original(row[b_url_idx])
        if not url:
            continue
        bl = _num(row[b_bl_idx])
        rd = _num(row[b_rd_idx])
        backlink_map[url] = {
            "backlinks": bl,
            "referringDomains": rd,
        }
        bl_vals.append(bl)
        rd_vals.append(rd)

    # Normalisierungsbereiche für Linkpotenzial
    min_ils, max_ils = robust_range(ils_vals) if ils_vals else (0.0, 1.0)
    min_prd, max_prd = robust_range(prd_vals) if prd_vals else (0.0, 1.0)

    bl_log_vals = [math.log1p(max(0.0, v)) for v in bl_vals]
    rd_log_vals = [math.log1p(max(0.0, v)) for v in rd_vals]
    lo_bl_log, hi_bl_log = robust_range(bl_log_vals) if bl_log_vals else (0.0, 1.0)
    lo_rd_log, hi_rd_log = robust_range(rd_log_vals) if rd_log_vals else (0.0, 1.0)
    min_bl, max_bl = robust_range(bl_vals) if bl_vals else (0.0, 1.0)
    min_rd, max_rd = robust_range(rd_vals) if rd_vals else (0.0, 1.0)

    
        


    # Spalten in Related-URL-Tabelle erkennen
    related_df = related_df.copy()
    related_df.columns = [str(c).strip() for c in related_df.columns]
    rel_header = [str(c).strip() for c in related_df.columns]

    rel_src_idx = find_column_index(rel_header, ["quelle","source","from","origin","quell-url","referring page url","referring page","referring url"])
    rel_dst_idx = find_column_index(rel_header, ["ziel","destination","target","ziel-url","ziel url","target url","target-url","zielseite"])
    rel_sim_idx = find_column_index(rel_header, ["similarity","similarität","similarity score","score","cosine similarity","cosinus ähnlichkeit"])

    if -1 in (rel_src_idx, rel_dst_idx, rel_sim_idx):
        if related_df.shape[1] >= 3:
            if rel_src_idx == -1:
                rel_src_idx = 0
            if rel_dst_idx == -1:
                rel_dst_idx = 1
            if rel_sim_idx == -1:
                rel_sim_idx = 2
            st.warning("Related URLs: Header nicht vollständig erkannt – Fallback auf Spaltenpositionen (1–3).")
        else:
            st.error("'Related URLs' braucht mindestens 3 Spalten (Quelle, Ziel, Similarity).")
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

    # PERSIST RELATED MAP für A3
    st.session_state["_related_map"] = related_map

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

    # Persistente Maps für A3/A4
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
    # Inlink-Counts je URL (für Mauerblümchen in A3)
    st.session_state["_inlink_count_map"] = {
        remember_original(r.iloc[m_url_idx]): _num(r.iloc[m_in_idx])
        for _, r in metrics_df.iterrows()
    }

    # ============================================
    # Scope-Filter für Analysen (Regex-basiert)
    # ============================================
    scope_mode = st.session_state.get("scope_mode", "Alle URLs (Standard)")
    allowed_urls_target = None   # URLs, die als Ziel-URL zulässig sind
    allowed_urls_source = None   # URLs, die als Linkgeber/Related zulässig sind
    allowed_urls_union  = None   # Vereinigt (falls später benötigt)

    if scope_mode == "Regex-basiert (nur passende URLs)":
        regex_target_text = st.session_state.get("scope_regex_target_text", "") or ""
        regex_source_text = st.session_state.get("scope_regex_source_text", "") or ""

        patterns_target: list[re.Pattern] = []
        patterns_source: list[re.Pattern] = []

        # Ziel-URL-Regeln
        for line in regex_target_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                patterns_target.append(re.compile(line))
            except re.error:
                st.warning(f"Ungültige Regex-Regel für Ziel-URLs ignoriert: {line}")
                continue

        # Source-/Related-Regeln
        for line in regex_source_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                patterns_source.append(re.compile(line))
            except re.error:
                st.warning(f"Ungültige Regex-Regel für Related-URLs ignoriert: {line}")
                continue

        if patterns_target or patterns_source:
            all_urls_for_scope = set()

            # URLs aus Related Map
            for t, rel_list in related_map.items():
                all_urls_for_scope.add(t)
                for src, _ in rel_list:
                    all_urls_for_scope.add(src)

            # URLs aus All Inlinks
            all_links = st.session_state.get("_all_links", set())
            for src, dst in all_links:
                all_urls_for_scope.add(src)
                all_urls_for_scope.add(dst)

            # Ziel-URL-Menge
            if patterns_target:
                allowed_target = set()
                for u in all_urls_for_scope:
                    for rx in patterns_target:
                        if rx.search(u):
                            allowed_target.add(u)
                            break
                allowed_urls_target = allowed_target

            # Source-/Related-URL-Menge
            if patterns_source:
                allowed_source = set()
                for u in all_urls_for_scope:
                    for rx in patterns_source:
                        if rx.search(u):
                            allowed_source.add(u)
                            break
                allowed_urls_source = allowed_source

            # Union (optional)
            if allowed_urls_target is not None or allowed_urls_source is not None:
                allowed_union = set()
                if allowed_urls_target is not None:
                    allowed_union.update(allowed_urls_target)
                if allowed_urls_source is not None:
                    allowed_union.update(allowed_urls_source)
                allowed_urls_union = allowed_union

    # in Session speichern, damit Analysen darauf zugreifen können
    st.session_state["_allowed_urls_target"] = allowed_urls_target
    st.session_state["_allowed_urls_source"] = allowed_urls_source
    st.session_state["_allowed_urls_union"]  = allowed_urls_union


    # ===============================
    # Analyse 1 – nur rendern, wenn Button gedrückt
    # ===============================
    if st.session_state.get("__show_a1__", False):
        st.markdown("## Analyse 1: Interne Verlinkungsmöglichkeiten")
        st.caption("Diese Analyse schlägt thematisch passende interne Verlinkungen vor, zeigt bestehende (Content-)Links und bewertet das Linkpotenzial der Linkgeber.")
        if not st.session_state.get("__gems_loading__", False):
            cols = ["Ziel-URL"]
            for i in range(1, int(locals().get("max_related", 10)) + 1):
                cols += [
                    f"Related URL {i}",
                    f"Ähnlichkeit {i}",
                    f"Link von Related URL {i} auf Ziel-URL bereits vorhanden?",
                    f"Link von Related URL {i} auf Ziel-URL aus Inhalt heraus vorhanden?",
                    f"Linkpotenzial Related URL {i}",
                ]
            rows_norm, rows_view = [], []
            max_related_local = int(locals().get("max_related", 10))

            # Scope-Einstellungen aus der Sidebar
            scope_mode = st.session_state.get("scope_mode", "Alle URLs (Standard)")
            allowed_urls_target = st.session_state.get("_allowed_urls_target")
            allowed_urls_source = st.session_state.get("_allowed_urls_source")

            for target, related_list in sorted(related_map.items()):
                # Regex-Scope: Ziel-URL ggf. komplett ausschließen
                if scope_mode == "Regex-basiert (nur passende URLs)" and allowed_urls_target is not None:
                    if target not in allowed_urls_target:
                        continue

                # Start: unveränderte Kandidaten
                filtered = list(related_list)

                

                # Regex-Scope (Variante 3): nur Quellen, die als Linkgeber erlaubt sind
                if scope_mode == "Regex-basiert (nur passende URLs)" and allowed_urls_source is not None:
                    filtered = [
                        (source, sim)
                        for (source, sim) in filtered
                        if source in allowed_urls_source
                    ]

                # Sortierung & Top-N
                related_sorted = sorted(filtered, key=lambda x: x[1], reverse=True)[:max_related_local]

                row_norm = [target]
                row_view = [disp(target)]

                for source, sim in related_sorted:
                    anywhere = "ja" if (source, target) in st.session_state["_all_links"] else "nein"
                    from_content = "ja" if (source, target) in st.session_state["_content_links"] else "nein"
                    final_score = source_potential_map.get(source, 0.0)
                    row_norm.extend([source, round(float(sim), 3), anywhere, from_content, final_score])
                    row_view.extend([disp(source), round(float(sim), 3), anywhere, from_content, final_score])

                # Zeilen auf die Spaltenanzahl auffüllen
                while len(row_norm) < len(cols):
                    row_norm.append(np.nan)
                while len(row_view) < len(cols):
                    row_view.append(np.nan)

                rows_norm.append(row_norm)
                rows_view.append(row_view)

            res1_df = pd.DataFrame(rows_norm, columns=cols)
            st.session_state.res1_df = res1_df


            long_rows = []
            max_i = int(locals().get("max_related", 10))
            
            # Seitendaten-Maps aufbauen (falls Crawl-Datei vorhanden)
            page_maps = {}
            if isinstance(crawl_df_a1, pd.DataFrame) and not crawl_df_a1.empty:
                page_maps = build_page_text_maps_for_a1(crawl_df_a1)
            
            title_map = page_maps.get("title", {})
            meta_map = page_maps.get("meta", {})
            h1_map = page_maps.get("h1", {})
            
            # Prüfen, welche Spalten vorhanden sind
            has_title = bool(title_map)
            has_meta = bool(meta_map)
            has_h1 = bool(h1_map)
            
            # Spalten-Liste dynamisch aufbauen
            cols = ["Ziel-URL"]
            if has_title:
                cols.append("Title Tag")
            if has_meta:
                cols.append("Meta Description")
            if has_h1:
                cols.append("H1")
            cols.extend([
                "Related URL",
                "Ähnlichkeit (Cosinus Ähnlichkeit)",
                "Link von Related URL auf Ziel-URL vorhanden?",
                "Link von Related URL auf Ziel-URL aus Inhalt heraus vorhanden?",
                "Linkpotenzial",
            ])
            
            for _, r in res1_df.iterrows():
                ziel = r["Ziel-URL"]
                ziel_norm = remember_original(ziel) if ziel else ""
                
                # Seitendaten für diese Ziel-URL holen
                title_val = title_map.get(ziel_norm, "") if has_title else ""
                meta_val = meta_map.get(ziel_norm, "") if has_meta else ""
                h1_val = h1_map.get(ziel_norm, "") if has_h1 else ""
                
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
                    
                    # Zeile aufbauen (dynamisch je nach vorhandenen Seitendaten)
                    row = [disp(ziel)]
                    if has_title:
                        row.append(title_val)
                    if has_meta:
                        row.append(meta_val)
                    if has_h1:
                        row.append(h1_val)
                    row.extend([
                        disp(src),
                        round(float(sim), 3) if pd.notna(sim) else np.nan,
                        anywhere,
                        from_content,
                        float(pot),
                    ])
                    
                    long_rows.append(row)
            
            res1_view_long = pd.DataFrame(long_rows, columns=cols)
            if not res1_view_long.empty:
                res1_view_long = res1_view_long.sort_values(
                    by=["Ziel-URL", "Ähnlichkeit (Cosinus Ähnlichkeit)"],
                    ascending=[True, False],
                    kind="mergesort"
                ).reset_index(drop=True)

            # KI-Ankertexte pro Ziel-URL (optional)
            if st.session_state.get("a1_enable_anchor_ai", False):
                cfg = {
                    "provider": st.session_state.get("a1_anchor_provider", "OpenAI"),
                    "openai_key": st.session_state.get("a1_openai_api_key", "").strip(),
                    "gemini_key": st.session_state.get("a1_gemini_api_key", "").strip(),
                    "fields": st.session_state.get("a1_prompt_fields", []),
                }

                page_maps = build_page_text_maps_for_a1(crawl_df_a1) if isinstance(crawl_df_a1, pd.DataFrame) else {"title": {}, "h1": {}, "meta": {}, "main": {}}

                top_kw_map = {}
                manual_kw_map = {}

                # Cache für bereits generierte Ankertexte
                cache_key = f"_anchor_cache_{hash(str(cfg))}"
                if cache_key not in st.session_state:
                    st.session_state[cache_key] = {}
                anchor_cache = st.session_state[cache_key]
                
                # URLs, die noch nicht im Cache sind
                urls_to_process = []
                for tgt in sorted(res1_view_long["Ziel-URL"].astype(str).unique()):
                    if tgt not in anchor_cache:
                        norm = remember_original(tgt)
                        if not norm:
                            norm = tgt
                        urls_to_process.append((tgt, norm))
                
                # Parallel generieren mit Progress-Bar
                if urls_to_process:
                    failed_urls = []
                    progress_bar = st.progress(0)
                    total = len(urls_to_process)
                    
                    def generate_single_anchor(tgt_norm_pair):
                        tgt, norm = tgt_norm_pair
                        try:
                            v1, v2, v3 = generate_anchor_variants_for_url(
                                norm,
                                page_maps=page_maps,
                                top_kw_map=top_kw_map,
                                manual_kw_map=manual_kw_map,
                                cfg=cfg,
                            )
                            return (tgt, (v1, v2, v3), None)
                        except Exception as e:
                            return (tgt, None, str(e))
                    
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = {executor.submit(generate_single_anchor, pair): pair for pair in urls_to_process}
                        completed = 0
                        
                        for future in as_completed(futures):
                            completed += 1
                            progress_bar.progress(completed / total)
                            tgt, result, error = future.result()
                            
                            if error:
                                failed_urls.append((tgt, error))
                            else:
                                anchor_cache[tgt] = result
                                st.session_state[cache_key] = anchor_cache
                    
                    progress_bar.empty()
                    
                    # Zeige Fehler an, falls welche aufgetreten sind
                    if failed_urls:
                        st.error(f"⚠️ Ankertext-Generierung fehlgeschlagen für {len(failed_urls)} URL(s):")
                        for url, error in failed_urls[:10]:  # Zeige max. 10 Fehler
                            with st.expander(f"Fehler für {disp(url)}"):
                                st.code(error)
                        if len(failed_urls) > 10:
                            st.caption(f"... und {len(failed_urls) - 10} weitere Fehler")

                res1_view_long["Ankertext (kurz)"] = res1_view_long["Ziel-URL"].map(lambda u: anchor_cache.get(u, ("", "", ""))[0])
                res1_view_long["Ankertext (beschreibend)"] = res1_view_long["Ziel-URL"].map(lambda u: anchor_cache.get(u, ("", "", ""))[1])
                res1_view_long["Ankertext (handlungsorientiert)"] = res1_view_long["Ziel-URL"].map(lambda u: anchor_cache.get(u, ("", "", ""))[2])

            st.dataframe(res1_view_long, use_container_width=True, hide_index=True)
            csv1 = res1_view_long.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download 'Interne Verlinkungsmöglichkeiten (Long-Format)' (CSV)",
                data=csv1, file_name="interne_verlinkungsmoeglichkeiten_long.csv", mime="text/csv", key="dl_interne_verlinkung_long",
            )

    # ===============================
    # Analyse 2 – nur rendern, wenn Button gedrückt
    # ===============================
    if st.session_state.get("__show_a2__", False):
        st.markdown("## Analyse 2: Potenziell zu entfernende Links")
        st.caption("Diese Analyse legt bestehende Links zwischen semantisch nicht stark verwandten URLs offen.")

        # A2-Settings aus Sidebar-State lesen
        not_similar_threshold = float(st.session_state.get("a2_not_sim", 0.60))
        backlink_weight_2x = bool(st.session_state.get("a2_weight2x", False))
        only_content_links = bool(st.session_state.get("a2_only_content", False))

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

        link_iterable = st.session_state["_content_links"] if only_content_links else st.session_state["_all_links"]

        if _has_emb:
            missing = [
                (src, dst) for (src, dst) in link_iterable
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
            if only_content_links and (quelle, ziel) not in st.session_state["_content_links"]:
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
# Analyse 3 (nur anzeigen, wenn ausgewählt)
# =========================================================
if A3_NAME in selected_analyses:

    # A3 wird über Button oben gestartet
    if run_clicked_a3:
        st.session_state["__gems_loading__"] = True
        st.session_state["__ready_gems__"] = False
        st.rerun()

        # Loader (ohne GIF, mit statischem Hinweis)
        if st.session_state.get("__gems_loading__", False):
            ph3 = st.session_state.get("__gems_ph__")
            if ph3 is None:
                ph3 = st.empty()
                st.session_state["__gems_ph__"] = ph3
            with ph3.container():
                c1, c2, c3 = st.columns([1, 2, 1])
                with c2:
                    st.markdown("""
                    <div style='padding: 12px; border-radius: 6px; background-color:#fff0f0; color:#b30000; border:1px solid #b30000; text-align:center;'>
                        ⏳ Analyse 3 läuft – bitte einen Moment warten…
                    </div>
                    """, unsafe_allow_html=True)


    # ====== Analyse 3: Berechnung & Ausgabe ======
    # Datenabhängigkeiten prüfen
    source_potential_map = st.session_state.get("_source_potential_map")
    related_map = st.session_state.get("_related_map")
    content_links = st.session_state.get("_content_links")
    all_links = st.session_state.get("_all_links")
    
    if not (
        isinstance(source_potential_map, dict) 
        and isinstance(related_map, dict) 
        and isinstance(content_links, set) 
        and isinstance(all_links, set)
    ):
        if run_clicked_a3:
            st.error("Für Analyse 3 fehlen vorbereitete Daten. Bitte prüfe, ob 'Related URLs', 'All Inlinks', 'Linkmetriken' und 'Backlinks' hochgeladen sind.")
        st.stop()
    else:
        # GSC (optional) – falls im Upload-Center bereitgestellt
        gsc_df_a3 = gsc_df_loaded

        if isinstance(gsc_df_a3, pd.DataFrame) and not gsc_df_a3.empty:
            df = gsc_df_a3.copy()
            df.columns = [str(c).strip() for c in df.columns]
            hdr = [_norm_header(c) for c in df.columns]

            def _fidx(names, default=None):
                names = {_norm_header(x) for x in names}
                for i, h in enumerate(hdr):
                    if h in names: return i
                for i, h in enumerate(hdr):
                    if any(n in h for n in names): return i
                return default

            url_idx   = _fidx({"url","page","seite","landingpage"}, 0)
            impr_idx  = _fidx({"impressions","impr","impressionen"}, None)
            clicks_idx= _fidx({"clicks","klicks"}, None)
            pos_idx   = _fidx({"position","avg position","durchschn. position"}, None)

            if url_idx is not None:
                df.iloc[:, url_idx] = df.iloc[:, url_idx].astype(str).map(remember_original)
                if impr_idx is not None:
                    df.iloc[:, impr_idx] = pd.to_numeric(df.iloc[:, impr_idx], errors="coerce").fillna(0)
                if clicks_idx is not None:
                    df.iloc[:, clicks_idx] = pd.to_numeric(df.iloc[:, clicks_idx], errors="coerce").fillna(0)
                if pos_idx is not None:
                    df.iloc[:, pos_idx] = pd.to_numeric(df.iloc[:, pos_idx], errors="coerce")

                gsc_agg = df.groupby(df.columns[url_idx]).agg({
                    (df.columns[impr_idx] if impr_idx is not None else df.columns[url_idx]): ("sum" if impr_idx is not None else "size"),
                    **({df.columns[clicks_idx]: "sum"} if clicks_idx is not None else {}),
                    **({df.columns[pos_idx]: "mean"} if pos_idx is not None else {})
                })
                gsc_agg.columns = [("impressions" if impr_idx is not None else "count"),
                                   *([ "clicks"] if clicks_idx is not None else []),
                                   *([ "position"] if pos_idx is not None else [])]
                gsc_agg = gsc_agg.reset_index(names="url")
            else:
                gsc_agg = pd.DataFrame(columns=["url","impressions","clicks","position"])
        else:
            gsc_agg = pd.DataFrame(columns=["url","impressions","clicks","position"])

        # Hilfs-Maps
        gsc_impr_map = {r["url"]: float(r.get("impressions", 0.0)) for _, r in gsc_agg.iterrows()} if "impressions" in gsc_agg.columns else {}
        gsc_pos_map  = {r["url"]: float(r.get("position", np.nan)) for _, r in gsc_agg.iterrows()} if "position" in gsc_agg.columns else {}

        # Inlink-Counts je URL (für Mauerblümchen)
        inlink_count_map = st.session_state.get("_inlink_count_map", {})

        # Offpage-Infos (für Dämpfung)
        backlink_map = st.session_state.get("_backlink_map", {})
        bl_vals  = [d.get("backlinks", 0.0) for d in backlink_map.values()]
        rd_vals  = [d.get("referringDomains", 0.0) for d in backlink_map.values()]
        bl_log   = np.asarray([np.log1p(max(0.0, float(v))) for v in bl_vals], dtype=float)
        rd_log   = np.asarray([np.log1p(max(0.0, float(v))) for v in rd_vals], dtype=float)
        lo_bl, hi_bl = robust_range(bl_log, 0.05, 0.95) if bl_log.size else (0.0, 1.0)
        lo_rd, hi_rd = robust_range(rd_log, 0.05, 0.95) if rd_log.size else (0.0, 1.0)

        def _offpage_norm(u: str) -> float:
            d = backlink_map.get(u, {"backlinks":0.0,"referringDomains":0.0})
            bln = robust_norm(np.log1p(_num(d.get("backlinks",0.0))), lo_bl, hi_bl)
            rdn = robust_norm(np.log1p(_num(d.get("referringDomains",0.0))), lo_rd, hi_rd)
            return 0.5*bln + 0.5*rdn

        # Gems bestimmen (Top-X % nach Linkpotenzial)
        gem_pct = float(st.session_state.get("a3_gem_pct", 10))
        max_targets_per_gem = int(st.session_state.get("a3_max_targets", 10))
        w_lihd = float(st.session_state.get("a3_w_lihd", 0.30))
        w_def  = float(st.session_state.get("a3_w_def", 0.30))
        w_rank = float(st.session_state.get("a3_w_rank", 0.30))
        w_orph = float(st.session_state.get("a3_w_orph", 0.10))
        thin_k = int(st.session_state.get("a3_thin_k", 2))
        rank_minmax = st.session_state.get("a3_rank_minmax", (8, 20))
        offpage_damp_enabled = bool(st.session_state.get("a3_offpage_enable", True))
        beta_offpage = float(st.session_state.get("a3_offpage_beta", 0.5))
        sort_choice = st.session_state.get("a3_sort_choice", "rank_mix")
        alpha_mix = float(st.session_state.get("a3_alpha_mix", 0.5))

        n_sources = len(source_potential_map)
        n_gems = max(1, int(math.ceil(gem_pct / 100.0 * n_sources)))
        gems = sorted(source_potential_map.items(), key=lambda x: x[1], reverse=True)[:n_gems]
        gem_set = {g for g, _ in gems}

        # Kandidaten erzeugen: Für jeden Gem alle semantisch nahen Ziele ohne bestehenden Content-Link
        rows = []
        for gem, gem_pot in gems:
            cand = related_map.get(gem, [])
            if not cand:
                continue
            # Normalisierung der Similarity auf Kandidatenebene
            sim_vals = np.asarray([float(s) for _, s in cand], dtype=float)
            s_lo, s_hi = robust_range(sim_vals, 0.05, 0.95) if sim_vals.size else (0.0, 1.0)

            for target, sim in cand:
                # nur fehlende Content-Links (Empfehlungen = noch nicht gesetzte Content-Links)
                if (gem, target) in st.session_state["_content_links"]:
                    continue

                sim_norm = robust_norm(float(sim), s_lo, s_hi)

                # Hidden Champions (nur wenn GSC Impressions vorhanden)
                hid_raw = gsc_impr_map.get(target, 0.0)
                if len(gsc_impr_map) > 0:
                    impr_arr = np.asarray(list(gsc_impr_map.values()), dtype=float)
                    i_lo, i_hi = robust_range(impr_arr, 0.05, 0.95)
                    hid_norm = robust_norm(hid_raw, i_lo, i_hi)
                else:
                    hid_norm = 0.0

                # Sprungbrett-URLs (nur wenn Position vorhanden)
                if target in gsc_pos_map and not pd.isna(gsc_pos_map[target]):
                    pos = float(gsc_pos_map[target])
                    lo_pos, hi_pos = rank_minmax
                    if lo_pos <= pos <= hi_pos:
                        rank_norm = 1.0
                    else:
                        if pos < lo_pos:
                            dist = lo_pos - pos
                        else:
                            dist = pos - hi_pos
                        rank_norm = max(0.0, 1.0 - (dist / max(1.0, hi_pos)))
                else:
                    rank_norm = 0.0

                # Mauerblümchen (Thin: wenige Inlinks)
                in_c = st.session_state.get("_inlink_count_map", {}).get(target, 0.0)
                orph_norm = 1.0 if in_c <= float(thin_k) else 0.0

                # PRIO (Linkbedarf)
                prio = (w_def * sim_norm) + (w_lihd * hid_norm) + (w_rank * rank_norm) + (w_orph * orph_norm)

                # Offpage-Dämpfung
                if offpage_damp_enabled:
                    off_n = _offpage_norm(target)
                    prio = prio * (1.0 - beta_offpage * off_n)

                # Sortier-Score
                if sort_choice == "prio_only":
                    sort_score = prio
                elif sort_choice == "sim_only":
                    sort_score = sim_norm
                else:  # rank_mix
                    sort_score = alpha_mix * sim_norm + (1.0 - alpha_mix) * prio

                rows.append({
                    "Gem": disp(gem),
                    "Gem (normiert)": gem,
                    "Gem-Linkpotenzial": float(gem_pot),
                    "Ziel-URL": disp(target),
                    "Ziel (normiert)": target,
                    "Ähnlichkeit": float(sim),
                    "Ähnlichkeit (norm)": float(sim_norm),
                    "HiddenChamp (norm)": float(hid_norm),
                    "Sprungbrett (norm)": float(rank_norm),
                    "Mauerblümchen (norm)": float(orph_norm),
                    "PRIO (Linkbedarf)": float(prio),
                    "SortScore": float(sort_score),
                })

        rec_df = pd.DataFrame(rows)
        if rec_df.empty:
            st.info("Keine Empfehlungen gefunden (ggf. sind für die Gems bereits Content-Links gesetzt oder es fehlen Related-URL-Daten).")
        else:
            # Top-Ziele je Gem begrenzen & sortieren
            rec_df = rec_df.sort_values(["Gem (normiert)", "SortScore"], ascending=[True, False])
            rec_df["Rang (Gem)"] = rec_df.groupby("Gem (normiert)")["SortScore"].rank(method="first", ascending=False).astype(int)
            rec_df = rec_df[rec_df["Rang (Gem)"] <= int(max_targets_per_gem)]

            # Gesamt-Ansicht
            view_cols = [
                "Gem","Gem-Linkpotenzial","Ziel-URL","Ähnlichkeit","PRIO (Linkbedarf)",
                "HiddenChamp (norm)","Sprungbrett (norm)","Mauerblümchen (norm)","Rang (Gem)"
            ]
            st.markdown("### Empfehlungen (gesamt)")
            st.dataframe(rec_df[view_cols].sort_values(["Gem","Rang (Gem)"]), use_container_width=True, hide_index=True)

            # Download (CSV + XLSX)
            csv_bytes = rec_df[view_cols + ["Ähnlichkeit (norm)","SortScore"]].to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download Empfehlungen (CSV)", data=csv_bytes, file_name="analyse3_empfehlungen.csv", mime="text/csv", key="a3_dl_csv")

            try:
                buf_rec = io.BytesIO()
                with pd.ExcelWriter(buf_rec, engine="xlsxwriter") as xw:
                    # Gesamt
                    rec_df[view_cols + ["Ähnlichkeit (norm)","SortScore"]].to_excel(xw, index=False, sheet_name="Gesamt")
                    # Pro Gem
                    for gem_name, grp in rec_df.groupby("Gem", sort=False):
                        sheet = re.sub(r"[^A-Za-z0-9]+", "_", gem_name)[:31] or "Gem"
                        grp[view_cols + ["Ähnlichkeit (norm)","SortScore"]].to_excel(xw, index=False, sheet_name=sheet)
                buf_rec.seek(0)
                st.download_button("Download Empfehlungen (XLSX)", data=buf_rec.getvalue(),
                                   file_name="analyse3_empfehlungen.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   key="a3_dl_xlsx")
            except Exception:
                pass

            # Kompakte per-Gem Ansicht
            st.markdown("### Schnellansicht je Gem")
            for gem_name, grp in rec_df.groupby("Gem", sort=False):
                with st.expander(f"Empfehlungen für: {gem_name}", expanded=False):
                    st.dataframe(
                        grp[view_cols].sort_values("Rang (Gem)"),
                        use_container_width=True, hide_index=True
                    )

            # ---------------------------------------------
            # Add-on: Near-Orphan-Analyse 2.0 – Mauerblümchen
            # ---------------------------------------------
            st.markdown("### Add-on: Near-Orphan-Analyse 2.0 – „Mauerblümchen mit Potenzial“")

            # Inlink-Daten & GSC müssen vorhanden sein
            if not inlink_count_map:
                st.info("Keine Inlink-Zahlen aus 'Linkmetriken' verfügbar – Near-Orphan-Analyse wird übersprungen.")
            elif not gsc_impr_map:
                st.info("Keine GSC-Impressions für Analyse 3 geladen – Near-Orphan-Analyse wird übersprungen.")
            else:
                impr_vals = np.asarray(list(gsc_impr_map.values()), dtype=float)
                impr_vals = impr_vals[np.isfinite(impr_vals) & (impr_vals > 0)]

                if impr_vals.size == 0:
                    st.info("Keine GSC-Impressions > 0 – Near-Orphan-Analyse wird übersprungen.")
                else:
                    # Nachfrage in Log-Skala normalisieren (robust)
                    impr_log = np.log1p(impr_vals)
                    lo_impr, hi_impr = robust_range(impr_log, 0.10, 0.95)

                    near_rows = []

                    for url, inl in inlink_count_map.items():
                        inl = float(inl)

                        # Nur „Mauerblümchen“: wenige interne Inlinks (Thin-Threshold aus A3)
                        if inl > float(thin_k):
                            continue

                        impr = float(gsc_impr_map.get(url, 0.0))
                        if impr <= 0:
                            # Keine Nachfrage → uninteressant für dieses Add-on
                            continue

                        pos = float(gsc_pos_map.get(url, np.nan)) if url in gsc_pos_map else np.nan

                        # Wenige Inlinks = hoher Wert (0 = viele Inlinks, 1 = gar keine)
                        inl_score = 1.0 - robust_norm(inl, 0.0, float(max(thin_k, 1)))

                        # Nachfrage-Signal (Impressions, log-transformiert)
                        impr_score = robust_norm(np.log1p(impr), lo_impr, hi_impr)

                        # Ranking-Sweet-Spot wie oben (optional Boost)
                        if np.isfinite(pos):
                            lo_pos, hi_pos = rank_minmax
                            if lo_pos <= pos <= hi_pos:
                                rank_score = 1.0
                            else:
                                if pos < lo_pos:
                                    dist = lo_pos - pos
                                else:
                                    dist = pos - hi_pos
                                rank_score = max(0.0, 1.0 - (dist / max(1.0, hi_pos)))
                        else:
                            rank_score = 0.0

                        # Heuristischer Gesamt-Score:
                        # - 50 % Nachfrage
                        # - 30 % Ranking-Sweet-Spot
                        # - 20 % „Mauerblümchenheit“ (wenig Inlinks)
                        near_score = (0.5 * impr_score) + (0.3 * rank_score) + (0.2 * inl_score)

                        near_rows.append([
                            disp(url),
                            inl,
                            impr,
                            pos if np.isfinite(pos) else np.nan,
                            round(near_score, 3),
                        ])

                    if not near_rows:
                        st.info("Keine Near-Orphan-Kandidaten gefunden (nach aktueller Thin-Schwelle & GSC-Daten).")
                    else:
                        near_df = pd.DataFrame(
                            near_rows,
                            columns=[
                                "Ziel-URL",
                                "Interne Inlinks",
                                "GSC-Impressions (agg.)",
                                "Ø Position (GSC)",
                                "Near-Orphan-Score",
                            ],
                        ).sort_values("Near-Orphan-Score", ascending=False)

                        st.dataframe(
                            near_df.head(200),
                            use_container_width=True,
                            hide_index=True
                        )

                        st.download_button(
                            "Download Near-Orphan-Analyse (CSV)",
                            data=near_df.to_csv(index=False).encode("utf-8-sig"),
                            file_name="analyse3_near_orphan_addon.csv",
                            mime="text/csv",
                            key="a3_dl_near_orphan_csv",
                        )

    # Status-Flags
    st.session_state["__gems_loading__"] = False
    st.session_state["__ready_gems__"] = True
    try:
        ph3 = st.session_state.get("__gems_ph__")
        if ph3 is not None:
            ph3.empty()
    except Exception:
        pass

# =========================================================
# Analyse 4: Anchor & Query Intelligence (Embeddings)
# =========================================================
if A4_NAME in selected_analyses:

    # A4 wird über Button oben gestartet
    if run_clicked_a4:
        st.session_state["__a4_loading__"] = True
        st.session_state["__ready_a4__"] = False
        st.rerun()
    
    # Loader-Anzeige (wenn Analyse 4 läuft, ohne GIF)
    if st.session_state.get("__a4_loading__", False):
        ph4 = st.session_state.get("__a4_ph__")
        if ph4 is None:
            ph4 = st.empty()
            st.session_state["__a4_ph__"] = ph4
        with ph4.container():
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.markdown("""
                <div style='padding: 12px; border-radius: 6px; background-color:#fff0f0; color:#b30000; border:1px solid #b30000; text-align:center;'>
                    ⏳ Analyse 4 läuft – bitte einen Moment warten…
                </div>
                """, unsafe_allow_html=True)

    # ---- Anchor-Inventar aus All Inlinks (inkl. ALT als Fallback) ----
    if inlinks_df is None:
        st.info("Für Analyse 4 wird die Datei **All Inlinks** benötigt. Bitte oben im Upload-Center hochladen.")
        st.stop()

    header = [str(c).strip() for c in inlinks_df.columns]
    src_idx = find_column_index(header, POSSIBLE_SOURCE)
    dst_idx = find_column_index(header, POSSIBLE_TARGET)
    if src_idx == -1 or dst_idx == -1:
        st.error("In 'All Inlinks' wurden die Spalten 'Quelle/Source' oder 'Ziel/Destination' nicht gefunden.")
        st.stop()

    # 🔎 Linkposition-Spalte suchen (Link Position, Linkposition, Position Link, ...)
    pos_idx = find_column_index(header, POSSIBLE_POSITION)

    def filter_inlinks_by_position(
        df: pd.DataFrame,
        pos_idx: int,
        key_suffix: str,
        label: str,
    ) -> pd.DataFrame:
        """
        Filtert die All-Inlinks-Tabelle für eine A4-Unteranalyse nach Linkpositionen.

        - Default: alle Positionen werden berücksichtigt.
        - User kann im Multiselect einzelne Positionen ausschließen.
        - Zusätzlich: globaler A4-Scope 'Nur Content-Links' nutzt is_content_position().
        - key_suffix: eindeutiger Suffix pro Unteranalyse (z. B. "over_anchor", "gsc_cov", "kw_cov" usw.).
        """
        if df is None or df.empty:
            return df

        # NEU: globaler A4-Scope – nur Content-Links?
        scope = st.session_state.get("a4_link_scope", "Alle Links (Standard)")
        if scope.startswith("Nur") and pos_idx != -1:
            mask_content = df.iloc[:, pos_idx].apply(is_content_position)
            df = df[mask_content]
        elif scope.startswith("Nur") and pos_idx == -1:
            # Nur einmal warnen
            if not st.session_state.get("_a4_pos_missing_warned", False):
                st.info(
                    "Die Option 'Nur Content-Links' ist aktiv, "
                    "aber in 'All Inlinks' wurde keine Linkpositions-Spalte erkannt – "
                    "es werden daher alle Links verwendet."
                )
                st.session_state["_a4_pos_missing_warned"] = True

        # Wenn es keine Positionsspalte gibt, können wir danach nicht filtern
        if pos_idx == -1:
            return df

        pos_series = df.iloc[:, pos_idx].dropna().astype(str)
        pos_values = sorted(pos_series.unique())

        if not pos_values:
            return df

        exclude_key = f"a4_pos_exclude_{key_suffix}"

        excluded = st.multiselect(
            label,
            options=pos_values,
            default=[],
            key=exclude_key,
            help=(
                "Standard: alle Linkpositionen werden berücksichtigt. "
                "Wähle optional Positionen aus, die für diese Analyse ignoriert werden sollen "
                "(z. B. Navigation, Footer)."
            ),
        )

        if excluded:
            mask = ~df.iloc[:, pos_idx].astype(str).isin(excluded)
            return df[mask]

        return df



    def extract_anchor_inventory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Baut das Anchor-Inventar aus All Inlinks:
        - Primär: Anchor-Spalte
        - Fallback: wenn Anchor leer/NaN → ALT-Spalte
        """
        rows = []
        hdr = [str(c).strip() for c in df.columns]

        # Spalten-Indizes für Anchor & ALT ermitteln
        a_idx = find_column_index(hdr, POSSIBLE_ANCHOR)
        alt_i = find_column_index(hdr, POSSIBLE_ALT)

        for row in df.itertuples(index=False, name=None):
            # Ziel-URL normalisieren/merken
            dst = remember_original(row[dst_idx])
            if not dst:
                continue

            # 1) Primärwert aus Anchor-Spalte
            anchor_val = None
            if a_idx != -1:
                anchor_val = row[a_idx]

            # 2) Fallback: wenn Anchor fehlt/leer/NaN → ALT verwenden
            anchor_is_empty = (
                anchor_val is None
                or (isinstance(anchor_val, str) and anchor_val.strip() == "")
                or (not isinstance(anchor_val, str) and pd.isna(anchor_val))
            )

            if (a_idx == -1 or anchor_is_empty) and alt_i != -1:
                anchor_val = row[alt_i]

            # 3) Finalen Anchor aufbereiten
            anchor = str(anchor_val or "").strip()
            if not anchor:
                # weder Anchor noch ALT → ignorieren
                continue

            rows.append([normalize_url(dst), anchor])

        if not rows:
            return pd.DataFrame(columns=["target", "anchor", "count"])

        tmp = pd.DataFrame(rows, columns=["target", "anchor"])
        agg = (
            tmp.groupby(["target", "anchor"], as_index=False)
               .size()
               .rename(columns={"size": "count"})
        )
        return agg

    # Globale (ungefilterte) Basis: Ankerinventar intern
    anchor_inv_internal_global = extract_anchor_inventory(inlinks_df)

    # NEU: Offpage-Ankertexte ggf. ergänzen (für Visualisierungen / Matrix / Shared / Treemap)
    include_offpage_anchors = bool(st.session_state.get("a4_include_offpage_anchors", False))
    
    # Basis: interne Anker (werden immer verwendet)
    anchor_inv_vis_global = anchor_inv_internal_global.copy()  # für Visualisierung
    offpage_anchor_inv = pd.DataFrame(columns=["target", "anchor", "count"])
    
    if include_offpage_anchors and isinstance(offpage_anchors_df, pd.DataFrame) and not offpage_anchors_df.empty:
        df_off = offpage_anchors_df.copy()
        df_off.columns = [str(c).strip() for c in df_off.columns]
        hdr_off = [str(c).strip() for c in df_off.columns]
    
        # Ziel-URL & Anchor-Spalte suchen (wie bei Inlinks)
        off_tgt_idx = find_column_index(hdr_off, POSSIBLE_TARGET)
        off_anc_idx = find_column_index(hdr_off, POSSIBLE_ANCHOR)
    
        if off_tgt_idx == -1 or off_anc_idx == -1:
            st.warning(
                "In der Offpage-Ankerdatei wurden Ziel-URL oder Anker-Spalte nicht erkannt. "
                "Offpage-Ankertexte werden für A4 ignoriert."
            )
        else:
            rows = []
            for row in df_off.itertuples(index=False, name=None):
                dst = remember_original(row[off_tgt_idx])
                if not dst:
                    continue
                anchor_val = row[off_anc_idx]
                anchor = str(anchor_val or "").strip()
                if not anchor:
                    continue
                rows.append([normalize_url(dst), anchor])
    
            if rows:
                tmp = pd.DataFrame(rows, columns=["target", "anchor"])
                offpage_anchor_inv = (
                    tmp.groupby(["target", "anchor"], as_index=False)
                       .size()
                       .rename(columns={"size": "count"})
                )
    
            if not offpage_anchor_inv.empty:
                # interne + Offpage-Anker kumulieren (noch OHNE source_flag)
                anchor_inv_vis_global = (
                    pd.concat([anchor_inv_internal_global, offpage_anchor_inv], ignore_index=True)
                      .groupby(["target", "anchor"], as_index=False)["count"]
                      .sum()
                )

    def build_anchor_vis_for_subset(anchor_internal_subset: pd.DataFrame) -> pd.DataFrame:
        """
        Kombiniert gefilterte interne Anker mit Offpage-Ankern (falls aktiviert)
        und hängt source_flag an.
        """
        anchor_inv_vis_local = anchor_internal_subset.copy()

        if include_offpage_anchors and not offpage_anchor_inv.empty:
            anchor_inv_vis_local = (
                pd.concat([anchor_internal_subset, offpage_anchor_inv], ignore_index=True)
                  .groupby(["target", "anchor"], as_index=False)["count"]
                  .sum()
            )

        if anchor_inv_vis_local.empty:
            return anchor_inv_vis_local.assign(source_flag=pd.NA)

        tmp = anchor_inv_vis_local.merge(
            anchor_inv_internal_global.assign(internal_count=lambda df: df["count"])[["target", "anchor", "internal_count"]],
            on=["target", "anchor"],
            how="left"
        ).merge(
            offpage_anchor_inv.assign(offpage_count=lambda df: df["count"])[["target", "anchor", "offpage_count"]],
            on=["target", "anchor"],
            how="left"
        )

        tmp["internal_count"] = tmp["internal_count"].fillna(0)
        tmp["offpage_count"] = tmp["offpage_count"].fillna(0)

        def _src_flag(row):
            has_int = row["internal_count"] > 0
            has_off = row["offpage_count"] > 0
            if has_int and has_off:
                return "intern + Offpage"
            elif has_int:
                return "nur intern"
            elif has_off:
                return "nur Offpage"
            else:
                return "unbekannt"

        tmp["source_flag"] = tmp.apply(_src_flag, axis=1)
        return tmp[["target", "anchor", "count", "source_flag"]]

    # Globale sichtbare Anker + Flags
    anchor_inv_vis_global = build_anchor_vis_for_subset(anchor_inv_internal_global)

    # Für spätere Verwendung (z. B. Sidebar)
    st.session_state["_anchor_inv_internal"] = anchor_inv_internal_global
    st.session_state["_anchor_inv_vis"] = anchor_inv_vis_global

    # --------------------------------------------------------
    # 1) Over-Anchor ≥ Schwellen (absolut)
    # --------------------------------------------------------
    enable_over_anchor = st.session_state.get("a4_enable_over_anchor", True)
    over_anchor_df = pd.DataFrame(columns=["Ziel-URL", "Anchor", "Count", "TopAnchorShare(%)"])

    if enable_over_anchor:
        # Inlinks nach Positionen filtern
        inlinks_over = filter_inlinks_by_position(
            inlinks_df,
            pos_idx,
            key_suffix="over_anchor",
            label="Linkpositionen, die für den Over-Anchor-Check ausgeschlossen werden sollen",
        )

        anchor_inv_internal_over = extract_anchor_inventory(inlinks_over)

        if not anchor_inv_internal_over.empty:
            # Gesamtanzahl Anker je Ziel-URL
            totals = anchor_inv_internal_over.groupby("target")["count"].sum().rename("total")
            tmp = anchor_inv_internal_over.merge(totals, on="target", how="left")

            # Anteil dieses Ankers an allen Ankern der URL (nur Info)
            tmp["share"] = np.where(
                tmp["total"] > 0,
                (100.0 * tmp["count"] / tmp["total"]),
                0.0,
            ).round(2)

            # Nur noch absolute Schwelle: wie viele Vorkommen muss ein Anker haben?
            top_anchor_abs = int(st.session_state.get("a4_top_anchor_abs", 200))

            # Filter: Count >= Schwelle (absolut)
            filt = (tmp["count"] >= top_anchor_abs)

            over_anchor_df = tmp.loc[filt, ["target", "anchor", "count", "share"]].copy()

            # Ziel-URL für Ausgabe in Original-Form bringen
            over_anchor_df["Ziel-URL"] = over_anchor_df["target"].map(disp)

            # Reihenfolge der Spalten neu setzen
            over_anchor_df = over_anchor_df[["Ziel-URL", "anchor", "count", "share"]]
            over_anchor_df.columns = ["Ziel-URL", "Anchor", "Count", "TopAnchorShare(%)"]

    # ---- GSC laden (aus Upload oder ggf. von Analyse 3) ----
    # GSC-Daten aus Upload-Center verwenden
    if gsc_df_loaded is not None:
        gsc_df = gsc_df_loaded.copy()
        st.session_state["__gsc_df_raw__"] = gsc_df
    else:
        gsc_df = st.session_state.get("__gsc_df_raw__", None)

    # GSC-Coverage nur wenn aktiviert
    enable_gsc_coverage = st.session_state.get("a4_enable_gsc_coverage", True)
    
    # Platzhalter für später
    gsc_tab1_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])
    gsc_tab2_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])
    gsc_tab3_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])

    cov_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Match-Typ", "MatchBool", "Fund-Count"])
    
    # GSC-Coverage Settings (immer lesen, auch wenn deaktiviert)
    brand_mode = st.session_state.get("a4_brand_mode", "Nur Non-Brand")
    metric_choice = st.session_state.get("a4_metric_choice", "Impressions")
    min_clicks = int(st.session_state.get("a4_min_clicks", 50))
    min_impr = int(st.session_state.get("a4_min_impr", 500))
    topN_default = int(st.session_state.get("a4_topN", 0))
    auto_variants = bool(st.session_state.get("a4_auto_variants", True))

    if enable_gsc_coverage and isinstance(gsc_df, pd.DataFrame) and not gsc_df.empty:
        df = gsc_df.copy()
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
    
        url_i = _find_idx({"url","page","seite","address","adresse","landingpage","landing page"}, 0)
        q_i   = _find_idx({"query","suchanfrage","suchbegriff"}, 1)
        c_i   = _find_idx({"clicks","klicks"}, None)
        im_i  = _find_idx({"impressions","impr","impressionen","suchimpressionen","search impressions"}, None)
    
        if url_i is None or q_i is None or (c_i is None and im_i is None):
            st.warning("GSC-Datei für A4 benötigt: **URL + Query + (Clicks oder Impressions)**. Bitte Datei prüfen.")
        else:
            # -------------------------------
            # Brand-Liste nur für GSC-Coverage bauen
            # -------------------------------
            brand_text = st.session_state.get("a4_brand_text", "")
            brand_file = st.session_state.get("a4_brand_file", None)

            raw = str(brand_text or "")
            tokens = re.split(r"[,\n;]+", raw)
            brand_list = [t.strip().lower() for t in tokens if t.strip()]

            if brand_file is not None:
                try:
                    import io
                    df_br = pd.read_csv(io.BytesIO(brand_file.getvalue()), header=None)
                    extra = [
                        str(v).strip().lower()
                        for v in df_br.iloc[:, 0].dropna().tolist()
                        if str(v).strip()
                    ]
                    brand_list.extend(extra)
                except Exception as e:
                    st.warning(
                        f"Konnte Brand-Datei nicht lesen ({e}) – Brand-Liste wird nur aus dem Textfeld gebaut."
                    )

            brand_list = sorted(set(brand_list))

            df.iloc[:, url_i] = df.iloc[:, url_i].astype(str).map(remember_original)
            df.iloc[:, q_i]   = df.iloc[:, q_i].astype(str).fillna("").str.strip()
            if c_i is not None:
                df.iloc[:, c_i] = pd.to_numeric(df.iloc[:, c_i], errors="coerce").fillna(0)
            if im_i is not None:
                df.iloc[:, im_i] = pd.to_numeric(df.iloc[:, im_i], errors="coerce").fillna(0)

            def brand_filter(row) -> bool:
                if not brand_list or brand_mode == "Beides":
                    return True
                q = str(row.iloc[q_i])
                is_b = is_brand_query(q, brand_list, auto_variants)
                if brand_mode == "Nur Non-Brand":
                    return not is_b
                elif brand_mode == "Nur Brand":
                    return is_b
                else:
                    return True

            df = df[df.apply(brand_filter, axis=1)]
    
            if c_i is not None and metric_choice == "Clicks":
                df = df[df.iloc[:, c_i] >= int(min_clicks)]
            if im_i is not None and metric_choice == "Impressions":
                df = df[df.iloc[:, im_i] >= int(min_impr)]
    
            if df.empty:
                cov_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Match-Typ", "MatchBool", "Fund-Count"])
            else:
                metric_col = c_i if metric_choice == "Clicks" else im_i

                if metric_choice == "Clicks" and im_i is not None:
                    df = df.sort_values(
                        by=[df.columns[url_i], df.columns[c_i], df.columns[im_i]],
                        ascending=[True, False, False],
                    )
                elif metric_choice == "Impressions" and c_i is not None:
                    df = df.sort_values(
                        by=[df.columns[url_i], df.columns[im_i], df.columns[c_i]],
                        ascending=[True, False, False],
                    )
                else:
                    df = df.sort_values(
                        by=[df.columns[url_i], df.columns[metric_col]],
                        ascending=[True, False],
                    )

                # Top-20 % je URL (mind. 1) + Top-N-Limit
                top_rows = []
                for u, grp in df.groupby(df.columns[url_i], sort=False):
                    n = max(1, int(math.ceil(0.2 * len(grp))))
                    if int(topN_default) > 0:
                        n = max(1, min(n, int(topN_default)))
                    top_rows.append(grp.head(n))
    
                df_top = pd.concat(top_rows) if top_rows else pd.DataFrame(columns=df.columns)
    
                # Anchor-Inventar für GSC-Coverage (Positionsfilter)
                inlinks_gsc = filter_inlinks_by_position(
                    inlinks_df,
                    pos_idx,
                    key_suffix="gsc_cov",
                    label="Linkpositionen, die für die GSC-Query-Coverage ausgeschlossen werden sollen",
                )
                anchor_inv_internal_gsc = extract_anchor_inventory(inlinks_gsc)

                # Anchor-Inventar als Multiset: target -> {anchor: count}
                inv_map: Dict[str, Dict[str, int]] = {}
                for _, r in anchor_inv_internal_gsc.iterrows():
                    inv_map.setdefault(str(r["target"]), {})[str(r["anchor"])] = int(r["count"])
    
                # Embedding-Modell vorbereiten
                check_exact = bool(st.session_state.get("a4_cov_exact", True))
                check_embed = bool(st.session_state.get("a4_cov_embed", False))
                embed_thresh = float(st.session_state.get("a4_cov_embed_thresh", 0.80))
                embed_model_name = st.session_state.get(
                    "a4_cov_embed_model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                )

                model = None
                if check_embed:
                    try:
                        from sentence_transformers import SentenceTransformer
                        if (
                            "_A4_EMB_MODEL_NAME" not in st.session_state
                            or st.session_state.get("_A4_EMB_MODEL_NAME") != embed_model_name
                        ):
                            st.session_state["_A4_EMB_MODEL"] = SentenceTransformer(embed_model_name)
                            st.session_state["_A4_EMB_MODEL_NAME"] = embed_model_name
                        model = st.session_state["_A4_EMB_MODEL"]
                    except Exception as e:
                        st.warning(f"Embedding-Modell konnte nicht geladen werden ({e}). Embedding-Abgleich wird übersprungen.")
                        check_embed = False
    
                def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
                    A = A.astype(np.float32, copy=False)
                    B = B.astype(np.float32, copy=False)
                    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
                    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
                    return A @ B.T
    
                def _norm_text_for_emb(s: str) -> str:
                    s = str(s or "").strip().lower()
                    s = re.sub(r"\s+", " ", s)
                    return s
    
                # Cache Anchor-Embeddings je Ziel-URL
                anchor_emb_cache: Dict[str, Tuple[List[str], List[str], Optional[np.ndarray]]] = {}

                coverage_rows = []  # eine Zeile pro URL+Query

                for u, grp in df_top.groupby(df.columns[url_i], sort=False):
                    url_raw = str(u)
                    url_norm = normalize_url(url_raw)
                    inv = inv_map.get(url_norm, {})
                
                    a_names = list(inv.keys())
                    a_emb = None
                    a_names_norm: List[str] = []

                    # Embeddings je URL vorbereiten
                    if check_embed and model is not None and len(a_names) > 0:
                        if url_norm not in anchor_emb_cache:
                            try:
                                a_names_norm = [_norm_text_for_emb(a) for a in a_names]
                                a_emb = np.asarray(
                                    model.encode(a_names_norm, batch_size=64, show_progress_bar=False)
                                )
                                anchor_emb_cache[url_norm] = (a_names, a_names_norm, a_emb)
                            except Exception:
                                anchor_emb_cache[url_norm] = (a_names, [], None)
                                a_emb = None
                        else:
                            a_names, a_names_norm, a_emb = anchor_emb_cache[url_norm]
                
                    for _, rr in grp.iterrows():
                        q = str(rr.iloc[q_i]).strip()
                        if not q:
                            continue
                
                        found = False
                        found_cnt = 0
                        match_type_parts = []
                
                        # Exact Match
                        if check_exact and a_names:
                            cnt = sum(inv.get(a, 0) for a in a_names if a.lower() == q.lower())
                            if cnt > 0:
                                found = True
                                found_cnt = max(found_cnt, cnt)
                                match_type_parts.append("Exact")
                
                        # Embedding Match
                        if (not found) and check_embed and model is not None and a_emb is not None and len(a_names) > 0:
                            try:
                                q_norm = _norm_text_for_emb(q)
                                q_emb = model.encode([q_norm], show_progress_bar=False)
                                q_emb = np.asarray(q_emb)
                
                                S = cosine_sim_matrix(q_emb, a_emb)[0]
                                idxs = np.where(S >= float(embed_thresh))[0]
                                if idxs.size > 0:
                                    found = True
                                    cnt = int(sum(inv.get(a_names[i], 0) for i in idxs))
                                    found_cnt = max(found_cnt, cnt)
                                    match_type_parts.append("Embedding")
                            except Exception:
                                pass
                
                        match_type = "+".join(match_type_parts) if match_type_parts else "—"
                
                        coverage_rows.append([
                            disp(url_norm),
                            q,
                            match_type,
                            bool(found),
                            int(found_cnt),
                        ])
    
                cov_df = pd.DataFrame(
                    coverage_rows,
                    columns=["Ziel-URL", "Query", "Match-Typ", "MatchBool", "Fund-Count"]
                )
    
            # ---------- Tabs bauen (auf Basis von cov_df) ----------
            if not cov_df.empty:
                agg = (
                    cov_df.groupby("Ziel-URL")
                    .agg(
                        Top_Queries=("Query", "size"),
                        Treffer=("MatchBool", "sum"),
                    )
                    .reset_index()
                )
                agg["Coverage"] = np.where(
                    agg["Top_Queries"] > 0,
                    agg["Treffer"] / agg["Top_Queries"],
                    0.0,
                )

                # Tab 1: URLs mit < 50 % Abdeckung ihrer Top-Queries
                low_cov_urls = agg[agg["Coverage"] < 0.5][["Ziel-URL"]]
                tab1 = cov_df.merge(low_cov_urls, on="Ziel-URL", how="inner")
                tab1["Als_Anker_vorhanden?"] = np.where(tab1["MatchBool"], "ja", "nein")
                gsc_tab1_df = tab1[["Ziel-URL", "Query", "Als_Anker_vorhanden?"]].copy()

                # Tab 2: URLs, deren Top-3-Queries alle nicht als Anchor vorkommen
                rows2 = []
                for url, grp in cov_df.groupby("Ziel-URL", sort=False):
                    top3 = grp.head(3)
                    if top3.empty:
                        continue
                    if not top3["MatchBool"].any():
                        for _, r in top3.iterrows():
                            rows2.append([
                                url,
                                r["Query"],
                                "nein",
                            ])
                gsc_tab2_df = pd.DataFrame(
                    rows2,
                    columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"],
                )

                # Tab 3: URLs, deren Top-Query nicht als Anchor vorkommt
                rows3 = []
                for url, grp in cov_df.groupby("Ziel-URL", sort=False):
                    top1 = grp.head(1)
                    if top1.empty:
                        continue
                    r = top1.iloc[0]
                    if not r["MatchBool"]:
                        rows3.append([
                            url,
                            r["Query"],
                            "nein",
                        ])
                gsc_tab3_df = pd.DataFrame(
                    rows3,
                    columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"],
                )

            else:
                gsc_tab1_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])
                gsc_tab2_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])
                gsc_tab3_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])
    else:
        cov_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Match-Typ", "MatchBool", "Fund-Count"])
        gsc_tab1_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])
        gsc_tab2_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])
        gsc_tab3_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])

    # ============================
    # AUSGABE A4 – 1)–2)
    # ============================

    # 1) Over-Anchor-Check
    if enable_over_anchor:
        st.markdown("#### 1) Over-Anchor-Check (identische Anker pro Ziel-URL)")
        if over_anchor_df.empty:
            st.info("Keine Over-Anchor-Fälle nach den gewählten Schwellen gefunden.")
        else:
            st.dataframe(over_anchor_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Over-Anchor-Check (CSV)",
                data=over_anchor_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="a4_over_anchor_check.csv",
                mime="text/csv",
                key="a4_dl_over_anchor_csv",
            )
            try:
                buf_oa = io.BytesIO()
                with pd.ExcelWriter(buf_oa, engine="xlsxwriter") as xw:
                    over_anchor_df.to_excel(xw, index=False, sheet_name="Over-Anchor")
                buf_oa.seek(0)
                st.download_button(
                    "Download Over-Anchor-Check (XLSX)",
                    data=buf_oa.getvalue(),
                    file_name="a4_over_anchor_check.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="a4_dl_over_anchor_xlsx",
                )
            except Exception:
                pass

    # 2) GSC-Query-Coverage – Tabs 1–3
    if enable_gsc_coverage:
        st.markdown("#### 2) GSC-Query-Coverage (Top-20 % je URL, zusätzlich Top-N-Limit)")

        st.markdown("**Tab 1: URLs mit < 50 % Abdeckung ihrer Top-Queries**")
        if gsc_tab1_df.empty:
            st.info("Keine URLs mit weniger als 50 % Anchor-Abdeckung der Top-Queries gefunden.")
        else:
            st.dataframe(gsc_tab1_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download GSC-Coverage – Tab 1 (CSV)",
                data=gsc_tab1_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="a4_gsc_coverage_tab1_urls_unter_50_prozent.csv",
                mime="text/csv",
                key="a4_dl_cov_tab1_csv"
            )

        st.markdown("---")
        st.markdown("**Tab 2: URLs, deren Top-3-Queries alle nicht als Anchor vorkommen**")
        if gsc_tab2_df.empty:
            st.info("Keine URLs, bei denen alle Top-3-Queries nicht als Anchor vorkommen.")
        else:
            st.dataframe(gsc_tab2_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download GSC-Coverage – Tab 2 (CSV)",
                data=gsc_tab2_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="a4_gsc_coverage_tab2_top3_ohne_anchor.csv",
                mime="text/csv",
                key="a4_dl_cov_tab2_csv"
            )

        st.markdown("---")
        st.markdown("**Tab 3: URLs, deren Top-Query nicht als Anchor vorkommt**")
        if gsc_tab3_df.empty:
            st.info("Keine URLs, bei denen die Top-Query nicht als Anchor vorkommt.")
        else:
            st.dataframe(gsc_tab3_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download GSC-Coverage – Tab 3 (CSV)",
                data=gsc_tab3_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="a4_gsc_coverage_tab3_top1_ohne_anchor.csv",
                mime="text/csv",
                key="a4_dl_cov_tab3_csv"
            )

        # XLSX mit 3 Tabs
        try:
            buf_cov = io.BytesIO()
            with pd.ExcelWriter(buf_cov, engine="xlsxwriter") as xw:
                gsc_tab1_df.to_excel(xw, index=False, sheet_name="Tab1_Coverage")
                gsc_tab2_df.to_excel(xw, index=False, sheet_name="Tab2_Top3_NoAnchor")
                gsc_tab3_df.to_excel(xw, index=False, sheet_name="Tab3_Top1_NoAnchor")
            buf_cov.seek(0)
            st.download_button(
                "Download GSC-Coverage (XLSX, 3 Tabs)",
                data=buf_cov.getvalue(),
                file_name="a4_gsc_coverage_3tabs.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="a4_dl_cov_xlsx_multi"
            )
        except Exception:
            pass

    # 3) Keyword-Coverage mit eigener Keyword-Datei
    if st.session_state.get("a4_enable_kw_coverage", True):
        st.markdown("#### 3) Keyword-Ankertext-Coverage mit eigener Keyword-Datei")
    
        kw_df = kw_df_a4
        if not isinstance(kw_df, pd.DataFrame) or kw_df.empty:
            st.info("Für die Keyword-Ankertext-Coverage wurde noch keine Keyword-Datei hochgeladen.")
        else:
            df_kw = kw_df.copy()
            df_kw.columns = [str(c).strip() for c in df_kw.columns]

            if df_kw.shape[1] < 2:
                st.warning("Die Keyword-Datei benötigt mindestens 2 Spalten: URL + Keyword(s).")
            else:
                url_col = df_kw.columns[0]
                keyword_cols = list(df_kw.columns[1:])

                # Long-Ansicht aus der Rohdatei erzeugen: eine Zeile = URL + Keyword
                pairs = []
                for _, r in df_kw.iterrows():
                    url_raw = str(r[url_col]).strip()
                    if not url_raw:
                        continue
                    for c in keyword_cols:
                        kw_val = str(r[c]).strip()
                        if not kw_val:
                            continue
                        pairs.append([url_raw, kw_val])

            if not pairs:
                st.info("In der Keyword-Datei wurden keine gültigen URL-Keyword-Kombinationen gefunden.")
            else:
                kw_long = pd.DataFrame(pairs, columns=["url_raw", "keyword"])
                kw_long["url_norm"] = kw_long["url_raw"].map(lambda u: normalize_url(u))

                # Anchor-Inventar für Keyword-Coverage (Positionsfilter)
                inlinks_kw = filter_inlinks_by_position(
                    inlinks_df,
                    pos_idx,
                    key_suffix="kw_cov",
                    label="Linkpositionen, die für die Keyword-Coverage ausgeschlossen werden sollen",
                )
                anchor_inv_internal_kw = extract_anchor_inventory(inlinks_kw)

                # Anchor-Inventar in Map-Struktur bringen
                inv_map = {}
                for _, r in anchor_inv_internal_kw.iterrows():
                    tgt = str(r["target"])
                    anc = str(r["anchor"])
                    cnt = int(r["count"])
                    inv_map.setdefault(tgt, {})[anc] = cnt

                if not inv_map:
                    st.info("Es liegen noch keine Anchor-Daten vor (All Inlinks / Anchor-Inventar).")
                else:
                    total_anchor_map = {
                        tgt: sum(cnts.values()) for tgt, cnts in inv_map.items()
                    }

                    # Matching-Optionen (Exact / Embedding) – Defaults, fallen auf bestehende Coverage-Settings zurück
                    check_exact = bool(st.session_state.get("a4_cov_exact", True))
                    check_embed = bool(st.session_state.get("a4_cov_embed", False))
                    embed_thresh = float(st.session_state.get("a4_cov_embed_thresh", 0.80))
                    embed_model_name = st.session_state.get(
                        "a4_cov_embed_model",
                        "sentence-transformers/all-MiniLM-L6-v2",
                    )

                    # Ausgabeformat: Wide vs. Long
                    view_mode = st.radio(
                        "Ausgabeformat für Keyword-Coverage",
                        ["Wide (URL + Keywords nebeneinander)", "Long (je Zeile: URL + Keyword)"],
                        index=0,
                        key="a4_kw_view_mode",
                    )

                    # Embedding-Helfer & Modell (nur bei Bedarf)
                    model = None
                    anchor_emb_cache = {}

                    def _norm_text_for_emb_kw(s: str) -> str:
                        s = str(s or "").strip().lower()
                        s = re.sub(r"\s+", " ", s)
                        return s

                    def cosine_sim_matrix_kw(A: np.ndarray, B: np.ndarray) -> np.ndarray:
                        A = A.astype(np.float32, copy=False)
                        B = B.astype(np.float32, copy=False)
                        A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
                        B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
                        return A @ B.T

                    if check_embed:
                        try:
                            from sentence_transformers import SentenceTransformer
                            if (
                                "_A4_EMB_MODEL_NAME" not in st.session_state
                                or st.session_state.get("_A4_EMB_MODEL_NAME") != embed_model_name
                            ):
                                st.session_state["_A4_EMB_MODEL"] = SentenceTransformer(embed_model_name)
                                st.session_state["_A4_EMB_MODEL_NAME"] = embed_model_name
                            model = st.session_state["_A4_EMB_MODEL"]
                        except Exception as e:
                            st.warning(
                                f"Embedding-Modell konnte nicht geladen werden ({e}). "
                                "Embedding-Matches werden für die Keyword-Coverage deaktiviert."
                            )
                            check_embed = False

                    result_rows = []

                    for url_raw, grp in kw_long.groupby("url_raw", sort=False):
                        url_norm = normalize_url(url_raw)
                        inv = inv_map.get(url_norm, {})
                        total_count = total_anchor_map.get(url_norm, 0)
                    
                        a_names = list(inv.keys())
                        a_emb = None
                        a_names_norm = []
                    
                        if check_embed and model is not None and a_names:
                            if url_norm not in anchor_emb_cache:
                                try:
                                    a_names_norm = [_norm_text_for_emb_kw(a) for a in a_names]
                                    a_emb = np.asarray(
                                        model.encode(a_names_norm, batch_size=64, show_progress_bar=False)
                                    )
                                    anchor_emb_cache[url_norm] = (a_names, a_names_norm, a_emb)
                                except Exception:
                                    anchor_emb_cache[url_norm] = (a_names, [], None)
                                    a_emb = None
                            else:
                                a_names, a_names_norm, a_emb = anchor_emb_cache[url_norm]
                    
                        for _, row in grp.iterrows():
                            kw_str = str(row["keyword"]).strip()
                            if not kw_str:
                                continue
                    
                            found = False
                            used_count = 0
                    
                            # 1) Exact Match
                            if check_exact and a_names:
                                kw_lower = kw_str.lower()
                                cnt_exact = sum(
                                    cnt for a, cnt in inv.items()
                                    if a.lower() == kw_lower
                                )
                                if cnt_exact > 0:
                                    found = True
                                    used_count = cnt_exact
                    
                            # 2) Embedding Match
                            if (not found) and check_embed and model is not None and a_emb is not None and a_names:
                                try:
                                    q_norm = _norm_text_for_emb_kw(kw_str)
                                    q_emb = np.asarray(model.encode([q_norm], show_progress_bar=False))
                                    sims = cosine_sim_matrix_kw(q_emb, a_emb)[0]
                                    idxs = np.where(sims >= float(embed_thresh))[0]
                                    if idxs.size > 0:
                                        cnt_emb = int(sum(inv[a_names[i]] for i in idxs))
                                        if cnt_emb > 0:
                                            found = True
                                            used_count = cnt_emb
                                except Exception:
                                    pass
                    
                            if total_count > 0 and used_count > 0:
                                share_pct = round(100.0 * used_count / float(total_count), 2)
                            else:
                                share_pct = 0.0
                    
                            result_rows.append([
                                url_raw,
                                kw_str,
                                "ja" if found else "nein",
                                share_pct,
                            ])
                    
                    # JETZT erst DataFrame bauen
                    result_df = pd.DataFrame(
                        result_rows,
                        columns=["URL", "Keyword", "Kommt als Ankertext vor?", "Ankertext-Share (%)"],
                    )


                    if not result_rows:
                        st.info("Keine Keyword-Matches gegen das Anchor-Inventar gefunden.")
                    else:
                        if view_mode.startswith("Wide"):
                            wide_rows = []
                            grouped = result_df.groupby("URL", sort=False)
                            max_k = grouped.size().max()

                            cols = ["URL"]
                            for i in range(1, max_k + 1):
                                cols.extend([
                                    f"Keyword {i}",
                                    f"Kommt als Ankertext vor? {i}",
                                    f"Ankertext-Share Keyword {i} (%)",
                                ])

                            for url, grp in grouped:
                                row = [url]
                                for _, r in grp.reset_index(drop=True).iterrows():
                                    row.extend([
                                        r["Keyword"],
                                        r["Kommt als Ankertext vor?"],
                                        float(r["Ankertext-Share (%)"]),
                                    ])
                                while len(row) < len(cols):
                                    row.append("")
                                wide_rows.append(row)

                            wide_df = pd.DataFrame(wide_rows, columns=cols)
                            st.dataframe(wide_df, use_container_width=True, hide_index=True)
                            st.download_button(
                                "Download Keyword-Coverage (Wide, CSV)",
                                data=wide_df.to_csv(index=False).encode("utf-8-sig"),
                                file_name="a4_keyword_coverage_wide.csv",
                                mime="text/csv",
                                key="a4_dl_kw_cov_wide_csv",
                            )
                        else:
                            st.dataframe(result_df, use_container_width=True, hide_index=True)
                            st.download_button(
                                "Download Keyword-Coverage (Long, CSV)",
                                data=result_df.to_csv(index=False).encode("utf-8-sig"),
                                file_name="a4_keyword_coverage_long.csv",
                                mime="text/csv",
                                key="a4_dl_kw_cov_long_csv",
                            )

    # ============================
    # 4) Anchor-Inventar (Wide)
    # ============================
    if st.session_state.get("a4_enable_anchor_matrix", True):

        # Linkpositionen für Matrix (Wide) filtern
        inlinks_matrix = filter_inlinks_by_position(
            inlinks_df,
            pos_idx,
            key_suffix="matrix_wide",
            label="Linkpositionen, die für die Anchor-Matrix (Wide) ausgeschlossen werden sollen",
        )

        anchor_internal_matrix = extract_anchor_inventory(inlinks_matrix)
        anchor_inv_check = build_anchor_vis_for_subset(anchor_internal_matrix)

        if not anchor_inv_check.empty:
            inv_sorted = anchor_inv_check.sort_values(
                ["target", "count"],
                ascending=[True, False]
            ).copy()

            has_flag = (
                "source_flag" in inv_sorted.columns
                and bool(st.session_state.get("a4_include_offpage_anchors", False))
            )

            totals = inv_sorted.groupby("target")["count"].transform("sum")
            inv_sorted["share"] = np.where(
                totals > 0,
                (100.0 * inv_sorted["count"] / totals),
                0.0,
            ).round(2)

            max_n = int(inv_sorted.groupby("target")["anchor"].size().max())

            if has_flag:
                cols = ["Ziel-URL"] + [
                    x
                    for i in range(1, max_n + 1)
                    for x in (
                        f"Ankertext {i}",
                        f"Count Ankertext {i}",
                        f"TopAnchorShare Ankertext {i} (%)",
                        f"Quelle Ankertext {i}",
                    )
                ]
            else:
                cols = ["Ziel-URL"] + [
                    x
                    for i in range(1, max_n + 1)
                    for x in (
                        f"Ankertext {i}",
                        f"Count Ankertext {i}",
                        f"TopAnchorShare Ankertext {i} (%)",
                    )
                ]

            rows = []
            for tgt, grp in inv_sorted.groupby("target", sort=False):
                row = [disp(tgt)]
                if has_flag:
                    for a, c, s, src_flag in zip(
                        grp["anchor"].astype(str),
                        grp["count"].astype(int),
                        grp["share"].astype(float),
                        grp["source_flag"].astype(str),
                    ):
                        row += [a, c, float(s), src_flag]
                else:
                    for a, c, s in zip(
                        grp["anchor"].astype(str),
                        grp["count"].astype(int),
                        grp["share"].astype(float),
                    ):
                        row += [a, c, float(s)]

                while len(row) < len(cols):
                    row.append("")
                rows.append(row)

            anchor_wide_df = pd.DataFrame(rows, columns=cols)

            st.markdown("#### 4) Anchor-Inventar (Wide)")
            st.dataframe(anchor_wide_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download Anchor-Inventar (Wide) – CSV",
                data=anchor_wide_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="a4_anchor_inventar_wide.csv",
                mime="text/csv",
                key="a4_dl_anchor_wide_csv_main",
            )

    # 4b) Anchor-Inventar (Long)
    if st.session_state.get("a4_enable_anchor_matrix_long", False):

        inlinks_matrix_long = filter_inlinks_by_position(
            inlinks_df,
            pos_idx,
            key_suffix="matrix_long",
            label="Linkpositionen, die für die Anchor-Matrix (Long) ausgeschlossen werden sollen",
        )

        anchor_internal_matrix_long = extract_anchor_inventory(inlinks_matrix_long)
        anchor_inv_check = build_anchor_vis_for_subset(anchor_internal_matrix_long)

        if not anchor_inv_check.empty:
            totals = anchor_inv_check.groupby("target")["count"].sum().rename("total")
            tmp = anchor_inv_check.merge(totals, on="target", how="left")
            tmp["TopAnchorShare(%)"] = np.where(
                tmp["total"] > 0,
                (100.0 * tmp["count"] / tmp["total"]).round(2),
                0.0
            )

            long_df = tmp.copy()
            long_df["Ziel-URL"] = long_df["target"].map(disp)
            long_df = long_df[["Ziel-URL", "anchor", "count", "TopAnchorShare(%)"]]

            if "source_flag" in tmp.columns:
                long_df["Quelle"] = tmp["source_flag"]

            st.markdown("#### URL-Ankertext-Matrix (Long-Format)")
            st.dataframe(long_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download URL-Ankertext-Matrix (Long-Format, CSV)",
                data=long_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="a4_anchor_inventar_long.csv",
                mime="text/csv",
                key="a4_dl_anchor_long_csv",
            )
        else:
            st.info("Keine Anchor-Daten für die URL-Ankertext-Matrix (Long-Format) verfügbar.")

    # 5) Shared-Ankertexte
    if st.session_state.get("a4_shared_enable", True):

        inlinks_shared = filter_inlinks_by_position(
            inlinks_df,
            pos_idx,
            key_suffix="shared",
            label="Linkpositionen, die für Shared-Ankertexte ausgeschlossen werden sollen",
        )

        anchor_internal_shared = extract_anchor_inventory(inlinks_shared)
        anchor_inv_check = build_anchor_vis_for_subset(anchor_internal_shared)

        if not anchor_inv_check.empty:
            st.markdown("#### 5) Shared-Ankertexte (gleicher Ankertext für mehrere Ziel-URLs)")

            min_urls_per_anchor = int(st.session_state.get("a4_shared_min_urls", 2))
            ignore_nav = bool(st.session_state.get("a4_shared_ignore_nav", True))

            df_shared = anchor_inv_check.copy()
            df_shared["target"] = df_shared["target"].astype(str)
            df_shared["anchor"] = df_shared["anchor"].astype(str)

            if ignore_nav:
                df_shared = df_shared[
                    ~df_shared["anchor"].str.strip().str.lower().isin(NAVIGATIONAL_ANCHORS)
                ]

            has_flag = "source_flag" in df_shared.columns

            anchor_source_map = {}
            if has_flag and bool(st.session_state.get("a4_include_offpage_anchors", False)):

                def _merge_flags(flags):
                    flags = {f for f in flags if pd.notna(f)}
                    if not flags:
                        return "unbekannt"
                    if "intern + Offpage" in flags:
                        return "intern + Offpage"
                    has_int = "nur intern" in flags
                    has_off = "nur Offpage" in flags
                    if has_int and has_off:
                        return "intern + Offpage"
                    if has_int:
                        return "nur intern"
                    if has_off:
                        return "nur Offpage"
                    return sorted(flags)[0]

                anchor_source_map = (
                    df_shared.groupby("anchor")["source_flag"]
                    .agg(lambda s: _merge_flags(set(s)))
                    .to_dict()
                )

            grouped = (
                df_shared.groupby("anchor")["target"]
                .agg(lambda s: sorted({disp(t) for t in s}))
                .reset_index(name="urls")
            )
            grouped["url_count"] = grouped["urls"].apply(len)
            grouped = grouped[grouped["url_count"] >= min_urls_per_anchor]

            if has_flag and anchor_source_map:
                grouped["Quelle"] = grouped["anchor"].map(anchor_source_map).fillna("unbekannt")

            if grouped.empty:
                st.info("Keine Shared-Ankertexte nach den gesetzten Filtern gefunden.")
            else:
                rows = []
                for _, r in grouped.sort_values("url_count", ascending=False).iterrows():
                    anchor_txt = r["anchor"]
                    urls = r["urls"]
                    url_count = int(r["url_count"])
                    quelle_val = r.get("Quelle", None) if "Quelle" in grouped.columns else None

                    for u in urls:
                        if quelle_val is not None:
                            rows.append([anchor_txt, quelle_val, u, url_count])
                        else:
                            rows.append([anchor_txt, u, url_count])

                if "Quelle" in grouped.columns:
                    shared_long_df = pd.DataFrame(
                        rows,
                        columns=["Ankertext", "Quelle", "Ziel-URL", "Anzahl_Ziel-URLs_für_Anker"]
                    )
                else:
                    shared_long_df = pd.DataFrame(
                        rows,
                        columns=["Ankertext", "Ziel-URL", "Anzahl_Ziel-URLs_für_Anker"]
                    )

                st.dataframe(shared_long_df, use_container_width=True, hide_index=True)

                st.download_button(
                    "Download Shared-Ankertexte (CSV)",
                    data=shared_long_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="a4_shared_ankertexte_long.csv",
                    mime="text/csv",
                    key="a4_dl_shared_csv_main",
                )

    # ============================
    # Treemap-Visualisierung (Hauptbereich)
    # ============================
    show_treemap = bool(st.session_state.get("a4_show_treemap", True))
    
    if show_treemap and _HAS_PLOTLY:
        inlinks_treemap = filter_inlinks_by_position(
            inlinks_df,
            pos_idx,
            key_suffix="treemap",
            label="Linkpositionen, die für die Treemap ausgeschlossen werden sollen",
        )

        anchor_internal_treemap = extract_anchor_inventory(inlinks_treemap)
        anchor_inv_check = build_anchor_vis_for_subset(anchor_internal_treemap)

        if not anchor_inv_check.empty:
            st.markdown("#### Treemap der häufigsten Anchors je Ziel-URL")
    
            treemap_topK = int(st.session_state.get("a4_treemap_topk", 12))
            treemap_url_mode = st.session_state.get("a4_treemap_url_mode", "Alle URLs")
            selected_urls = st.session_state.get("a4_treemap_selected_urls") or []
    
            all_targets = sorted(anchor_inv_check["target"].astype(str).unique())
    
            if treemap_url_mode == "Ausgewählte URLs" and selected_urls:
                targets = [u for u in selected_urls if u in all_targets]
            else:
                targets = all_targets
    
            if not targets:
                st.info("Keine passenden Ziel-URLs für die Treemap-Auswertung gefunden.")
            else:
                preview_url = st.selectbox(
                    "URL für Treemap-Vorschau auswählen",
                    options=targets,
                    format_func=lambda u: disp(u),
                    key="a4_treemap_preview_url"
                )
    
                import re
    
                def build_treemap_for_target(target: str):
                    grp = anchor_inv_check[anchor_inv_check["target"].astype(str) == str(target)].copy()
                    grp = grp.sort_values("count", ascending=False).head(treemap_topK)
                    if grp.empty:
                        return None, None

                    df_t = grp[["anchor", "count"]].rename(columns={"anchor": "Anchor", "count": "Count"})

                    fig = px.treemap(
                        df_t,
                        path=["Anchor"],
                        values="Count",
                        title=f"Treemap: häufigste Anchors für {disp(target)}"
                    )

                    fig.update_traces(
                        textposition="middle center",
                        textinfo="label+value",
                        textfont_size=16,
                    )

                    html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

                    return fig, html_bytes

                fig, html_bytes = build_treemap_for_target(preview_url)
    
                if fig is None:
                    st.info("Für die ausgewählte URL liegen keine Ankerdaten vor.")
                else:
                    st.plotly_chart(fig, use_container_width=True)
    
                    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", str(preview_url))[:60] or "url"
                    st.download_button(
                        "Treemap für ausgewählte URL herunterladen (HTML)",
                        data=html_bytes,
                        file_name=f"treemap_anchors_{safe_name}.html",
                        mime="text/html",
                        key="a4_dl_treemap_single"
                    )
    
                export_rows = []
                for tgt in targets:
                    grp = anchor_inv_check[anchor_inv_check["target"].astype(str) == str(tgt)].copy()
                    grp = grp.sort_values("count", ascending=False).head(treemap_topK)
                    for _, row in grp.iterrows():
                        export_rows.append([disp(tgt), str(row["anchor"]), int(row["count"])])
    
                if export_rows:
                    treemap_export_df = pd.DataFrame(export_rows, columns=["Ziel-URL", "Anchor", "Count"])
                    st.download_button(
                        "Treemap-Basisdaten (alle ausgewählten URLs, CSV)",
                        data=treemap_export_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="treemap_anchor_basisdaten.csv",
                        mime="text/csv",
                        key="a4_dl_treemap_csv"
                    )
        else:
            st.info("Keine Anchor-Daten für die Treemap-Auswertung verfügbar.")
    
    elif show_treemap and not _HAS_PLOTLY:
        st.info("Plotly ist nicht verfügbar – Treemap wird übersprungen.")

    # Abschluss: Loader abbauen, Status setzen
    st.session_state["__a4_loading__"] = False
    st.session_state["__ready_a4__"] = True
    try:
        ph4 = st.session_state.get("__a4_ph__")
        if ph4 is not None:
            ph4.empty()
    except Exception:
        pass
