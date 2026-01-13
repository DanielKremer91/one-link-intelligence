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

  Folge mich auf <a href="https://www.linkedin.com/in/daniel-kremer-b38176264/" target="_blank">LinkedIn</a> für mehr SEO-Insights und Tool-Updates

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

                    pairs.append([urls[i], urls[j], s])

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

    - Seitendaten (Title/H1/Meta) - OHNE Main Content

    - GSC-Top-Keywords

    - manuelle Keywords

    und ruft entweder OpenAI oder Gemini auf.

    KEIN FALLBACK - bei Fehlern wird eine Exception geworfen.

    """

    title_map = page_maps.get("title", {})

    h1_map    = page_maps.get("h1", {})

    meta_map  = page_maps.get("meta", {})

    fields = cfg.get("fields", [])

    use_gsc = cfg.get("use_gsc", "Nicht verwenden")

    gsc_metric = "Clicks" if use_gsc == "Top-Keyword nach Klicks" else "Impressions"

    use_manual = cfg.get("use_manual", False)

    title = title_map.get(url, "")

    h1    = h1_map.get(url, "")

    meta  = meta_map.get(url, "")

    

    # GSC: Liste von Keywords (Top 10)

    top_kws = top_kw_map.get(url, []) if use_gsc != "Nicht verwenden" else []

    manual_kws = manual_kw_map.get(url, []) if use_manual else []

    

    # Marke aus Seitendaten entfernen (falls vorhanden)

    def remove_brand_from_text(text: str) -> str:

        """Entfernt Markennamen und Pipe-Zeichen aus Text."""

        if not text:

            return text

        # Entferne Pipe-Zeichen und alles danach (typisch: "| MARKENNAME")

        text = re.sub(r'\s*\|\s*[^|]*$', '', text)

        text = re.sub(r'\s*\|\s*', ' ', text)  # Entferne alle Pipe-Zeichen

        return text.strip()

    

    # Marke aus Seitendaten entfernen

    title = remove_brand_from_text(title)

    h1 = remove_brand_from_text(h1)

    meta = remove_brand_from_text(meta)

    

    blocks = []

    blocks.append(f"Ziel-URL: {url}")

    

    # Priorisierung: 1) Manuelle Keywords (höchste Priorität)

    if manual_kws:

        blocks.append(

            "⚠️ HÖCHSTE PRIORITÄT – Manuell hochgeladene Keywords (diese MÜSSEN bevorzugt verwendet werden): "

            + ", ".join(manual_kws)

        )

    

    # Priorisierung: 2) GSC Top-Keywords (zweithöchste Priorität)

    if top_kws:

        blocks.append(

            f"⚠️ HOHE PRIORITÄT – Top-Keywords aus Search Console ({gsc_metric}, Top 10, nutze ALLE für Varianten): "

            + ", ".join(top_kws)

        )

    

    # Priorisierung: 3) Seitendaten (Title, H1, Meta Description) - Marke bereits entfernt

    # WICHTIG: Main Content wird NICHT mehr verwendet

    if "Title" in fields and title:

        blocks.append(f"Title: {title}")

    if "H1" in fields and h1:

        blocks.append(f"H1: {h1}")

    if "Meta Description" in fields and meta:

        blocks.append(f"Meta Description: {meta}")

    

    page_info = "\n".join(blocks)

    

    system_prompt = (

        "Du bist ein erfahrener SEO-Experte. "

        "Du erstellst präzise, natürliche und sinnvolle Ankertexte für interne Verlinkungen "

        "und hältst dich strikt an das geforderte Ausgabeformat. "

        "Du orientierst dich stark an den vorhandenen Formulierungen aus Title Tag, H1 und Meta Description."

    )

    

    user_prompt = f"""

    Erzeuge 3 unterschiedliche Ankertexte für eine interne Verlinkung zu der folgenden Zielseite.

    

    Seitendaten:

    {page_info}

    

    WICHTIG – Priorisierung und Vorgehen für die Ankertext-Generierung:

    

    1. HÖCHSTE PRIORITÄT: Manuell hochgeladene Keywords (falls vorhanden) – diese MÜSSEN bevorzugt verwendet werden.

    

    2. HOHE PRIORITÄT: Top-Keywords aus Search Console (falls vorhanden) – nutze ALLE bereitgestellten GSC-Keywords für Varianten, nicht nur das stärkste. Wenn mehrere Keywords vorhanden sind, baue verschiedene Varianten mit unterschiedlichen Keywords ein.

    

    3. KERN-AUFGABE: Leite das Hauptthema aus dem Title Tag ab (z.B. "Zeckenbiss beim Hund" aus "Zeckenbiss beim Hund: So reagierst du richtig | FRESSNAPF").

    

    4. BESCHREIBENDE ELEMENTE: Extrahiere und nutze beschreibende Elemente, Adjektive, Verben und wichtige Begriffe aus:

       - Title Tag (z.B. "richtig reagieren" aus "So reagierst du richtig")

       - H1 (z.B. "so reagierst du richtig")

       - Meta Description (z.B. "Erreger und Symptome", "Krankheiten übertragen")

    

    5. ORIENTIERUNG: Die Ankertexte sollen sich STARK an den vorhandenen Formulierungen aus Title, H1 und Meta Description orientieren. Nutze die exakten Formulierungen, Adjektive, Verben und beschreibenden Begriffe aus diesen Feldern.

    

    Ziel:

    - Der Ankertext soll das Hauptthema der Zielseite klar widerspiegeln (abgeleitet aus dem Title Tag).

    - Nutze beschreibende Elemente aus Title, H1 und Meta Description (z.B. "richtig reagieren", "Erreger und Symptome").

    - Kombiniere das Hauptthema mit den beschreibenden Elementen.

    - Vermeide harte Überoptimierung (keine unnatürlich gehäuften Keywords).

    - Sei kreativ, aber bleibe nah an den vorhandenen Formulierungen.

    

    Erzeuge GENAU diese 3 Varianten:

    1. Kurz & präzise – 2–3 Wörter, fokussiert auf das Hauptthema (aus Title Tag das Hauptkeyword bzw. Hauptthema ableiten) + ggf. ein beschreibendes Element. Bleibe aber bei maximal 3 Wörter.

    2. Beschreibend – 3–5 Wörter, mit mehr Kontext, nutze beschreibende Elemente aus Title/H1/Meta.

    3. Handlungsorientiert – 3–5 Wörter, mit klarer Nutzen- oder Handlungsorientierung, nutze Verben und aktive Formulierungen aus Title/H1/Meta, aber ohne generische Floskeln wie „hier klicken".

    

    Strikte Regeln:

    - Sprache: ausschließlich natürliches, korrektes Deutsch.

    - Beachte die Höchstzahl an Wörter. Nicht mehr als 3 Wörter bei Kurz & präzise. Maximal 5 Wörter bei Beschreibend und Handlungsorientiert.

    - Rechtschreibung: Achte auf korrekte Groß- und Kleinschreibung. Wenn Search Console Keywords z.B. "katzen agility" (kleingeschrieben) enthalten, schreibe im Ankertext korrekt "Katzen Agility" (mit Großbuchstaben am Wortanfang). Wenn im Title Tag "Zeckenbiss beim Hund" steht, schreibe auch im Ankertext "Zeckenbiss beim Hund" (nicht "zeckenbiss beim hund").

    - KEIN Pipe-Zeichen (|): Das Zeichen "|" darf NIEMALS im Ankertext vorkommen.

    - KEINE Marke: Wenn in den Seitendaten eine Marke vorkommt (z.B. "| FRESSNAPF" im Title), darf diese Marke NIEMALS im Ankertext erscheinen. Entferne Markennamen komplett aus den Ankertexten.

    - Keine generischen Phrasen wie "hier klicken", "mehr erfahren", "weiterlesen", "klicke hier" oder ähnlich.

    - Keine Emojis, keine Anführungszeichen, keine Nummerierung, keine Aufzählungszeichen.

    - Keine HTML-Tags und keine Sonderzeichen wie <, >, #, *, /, |.

    - Keine reinen Brand-Ankertexte; Markennamen sind komplett zu vermeiden.

    - Jede der 3 Varianten muss sich in Bedeutung und Wortlaut klar von den anderen unterscheiden.

    - ORIENTIERUNG AN FORMULIERUNGEN: Orientiere dich an Formulierungen, Begriffen und Phrasen aus Title, H1 und Meta Description.

    

    AUSGABEFORMAT:

    Die 3 Ankertexte müssen mit "|||" (drei Pipe-Zeichen) getrennt werden, eine pro Zeile oder in einer Zeile.

    Beispiel:

    Zeckenbiss beim Hund|||richtig reagieren|||Erreger und Symptome

    

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

            resp = client.chat.completions.create(
                model=cfg.get("openai_model", "gpt-5.1"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                temperature=0.5,
            )
            text = resp.choices[0].message.content
            
            if not text or not text.strip():
                raise ValueError(f"OpenAI API hat leeren Text zurückgegeben. Response: {resp}")

        else:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            # Versuche verschiedene Modellnamen (Fallback auf verfügbare Modelle)
            model_candidates = [
                cfg.get("gemini_model"),  # Vom User konfiguriert
                "gemini-1.5-flash-latest",  # Meist verfügbar
                "gemini-1.5-pro-latest",   # Alternative Pro-Version
                "gemini-1.5-flash",        # Ohne -latest Suffix
                "gemini-1.5-pro",          # Original (falls doch verfügbar)
                "gemini-pro",              # Ältere Version
            ]
            
            model = None
            last_error = None
            
            for model_name in model_candidates:
                if not model_name:
                    continue
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
                raise RuntimeError(
                    f"Kein verfügbares Gemini-Modell gefunden. Versuchte Modelle: {[m for m in model_candidates if m]}\n"
                    f"Letzter Fehler: {str(last_error)}\n"
                    f"Bitte prüfe die verfügbaren Modelle mit: genai.list_models()"
                ) from last_error

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


