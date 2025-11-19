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
POSSIBLE_POSITION = ["linkposition", "link position", "position"]

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


    # Parsing wie in A1/A2/A3
    emb_df = emb_df.copy()
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

    url_i = pick_col(["url", "urls", "page", "seite", "address", "adresse", "landingpage", "landing page"], 0)
    emb_i = pick_col(
        [
            "embedding","embeddings","embedding json","embedding_json","text embedding",
            "openai embedding","sentence embedding","vector","vec","content embedding",
            "sf embedding","embedding 1536","embedding_1536"
        ],
        1 if emb_df.shape[1] >= 2 else None
    )
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
        return None

    dims = [v.size for v in vecs]
    max_dim = max(dims)
    V = np.zeros((len(vecs), max_dim), dtype=np.float32)
    for i, v in enumerate(vecs):
        d = min(max_dim, v.size)
        V[i, :d] = v[:d]
    V = l2_normalize(V)
    backend_eff = prefer_backend
    df_rel = build_related_auto(list(urls), V, int(top_k), float(sim_threshold), backend_eff, mem_budget_gb=1.5)
    return df_rel


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
                st.caption("Gleicht die Top 20% der Suchanfragen einer URL (basierend auf Search Console Daten) mit vorhandenen Ankertexten ab und identifiziert fehlende oder falsch verlinkte Queries.")
                
                # Wichtig: Reihenfolge – zuerst welche Queries berücksichtigen
                brand_mode = st.radio(
                    "Sollen auch Brand-Suchanfragen bei dieser Analyse berücksichtigt werden?",
                    ["Nur Non-Brand", "Nur Brand", "Beides"],
                    index=0,
                    horizontal=True,
                    key="a4_brand_mode",
                    help="Filtert GSC-Queries nach Brand/Non-Brand bevor die Auswertung startet."
                )
                
                # Brand-Schreibweisen IMMER anzeigen, wenn GSC-Coverage aktiv ist
                brand_text = st.text_area(
                    "Brand-Schreibweisen (eine pro Zeile oder komma-getrennt)",
                    value=st.session_state.get("a4_brand_text", ""),
                    key="a4_brand_text",
                    help="Liste von Marken-Schreibweisen; wird für Brand/Non-Brand-Erkennung verwendet."
                )
                brand_file = st.file_uploader(
                    "Optional: Brand-Liste (1 Spalte)",
                    type=["csv", "xlsx", "xlsm", "xls"],
                    key="a4_brand_file",
                    help="Einspaltige Liste; zusätzliche Spalten werden ignoriert."
                )
                auto_variants = st.checkbox(
                    "Branded Keywords auch als Brand-Keywords behandeln? (Kombis wie 'marke keyword', 'keyword marke', 'marke-keyword' usw.)",
                    value=st.session_state.get("a4_auto_variants", True),
                    key="a4_auto_variants",
                    help="Alle Keyword+Brand-Kombinationen werden als Brand-Queries erkannt."
                )

                # --- Relevanzgrundlage ---
                metric_choice = st.radio(
                    "Sollen die Top 20 % Suchanfragen auf Basis der Klicks oder Impressionen analysiert werden?",
                    ["Impressions", "Clicks"],
                    index=0,
                    horizontal=True,
                    key="a4_metric_choice"
                )
                
                st.caption(
                    "Soll der Abgleich der Search Console Queries mit den Ankertexten als Exact Match "
                    "oder auf Basis semantischer Ähnlichkeit erfolgen?"
                )

                check_exact = st.checkbox("Exact Match prüfen", value=True, key="a4_check_exact")
                check_embed = st.checkbox("Embedding Match prüfen", value=True, key="a4_check_embed")

                embed_model_name = st.selectbox(
                    "Sentence Transformer Modell (Embedding-Modell)",
                    [
                        "sentence-transformers/all-MiniLM-L6-v2",
                        "sentence-transformers/all-MiniLM-L12-v2",
                        "sentence-transformers/all-mpnet-base-v2",
                    ],
                    index=0,
                    help="Standard: all-MiniLM-L6-v2",
                    key="a4_embed_model",
                )
                embed_thresh = st.slider(
                    "Cosine-Schwelle (Embedding)",
                    0.50, 0.95, 0.75, 0.01,
                    key="a4_embed_thresh",
                    help="Nur Anchors mit Cosine Similarity ≥ Schwelle gelten als semantische Treffer."
                )

                help_text_schwellen = (
                    "Mit den Schwellen & Filtern reduzierst du Rauschen und fokussierst die Analyse auf wirklich relevante Suchanfragen:\n\n"
                    "• Mindest-Klicks/Query – wird nur angewendet, wenn oben Clicks ausgewählt ist. Filtert Suchanfragen mit zu wenigen Klicks heraus.\n"
                    "• Mindest-Impressions/Query – wird nur angewendet, wenn oben Impressions ausgewählt ist. Filtert Queries mit Impressionen unter dem gewählten Schwellenwert heraus.\n"
                    "• Top-N Queries pro URL – zusätzlicher Deckel nach der Top-20-%-Auswahl. Pro URL werden maximal N der stärksten Queries (aus dem Topf der Top-20 %) geprüft (mindestens 1).\n\n"
                    "Hinweise:\n"
                    "– Die Auswahl Impressions vs. Clicks steuert, welche Schwelle greift.\n"
                    "– Erst werden Marke/Non-Brand und Mindestwerte gefiltert, dann die Top-20-% berechnet, und anschließend per Top-N begrenzt."
                )
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    min_clicks = st.number_input(
                        "Mindest-Klicks/Query",
                        min_value=0,
                        value=50,
                        step=10,
                        key="a4_min_clicks",
                        help=help_text_schwellen
                    )
                with col_s2:
                    min_impr = st.number_input(
                        "Mindest-Impressions/Query",
                        min_value=0,
                        value=500,
                        step=50,
                        key="a4_min_impr",
                        help=help_text_schwellen
                    )
                with col_s3:
                    topN_default = st.number_input(
                        "Top-N Queries pro URL (zusätzliche Bedingung)",
                        min_value=0,  # 0 = kein Zusatz-Deckel
                        value=st.session_state.get("a4_topN", 0),
                        step=1,
                        key="a4_topN",
                        help=(
                            help_text_schwellen
                            + "\n\nHinweis: **0 = kein zusätzlicher Top-N-Deckel** (es gelten nur die Top-20 % je URL)."
                        )
                    )

            else:
                # Setze Defaults wenn deaktiviert
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

            # =========================
            # NEU: Semantische Ankertext-Passung / Drift
            # =========================
            enable_semantic_anchor = st.checkbox(
                "Semantische Ankertext-Passung / Anchor-Drift analysieren",
                value=True,
                key="a4_enable_semantic_anchor",
                help=(
                    "Vergleicht je Ziel-URL die Ankertexte semantisch mit dem Seiten-Content. "
                    "Output: Übersicht (Ø-Similarity & Drift) und Detailtabelle je Ankertext."
                ),
            )

            if enable_semantic_anchor:
                st.markdown("**Semantische Ankertext-Passung / Drift**")
                st.caption(
                    "Wir berechnen Embeddings für Seiteninhalt und Ankertexte und messen die Cosine Similarity. "
                    "So erkennst du URLs, deren Ankertexte nicht mehr sauber zum Content passen (Anchor-Drift)."
                )

                a4_emb_mode = st.radio(
                    "Auf welcher Basis sollen die Embeddings für die URLs berechnet werden?",
                    [
                        "Basierend auf einer oder mehreren Spalten der Crawl-Datei",
                        "Embedding-Spalte in Crawl-Datei nutzen",
                        "Separate Embedding-Datei (URL + Embedding)",
                    ],
                    index=0,
                    key="a4_emb_mode",
                    help=(
                        "• Basierend auf einer oder mehreren Spalten der Crawl-Datei: Embeddings werden aus Content-Spalten des Crawls erzeugt.\n"
                        "• Embedding-Spalte in Crawl-Datei nutzen: Embeddings liegen im Crawl bereits vor.\n"
                        "• Separate Embedding-Datei: eigene Datei mit URL + Embedding-Spalte nur für diese Analyse."
                    ),
                )

                # 🔥 NEU: Modell-Auswahl NUR für den Modus „Basierend auf einer oder mehreren Spalten der Crawl-Datei“
                if a4_emb_mode == "Basierend auf einer oder mehreren Spalten der Crawl-Datei":
                    st.selectbox(
                        "Sentence Transformer Modell (Embedding-Modell)",
                        [
                            "sentence-transformers/all-MiniLM-L6-v2",
                            "sentence-transformers/all-MiniLM-L12-v2",
                            "sentence-transformers/all-mpnet-base-v2",
                        ],
                        index=0,
                        key="a4_sem_embed_model",
                        help="Dieses Modell wird für die Seiten-Embeddings und die Ankertext-Embeddings in der Semantik-Analyse verwendet.",
                    )

                    st.markdown("**Welche Spalten aus der Crawl-Datei sollen für die Embedding-Berechnung genutzt werden?**")
                
                    # Spalten aus Session-State (beim Upload gesetzt)
                    crawl_cols_for_select = st.session_state.get("a4_crawl_columns", [])
                
                    if crawl_cols_for_select:
                        # Multiselect aus den tatsächlich vorhandenen Crawl-Spalten
                        default_sel = st.session_state.get("a4_text_cols_list", [])
                        default_sel = [c for c in default_sel if c in crawl_cols_for_select]
                
                        a4_text_cols_list = st.multiselect(
                            "Spalten aus Crawl wählen",
                            options=crawl_cols_for_select,
                            default=default_sel,
                            help="Diese Spalten werden zu einem Text pro URL kombiniert und für die Seiten-Embeddings genutzt.",
                            key="a4_text_cols_list",
                        )
                
                        # Optional: Freitext-Fallback zusätzlich anbieten (für spätere Crawls mit anderen Spaltennamen)
                        st.text_input(
                            "Zusätzliche Spaltennamen (optional, frei eintragen)",
                            key="a4_text_cols",
                            help="Kommagetrennt oder zeilenweise – falls du Spalten per Namen referenzieren möchtest, die im aktuellen Crawl (noch) nicht vorkommen."
                        )
                    else:
                        # Fallback, wenn noch kein Crawl geladen ist
                        st.text_input(
                            "Die ausgewählten Spalten werden kombiniert und die Embeddings berechnet – WICHTIG: Zuerst Crawl-Datei hochladen, dann Spalten auswählen",
                            key="a4_text_cols",
                            help="Kommagetrennt oder zeilenweise – wird als Textgrundlage für Seiten-Embeddings genutzt."
                        )


                # Embedding-Spalte aus der Crawl-Datei wählen / benennen
                if a4_emb_mode == "Embedding-Spalte in Crawl-Datei nutzen":
                    crawl_cols_for_select = st.session_state.get("a4_crawl_columns", [])

                    if crawl_cols_for_select:
                        st.selectbox(
                            "Embeddings-Spalte in der Crawl-Datei",
                            options=crawl_cols_for_select,
                            key="a4_emb_col",
                            help="Spalte wählen, die die Embeddings pro URL enthält (z. B. JSON oder Vektor).",
                        )
                    else:
                        st.text_input(
                            "Name der Embeddings-Spalte in der Crawl-Datei",
                            key="a4_emb_col",
                            help="Wird genutzt, um die Embeddings pro URL auszulesen (z. B. 'Embedding', 'Vector').",
                        )


                st.slider(
                    "Mindest-Cosine-Similarity (Ø Anker → Seite)",
                    0.0,
                    1.0,
                    0.70,
                    0.01,
                    key="a4_sem_sim_thresh",
                    help="URLs unterhalb dieser Ähnlichkeit werden als Drift-Fälle markiert.",
                )

            # Abstand / Trennlinie zur nächsten Unteranalyse
            st.markdown(
                "<div style='margin:18px 0; border-bottom:1px solid #eee;'></div>",
                unsafe_allow_html=True,
            )



            
            # --- Visualisierung (A4) ---
            st.markdown("**Ankertext-Matrix & Visualisierung**")
            st.caption(
                "Lasse dir je URL die häufigsten Ankertexte visuell oder als CSV ausgeben. "
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
            
            show_treemap = st.checkbox(
                "Treemap-Visualisierung aktivieren",
                value=True,
                key="a4_show_treemap",
                help=(
                    "Schaltet die Treemap ein/aus. Die Treemap zeigt je Ziel-URL die häufigsten Ankertexte. "
                    "Grundlage sind die Anker aus All Inlinks und – falls aktiviert – aus der Offpage-Ankerdatei."
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
needs_crawl_a4_sem = A4_NAME in selected_analyses and st.session_state.get("a4_enable_semantic_anchor", False)



# ===============================
# Upload-Center: nach Analysen getrennt + "Für mehrere Analysen benötigt"
# ===============================
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
crawl_df_a4 = None
emb_df_a4 = None

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
HELP_CRAWL_A4 = (
    "Crawl-Datei mit mindestens einer URL-Spalte und einer oder mehreren Content-Spalten "
    "(z. B. 'Main Content', 'H1', 'Title'). "
    "Diese Spalten können in A4 für die Berechnung der Seiten-Embeddings ausgewählt werden."
)



# =====================================================
# Gemeinsame Sektion (falls mehrfach benötigt)
# =====================================================

# 1) Globaler Eingabemodus: URLs + Embeddings vs. Related URLs
if needs_embeddings_or_related:
    mode = st.radio(
        "Eingabemodus (für Analysen 1–3, 5 & 6)",
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
    # Wenn der Radio nicht angezeigt wird (weil gerade keine Analyse Embeddings/Related braucht),
    # alten Wert weiterverwenden oder Default setzen
    mode = st.session_state.get("emb_rel_mode_global", "Related URLs")

# ❗ WICHTIG:
# KEIN st.session_state["emb_rel_mode_global"] = mode mehr!
# Der Wert wird vom Radio-Widget selbst in den Session State geschrieben.

# Gemeinsame Sektion (falls mehrfach benötigt)
if shared_uploads:
    st.markdown("### Für mehrere Analysen benötigt")

# Spalten IMMER anlegen, damit colA/colB existieren
colA, colB = st.columns(2)

# mode ist jetzt schon gesetzt – falls du später woanders nochmal drauf zugreifen willst,
# kannst du dort einfach st.session_state.get("emb_rel_mode_global", "Related URLs") verwenden.

# Spalten IMMER anlegen, damit colA/colB existieren
colA, colB = st.columns(2)

# Kein zweites Radio mehr – nur noch auslesen
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

# A1
if A1_NAME in selected_analyses:
    needs = []
    if needs_embeddings_or_related and "URLs + Embeddings" not in shared_uploads and "Related URLs" not in shared_uploads:
        # zeigen wir beide Alternativen, Steuerung per Radio oben: wähle 'mode'
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
    for (label, df) in upload_for_analysis("Analyse 1 interne Verlinkungsmöglichkeiten finden – erforderliche Dateien", needs):
        if "Embeddings" in label: emb_df = df
        if "Related URLs" in label: related_df = df
        if "Inlinks" in label: inlinks_df = df
        if "Linkmetriken" in label: metrics_df = df
        if "Backlinks" in label: backlinks_df = df

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
    # Offpage-Ankerdatei nur anbieten, wenn Option aktiviert
    if st.session_state.get("a4_include_offpage_anchors", False):
        needs.append((
            "Offpage-Ankertexte (CSV/Excel)",
            "up_offpage_anchors_a4",
            "Struktur: mindestens Ziel-URL + Ankertext. "
            "Ziel-URL wird ähnlich wie in 'All Inlinks' erkannt, Ankertext über Spaltennamen wie "
            "'Anchor', 'Anchor Text', 'Anker', 'Ankertext' etc."
        ))

    # NEU: Crawl + optionale Embedding-Datei für semantische Ankeranalyse
    if st.session_state.get("a4_enable_semantic_anchor", False):
        a4_mode = st.session_state.get("a4_emb_mode", "Basierend auf einer oder mehreren Spalten der Crawl-Datei")

        # Crawl nur, wenn wir ihn wirklich brauchen
        if a4_mode in ["Basierend auf einer oder mehreren Spalten der Crawl-Datei", "Embedding-Spalte in Crawl-Datei nutzen"]:
            needs.append(("Crawl (CSV/Excel)", "up_crawl_a4", HELP_CRAWL_A4))

        # Separate Embedding-Datei nur im entsprechenden Modus
        if a4_mode == "Separate Embedding-Datei (URL + Embedding)":
            needs.append((
                "URLs + Embeddings für semantische Ankeranalyse (CSV/Excel)",
                "up_emb_a4",
                HELP_EMB
            ))


    if needs:
        for (label, df) in upload_for_analysis(
            "Analyse 4 Ankertexte analysieren – erforderliche Dateien", needs
        ):
            if "Inlinks" in label:
                inlinks_df = df
            if "Search Console" in label:
                gsc_df_loaded = df
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
            related_df = build_related_auto(list(urls), V, int(locals().get("max_related", 10)), float(locals().get("sim_threshold", 0.8)), backend_eff, mem_budget_gb=1.5)
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
            for target, related_list in sorted(related_map.items()):
                related_sorted = sorted(related_list, key=lambda x: x[1], reverse=True)[: int(locals().get("max_related", 10))]
                row_norm = [target]
                row_view = [disp(target)]
                for source, sim in related_sorted:
                    anywhere = "ja" if (source, target) in st.session_state["_all_links"] else "nein"
                    from_content = "ja" if (source, target) in st.session_state["_content_links"] else "nein"
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
            max_i = int(locals().get("max_related", 10))
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

    # -------------------------------------------------
    # Ab hier: komplette A4-Auswertung
    # -------------------------------------------------

    # ---- Anchor-Inventar aus All Inlinks (inkl. ALT als Fallback) ----
    # WICHTIG: nur ausführen, wenn A4 aktiv ist (wir sind im if A4_NAME in selected_analyses)
    if inlinks_df is None:
        st.info("Für Analyse 4 wird die Datei **All Inlinks** benötigt. Bitte oben im Upload-Center hochladen.")
        st.stop()

    header = [str(c).strip() for c in inlinks_df.columns]
    src_idx = find_column_index(header, POSSIBLE_SOURCE)
    dst_idx = find_column_index(header, POSSIBLE_TARGET)
    if src_idx == -1 or dst_idx == -1:
        st.error("In 'All Inlinks' wurden die Spalten 'Quelle/Source' oder 'Ziel/Destination' nicht gefunden.")
        st.stop()

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

    anchor_inv_internal = extract_anchor_inventory(inlinks_df)

    # NEU: Offpage-Ankertexte ggf. ergänzen
    include_offpage_anchors = bool(st.session_state.get("a4_include_offpage_anchors", False))
    
    # Basis: interne Anker (werden immer verwendet)
    anchor_inv_vis = anchor_inv_internal.copy()  # für Visualisierung (Treemap/Matrix/Shared)
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
                anchor_inv_vis = (
                    pd.concat([anchor_inv_internal, offpage_anchor_inv], ignore_index=True)
                      .groupby(["target", "anchor"], as_index=False)["count"]
                      .sum()
                )
    
    # Quelle-Flag („nur intern“, „nur Offpage“, „intern + Offpage“)
    if not anchor_inv_vis.empty:
        tmp = anchor_inv_vis.merge(
            anchor_inv_internal.assign(internal_count=lambda df: df["count"])[["target", "anchor", "internal_count"]],
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
    
        # zurück auf schlanke Struktur + Flag
        anchor_inv_vis = tmp[["target", "anchor", "count", "source_flag"]]
    else:
        # Fallback – Struktur konsistent halten
        anchor_inv_vis = anchor_inv_vis.assign(source_flag=pd.NA)
    
    # Für spätere Verwendung (Sidebar: Matrix, Shared, Treemap)
    st.session_state["_anchor_inv_internal"] = anchor_inv_internal
    st.session_state["_anchor_inv_vis"] = anchor_inv_vis

  
# --------------------------------------------------------
# Over-Anchor ≥ Schwellen (absolut / share)
# --------------------------------------------------------

# Sicherstellen, dass anchor_inv_internal existiert (auch wenn A4 noch nicht gelaufen ist)
anchor_inv_internal = st.session_state.get("_anchor_inv_internal", pd.DataFrame())

enable_over_anchor = st.session_state.get("a4_enable_over_anchor", True)
over_anchor_df = pd.DataFrame(columns=["Ziel-URL", "Anchor", "Count", "TopAnchorShare(%)"])

# 1) Over-Anchor-Check
if enable_over_anchor and not anchor_inv_internal.empty:
    # Gesamtanzahl Anker je Ziel-URL
    totals = anchor_inv_internal.groupby("target")["count"].sum().rename("total")
    tmp = anchor_inv_internal.merge(totals, on="target", how="left")

    # Anteil dieses Ankers an allen Ankern der URL (nur Info, kein Filter mehr)
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
            # Werte robust aus dem Session-State holen (falls A4-UI noch nicht genutzt wurde)
            brand_text = st.session_state.get("a4_brand_text", "")
            brand_file = st.session_state.get("a4_brand_file", None)

            # 1) Brand-Begriffe aus dem Freitext-Feld
            raw = str(brand_text or "")
            # Split an Komma, Zeilenumbruch, Semikolon
            tokens = re.split(r"[,\n;]+", raw)
            brand_list = [t.strip().lower() for t in tokens if t.strip()]

            # 2) Optional: Brand-Begriffe aus Datei (erste Spalte)
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

            # 3) Duplikate entfernen
            brand_list = sorted(set(brand_list))



            # -------------------------------
            # Normalisierung & Basisfilter
            # -------------------------------
            df.iloc[:, url_i] = df.iloc[:, url_i].astype(str).map(remember_original)
            df.iloc[:, q_i]   = df.iloc[:, q_i].astype(str).fillna("").str.strip()
            if c_i is not None:
                df.iloc[:, c_i] = pd.to_numeric(df.iloc[:, c_i], errors="coerce").fillna(0)
            if im_i is not None:
                df.iloc[:, im_i] = pd.to_numeric(df.iloc[:, im_i], errors="coerce").fillna(0)

            # Brand-Filter
            def brand_filter(row) -> bool:
                # Wenn keine Brandliste oder Modus "Brand + Non-Brand" → nichts filtern
                if not brand_list or brand_mode == "Brand + Non-Brand":
                    return True

                q = str(row.iloc[q_i])
                is_b = is_brand_query(q)

                if brand_mode == "Nur Non-Brand":
                    return not is_b
                elif brand_mode == "Nur Brand":
                    return is_b
                else:
                    return True

            df = df[df.apply(brand_filter, axis=1)]
    
            # Mindestschwellen
            if c_i is not None and metric_choice == "Clicks":
                df = df[df.iloc[:, c_i] >= int(min_clicks)]
            if im_i is not None and metric_choice == "Impressions":
                df = df[df.iloc[:, im_i] >= int(min_impr)]
    
            if df.empty:
                cov_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Match-Typ", "MatchBool", "Fund-Count"])
            else:
                # Top-20 % je URL (mind. 1) + Top-N-Limit
                metric_col = c_i if metric_choice == "Clicks" else im_i

                # Sortierung je nach gewählter Metric:
                # - Basis: gewählte Metrik (Clicks oder Impressions) absteigend
                # - Tie-Breaker: andere Metrik (falls vorhanden) ebenfalls absteigend
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
                    # Fallback, falls nur eine der beiden Metriken vorhanden ist
                    df = df.sort_values(
                        by=[df.columns[url_i], df.columns[metric_col]],
                        ascending=[True, False],
                    )

    
                top_rows = []
                for u, grp in df.groupby(df.columns[url_i], sort=False):
                    n = max(1, int(math.ceil(0.2 * len(grp))))
                    if int(topN_default) > 0:
                        n = max(1, min(n, int(topN_default)))
                    top_rows.append(grp.head(n))
    
                df_top = pd.concat(top_rows) if top_rows else pd.DataFrame(columns=df.columns)
    
                # Anchor-Inventar als Multiset: target -> {anchor: count}
                inv_map: Dict[str, Dict[str, int]] = {}
                for _, r in anchor_inv_internal.iterrows():
                    inv_map.setdefault(str(r["target"]), {})[str(r["anchor"])] = int(r["count"])
    
                # Embedding-Modell vorbereiten
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
                        """
                        Vereinheitlicht Text für Embeddings:
                        - None / NaN -> ""
                        - trim
                        - lowercase
                        - Mehrfach-Leerzeichen auf eins reduzieren
                        """
                        s = str(s or "").strip().lower()
                        s = re.sub(r"\s+", " ", s)
                        return s
    
                # Cache Anchor-Embeddings je Ziel-URL
                # Struktur: url -> (a_names_original, a_names_norm, a_emb_matrix)
                anchor_emb_cache: Dict[str, Tuple[List[str], List[str], Optional[np.ndarray]]] = {}

                coverage_rows = []  # eine Zeile pro URL+Query

                for u, grp in df_top.groupby(df.columns[url_i], sort=False):
                    # u = Original-URL aus GSC
                    url_raw = str(u)
                    url_norm = normalize_url(url_raw)
                    inv = inv_map.get(url_norm, {})
                
                    # Ankernamen für diese URL
                    a_names = list(inv.keys())
                    a_emb = None
                    a_names_norm: List[str] = []

                    # Embeddings je URL vorbereiten
                    if check_embed and model is not None and len(a_names) > 0:
                        if url_norm not in anchor_emb_cache:
                            try:
                                # Normalisierte Versionen für das Embedding
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
                        if check_exact:
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
                                    # Counts kommen weiterhin aus den ORIGINAL-Ankernamen
                                    cnt = int(sum(inv.get(a_names[i], 0) for i in idxs))
                                    found_cnt = max(found_cnt, cnt)
                                    match_type_parts.append("Embedding")
                            except Exception:
                                pass
                
                        match_type = "+".join(match_type_parts) if match_type_parts else "—"
                
                        coverage_rows.append([
                            disp(url_norm),  # oder disp(url_raw), beide ok
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
                # Coverage je URL berechnen
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

                # --- Tab 1: Alle Top-Queries je URL mit < 50 % Abdeckung ---
                low_cov_urls = agg[agg["Coverage"] < 0.5][["Ziel-URL"]]
                tab1 = cov_df.merge(low_cov_urls, on="Ziel-URL", how="inner")
                tab1["Als_Anker_vorhanden?"] = np.where(tab1["MatchBool"], "ja", "nein")
                gsc_tab1_df = tab1[["Ziel-URL", "Query", "Als_Anker_vorhanden?"]].copy()

                # --- Tab 2: URLs, deren Top-3-Queries alle NICHT als Anchor vorkommen ---
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
                                "nein",  # als Anker vorhanden?
                            ])
                gsc_tab2_df = pd.DataFrame(
                    rows2,
                    columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"],
                )

                # --- Tab 3: URLs, deren Top-Query NICHT als Anchor vorkommt ---
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
                            "nein",  # als Anker vorhanden?
                        ])
                gsc_tab3_df = pd.DataFrame(
                    rows3,
                    columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"],
                )

            else:
                gsc_tab1_df = pd.DataFrame(columns=["Ziel-URL", "Coverage_%", "Treffer", "Top_Queries"])
                gsc_tab2_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Match-Typ"])
                gsc_tab3_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Match-Typ"])
    else:
        cov_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Match-Typ", "MatchBool", "Fund-Count"])
        gsc_tab1_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])
        gsc_tab2_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])
        gsc_tab3_df = pd.DataFrame(columns=["Ziel-URL", "Query", "Als_Anker_vorhanden?"])


    # ============================
    # AUSGABE A4 – 1)–4)
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

    

    
# 4) Anchor-Inventar (Wide) – nur wenn in der Sidebar aktiviert
if st.session_state.get("a4_enable_anchor_matrix", True):
    anchor_inv_check = anchor_inv_vis.copy()

    if not anchor_inv_check.empty:
        inv_sorted = anchor_inv_check.sort_values(
            ["target", "count"],
            ascending=[True, False]
        ).copy()

        has_flag = (
            "source_flag" in inv_sorted.columns
            and bool(st.session_state.get("a4_include_offpage_anchors", False))
        )

        # Gesamtanzahl Anker je Ziel-URL → Anteil pro Anker berechnen
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

        # CSV-Download (Wide)
        st.download_button(
            "Download Anchor-Inventar (Wide) – CSV",
            data=anchor_wide_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="a4_anchor_inventar_wide.csv",
            mime="text/csv",
            key="a4_dl_anchor_wide_csv_main",
        )


# 4b) Anchor-Inventar (Long) – nur wenn separat aktiviert
if st.session_state.get("a4_enable_anchor_matrix_long", False):
    anchor_inv_check = anchor_inv_vis.copy()

    if not anchor_inv_check.empty:
        # TopAnchorShare(%) pro Ziel-URL + Anker berechnen
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

        # Optional: Quelle (intern / Offpage / intern + Offpage), falls vorhanden
        if "source_flag" in tmp.columns:
            long_df["Quelle"] = tmp["source_flag"]

        st.markdown("#### URL-Ankertext-Matrix (Long-Format)")
        st.dataframe(long_df, use_container_width=True, hide_index=True)

        # CSV-Export (Long)
        st.download_button(
            "Download URL-Ankertext-Matrix (Long-Format, CSV)",
            data=long_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="a4_anchor_inventar_long.csv",
            mime="text/csv",
            key="a4_dl_anchor_long_csv",
        )
    else:
        st.info("Keine Anchor-Daten für die URL-Ankertext-Matrix (Long-Format) verfügbar.")



    # 5) Shared-Ankertexte – nur wenn in der Sidebar aktiviert
    if st.session_state.get("a4_shared_enable", True):
        anchor_inv_check = anchor_inv_vis.copy()
    
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
    
            # Aggregierte Quelle pro Anchor ermitteln (nur wenn Offpage aktiv)
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
    
            # Gruppieren: ein Eintrag pro Anchor → Liste der URLs
            grouped = (
                df_shared.groupby("anchor")["target"]
                .agg(lambda s: sorted({disp(t) for t in s}))  # disp() = Original-URL-Format
                .reset_index(name="urls")
            )
            grouped["url_count"] = grouped["urls"].apply(len)
            grouped = grouped[grouped["url_count"] >= min_urls_per_anchor]
    
            if has_flag and anchor_source_map:
                grouped["Quelle"] = grouped["anchor"].map(anchor_source_map).fillna("unbekannt")
    
            if grouped.empty:
                st.info("Keine Shared-Ankertexte nach den gesetzten Filtern gefunden.")
            else:
                # 🔁 NEU: Long-Format
                # Eine Zeile = Ankertext + Ziel-URL (untereinander), optional mit Quelle + Anzahl URLs pro Anker
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
    
                # CSV
                st.download_button(
                    "Download Shared-Ankertexte (CSV)",
                     data=shared_long_df.to_csv(index=False).encode("utf-8-sig"),
                     file_name="a4_shared_ankertexte_long.csv",
                     mime="text/csv",
                     key="a4_dl_shared_csv_main",
                )
    
                # XLSX
                try:
                    buf_shared = io.BytesIO()
                    with pd.ExcelWriter(buf_shared, engine="xlsxwriter") as xw:
                        shared_long_df.to_excel(xw, index=False, sheet_name="Shared-Ankertexte")
                        ws = xw.sheets["Shared-Ankertexte"]
                        for col_idx, col_name in enumerate(shared_long_df.columns, start=1):
                            max_len_col = max(
                                len(str(col_name)),
                                *(
                                    len(str(v))
                                    for v in shared_long_df.iloc[:, col_idx - 1].astype(str).values[:1000]
                                ),
                            )
                            ws.set_column(col_idx - 1, col_idx - 1, min(max_len_col + 2, 60))
                    buf_shared.seek(0)
                    st.download_button(
                        "Download Shared-Ankertexte (XLSX)",
                        data=buf_shared.getvalue(),
                        file_name="a4_shared_ankertexte_long.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="a4_dl_shared_xlsx_main",
                    )
                except Exception as e:
                    st.warning(f"XLSX-Export (Shared-Ankertexte) nicht möglich: {e}")



    # ============================
    # Optional: Treemap-Visualisierung (Hauptbereich)
    # ============================
    show_treemap = bool(st.session_state.get("a4_show_treemap", True))
    
    # Anchor-Inventar aus Session holen (kommt aus A4-Berechnung)
    anchor_inv_check = st.session_state.get("_anchor_inv_vis", pd.DataFrame())
    
    if show_treemap and _HAS_PLOTLY and not anchor_inv_check.empty:
        st.markdown("#### Treemap der häufigsten Anchors je Ziel-URL")
    
        treemap_topK = int(st.session_state.get("a4_treemap_topk", 12))
        treemap_url_mode = st.session_state.get("a4_treemap_url_mode", "Alle URLs")
        selected_urls = st.session_state.get("a4_treemap_selected_urls") or []
    
        all_targets = sorted(anchor_inv_check["target"].astype(str).unique())
    
        # Bestimme, welche URLs final verwendet werden
        if treemap_url_mode == "Ausgewählte URLs" and selected_urls:
            targets = [u for u in selected_urls if u in all_targets]
        else:
            targets = all_targets
    
        if not targets:
            st.info("Keine passenden Ziel-URLs für die Treemap-Auswertung gefunden.")
        else:
            # Vorschau: eine URL auswählen
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

                # einfache Struktur: nur Anchor + Count für diese URL
                df_t = grp[["anchor", "count"]].rename(columns={"anchor": "Anchor", "count": "Count"})

                fig = px.treemap(
                    df_t,
                    path=["Anchor"],
                    values="Count",
                    title=f"Treemap: häufigste Anchors für {disp(target)}"
                )

                # Text in den Kästchen zentrieren und etwas größer machen
                fig.update_traces(
                    textposition="middle center",
                    textinfo="label+value",
                    textfont_size=16,
                )

                # HTML für Download
                html_bytes = fig.to_html(full_html=True, include_plotlyjs="cdn").encode("utf-8")

                return fig, html_bytes

    
            fig, html_bytes = build_treemap_for_target(preview_url)
    
            if fig is None:
                st.info("Für die ausgewählte URL liegen keine Ankerdaten vor.")
            else:
                # Vorschau im Hauptbereich
                st.plotly_chart(fig, use_container_width=True)
    
                # Dateiname etwas entschärfen
                safe_name = re.sub(r"[^A-Za-z0-9]+", "_", str(preview_url))[:60] or "url"
                st.download_button(
                    "Treemap für ausgewählte URL herunterladen (HTML)",
                    data=html_bytes,
                    file_name=f"treemap_anchors_{safe_name}.html",
                    mime="text/html",
                    key="a4_dl_treemap_single"
                )
    
            # Basisdaten für alle ausgewählten URLs als CSV-Download
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


