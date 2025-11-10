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

# =====================================================================
# NEU: Analyse-Auswahl im Hauptbereich + Zentrales Upload-Center
# =====================================================================
A1_NAME = "Interne Verlinkungsmöglichkeiten finden"
A2_NAME = "Unpassende interne Links entfernen"
A3_NAME = "SEO-Potenziallinks finden"
A4_NAME = "Ankertexte analysieren"

st.markdown("---")
st.header("Welche Analysen möchtest du durchführen?")
selected_analyses = st.multiselect(
    "Mehrfachauswahl möglich",
    options=[A1_NAME, A2_NAME, A3_NAME, A4_NAME],
    default=[],
)

with st.sidebar:
    # Sidebar nur zeigen, wenn mind. eine Analyse gewählt ist
    if selected_analyses:
        st.header("Einstellungen")

        # Gemeinsame Settings nur für A1/A2/A3
        if any(a in selected_analyses for a in [A1_NAME, A2_NAME, A3_NAME]):
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

            # WICHTIG: "Entfernung von Links" NUR WENN A2 gewählt
            if A2_NAME in selected_analyses:
                st.subheader("Entfernung von Links (nur für Analyse 2)")
                not_similar_threshold = st.slider("Unähnlichkeitsschwelle (schwache Links)", 0.0, 1.0, 0.60, 0.01, key="a2_not_sim")
                backlink_weight_2x = st.checkbox("Backlinks/Ref. Domains doppelt gewichten", value=False, key="a2_weight2x")
            else:
                # Defaults ablegen, falls A2 nicht aktiv ist (werden unten bei A2 gelesen)
                st.session_state.setdefault("a2_not_sim", 0.60)
                st.session_state.setdefault("a2_weight2x", False)

        # A4-Settings in die Sidebar verschoben
        if A4_NAME in selected_analyses:
            st.subheader("Analyse 4 – Einstellungen")
            st.caption("Brand-/Matching-Optionen, Schwellen & Visualisierung")

            brand_text = st.text_area("Brand-Schreibweisen (eine pro Zeile oder komma-getrennt)", value="", key="a4_brand_text")
            brand_file = st.file_uploader("Optional: Brand-Liste (1 Spalte)", type=["csv","xlsx","xlsm","xls"], key="a4_brand_file")
            auto_variants = st.checkbox("Automatisch Varianten erzeugen (z. B. „bora kochfeld“, „bora-kochfeld“)", value=True, key="a4_auto_variants")
            head_nouns_text = st.text_input("Head-Nomen (kommagetrennt, editierbar)", value="kochfeld, kochfeldabzug, system, kochfelder", key="a4_head_nouns")
            brand_mode = st.radio("Welche Queries berücksichtigen?", ["Nur Non-Brand", "Nur Brand", "Beides"], index=0, horizontal=True, key="a4_brand_mode")

            st.markdown("**Matching**")
            metric_choice = st.radio("GSC-Bewertung nach …", ["Impressions", "Clicks"], index=0, horizontal=True, key="a4_metric_choice")
            check_exact = st.checkbox("Exact Match prüfen", value=True, key="a4_check_exact")
            check_embed = st.checkbox("Embedding Match prüfen", value=True, key="a4_check_embed")

            embed_model_name = st.selectbox(
                "Embedding-Modell",
                ["sentence-transformers/all-MiniLM-L6-v2",
                 "sentence-transformers/all-MiniLM-L12-v2",
                 "sentence-transformers/all-mpnet-base-v2"],
                index=0,
                help="Standard: all-MiniLM-L6-v2",
                key="a4_embed_model",
            )
            embed_thresh = st.slider("Cosine-Schwelle (Embedding)", 0.50, 0.95, 0.75, 0.01, key="a4_embed_thresh")

            st.markdown("**Schwellen & Filter**")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                min_clicks = st.number_input("Mindest-Klicks/Query", min_value=0, value=50, step=10, key="a4_min_clicks")
            with col_s2:
                min_impr   = st.number_input("Mindest-Impressions/Query", min_value=0, value=500, step=50, key="a4_min_impr")
            with col_s3:
                topN_default = st.number_input("Top-N Queries pro URL (zusätzliche Bedingung)", min_value=1, value=10, step=1, key="a4_topN")

            col_o1, col_o2 = st.columns(2)
            with col_o1:
                top_anchor_abs = st.number_input("Schwelle identischer Anker (absolut)", min_value=1, value=200, step=10, key="a4_top_anchor_abs")
            with col_o2:
                top_anchor_share = st.slider("Schwelle TopAnchorShare (%)", 0, 100, 60, 1, key="a4_top_anchor_share")

            st.markdown("**Visualisierung**")
            show_treemap = st.checkbox("Treemap-Visualisierung aktivieren", value=True, key="a4_show_treemap")
            treemap_topK = st.number_input("Treemap: Top-K Anchors anzeigen", min_value=3, max_value=50, value=12, step=1, key="a4_treemap_topk")

    else:
        # Optional: ganz leere Sidebar oder ein kleiner Hinweis
        st.caption("Wähle oben mindestens eine Analyse aus, um Einstellungen zu sehen.")

# Benötigte Inputs je Analyse
needs_embeddings_or_related = any(a in selected_analyses for a in [A1_NAME, A2_NAME, A3_NAME])
needs_inlinks               = any(a in selected_analyses for a in [A1_NAME, A2_NAME, A4_NAME])
needs_metrics               = any(a in selected_analyses for a in [A1_NAME, A2_NAME, A3_NAME])
needs_backlinks             = any(a in selected_analyses for a in [A1_NAME, A2_NAME, A3_NAME])
needs_gsc                   = any(a in selected_analyses for a in [A3_NAME])  # optional

# Eingabemodus nur zeigen, wenn 1/2/3 aktiv
mode = "Related URLs"
if needs_embeddings_or_related:
    st.subheader("Eingabemodus (für Analysen 1–3)")
    mode = st.radio(
        "Bitte wählen:",
        ["URLs + Embeddings", "Related URLs"],
        horizontal=True,
        help="Bei Embeddings berechnet das Tool die 'Related URLs' selbst. Oder fertige 'Related URLs' hochladen.",
    )

# Zentrales Upload-Center
st.markdown("---")
st.subheader("Benötigte Dateien hochladen")

emb_df = related_df = inlinks_df = metrics_df = backlinks_df = None
gsc_df_loaded = None

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

col_left, col_right = st.columns(2)

# A) Embeddings oder Related
if needs_embeddings_or_related:
    if mode == "URLs + Embeddings":
        up_emb = st.file_uploader(
            "URLs + Embeddings (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="up_emb_global",
            help="Mindestens: URL + Embedding-Spalte (JSON-Array ODER Zahlen, getrennt durch Komma/Whitespace/; / |).",
        )
        emb_df = _read_up("URLs + Embeddings", up_emb, required=True)
    else:
        up_related = st.file_uploader(
            "Related URLs (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="up_related_global",
            help="Genau 3 Spalten: Ziel-URL, Quell-URL, Similarity (0–1).",
        )
        related_df = _read_up("Related URLs", up_related, required=True)

# B) All Inlinks
if needs_inlinks:
    with col_left:
        up_inlinks = st.file_uploader(
            "All Inlinks (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="up_inlinks_global",
            help="Screaming Frog → Massenexport → Links → Alle Inlinks. (Optional: Anchor Text-Spalte)",
        )
        inlinks_df = _read_up("All Inlinks", up_inlinks, required=True)

# C) Linkmetriken
if needs_metrics:
    with col_right:
        up_metrics = st.file_uploader(
            "Linkmetriken (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="up_metrics_global",
            help="Erste 4 Spalten: URL, Score (Interner Link Score), Inlinks, Outlinks.",
        )
        metrics_df = _read_up("Linkmetriken", up_metrics, required=True)

# D) Backlinks
if needs_backlinks:
    with col_left:
        up_backlinks = st.file_uploader(
            "Backlinks (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="up_backlinks_global",
            help="Erste 3 Spalten: URL, Backlinks, Referring Domains.",
        )
        backlinks_df = _read_up("Backlinks", up_backlinks, required=True)

# E) Search Console (optional für A3)
if needs_gsc:
    with col_right:
        up_gsc = st.file_uploader(
            "Search Console Daten (optional, CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="up_gsc_global",
            help="Mindestens: URL, Impressions · Optional: Clicks, Position.",
        )
        if up_gsc is not None:
            gsc_df_loaded = _read_up("Search Console", up_gsc, required=False)

# Getrennte Start-Buttons für A1 & A2
st.markdown("---")
start_cols = st.columns(4)
run_clicked_a1 = run_clicked_a2 = False

if A1_NAME in selected_analyses:
    with start_cols[0]:
        run_clicked_a1 = st.button("Let's Go (Analyse 1)", type="secondary", key="btn_a1")

if A2_NAME in selected_analyses:
    with start_cols[1]:
        run_clicked_a2 = st.button("Let's Go (Analyse 2)", type="secondary", key="btn_a2")

# A3 eigener Button kommt später im A3-Block
# A4 eigener Button kommt im A4-Block

# Kompatibilität: gemeinsame Vorverarbeitung triggern, wenn einer der beiden startet
run_clicked = bool(run_clicked_a1 or run_clicked_a2)

# Merker für Sichtbarkeit
if run_clicked_a1:
    st.session_state["__show_a1__"] = True
if run_clicked_a2:
    st.session_state["__show_a2__"] = True

# Spinner beim Start von A1/A2
if run_clicked:
    placeholder = st.empty()
    with placeholder.container():
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.image("https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDJweGExcHhhOWZneTZwcnAxZ211OWJienY5cWQ1YmpwaHR0MzlydiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/dBRaPog8yxFWU/giphy.gif", width=280)
            st.caption("Die Berechnungen laufen – Zeit für eine kleine Stärkung …")

# Gate nur anwenden, wenn A1 oder A2 überhaupt gewählt wurden
if (A1_NAME in selected_analyses or A2_NAME in selected_analyses) and (not run_clicked) and (not st.session_state.get("ready", False)):
    st.info("Bitte Dateien für die gewählten Analysen hochladen und auf **Let's Go** klicken.")
    st.stop()

# =====================================================================
# Vorverarbeitung & Validierung NUR für A1/A2 (und ggf. Embeddings/Related)
# =====================================================================
if (A1_NAME in selected_analyses or A2_NAME in selected_analyses) and (run_clicked or st.session_state.ready):

    # Validierung & ggf. Related aus Embeddings bauen
    if needs_embeddings_or_related and mode == "URLs + Embeddings":
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

    # Anchor/ALT Spalten (für Analyse 4 – werden unten erneut verwendet)
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
    # Analyse 2 – nur rendern, wenn Button gedrückt
    # ===============================
    if st.session_state.get("__show_a2__", False):
        st.markdown("## Analyse 2: Potenziell zu entfernende Links")
        st.caption("Diese Analyse legt bestehende Links zwischen semantisch nicht stark verwandten URLs offen.")

        # A2-Settings aus Sidebar-State lesen (da nur dort gerendert)
        not_similar_threshold = float(st.session_state.get("a2_not_sim", 0.60))
        backlink_weight_2x = bool(st.session_state.get("a2_weight2x", False))

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
                (src, dst) for (src, dst) in st.session_state["_all_links"]
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
# Analyse 3 (nur anzeigen, wenn ausgewählt)
# =========================================================
if A3_NAME in selected_analyses:

    st.markdown("---")
    st.subheader("Analyse 3: Was sind starke Linkgeber („Gems“) & welche URLs diese verlinken sollten (⇒ SEO-Potenziallinks)")
    st.caption("Diese Analyse identifiziert die aus SEO-Gesichtspunkten wertvollsten, aber noch nicht gesetzten, Content-Links.")

    # --- Steuer-UI (im Hauptbereich) ---
    gem_pct = st.slider("Anteil starker Linkgeber (Top-X %)", 1, 30, 10, step=1, key="a3_gem_pct")
    max_targets_per_gem = st.number_input("Top-Ziele je Gem", min_value=1, max_value=50, value=10, step=1, key="a3_max_targets")

    # GSC Upload (optional für A3)
    gsc_up = st.file_uploader("Search Console Daten (CSV/Excel)", type=["csv","xlsx","xlsm","xls"], key="a3_gsc_up")

    st.markdown("#### Linkbedarf-Gewichtung für Zielseiten")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with bordered_container():
            st.markdown("Gewicht: **Hidden Champions**")
            w_lihd = st.slider("", 0.0, 1.0, 0.30, 0.05, disabled=(gsc_up is None), key="a3_w_lihd")
    with col2:
        with bordered_container():
            st.markdown("Gewicht: **Semantische Linklücke**")
            w_def  = st.slider("", 0.0, 1.0, 0.30, 0.05, key="a3_w_def")
    with col3:
        with bordered_container():
            st.markdown("Gewicht: **Sprungbrett-URLs**")
            w_rank = st.slider("", 0.0, 1.0, 0.30, 0.05, key="a3_w_rank")
            st.caption("Sprungbrett-URLs – Feineinstellung")
            rank_minmax = st.slider("Ranking Sprungbrett-URL (Positionsbereich)", 1, 50, (8, 20), 1, key="a3_rank_minmax")
    with col4:
        with bordered_container():
            st.markdown("Gewicht: **Mauerblümchen**")
            w_orph = st.slider("", 0.0, 1.0, 0.10, 0.05, key="a3_w_orph")
            st.caption("Mauerblümchen – Feineinstellung")
            thin_k = st.slider("Thin-Schwelle (Inlinks ≤ K)", 0, 10, 2, 1, key="a3_thin_k")

    with st.expander("Offpage-Einfluss (Backlinks & Ref. Domains)", expanded=False):
        offpage_damp_enabled = st.checkbox("Offpage-Dämpfung anwenden", value=True, key="a3_offpage_enable")
        beta_offpage = st.slider(
            "Stärke der Dämpfung", 0.0, 1.0, 0.5, 0.05,
            disabled=not st.session_state.get("a3_offpage_enable", False),
            key="a3_offpage_beta"
        )

    with st.expander("Reihenfolge der Empfehlungen - *OPTIONAL*", expanded=False):
        sort_labels = {"rank_mix":"Mix (Nähe & Linkbedarf kombiniert)","prio_only":"Nur Linkbedarf","sim_only":"Nur inhaltliche Nähe"}
        sort_choice = st.radio("Sortierung", options=["rank_mix","prio_only","sim_only"],
                               index=0, format_func=lambda k: sort_labels.get(k, k), horizontal=True, key="a3_sort_choice")
        alpha_mix = st.slider("Gewichtung: inhaltliche Nähe vs. Linkbedarf", 0.0, 1.0, 0.5, 0.05, key="a3_alpha_mix")

    # A3 starten
    run_gems = st.button("Let's Go (Analyse 3)", type="secondary", key="btn_a3")
    if run_gems:
        st.session_state["__gems_loading__"] = True
        st.session_state["__ready_gems__"] = False
        st.rerun()

    # Loader
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

    # ====== Analyse 3: Berechnung & Ausgabe ======
    # Datenabhängigkeiten prüfen
    source_potential_map = st.session_state.get("_source_potential_map")
    related_map = st.session_state.get("_related_map")
    content_links = st.session_state.get("_content_links")
    all_links = st.session_state.get("_all_links")

    if not (isinstance(source_potential_map, dict) and isinstance(related_map, dict) and isinstance(content_links, set) and isinstance(all_links, set)):
        st.info("Für Analyse 3 werden die vorbereiteten Daten benötigt (Linkmetriken/Backlinks/Inlinks & Related). Bitte zuvor Analyse 1 oder 2 einmal laufen lassen.")
    else:
        # GSC (optional) laden und aggregieren
        gsc_df_a3 = None
        if gsc_up is not None:
            gsc_df_a3 = read_any_file_cached(getattr(gsc_up, "name", ""), gsc_up.getvalue())
            # Für A4 bereitstellen
            st.session_state["__gsc_df_raw__"] = gsc_df_a3

        def _norm_header_local(s: str) -> str:
            return _norm_header(s)

        if isinstance(gsc_df_a3, pd.DataFrame) and not gsc_df_a3.empty:
            df = gsc_df_a3.copy()
            df.columns = [str(c).strip() for c in df.columns]
            hdr = [_norm_header_local(c) for c in df.columns]

            def _fidx(names, default=None):
                names = {_norm_header_local(x) for x in names}
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
                df.iloc[:, url_idx] = df.iloc[:, url_idx].astype(str).map(normalize_url)
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
                if (gem, target) in content_links:
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
                in_c = inlink_count_map.get(target, 0.0)
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
                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
                    # Gesamt
                    rec_df[view_cols + ["Ähnlichkeit (norm)","SortScore"]].to_excel(xw, index=False, sheet_name="Gesamt")
                    # Pro Gem
                    for gem_name, grp in rec_df.groupby("Gem", sort=False):
                        sheet = re.sub(r"[^A-Za-z0-9]+", "_", gem_name)[:31] or "Gem"
                        grp[view_cols + ["Ähnlichkeit (norm)","SortScore"]].to_excel(xw, index=False, sheet_name=sheet)
                bio.seek(0)
                st.download_button("Download Empfehlungen (XLSX)", data=bio.getvalue(),
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

    st.markdown("---")
    st.subheader("🔎 Analyse 4: Anchor & Query Intelligence (Embeddings)")
    st.caption("Verknüpft Suchanfragen, Ankertexte und Zielseiten via Embeddings. Findet Over-Anchors, fehlende Query-Anchors, Leader-Konflikte und nicht verlinkte Ziel-Keywords.")

    # -----------------------------
    # Uploads (Hauptbereich) – Settings kommen aus Sidebar
    # -----------------------------
    st.markdown("#### Uploads")
    gsc_up_a4 = st.file_uploader("Search Console: URL | Query | Clicks | Impressions", type=["csv","xlsx","xlsm","xls"], key="a4_gsc_up_main")
    kwmap_up  = st.file_uploader("Keyword-Zielvorgaben: URL + Keyword-Spalten", type=["csv","xlsx","xlsm","xls"], key="a4_kwmap_up")

    st.divider()

    # Start-Button (manuelle Berechnung starten)
    run_a4 = st.button("Let's Go (Analyse 4)", type="secondary", key="btn_a4")
    if run_a4:
        st.session_state["__a4_loading__"] = True
        st.session_state["__ready_a4__"] = False
        st.rerun()

    # Loader-Anzeige (wenn Analyse 4 läuft)
    if st.session_state.get("__a4_loading__", False):
        ph4 = st.session_state.get("__a4_ph__")
        if ph4 is None:
            ph4 = st.empty()
            st.session_state["__a4_ph__"] = ph4
        with ph4.container():
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                try:
                    st.image("https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExczFvdm16cHZsd3J2NGZ2eDVvOWE1b3k2OXJpajZodDliamxjdzVybCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WS6C15O7L8Oqk/giphy.gif", width=280)
                except Exception:
                    st.write("⏳")
                st.caption("Analyse 4 läuft … gleich geht’s los!")

    # -------------------------------------------------
    # Ab hier: komplette A4-Auswertung
    # -------------------------------------------------

    # Sidebar-Settings holen
    brand_text = st.session_state.get("a4_brand_text", "")
    brand_file = st.session_state.get("a4_brand_file", None)
    auto_variants = st.session_state.get("a4_auto_variants", True)
    head_nouns_text = st.session_state.get("a4_head_nouns", "kochfeld, kochfeldabzug, system, kochfelder")
    brand_mode = st.session_state.get("a4_brand_mode", "Nur Non-Brand")

    metric_choice = st.session_state.get("a4_metric_choice", "Impressions")
    check_exact = bool(st.session_state.get("a4_check_exact", True))
    check_embed = bool(st.session_state.get("a4_check_embed", True))
    embed_model_name = st.session_state.get("a4_embed_model", "sentence-transformers/all-MiniLM-L6-v2")
    embed_thresh = float(st.session_state.get("a4_embed_thresh", 0.75))

    min_clicks = int(st.session_state.get("a4_min_clicks", 50))
    min_impr = int(st.session_state.get("a4_min_impr", 500))
    topN_default = int(st.session_state.get("a4_topN", 10))
    top_anchor_abs = int(st.session_state.get("a4_top_anchor_abs", 200))
    top_anchor_share = int(st.session_state.get("a4_top_anchor_share", 60))

    show_treemap = bool(st.session_state.get("a4_show_treemap", True))
    treemap_topK = int(st.session_state.get("a4_treemap_topk", 12))

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

    def read_single_col_file_obj(up_obj) -> List[str]:
        if up_obj is None:
            return []
        try:
            df = read_any_file_cached(getattr(up_obj, "name", ""), up_obj.getvalue())
        except Exception:
            return []
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
    brand_list += read_single_col_file_obj(brand_file)
    brand_list = sorted({b.strip() for b in brand_list if b.strip()})
    head_nouns = [x.strip() for x in head_nouns_text.split(",") if x.strip()]
    brand_all_terms = make_brand_variants(brand_list, head_nouns, auto_variants)

    def is_brand_query(q: str) -> bool:
        s = (q or "").lower().strip()
        if not s:
            return False
        tokens = [re.escape(t.lower()) for t in brand_all_terms]
        if not tokens:
            return False
        pat = r"(?:^|[^a-z0-9äöüß])(" + "|".join(tokens) + r")(?:$|[^a-z0-9äöüß])"
        return re.search(pat, s, flags=re.IGNORECASE) is not None

    # Navigative/generische Anchors ausschließen (für Konflikte)
    def is_navigational(anchor: str) -> bool:
        return (anchor or "").strip().lower() in NAVIGATIONAL_ANCHORS

    # ---- Anchor-Inventar aus All Inlinks (inkl. ALT als Fallback) ----
    if 'inlinks_df' not in locals() or inlinks_df is None:
        st.error("Für Analyse 4 wird die Datei **All Inlinks** benötigt.")
        st.stop()

    header = [str(c).strip() for c in inlinks_df.columns]
    src_idx = find_column_index(header, POSSIBLE_SOURCE)
    dst_idx = find_column_index(header, POSSIBLE_TARGET)
    if src_idx == -1 or dst_idx == -1:
        st.error("In 'All Inlinks' wurden die Spalten 'Quelle/Source' oder 'Ziel/Destination' nicht gefunden.")
        st.stop()

    def extract_anchor_inventory(df: pd.DataFrame) -> pd.DataFrame:
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

    # ---- Over-Anchor ≥ Schwellen (absolut / share) ----
    over_anchor_df = pd.DataFrame(columns=["Ziel-URL","Anchor","Count","TopAnchorShare(%)"])
    if not anchor_inv.empty:
        totals = anchor_inv.groupby("target")["count"].sum().rename("total")
        tmp = anchor_inv.merge(totals, on="target", how="left")
        tmp["share"] = (100.0 * tmp["count"] / tmp["total"]).round(2)
        filt = (tmp["count"] >= int(top_anchor_abs)) | (tmp["share"] >= float(top_anchor_share))
        over_anchor_df = tmp.loc[filt, ["target","anchor","count","share"]].copy()
        over_anchor_df.columns = ["Ziel-URL","Anchor","Count","TopAnchorShare(%)"]

    # ---- GSC laden (aus Upload oder ggf. von Analyse 3) ----
    if gsc_up_a4 is not None:
        gsc_df = read_any_file_cached(getattr(gsc_up_a4, "name", ""), gsc_up_a4.getvalue())
        st.session_state["__gsc_df_raw__"] = gsc_df
    else:
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

            # Brand-Filter
            def brand_filter(row) -> bool:
                q = str(row.iloc[q_i])
                is_b = is_brand_query(q)
                if brand_mode == "Nur Non-Brand":
                    return (not is_b)
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

            # Top-20% je URL (mind. 1) und Top-N-Grenze
            metric_col = c_i if metric_choice == "Clicks" else im_i
            df = df.sort_values(by=[df.columns[url_i], df.columns[metric_col]], ascending=[True, False])
            top_rows = []
            for u, grp in df.groupby(df.columns[url_i], sort=False):
                n = max(1, int(math.ceil(0.2 * len(grp))))
                n = max(1, min(n, int(topN_default)))
                top_rows.append(grp.head(n))
            df_top = pd.concat(top_rows) if top_rows else pd.DataFrame(columns=df.columns)

            # Anchor-Inventar als Multiset: target -> {anchor: count}
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
                A = A.astype(np.float32, copy=False)
                B = B.astype(np.float32, copy=False)
                A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
                B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
                return A @ B.T

            # Cache Anchor-Embeddings je target
            anchor_emb_cache: Dict[str, Tuple[List[str], Optional[np.ndarray]]] = {}
            if check_embed and model is not None:
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

            # 1) Coverage der Top-Queries als Anchors
            issues_rows = []
            for u, grp in df_top.groupby(df_top.columns[url_i], sort=False):
                inv = inv_map.get(u, {})
                a_names = list(inv.keys())
                a_emb = None
                if check_embed and model is not None and (u in anchor_emb_cache):
                    a_names, a_emb = anchor_emb_cache.get(u, (a_names, None))

                for _, rr in grp.iterrows():
                    q = str(rr.iloc[q_i]).strip()
                    if not q:
                        continue
                    found = False
                    found_cnt = 0
                    match_type = []
                    # Exact
                    if check_exact:
                        cnt = sum(inv.get(a, 0) for a in a_names if a.lower() == q.lower())
                        if cnt > 0:
                            found = True
                            found_cnt = max(found_cnt, cnt)
                            match_type.append("Exact")
                    # Embedding
                    if (not found or check_embed) and check_embed and model is not None and a_emb is not None and len(a_names) > 0:
                        try:
                            q_emb = model.encode([q], show_progress_bar=False)
                            S = cosine_sim_matrix(np.asarray(q_emb), a_emb)[0]
                            idxs = np.where(S >= float(embed_thresh))[0]
                            if idxs.size > 0:
                                found = True
                                cnt = int(sum(inv.get(a_names[i], 0) for i in idxs))
                                found_cnt = max(found_cnt, cnt)
                                match_type.append("Embedding")
                        except Exception:
                            pass
                    if not found:
                        issues_rows.append([disp(u), q, "+".join(match_type) if match_type else "—", "nein", 0, "Top-Query kommt nicht als Anchor vor"])
            if issues_rows:
                gsc_issues_df = pd.DataFrame(issues_rows, columns=["Ziel-URL","Query","Match-Typ","Anker gefunden?","Fund-Count","Hinweis"])

            # 2) Leader-Konflikte
            if metric_choice == "Clicks" and c_i is not None:
                lead_ser = df.groupby(df.columns[q_i]).apply(lambda x: x.loc[x.iloc[:, c_i].idxmax(), df.columns[url_i]])
                lead_val = df.groupby(df.columns[q_i]).apply(lambda x: float(x.iloc[:, c_i].max()))
            else:
                lead_ser = df.groupby(df.columns[q_i]).apply(lambda x: x.loc[x.iloc[:, im_i].idxmax(), df.columns[url_i]])
                lead_val = df.groupby(df.columns[q_i]).apply(lambda x: float(x.iloc[:, im_i].max()))
            leader_map = {q: normalize_url(u) for q, u in lead_ser.to_dict().items() if isinstance(u, str)}
            leader_val_map = {q: float(v) for q, v in lead_val.to_dict().items()}

            leader_rows = []
            if check_exact or check_embed:
                # Precompute Query-Embeddings einmal
                q_emb_cache: Dict[str, np.ndarray] = {}
                if check_embed and model is not None:
                    try:
                        all_qs = list(leader_map.keys())
                        if all_qs:
                            E = model.encode(all_qs, batch_size=64, show_progress_bar=False)
                            for q, e in zip(all_qs, E):
                                q_emb_cache[q] = np.asarray(e)
                    except Exception:
                        pass

                for q, lead_u in leader_map.items():
                    # Kandidaten-Ziele mit passenden Anchors
                    exact_targets = set()
                    if check_exact:
                        sub = anchor_inv[anchor_inv["anchor"].str.lower() == q.lower()]
                        exact_targets.update(normalize_url(t) for t in sub["target"].tolist())

                    embed_targets = set()
                    if check_embed and model is not None and q in q_emb_cache:
                        qv = q_emb_cache[q][None, :]
                        for tgt, (a_names, a_emb) in anchor_emb_cache.items():
                            if a_emb is None or len(a_names) == 0:
                                continue
                            S = cosine_sim_matrix(qv, a_emb)[0]
                            if (S >= float(embed_thresh)).any():
                                embed_targets.add(tgt)

                    for tgt in exact_targets.union(embed_targets):
                        if not lead_u or not tgt or tgt == lead_u:
                            continue
                        # navigative-only Ausschluss
                        anchors_for_tgt = anchor_inv[anchor_inv["target"] == tgt]["anchor"].astype(str).tolist()
                        nav_only = True
                        if check_exact and (q.lower() in [a.lower() for a in anchors_for_tgt]):
                            if not all(is_navigational(a) for a in anchors_for_tgt if a.lower() == q.lower()):
                                nav_only = False
                        if check_embed and model is not None and q in q_emb_cache:
                            try:
                                for a in anchors_for_tgt:
                                    a_emb = model.encode([a], show_progress_bar=False)
                                    s = float(cosine_sim_matrix(q_emb_cache[q][None, :], np.asarray(a_emb))[0,0])
                                    if s >= float(embed_thresh) and not is_navigational(a):
                                        nav_only = False
                                        break
                            except Exception:
                                pass
                        if not nav_only:
                            leader_rows.append([q, disp(tgt), disp(lead_u), int(leader_val_map.get(q, 0)), "Anchor navigativ? (ausgeschlossen): nein"])

            if leader_rows:
                leader_conflicts_df = pd.DataFrame(leader_rows, columns=["Query","Verlinkte URL (aktueller Link)","Leader-URL","Leader-Wert","Hinweis (navigativ ausgeschlossen?)"])

    # ---- Keyword-Zielvorgaben prüfen (Exact &/oder Embedding) ----
    kw_missing_df = pd.DataFrame(columns=["Ziel-URL","Keyword","Hinweis","Match-Typ"])
    if kwmap_up is not None:
        kw_df = read_any_file_cached(getattr(kwmap_up, "name", ""), kwmap_up.getvalue())
        if isinstance(kw_df, pd.DataFrame) and not kw_df.empty:
            dff = kw_df.copy()
            dff.columns = [str(c).strip() for c in dff.columns]
            url_i = find_column_index(dff.columns.tolist(), ["url","urls","page","seite","address","adresse","ziel url","ziel-url"])
            if url_i == -1:
                url_i = 0
            kw_cols = [c for i, c in enumerate(dff.columns) if i != url_i and re.search(r"(keyword|suchanfrage|suchbegriff|query)", _norm_header(c))]
            if not kw_cols:
                kw_cols = [c for i, c in enumerate(dff.columns) if i != url_i]

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
                            S = (q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True)+1e-12)) @ (a_emb_u / (np.linalg.norm(a_emb_u, axis=1, keepdims=True)+1e-12)).T
                            S = np.asarray(S)[0]
                            if (S >= float(embed_thresh)).any():
                                found = True; match_t.append("Embedding")
                        except Exception:
                            pass
                    if not found:
                        kw_missing_df.loc[len(kw_missing_df)] = [disp(url), kw, "Ziel-Keyword nicht als Anchor verlinkt", "+".join(match_t) if match_t else "—"]

    # ---- Treemap-Visualisierung (optional) ----
    try:
        import plotly.express as px  # falls oben fehlte
        _HAS_PLOTLY = True
    except Exception:
        pass

    if show_treemap and _HAS_PLOTLY and not anchor_inv.empty:
        st.markdown("### Treemap: Ankerverteilung pro Ziel-URL")
        targets_sorted = sorted(anchor_inv["target"].unique())
        sel = st.selectbox("Ziel-URL wählen", [disp(t) for t in targets_sorted], index=0, key="a4_treemap_sel")
        t_norm = None
        for k, v in st.session_state["_ORIG_MAP"].items():
            if v == sel:
                t_norm = k; break
        if t_norm is None:
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
        try:
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
                for name, df in dfs.items():
                    df_use = df.copy()
                    if df_use.empty:
                        df_use = df_use.reindex(columns=["—"])
                    df_use.to_excel(xw, index=False, sheet_name=name[:31])
            bio.seek(0)
            return bio.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "analyse4_anchor_query_intelligence.xlsx"
        except Exception:
            pass
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

    st.session_state["__a4_loading__"] = False
    st.session_state["__ready_a4__"] = True
    try:
        ph4 = st.session_state.get("__a4_ph__")
        if ph4 is not None:
            ph4.empty()
    except Exception:
        pass

