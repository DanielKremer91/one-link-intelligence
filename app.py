import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ===============================
# Helpers
# ===============================

POSSIBLE_SOURCE = ["quelle", "source", "from", "origin"]
POSSIBLE_TARGET = ["ziel", "destination", "to", "target"]
POSSIBLE_POSITION = ["linkposition", "link position", "position"]


def find_column_index(header: List[str], possible_names: List[str]) -> int:
    lower = [str(h).strip().lower() for h in header]
    for i, h in enumerate(lower):
        if h in possible_names:
            return i
    return -1


def normalize_url(u: str, remove_www: bool = False) -> str:
    """URL-Kanonisierung: Protokoll ergänzen, Tracking-Parameter entfernen, Query sortieren, Trailing Slash normalisieren."""
    try:
        s = str(u or "").strip()
        if not s:
            return ""
        if not s.lower().startswith(("http://", "https://")):
            s = "https://" + s
        from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

        p = urlparse(s)
        fragment = ""  # Hash droppen

        # Tracking-Parameter entfernen
        qs = [
            (k, v)
            for (k, v) in parse_qsl(p.query, keep_blank_values=True)
            if k.lower()
            not in {
                "utm_source",
                "utm_medium",
                "utm_campaign",
                "utm_term",
                "utm_content",
                "gclid",
                "fbclid",
                "mc_cid",
                "mc_eid",
                "pk_campaign",
                "pk_kwd",
            }
        ]
        qs.sort(key=lambda kv: kv[0])
        query = urlencode(qs)

        hostname = (p.hostname or "").lower()
        if remove_www and hostname.startswith("www."):
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
        token in pos_norm for token in ["inhalt", "content", "body", "main", "artikel", "article"]
    )


# --- Embedding parser helper (robust) ---
def parse_vec(x) -> Optional[np.ndarray]:
    """
    Robust gegen verschiedene Formate:
    - JSON-Array: "[0.1, -0.2, ...]"
    - Kommagetrennt: "0.1, -0.2, ..."
    - Whitespace/; / | getrennt
    """
    import json
    import re

    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=float)

    s = str(x).strip()
    if not s:
        return None

    # JSON-Array?
    if s.startswith("[") and s.endswith("]"):
        try:
            return np.asarray(json.loads(s), dtype=float)
        except Exception:
            # Fallback unten versuchen
            pass

    # Klammern raus, dann an Komma/Whitespace/; / | splitten
    s_clean = re.sub(r"[\[\]]", "", s)
    parts = [p for p in re.split(r"[,\s;|]+", s_clean) if p]

    try:
        vec = np.asarray([float(p) for p in parts], dtype=float)
        return vec if vec.size > 0 else None
    except Exception:
        return None


def read_any_file(f) -> Optional[pd.DataFrame]:
    if f is None:
        return None
    name = (getattr(f, "name", "") or "").lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(f)
        else:
            return pd.read_excel(f)
    except Exception as e:
        st.error(f"Fehler beim Lesen von {getattr(f, 'name', 'Datei')}: {e}")
        return None


def build_related_from_embeddings(
    urls: List[str],
    V: np.ndarray,
    top_k: int,
    sim_threshold: float,
    backend: str,
    faiss_available: bool,
) -> pd.DataFrame:
    """
    Erzeugt eine DataFrame wie das GAS-Tab 'Related URLs' mit Spalten: Ziel, Quelle, Similarity
    - V muss L2-normalisiert sein.
    """
    n = V.shape[0]
    if n < 2:
        # keine Paare möglich
        return pd.DataFrame(columns=["Ziel", "Quelle", "Similarity"])

    K = int(top_k)
    pairs = []  # (target, source, sim)

    if backend == "Schnell (FAISS)" and faiss_available:
        import faiss  # type: ignore

        dim = V.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product == Cosine (bei L2-Norm)
        Vf = V.astype("float32")
        index.add(Vf)
        topk = min(K + 1, n)  # +1, Self-Match fällt raus
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
    else:
        # Exakte Brute-Force Variante (NumPy)
        sims_full = V @ V.T  # N x N
        np.fill_diagonal(sims_full, -1.0)  # Self-Match ausschließen
        for i in range(n):
            sims = sims_full[i]
            if K < n - 1:
                idx = np.argpartition(sims, -K)[-K:]
            else:
                idx = np.argsort(sims)
            idx = idx[np.argsort(sims[idx])][::-1]  # absteigend sortiert
            taken = 0
            for j in idx:
                s = float(sims[j])
                if s < sim_threshold:
                    continue
                pairs.append([urls[i], urls[j], s])
                taken += 1
                if taken >= K:
                    break

    if not pairs:
        return pd.DataFrame(columns=["Ziel", "Quelle", "Similarity"])

    df_rel = pd.DataFrame(pairs, columns=["Ziel", "Quelle", "Similarity"])
    return df_rel


# ===============================
# UI / Settings
# ===============================

st.set_page_config(page_title="ONE Link Intelligence (Streamlit)", layout="wide")
st.title("ONE Link Intelligence – Streamlit Edition")
st.caption("Konvertiert die Google Apps Script Logik in eine lokale, dateibasierte Streamlit-App.")

with st.sidebar:
    st.header("Einstellungen")
    remove_www = st.checkbox("www. beim Normalisieren entfernen", value=False)
    show_numbers_in_heatmap = st.checkbox("Linkpotenzial-Zahlen anzeigen", value=True)

    st.subheader("Gewichtungen (nur für Linkpotenzial)")
    w_ils = st.slider("Interner Link Score", 0.0, 1.0, 0.30, 0.01)
    w_pr = st.slider("PageRank Horder Score", 0.0, 1.0, 0.35, 0.01)
    w_rd = st.slider("Referring Domains", 0.0, 1.0, 0.20, 0.01)
    w_bl = st.slider("Backlinks", 0.0, 1.0, 0.15, 0.01)

    w_sum = w_ils + w_pr + w_rd + w_bl
    if not math.isclose(w_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        st.warning(f"Gewichtungs-Summe = {w_sum:.2f} (sollte 1.0 sein)")

    st.subheader("Schwellen & Limits")
    sim_threshold = st.slider("Ähnlichkeitsschwelle (Related URLs)", 0.0, 1.0, 0.80, 0.01)
    max_related = st.number_input("Max. Related pro Ziel", min_value=1, max_value=50, value=10, step=1)
    not_similar_threshold = st.slider(
        "Unähnlichkeits-Schwelle (zweite Analyse) – Links ≤ Wert gelten als schwach", 0.0, 1.0, 0.60, 0.01
    )
    backlink_weight_2x = st.checkbox("Backlink-Gewicht in 'schwache Links' verdoppeln", value=False)

    st.subheader("Matching-Backend")
    backend = st.radio("Methode", ["Exakt (NumPy)", "Schnell (FAISS)"], horizontal=True)
    try:
        import faiss  # type: ignore

        faiss_available = True
    except Exception:
        faiss_available = False
        if backend == "Schnell (FAISS)":
            st.warning("FAISS ist in dieser Umgebung nicht verfügbar – wechsle auf 'Exakt (NumPy)'.")
            backend = "Exakt (NumPy)"

st.markdown("---")

# ===============================
# Data ingestion
# ===============================

st.subheader("Daten laden")
mode = st.radio(
    "Eingabemodus", ["Ein Excel mit Tabs", "Einzeltabellen hochladen", "URLs + Embeddings"], horizontal=True
)

related_df = inlinks_df = metrics_df = backlinks_df = None
emb_df = None

if mode == "Ein Excel mit Tabs":
    xls = st.file_uploader(
        "Excel hochladen (enthält Tabs: Related URLs, All Inlinks, Linkmetriken, Backlinks)",
        type=["xlsx", "xlsm", "xls"],
    )
    if xls is not None:
        try:
            xls_obj = pd.ExcelFile(xls)

            def get_sheet_like(candidates):
                low = [s.lower() for s in xls_obj.sheet_names]
                for cand in candidates:
                    if cand.lower() in low:
                        return xls_obj.sheet_names[low.index(cand.lower())]
                return None

            tab_related = get_sheet_like(["Related URLs", "Related", "related urls"]) or xls_obj.sheet_names[0]
            tab_inlinks = get_sheet_like(["All Inlinks", "Inlinks", "all inlinks"]) or xls_obj.sheet_names[
                1 if len(xls_obj.sheet_names) > 1 else 0
            ]
            tab_metrics = get_sheet_like(["Linkmetriken", "Metrics", "linkmetriken"]) or xls_obj.sheet_names[
                2 if len(xls_obj.sheet_names) > 2 else 0
            ]
            tab_backlinks = get_sheet_like(["Backlinks", "Ref Domains", "backlinks"]) or xls_obj.sheet_names[
                3 if len(xls_obj.sheet_names) > 3 else 0
            ]

            related_df = pd.read_excel(xls_obj, sheet_name=tab_related)
            inlinks_df = pd.read_excel(xls_obj, sheet_name=tab_inlinks)
            metrics_df = pd.read_excel(xls_obj, sheet_name=tab_metrics)
            backlinks_df = pd.read_excel(xls_obj, sheet_name=tab_backlinks)
            st.success("Excel erfolgreich gelesen.")
        except Exception as e:
            st.error(f"Fehler beim Lesen der Excel-Datei: {e}")

elif mode == "Einzeltabellen hochladen":
    col1, col2 = st.columns(2)
    with col1:
        related_up = st.file_uploader("Related URLs (CSV/Excel)", type=["csv", "xlsx", "xlsm", "xls"], key="rel")
        metrics_up = st.file_uploader("Linkmetriken (CSV/Excel)", type=["csv", "xlsx", "xlsm", "xls"], key="met")
    with col2:
        inlinks_up = st.file_uploader("All Inlinks (CSV/Excel)", type=["csv", "xlsx", "xlsm", "xls"], key="inl")
        backlinks_up = st.file_uploader("Backlinks (CSV/Excel)", type=["csv", "xlsx", "xlsm", "xls"], key="bl")

    related_df = read_any_file(related_up)
    inlinks_df = read_any_file(inlinks_up)
    metrics_df = read_any_file(metrics_up)
    backlinks_df = read_any_file(backlinks_up)

elif mode == "URLs + Embeddings":
    st.write(
        "Lade eine Tabelle mit **URL** und **Embedding** (JSON-Array oder durch Komma/Leerzeichen getrennte Zahlen). "
        "Zusätzlich werden **All Inlinks**, **Linkmetriken** und **Backlinks** benötigt."
    )
    up_emb = st.file_uploader("URLs + Embeddings (CSV/Excel)", type=["csv", "xlsx", "xlsm", "xls"], key="embs")
    col1, col2 = st.columns(2)
    with col1:
        inlinks_up = st.file_uploader("All Inlinks (CSV/Excel)", type=["csv", "xlsx", "xlsm", "xls"], key="inl2")
        metrics_up = st.file_uploader("Linkmetriken (CSV/Excel)", type=["csv", "xlsx", "xlsm", "xls"], key="met2")
    with col2:
        backlinks_up = st.file_uploader("Backlinks (CSV/Excel)", type=["csv", "xlsx", "xlsm", "xls"], key="bl2")

    emb_df = read_any_file(up_emb)
    inlinks_df = read_any_file(inlinks_up)
    metrics_df = read_any_file(metrics_up)
    backlinks_df = read_any_file(backlinks_up)

    # Wenn Embeddings geladen wurden: Related-URLs aus Embeddings ableiten
    if emb_df is not None and not emb_df.empty:
        # Spalten erkennen: URL + Embedding
        cols = [c for c in emb_df.columns]
        url_col = None
        emb_col = None
        for c in cols:
            lc = str(c).strip().lower()
            if url_col is None and lc in ("url", "urls", "page", "seite", "adresse", "address"):
                url_col = c
            if emb_col is None and lc in ("embedding", "embeddings", "vector", "embedding_json", "vec"):
                emb_col = c
        if url_col is None:
            url_col = cols[0]
        if emb_col is None and len(cols) >= 2:
            emb_col = cols[1]

        # URLs + Vektoren extrahieren
        urls: List[str] = []
        vecs: List[np.ndarray] = []
        for _, r in emb_df.iterrows():
            u = normalize_url(r[url_col], remove_www)
            v = parse_vec(r[emb_col])
            if not u or v is None:
                continue
            urls.append(u)
            vecs.append(v)

        if len(vecs) < 2:
            st.error("Zu wenige gültige Embeddings erkannt (mindestens 2 benötigt).")
            related_df = pd.DataFrame(columns=["Ziel", "Quelle", "Similarity"])
        else:
            # auf gleiche Dimensionalität bringen (pad/truncate)
            max_dim = max(v.size for v in vecs)
            V = np.zeros((len(vecs), max_dim), dtype=float)
            for i, v in enumerate(vecs):
                d = min(max_dim, v.size)
                V[i, :d] = v[:d]

            # L2-Norm
            norms = np.linalg.norm(V, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            V = V / norms

            related_df = build_related_from_embeddings(
                urls=urls,
                V=V,
                top_k=int(max_related),
                sim_threshold=float(sim_threshold),
                backend=backend,
                faiss_available=bool(faiss_available),
            )

# Prüfen, ob alles da ist
have_all = all(df is not None for df in [related_df, inlinks_df, metrics_df, backlinks_df])

if not have_all:
    st.info("Bitte alle benötigten Tabellen laden, um die Analysen zu starten.")
    st.stop()

# ===============================
# Normalization maps / data prep
# ===============================

# Build metrics map (URL -> score, prDiff)
metrics_df = metrics_df.copy()
metrics_df.columns = [str(c).strip() for c in metrics_df.columns]

# Erwartet: [url, score, inlinks, outlinks] (wie im GAS)
if metrics_df.shape[1] < 4:
    st.error("'Linkmetriken' braucht mindestens 4 Spalten: URL, Score, Inlinks, Outlinks (in dieser Reihenfolge).")
    st.stop()

metrics_df.iloc[:, 0] = metrics_df.iloc[:, 0].astype(str)

metrics_map: Dict[str, Dict[str, float]] = {}
min_ils, max_ils = float("inf"), float("-inf")
min_prd, max_prd = float("inf"), float("-inf")

for _, r in metrics_df.iterrows():
    u = normalize_url(r.iloc[0], remove_www)
    if not u:
        continue
    score = float(pd.to_numeric(r.iloc[1], errors="coerce") or 0)
    inlinks = float(pd.to_numeric(r.iloc[2], errors="coerce") or 0)
    outlinks = float(pd.to_numeric(r.iloc[3], errors="coerce") or 0)
    prdiff = inlinks - outlinks
    metrics_map[u] = {"score": score, "prDiff": prdiff}
    min_ils, max_ils = min(min_ils, score), max(max_ils, score)
    min_prd, max_prd = min(min_prd, prdiff), max(max_prd, prdiff)

# Backlinks map (URL -> backlinks, referring domains)
backlinks_df = backlinks_df.copy()
if backlinks_df.shape[1] < 3:
    st.error("'Backlinks' braucht mindestens 3 Spalten: URL, Backlinks, Referring Domains (in dieser Reihenfolge).")
    st.stop()

backlink_map: Dict[str, Dict[str, float]] = {}
min_rd, max_rd = float("inf"), float("-inf")
min_bl, max_bl = float("inf"), float("-inf")

for _, r in backlinks_df.iterrows():
    u = normalize_url(r.iloc[0], remove_www)
    if not u:
        continue
    bl = float(pd.to_numeric(r.iloc[1], errors="coerce") or 0)
    rd = float(pd.to_numeric(r.iloc[2], errors="coerce") or 0)
    backlink_map[u] = {"backlinks": bl, "referringDomains": rd}
    min_bl, max_bl = min(min_bl, bl), max(max_bl, bl)
    min_rd, max_rd = min(min_rd, rd), max(max_rd, rd)

# Inlinks: gather all and content links
inlinks_df = inlinks_df.copy()
header = [str(c).strip() for c in inlinks_df.columns]

src_idx = find_column_index(header, POSSIBLE_SOURCE)
dst_idx = find_column_index(header, POSSIBLE_TARGET)
pos_idx = find_column_index(header, POSSIBLE_POSITION)

if src_idx == -1 or dst_idx == -1:
    st.error("In 'All Inlinks' wurden die Spalten 'Quelle/Source' oder 'Ziel/Destination' nicht gefunden.")
    st.stop()

all_links = set()
content_links = set()

for _, r in inlinks_df.iterrows():
    source = normalize_url(r.iloc[src_idx], remove_www)
    target = normalize_url(r.iloc[dst_idx], remove_www)
    if not source or not target:
        continue
    key = f"{source}→{target}"
    all_links.add(key)
    if pos_idx != -1 and is_content_position(r.iloc[pos_idx]):
        content_links.add(key)

# Related map (bidirectional, thresholded)
related_df = related_df.copy()
if related_df.shape[1] < 3:
    st.error("'Related URLs' braucht mindestens 3 Spalten: Ziel, Quelle, Similarity (in dieser Reihenfolge).")
    st.stop()

related_map: Dict[str, List[Tuple[str, float]]] = {}
processed_pairs = set()

for _, r in related_df.iterrows():
    urlA = normalize_url(r.iloc[0], remove_www)
    urlB = normalize_url(r.iloc[1], remove_www)
    try:
        sim = float(str(r.iloc[2]).replace(",", "."))
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

# ===============================
# Analyse 1: Interne Verlinkungsmöglichkeiten
# ===============================

st.markdown("## Analyse 1: Interne Verlinkungsmöglichkeiten")

cols = ["Ziel-URL"]
for i in range(1, int(max_related) + 1):
    cols.extend(
        [f"Related URL {i}", f"überhaupt verlinkt {i}?", f"aus Inhalt heraus verlinkt {i}?", f"Linkpotenzial {i}"]
    )

rows = []

for target, related_list in sorted(related_map.items()):
    related_sorted = sorted(related_list, key=lambda x: x[1], reverse=True)[: int(max_related)]
    row = [target]
    for source, sim in related_sorted:
        key = f"{source}→{target}"
        anywhere = "ja" if key in all_links else "nein"
        from_content = "ja" if key in content_links else "nein"

        m = metrics_map.get(source, {"score": 0.0, "prDiff": 0.0})
        ils_raw = float(m.get("score", 0.0))
        pr_raw = float(m.get("PrDiff".lower(), m.get("prDiff", 0.0)))  # robust gegen Key-Case

        bl = backlink_map.get(source, {"backlinks": 0.0, "referringDomains": 0.0})
        bl_raw = float(bl.get("backlinks", 0.0))
        rd_raw = float(bl.get("referringDomains", 0.0))

        # Safe normalization
        norm_ils = (ils_raw - min_ils) / (max_ils - min_ils) if max_ils > min_ils else 0.0
        norm_pr = (pr_raw - min_prd) / (max_prd - min_prd) if max_prd > min_prd else 0.0
        norm_bl = (bl_raw - min_bl) / (max_bl - min_bl) if max_bl > min_bl else 0.0
        norm_rd = (rd_raw - min_rd) / (max_rd - min_rd) if max_rd > min_rd else 0.0

        final_score = (w_ils * norm_ils) + (w_pr * norm_pr) + (w_bl * norm_bl) + (w_rd * norm_rd)
        final_score = round(final_score, 2)

        # Hinweis: Streamlit bietet keine Heatmap im DataFrame. Wir zeigen die Zahl an.
        row.extend([source, anywhere, from_content, final_score])

    # pad
    while len(row) < len(cols):
        row.append("")
    rows.append(row)

res1_df = pd.DataFrame(rows, columns=cols)
st.dataframe(res1_df, use_container_width=True, hide_index=True)

csv1 = res1_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download 'Interne Verlinkungsmöglichkeiten' (CSV)",
    data=csv1,
    file_name="interne_verlinkungsmoeglichkeiten.csv",
    mime="text/csv",
)

# ===============================
# Analyse 2: Potenziell zu entfernende Links
# ===============================

st.markdown("## Analyse 2: Potenziell zu entfernende Links")

# Build similarity map for both directions
sim_map: Dict[str, float] = {}
processed_pairs2 = set()

for _, r in related_df.iterrows():
    a = normalize_url(r.iloc[1], remove_www)  # Quelle (wie im GAS)
    b = normalize_url(r.iloc[0], remove_www)  # Ziel   (wie im GAS)
    try:
        sim = float(str(r.iloc[2]).replace(",", "."))
    except Exception:
        continue
    if not a or not b:
        continue
    pair_key = "↔".join(sorted([a, b]))
    if pair_key in processed_pairs2:
        continue
    sim_map[f"{a}→{b}"] = sim
    sim_map[f"{b}→{a}"] = sim
    processed_pairs2.add(pair_key)

# PageRank-Waster-ähnlicher Rohwert und backlink-adjusted Score
raw_score_map: Dict[str, float] = {}
for _, r in metrics_df.iterrows():
    u = normalize_url(r.iloc[0], remove_www)
    inl = float(pd.to_numeric(r.iloc[2], errors="coerce") or 0)
    outl = float(pd.to_numeric(r.iloc[3], errors="coerce") or 0)
    raw_score_map[u] = outl - inl

adjusted_score_map: Dict[str, float] = {}
for u, raw in raw_score_map.items():
    bl = backlink_map.get(u, {"backlinks": 0.0, "referringDomains": 0.0})
    impact = bl.get("backlinks", 0.0) * 0.5 + bl.get("referringDomains", 0.0) * 0.5
    factor = 2.0 if backlink_weight_2x else 1.0
    malus = 5.0 * factor if impact == 0 else 0.0
    adjusted = (raw or 0.0) - (factor * impact) + malus
    adjusted_score_map[u] = adjusted

# Build output
out_rows = []
rest_cols = [c for i, c in enumerate(header) if i not in (src_idx, dst_idx)]
out_header = ["Quelle", "Ziel", "PageRank Waster (Farbindikator)", "Semantische Ähnlichkeit", *rest_cols]

for _, r in inlinks_df.iterrows():
    quelle = normalize_url(r.iloc[src_idx], remove_www)
    ziel = normalize_url(r.iloc[dst_idx], remove_www)
    if not quelle or not ziel:
        continue

    k1 = f"{quelle}→{ziel}"
    k2 = f"{ziel}→{quelle}"
    sim = sim_map.get(k1, sim_map.get(k2, np.nan))

    # Weak links: similarity <= threshold OR missing
    if not (isinstance(sim, (int, float)) and not np.isnan(sim)):
        sim_display = "Ähnlichkeit unter Schwelle oder nicht erfasst"
        is_weak = True
    else:
        sim_display = sim
        is_weak = sim <= not_similar_threshold

    if not is_weak:
        continue

    rest = [r.iloc[i] for i in range(len(header)) if i not in (src_idx, dst_idx)]
    out_rows.append([quelle, ziel, "", sim_display, *rest])

out_df = pd.DataFrame(out_rows, columns=out_header)

# Coloring by adjusted score (simple buckets)
colors = []
for _, row in out_df.iterrows():
    q = row["Quelle"]
    score = adjusted_score_map.get(q)
    if score is None or (isinstance(score, float) and np.isnan(score)):
        colors.append("#ffffff")
    elif score >= 50:
        colors.append("#ffcccc")
    elif score >= 25:
        colors.append("#fff2cc")
    else:
        colors.append("#ccffcc")

out_df["Farbcode (intern)"] = colors
st.dataframe(out_df, use_container_width=True, hide_index=True)

csv2 = out_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download 'Potenziell zu entfernende Links' (CSV)",
    data=csv2,
    file_name="potenziell_zu_entfernende_links.csv",
    mime="text/csv",
)

st.markdown(
    """
**Hinweise**
- Die Linkpotenzial-Berechnung nutzt **nur** die vier Metriken (Interner Link Score, PageRank Horder Score, Backlinks, Referring Domains). Die Ähnlichkeit wirkt als **Filter** über die Schwelle.
- Für Heatmaps innerhalb von Streamlit wird hier der Score angezeigt; echte Zell-Hintergründe wie in Google Sheets sind in CSV-Exports natürlich nicht enthalten.
- Die Spaltenerkennung in *All Inlinks* ist robust gegen deutsch/englische Varianten (Quelle/Source, Ziel/Destination, Position).
- Für bestmögliche Konsistenz sollten alle URLs im Input einheitlich sein (gleiche Protokolle, www-Konvention). Optional kannst du "www." während der Normalisierung entfernen.
"""
)
