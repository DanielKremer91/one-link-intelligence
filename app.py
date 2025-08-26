import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ===============================
# Page config & Branding
# ===============================
st.set_page_config(page_title="ONE Link Intelligence", layout="wide")

# Session-State initialisieren (für persistente Outputs)
if "ready" not in st.session_state:
    st.session_state.ready = False
if "res1_df" not in st.session_state:
    st.session_state.res1_df = None
if "out_df" not in st.session_state:
    st.session_state.out_df = None

# Remote-Logo robust laden (kein Crash, wenn Bild nicht geht)
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
<div style="background-color: #f2f2f2; color: #000000; padding: 15px 20px; border-radius: 6px; font-size: 1.2em; max-width: 900px; margin-bottom: 1.5em; line-height: 1.5;">
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

POSSIBLE_SOURCE = ["quelle", "source", "from", "origin"]
POSSIBLE_TARGET = ["ziel", "destination", "to", "target"]
POSSIBLE_POSITION = ["linkposition", "link position", "position"]

def _num(x, default: float = 0.0) -> float:
    """Robuste Numerik: NaN/None sicher auf Default."""
    v = pd.to_numeric(x, errors="coerce")
    return default if pd.isna(v) else float(v)

def _safe_minmax(lo, hi) -> Tuple[float, float]:
    """Sichere Min/Max-Ranges (verhindert +/-inf oder hi<=lo)."""
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else (0.0, 1.0)

def find_column_index(header: List[str], possible_names: List[str]) -> int:
    lower = [str(h).strip().lower() for h in header]
    for i, h in enumerate(lower):
        if h in possible_names:
            return i
    return -1

def normalize_url(u: str) -> str:
    """URL-Kanonisierung: Protokoll ergänzen, Tracking-Parameter entfernen, Query sortieren,
    Trailing Slash normalisieren, www. entfernen."""
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
        # www. automatisch entfernen
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
        token in pos_norm for token in ["inhalt", "content", "body", "main", "artikel", "article"]
    )

# --- Embedding parser helper (robust) ---
def parse_vec(x) -> Optional[np.ndarray]:
    """
    Robust gegen verschiedene Formate:
    - JSON-Array: "[0.1, -0.2, ...]"
    - Komma-getrennt: "0.1, -0.2, ..."
    - Whitespace / ; / | getrennt
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
            pass

    # Klammern raus, dann an Komma/Whitespace/; / | splitten
    s_clean = re.sub(r"[\[\]]", "", s)
    parts = [p for p in re.split(r"[,\s;|]+", s_clean) if p]

    try:
        vec = np.asarray([float(p) for p in parts], dtype=float)
        return vec if vec.size > 0 else None
    except Exception:
        return None

# --- Robuster Datei-Leser mit Encoding- und Delimiter-Erkennung ---
def read_any_file(f) -> Optional[pd.DataFrame]:
    """CSV/Excel robust lesen: probiert mehrere Encodings; snifft Delimiter (Komma/Semikolon/Tab)."""
    if f is None:
        return None
    name = (getattr(f, "name", "") or "").lower()
    try:
        if name.endswith(".csv"):
            # Mehrere Encodings probieren
            for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
                try:
                    f.seek(0)
                    # Delimiter sniffer braucht den python-Parser
                    return pd.read_csv(
                        f,
                        sep=None,            # Trenner automatisch erkennen
                        engine="python",     # nötig für sep=None
                        encoding=enc,
                        on_bad_lines="skip", # robust gegen Ausreißer
                    )
                except UnicodeDecodeError:
                    continue
                except Exception:
                    # Mehrere Fallback-Delimiter testen
                    for sep_try in [";", ",", "\t"]:
                        try:
                            f.seek(0)
                            return pd.read_csv(
                                f, sep=sep_try, engine="python", encoding=enc, on_bad_lines="skip"
                            )
                        except Exception:
                            continue
            raise ValueError("Kein passendes Encoding/Trennzeichen gefunden.")
        else:
            f.seek(0)
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
    - Bei FAISS verwenden wir Inner Product auf L2-normalisierten Vektoren (entspricht Cosine).
    """
    n = V.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["Ziel", "Quelle", "Similarity"])

    K = int(top_k)
    pairs = []  # (target, source, sim)

    if backend == "Schnell (FAISS)" and faiss_available:
        import faiss  # type: ignore

        dim = V.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product == Cosine bei L2-Norm
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

# =============================
# Hilfe / Tool-Dokumentation (Expander)
# =============================
with st.expander("❓ Hilfe / Tool-Dokumentation", expanded=False):
    st.markdown("""
## Was macht das Tool ONE Link Intelligence?

**ONE Link Intelligence** besteht aus zwei Analysen:  

1. **Interne Links finden**  
   - Auf Basis semantischer Ähnlichkeit wird geprüft, ob thematisch verwandte Seiten bereits intern miteinander verlinkt sind.  
   - Das Tool schlägt zusätzlich sinnvolle interne Links vor und bewertet deren Potenzial mit einem **Linkpotenzial-Score**.  

2. **Unpassende Links identifizieren**  
   - Analysiert bestehende interne Links und erkennt solche, die thematisch unpassend oder schwach sind.  
   - Grundlage ist die semantische Ähnlichkeit sowie ein vereinfachter *PageRank-Waster-Ansatz* (Seiten mit vielen Outlinks, aber wenigen Inlinks).  

Beide Tools zahlen direkt auf die **Optimierung deiner internen Verlinkung** ein.
""")

    st.markdown("""
### 🔄 Input-Dateien

- **Option 1: URLs + Embeddings**  
  Tabelle mit mindestens zwei Spalten:  
  - **URL** (Spaltenname: z. B. URL, Adresse, Address, Page, Seite)  
  - **Embeddings** (Spaltenname: z. B. Embedding, Embeddings, Vector). Werte können als JSON-Array ([0.1, 0.2, ...]) oder durch Komma/Leerzeichen/;/| getrennt vorliegen.  

  Zusätzlich erforderlich:  
  - **All Inlinks** (CSV/Excel, aus Screaming Frog: *Massenexport → Links → Alle Inlinks*) — enthält mindestens: **Quelle/Source**, **Ziel/Destination**, optional **Linkposition/Link Position**  
  - **Linkmetriken** (CSV/Excel) — **erste 4 Spalten** in dieser Reihenfolge: **URL**, **Score**, **Inlinks**, **Outlinks**  
  - **Backlinks** (CSV/Excel) — **erste 3 Spalten** in dieser Reihenfolge: **URL**, **Backlinks**, **Referring Domains**  

- **Option 2: Related URLs**  
  Tabelle mit mindestens drei Spalten:  
  - **Ziel-URL**, **Quell-URL**, **Ähnlichkeitswert (0–1)** (z. B. aus Screaming Frog *Massenexport → Inhalt → Semantisch ähnlich*).  

Hinweis: Spaltenerkennung ist tolerant gegenüber deutsch/englischen Varianten.  
Trennzeichen (Komma/Semikolon) und Encodings (UTF-8/UTF-8-SIG/Windows-1252/Latin-1) werden automatisch erkannt.  
URLs werden kanonisiert (Protokoll ergänzt, www. entfernt, Tracking-Parameter entfernt, Pfade vereinheitlicht).
""")

    st.markdown("""
### ⚙️ Gewichtung (Linkpotenzial)

Die Berechnung des Linkpotenzials basiert auf folgenden Faktoren:  

- **Interner Link Score**  
  Bewertet, wie wichtig eine Seite im internen Linkgraph ist (ähnlich dem Link Score in Screaming Frog). Je höher der Wert, desto stärker kann die Seite Linkpower weitergeben.  

- **PageRank-Horder-Score**  
  Was ist ein *PageRank-Horder*? Vereinfacht gesagt: Je mehr eingehende Links (intern & extern) und je weniger ausgehende Links eine URL hat, desto mehr Linkpower kann sie „vererben“. Das „Robin-Hood-Prinzip“ – take it from the rich, give it to the poor.  

- **Backlinks** & **Referring Domains**  
  Berücksichtigen externe Signale (Autorität/Vertrauen) der Quell-URL.  

💡 **Interpretation des Linkpotenzial-Scores in der Output-Datei:**  
Der Wert ist **relativ** – er zeigt im Verhältnis zu den anderen, wie lukrativ ein Link wäre.  
Je **höher** der Score im Vergleich zu den übrigen, desto sinnvoller ist die Verlinkung.

*Hinweis:* Die Ermittlung **zu entfernender Links** berücksichtigt **alle Similarities** (nicht nur ≥ Schwelle), damit auch sehr schwache Verbindungen sichtbar werden.
""")

    st.markdown(
        """
### 🧪 Optional: Visualisierung & Gems (nachgelagert)

- Interaktive Graphen (IST/SOLL/Zukunft) werden **erst geladen**, wenn du sie aktivierst – die initiale Analyse bleibt schnell.
- **Gems** = stärkste Linkgeber (Top-X % nach Linkpotenzial; X per Slider bis max. 30 %).
- Empfehlungen je Gem:
  - **Nur Similarity** oder **Similarity + Opportunity (GSC)**  
    Opportunity basiert auf *log1p(Impressions)* → Min-Max und **CTR = Clicks/(Impressions+1)**:  
    `opp = norm_impr * (1 - CTR)`; Gesamtscore `rank = α·Similarity + β·opp`.
- Zukunfts-Graph zeigt **nur Änderungen**: neue Links grün, entfernte Links werden ausgeblendet.
- Kanten-Limits & Suche sorgen für Übersichtlichkeit (Highlight der gesuchten URL + Nachbarn; Rest ausgegraut).
""")
# ===============================
# Sidebar Controls (mit Tooltips)
# ===============================
with st.sidebar:

    st.header("Einstellungen")

    # Matching-Backend (weiter oben, ausführliche Hilfe)
    try:
        import faiss  # type: ignore
        faiss_available = True
    except Exception:
        faiss_available = False

    backend_default = "Schnell (FAISS)" if faiss_available else "Exakt (NumPy)"
    backend = st.radio(
        "Matching-Backend",
        ["Exakt (NumPy)", "Schnell (FAISS)"],
        index=0 if backend_default=="Exakt (NumPy)" else 1,
        horizontal=True,
        help=("Bestimmt, wie semantische Nachbarn ermittelt werden (Cosine Similarity):\n\n"
              "- **Exakt (NumPy)**: O(N²), sehr genau. Gut bis ca. 2.000–5.000 URLs (abhängig von RAM & Dim.).\n"
              "- **Schnell (FAISS)**: Approximate Nearest Neighbor, sehr schnell & speichereffizient. "
              "Empfohlen ab ~5.000–10.000 URLs oder wenn NumPy zu langsam wird.\n\n"
              "Beide liefern Cosine-Similarity (0–1). Wenn 'faiss-cpu' nicht installiert ist, fällt die App automatisch auf NumPy zurück.")
    )
    if not faiss_available and backend == "Schnell (FAISS)":
        st.warning("FAISS ist in dieser Umgebung nicht verfügbar – wechsle auf 'Exakt (NumPy)'.")
        backend = "Exakt (NumPy)"

    st.subheader("Gewichtung (Linkpotenzial)")
    st.caption(
        "Das Linkpotenzial gewichtet die Autorität/Relevanz der **Quell-URL**. "
    )
    w_ils = st.slider(
        "Interner Link Score",
        0.0, 1.0, 0.30, 0.01,
        help=("Interner Link Score (Screaming Frog): PageRank-ähnliches Maß für interne Linkpopularität aus dem Crawl. "
              "Höherer ILS ⇒ Quelle kann mehr interne Linkkraft vererben.")
    )
    w_pr = st.slider(
        "PageRank-Horder-Score",
        0.0, 1.0, 0.35, 0.01,
        help=("Was ist ein PageRank-Horder?\n\n"
              "Je mehr eingehende Links (intern & extern) und je weniger ausgehende Links eine URL hat, "
              "desto mehr Linkpower kann sie „vererben“. Das „Robin-Hood-Prinzip“ – take it from the rich, give it to the poor. "
              "Solche URLs werden in der Linkpotenzial-Kalkulation höher priorisiert.")
    )
    w_rd = st.slider(
        "Referring Domains",
        0.0, 1.0, 0.20, 0.01,
        help="Externe verweisende Domains der Quell-URL (Autorität/Vertrauen)."
    )
    w_bl = st.slider(
        "Backlinks",
        0.0, 1.0, 0.15, 0.01,
        help="Externe Backlinks der Quell-URL."
    )
    w_sum = w_ils + w_pr + w_rd + w_bl
    if not math.isclose(w_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        st.warning(f"Gewichtungs-Summe = {w_sum:.2f} (sollte 1.0 sein)")

    st.subheader("Schwellen & Limits (Related-Ermittlung)")
    sim_threshold = st.slider(
        "Ähnlichkeitsschwelle",
        0.0, 1.0, 0.80, 0.01,
        help="Nur Paare mit Cosine Similarity ≥ diesem Wert gelten als „related“."
    )
    max_related = st.number_input(
        "Anzahl Related URLs",
        min_value=1, max_value=50, value=10, step=1,
        help="Wie viele semantisch ähnliche Seiten sollen pro Ziel-URL in die Analyse einbezogen werden?"
    )

    st.subheader("Entfernung von Links")
    not_similar_threshold = st.slider(
        "Unähnlichkeits-Schwelle (schwache Links)",
        0.0, 1.0, 0.60, 0.01,
        help=("Interne Links gelten als schwach, wenn deren semantische Ähnlichkeit ≤ diesem Wert liegt. "
              "Beispiel: 0.60 → alle Links ≤ 0.60 werden als potenziell zu entfernend gelistet.")
    )
    backlink_weight_2x = st.checkbox(
        "Backlinks/Ref. Domains doppelt gewichten",
        value=False,
        help=("Erhöht den Dämpfungseffekt externer Autorität auf den Waster-Score. "
              "Wenn aktiv, wirken Backlinks & Ref. Domains doppelt so stark.")
    )

# etwas CSS für den roten Button (wir nutzen ihn später für „Let's Go“)
st.markdown(
    """
<style>
div.stButton > button[kind="secondary"] {
  background-color: #e02424 !important;
  color: white !important;
  border: 1px solid #e02424 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

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
    st.write(
        "Lade eine Datei mit **URL** und **Embedding** (JSON-Array oder Zahlen, getrennt durch Komma/Whitespace/;/|). "
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

elif mode == "Related URLs":
    st.write(
        "Lade die vier Tabellen: **Related URLs**, **All Inlinks**, **Linkmetriken**, **Backlinks** "
        "(CSV/Excel; Trennzeichen & Encodings werden automatisch erkannt)."
    )
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

# ===============================
# Let's Go Button (startet Berechnungen)
# ===============================
run_clicked = st.button("Let's Go", type="secondary")  # durch CSS rot

# Wenn noch nichts gerechnet wurde UND Button nicht geklickt: Hinweis & Abbruch
if not run_clicked and not st.session_state.ready:
    st.info("Bitte Dateien hochladen und auf **Let's Go** klicken, um die Analysen zu starten.")
    st.stop()

# GIF nur anzeigen, wenn jetzt gerechnet wird
if run_clicked:
    placeholder = st.empty()
    # kleiner + mittig anzeigen
    with placeholder.container():
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.image(
                "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDJweGExcHhhOWZneTZwcnAxZ211OWJienY5cWQ1YmpwaHR0MzlydiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/dBRaPog8yxFWU/giphy.gif",
                width=280
            )
        st.caption("Die Berechnungen laufen – Zeit für eine kleine Stärkung …")

# ===============================
# Validierung & ggf. Ableitung Related aus Embeddings
# ===============================
if run_clicked or st.session_state.ready:
    if mode == "URLs + Embeddings":
        if emb_df is None or any(df is None for df in [inlinks_df, metrics_df, backlinks_df]):
            st.error("Bitte alle benötigten Dateien hochladen (Embeddings, All Inlinks, Linkmetriken, Backlinks).")
            st.stop()

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
                u = normalize_url(r[url_col])
                v = parse_vec(r[emb_col])
                if not u or v is None:
                    continue
                urls.append(u)
                vecs.append(v)

            if len(vecs) < 2:
                st.error("Zu wenige gültige Embeddings erkannt (mindestens 2 benötigt).")
                st.stop()

            # Dimensionalitäts-Check / -Harmonisierung
            dims = [v.size for v in vecs]
            max_dim = max(dims)
            V = np.zeros((len(vecs), max_dim), dtype=float)
            shorter = sum(1 for d in dims if d < max_dim)
            for i, v in enumerate(vecs):
                d = min(max_dim, v.size)
                V[i, :d] = v[:d]

            # L2-Norm
            norms = np.linalg.norm(V, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            V = V / norms
            V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)  # <- robust gegen NaN/Inf


            if shorter > 0:
                st.caption(f"⚠️ {shorter} Embeddings hatten geringere Dimensionen und wurden auf {max_dim} gepaddet.")

            related_df = build_related_from_embeddings(
                urls=urls,
                V=V,
                top_k=int(max_related),
                sim_threshold=float(sim_threshold),
                backend=backend,
                faiss_available=bool(faiss_available),
            )

            # ---- Embeddings & URLs im Session-State ablegen (nur in diesem Modus) ----
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
        u = normalize_url(r.iloc[0])
        if not u:
            continue
        score = _num(r.iloc[1])
        inlinks = _num(r.iloc[2])
        outlinks = _num(r.iloc[3])
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
        u = normalize_url(r.iloc[0])
        if not u:
            continue
        bl = _num(r.iloc[1])
        rd = _num(r.iloc[2])
        backlink_map[u] = {"backlinks": bl, "referringDomains": rd}
        min_bl, max_bl = min(min_bl, bl), max(max_bl, bl)
        min_rd, max_rd = min(min_rd, rd), max(max_rd, rd)

    # Safe ranges
    min_ils, max_ils = _safe_minmax(min_ils, max_ils)
    min_prd, max_prd = _safe_minmax(min_prd, max_prd)
    min_bl, max_bl = _safe_minmax(min_bl, max_bl)
    min_rd, max_rd = _safe_minmax(min_rd, max_rd)

    # Inlinks: gather all and content links  (=> Keys als Tupel (src, dst))
    inlinks_df = inlinks_df.copy()
    header = [str(c).strip() for c in inlinks_df.columns]

    src_idx = find_column_index(header, POSSIBLE_SOURCE)
    dst_idx = find_column_index(header, POSSIBLE_TARGET)
    pos_idx = find_column_index(header, POSSIBLE_POSITION)

    if src_idx == -1 or dst_idx == -1:
        st.error("In 'All Inlinks' wurden die Spalten 'Quelle/Source' oder 'Ziel/Destination' nicht gefunden.")
        st.stop()

    all_links: set[Tuple[str, str]] = set()
    content_links: set[Tuple[str, str]] = set()

    for _, r in inlinks_df.iterrows():
        source = normalize_url(r.iloc[src_idx])
        target = normalize_url(r.iloc[dst_idx])
        if not source or not target:
            continue
        key = (source, target)
        all_links.add(key)
        if pos_idx != -1 and is_content_position(r.iloc[pos_idx]):
            content_links.add(key)

    # Für Visualisierung vormerken
    st.session_state["_all_links"] = all_links
    st.session_state["_content_links"] = content_links

    # Related map (bidirectional, thresholded)
    related_df = related_df.copy()
    if related_df.shape[1] < 3:
        st.error("'Related URLs' braucht mindestens 3 Spalten: Ziel, Quelle, Similarity (in dieser Reihenfolge).")
        st.stop()

    related_map: Dict[str, List[Tuple[str, float]]] = {}
    processed_pairs = set()

    for _, r in related_df.iterrows():
        urlA = normalize_url(r.iloc[0])  # Ziel
        urlB = normalize_url(r.iloc[1])  # Quelle
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

    # Precompute "Linkpotenzial" als reine Quelleigenschaft (ohne Similarity)
    source_potential_map: Dict[str, float] = {}
    for u, m in metrics_map.items():
        ils_raw = _num(m.get("score"))
        pr_raw = _num(m.get("prDiff"))
        bl = backlink_map.get(u, {"backlinks": 0.0, "referringDomains": 0.0})
        bl_raw = _num(bl.get("backlinks"))
        rd_raw = _num(bl.get("referringDomains"))

        # Safe normalization
        norm_ils = (ils_raw - min_ils) / (max_ils - min_ils) if max_ils > min_ils else 0.0
        norm_pr = (pr_raw - min_prd) / (max_prd - min_prd) if max_prd > min_prd else 0.0
        norm_bl = (bl_raw - min_bl) / (max_bl - min_bl) if max_bl > min_bl else 0.0
        norm_rd = (rd_raw - min_rd) / (max_rd - min_rd) if max_rd > min_rd else 0.0

        final_score = (w_ils * norm_ils) + (w_pr * norm_pr) + (w_bl * norm_bl) + (w_rd * norm_rd)
        source_potential_map[u] = round(final_score, 4)

    st.session_state["_source_potential_map"] = source_potential_map
    st.session_state["_metrics_map"] = metrics_map
    st.session_state["_backlink_map"] = backlink_map
    st.session_state["_norm_ranges"] = {
        "ils": (min_ils, max_ils),
        "prd": (min_prd, max_prd),
        "bl": (min_bl, max_bl),
        "rd": (min_rd, max_rd),
    }

    # ===============================
    # Analyse 1: Interne Verlinkungsmöglichkeiten
    # ===============================
    st.markdown("## Analyse 1: Interne Verlinkungsmöglichkeiten")

    cols = ["Ziel-URL"]
    for i in range(1, int(max_related) + 1):
        cols.extend([
            f"Related URL {i}",
            f"Ähnlichkeit {i}",
            f"überhaupt verlinkt {i}?",
            f"aus Inhalt heraus verlinkt {i}?",
            f"Linkpotenzial {i}",
        ])

    rows = []

    for target, related_list in sorted(related_map.items()):
        related_sorted = sorted(related_list, key=lambda x: x[1], reverse=True)[: int(max_related)]
        row = [target]
        for source, sim in related_sorted:
            anywhere = "ja" if (source, target) in all_links else "nein"
            from_content = "ja" if (source, target) in content_links else "nein"

            final_score = source_potential_map.get(source, 0.0)
            row.extend([source, round(float(sim), 3), anywhere, from_content, final_score])

        # pad
        while len(row) < len(cols):
            row.append(np.nan)

        rows.append(row)

    res1_df = pd.DataFrame(rows, columns=cols)
    st.session_state.res1_df = res1_df  # -> persistieren
    
    # Arrow-kompatibel: Similarity-Spalten numeric casten
    sim_cols = [c for c in res1_df.columns if c.startswith("Ähnlichkeit ")]
    for c in sim_cols:
        res1_df[c] = pd.to_numeric(res1_df[c], errors="coerce")
        
    st.dataframe(res1_df, use_container_width=True, hide_index=True)

    csv1 = res1_df.to_csv(index=False).encode("utf-8-sig")
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
    sim_map: Dict[Tuple[str, str], float] = {}
    processed_pairs2 = set()

    for _, r in related_df.iterrows():
        a = normalize_url(r.iloc[1])  # Quelle (wie im GAS)
        b = normalize_url(r.iloc[0])  # Ziel   (wie im GAS)
        try:
            sim = float(str(r.iloc[2]).replace(",", "."))
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

    # PageRank-Waster-ähnlicher Rohwert und backlink-adjusted Score
    raw_score_map: Dict[str, float] = {}
    for _, r in metrics_df.iterrows():
        u = normalize_url(r.iloc[0])
        inl = _num(r.iloc[2])
        outl = _num(r.iloc[3])
        raw_score_map[u] = outl - inl

    adjusted_score_map: Dict[str, float] = {}
    for u, raw in raw_score_map.items():
        bl = backlink_map.get(u, {"backlinks": 0.0, "referringDomains": 0.0})
        impact = _num(bl.get("backlinks")) * 0.5 + _num(bl.get("referringDomains")) * 0.5
        factor = 2.0 if backlink_weight_2x else 1.0
        malus = 5.0 * factor if impact == 0 else 0.0
        adjusted = (raw or 0.0) - (factor * impact) + malus
        adjusted_score_map[u] = adjusted

    # Build output
    out_rows = []
    rest_cols = [c for i, c in enumerate(header) if i not in (src_idx, dst_idx)]
    out_header = ["Quelle", "Ziel", "PageRank Waster (Farbindikator)", "Semantische Ähnlichkeit", *rest_cols]

    for _, r in inlinks_df.iterrows():
        quelle = normalize_url(r.iloc[src_idx])
        ziel = normalize_url(r.iloc[dst_idx])
        if not quelle or not ziel:
            continue

        sim = sim_map.get((quelle, ziel), sim_map.get((ziel, quelle), np.nan))

        # Weak links: similarity ≤ threshold OR missing
        if not (isinstance(sim, (int, float)) and not np.isnan(sim)):
            sim_display = "Ähnlichkeit unter Schwelle oder nicht erfasst"
            is_weak = True
        else:
            sim_display = sim
            is_weak = sim <= not_similar_threshold

        if not is_weak:
            continue

        rest = [r.iloc[i] for i in range(len(header)) if i not in (src_idx, dst_idx)]
        # Sichtbare Spalte bleibt unverändert (dein Wunsch), wir füllen sie weiterhin leer:
        out_rows.append([quelle, ziel, "", sim_display, *rest])

    out_df = pd.DataFrame(out_rows, columns=out_header)

    # Coloring by adjusted score (simple buckets) – als Spalte
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
    st.session_state.out_df = out_df  # -> persistieren
    st.dataframe(out_df, use_container_width=True, hide_index=True)

    csv2 = out_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download 'Potenziell zu entfernende Links' (CSV)",
        data=csv2,
        file_name="potenziell_zu_entfernende_links.csv",
        mime="text/csv",
    )

    # Am Ende der Berechnungen:
    if run_clicked:
        try:
            placeholder.empty()
        except Exception:
            pass
        st.success("✅ Berechnung abgeschlossen!")
        st.session_state.ready = True

# =========================================================
# Analyse 3 (gemergt): Gems + priorisierte Link-Empfehlungen (Similarity ⨉ Dringlichkeit)
# =========================================================
st.markdown("---")
st.subheader("Analyse 3: Starke Linkgeber („Gems“) & priorisierte Link-Empfehlungen")

with st.expander("ℹ️ Erklärung & Priorisierungslogik", expanded=False):
    st.markdown("""
**Ziel:** Für jede starke Linkgeber-Seite (**Gem**) die besten Ziel-URLs finden – thematisch passend **und** mit **hoher Dringlichkeit**.

**Gems** bestimmen wir über den **Linkpotenzial-Score** (aus ILS, PageRank-Horder, Backlinks, Ref. Domains).  
**Dringlichkeit** („PRIO“) setzt sich aus folgenden Signalen (∈ [0,1]) zusammen:

- **Opportunity (Opp)** = `MinMax(log1p(Impressions)) × (1 − CTR)`  
  → viel Sichtbarkeit, aber (noch) wenig Klicks ⇒ hohes Potenzial.
- **Low-ILS-High-Demand (LIHD)** = `(1 − ILS_norm) × Demand_norm`
  → viel Nachfrage, aber intern unterversorgt.
- **Inlinks-Defizit (similarity-gewichtet)**  
  → Anteil der **nahen** Quellen (Related), die **noch nicht** auf die Ziel-URL verlinken.
- **Ranking-Sweet-Spot** (Standard **8–20**):  
  URLs mit durchschnittlicher Position im Bereich **8–20** werden bevorzugt (einstellbar).
- **Orphan/Thin-Linking:**  
  Orphan = keine Inlinks; Thin = Inlinks ≤ K.

**Ranking je (Gem, Ziel):**  
`Rank = α · Similarity + (1 − α) · PRIO(target)`  
(β/Opportunity entfällt, weil Opp schon in **PRIO** steckt.)
""")

# --------------------------
# UI: Gems + Zielanzahl
# --------------------------
gem_pct = st.slider(
    "Anteil starker Linkgeber (Top-X %)",
    1, 30, 10, step=1,
    help="Welche obersten X % nach Linkpotenzial gelten als „Gems“?"
)
max_targets_per_gem = st.number_input(
    "Top-Ziele je Gem (Anzahl Spalten)",
    min_value=1, max_value=50, value=10, step=1,
    help="Wie viele Ziel-URLs pro Gem in der Breiten-Tabelle gezeigt werden."
)

# --------------------------
# UI: Gewichtung der Dringlichkeits-Signale (PRIO)
# --------------------------
st.markdown("#### Gewichtung Dringlichkeit (PRIO)")
colA, colB, colC = st.columns(3)
with colA:
    w_opp   = st.slider("Gewicht: Opportunity",     0.0, 1.0, 0.50, 0.05)
    w_lihd  = st.slider("Gewicht: LIHD",            0.0, 1.0, 0.30, 0.05)
with colB:
    w_def   = st.slider("Gewicht: Inlinks-Defizit", 0.0, 1.0, 0.20, 0.05)
    w_rank  = st.slider("Gewicht: Ranking 8–20",    0.0, 1.0, 0.20, 0.05,
                        help="Bevorzugt Seiten mit durchschnittlicher Position im eingestellten Bereich (unten).")
with colC:
    w_orph  = st.slider("Gewicht: Orphan/Thin",     0.0, 1.0, 0.10, 0.05)
    thin_k  = st.slider("Thin-Schwelle K (Inlinks ≤ K)", 0, 10, 2, 1)

# Ranking-Sweet-Spot (8–20) inkl. weichem Rand
rank_minmax = st.slider(
    "Ranking-Sweet-Spot (Positionen)", 1, 50, (8, 20), 1,
    help="Bereich der durchschnittlichen Position, der als „sweet spot“ gilt (Standard 8–20)."
)
rank_falloff = st.slider(
    "Weicher Rand (Falloff, ± Positionen)", 0, 10, 2, 1,
    help="Außerhalb des Sweet-Spot fällt das Gewicht innerhalb dieses Randes linear auf 0 ab."
)

# --------------------------
# UI: Ranking-Mischung (Similarity vs. PRIO)
# --------------------------
alpha = st.slider(
    "Balance α: Similarity ↔ Dringlichkeit (PRIO)",
    0.0, 1.0, 0.6, 0.05,
    help="α höher = stärker nach semantischer Nähe sortieren; α niedriger = stärker nach Dringlichkeit (PRIO)."
)

# --------------------------
# Optional: GSC-Datei laden (URL, Impressions, Clicks[, Position])
# --------------------------
gsc_up = st.file_uploader(
    "Optional: GSC-Daten (CSV/Excel) – Spalten: URL, Impressions, Clicks, [Position]",
    type=["csv", "xlsx"], key="gsc_up_merged"
)

# =========================================================
# Daten holen (aus früheren Schritten)
# =========================================================
res1_df: Optional[pd.DataFrame] = st.session_state.get("res1_df")
source_potential_map: Dict[str, float] = st.session_state.get("_source_potential_map", {})
metrics_map: Dict[str, Dict[str, float]] = st.session_state.get("_metrics_map", {})
norm_ranges: Dict[str, Tuple[float, float]] = st.session_state.get("_norm_ranges", {})
all_links: set = st.session_state.get("_all_links", set())

# Gems bestimmen
if source_potential_map:
    sorted_sources = sorted(source_potential_map.items(), key=lambda x: x[1], reverse=True)
    cutoff_idx = max(1, int(len(sorted_sources) * gem_pct / 100))
    gems = [u for u, _ in sorted_sources[:cutoff_idx]]
else:
    gems = []

# =========================================================
# GSC verarbeiten → Opp/Demand/Position ermitteln
# (robuste Spaltenerkennung)
# =========================================================
opp_map: Dict[str, float] = {}
demand_map: Dict[str, float] = {}
pos_map: Dict[str, float] = {}

if gsc_up is not None:
    gsc_df = read_any_file(gsc_up)
    if gsc_df is not None and not gsc_df.empty:
        df = gsc_df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # Spalten indices heuristisch
        def _find_col(cands: List[str], default_idx: Optional[int]=None) -> Optional[int]:
            low = [c.lower() for c in df.columns]
            for k in cands:
                if k in low:
                    return low.index(k)
            return default_idx

        url_idx = 0  # URL = erste Spalte, falls nicht anders gefunden
        impr_idx = _find_col(
            ["impressions", "impr", "anzeigen", "impressionen"], default_idx=1 if df.shape[1] > 1 else 0
        )
        clk_idx = _find_col(
            ["clicks", "klicks", "click"], default_idx=2 if df.shape[1] > 2 else None
        )
        pos_idx = _find_col(
            ["position", "avg position", "average position", "ranking", "durchschnittliche position", "rang"],
            default_idx=None
        )

        # URLs normieren
        df.iloc[:, url_idx] = df.iloc[:, url_idx].astype(str).map(normalize_url)
        urls_series = df.iloc[:, url_idx]

        # Impressions/Clicks robust casten
        impr = pd.to_numeric(df.iloc[:, impr_idx], errors="coerce").fillna(0) if impr_idx is not None else pd.Series([0]*len(df))
        clicks = pd.to_numeric(df.iloc[:, clk_idx],  errors="coerce").fillna(0) if clk_idx is not None else pd.Series([0]*len(df))

        # Demand = Min-Max von log1p(Impressions)
        log_impr = np.log1p(impr)
        if (log_impr.max() - log_impr.min()) > 0:
            demand_norm = (log_impr - log_impr.min()) / (log_impr.max() - log_impr.min())
        else:
            demand_norm = np.zeros_like(log_impr)

        ctr = clicks / (impr + 1)
        opp_vals = demand_norm * (1 - ctr)

        for u, d, o in zip(urls_series, demand_norm, opp_vals):
            if u:
                demand_map[str(u)] = float(d)
                opp_map[str(u)] = float(o)

        # Position (optional)
        if pos_idx is not None:
            pos_series = pd.to_numeric(df.iloc[:, pos_idx], errors="coerce")
            for u, p in zip(urls_series, pos_series):
                if pd.notna(p) and str(u):
                    pos_map[str(u)] = float(p)

        # Für ggf. spätere Nutzung merken
        st.session_state["__gsc_df_raw__"] = df.copy()

# =========================================================
# Hilfsfunktionen für PRIO-Signale
# =========================================================
from collections import defaultdict

# Inbound-Counts für Orphan/Thin
inbound_count = defaultdict(int)
for s, t in all_links:
    inbound_count[t] += 1

min_ils, max_ils = norm_ranges.get("ils", (0.0, 1.0))

def ils_norm_for(u: str) -> float:
    m = metrics_map.get(u)
    if not m:
        return 0.0
    x = float(m.get("score", 0.0))
    if max_ils > min_ils:
        return float(np.clip((x - min_ils) / (max_ils - min_ils), 0.0, 1.0))
    return 0.0

def lihd_for(u: str) -> float:
    if u not in demand_map:
        return 0.0
    return float((1.0 - ils_norm_for(u)) * demand_map[u])

def deficit_weighted_for(target: str) -> float:
    if not isinstance(res1_df, pd.DataFrame):
        return 0.0
    row = res1_df.loc[res1_df["Ziel-URL"] == target]
    if row.empty:
        return 0.0
    r = row.iloc[0]
    sum_all, sum_missing, i = 0.0, 0.0, 1
    while True:
        col_sim = f"Ähnlichkeit {i}"
        col_src = f"Related URL {i}"
        col_any = f"überhaupt verlinkt {i}?"
        if col_sim not in res1_df.columns or col_src not in res1_df.columns:
            break
        sim_val = r.get(col_sim, np.nan)
        src_val = normalize_url(r.get(col_src, ""))
        if pd.isna(sim_val) or not src_val:
            i += 1
            continue
        try:
            simf = float(sim_val)
        except Exception:
            simf = 0.0
        sum_all += max(0.0, simf)
        anywhere = str(r.get(col_any, "nein")).strip().lower()
        if anywhere != "ja":
            sum_missing += max(0.0, simf)
        i += 1
    if sum_all <= 0:
        return 0.0
    return float(np.clip(sum_missing / sum_all, 0.0, 1.0))

def rank_sweetspot_for(u: str, lo: int, hi: int, falloff: int) -> float:
    """Gewicht 1.0 im [lo,hi]; außerhalb linearer Abfall über 'falloff' Positionen auf 0."""
    p = pos_map.get(u)
    if p is None:
        return 0.0
    # im Sweet-Spot
    if lo <= p <= hi:
        return 1.0
    # links vom Spot
    if falloff > 0 and (lo - falloff) <= p < lo:
        return float(1.0 - (lo - p) / falloff)
    # rechts vom Spot
    if falloff > 0 and hi < p <= (hi + falloff):
        return float(1.0 - (p - hi) / falloff)
    return 0.0

def orphan_score_for(u: str, k: int) -> float:
    inl = int(inbound_count.get(u, 0))
    orphan = 1.0 if inl == 0 else 0.0
    thin   = 1.0 if inl <= k else 0.0
    # leichtes Gewicht für Thin; Vollgewicht für Orphan
    return float(max(orphan, 0.5 * thin))

# =========================================================
# PRIO je Ziel berechnen und cachen
# =========================================================
target_priority_map: Dict[str, float] = {}

if isinstance(res1_df, pd.DataFrame) and not res1_df.empty:
    for _, row in res1_df.iterrows():
        u = normalize_url(row["Ziel-URL"])
        if not u:
            continue

        opp  = float(opp_map.get(u, 0.0))
        lihd = lihd_for(u)
        ddef = deficit_weighted_for(u)
        rnk  = rank_sweetspot_for(u, lo=rank_minmax[0], hi=rank_minmax[1], falloff=rank_falloff)
        orph = orphan_score_for(u, thin_k)

        weights = np.array([w_opp, w_lihd, w_def, w_rank, w_orph], dtype=float)
        comps   = np.array([opp,   lihd,  ddef,  rnk,   orph  ], dtype=float)
        denom = weights.sum()
        prio = float((weights @ comps) / denom) if denom > 0 else 0.0

        target_priority_map[u] = prio

# =========================================================
# Empfehlungen pro Gem bauen (Rank = α·Sim + (1−α)·PRIO(target))
# =========================================================
gem_rows: List[List] = []
if isinstance(res1_df, pd.DataFrame) and not res1_df.empty and gems:
    for gem in gems:
        for _, row in res1_df.iterrows():
            target = normalize_url(row["Ziel-URL"])
            i = 1
            while True:
                col_src = f"Related URL {i}"
                col_sim = f"Ähnlichkeit {i}"
                col_any = f"überhaupt verlinkt {i}?"
                if col_src not in res1_df.columns:
                    break
                src = normalize_url(row.get(col_src, ""))
                if not src or src != gem:
                    i += 1
                    continue
                simv = row.get(col_sim, 0.0)
                try:
                    simf = float(simv)
                except Exception:
                    simf = 0.0
                prio_t = float(target_priority_map.get(target, 0.0))
                rank_score = alpha * simf + (1.0 - alpha) * prio_t
                exists = str(row.get(col_any, "nein"))
                gem_rows.append([gem, target, simf, prio_t, rank_score, exists])
                i += 1

# Pro Gem top-N schneiden & sortieren (nach Rank, dann Similarity)
if gem_rows:
    import itertools
    gem_rows.sort(key=lambda r: r[0])
    final_rows: List[List] = []
    for gem_key, group in itertools.groupby(gem_rows, key=lambda r: r[0]):
        grp = list(group)
        grp = sorted(grp, key=lambda r: (r[4], r[2]), reverse=True)[:int(max_targets_per_gem)]
        final_rows.extend(grp)
    gem_rows = final_rows

# =========================================================
# Ausgabe: breite Tabelle wie Analyse 1
# =========================================================
if gem_rows:
    from collections import defaultdict
    by_gem: Dict[str, List[Tuple[str, float, float, float, str]]] = defaultdict(list)
    for gem, target, simv, prio_t, rankv, exists in gem_rows:
        by_gem[gem].append((target, float(simv), float(prio_t), float(rankv), str(exists)))

    # Spaltenkopf
    cols = ["Gem (Quelle)", "Linkpotenzial (Quelle)"]
    for i in range(1, int(max_targets_per_gem) + 1):
        cols += [
            f"Ziel {i}",
            f"Similarity {i}",
            f"PRIO {i}",
            f"Rank {i}",
            f"Bereits verlinkt {i}?",
        ]

    def pot_for(g: str) -> float:
        return float(st.session_state.get("_source_potential_map", {}).get(normalize_url(g), 0.0))

    ordered_gems = sorted(by_gem.keys(), key=pot_for, reverse=True)
    rows = []
    for gem in ordered_gems:
        items = sorted(by_gem[gem], key=lambda t: (t[3], t[1]), reverse=True)[: int(max_targets_per_gem)]
        row = [gem, round(pot_for(gem), 3)]
        for i in range(int(max_targets_per_gem)):
            if i < len(items):
                target, simv, prio_t, rankv, exists = items[i]
                row += [target, round(simv, 3), round(prio_t, 3), round(rankv, 3), exists]
            else:
                row += [np.nan, np.nan, np.nan, np.nan, ""]
        rows.append(row)

    gem_wide_df = pd.DataFrame(rows, columns=cols)
    st.dataframe(gem_wide_df, use_container_width=True, hide_index=True)

    csv_gem = gem_wide_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download 'Gems & Empfehlungen (breit)' (CSV)",
        data=csv_gem,
        file_name="gems_empfehlungen_breit.csv",
        mime="text/csv",
    )
else:
    st.caption("Keine Gems-Empfehlungen gefunden – prüfe GSC-Upload/Signale oder Gem-Perzentil.")



