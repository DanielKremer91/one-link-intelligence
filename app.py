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

    backend = st.radio(
        "Matching-Backend",
        ["Exakt (NumPy)", "Schnell (FAISS)"],
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
    placeholder.image(
        "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExNDJweGExcHhhOWZneTZwcnAxZ211OWJienY5cWQ1YmpwaHR0MzlydiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/dBRaPog8yxFWU/giphy.gif",
        caption="Die Berechnungen laufen – Zeit für eine kleine Stärkung, bevor es losgeht …",
        use_container_width=True
    )

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
            row.append("")
        rows.append(row)

    res1_df = pd.DataFrame(rows, columns=cols)
    st.session_state.res1_df = res1_df  # -> persistieren
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
# NEU: Gems & Empfehlungen
# =========================================================
st.markdown("---")
st.subheader("Analyse 3: Gems & Empfehlungen (Optional)")

with st.expander("ℹ️ Erklärung Gems & Opportunity-Score", expanded=False):
    st.markdown("""
**Gems (Top-X % Linkgeber):**  
Die stärksten Linkgeberseiten nach Linkpotenzial. Standard: oberste 10 %, einstellbar bis max. 30 %.  

**Opportunity-Ranking (Optional mit GSC):**  
Wir kombinieren Similarity (thematische Nähe) mit einer *Opportunity-Bewertung* aus GSC-Daten:  

- CTR = Clicks / (Impressions + 1)  
- normierte Impressions (Log + Min-Max)  
- opp = norm(Impressions) × (1 − CTR)  

Der Gesamtscore lautet:  
`rank_score = α × Similarity + β × opp`  

👉 Default: α=0.6, β=0.4. Wer’s einfacher mag, setzt β=0 und nutzt nur Similarity.
""")

# Steuerung: Perzentil-Slider
gem_pct = st.slider("Top-X % als Gems auswählen", 1, 30, 10, step=1,
                    help="Die obersten X % nach Linkpotenzial werden als Gems genutzt. Maximal 30 %.")

# Optional: GSC-Datei laden
gsc_up = st.file_uploader("Optional: GSC-Daten (CSV/Excel)", type=["csv", "xlsx"], key="gsc_up")
alpha = st.slider("α Gewicht für Similarity", 0.0, 1.0, 0.6, 0.05)
beta = st.slider("β Gewicht für Opportunity", 0.0, 1.0, 0.4, 0.05)
min_impr = st.number_input("Min. Impressions (für Opportunity-Berechnung)", 0, 10000, 100, step=50)

# Gems bestimmen
source_potential_map = st.session_state.get("_source_potential_map", {})
if source_potential_map:
    sorted_sources = sorted(source_potential_map.items(), key=lambda x: x[1], reverse=True)
    cutoff_idx = max(1, int(len(sorted_sources) * gem_pct / 100))
    gems = [u for u, _ in sorted_sources[:cutoff_idx]]
else:
    gems = []

# GSC verarbeiten (falls hochgeladen)
gsc_map = {}
if gsc_up is not None:
    gsc_df = read_any_file(gsc_up)
    if gsc_df is not None and not gsc_df.empty and gsc_df.shape[1] >= 3:
        # Erste Spalte: URL, dann Impressions, Clicks
        gsc_df.iloc[:, 0] = gsc_df.iloc[:, 0].astype(str)
        urls_norm = gsc_df.iloc[:, 0].map(normalize_url)
        impressions = pd.to_numeric(gsc_df.iloc[:, 1], errors="coerce").fillna(0)
        clicks = pd.to_numeric(gsc_df.iloc[:, 2], errors="coerce").fillna(0)

        # Log + Min-Max Normalisierung
        log_impr = np.log1p(impressions)
        if log_impr.max() > log_impr.min():
            norm_impr = (log_impr - log_impr.min()) / (log_impr.max() - log_impr.min())
        else:
            norm_impr = np.zeros_like(log_impr)

        ctr = clicks / (impressions + 1)
        opp = norm_impr * (1 - ctr)

        for u, o, imp in zip(urls_norm, opp, impressions):
            if u and imp >= min_impr:
                gsc_map[u] = float(o)

# Empfehlungen pro Gem bauen
res1 = st.session_state.get("res1_df")
gem_rows = []
if isinstance(res1, pd.DataFrame) and not res1.empty and gems:
    for gem in gems:
        # finde alle Zeilen, wo gem als Quelle auftaucht
        for _, row in res1.iterrows():
            target = normalize_url(row["Ziel-URL"])
            for i in range(1, int(max_related) + 1):
                src = normalize_url(row.get(f"Related URL {i}", ""))
                if not src or src != gem:
                    continue
                simv = row.get(f"Ähnlichkeit {i}", 0.0)
                potv = row.get(f"Linkpotenzial {i}", 0.0)
                exists = row.get(f"überhaupt verlinkt {i}?", "nein")

                oppv = gsc_map.get(target, 0.0)
                rank_score = alpha * float(simv) + beta * float(oppv)

                gem_rows.append([gem, target, simv, potv, oppv, rank_score, exists])

# Ausgabe-Tabelle
if gem_rows:
    gem_df = pd.DataFrame(gem_rows, columns=[
        "Gem (Quelle)", "Ziel", "Similarity", "Linkpotenzial (Quelle)",
        "Opportunity (Ziel)", "Rank-Score", "Bereits verlinkt?"
    ])
    # sortieren nach Rank-Score
    gem_df = gem_df.sort_values("Rank-Score", ascending=False)
    st.dataframe(gem_df, use_container_width=True, hide_index=True)

    csv_gem = gem_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download 'Gems & Empfehlungen' (CSV)",
        data=csv_gem,
        file_name="gems_empfehlungen.csv",
        mime="text/csv",
    )
else:
    st.caption("Keine Gems-Empfehlungen gefunden – ggf. GSC-Daten hochladen oder Perzentil anpassen.")
# =========================================================
# Visualisierung (nachgelagert, optional & performant)
# =========================================================
st.markdown("---")
st.subheader("Optional: Visualisierung interne Verlinkung (Vorher / Zukunft)")

# Nur anbieten, wenn erste Analysen gelaufen sind und wir Embeddings haben:
can_visualize = st.session_state.ready and ("_emb_urls" in st.session_state) and ("_emb_matrix" in st.session_state)

if not can_visualize:
    st.caption("ℹ️ Die Visualisierung steht zur Verfügung, wenn die Analysen gelaufen sind **und** Embeddings hochgeladen wurden.")
else:
    # Opt-in, damit initial keine Kosten entstehen
    enable_viz = st.checkbox("Visualisierung aktivieren", value=False,
                             help="Die Graph-Visualisierung wird erst berechnet, wenn du diese Option aktivierst.")
    if enable_viz:
        # Lazy imports für Plotly/Sklearn/UMAP
        import plotly.graph_objects as go

        # UMAP optional
        HAVE_UMAP = False
        try:
            import umap  # type: ignore
            HAVE_UMAP = True
        except Exception:
            HAVE_UMAP = False

        # --- UI: Konfiguration ---
        st.markdown("#### Anzeige & Filter")

        colA, colB, colC = st.columns(3)
        with colA:
            layout_algo = st.radio(
                "2D-Layout",
                ["t-SNE", "UMAP (empfohlen bei vielen URLs)"] if HAVE_UMAP else ["t-SNE"],
                horizontal=False,
                help="Wähle das Reduktionsverfahren für die 2D-Positionen. Beide liefern vergleichbare Cluster."
            )
        with colB:
            only_content_links = st.checkbox(
                "IST: nur Content-Links zeigen",
                value=False,  # False = Header/Footer/Sidebar sind initial dabei (dein Wunsch)
                help="Wenn aktiviert, werden nur Links aus dem Inhaltsbereich (Content) für den IST-Graphen gezeigt."
            )
        with colC:
            future_only = st.checkbox(
                "Nur Änderungen (Zukunfts-Graph)",
                value=True,
                help="Blendet IST-Links aus. Zeigt nur neue empfohlene Links in grün. Entfernte Links werden nicht dargestellt."
            )

        st.markdown("#### Empfohlene neue Links (Selektion)")

        colN, colM, colG = st.columns(3)
        with colN:
            max_new_total = st.slider(
                "Max. neue Links gesamt (N)", 10, 1000, 30, step=10,
                help="Globale Obergrenze neuer Links in der Visualisierung. Hält den Graph übersichtlich."
            )
        with colM:
            max_new_per_gem = st.slider(
                "Max. neue Links pro Gem (M)", 1, 50, 3, step=1,
                help="Begrenzt Empfehlungen pro starkem Linkgeber. Verhindert Übergewicht einzelner Seiten."
            )
        with colG:
            gem_pct_vis = st.slider(
                "Gems-Perzentil (Top-X %)", 1, 30, int(st.session_state.get("last_gem_pct", 10)), step=1,
                help="Welche Top-% nach Linkpotenzial werden als starke Linkgeber betrachtet?",
            )
            # speichere für UX
            st.session_state["last_gem_pct"] = gem_pct_vis

        st.markdown("#### Ranking-Methode")
        rank_mode = st.radio(
            "Wie sollen neue Links priorisiert werden?",
            ["Nur Similarity", "Similarity + Opportunity (GSC)"],
            horizontal=True,
            help="Wenn GSC-Daten geladen wurden, kannst du Similarity mit Opportunity (Impr & CTR) kombinieren."
        )
        if rank_mode == "Nur Similarity":
            alpha_viz, beta_viz = 1.0, 0.0
        else:
            alpha_viz = alpha
            beta_viz = beta

        st.markdown("#### Bubble-Größe")
        bubble_mode = st.radio(
            "Knotengröße",
            ["Konstant", "Linkpotenzial", "Performance-Spalte (optional)"],
            index=1,
            help="Empfehlung: Linkpotenzial. Alternativ konstant oder eine numerische Spalte aus einer Datei."
        )

        perf_up = None
        perf_df = None
        perf_numeric_cols = []
        if bubble_mode == "Performance-Spalte (optional)":
            perf_up = st.file_uploader(
                "Performance-Datei für Bubble-Skalierung (CSV/Excel)",
                type=["csv", "xlsx", "xlsm", "xls"],
                key="viz_perf2"
            )
            if perf_up is not None:
                perf_df = read_any_file(perf_up)
                if perf_df is not None and not perf_df.empty:
                    # Finde numerische Spalten robust
                    for c in perf_df.columns:
                        if c is None:
                            continue
                        s = pd.to_numeric(pd.Series(perf_df[c]), errors="coerce")
                        if s.notna().mean() > 0.6 and s.nunique(dropna=True) > 5:
                            perf_numeric_cols.append(c)
        if bubble_mode == "Performance-Spalte (optional)" and perf_numeric_cols:
            size_by = st.selectbox("Bubble-Größe nach Spalte", perf_numeric_cols, index=0)
            size_min = st.slider("Min-Blasengröße (px)", 2, 20, 4)
            size_max = st.slider("Max-Blasengröße (px)", 6, 40, 14)
        else:
            size_by = None
            size_min, size_max = 6, 14

        # Interaktive Suche
        st.markdown("#### Suche & Highlight")
        search_q = st.text_input("URL-Suche (Teilstring reicht)", value="", help="Treffer-URL wird hervorgehoben. Nachbarn (ein-/ausgehend) werden abgesetzt eingefärbt, Rest grau.")

        # --- Daten vorbereiten ---
        urls = st.session_state["_emb_urls"]
        X = st.session_state["_emb_matrix"]  # float32, L2-normalisiert
        idx_by_url = st.session_state.get("_emb_index_by_url", {u: i for i, u in enumerate(urls)})

        # IST-Kanten (nur Content, wenn gewünscht)
        all_links = st.session_state.get("_all_links", set())
        content_links = st.session_state.get("_content_links", set())
        ist_edges = content_links if only_content_links else all_links

        # Kanten, die in Zukunft entfernt werden (aus Analyse 2)
        removed_edges = set()
        out_df = st.session_state.get("out_df")
        if isinstance(out_df, pd.DataFrame) and not out_df.empty:
            for _, rr in out_df.iterrows():
                q = normalize_url(rr["Quelle"])
                z = normalize_url(rr["Ziel"])
                if q and z:
                    removed_edges.add((q, z))

        # SOLL-Kandidaten aus res1_df + Gems
        source_potential_map = st.session_state.get("_source_potential_map", {})
        res1 = st.session_state.get("res1_df")

        # Gems für Visualisierung errechnen (unabhängig vom Tab "Analyse 3", damit vis standalone ist)
        if source_potential_map:
            sorted_sources = sorted(source_potential_map.items(), key=lambda x: x[1], reverse=True)
            cutoff_idx = max(1, int(len(sorted_sources) * gem_pct_vis / 100))
            gems_vis = {u for u, _ in sorted_sources[:cutoff_idx]}
        else:
            gems_vis = set()

        # GSC-Opp aus Analyse 3 (falls geladen)
        gsc_map = {}
        if "gsc_up" in st.session_state:  # nicht zuverlässig – fallback: aus gem_df nicht trivial -> belassen
            pass  # noop
        # Wir können die schon berechnete Map aus Analyse 3 nicht sicher speichern => nutzen ggf. opp=0

        # Vorschlagskandidaten (Quelle in Gems, Link existiert noch nicht, Ziel erreichbar)
        suggested_edges = []  # (source, target, sim, potential, opp, rank)
        if isinstance(res1, pd.DataFrame) and not res1.empty and gems_vis:
            for _, row in res1.iterrows():
                target = normalize_url(row["Ziel-URL"])
                for i in range(1, int(max_related) + 1):
                    col_src = f"Related URL {i}"
                    col_sim = f"Ähnlichkeit {i}"
                    col_any = f"überhaupt verlinkt {i}?"
                    col_pot = f"Linkpotenzial {i}"
                    if col_src not in res1.columns:
                        continue
                    src = normalize_url(row.get(col_src, ""))
                    if not src or src not in gems_vis:
                        continue
                    if (src, target) in all_links:
                        continue  # existiert bereits
                    try:
                        simv = float(row.get(col_sim, 0.0))
                    except Exception:
                        simv = 0.0
                    try:
                        potv = float(row.get(col_pot, 0.0))
                    except Exception:
                        potv = 0.0
                    oppv = gsc_map.get(target, 0.0)
                    rank_score = alpha_viz * simv + beta_viz * oppv
                    suggested_edges.append((src, target, simv, potv, oppv, rank_score))

        # Sortieren & limitieren: erst pro Gem (M), dann global (N)
        final_suggested = []
        if suggested_edges:
            # Gruppieren nach Quelle (Gem)
            from collections import defaultdict
            by_gem = defaultdict(list)
            for s, t, simv, potv, oppv, rnk in suggested_edges:
                by_gem[s].append((s, t, simv, potv, oppv, rnk))
            # pro Gem sortieren & cutten
            per_gem_lists = []
            for g, lst in by_gem.items():
                lst_sorted = sorted(lst, key=lambda x: (x[5], x[2]), reverse=True)  # rank, dann similarity
                per_gem_lists.extend(lst_sorted[:max_new_per_gem])
            # global sortieren & cutten
            final_suggested = sorted(per_gem_lists, key=lambda x: (x[5], x[2]), reverse=True)[:max_new_total]

        # Bubble-Size vorbereiten
        node_sizes = np.full(len(urls), float(size_min), dtype=float)
        if bubble_mode == "Linkpotenzial":
            for i, u in enumerate(urls):
                node_sizes[i] = size_min + (size_max - size_min) * source_potential_map.get(normalize_url(u), 0.0)
        elif bubble_mode == "Performance-Spalte (optional)" and perf_df is not None and size_by is not None:
            perf_df_local = perf_df.copy()
            perf_df_local["__join"] = perf_df_local.iloc[:, 0].astype(str).apply(normalize_url)
            perf_df_local["__val"] = pd.to_numeric(perf_df_local[size_by], errors="coerce")
            perf_df_local = perf_df_local.dropna(subset=["__val"])
            mvals: Dict[str, float] = {normalize_url(r["__join"]): float(r["__val"]) for _, r in perf_df_local.iterrows() if isinstance(r["__join"], str)}
            vals = np.array([mvals.get(normalize_url(u), np.nan) for u in urls], dtype=float)
            valid = ~np.isnan(vals)
            if valid.any():
                v = vals[valid]
                lo, hi = np.percentile(v, 5), np.percentile(v, 95)
                v = np.clip(v, lo, hi)
                if hi > lo:
                    v = (v - lo) / (hi - lo)
                else:
                    v = np.zeros_like(v)
                v = size_min + v * (size_max - size_min)
                node_sizes[valid] = v

        # 2D-Layout berechnen (nur einmal pro Einstellung)
        @st.cache_data(show_spinner=False)
        def compute_layout(X_in: np.ndarray, algo: str):
            from sklearn.manifold import TSNE
            if algo.startswith("UMAP"):
                import umap  # type: ignore
                reducer = umap.UMAP(
                    n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42
                )
                Y = reducer.fit_transform(X_in)
                return Y
            else:
                n = X_in.shape[0]
                perplexity = max(5, min(30, n - 1))
                tsne = TSNE(
                    n_components=2,
                    metric="euclidean",
                    method="barnes_hut",
                    init="pca",
                    learning_rate="auto",
                    n_iter=600,
                    random_state=42,
                    perplexity=perplexity
                )
                Y = tsne.fit_transform(X_in)
                return Y

        Y = compute_layout(X, layout_algo)
        xs, ys = Y[:, 0], Y[:, 1]

        # Helper: baue Linien-Trace
        def build_edge_trace(edge_list, color, width=1.0, opacity=0.25, cap=10000):
            Xs, Ys = [], []
            cnt = 0
            for (a, b) in edge_list:
                if cnt >= cap:
                    break
                ia = idx_by_url.get(a)
                ib = idx_by_url.get(b)
                if ia is None or ib is None:
                    continue
                Xs += [xs[ia], xs[ib], None]
                Ys += [ys[ia], ys[ib], None]
                cnt += 1
            return go.Scattergl(
                x=Xs, y=Ys,
                mode="lines",
                line=dict(color=color, width=width),
                opacity=opacity,
                hoverinfo="skip",
                showlegend=False
            )

        # Kanten für Anzeige vorbereiten
        ist_pairs = list(ist_edges)
        # Entfernte Kanten im Zukunftsmodus nicht anzeigen (werden ohnehin durch future_only ausgeblendet)
        if future_only:
            ist_pairs = []

        # Neue Kanten (grün)
        new_pairs = [(s, t) for (s, t, *_rest) in final_suggested]

        # Suche / Highlight bestimmen
        highlight_nodes = set()
        neighbor_nodes = set()
        if search_q:
            # Match per Teilstring auf normalisierten URLs
            candidates = [u for u in urls if search_q.lower() in normalize_url(u).lower()]
            if candidates:
                # nimm den ersten besten Treffer (oder erweitern zu Multiselect, falls gewünscht)
                focus = normalize_url(candidates[0])
                highlight_nodes.add(focus)
                # Nachbarn: eingehend/ausgehend aus IST + neuen Kanten
                for (a, b) in list(ist_edges) + new_pairs:
                    if a == focus:
                        neighbor_nodes.add(b)
                    if b == focus:
                        neighbor_nodes.add(a)

        # Basis-Node-Trace (grau, wenn Suche aktiv)
        base_colors = []
        base_sizes = []
        for u in urls:
            nu = normalize_url(u)
            if search_q:
                if nu in highlight_nodes:
                    base_colors.append("#ff8c00")  # Fokus (orange)
                elif nu in neighbor_nodes:
                    base_colors.append("#ffd08a")  # Nachbarn (heller)
                else:
                    base_colors.append("#cfcfcf")  # grau
            else:
                base_colors.append("#4F8EF7")
            base_sizes.append(node_sizes[idx_by_url.get(nu, 0)])

        node_trace = go.Scattergl(
            x=xs, y=ys,
            mode="markers",
            marker=dict(
                size=base_sizes,
                color=base_colors,
                line=dict(width=0.5, color="white"),
                opacity=0.9 if search_q else 0.8
            ),
            text=urls,
            hovertemplate="%{text}",
            name="Seiten"
        )

        # Edge-Traces
        ist_trace = build_edge_trace(ist_pairs, color="lightgray", width=1.0, opacity=0.25, cap=10000)
        new_trace = build_edge_trace(new_pairs, color="#2ecc71", width=2.0, opacity=0.7, cap=max_new_total)  # grün

        # Figur aufbauen
        fig = go.Figure()
        if not future_only and len(ist_pairs) > 0:
            fig.add_trace(ist_trace)
        if len(new_pairs) > 0:
            fig.add_trace(new_trace)
        fig.add_trace(node_trace)
        fig.update_layout(
            title="Interne Verlinkung – Zukunftsszenario" if future_only else "Interne Verlinkung – IST + empfohlene neue Links",
            template="plotly_white",
            height=780,
            margin=dict(l=10, r=10, t=50, b=10),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Export
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        fname = "interne_verlinkung_zukunft.html" if future_only else "interne_verlinkung_ist_plus_neu.html"
        st.download_button(
            "📥 HTML-Export der Visualisierung",
            data=html_bytes,
            file_name=fname,
            mime="text/html"
        )

        # Sanfte Hinweise bei Limits
        if len(ist_edges) > 10000 and not future_only:
            st.info("Hinweis: IST-Kanten wurden auf 10.000 begrenzt, um die Darstellung performant zu halten.")
        if len(new_pairs) >= max_new_total:
            st.info(f"Hinweis: Neue Links wurden auf N={max_new_total} begrenzt. Passe N/M an, um mehr/anders zu sehen.")
        # Kleine KPI-Zusammenfassung zur Einordnung
        with st.container():
            c1, c2, c3 = st.columns(3)
            c1.metric("Gems (Quelle)", f"{len(gems_vis)}")
            c2.metric("Neue Links (gezeigt)", f"{len(new_pairs)}", help="Anzahl grüner Kanten im aktuellen Setting.")
            c3.metric("IST-Kanten einbezogen", f"{0 if future_only else min(len(ist_edges), 10000)}")

        # Einsteiger-Hinweise
        with st.expander("❓ Hinweise für Einsteiger (Ranking & Opportunity)", expanded=False):
            st.markdown("""
**Ranking-Modi:**  
- *Nur Similarity*: sortiert nur nach thematischer Nähe (empfohlen, wenn keine GSC-Daten).  
- *Similarity + Opportunity (GSC)*: kombiniert Nähe mit Chancen aus GSC.  
  - CTR = Clicks / (Impressions + 1)  
  - normierte Impressions = log1p(Impressions) → Min-Max  
  - **Opportunity** = norm(Impressions) × (1 − CTR)  
  - **Rank** = α × Similarity + β × Opportunity  
  - Größere α → stärker thematisch, größere β → stärker auf „viel Impressions, niedrige CTR“.

**Gems (Top-X %):**  
Starke Linkgeber. Nur diese Quellen dürfen neue Links setzen. Reduziert Rauschen & fokussiert Maßnahmen.

**Graph bleibt handlich durch:**  
- Limit *N* (gesamt) und *M* (pro Gem),  
- „Nur Änderungen“ (blendet IST aus),  
- Suche mit Nachbarschafts-Highlight (alles andere wird grau).
""")
