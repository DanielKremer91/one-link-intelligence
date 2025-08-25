import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ===============================
# Page config & Branding
# ===============================
st.set_page_config(page_title="ONE Link Intelligence", layout="wide")

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
    """CSV/Excel robust lesen: probiert mehrere Encodings; snifft Delimiter (Komma/Semikolon)."""
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
                    # Fallback auf Standard-Trenner (mit gleichem Encoding)
                    try:
                        f.seek(0)
                        return pd.read_csv(
                            f, sep=";", engine="python", encoding=enc, on_bad_lines="skip"
                        )
                    except Exception:
                        f.seek(0)
                        return pd.read_csv(
                            f, sep=",", engine="python", encoding=enc, on_bad_lines="skip"
                        )
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
  - **URL** (Spaltenname: z. B. `URL`, `Adresse`, `Address`, `Page`, `Seite`)  
  - **Embeddings** (Spaltenname: z. B. `Embedding`, `Embeddings`, `Vector`). Werte können als JSON-Array (`[0.1, 0.2, ...]`) oder durch Komma/Leerzeichen/`;`/`|` getrennt vorliegen.  

  Zusätzlich erforderlich:  
  - **All Inlinks** (CSV/Excel, aus Screaming Frog: *Massenexport → Links → Alle Inlinks*) — enthält mindestens: **Quelle/Source**, **Ziel/Destination**, optional **Linkposition/Link Position**  
  - **Linkmetriken** (CSV/Excel) — **erste 4 Spalten** in dieser Reihenfolge: **URL**, **Score**, **Inlinks**, **Outlinks**  
  - **Backlinks** (CSV/Excel) — **erste 3 Spalten** in dieser Reihenfolge: **URL**, **Backlinks**, **Referring Domains**  

- **Option 2: Related URLs**  
  Tabelle mit mindestens drei Spalten:  
  - **Ziel-URL**, **Quell-URL**, **Ähnlichkeitswert (0–1)** (z. B. aus Screaming Frog *Massenexport → Inhalt → Semantisch ähnlich*).  

Hinweis: Spaltenerkennung ist tolerant gegenüber deutsch/englischen Varianten.  
Trennzeichen (Komma/Semikolon) und Encodings (UTF-8/UTF-8-SIG/Windows-1252/Latin-1) werden automatisch erkannt.  
URLs werden kanonisiert (Protokoll ergänzt, `www.` entfernt, Tracking-Parameter entfernt, Pfade vereinheitlicht).
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
Der Wert ist **relativ** – er zeigt im Verhältnis zu den anderen Vorschlägen, wie lukrativ ein Link wäre.  
Je **höher** der Score im Vergleich zu den übrigen, desto sinnvoller ist die Verlinkung.
""")

    st.markdown("""
### 🚫 Unpassende Links

Die zweite Analyse identifiziert interne Links, die thematisch nicht passen oder potenziell „SEO-Power verschwenden“.  
- Maßstab: **Semantische Ähnlichkeit** (unterhalb der gewählten Unähnlichkeitsschwelle).  
- Zusätzlich fließt ein vereinfachter *PageRank-Waster-Wert* ein (viele Outlinks, wenige Inlinks → Kandidat).  
""")

    st.markdown("""
### 📤 Output (Ergebnisse)

1. **Interne Verlinkungsmöglichkeiten** — vorgeschlagene interne Links inkl. Linkpotenzial **und Ähnlichkeitswert**.  
2. **Potenziell zu entfernende Links** — bestehende Links, die thematisch unpassend sind oder von „PageRank-Wastern“ ausgehen.  

Beide Ergebnisse sind als CSV downloadbar.
""")

    st.markdown(
        """
<div style="margin-top: 0.5rem; background:#fff8e6; border:1px solid #ffd28a; border-radius:8px; padding:10px 12px; color:#000;">
  <strong>❗WICHTIG:</strong> Achte beim Export aus Screaming Frog / Excel auf echte Spaltentrenner (Komma/Semikolon). 
  Falls du Ein-Spalten-CSVs bekommst, als <em>UTF-8</em> oder <em>Windows-1252 (Latin-1)</em> neu speichern. 
  Die App kann beides lesen, überspringt aber defekte Zeilen.
</div>
""",
        unsafe_allow_html=True,
    )

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

# etwas CSS für den roten Button
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
        "Lade eine Datei mit **URL** und **Embedding** (JSON-Array oder Zahlen, getrennt durch Komma/Whitespace/`;`/`|`). "
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

if not run_clicked:
    st.info("Bitte Dateien hochladen und auf **Let's Go** klicken, um die Analysen zu starten.")
    st.stop()

# GIF anzeigen, solange gerechnet wird
placeholder = st.empty()
placeholder.image(
    "https://media.giphy.com/media/.../giphy.gif",
    caption="Berechnungen laufen – wir geben Gas …",
    use_container_width=True
)

# ===============================
# Validierung & ggf. Ableitung Related aus Embeddings
# ===============================
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
    u = normalize_url(r.iloc[0])
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
    source = normalize_url(r.iloc[src_idx])
    target = normalize_url(r.iloc[dst_idx])
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
    urlA = normalize_url(r.iloc[0])
    urlB = normalize_url(r.iloc[1])
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
        key = f"{source}→{target}"
        anywhere = "ja" if key in all_links else "nein"
        from_content = "ja" if key in content_links else "nein"

        m = metrics_map.get(source, {"score": 0.0, "prDiff": 0.0})
        ils_raw = float(m.get("score", 0.0))
        pr_raw = float(m.get("prDiff", 0.0))

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

        row.extend([source, round(float(sim), 3), anywhere, from_content, final_score])

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
    sim_map[f"{a}→{b}"] = sim
    sim_map[f"{b}→{a}"] = sim
    processed_pairs2.add(pair_key)

# PageRank-Waster-ähnlicher Rohwert und backlink-adjusted Score
raw_score_map: Dict[str, float] = {}
for _, r in metrics_df.iterrows():
    u = normalize_url(r.iloc[0])
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
    quelle = normalize_url(r.iloc[src_idx])
    ziel = normalize_url(r.iloc[dst_idx])
    if not quelle or not ziel:
        continue

    k1 = f"{quelle}→{ziel}"
    k2 = f"{ziel}→{quelle}"
    sim = sim_map.get(k1, sim_map.get(k2, np.nan))

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
st.dataframe(out_df, use_container_width=True, hide_index=True)

csv2 = out_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download 'Potenziell zu entfernende Links' (CSV)",
    data=csv2,
    file_name="potenziell_zu_entfernende_links.csv",
    mime="text/csv",
)

# Am Ende der Berechnungen:
placeholder.empty()
st.success("✅ Berechnung abgeschlossen!")
