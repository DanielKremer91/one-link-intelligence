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

def _num(x, default: float = 0.0) -> float:
    """Robuste Numerik: NaN/None sicher auf Default."""
    v = pd.to_numeric(x, errors="coerce")
    return default if pd.isna(v) else float(v)

def _safe_minmax(lo, hi) -> Tuple[float, float]:
    """Sichere Min/Max-Ranges (verhindert +/-inf oder hi<=lo)."""
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) and hi > lo else (0.0, 1.0)

def robust_range(values, lo_q: float = 0.05, hi_q: float = 0.95) -> Tuple[float, float]:
    """Robuste Spannweite über Quantile (z. B. 5–95%). Fällt zurück auf min/max, wenn nötig."""
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
    """Normierung auf 0..1 innerhalb [lo, hi] (clip), robust gegen Ausreißer."""
    if not np.isfinite(x) or hi <= lo:
        return 0.0
    v = (float(x) - lo) / (hi - lo)
    return float(np.clip(v, 0.0, 1.0))


def find_column_index(header: List[str], possible_names: List[str]) -> int:
    lower = [str(h).strip().lower() for h in header]
    for i, h in enumerate(lower):
        if h in possible_names:
            return i
    return -1

def normalize_url(u: str) -> str:
    """URL-Kanonisierung: Protokoll ergänzen, Tracking-Parameter entfernen, Query sortieren,
    Trailing Slash normalisieren, www. entfernen (nur intern für Keys)."""
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

# --- Anzeige-Originale merken/anzeigen ---
_ORIG_MAP: Dict[str, str] = {}  # key: canonical -> original (bevorzugt mit Slash, falls verfügbar)

def remember_original(raw: str) -> str:
    """Merkt sich die original eingegebene URL-Form für die Anzeige, liefert den kanonischen Key zurück."""
    s = str(raw or "").strip()
    if not s:
        return ""
    key = normalize_url(s)
    if not key:
        return ""
    prev = _ORIG_MAP.get(key)
    if prev is None:
        _ORIG_MAP[key] = s
    else:
        # Bevorzuge eine Version MIT trailing slash in der Anzeige
        if (not prev.endswith("/")) and s.endswith("/"):
            _ORIG_MAP[key] = s
    return key

def disp(key_or_url: str) -> str:
    """Gibt die gemerkte Original-URL-Form zur Anzeige zurück (Fallback: Eingabe)."""
    return _ORIG_MAP.get(str(key_or_url), str(key_or_url))

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

# --- Robuster Datei-Leser mit Encoding- und Delimiter-Erkennung ---
def read_any_file(f) -> Optional[pd.DataFrame]:
    """CSV/Excel robust lesen: probiert mehrere Encodings; snifft Delimiter (Komma/Semikolon/Tab)."""
    if f is None:
        return None
    name = (getattr(f, "name", "") or "").lower()
    try:
        if name.endswith(".csv"):
            for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
                try:
                    f.seek(0)
                    return pd.read_csv(
                        f,
                        sep=None,
                        engine="python",
                        encoding=enc,
                        on_bad_lines="skip",
                    )
                except UnicodeDecodeError:
                    continue
                except Exception:
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
) -> pd.DataFrame:
    """
    Baut 'Related URLs' (Ziel, Quelle, Similarity).
    Erwartet L2-normalisierte Vektoren V.
    Nutzt FAISS (Inner Product) falls gewählt & verfügbar, sonst exaktes NumPy-Chunking.
    """
    n = V.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["Ziel", "Quelle", "Similarity"])

    K = int(top_k)
    pairs: List[List[object]] = []

    # ---- Versuch: FAISS ----
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
            backend = "Exakt (NumPy)"  # Fallback

    # ---- Exakt (NumPy) mit 2D-Chunking ----
    if backend == "Exakt (NumPy)":
        Vf = V.astype(np.float32, copy=False)
        row_chunk = 512
        col_chunk = 2048

        for r0 in range(0, n, row_chunk):
            r1 = min(r0 + row_chunk, n)
            block = Vf[r0:r1]   # (b, d)
            b = r1 - r0

            top_vals = np.full((b, K), -np.inf, dtype=np.float32)
            top_idx  = np.full((b, K), -1,      dtype=np.int32)

            for c0 in range(0, n, col_chunk):
                c1 = min(c0 + col_chunk, n)
                part = Vf[c0:c1]             # (c, d)
                sims = block @ part.T        # (b, c)

                # Self-Matches maskieren
                if (c0 < r1) and (c1 > r0):
                    o0 = max(r0, c0); o1 = min(r1, c1)
                    br = np.arange(o0 - r0, o1 - r0)
                    bc = np.arange(o0 - c0, o1 - c0)
                    sims[br, bc] = -1.0

                # lokale Top-K
                if K < sims.shape[1]:
                    part_idx = np.argpartition(sims, -K, axis=1)[:, -K:]
                else:
                    part_idx = np.argsort(sims, axis=1)
                rows = np.arange(b)[:, None]
                cand_vals = sims[rows, part_idx]
                cand_idx  = part_idx + c0

                # mit globalen Top-K mergen
                all_vals = np.concatenate([top_vals, cand_vals], axis=1)
                all_idx  = np.concatenate([top_idx,  cand_idx],  axis=1)
                sel = np.argpartition(all_vals, -K, axis=1)[:, -K:]
                top_vals = all_vals[rows, sel]
                top_idx  = all_idx[rows,  sel]

                order = np.argsort(top_vals, axis=1)[:, ::-1]
                top_vals = top_vals[rows, order]
                top_idx  = top_idx[rows,  order]

            # Ergebnisse einsammeln
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

    # --- Rückgabe ---
    if not pairs:
        return pd.DataFrame(columns=["Ziel", "Quelle", "Similarity"])
    return pd.DataFrame(pairs, columns=["Ziel", "Quelle", "Similarity"])


# ======== Caching-Helper ========

@st.cache_data(show_spinner=False)
def read_any_file_cached(filename: str, raw: bytes) -> Optional[pd.DataFrame]:
    """
    Gleiches Verhalten wie read_any_file(), aber cachebar:
    Cache-Key = (filename, raw-bytes).
    """
    from io import BytesIO
    if not raw:
        return None
    name = (filename or "").lower()
    try:
        if name.endswith(".csv"):
            # Versuche mehrere Encodings; bei Fehlern zusätzlich feste Separatoren
            for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
                try:
                    return pd.read_csv(
                        BytesIO(raw),
                        sep=None,
                        engine="python",
                        encoding=enc,
                        on_bad_lines="skip",
                    )
                except UnicodeDecodeError:
                    continue
                except Exception:
                    for sep_try in [";", ",", "\t"]:
                        try:
                            return pd.read_csv(
                                BytesIO(raw),
                                sep=sep_try,
                                engine="python",
                                encoding=enc,
                                on_bad_lines="skip",
                            )
                        except Exception:
                            continue
            raise ValueError("Kein passendes Encoding/Trennzeichen gefunden.")
        else:
            return pd.read_excel(BytesIO(raw))
    except Exception as e:
        st.error(f"Fehler beim Lesen von {filename or 'Datei'}: {e}")
        return None

# ===== Backend-Diagnose & Auto-Switch =====
def _faiss_available() -> bool:
    try:
        import faiss  # type: ignore
        return True
    except Exception:
        return False

def _numpy_footprint_gb(n: int) -> float:
    """Grober RAM-Bedarf für volle NxN-Ähnlichkeitsmatrix in float64."""
    return (n * n * 8) / (1024**3)

def choose_backend(prefer: str, n_items: int, mem_budget_gb: float = 1.5) -> Tuple[str, str]:
    """
    Liefert (effective_backend, reason).
    prefer: "Exakt (NumPy)" oder "Schnell (FAISS)".
    Schaltet bei Bedarf um (RAM-Budget / FAISS-Verfügbarkeit).
    """
    faiss_ok = _faiss_available()
    if prefer == "Schnell (FAISS)":
        if faiss_ok:
            return "Schnell (FAISS)", "FAISS gewählt"
        else:
            return "Exakt (NumPy)", "FAISS nicht installiert → NumPy"

    # prefer NumPy
    est = _numpy_footprint_gb(max(0, int(n_items)))
    if est > mem_budget_gb and faiss_ok:
        return "Schnell (FAISS)", f"NumPy-Schätzung {est:.2f} GB > Budget {mem_budget_gb:.2f} GB → FAISS"
    return "Exakt (NumPy)", "NumPy innerhalb Budget"

@st.cache_data(show_spinner=False)
def build_related_cached(
    urls: tuple,
    V: np.ndarray,
    top_k: int,
    sim_threshold: float,
    backend: str,
    _v: int = 1,
) -> pd.DataFrame:
    Vf = V.astype("float32", copy=False)
    return build_related_from_embeddings(
        urls=list(urls),
        V=Vf,
        top_k=top_k,
        sim_threshold=sim_threshold,
        backend=backend,
    )

def build_related_auto(
    urls: List[str],
    V: np.ndarray,
    top_k: int,
    sim_threshold: float,
    prefer_backend: str,
    mem_budget_gb: float = 1.5,
) -> pd.DataFrame:
    """Wählt Backend automatisch und fällt robust um (NumPy<->FAISS)."""
    n = int(V.shape[0])
    eff_backend, reason = choose_backend(prefer_backend, n, mem_budget_gb)
    if eff_backend != prefer_backend:
        st.info(f"Backend auf **{eff_backend}** umgestellt ({reason}).")

    try:
        return build_related_cached(tuple(urls), V, int(top_k), float(sim_threshold), eff_backend, _v=1)
    except MemoryError:
        # NumPy am Limit? -> auf FAISS wechseln
        if eff_backend == "Exakt (NumPy)" and _faiss_available():
            st.warning("NumPy ist am Speicherlimit – Wechsel auf **FAISS**.")
            return build_related_cached(tuple(urls), V, int(top_k), float(sim_threshold), "Schnell (FAISS)", _v=1)
        raise
    except Exception as e:
        # FAISS zickt? -> Fallback auf NumPy
        if eff_backend == "Schnell (FAISS)":
            st.warning(f"FAISS-Indexierung fehlgeschlagen ({e}). Fallback auf **NumPy**.")
            return build_related_cached(tuple(urls), V, int(top_k), float(sim_threshold), "Exakt (NumPy)", _v=1)
        else:
            raise


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
  - **All Inlinks** (CSV/Excel, aus Screaming Frog: *Massenexport → Links → Alle Inlinks*) — enthält mindestens: **Quelle/Source**, **Ziel/Destination**, optional **Linkposition/Link Position**. Weitere Spalten sind nicht erforderlich. **Tipp**: Lösche alle Linktypen, die keine Hyperlinks sind und die anderen Spalten vor dem Upload heraus, um eine geringere Dateigröße zu erzielen.
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

# ===============================
# Sidebar Controls (mit Tooltips)
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
    st.caption(
        "Das Linkpotenzial gibt Aufschluss über die Lukrativität der **Quell-URL** als Linkgeber. "
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
        help="Externe verweisende Domains der Quell-URL."
    )
    w_bl = st.slider(
        "Backlinks",
        0.0, 1.0, 0.15, 0.01,
        help="Externe Backlinks der Quell-URL."
    )
    w_sum = w_ils + w_pr + w_rd + w_bl
    if not math.isclose(w_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        st.warning(f"Gewichtungs-Summe = {w_sum:.2f} (sollte 1.0 sein)")

    st.subheader("Schwellenwerte & Limits (Related URLs Ermittlung)")
    sim_threshold = st.slider(
        "Ähnlichkeitsschwelle",
        0.0, 1.0, 0.80, 0.01,
        help="Nur URL-Paare mit Cosine Similarity ≥ diesem Wert gelten als „related“."
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
CSS_ACTION_BUTTONS = """
<style>
/* ONE Red */
:root {
  --one-red: #e02424;
  --one-red-hover: #c81e1e;
}

/* Standard-Buttons (z. B. Let's Go mit kind="secondary") */
div.stButton > button[kind="secondary"] {
  background-color: var(--one-red) !important;
  color: #fff !important;
  border: 1px solid var(--one-red) !important;
  border-radius: 6px !important;
}
div.stButton > button[kind="secondary"]:hover {
  background-color: var(--one-red-hover) !important;
  border-color: var(--one-red-hover) !important;
}

/* Download-Buttons – Streamlit rendert je nach Version <button> oder <a> */
div.stDownloadButton > button,
div.stDownloadButton > a {
  background-color: var(--one-red) !important;
  color: #fff !important;
  border: 1px solid var(--one-red) !important;
  border-radius: 6px !important;
  text-decoration: none !important;
}
div.stDownloadButton > button:hover,
div.stDownloadButton > a:hover {
  background-color: var(--one-red-hover) !important;
  border-color: var(--one-red-hover) !important;
  color: #fff !important;
}
</style>
"""
st.markdown(CSS_ACTION_BUTTONS, unsafe_allow_html=True)


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

    up_emb = st.file_uploader(
        "URLs + Embeddings (CSV/Excel)",
        type=["csv", "xlsx", "xlsm", "xls"],
        key="embs",
        help="""CSV oder Excel. Mindestanforderungen:
- Spalten: **URL** und **Embedding** (Reihenfolge egal).
- URL-Spalte wird erkannt über Header: `url`, `urls`, `page`, `seite`, `adresse`, `address`; sonst 1. Spalte.
- Embedding-Spalte wird erkannt über Header: `embedding`, `embeddings`, `vector`, `embedding_json`, `vec`; sonst 2. Spalte.
- Embedding-Werte: JSON-Array wie `[0.12, -0.34, ...]` **oder** Zahlen getrennt durch Komma/Leerzeichen/`;`/`|`."""
    )

    col1, col2 = st.columns(2)
    with col1:
        inlinks_up = st.file_uploader(
            "All Inlinks (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="inl2",
            help="""CSV oder Excel. Reihenfolge egal, es zählen die **Headernamen**:
- Quelle/Source: `quelle`, `source`, `from`, `origin`, `quell-url`
- Ziel/Destination: `ziel`, `destination`, `to`, `target`, `ziel-url`
- Linkposition: `linkposition`, `link position`, `position`
Fehlt *Linkposition*, gilt „aus Inhalt heraus verlinkt?“ = **„nein“** für alle."""
        )
        metrics_up = st.file_uploader(
            "Linkmetriken (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="met2",
            help="""CSV oder Excel. **Erste 4 Spalten in genau dieser Reihenfolge**:
1) **URL**, 2) **Score** (Interner Link Score), 3) **Inlinks**, 4) **Outlinks**.
Weitere Spalten werden ignoriert. Zahlen dürfen `,` oder `.` enthalten."""
        )
    with col2:
        backlinks_up = st.file_uploader(
            "Backlinks (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="bl2",
            help="""CSV oder Excel. **Erste 3 Spalten in genau dieser Reihenfolge**:
1) **URL**, 2) **Backlinks**, 3) **Referring Domains**.
Weitere Spalten werden ignoriert."""
        )

    # Dateien erst NACH den with-Blöcken einlesen (immer noch im if-Block!)
    emb_df       = read_any_file_cached(getattr(up_emb,       "name", ""), up_emb.getvalue())         if up_emb       else None
    inlinks_df   = read_any_file_cached(getattr(inlinks_up,   "name", ""), inlinks_up.getvalue())     if inlinks_up   else None
    metrics_df   = read_any_file_cached(getattr(metrics_up,   "name", ""), metrics_up.getvalue())     if metrics_up   else None
    backlinks_df = read_any_file_cached(getattr(backlinks_up, "name", ""), backlinks_up.getvalue())   if backlinks_up else None


elif mode == "Related URLs":
    st.write(
        "Lade die vier Tabellen: **Related URLs**, **All Inlinks**, **Linkmetriken**, **Backlinks** "
        "(CSV/Excel; Trennzeichen & Encodings werden automatisch erkannt)."
    )

    col1, col2 = st.columns(2)
    with col1:
        related_up = st.file_uploader(
            "Related URLs (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="rel",
            help="""CSV oder Excel. **Genau 3 Spalten in dieser Reihenfolge**:
1) **Ziel-URL**, 2) **Quell-URL**, 3) **Similarity (0–1)**.
Spaltennamen sind egal – es zählt die Position. Similarity darf `,` oder `.` als Dezimaltrenner haben."""
        )
        metrics_up = st.file_uploader(
            "Linkmetriken (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="met",
            help="""CSV oder Excel. **Erste 4 Spalten in genau dieser Reihenfolge**:
1) **URL**, 2) **Score** (Interner Link Score), 3) **Inlinks**, 4) **Outlinks**.
Weitere Spalten werden ignoriert. Zahlen dürfen `,` oder `.` enthalten."""
        )
    with col2:
        inlinks_up = st.file_uploader(
            "All Inlinks (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="inl",
            help="""CSV oder Excel. Reihenfolge egal, es zählen die **Headernamen**:
- Quelle/Source: `quelle`, `source`, `from`, `origin`, `quell-url`
- Ziel/Destination: `ziel`, `destination`, `to`, `target`, `ziel-url`
- (optional) Linkposition: `linkposition`, `link position`, `position`
Fehlt *Linkposition*, gilt „aus Inhalt heraus verlinkt?“ = **„nein“** für alle."""
        )
        backlinks_up = st.file_uploader(
            "Backlinks (CSV/Excel)",
            type=["csv", "xlsx", "xlsm", "xls"],
            key="bl",
            help="""CSV oder Excel. **Erste 3 Spalten in genau dieser Reihenfolge**:
1) **URL**, 2) **Backlinks**, 3) **Referring Domains**.
Weitere Spalten werden ignoriert."""
        )

    # Dateien einlesen (immer noch im elif-Block!)
    related_df   = read_any_file_cached(getattr(related_up,  "name", ""), related_up.getvalue())    if related_up  else None
    inlinks_df   = read_any_file_cached(getattr(inlinks_up,  "name", ""), inlinks_up.getvalue())    if inlinks_up  else None
    metrics_df   = read_any_file_cached(getattr(metrics_up,  "name", ""), metrics_up.getvalue())    if metrics_up  else None
    backlinks_df = read_any_file_cached(getattr(backlinks_up,"name", ""), backlinks_up.getvalue())  if backlinks_up else None



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
                nkey = remember_original(r[url_col])
                v = parse_vec(r[emb_col])
                if not nkey or v is None:
                    continue
                urls.append(nkey)  # kanonischer Key
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
            V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

            if shorter > 0:
                st.caption(f"⚠️ {shorter} Embeddings hatten geringere Dimensionen und wurden auf {max_dim} gepaddet.")

            related_df = build_related_auto(
                urls=list(urls),
                V=V,  # castet intern auf float32
                top_k=int(max_related),
                sim_threshold=float(sim_threshold),
                prefer_backend=backend,   # "Exakt (NumPy)" oder "Schnell (FAISS)"
                mem_budget_gb=1.5,        # optional
            )



            # Embeddings & URLs im Session-State ablegen
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

    if metrics_df.shape[1] < 4:
        st.error("'Linkmetriken' braucht mindestens 4 Spalten: URL, Score, Inlinks, Outlinks (in dieser Reihenfolge).")
        st.stop()

    metrics_df.iloc[:, 0] = metrics_df.iloc[:, 0].astype(str)

    metrics_map: Dict[str, Dict[str, float]] = {}
    for _, r in metrics_df.iterrows():
        u = remember_original(r.iloc[0])
        if not u:
            continue
        score = _num(r.iloc[1])
        inlinks = _num(r.iloc[2])
        outlinks = _num(r.iloc[3])
        prdiff = inlinks - outlinks
        metrics_map[u] = {"score": score, "prDiff": prdiff}


    # Backlinks map (URL -> backlinks, referring domains)
    backlinks_df = backlinks_df.copy()
    if backlinks_df.shape[1] < 3:
        st.error("'Backlinks' braucht mindestens 3 Spalten: URL, Backlinks, Referring Domains (in dieser Reihenfolge).")
        st.stop()

    backlink_map: Dict[str, Dict[str, float]] = {}
    for _, r in backlinks_df.iterrows():
        u = remember_original(r.iloc[0])
        if not u:
            continue
        bl = _num(r.iloc[1])
        rd = _num(r.iloc[2])
        backlink_map[u] = {"backlinks": bl, "referringDomains": rd}


    # --- Robuste Ranges (Quantile) ---
    ils_vals = [m["score"] for m in metrics_map.values()]
    prd_vals = [m["prDiff"] for m in metrics_map.values()]
    bl_vals  = [b["backlinks"] for b in backlink_map.values()]
    rd_vals  = [b["referringDomains"] for b in backlink_map.values()]

    min_ils, max_ils = robust_range(ils_vals, 0.05, 0.95)
    min_prd, max_prd = robust_range(prd_vals, 0.05, 0.95)
    min_bl,  max_bl  = robust_range(bl_vals,  0.05, 0.95)
    min_rd,  max_rd  = robust_range(rd_vals,  0.05, 0.95)

    # --- Offpage: Log-Skalen + robuste Ranges für die Dämpfung ---
    bl_log_vals = [float(np.log1p(max(0.0, v))) for v in bl_vals]
    rd_log_vals = [float(np.log1p(max(0.0, v))) for v in rd_vals]
    lo_bl_log, hi_bl_log = robust_range(bl_log_vals, 0.05, 0.95)
    lo_rd_log, hi_rd_log = robust_range(rd_log_vals, 0.05, 0.95)


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

    # Fallback: Embeddings aus Analyse 1, um fehlende Similarities direkt zu berechnen
    _idx_map = st.session_state.get("_emb_index_by_url")
    _Vmat    = st.session_state.get("_emb_matrix")
    _has_emb = isinstance(_idx_map, dict) and isinstance(_Vmat, np.ndarray)

    for _, r in inlinks_df.iterrows():
        source = remember_original(r.iloc[src_idx])
        target = remember_original(r.iloc[dst_idx])
        if not source or not target:
            continue
        key = (source, target)
        all_links.add(key)
        if pos_idx != -1 and is_content_position(r.iloc[pos_idx]):
            content_links.add(key)

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
        urlA = remember_original(r.iloc[0])  # Ziel (Key + Original merken)
        urlB = remember_original(r.iloc[1])  # Quelle (Key + Original merken)
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

        norm_ils = robust_norm(ils_raw, min_ils, max_ils)
        norm_pr  = robust_norm(pr_raw,  min_prd, max_prd)
        norm_bl  = robust_norm(bl_raw,  min_bl,  max_bl)   # hier OHNE Log, bewusst linear
        norm_rd  = robust_norm(rd_raw,  min_rd,  max_rd)   # hier OHNE Log, bewusst linear


        final_score = (w_ils * norm_ils) + (w_pr * norm_pr) + (w_bl * norm_bl) + (w_rd * norm_rd)
        source_potential_map[u] = round(final_score, 4)

    st.session_state["_source_potential_map"] = source_potential_map
    st.session_state["_metrics_map"] = metrics_map
    st.session_state["_backlink_map"] = backlink_map
    st.session_state["_norm_ranges"] = {
        # Robuste (Quantil-)Ranges für lineare Normierungen
        "ils": (min_ils, max_ils),
        "prd": (min_prd, max_prd),
        "bl":  (min_bl,  max_bl),
        "rd":  (min_rd,  max_rd),
        # Zusätzlich: Log-basierte robuste Ranges NUR für die Offpage-Dämpfung
        "bl_log": (lo_bl_log, hi_bl_log),
        "rd_log": (lo_rd_log, hi_rd_log),
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

    rows_norm = []
    rows_view = []

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

        # pad
        while len(row_norm) < len(cols):
            row_norm.append(np.nan)
        while len(row_view) < len(cols):
            row_view.append(np.nan)

        rows_norm.append(row_norm)
        rows_view.append(row_view)

    # interne (kanonische) DF für spätere Berechnungen
    res1_df = pd.DataFrame(rows_norm, columns=cols)
    st.session_state.res1_df = res1_df

    # Anzeige-DF
    res1_view_df = pd.DataFrame(rows_view, columns=cols)
    sim_cols = [c for c in res1_view_df.columns if c.startswith("Ähnlichkeit ")]
    for c in sim_cols:
        res1_view_df[c] = pd.to_numeric(res1_view_df[c], errors="coerce")

    st.dataframe(res1_view_df, use_container_width=True, hide_index=True)

    csv1 = res1_view_df.to_csv(index=False).encode("utf-8-sig")
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

    # 1) Similarity-Map aus Related (beidseitig)
    sim_map: Dict[Tuple[str, str], float] = {}
    processed_pairs2 = set()

    for _, r in related_df.iterrows():
        a = remember_original(r.iloc[1])  # Quelle
        b = remember_original(r.iloc[0])  # Ziel
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

    # 2) Fehlende Similarities für ALLE Inlinks nachrechnen (nur wenn Embeddings da)
    if _has_emb:
        for (src, dst) in all_links:
            if (src, dst) in sim_map or (dst, src) in sim_map:
                continue
            i = _idx_map.get(src); j = _idx_map.get(dst)
            if i is None or j is None:
                continue
            # Cosine = Dot-Product (L2-normalisiert)
            sim_val = float(np.dot(_Vmat[i], _Vmat[j]))
            sim_map[(src, dst)] = sim_val
            sim_map[(dst, src)] = sim_val

    # 3) Waster-Score (Quelle) + Klassifikation
    raw_score_map: Dict[str, float] = {}
    for _, r in metrics_df.iterrows():
        u = remember_original(r.iloc[0])
        inl = _num(r.iloc[2])
        outl = _num(r.iloc[3])
        # „Waster“: viele Outlinks, wenig Inlinks
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
        q70 = float(np.quantile(w_vals, 0.70))   # ab hier „mittel“
        q90 = float(np.quantile(w_vals, 0.90))   # ab hier „hoch“

    def waster_class_for(u: str) -> Tuple[str, float]:
        score = float(adjusted_score_map.get(u, 0.0))
        if score >= q90:
            return "hoch", score
        elif score >= q70:
            return "mittel", score
        else:
            return "niedrig", score

    # 4) Output bauen (Anzeige mit Original-URLs)
    out_rows = []
    rest_cols = [c for i, c in enumerate(header) if i not in (src_idx, dst_idx)]
    out_header = [
        "Quelle",
        "Ziel",
        "Waster-Klasse (Quelle)",
        "Waster-Score (Quelle)",
        "Semantische Ähnlichkeit",
        *rest_cols,
    ]

    for _, r in inlinks_df.iterrows():
        quelle = remember_original(r.iloc[src_idx])
        ziel   = remember_original(r.iloc[dst_idx])
        if not quelle or not ziel:
            continue

        # Waster-Klasse/-Score für die QUELL-URL
        w_class, w_score = waster_class_for(normalize_url(quelle))

        # Similarity aus Map (beidseitig) …
        sim = sim_map.get((quelle, ziel), sim_map.get((ziel, quelle), np.nan))

        # … sonst optional on-the-fly aus Embeddings berechnen
        if not (isinstance(sim, (int, float)) and np.isfinite(sim)) and _has_emb:
            i = _idx_map.get(normalize_url(quelle))
            j = _idx_map.get(normalize_url(ziel))
            if i is not None and j is not None:
                sim = float(np.dot(_Vmat[i], _Vmat[j]))  # Cosine (L2-normalisiert)

        # Filter/Anzeige:
        # - numerisch: nur Links <= not_similar_threshold in den „zu entfernen“-Report
        # - nicht vorhanden: mit Hinweistext ausgeben
        if isinstance(sim, (int, float)) and np.isfinite(sim):
            sim_display = round(float(sim), 3)
            if float(sim) > float(not_similar_threshold):
                continue  # stark genug -> nicht entfernen
        else:
            sim_display = (
                "Cosine Similarity nicht erfasst / URL-Paar nicht im Input-Dokument vorhanden"
                if mode == "Related URLs" else
                "Cosine Similarity nicht erfasst"
            )

        rest = [r.iloc[i] for i in range(len(header)) if i not in (src_idx, dst_idx)]
        out_rows.append([
            disp(quelle),
            disp(ziel),
            w_class,
            round(float(w_score), 3),
            sim_display,
            *rest,
        ])

    out_df = pd.DataFrame(out_rows, columns=out_header)
    st.session_state.out_df = out_df
    st.dataframe(out_df, use_container_width=True, hide_index=True)

    csv2 = out_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download 'Potenziell zu entfernende Links' (CSV)",
        data=csv2,
        file_name="potenziell_zu_entfernende_links.csv",
        mime="text/csv",
    )

    # kleines Aufräumen am Ende des Runs
    if run_clicked:
        try:
            placeholder.empty()
        except Exception:
            pass
        st.success("✅ Berechnung abgeschlossen!")
        st.session_state.ready = True


# =========================================================
# Analyse 3: Gems & „Cheat-Sheet der internen Verlinkung“ (Similarity × PRIO, ohne Opportunity)
# =========================================================
st.markdown("---")
st.subheader("Analyse 3 (optional): Was sind starke Linkgeber („Gems“) & welche URLs diese verlinken sollten")

with st.expander("Erklärung: Wie werden Gems & Zielseiten bestimmt?", expanded=False):
    st.markdown('''
## Wie werden Gems & Zielseiten bestimmt?

**Kurz & klar:**

1) **Gems identifizieren**  
   Wir bewerten alle Seiten nach dem **Linkpotenzial-Score** (Mix aus *Interner Link Score*, *PageRank-Horder*, *Backlinks*, *Ref. Domains*).  
   Die obersten *X %* (Slider **„Anteil starker Linkgeber“**) gelten als **Gems** = beste internen Linkgeber.

2) **Kandidaten-Ziele finden**  
   Für jede Gem-URL nehmen wir Ziel-URLs aus **Analyse 1**, bei denen die Gem-URL als **Related URL** auftaucht **und** es **noch keinen Content-Link** vom Gem → Ziel gibt.  
   Die inhaltliche Nähe steuert die **Ähnlichkeitsschwelle** in der Seitenleiste.

3) **Linkbedarf (PRIO) berechnen**  
   Jede Ziel-URL erhält einen **PRIO-Wert** aus vier Signalen (Gewichte per Slider; Summe muss nicht 1 sein — wir normalisieren intern):
   - **Hidden Champtions** *(Search Console Daten nötig)*  
     Formel: `Hidden Champions = (1 − ILS_norm) × Demand_norm × (1 − β × Offpage_norm)`  
     Bedeutet: viel Such-Nachfrage (Search Console Impressions) + schwacher interner Link-Score ⇒ höherer Linkbedarf.  
     **Offpage-Dämpfung** (standardmäßig aktiv): viele Backlinks/Ref. Domains reduzieren den Linkbedarf einer URL; Stärke des Einflusses von Offpage-Signalen über **Regler** steuerbar.
   - **Semantische Linklücke**  
     Anteil semantisch ähnlicher URLs, die **noch nicht aus dem Content** heraus auf eine andere thematisch nah verwandte URL verlinken.  
     **Similarity** dient als Gewicht: je ähnlicher sich Seiten sind, desto wichtiger der fehlende Link.  
     Ebenfalls **Offpage-gedämpft**: starke Offpage-Signale des **Ziels** reduzieren den Linkbedarf.
   - **Ranking Sprungbrett-URLs** *(Search Console Position nötig)*  
     Bonus für URLs, deren **durchschnittliche Position** im eingestellten **Sweet-Spot (z. B. 8–20)** liegt.
     Die Erfahrung zeigt, durch gezielte Optimierungsmaßnahmen (z.B. Verbesserung der internen Verlinkung) können sog.**Sprungbrett-URLs** schneller bessere Rankings erreichen als wenn eine Seite von Position 50 auf die erste Ergebnisseite gehoben werden soll.
   - **Mauerblümchen**  
     Gar nicht (Orphan) oder nur sehr schwach (Thin) intern verlinkt. **Orphan** = 0 interne Inlinks. **Thin** = Inlinks ≤ **K** (Slider **„Thin-Schwelle“**). Analyse ezieht sich rein auf Links aus dem **Content**.

4) **Ausgabe & Reihenfolge der Empfehlungen**  
   Pro Gem listen wir die **Top-Z Ziele** (Slider **„Top-Ziele je Gem“**).  
   Die Reihenfolge steuerst du über **„Reihenfolge der Empfehlungen“**:
   - **Mix (inhaltliche Nähe + Linkbedarf)** → Sortwert `= α · Similarity + (1 − α) · PRIO`  
     Den Mix justierst du mit **„Gewichtung: inhaltliche Nähe vs. Linkbedarf“ (α)**.
   - **Nur Linkbedarf** → sortiert nach PRIO.  
   - **Nur inhaltliche Nähe** → sortiert nach Similarity.

**Hinweise:**
- **Search Console Daten erforderlich** für Hidden Champions (Impressions) und Ranking Sprungbrett-URLs (Position).  
- **Offpage-Dämpfung** betrifft **Hidden Champions** und **Semantische Linklücke** und ist standardmäßig **aktiv** (abschaltbar).   
- Alle Normalisierungen (z. B. ILS, Impressions, Offpage-Signale) erfolgen **relativ zu deinen hochgeladenen Daten**.
''')


# --------------------------
# Eingänge / Session aus Analyse 1+2 (werden für die Berechnung gebraucht)
# --------------------------
res1_df: Optional[pd.DataFrame] = st.session_state.get("res1_df")
source_potential_map: Dict[str, float] = st.session_state.get("_source_potential_map", {})
metrics_map: Dict[str, Dict[str, float]] = st.session_state.get("_metrics_map", {})
norm_ranges: Dict[str, Tuple[float, float]] = st.session_state.get("_norm_ranges", {})
all_links: set = st.session_state.get("_all_links", set())

# --------------------------
# Gems + Zielanzahl (Regler sind IMMER sichtbar)
# --------------------------
gem_pct = st.slider(
    "Anteil starker Linkgeber (Top-X %)",
    1, 30, 10, step=1,
    help="Welche obersten X % nach Linkpotenzial sollen als 'Gems' aka starke Linkgeber-URLs gelten?"
)
max_targets_per_gem = st.number_input(
    "Top-Ziele je Gem",
    min_value=1, max_value=50, value=10, step=1,
    help="Wie viele Ziel-URLs pro Gem in der Output-Tabelle gezeigt werden."
)

# --------------------------
# Gewichtung Dringlichkeit (PRIO) inkl. GSC-Upload (direkt hier)
# --------------------------
st.markdown("#### Linkbedarf-Gewichtung")

gsc_up = st.file_uploader(
    "Search Console Daten (CSV/Excel)",
    type=["csv", "xlsx", "xlsm", "xls"],
    key="gsc_up_merged_no_opp",
    help=(
        "Erforderlich: URL, Impressions · Optional: Clicks, Position\n"
        "Spalten dürfen in **beliebiger Reihenfolge** stehen. Erkannte Header (Beispiele):\n"
        "• URL: url, page, seite, address/adresse\n"
        "• Impressions: impressions, impr, search impressions, impressions_total\n"
        "• Clicks: clicks, klicks\n" 
        "• Position: position, avg/average position, (de) durchschnittliche/durchschn. position\n"
        "Impressions werden per log1p normalisiert. URLs werden intern normalisiert; Anzeige bleibt im Original."
    ),
)

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

    # --- Spalten robust erkennen (Header-basiert, mit Fallback auf Positionen) ---
    hdr = [str(c).strip().lower() for c in df.columns]

    def _find_idx(candidates, default=None):
        for i, h in enumerate(hdr):
            if h in candidates:
                return i
        return default

    url_idx  = _find_idx({"url", "page", "seite", "address", "adresse"}, 0)
    impr_idx = _find_idx({"impressions", "impr", "search impressions", "impressions_total"}, 1)
    click_idx = _find_idx({"clicks", "klicks"}, 2 if df.shape[1] >= 3 else None)
    pos_idx   = _find_idx(
        {"position", "avg position", "average position", "durchschnittliche position", "durchschn. position"},
        3 if df.shape[1] >= 4 else None
    )

    # URLs normalisieren (nur für den Key), aber Anzeige bleibt Original via disp()
    df.iloc[:, url_idx] = df.iloc[:, url_idx].astype(str).map(normalize_url)
    urls_series = df.iloc[:, url_idx]

    # Nachfrage (Impressions) -> log1p + MinMax
    impr = pd.to_numeric(df.iloc[:, impr_idx], errors="coerce").fillna(0)
    log_impr = np.log1p(impr)
    if (log_impr.max() - log_impr.min()) > 0:
        demand_norm = (log_impr - log_impr.min()) / (log_impr.max() - log_impr.min())
    else:
        demand_norm = np.zeros_like(log_impr)

    demand_map = {}
    for u, d in zip(urls_series, demand_norm):
        demand_map[str(u)] = float(d)

    # Position (optional, egal an welcher Stelle)
    has_pos = False
    pos_map = {}
    if pos_idx is not None and pos_idx < df.shape[1]:
        pos_series = pd.to_numeric(df.iloc[:, pos_idx], errors="coerce")
        for u, p in zip(urls_series, pos_series):
            if pd.notna(p) and str(u):
                pos_map[str(u)] = float(p)
        has_pos = len(pos_map) > 0

    # für spätere Nutzung verfügbar halten
    st.session_state["__gsc_df_raw__"] = df.copy()


# PRIO-Regler (GSC-abhängige Slider automatisch ausgrauen)
colA, colB = st.columns(2)
with colA:
    w_lihd = st.slider(
        "Gewicht: Hidden Champions",
        0.0, 1.0, 0.30, 0.05, disabled=not has_gsc,
        help="Heißt: viel Such-Nachfrage (Search Console Impressions), aber zu schwach verlinkt ⇒ höherer Linkbedarf."
    )
    w_def  = st.slider(
        "Gewicht: Semantische Linklücke",
        0.0, 1.0, 0.30, 0.05,
        help="Fehlen Links von semantisch ähnlichen URLs? --> Anteil der 'Related' Quellen, die noch nicht aus dem Content heraus verlinken. Similarity dient als **Gewicht**, heißt: Je ähnlicher die Themen, desto stärker fällt der fehlende Link ins Gewicht."
    )
with colB:
    w_rank = st.slider(
        "Gewicht: Sprungbrett-URLs",
        0.0, 1.0, 0.30, 0.05, disabled=not has_pos,
        help="URLs mit durchschnittlicher Rankingosition im eingestellten Sprungbrett-Bereich (z. B. 8–20) erhalten ein höheres Gewicht. Benötigt die SC-Position für die URL."
    )
    w_orph = st.slider(
        "Gewicht: Mauerblümchen",
        0.0, 1.0, 0.10, 0.05,
        help="Seiten mit schlechter Linkversorgung aus dem Content heraus (Navi / Footer wird hier nicht betrachtet). Orphan = 0 interne Inlinks. Thin = sehr wenige Inlinks. Hebt 'vergessene' Seiten hervor."
    )

# Info zur Summe (nur Hinweis, wir normalisieren intern)
eff_sum = (0 if not has_gsc else w_lihd) + w_def + (0 if not has_pos else w_rank) + w_orph
if not math.isclose(eff_sum, 1.0, rel_tol=1e-3, abs_tol=1e-3):
    st.caption(f"ℹ️ Aktuelle PRIO-Gewichtungs-Summe: {eff_sum:.2f}. (kein Problem, wenn > 1 oder < 1, wird intern normalisiert)")

# --- Offpage-Dämpfung (standardmäßig aktiv) ---
st.markdown("##### Offpage-Einfluss (Backlinks & Ref. Domains)")
st.caption("Seiten mit Backlinks von vielen verschiedenen Domains bekommen etwas weniger Dringlichkeit / Linkbedarf verliehen. Wir beziehen für ein realisitischers Gesamtbild standardmäßig gemäß des TIPR-Ansatzes auch die Offpage-Daten in die Optimierung der internen Verlinkung mit ein.")
offpage_damp_enabled = st.checkbox(
    "Offpage-Dämpfung auf Hidden Champions & Semantische Linklücke anwenden",
    value=True,
    help="Offpage-Dämpfung: Seiten mit Backlinks von vielen verschiedenen Domains (Referring Domains) bekommen etwas weniger Dringlichkeit / Linkbedarf."
)
beta_offpage = st.slider(
    "Stärke der Dämpfung durch Offpage-Signale",
    0.0, 1.0, 0.30, 0.05,
    disabled=not offpage_damp_enabled,
    help="0 = keine Dämpfung. Höherer Wert = stärkere Dämpfung -->  stärkere Reduktion des Bedarfs, interne Links setzen zu müssen für URLs mit vielen Backlinks/Ref. Domains."
)


# Abgrenzende Überschrift für Thin-Schwelle
st.markdown("##### Mauerblümchen-Definition")

thin_k = st.slider(
    "Thin-Schwelle (Inlinks ≤ K)", 0, 10, 2, 1,
    help="Ab wie vielen eingehenden internen Links gilt eine Seite als 'thin' (sehr schwach verlinkt) bzw. ab welcher Anzahl an eingehenden Links nicht mehr als schwach verlinkt?"
)

# Ranking-Sweet-Spot (direkt sichtbar)
rank_minmax = st.slider(
    "Ranking Sprungbrett-URL (Positionsbereich)",
    1, 50, (8, 20), 1,
    help="Bereich der durchschnittlichen Position einer (Sprungbrett-)URL, der bevorzugt wird (z. B. 8–20). These: URLs mit Rankings in diesem Bereich haben schon eine gewisse Relevanz aufgebaut, eine Optimierung in Form der Verbesserung der internen Verlinkung zahlt sich hier schneller aus als wenn eine URL von Position 50 auf Position 1 gehoben werden soll.",
    disabled=not has_pos
)

# Sortierlogik (laienfreundliche Labels, gleiche Mechanik)
sort_labels = {
    "rank_mix":   "Mix (Nähe & Linkbearf kombiniert)",
    "prio_only":  "Nur Linkbedarf",
    "sim_only":   "Nur inhaltliche Nähe",
}
sort_choice = st.radio(
    "Reihenfolge der Empfehlungen",
    options=list(sort_labels.keys()),
    format_func=lambda k: sort_labels[k],
    horizontal=True,
    help=("Hier legst du fest, **in welcher Reihenfolge die Ziel-URLs pro Gem** angezeigt werden:\n"
          "• Empfehlungsmix: Kombination aus inhaltlicher Nähe (Similarity) und Linkbedarf (PRIO)\n"
          "• Nur Linkbedarf: Seiten mit höchster PRIO zuerst\n"
          "• Nur inhaltliche Nähe: Seiten mit höchster Similarity zuerst")
)
alpha = st.slider(
    "Gewichtung: inhaltliche Nähe vs. Linkbedarf",
    0.0, 1.0, 0.6, 0.05,
    help=("Gilt nur für den **Mix**: Links = Linkbedarf wichtiger, "
          "Rechts = inhaltliche Nähe wichtiger.")
)


# --- Let's Go Button jetzt UNTEN, nach Balance ---
if "__gems_loading__" not in st.session_state:
    st.session_state["__gems_loading__"] = False
if "__ready_gems__" not in st.session_state:
    st.session_state["__ready_gems__"] = False

run_gems = st.button("Let's Go (Analyse 3)", type="secondary")

# Klick setzt Lade-Flag (persistiert über Reruns)
if run_gems:
    st.session_state["__gems_loading__"] = True
    st.session_state["__ready_gems__"] = False

# GIF anzeigen, solange Lade-Flag aktiv ist
if st.session_state["__gems_loading__"]:
    with st.container():
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(
                "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExcnY0amo3NThxZnpnb3I4dDB6NWF2a2RkZm9uaXJ0bml1bG5lYm1mciZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6HypNJJjcfnZ1bzWDs/giphy.gif",
                width=280,
            )
            st.caption("Analyse 3 läuft … Wir geben Gas – versprochen!")

# Gate: nur stoppen, wenn weder geladen wird noch Ergebnis vorliegt
if not (st.session_state["__gems_loading__"] or st.session_state.get("__ready_gems__", False)):
    st.info("Stell die Regler ein und lade ggf. **Search Console Daten**. Dann klicke auf **Let's Go (Analyse 3)**.")
    st.stop()


# --------------------------
# Hilfsfunktionen für PRIO-Signale
# --------------------------
from collections import defaultdict

# Inbound-Counts für Orphan/Thin
inbound_count = defaultdict(int)
for s, t in st.session_state.get("_content_links", set()):
    inbound_count[t] += 1

min_ils, max_ils = norm_ranges.get("ils", (0.0, 1.0))

# --- Offpage: normalisierte externe Autorität & Dämpfungsfaktor ---
lo_bl_log, hi_bl_log = norm_ranges.get("bl_log", (0.0, 1.0))
lo_rd_log, hi_rd_log = norm_ranges.get("rd_log", (0.0, 1.0))
backlink_map: Dict[str, Dict[str, float]] = st.session_state.get("_backlink_map", {})

def _safe_norm(x: float, lo: float, hi: float) -> float:
    if hi > lo:
        v = (float(x) - lo) / (hi - lo)
        return float(np.clip(v, 0.0, 1.0))
    return 0.0

def ext_auth_norm_for(u: str) -> float:
    """Externe Autorität (0..1) für die Dämpfung – log-transformiert + robust skaliert."""
    bl = backlink_map.get(u, {})
    bl_raw = float(bl.get("backlinks", 0.0) or 0.0)
    rd_raw = float(bl.get("referringDomains", 0.0) or 0.0)

    bl_log = float(np.log1p(max(0.0, bl_raw)))
    rd_log = float(np.log1p(max(0.0, rd_raw)))

    bl_n = robust_norm(bl_log, lo_bl_log, hi_bl_log)
    rd_n = robust_norm(rd_log, lo_rd_log, hi_rd_log)
    return 0.5 * (bl_n + rd_n)


def damp_factor(u: str) -> float:
    """
    Logistische Offpage-Dämpfung (sanfter als linear):
      - x  = externe Autorität in [0,1] (aus ext_auth_norm_for)
      - s  = Sigmoid( x; k, m ) auf [0,1] renormiert
      - Dämpfung = 1 − β · s

    Effekte:
      * wenig Offpage  → nahe 1.0 (kaum Dämpfung)
      * mittel Offpage → moderat gedämpft
      * viel Offpage   → nähert sich 1 − β (stärker gedämpft)
    """
    if not offpage_damp_enabled:
        return 1.0

    # Externe Autorität (bereits 0..1-normalisiert, idealerweise robust/log-transformiert)
    x = float(np.clip(ext_auth_norm_for(u), 0.0, 1.0))

    # Form der S-Kurve: k = Steilheit, m = Wendepunkt
    k = 6.0   # typ. 4–10; höher = steiler
    m = 0.5   # Mittelpunkt (0..1)

    # Sigmoid und anschließende Renormierung auf exakt [0,1]
    s  = 1.0 / (1.0 + np.exp(-k * (x - m)))
    s0 = 1.0 / (1.0 + np.exp(-k * (0.0 - m)))
    s1 = 1.0 / (1.0 + np.exp(-k * (1.0 - m)))
    s_norm = (s - s0) / (s1 - s0)
    s_norm = float(np.clip(s_norm, 0.0, 1.0))

    # Endgültiger Dämpfungsfaktor
    return float(np.clip(1.0 - beta_offpage * s_norm, 0.0, 1.0))


def ils_norm_for(u: str) -> float:
    m = metrics_map.get(u)
    if not m:
        return 0.0
    x = float(m.get("score", 0.0))
    return float(np.clip((x - min_ils) / (max_ils - min_ils), 0.0, 1.0)) if max_ils > min_ils else 0.0

def lihd_for(u: str) -> float:
    """Low Internal, High Demand (Hidden Champions) = (1 − ILS_norm) · Demand_norm · damp(u)"""
    if not has_gsc:
        return 0.0
    d = float(demand_map.get(u, 0.0))
    base = float((1.0 - ils_norm_for(u)) * d)
    return base * damp_factor(u)

def rank_sweetspot_for(u: str, lo: int, hi: int) -> float:
    """1, wenn Position im Sweet-Spot [lo, hi], sonst 0."""
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
    """
    Similarity-gewichteter Anteil noch fehlender Content-Links.
    Danach Offpage-Dämpfung des ZIELS: starke externe Autorität => geringerer Linkbedarf.
    """
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
        col_sim  = f"Ähnlichkeit {i}"
        col_src  = f"Related URL {i}"
        col_cont = f"aus Inhalt heraus verlinkt {i}?"
        if col_sim not in res1_df.columns or col_src not in res1_df.columns:
            break

        sim_val = r.get(col_sim, np.nan)
        src_val = normalize_url(r.get(col_src, ""))
        if pd.isna(sim_val) or not src_val:
            i += 1
            continue

        simf = float(sim_val) if pd.notna(sim_val) else 0.0
        simf = max(0.0, simf)
        sum_all += simf

        # "fehlend" = es existiert KEIN Content-Link Quelle -> Ziel
        from_content = str(r.get(col_cont, "nein")).strip().lower()
        if from_content != "ja":
            sum_missing += simf

        i += 1

    ratio = float(np.clip(sum_missing / sum_all, 0.0, 1.0)) if sum_all > 0 else 0.0
    # Offpage-Dämpfung auf das ZIEL anwenden (viel Autorität => geringerer Linkbedarf)
    return ratio * damp_factor(target)


# --------------------------
# Gems bestimmen (aus Linkpotenzial)
# --------------------------
if source_potential_map:
    sorted_sources = sorted(source_potential_map.items(), key=lambda x: x[1], reverse=True)
    cutoff_idx = max(1, int(len(sorted_sources) * gem_pct / 100))
    gems = [u for u, _ in sorted_sources[:cutoff_idx]]
else:
    gems = []

# --------------------------
# PRIO je Ziel berechnen (normalisiert über aktive Gewichte)
# --------------------------
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

# --------------------------
# Empfehlungen pro Gem bauen (nur: kein Content-Link vorhanden)
# --------------------------
if not isinstance(res1_df, pd.DataFrame) or res1_df.empty or not gems:
    st.session_state["__gems_loading__"] = False
    st.caption("Keine Gem-Daten/Analyse 1 fehlt. Bitte erst Schritt 1+2 ausführen.")
    st.stop()

gem_rows: List[List] = []
for gem in gems:
    for _, row in res1_df.iterrows():
        target = normalize_url(row["Ziel-URL"])
        i = 1
        while True:
            col_src  = f"Related URL {i}"
            col_sim  = f"Ähnlichkeit {i}"
            col_cont = f"aus Inhalt heraus verlinkt {i}?"
            if col_src not in res1_df.columns:
                break
            src = normalize_url(row.get(col_src, ""))
            if not src or src != gem:
                i += 1
                continue
            # bereits Content-Link? → skip
            from_content = str(row.get(col_cont, "nein")).strip().lower()
            if from_content == "ja":
                i += 1
                continue

            simf = float(row.get(col_sim, 0.0) or 0.0)
            prio_t = float(target_priority_map.get(target, 0.0))

            if sort_choice == "prio_only":
                sort_score = prio_t
            elif sort_choice == "sim_only":
                sort_score = simf
            else:  # "rank_mix"
                sort_score = alpha * simf + (1.0 - alpha) * prio_t


            gem_rows.append([gem, target, simf, prio_t, sort_score])
            i += 1

# Pro Gem top-N schneiden & sortieren
if gem_rows:
    import itertools
    gem_rows.sort(key=lambda r: r[0])  # nach Gem gruppieren
    final_rows: List[List] = []
    for gem_key, group in itertools.groupby(gem_rows, key=lambda r: r[0]):
        grp = list(group)
        # Sortierung je Modus (absteigend)
        if sort_choice == "prio_only":
            grp = sorted(grp, key=lambda r: (r[3], r[2]), reverse=True)   # PRIO, Tie-Break: Sim
        elif sort_choice == "sim_only":
            grp = sorted(grp, key=lambda r: (r[2], r[3]), reverse=True)   # Sim,  Tie-Break: PRIO
        else:  # "rank_mix"
            grp = sorted(grp, key=lambda r: (r[4], r[2], r[3]), reverse=True)  # Sortwert, dann Sim/PRIO
        final_rows.extend(grp[:int(max_targets_per_gem)])
    gem_rows = final_rows

# --------------------------
# Ausgabe: Breite Tabelle + Download  (→ HIER disp() FÜR ANZEIGE)
# --------------------------
if gem_rows:
    from collections import defaultdict
    by_gem: Dict[str, List[Tuple[str, float, float, float]]] = defaultdict(list)
    for gem, target, simv, prio_t, sortv in gem_rows:
        by_gem[gem].append((target, float(simv), float(prio_t), float(sortv)))

    cols = ["Gem (Quelle)", "Linkpotenzial (Quelle)"]
    for i in range(1, int(max_targets_per_gem) + 1):
        cols += [f"Ziel {i}", f"Similarity (inhaltliche Nähe) {i}", f"Linkbedarf PRIO {i}", f"Score für Sortierung {i}"]

    def pot_for(g: str) -> float:
        return float(st.session_state.get("_source_potential_map", {}).get(normalize_url(g), 0.0))

    ordered_gems = sorted(by_gem.keys(), key=pot_for, reverse=True)
    rows = []
    for gem in ordered_gems:
        items = by_gem[gem]
        row = [disp(gem), round(pot_for(gem), 3)]              # ← disp(gem)
        for i in range(int(max_targets_per_gem)):
            if i < len(items):
                target, simv, prio_t, sortv = items[i]
                row += [disp(target),                           # ← disp(target)
                        round(simv, 3), round(prio_t, 3), round(sortv, 3)]
            else:
                row += [np.nan, np.nan, np.nan, np.nan]
        rows.append(row)

    cheat_df = pd.DataFrame(rows, columns=cols)
    st.dataframe(cheat_df, use_container_width=True, hide_index=True)

    csv_cheat = cheat_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download »Cheat-Sheet der internen Verlinkung« (CSV)",
        data=csv_cheat,
        file_name="Cheat-Sheet der internen Verlinkung.csv",
        mime="text/csv",
    )

    st.session_state["__gems_loading__"] = False
    st.success("✅ Analyse abgeschlossen!")
    st.session_state["__ready_gems__"] = True

else:
    st.session_state["__gems_loading__"] = False
    st.caption("Keine Gem-Empfehlungen gefunden – prüfe GSC-Upload/Signale, Gem-Perzentil oder Similarity/PRIO-Gewichte.")

