"""
TASC Corporate Naming Engine — Streamlit deployment version.

This is a rewrite of the original Colab/ipywidgets notebook as a standalone
web app. Key differences from the notebook:
  - No google.colab imports; the brief file is bundled in the repo instead
    of uploaded each session.
  - The Anthropic API key comes from Streamlit secrets, never typed by users.
  - State that used to live in plain Python globals (name_pool,
    verified_results) now lives in st.session_state, which Streamlit keeps
    separate per browser session automatically.
  - If the bundled brief is short enough, the whole document is sent in the
    prompt directly (SIMPLE_MODE) instead of chunking + embedding it. This
    avoids bundling sentence-transformers/torch, which are large and can
    blow past free-tier RAM limits. If the brief is long, the app
    automatically falls back to the original embedding-based retrieval.
"""

import io
import json
import re
import socket
import time

import docx
import numpy as np
import streamlit as st
import anthropic

# ── Optional password gate ───────────────────────────────────────────────
# Because this app calls the Anthropic API with a key you pay for, anyone
# with the URL could otherwise run up usage. If you set APP_PASSWORD in
# secrets, visitors must enter it once per session. If you don't set it,
# the app is open to anyone with the link (fine for a small internal team).
def check_password():
    app_password = st.secrets.get("APP_PASSWORD")
    if not app_password:
        return True
    if st.session_state.get("authed"):
        return True
    st.title("TASC Naming Engine")
    pw = st.text_input("Password", type="password")
    if pw == app_password:
        st.session_state.authed = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    return False


st.set_page_config(page_title="TASC Naming Engine", page_icon="🏛️", layout="wide")

if not check_password():
    st.stop()

# ── Config — same rules as the notebook, pulled directly from the brief ──
MAX_NAMES = 100
DEFAULT_NAME_LENGTH = 5
MIN_NAME_LENGTH = 3
MAX_NAME_LENGTH = 10
BRIEF_PATH = "TASC_Naming_Brief_v1.docx"   # bundled in the repo, next to app.py
SIMPLE_MODE_CHAR_THRESHOLD = 12000         # ~3k tokens; below this, skip RAG entirely
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GEN_MODEL = "claude-sonnet-5"

ALLOWED_ENDINGS = ["aq", "ar", "ad", "ok", "id", "or", "on", "ex", "ax"]
BANNED_ENDINGS = ["a", "la", "ra", "li", "ri", "ma", "na", "ya", "in", "en"]

PERMANENT_BLACKLIST = {
    "spyne": "REJECTED in brief -- live commercial conflict + phonetic clash with 'Spine'.",
    "asas": "REJECTED in brief -- Ismaili theological term + banned Iraqi newspaper name.",
    "armoud": "REJECTED in brief -- Moroccan village name + personal name + registered tyre brand.",
    "hardir": "NOT RECOMMENDED in brief -- no usable meaning + rhymes with 'Amir'.",
}

GEO_EXCLUSIONS = ["qatar", "kuwait", "jordan", "iraq", "tunisia", "algeria", "mosul",
                   "wasit", "gulf", "levant", "mena", "arabia", "arabian"]

SATURATED_WORDS = ["foundation", "trust", "independence", "pioneer", "bridge",
                    "union", "link", "asas", "etihad", "ittihad"]

EUROPEAN_SOUNDING_HINTS = ["tech", "soft", "ify", "app", "ly", "io", "labs", "systems", "corp"]

client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ── Load the bundled brief (once per app instance, shared across users) ──
@st.cache_resource(show_spinner="Loading naming brief...")
def load_brief_text():
    d = docx.Document(BRIEF_PATH)
    paras = [p.text.strip() for p in d.paragraphs if p.text.strip()]
    return "\n".join(paras)


BRIEF_TEXT = load_brief_text()
SIMPLE_MODE = len(BRIEF_TEXT) <= SIMPLE_MODE_CHAR_THRESHOLD


def chunk_text(text: str, target_chars: int = 700, overlap_chars: int = 120):
    paras = text.split("\n")
    chunks, current, current_len = [], [], 0
    for p in paras:
        current.append(p)
        current_len += len(p)
        if current_len >= target_chars:
            chunks.append("\n".join(current))
            tail = "\n".join(current)[-overlap_chars:]
            current, current_len = [tail], len(tail)
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]


if not SIMPLE_MODE:
    # Only imported/loaded if the brief is too long to send in full —
    # keeps the simple-mode deployment free of the sentence-transformers
    # dependency entirely.
    from sentence_transformers import SentenceTransformer

    @st.cache_resource(show_spinner="Building retrieval index (one-time)...")
    def build_index():
        embedder = SentenceTransformer(EMBED_MODEL_NAME)
        chunks = chunk_text(BRIEF_TEXT)
        vectors = np.array(embedder.encode(chunks, normalize_embeddings=True))
        return embedder, chunks, vectors

    _embedder, _chunks, _vectors = build_index()

    def retrieve(query: str, k: int = 4):
        q_vec = _embedder.encode([query], normalize_embeddings=True)[0]
        sims = _vectors @ q_vec
        top_idx = np.argsort(-sims)[:k]
        return [_chunks[i] for i in top_idx]

    def retrieve_generation_context():
        queries = [
            "naming criteria priority order pronounceable institutional register",
            "hard exclusions geographic personal names religious sectarian shareholder favoritism",
            "five directions sovereign acronym Arabic phonetic coined celestial single institutional invented English",
            "fused root method institutional ending test concept roots Arabic morphemes",
            "client submitted candidates rejected",
        ]
        seen, out = set(), []
        for q in queries:
            for c in retrieve(q, k=3):
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return "\n\n---\n\n".join(out)

    def retrieve_verification_context(name: str):
        queries = [
            "due diligence requirements zero conflict standard six steps",
            "hard exclusions geographic personal names religious sectarian",
            f"institutional ending test personal name screen {name}",
        ]
        seen, out = set(), []
        for q in queries:
            for c in retrieve(q, k=3):
                if c not in seen:
                    seen.add(c)
                    out.append(c)
        return "\n\n---\n\n".join(out)[:2500]

else:
    def retrieve_generation_context():
        return BRIEF_TEXT

    def retrieve_verification_context(name: str):
        return BRIEF_TEXT


# ── Local rule-based checks (unchanged from the notebook) ────────────────
def has_hard_consonant_cluster(name: str, max_cluster: int = 3) -> bool:
    vowels = set("aeiouAEIOU")
    cluster = 0
    for ch in name:
        if ch.isalpha() and ch not in vowels:
            cluster += 1
            if cluster > max_cluster:
                return True
        else:
            cluster = 0
    return False


def local_prescreen(name: str, target_length: int = None):
    flags = []
    n = name.strip().lower()
    if n in PERMANENT_BLACKLIST:
        flags.append(f"Already ruled on in brief: {PERMANENT_BLACKLIST[n]}")
        return flags
    if not re.fullmatch(r"[a-zA-Z]+", name.strip()):
        flags.append("Contains characters outside plain Latin letters -- check spelling.")
    if target_length and len(name.strip()) != target_length:
        flags.append(f"Length is {len(name.strip())} letters, not the requested {target_length}.")
    bad_hits = [e for e in BANNED_ENDINGS if n.endswith(e)]
    if bad_hits:
        flags.append(f"Ends in a name-pattern suffix ('-{bad_hits[0]}') -- brief requires a hard institutional ending.")
    elif not any(n.endswith(e) for e in ALLOWED_ENDINGS):
        flags.append("Does not end in an approved institutional ending (-aq,-ar,-ad,-ok,-id,-or,-on,-ex,-ax).")
    for geo in GEO_EXCLUSIONS:
        if geo in n:
            flags.append(f"Contains/echoes a geographic term ('{geo}') -- brief excludes single-market references.")
    for word in SATURATED_WORDS:
        if word in n:
            flags.append(f"Overlaps saturated institutional vocabulary ('{word}').")
    for hint in EUROPEAN_SOUNDING_HINTS:
        if hint in n:
            flags.append(f"Contains a Western/tech-startup-sounding fragment ('{hint}') -- should sound Arabic-rooted, not European-coined.")
    if has_hard_consonant_cluster(n):
        flags.append("Long consonant cluster -- may not be easy to pronounce on first read.")
    if 2 <= len(n) <= 4 and n.isalpha():
        flags.append("Very short -- manually confirm it isn't a common given name.")
    return flags


def check_domain_available(name: str) -> str:
    domain = f"{name.lower()}.com"
    try:
        socket.gethostbyname(domain)
        return f"{domain}: DNS resolves -- likely registered/in use, needs manual check."
    except socket.gaierror:
        return f"{domain}: DNS does not resolve -- likely unregistered or parked."


def web_search_snippets(query: str, max_results: int = 5) -> str:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"(search failed: {e})"
    if not results:
        return "(no results found)"
    lines = []
    for r in results:
        lines.append(f"- {r.get('title','')}: {r.get('body','')[:200]} ({r.get('href','')})")
    return "\n".join(lines)


def extract_text(resp):
    return "".join(block.text for block in resp.content if block.type == "text")


def call_claude_with_retry(system_prompt, user_message, max_tokens, max_attempts=4, effort="low"):
    for attempt in range(max_attempts):
        try:
            return client.messages.create(
                model=GEN_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
            )
        except anthropic.RateLimitError:
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt + 1)
                continue
            raise
        except Exception as e:
            if "429" in str(e) and attempt < max_attempts - 1:
                time.sleep(2 ** attempt + 1)
                continue
            raise
    return None


# ── Generation ─────────────────────────────────────────────────────────
def generate_batch(blacklist, target_length, log, max_attempts=3):
    context = retrieve_generation_context()
    seen_lower = set(b.lower() for b in blacklist)
    collected = []

    for attempt in range(max_attempts):
        still_needed = min(5, 10 - len(collected))
        if still_needed <= 0:
            break

        system_prompt = f'''You are a corporate naming engine working strictly from the naming
brief excerpts provided below. Do not use outside knowledge of brand naming trends.
Do not invent naming criteria that are not in the excerpts.

BRIEF EXCERPTS (ground truth -- do not contradict or go beyond this):
{context}

RULES YOU MUST FOLLOW EXACTLY:
- Generate exactly {still_needed} candidates.
- Every candidate name should aim to be exactly {target_length} letters long (a-z only, no spaces, no hyphens).
- Every candidate must end in one of: {", ".join(ALLOWED_ENDINGS)}.
- Never end a candidate in: {", ".join(BANNED_ENDINGS)}.
- Names must be easy to pronounce on first read in both English and Arabic -- no unusual consonant clusters.
- Names must NOT sound European or Western-coined -- avoid tech-startup naming patterns (e.g. -tech, -ify, -ly, -io).
- Never repeat, reuse, or lightly modify any name in this blacklist: {json.dumps(list(seen_lower))}.
- Never produce a name that is geographically specific to one MENA country, reads as a
  personal name, or overlaps saturated words (foundation, trust, independence, pioneer,
  bridge, union, link).
- Note which construction method from the excerpts you used and why, briefly.

Python will independently re-check length, endings, and banned patterns after you respond,
so do not spend time re-verifying each candidate yourself -- just generate your best {still_needed}
candidates in one pass.

Return ONLY a valid JSON array (no markdown fences, no preamble, no commentary):
[
  {{
    "name": "string",
    "direction": "one of: Sovereign-Style Acronym | Arabic-Phonetic Coined | Celestial Register | Single Institutional Arabic Word | Invented Institutional English",
    "method": "short label, e.g. 'Fused Root Method: Wasl + Madar'",
    "roots_used": "which concept roots/letters were used, or 'n/a'",
    "logic": "1-2 sentence plain-English explanation of the meaning/construction",
    "method_rationale": "why this method was chosen over an alternative also available in the excerpts, or 'only one method applied'",
    "ending": "the final 2-3 letters"
  }}
]'''
        resp = call_claude_with_retry(
            system_prompt=system_prompt,
            user_message=f"Generate {still_needed} candidates now, each exactly {target_length} letters.",
            max_tokens=8000,
            effort="low",
        )
        if resp is None:
            log.append(f"[attempt {attempt}] API returned None after retries (repeated rate limiting).")
            continue

        block_types = [b.type for b in resp.content]
        log.append(f"[attempt {attempt}] stop_reason={resp.stop_reason} | blocks={block_types} | "
                    f"in_tok={resp.usage.input_tokens} out_tok={resp.usage.output_tokens}")

        raw = extract_text(resp).strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        if not raw:
            log.append(f"[attempt {attempt}] Empty text after extraction — no usable content in this response.")
            continue

        try:
            candidates = json.loads(raw)
        except json.JSONDecodeError as e:
            log.append(f"[attempt {attempt}] JSON parse failed: {e}")
            continue

        if not isinstance(candidates, list):
            log.append(f"[attempt {attempt}] Expected a list, got {type(candidates)}")
            continue

        for c in candidates:
            try:
                name = c.get("name", "").strip()
            except AttributeError:
                continue
            if not name:
                continue
            n = name.lower()
            if n in seen_lower:
                continue
            if not re.fullmatch(r"[a-zA-Z]+", name):
                continue
            if len(name) != target_length:
                continue
            if any(n.endswith(e) for e in BANNED_ENDINGS):
                continue
            if not any(n.endswith(e) for e in ALLOWED_ENDINGS):
                continue
            if any(h in n for h in EUROPEAN_SOUNDING_HINTS):
                continue
            c["target_length"] = target_length
            seen_lower.add(n)
            collected.append(c)
            if len(collected) >= 10:
                break

    return collected[:10]


def run_verification(name: str) -> str:
    pre_flags = local_prescreen(name)
    if any("Already ruled on" in f for f in pre_flags):
        return f"[REJECTED] {pre_flags[0]}"

    context = retrieve_verification_context(name)
    domain_status = check_domain_available(name)
    search_results = "\n\n".join([
        f"QUERY: \"{name}\" company trademark\n{web_search_snippets(f'\"{name}\" company trademark')}",
        f"QUERY: {name} LinkedIn Crunchbase\n{web_search_snippets(f'{name} LinkedIn Crunchbase')}",
        f"DOMAIN CHECK: {domain_status}",
    ])

    system_prompt = f'''You are running the brief's due-diligence protocol on ONE
candidate name: "{name}".

RELEVANT BRIEF EXCERPTS:
{context}

LOCAL RULE-BASED PRE-SCREEN FINDINGS (already computed, treat as facts):
{json.dumps(pre_flags) if pre_flags else "No rule-based flags found."}

LIVE WEB SEARCH RESULTS (already fetched, treat as facts -- do not claim to search yourself):
{search_results}

Only treat a search result as a genuine commercial conflict if "{name}" appears as the
COMPLETE name of a company, brand, or product -- not as a substring inside a longer
phrase or title, and not as a shared prefix with a different word. Do not flag
informational pages (Wikipedia definitions of unrelated companies, USPTO's general
trademark-search homepage, generic help pages) -- these appear in nearly every search
and carry no signal.

The phonetic-neighbor test (checking words that differ by one letter/vowel) applies ONLY
to the personal-name screen -- i.e. whether the candidate itself sounds like a person's
name. It does NOT apply to the commercial-conflict search.

You cannot personally consult native Gulf/Levantine/Iraqi Arabic speakers -- say so
explicitly and flag dialect + cultural/political sensitivity review as an OUTSTANDING
human step per the brief's Step 5 and Step 6.

Give your verdict using exactly one of these three labels, with these precise meanings:

[REJECTED] -- use when either:
  (a) the name matches a PERMANENT_BLACKLIST entry or a hard exclusion in the brief with
      certainty, or
  (b) local pre-screen findings above already flag a rule violation, or
  (c) a search result shows a company/brand LITERALLY using the exact candidate string as
      its name or registered trademark -- regardless of industry or country, or
  (d) search results confirm the candidate is an actual, currently-used personal name --
      REQUIRES [REJECTED] under the personal-names hard exclusion, not RISK.

[RISK] -- use when there is a plausible-but-unconfirmed concern that a human must resolve,
  such as: the candidate begins with, ends with, or closely resembles a known common
  Arabic/English/Persian given name (cite the specific name and why); a search result
  exists but is ambiguous, unrelated-industry, or inconclusive; domain status could not
  be fully confirmed.

[SAFE] -- no rule-based flags, no hard-exclusion matches, no exact-string commercial
  matches, and no meaningful personal-name-neighbor concern. Standard outstanding
  due-diligence steps still apply.

When multiple concerns exist, list each one under a clear label (COMMERCIAL CONFLICT /
PERSONAL NAME / GEOGRAPHIC / SECTARIAN / OTHER) rather than blending them into one
sentence.'''

    resp = call_claude_with_retry(
        system_prompt=system_prompt,
        user_message=f"Give the verdict for '{name}' now.",
        max_tokens=6000,
        effort="low",
    )
    if resp is None:
        return "[ERROR] Rate limit hit repeatedly — wait a minute and click Verify again."
    return extract_text(resp) or "No verdict text returned."


# ── Session state ──────────────────────────────────────────────────────
if "name_pool" not in st.session_state:
    st.session_state.name_pool = []
if "verified_results" not in st.session_state:
    st.session_state.verified_results = {}
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []

# ── UI ─────────────────────────────────────────────────────────────────
st.title("🏛️ TASC Corporate Naming Engine")
st.caption("Generates naming candidates strictly from the TASC naming brief, then runs due diligence on demand.")

with st.sidebar:
    st.subheader("Controls")
    target_length = st.slider("Target letters", MIN_NAME_LENGTH, MAX_NAME_LENGTH, DEFAULT_NAME_LENGTH)
    pool_full = len(st.session_state.name_pool) >= MAX_NAMES
    gen_label = "Pool full (100)" if pool_full else (
        "Generate 10 names" if not st.session_state.name_pool else "Load 10 more names"
    )
    if st.button(gen_label, type="primary", disabled=pool_full):
        with st.spinner("Generating..."):
            blacklist = [c["name"] for c in st.session_state.name_pool] + list(PERMANENT_BLACKLIST.keys())
            new_batch = generate_batch(blacklist, target_length, st.session_state.debug_log)
            room = MAX_NAMES - len(st.session_state.name_pool)
            st.session_state.name_pool.extend(new_batch[:room])
            if not new_batch:
                st.session_state.debug_log.append("Generation returned 0 candidates this round.")
        st.rerun()

    st.divider()
    st.write("**Add a custom name**")
    custom_name = st.text_input("Name", key="custom_name_input", label_visibility="collapsed")
    if st.button("Add to pool") and custom_name.strip():
        clean = custom_name.strip()
        existing = [c["name"].lower() for c in st.session_state.name_pool]
        if clean.lower() not in existing and len(st.session_state.name_pool) < MAX_NAMES:
            st.session_state.name_pool.append({
                "name": clean, "direction": "User-submitted", "method": "Manual entry",
                "roots_used": "n/a", "logic": "Manually entered for review.",
                "method_rationale": "n/a", "ending": clean[-2:] if len(clean) > 2 else clean,
                "target_length": len(clean),
            })
            st.rerun()

    st.divider()
    if st.button("Reset pool"):
        st.session_state.name_pool = []
        st.session_state.verified_results = {}
        st.session_state.debug_log = []
        st.rerun()

    if st.session_state.debug_log:
        with st.expander("Diagnostics"):
            for line in st.session_state.debug_log[-20:]:
                st.text(line)

st.write(f"**Names in pool: {len(st.session_state.name_pool)} / {MAX_NAMES}**")

for item in st.session_state.name_pool:
    name = item["name"]
    with st.container(border=True):
        st.markdown(f"#### {name}")
        st.caption(f"Direction: {item.get('direction','-')} | Method: {item.get('method','-')} | Ending: {item.get('ending','-')}")
        st.write(item.get("logic", ""))
        if item.get("method_rationale") not in (None, "", "n/a"):
            st.caption(f"Method choice: {item['method_rationale']}")

        flags = local_prescreen(name, target_length=item.get("target_length"))
        if flags:
            st.warning("Local pre-screen flags:\n" + "\n".join(f"- {f}" for f in flags))

        already_verified = name in st.session_state.verified_results
        verify_label = "Re-verify" if already_verified else "Verify"
        if st.button(verify_label, key=f"verify_{name}"):
            with st.spinner(f"Running due diligence on '{name}'..."):
                verdict = run_verification(name)
                st.session_state.verified_results[name] = verdict
            st.rerun()

        if already_verified:
            st.markdown(f"**Verdict for {name}:**\n\n{st.session_state.verified_results[name]}")
