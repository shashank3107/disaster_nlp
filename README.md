# Disaster Tweet Intelligence — Streamlit App

An interactive application that takes a CSV/TSV of tweets (and/or scraped news
articles) for one event and runs the full pipeline with every step visible:

1. **Upload & Classify** — RoBERTa checkpoints predict *informative*, *humanitarian
   category*, and *damage severity*. Non-English (incl. Indian-language) text is
   auto-translated to English first, and news articles are sliced into tweet-sized
   snippets so they classify alongside tweets.
2. **Knowledge Graph** — BERT-NER → relation extraction → triples → graph
   (each stage shown live), with an embedded interactive graph.
3. **Chatbot** — hybrid Graph-RAG retrieval + **Gemini API** grounded answers
   with tweet citations.
4. **Resource Dashboard** — extracts resource needs vs. availability, raises unmet-
   need alerts, and plots an **affected-areas map** with circled hotspots.

CPU-only: classification, NER, and embeddings run on CPU; generation uses the
Gemini API, so **no GPU is needed**.

---

## Input formats

You can upload **tweets**, **news articles**, or both — they are classified together
in a single run. Both CSV and **TSV** (tab-separated) files are accepted; the
delimiter is picked automatically from the file extension (`.csv` vs `.tsv`/`.tab`).

### Tweets

One file per event, with these columns:

| column | example |
|--------|---------|
| `tweet_url` | http://x.com/123 |
| `tweet_id` | 123 |
| `username` | reliefnow |
| `tweet_text` | Red Cross volunteers rescuing families in Houston #Harvey |
| `timestamp` | 2017-08-28T10:00:00 |

`username`, `timestamp`, and `tweet_url` are kept as provenance on the tweet nodes
(used for citations); the graph node types are unchanged.

### News articles

A scraped-article CSV/TSV is also supported (separate uploader). It needs at least
one text column — `full_text`, `snippet`, or `title` — and optionally
`article_url`, `source`, `author`, `published_date`, `scrape_status` for provenance.
Each article is converted into tweet-sized records:

- Text source per article: `full_text` when `scrape_status == ok`, else `snippet`,
  else `title`.
- Encoding artifacts (mojibake like `â`, `Â`) are repaired.
- The headline is kept as the first snippet, then the body is split on sentence
  boundaries into ≤280-character chunks.
- A **Max snippets per article** slider caps each article's footprint (default 15).

The resulting snippets are plain tweet-shaped records (no special tag), so they flow
through classification, the KG, the chatbot, and the dashboard exactly like tweets.

### Re-loading classified data

After classification you can **save** the results (see below) and later **re-load
them** via the "load previously classified data" expander to skip classification and
jump straight to the knowledge graph. Expected columns:
`tweet_id, tweet_text, text_info, text_human, text_damage`.

---

## Local setup

### 1. Get the code + Python env
```bash
git clone <your-repo>            # or copy the disaster_nlp folder
cd disaster_nlp
python -m venv .venv && source .venv/bin/activate     # or conda
pip install -r requirements_app.txt
```

### 2. Copy the trained classifier checkpoints
The app loads RoBERTa checkpoints from `experiments/<model>_<task>/best/`. Copy
these three folders from the HPC to the same path locally:

```
experiments/roberta_informative/best/
experiments/roberta_humanitarian/best/
experiments/roberta_damage/best/
```
e.g.
```bash
rsync -av <hpc>:~/disaster_nlp/experiments/roberta_informative ./experiments/
rsync -av <hpc>:~/disaster_nlp/experiments/roberta_humanitarian ./experiments/
rsync -av <hpc>:~/disaster_nlp/experiments/roberta_damage ./experiments/
```
Each `best/` folder needs `config.json`, `model.safetensors`, and the tokenizer
files. (DistilBERT checkpoints work too — selectable in the sidebar.)

### 3. Get a Gemini API key
Create one at Google AI Studio, then either set it in your environment:
```bash
export GEMINI_API_KEY="your-key"
```
or paste it into the app sidebar at runtime.

### 4. Run
```bash
streamlit run app.py
```
Opens at http://localhost:8501.

---

## First-run model downloads

On first use the app downloads (and caches) two small models from HuggingFace:
- `dslim/bert-base-NER` (~430 MB) for entity extraction
- `BAAI/bge-small-en-v1.5` (~130 MB) for embeddings

These are cached in `~/.cache/huggingface` and reused thereafter.

Two features make lightweight **network calls** instead (no model download):
- **Translation** uses Google Translate via `deep-translator` (no API key).
- **Geocoding** for the map uses a built-in gazetteer first, then OpenStreetMap's
  Nominatim for unknown place names. Results are cached to `.geocode_cache.json` in
  the project root so each place is looked up only once.

---

## Sidebar options

| Option | Purpose |
|--------|---------|
| Event name | Label for the central Event node |
| Gemini API key / model | Chatbot generation (default `gemini-2.0-flash`) |
| Classifier family | `roberta` (best) or `distilbert` |
| Checkpoints dir | Where `*_<task>/best` live (default `./experiments`) |
| Device | `cpu` (default) or `cuda` |
| Use BERT-NER | Toggle off to use the fast regex fallback |
| Translate non-English → English | Auto-detect and translate non-English text before classifying (on by default) |
| Graph: only informative tweets | Restrict the KG to tweets classified informative |
| top-k / semantic candidates | Retrieval breadth for the chatbot |

The **Max snippets per article** slider lives next to the news uploader on the
Upload tab (not the sidebar), since it only applies to news ingestion.

---

## Notes

- **Gemini model id.** Default is `gemini-2.0-flash`; if your key doesn't have it,
  change the field to a flash model your account supports (e.g. `gemini-1.5-flash`).
- **No key?** The chatbot still works in retrieval-only mode — it shows the top
  retrieved tweets instead of a generated answer.
- **Grounding.** The Gemini call uses the same system prompt as the local LLM
  path: answer only from the retrieved tweets, cite `tweet_id`s, say so if the
  answer isn't in context.
- **Scale.** CPU classification of a few thousand tweets takes a couple of minutes;
  a progress bar is shown. For very large CSVs, run on a machine with a GPU and set
  Device = `cuda`.
- **Translation.** Non-English text (native Indian scripts — Devanagari, Tamil,
  Telugu, Bengali, etc. — plus best-effort romanized/Hinglish) is translated to
  English before classification; the original is kept in a `text_original` column.
  Translation needs internet and is subject to Google's free-endpoint rate limits.
  Toggle it off in the sidebar to classify text as-is.
- **Saving / re-loading results.** After classification, use **Save classified data
  (CSV)** to export `tweet_id, tweet_text, text_info, text_human, text_damage`. You
  can re-upload that file later to rebuild the knowledge graph without re-running the
  classifiers. (This compact format doesn't store provenance, so re-loaded items
  have blank url/username/timestamp.)
- **News articles.** Slicing uses rule-based sentence chunking (offline,
  deterministic), so some boilerplate from `full_text` may become snippets — lower
  **Max snippets per article** to cap each article's footprint. With a Gemini key,
  the Resource Dashboard can still extract place names from these snippets for the
  affected-areas map.
- **Affected-areas map.** Circle size = number of need mentions; red = unmet need
  (need > availability), orange = needs being met, green = availability only.
  Locations come from the Gemini resource extractor, so without a Gemini key the map
  will be empty.
