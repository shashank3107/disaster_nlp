"""
Resource need / availability extraction from disaster tweets.

This is the layer that turns tweets into the *actionable* signal the thesis
objectives need: for each tweet, what resource is involved (food / water /
shelter / medical / sanitation / …), whether it is a NEED (a request/shortage)
or an AVAILABILITY (an offer/distribution), where, and how much.

Grounded in the FIRE IRMiDis line of work (identifying need-tweets and
availability-tweets and matching them during disasters).

Two backends:
  * Gemini (primary) — few-shot JSON extraction; needs an API key, no training data.
  * Keyword/cue fallback — runs fully offline so the stage always produces output.

It also provides aggregation + a simple need↔availability matcher used by the
dashboard for alerts and allocation suggestions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Controlled resource vocabulary.
RESOURCE_TYPES = [
    "food", "water", "shelter", "medical", "sanitation",
    "rescue", "power", "money", "clothing",
]

# Cue words for the offline fallback.
_NEED_CUES = [
    "need", "needed", "require", "required", "shortage", "short of", "without",
    "lack", "lacking", "no water", "no food", "send", "please send", "sos",
    "help needed", "stranded", "trapped", "urgent", "requesting", "appeal", "stuck",
]
_AVAIL_CUES = [
    "distribut", "providing", "provide", "provided", "available", "donat",
    "supply", "supplying", "supplied", "relief camp", "set up", "delivered",
    "delivering", "offering", "offer", "we have", "handed over", "dispatch",
    "sent relief", "reaching", "rescued", "evacuated",
]
_RESOURCE_CUES = {
    "water":      ["water", "drinking water"],
    "food":       ["food", "meal", "ration", "grocery", "groceries", "milk", "bread", "hunger", "langar"],
    "shelter":    ["shelter", "tent", "camp", "housing", "accommodation", "roof", "displaced"],
    "medical":    ["medical", "medicine", "hospital", "doctor", "ambulance", "first aid", "injured", "health", "blood", "wounded"],
    "sanitation": ["sanitation", "toilet", "hygiene", "sanitary"],
    "rescue":     ["rescue", "evacuat", "trapped", "stranded", "boat", "ndrf", "sdrf", "relief operation"],
    "power":      ["power", "electricity", "generator", "fuel", "diesel", "charging", "power line"],
    "money":      ["money", "fund", "cash", "donation", "financial", "compensation", "ex-gratia", "relief fund"],
    "clothing":   ["clothing", "clothes", "blanket", "warm"],
}


@dataclass
class ResourceInfo:
    tweet_id:  str
    stance:    str = "none"          # need | availability | none
    resources: List[str] = field(default_factory=list)
    location:  str = ""
    quantity:  str = ""

    def as_dict(self) -> Dict:
        return {
            "tweet_id": self.tweet_id, "stance": self.stance,
            "resources": self.resources, "location": self.location,
            "quantity": self.quantity,
        }


_LLM_INSTRUCTION = (
    "You extract disaster-resource information from tweets for emergency response. "
    "Return ONLY a JSON array — one object per input tweet — with keys exactly:\n"
    '  "id": the tweet id (string)\n'
    '  "stance": one of "need", "availability", "none"\n'
    '  "resources": a subset of '
    '["food","water","shelter","medical","sanitation","rescue","power","money","clothing"]\n'
    '  "location": the most specific place named, else ""\n'
    '  "quantity": any amount/number mentioned, else ""\n'
    'Rules: "need" = someone requests, lacks, or is short of a resource; '
    '"availability" = someone offers, provides, distributes, or has a resource; '
    '"none" = neither (general news/opinion). Use [] for resources when none apply.'
)


class ResourceExtractor:
    """Extract ResourceInfo per tweet (Gemini primary, keyword fallback)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        use_llm: bool = True,
    ) -> None:
        self.api_key = api_key
        self.model_name = model
        self.use_llm = use_llm and bool(api_key)
        self._model = None

    # ── backends ──────────────────────────────────────────────

    def _ensure_model(self):
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(
                self.model_name, system_instruction=_LLM_INSTRUCTION)
        return self._model

    def _llm_extract(self, chunk: List[Dict]) -> Dict[str, ResourceInfo]:
        model = self._ensure_model()
        lines = [f'id={r["tweet_id"]} :: {r["text"]}' for r in chunk]
        prompt = "Tweets:\n" + "\n".join(lines)
        resp = model.generate_content(
            prompt, generation_config={"temperature": 0.0, "max_output_tokens": 2048})
        text = (resp.text or "").strip()
        # strip ```json fences if present
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        data = json.loads(text)
        out: Dict[str, ResourceInfo] = {}
        for obj in data:
            tid = str(obj.get("id", ""))
            res = [r for r in (obj.get("resources") or []) if r in RESOURCE_TYPES]
            stance = obj.get("stance", "none")
            if stance not in ("need", "availability", "none"):
                stance = "none"
            out[tid] = ResourceInfo(
                tweet_id=tid, stance=stance, resources=res,
                location=str(obj.get("location", "") or ""),
                quantity=str(obj.get("quantity", "") or ""),
            )
        return out

    @staticmethod
    def _kw_one(rec: Dict) -> ResourceInfo:
        low = rec["text"].lower()
        resources = [r for r, cues in _RESOURCE_CUES.items() if any(c in low for c in cues)]
        has_need  = any(c in low for c in _NEED_CUES)
        has_avail = any(c in low for c in _AVAIL_CUES)
        if has_need:
            stance = "need"            # prioritise needs for alerting
        elif has_avail:
            stance = "availability"
        else:
            stance = "none"
        # No reliable offline location extraction; left blank (Gemini fills it).
        return ResourceInfo(tweet_id=rec["tweet_id"], stance=stance, resources=resources)

    def _kw_extract(self, chunk: List[Dict]) -> Dict[str, ResourceInfo]:
        return {r["tweet_id"]: self._kw_one(r) for r in chunk}

    # ── public ────────────────────────────────────────────────

    def extract(
        self,
        records: List[Dict],
        batch_size: int = 20,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, ResourceInfo]:
        out: Dict[str, ResourceInfo] = {}
        n = len(records)
        for i in range(0, n, batch_size):
            chunk = records[i:i + batch_size]
            if self.use_llm:
                try:
                    out.update(self._llm_extract(chunk))
                except Exception as e:        # malformed JSON / API hiccup
                    logger.warning("LLM resource extraction failed (%s); using keywords.", e)
                    out.update(self._kw_extract(chunk))
            else:
                out.update(self._kw_extract(chunk))
            if progress_cb:
                progress_cb(f"extracted {min(i + batch_size, n)}/{n}", min((i + batch_size) / n, 1.0))
        # ensure every tweet has an entry
        for r in records:
            out.setdefault(r["tweet_id"], ResourceInfo(tweet_id=r["tweet_id"]))
        return out


# ── tidy frame + matching/alerts (used by the dashboard) ─────

def resource_dataframe(records: List[Dict], resinfo: Dict[str, ResourceInfo]) -> pd.DataFrame:
    """One row per (tweet × resource), enriched with stance/location/time."""
    rows = []
    by_id = {r["tweet_id"]: r for r in records}
    for tid, ri in resinfo.items():
        rec = by_id.get(tid, {})
        resources = ri.resources or (["unspecified"] if ri.stance != "none" else [])
        loc = ri.location or "unknown"
        for res in resources:
            rows.append({
                "tweet_id":  tid,
                "stance":    ri.stance,
                "resource":  res,
                "location":  loc,
                "quantity":  ri.quantity,
                "timestamp": rec.get("timestamp", ""),
                "text":      rec.get("text", ""),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def match_need_availability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate need vs availability per (location, resource) — the basis for
    alerts (unmet needs) and allocation suggestions.
    """
    if df.empty:
        return pd.DataFrame(columns=["location", "resource", "need", "availability", "gap"])
    pivot = (df[df["stance"].isin(["need", "availability"])]
             .pivot_table(index=["location", "resource"], columns="stance",
                          values="tweet_id", aggfunc="count", fill_value=0)
             .reset_index())
    for col in ("need", "availability"):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["gap"] = pivot["need"] - pivot["availability"]
    return pivot.sort_values("gap", ascending=False).reset_index(drop=True)


def unmet_alerts(match_df: pd.DataFrame) -> pd.DataFrame:
    """Rows where need outstrips availability — the actionable alerts."""
    if match_df.empty:
        return match_df
    return match_df[match_df["gap"] > 0].reset_index(drop=True)
