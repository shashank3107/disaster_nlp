"""
Geocoding for the resource dashboard map.

Resource extraction gives us place *names* ("Chennai", "Houston"), not
coordinates. To plot affected areas on a map we resolve each name to (lat, lon):

  1. A built-in gazetteer of common Indian cities/states + the CrisisMMD disaster
     locations — instant, offline, no rate limits.
  2. Nominatim (OpenStreetMap) for anything not in the gazetteer — online and
     rate-limited to ~1 req/s, so results are cached to a JSON file on disk and
     reused across runs.

Everything degrades gracefully: if geopy is missing or a lookup fails, the name is
simply skipped (it won't appear on the map) and the rest of the dashboard is
unaffected.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).resolve().parents[2] / ".geocode_cache.json"

# Lightweight gazetteer: name -> (lat, lon). Lower-cased keys.
# Covers major Indian cities/states (flood/cyclone-prone) and the seven CrisisMMD
# events, so the common cases never hit the network.
_GAZETTEER: Dict[str, Tuple[float, float]] = {
    # India — cities
    "chennai": (13.0827, 80.2707), "mumbai": (19.0760, 72.8777),
    "delhi": (28.6139, 77.2090), "new delhi": (28.6139, 77.2090),
    "kolkata": (22.5726, 88.3639), "bengaluru": (12.9716, 77.5946),
    "bangalore": (12.9716, 77.5946), "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567), "ahmedabad": (23.0225, 72.5714),
    "surat": (21.1702, 72.8311), "jaipur": (26.9124, 75.7873),
    "patna": (25.5941, 85.1376), "guwahati": (26.1445, 91.7362),
    "kochi": (9.9312, 76.2673), "cochin": (9.9312, 76.2673),
    "thiruvananthapuram": (8.5241, 76.9366), "kozhikode": (11.2588, 75.7804),
    "visakhapatnam": (17.6868, 83.2185), "bhubaneswar": (20.2961, 85.8245),
    "cuttack": (20.4625, 85.8830), "vijayawada": (16.5062, 80.6480),
    "coimbatore": (11.0168, 76.9558), "madurai": (9.9252, 78.1198),
    "nagpur": (21.1458, 79.0882), "lucknow": (26.8467, 80.9462),
    "kanpur": (26.4499, 80.3319), "varanasi": (25.3176, 82.9739),
    "srinagar": (34.0837, 74.7973), "dehradun": (30.3165, 78.0322),
    "shimla": (31.1048, 77.1734), "raipur": (21.2514, 81.6296),
    "ranchi": (23.3441, 85.3096), "bhopal": (23.2599, 77.4126),
    "indore": (22.7196, 75.8577), "amritsar": (31.6340, 74.8723),
    # India — states / regions
    "kerala": (10.8505, 76.2711), "tamil nadu": (11.1271, 78.6569),
    "karnataka": (15.3173, 75.7139), "maharashtra": (19.7515, 75.7139),
    "gujarat": (22.2587, 71.1924), "rajasthan": (27.0238, 74.2179),
    "assam": (26.2006, 92.9376), "bihar": (25.0961, 85.3131),
    "odisha": (20.9517, 85.0985), "orissa": (20.9517, 85.0985),
    "west bengal": (22.9868, 87.8550), "uttar pradesh": (26.8467, 80.9462),
    "uttarakhand": (30.0668, 79.0193), "telangana": (18.1124, 79.0193),
    "andhra pradesh": (15.9129, 79.7400), "punjab": (31.1471, 75.3412),
    "himachal pradesh": (31.1048, 77.1734), "jammu and kashmir": (33.7782, 76.5762),
    "madhya pradesh": (22.9734, 78.6569), "chhattisgarh": (21.2787, 81.8661),
    "jharkhand": (23.6102, 85.2799), "goa": (15.2993, 74.1240),
    # CrisisMMD events / international
    "houston": (29.7604, -95.3698), "texas": (31.9686, -99.9018),
    "florida": (27.6648, -81.5158), "puerto rico": (18.2208, -66.5901),
    "mexico city": (19.4326, -99.1332), "mexico": (23.6345, -102.5528),
    "california": (36.7783, -119.4179), "sri lanka": (7.8731, 80.7718),
    "srilanka": (7.8731, 80.7718), "colombo": (6.9271, 79.8612),
    "iran": (32.4279, 53.6880), "iraq": (33.2232, 43.6793),
}


def _load_cache() -> Dict[str, Optional[List[float]]]:
    try:
        return json.loads(_CACHE_PATH.read_text())
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Optional[List[float]]]) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(cache))
    except Exception as e:
        logger.warning("Could not persist geocode cache: %s", e)


def _nominatim_lookup(name: str) -> Optional[Tuple[float, float]]:
    try:
        from geopy.geocoders import Nominatim
        geocoder = Nominatim(user_agent="disaster_nlp_dashboard")
        loc = geocoder.geocode(name, timeout=10)
        time.sleep(1.0)  # respect Nominatim's ~1 req/s usage policy
        if loc:
            return (loc.latitude, loc.longitude)
    except Exception as e:
        logger.warning("Nominatim lookup failed for %r: %s", name, e)
    return None


def geocode(name: str) -> Optional[Tuple[float, float]]:
    """Resolve one place name to (lat, lon); None if it can't be found."""
    if not name:
        return None
    key = name.strip().lower()
    if not key or key == "unknown":
        return None
    if key in _GAZETTEER:
        return _GAZETTEER[key]
    return None  # single-name path: use geocode_many for network + caching


def geocode_many(names: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Resolve many place names, using the gazetteer first and a disk-cached
    Nominatim fallback for the rest. Returns only the names that resolved.
    """
    out: Dict[str, Tuple[float, float]] = {}
    cache = _load_cache()
    cache_dirty = False

    for raw in names:
        key = (raw or "").strip().lower()
        if not key or key == "unknown" or key in out:
            continue
        if key in _GAZETTEER:
            out[raw] = _GAZETTEER[key]
            continue
        if key in cache:
            if cache[key] is not None:
                out[raw] = (cache[key][0], cache[key][1])
            continue
        coords = _nominatim_lookup(key)
        cache[key] = [coords[0], coords[1]] if coords else None
        cache_dirty = True
        if coords:
            out[raw] = coords

    if cache_dirty:
        _save_cache(cache)
    return out
