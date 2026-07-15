import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("SUPABASE_URL", "http://localhost")
os.environ.setdefault("SUPABASE_KEY", "test")
os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
os.environ.setdefault("OLLAMA_MODEL", "test")

from main import inject_leaflet_map, MAP_INJECT_PLACEHOLDER

row = {
    "case_no": "TC1",
    "district": "Tai Po",
    "street": "Rd",
    "severity": "high",
    "status": "open",
    "latitude": 22.45,
    "longitude": 114.16,
    "case_date": "x",
    "complaint_type": "x",
    "tree_species": "y",
    "tree_count": "1",
    "contractor": "z",
}

sample = f"""<html><head></head><body>
<section id="kpi"><h2>KPI</h2></section>
<section id="summary"><h2>Summary</h2></section>
<section id="cases-map-section"><h2>Map</h2>{MAP_INJECT_PLACEHOLDER}</section>
<section id="details"><h2>Details</h2><table></table></section>
<footer>Disclaimer</footer>
</body></html>"""

out = inject_leaflet_map(sample, [row])
kpi = out.find('id="kpi"')
summary = out.find('id="summary"')
mapsec = out.find("cases-map-section")
details = out.find('id="details"')
footer = out.find("<footer")
canvas = out.find("cases-map-canvas")
assert kpi < summary < mapsec < details < footer
assert mapsec < canvas < details
assert out.lower().count("cases-map-section") == 1
print("OK: map placed between summary and details")
