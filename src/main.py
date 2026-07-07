import csv
import html
import os
import json
import re
import threading
import time
from collections import Counter
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

# =========================================================
# 環境設定
# =========================================================
load_dotenv(Path(__file__).resolve().parent / ".env")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
COMPANY_PROXY = os.getenv("COMPANY_PROXY")
NO_PROXY = os.getenv("NO_PROXY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
if COMPANY_PROXY:
    os.environ["HTTP_PROXY"] = COMPANY_PROXY
    os.environ["HTTPS_PROXY"] = COMPANY_PROXY
if NO_PROXY:
    os.environ["NO_PROXY"] = NO_PROXY

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY environment variable.")

if not OLLAMA_BASE_URL:
    print("[warning] OLLAMA_BASE_URL is not set; LLM requests will fail.", flush=True)
if not OLLAMA_MODEL:
    print("[warning] OLLAMA_MODEL is not set; LLM requests will fail.", flush=True)

# =========================================================
# FastAPI
# =========================================================
app = FastAPI(title="Tree Cases Chatbot API (Multi-Filter + Report)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # POC 用，正式記得收窄
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# Ollama OpenAI-compatible client
# =========================================================
http_client = httpx.Client(timeout=120.0, trust_env=True)

llm_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",
    http_client=http_client,
)

# =========================================================
# Supabase client
# =========================================================
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================================================
# Session store
# =========================================================
sessions: Dict[str, List[Dict[str, str]]] = {}

# =========================================================
# Prompt & 常數
# =========================================================
SYSTEM_PROMPT = (
    "你是一個有禮貌、清晰、實事求是的 AI 助手。"
    "你必須使用繁體中文回答。"
    "如果系統以 SYSTEM 訊息提供了案件統計結果和樣本記錄，你必須以該結果為唯一數據來源回答。"
    "不要胡亂估計或虛構資料，不需要自我修正；如統計不足以回答，請明確講出不足之處，並解釋可以點樣再查。"
)

# 判斷係咪同 DB 有關的關鍵字（越粗暴越唔易漏）
DB_KEYWORDS = [
    "案件", "個案", "案",
    "樹", "樹木",
    "地區", "區", "街道", "街",
    "投訴", "承辦商",
    "嚴重", "輕微", "中等",
    "幾多", "多少", "數量", "統計",
    "記錄", "紀錄", "詳細",
    "tc20", "tc19", "tc21",
]

# 防止 context 爆炸：最多傳幾多筆詳細記錄俾 LLM
MAX_RECORDS_FOR_DETAIL = 100
# assistant 回覆寫回 history 時，最多保留幾多字元
MAX_ASSISTANT_HISTORY_CHARS = 1200

# =========================================================
# 報告 template / disclaimer
# =========================================================
REPORT_DISCLAIMER = (
    "本報告由人工智能系統根據當前資料自動生成，"
    "僅供內部參考，並不構成任何專業意見或法律責任承諾。"
)

REPORT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "default": {
        "id": "default",
        "description": "三段式樹木個案報告：1. KPI 區塊、2. 分類/統計摘要圖表、3. 詳細個案表格。",
        "sections": [
            {
                "id": "kpi",
                "title": "關鍵指標",
                "requirements": [
                    "顯示最少 3 個 KPI：例如案件總數、涉及地區數量、承辦商數量、樹木總數。",
                    "可以用 card、grid 或其他 HTML 結構，但要清晰標題 + 數值。",
                ],
            },
            {
                "id": "summary",
                "title": "分類及統計摘要",
                "requirements": [
                    "最少做一個按地區分佈、一個按承辦商分佈嘅摘要。",
                    "你可以用 <table> 或 <ul> 做 summary，"
                    "亦可以用 <canvas> 配合 Chart.js 或自家 <div data-chart=\"...\"> 區塊留俾前端渲染。",
                ],
            },
            {
                "id": "details",
                "title": "詳細個案列表",
                "requirements": [
                    "用 <table> 顯示每一宗個案，包括至少：案件編號、日期、地區、街道、投訴類型、樹木種類、樹木數量、嚴重程度、承辦商、狀態。",
                    "每一行對應一宗個案。",
                ],
            },
        ],
    }
}

# =========================================================
# Helper constants
# =========================================================
DISTRICTS = [
    "中西區", "灣仔", "東區", "南區",
    "油尖旺", "深水埗", "九龍城", "黃大仙", "觀塘",
    "葵青", "荃灣", "屯門", "元朗", "北區", "大埔", "沙田", "西貢", "離島",
]

STATUSES = ["新個案", "跟進中", "已完成", "已轉介"]
SEVERITIES = ["輕微", "中等", "嚴重"]

DISTRICT_ALIASES = {
    "wan chai": "灣仔",
    "wanchai": "灣仔",
}

STATUS_ALIASES = {
    "完成": "已完成",
}

SEVERITY_ALIASES = {
    "嚴重個案": "嚴重",
}

# =========================================================
# Helper functions
# =========================================================
def normalize_text(text: str) -> str:
    s = text.strip().lower()
    for k, v in DISTRICT_ALIASES.items():
        s = s.replace(k, v)
    for k, v in STATUS_ALIASES.items():
        s = s.replace(k, v)
    for k, v in SEVERITY_ALIASES.items():
        s = s.replace(k, v)
    return s


def looks_like_db_query(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    return any(k in t for k in DB_KEYWORDS)


def classify_query_type(user_prompt: str) -> str:
    """
    粗分幾類 DB 問題：
    - aggregate_trees: 問總共幾多「棵」樹
    - detail: 要「詳細記錄 / 詳細資料 / 全部記錄」
    - count: 問「幾多宗 / 幾多單 / 幾多個個案」
    - generic_db: 其他 DB 類問題
    """
    t = user_prompt.strip()
    if any(k in t for k in ["幾多棵", "幾多棵樹", "總共幾多棵", "總共有幾多棵"]):
        return "aggregate_trees"
    if any(k in t for k in ["詳細記錄", "詳細紀錄", "詳細資料", "詳細情況", "全部記錄", "全部案件"]):
        return "detail"
    if any(k in t for k in ["幾多宗", "幾多單", "幾多單個案", "幾多個個案", "幾多個案件"]):
        return "count"
    return "generic_db"


def select_report_template(user_prompt: str) -> str:
    """
    根據 prompt 揀報告 template：
      - 而家只有 default，將來可加其他 layout，再用關鍵字判斷。
    """
    t = user_prompt.lower()
    # example 將來:
    # if "layout1" in t: return "layout1"
    return "default"


REPORT_CASE_FIELDS: List[Tuple[str, str]] = [
    ("case_no", "案件編號"),
    ("case_date", "日期"),
    ("district", "地區"),
    ("street", "街道"),
    ("complaint_type", "投訴類型"),
    ("tree_species", "樹木種類"),
    ("tree_count", "樹木數量"),
    ("severity", "嚴重程度"),
    ("contractor", "承辦商"),
    ("status", "狀態"),
]
CASE_PLACEHOLDER_RE = re.compile(r"\{\{\s*case\.(\w+)\s*\}\}", re.IGNORECASE)
CASE_ROW_TEMPLATE_RE = re.compile(
    r"<tr[^>]*>.*?\{\{\s*case\.\w+.*?</tr>",
    re.DOTALL | re.IGNORECASE,
)


def _expand_case_placeholders(text: str, case: Dict[str, str]) -> str:
    def repl(match: re.Match) -> str:
        return html.escape(case.get(match.group(1), ""), quote=True)

    return CASE_PLACEHOLDER_RE.sub(repl, text)


def build_cases_details_table(rows: List[Dict[str, str]]) -> str:
    header_cells = "".join(f"<th>{label}</th>" for _, label in REPORT_CASE_FIELDS)
    body_rows: List[str] = []
    for case in rows:
        cells = "".join(
            f"<td>{html.escape(case.get(key, ''), quote=True)}</td>"
            for key, _ in REPORT_CASE_FIELDS
        )
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<table class="cases-detail">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def finalize_report_html(html_text: str, rows: List[Dict[str, str]]) -> str:
    """
    Replace LLM template placeholders (e.g. {{case.case_no}}) with real case data.
    Keeps the streamed HTML layout when possible; falls back to a server-built table.
    """
    if not html_text or not rows:
        return html_text
    if "{{" not in html_text:
        return html_text

    finalized = html_text
    row_match = CASE_ROW_TEMPLATE_RE.search(finalized)
    if row_match:
        template_row = row_match.group(0)
        expanded_rows = "\n".join(
            _expand_case_placeholders(template_row, case) for case in rows
        )
        finalized = (
            finalized[: row_match.start()] + expanded_rows + finalized[row_match.end() :]
        )

    if CASE_PLACEHOLDER_RE.search(finalized):
        if len(rows) == 1:
            finalized = _expand_case_placeholders(finalized, rows[0])
        else:
            table_match = re.search(
                r"<table[^>]*>.*?\{\{\s*case\.\w+.*?</table>",
                finalized,
                re.DOTALL | re.IGNORECASE,
            )
            details_table = build_cases_details_table(rows)
            if table_match:
                finalized = (
                    finalized[: table_match.start()]
                    + details_table
                    + finalized[table_match.end() :]
                )
            elif "</body>" in finalized.lower():
                body_end = finalized.lower().rfind("</body>")
                finalized = (
                    finalized[:body_end]
                    + details_table
                    + finalized[body_end:]
                )
            else:
                finalized = finalized + details_table

    if CASE_PLACEHOLDER_RE.search(finalized):
        finalized = CASE_PLACEHOLDER_RE.sub("", finalized)

    return finalized


def extract_districts(text: str) -> List[str]:
    t = normalize_text(text)
    found: List[str] = []
    for d in DISTRICTS:
        if d.lower() in t:
            found.append(d)
    return list(dict.fromkeys(found))


def extract_statuses(text: str) -> List[str]:
    t = normalize_text(text)
    found: List[str] = []
    for s in STATUSES:
        if s.lower() in t:
            found.append(s)
    return list(dict.fromkeys(found))


def extract_severities(text: str) -> List[str]:
    t = normalize_text(text)
    found: List[str] = []
    for s in SEVERITIES:
        if s.lower() in t:
            found.append(s)
    return list(dict.fromkeys(found))


def extract_months(text: str) -> List[int]:
    months: List[int] = []
    for m in re.findall(r"(\d{1,2})月", text):
        month = int(m)
        if 1 <= month <= 12:
            months.append(month)
    return list(dict.fromkeys(months))


def extract_years(text: str) -> List[int]:
    years: List[int] = []
    for m in re.findall(r"(\d{4})年", text):
        y = int(m)
        if 2000 <= y <= 2100:
            years.append(y)
    return list(dict.fromkeys(years))


def extract_exact_dates(text: str) -> List[date]:
    dates: List[date] = []

    for m in re.findall(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text):
        y, mo, d = map(int, m)
        try:
            dates.append(date(y, mo, d))
        except ValueError:
            pass

    for m in re.findall(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text):
        y, mo, d = map(int, m)
        try:
            dates.append(date(y, mo, d))
        except ValueError:
            pass

    seen = set()
    unique_dates: List[date] = []
    for dt in dates:
        if dt not in seen:
            seen.add(dt)
            unique_dates.append(dt)
    return unique_dates


def extract_case_nos(text: str) -> List[str]:
    t = text.upper()
    nos = re.findall(r"TC\d{4,}", t)
    return list(dict.fromkeys(nos))


def extract_values_from_prompt(
    user_prompt: str, rows: List[Dict[str, Any]], col: str
) -> List[str]:
    text = user_prompt.strip()
    if not text:
        return []
    candidates = {safe_str(r.get(col)) for r in rows if r.get(col)}
    found: List[str] = []
    for val in sorted(candidates, key=len, reverse=True):
        if val and val in text:
            found.append(val)
    return list(dict.fromkeys(found))


def safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def fetch_all_cases(limit: int = 300) -> List[Dict[str, Any]]:
    response = (
        supabase.table("tree_cases")
        .select("*")
        .order("case_date", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def trim_history(history: List[Dict[str, str]], max_turns: int = 4) -> List[Dict[str, str]]:
    if len(history) <= max_turns * 2:
        return history
    return history[-max_turns * 2:]


def build_prompt_text(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.get("role", "unknown")
        content = m.get("content", "")
        parts.append(f"[{role.upper()}]\n{content}\n")
    return "\n".join(parts)


# =========================================================
# 精準統計 context（全部欄位支援多選 + 聚合）
# =========================================================
def build_precise_count_context(
    user_prompt: str,
    rows: List[Dict[str, Any]],
    query_type: str = "generic_db",
    max_records: int = MAX_RECORDS_FOR_DETAIL,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    districts = extract_districts(user_prompt)
    statuses = extract_statuses(user_prompt)
    severities = extract_severities(user_prompt)
    months = extract_months(user_prompt)
    years = extract_years(user_prompt)
    exact_dates = extract_exact_dates(user_prompt)
    case_nos = extract_case_nos(user_prompt)

    streets = extract_values_from_prompt(user_prompt, rows, "street")
    complaint_types = extract_values_from_prompt(user_prompt, rows, "complaint_type")
    tree_species_list = extract_values_from_prompt(user_prompt, rows, "tree_species")
    contractors = extract_values_from_prompt(user_prompt, rows, "contractor")

    matched: List[Dict[str, Any]] = []

    for r in rows:
        row_district = safe_str(r.get("district"))
        row_status = safe_str(r.get("status"))
        row_severity = safe_str(r.get("severity"))
        row_case_no = safe_str(r.get("case_no"))
        row_street = safe_str(r.get("street"))
        row_complaint = safe_str(r.get("complaint_type"))
        row_tree_species = safe_str(r.get("tree_species"))
        row_contractor = safe_str(r.get("contractor"))

        if districts and row_district not in districts:
            continue
        if statuses and row_status not in statuses:
            continue
        if severities and row_severity not in severities:
            continue
        if case_nos and row_case_no not in case_nos:
            continue
        if streets and row_street not in streets:
            continue
        if complaint_types and row_complaint not in complaint_types:
            continue
        if tree_species_list and row_tree_species not in tree_species_list:
            continue
        if contractors and row_contractor not in contractors:
            continue

        d_obj: datetime | None = None
        raw_date = r.get("case_date")
        if raw_date:
            try:
                d_obj = datetime.fromisoformat(str(raw_date))
            except Exception:
                d_obj = None

        if exact_dates and d_obj:
            if d_obj.date() not in exact_dates:
                continue
        else:
            if years and d_obj and d_obj.year not in years:
                continue
            if months and d_obj and d_obj.month not in months:
                continue

        matched.append(r)

    total_tree_count = 0
    for r in matched:
        try:
            total_tree_count += int(r.get("tree_count") or 0)
        except (TypeError, ValueError):
            continue

    filters_debug = {
        "matched_count": len(matched),
        "years": years,
        "months": months,
        "exact_dates": [d.isoformat() for d in exact_dates],
        "districts": districts,
        "statuses": statuses,
        "severities": severities,
        "case_nos": case_nos,
        "streets": streets,
        "complaint_types": complaint_types,
        "tree_species": tree_species_list,
        "contractors": contractors,
        "query_type": query_type,
        "total_tree_count": total_tree_count,
    }

    parts: List[str] = []
    parts.append("以下內容由後端 Python 根據資料庫作精準統計，回答時請優先使用。")
    parts.append(f"匹配案件總數: {len(matched)}")

    if query_type == "aggregate_trees" or tree_species_list:
        parts.append(f"樹木總數: {total_tree_count}")

    years_text = "、".join(map(str, years)) if years else "無"
    months_text = "、".join(map(str, months)) if months else "無"
    dates_text = "、".join(d.isoformat() for d in exact_dates) if exact_dates else "無"
    districts_text = "、".join(districts) if districts else "無"
    statuses_text = "、".join(statuses) if statuses else "無"
    severities_text = "、".join(severities) if severities else "無"
    case_no_text = "、".join(case_nos) if case_nos else "無"
    streets_text = "、".join(streets) if streets else "無"
    complaint_text = "、".join(complaint_types) if complaint_types else "無"
    tree_species_text = "、".join(tree_species_list) if tree_species_list else "無"
    contractors_text = "、".join(contractors) if contractors else "無"

    parts.append(f"年份條件: {years_text}")
    parts.append(f"月份條件: {months_text}")
    parts.append(f"精確日期條件: {dates_text}")
    parts.append(f"地區條件: {districts_text}")
    parts.append(f"狀態條件: {statuses_text}")
    parts.append(f"嚴重程度條件: {severities_text}")
    parts.append(f"案件編號條件: {case_no_text}")
    parts.append(f"街道條件: {streets_text}")
    parts.append(f"投訴類型條件: {complaint_text}")
    parts.append(f"樹木種類條件: {tree_species_text}")
    parts.append(f"承辦商條件: {contractors_text}")

    if matched and max_records > 0:
        top_districts = Counter(
            [safe_str(r.get("district")) for r in matched if safe_str(r.get("district"))]
        ).most_common(5)
        top_streets = Counter(
            [safe_str(r.get("street")) for r in matched if safe_str(r.get("street"))]
        ).most_common(5)
        top_contractors = Counter(
            [safe_str(r.get("contractor")) for r in matched if safe_str(r.get("contractor"))]
        ).most_common(5)

        if top_districts:
            parts.append("常見地區: " + "，".join([f"{n}({c})" for n, c in top_districts]))
        if top_streets:
            parts.append("常見街道: " + "，".join([f"{n}({c})" for n, c in top_streets]))
        if top_contractors:
            parts.append("常見承辦商: " + "，".join([f"{n}({c})" for n, c in top_contractors]))

        sample_lines: List[str] = []
        limit = min(max_records, len(matched))
        for r in matched[:limit]:
            sample_lines.append(
                " | ".join(
                    [
                        f"案件編號:{safe_str(r.get('case_no'))}",
                        f"日期:{safe_str(r.get('case_date'))}",
                        f"地區:{safe_str(r.get('district'))}",
                        f"街道:{safe_str(r.get('street'))}",
                        f"投訴類型:{safe_str(r.get('complaint_type'))}",
                        f"樹木種類:{safe_str(r.get('tree_species'))}",
                        f"樹木數量:{safe_str(r.get('tree_count'))}",
                        f"嚴重程度:{safe_str(r.get('severity'))}",
                        f"承辦商:{safe_str(r.get('contractor'))}",
                        f"狀態:{safe_str(r.get('status'))}",
                    ]
                )
            )

        if limit < len(matched):
            parts.append(
                f"以下為前 {limit} 筆符合條件的記錄（共 {len(matched)} 筆）:\n"
                + "\n".join(sample_lines)
            )
        else:
            parts.append("以下為所有符合條件的記錄:\n" + "\n".join(sample_lines))
    elif not matched:
        parts.append("沒有符合條件的案件。")

    return "\n".join(parts), filters_debug, matched


# =========================================================
# SSE helper
# =========================================================
def sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# Windows: WinError 233/64 often means the browser closed the SSE connection
# or the console pipe broke (e.g. CMD window closed).
_CLIENT_DISCONNECT_WINERRORS = {233, 64, 10054}


def _safe_print(message: str) -> None:
    try:
        print(message, flush=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) not in _CLIENT_DISCONNECT_WINERRORS:
            raise


def _is_client_disconnect(exc: BaseException) -> bool:
    if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
        return True
    if isinstance(exc, OSError):
        winerror = getattr(exc, "winerror", None)
        if winerror in _CLIENT_DISCONNECT_WINERRORS:
            return True
        if exc.errno in _CLIENT_DISCONNECT_WINERRORS:
            return True
    return False


# =========================================================
# Chat interaction CSV log (server-side only)
# Default: logs/chat_log_YYYY-MM-DD.csv (daily rotation)
# Override with CHAT_LOG_DIR / CHAT_LOG_FILE in .env
# =========================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_chat_log_dir() -> Path:
    raw = (os.getenv("CHAT_LOG_DIR") or "logs").strip() or "logs"
    path = Path(raw)
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


def _chat_log_base_name() -> str:
    raw = (os.getenv("CHAT_LOG_FILE") or "chat_log").strip() or "chat_log"
    if raw.lower().endswith(".csv"):
        raw = raw[:-4]
    return raw


CHAT_LOG_DIR = _resolve_chat_log_dir()
CHAT_LOG_BASE = _chat_log_base_name()


def _chat_log_csv_for_date(log_date: date) -> Path:
    return CHAT_LOG_DIR / f"{CHAT_LOG_BASE}_{log_date.isoformat()}.csv"


def _parse_log_date(fetched_at: str) -> date:
    try:
        return datetime.strptime(fetched_at, "%Y-%m-%d %H:%M:%S").date()
    except ValueError:
        return date.today()


def _chat_log_display_path(log_path: Optional[Path] = None) -> str:
    path = log_path or _chat_log_csv_for_date(date.today())
    try:
        return path.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


_safe_print(
    f"[chat log] daily rotation: {_chat_log_display_path(CHAT_LOG_DIR / f'{CHAT_LOG_BASE}_YYYY-MM-DD.csv')}",
)
_chat_log_lock = threading.Lock()
CHAT_LOG_COLUMNS = [
    "Fetched at",
    "User input",
    "Thinking",
    "Answer",
    "Prompt to LLM",
    "Handling time",
    "Model used",
]


def format_handling_time(ms: int) -> str:
    if ms < 1000:
        return f"{ms} ms"
    seconds = ms / 1000
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes} min {remaining_seconds:.1f} s"


def trim_log_field(text: str) -> str:
    if not text:
        return ""
    lines = text.lstrip(" \t\r\n").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def _csv_needs_header(csv_path: Path) -> bool:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return True
    try:
        with csv_path.open("r", encoding="utf-8-sig") as f:
            first_line = f.readline()
        return "Fetched at" not in first_line
    except OSError:
        return True


def log_chat_turn(
    fetched_at: str,
    user_input: str,
    thinking: str,
    answer: str,
    prompt_to_llm: str,
    handling_time: str,
    model_used: str,
) -> None:
    row = [
        fetched_at,
        user_input,
        trim_log_field(thinking),
        trim_log_field(answer),
        prompt_to_llm,
        handling_time,
        model_used,
    ]
    csv_path = _chat_log_csv_for_date(_parse_log_date(fetched_at))
    with _chat_log_lock:
        CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        write_header = _csv_needs_header(csv_path)
        with csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(
                f,
                delimiter=",",
                quoting=csv.QUOTE_ALL,
                lineterminator="\n",
            )
            if write_header:
                writer.writerow(CHAT_LOG_COLUMNS)
            writer.writerow(row)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
    _safe_print(f"[chat log] row appended -> {_chat_log_display_path(csv_path)}")


# =========================================================
# 主 streaming generator
# =========================================================
def openai_stream_generator(user_prompt: str, session_id: str):
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start_time = time.perf_counter()
    assistant_answer = ""
    assistant_thinking = ""
    prompt_text_logged = ""
    chat_logged = False

    def elapsed_ms() -> int:
        return round((time.perf_counter() - start_time) * 1000)

    def schedule_chat_log(
        answer: str,
        thinking: Optional[str] = None,
        prompt_to_llm: Optional[str] = None,
        model_used: Optional[str] = None,
    ) -> None:
        nonlocal chat_logged
        if chat_logged:
            return
        chat_logged = True
        handling_time = format_handling_time(elapsed_ms())
        thinking_val = assistant_thinking if thinking is None else thinking
        prompt_val = prompt_text_logged if prompt_to_llm is None else prompt_to_llm
        model_val = model_used or OLLAMA_MODEL or "unknown"

        def _write_log_row() -> None:
            try:
                # Brief delay so the SSE "done" event reaches the browser before
                # CSV creation (which can trigger Live Server reload on Windows).
                time.sleep(0.4)
                log_chat_turn(
                    fetched_at=fetched_at,
                    user_input=user_prompt,
                    thinking=thinking_val,
                    answer=answer,
                    prompt_to_llm=prompt_val,
                    handling_time=handling_time,
                    model_used=model_val,
                )
            except Exception as log_err:
                import traceback
                _safe_print(f"chat log write failed: {log_err}")
                _safe_print(traceback.format_exc())

        threading.Thread(target=_write_log_row, daemon=True).start()

    try:
        history = trim_history(sessions.get(session_id, []))
        is_db_query = looks_like_db_query(user_prompt)
        filters_debug: Dict[str, Any] = {}
        rows_loaded = 0
        query_type = None

        # ========== DB 問題 ==========
        if is_db_query:
            rows = fetch_all_cases(limit=300)
            rows_loaded = len(rows)
            query_type = classify_query_type(user_prompt)

            if query_type == "detail":
                max_records = MAX_RECORDS_FOR_DETAIL
            elif query_type == "aggregate_trees":
                max_records = 0
            elif query_type == "count":
                max_records = 0
            else:
                max_records = 100

            summary_text, filters_debug, matched = build_precise_count_context(
                user_prompt, rows, query_type=query_type, max_records=max_records
            )

            # ---- 純數量 / 樹木總數：backend 直接回答 ----
            if query_type in ("count", "aggregate_trees"):
                dbg_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": summary_text},
                    {"role": "user", "content": user_prompt},
                ]
                prompt_text = build_prompt_text(dbg_messages)
                prompt_text_logged = prompt_text
                yield sse_event({"type": "prompt", "text": prompt_text})

                matched_count = filters_debug.get("matched_count", 0)
                streets = filters_debug.get("streets") or []
                districts = filters_debug.get("districts") or []
                severities = filters_debug.get("severities") or []
                tree_species = filters_debug.get("tree_species") or []
                total_tree_count = filters_debug.get("total_tree_count", 0)

                cond_parts: List[str] = []
                if streets:
                    cond_parts.append(f"街道為「{'、'.join(streets)}」")
                if districts:
                    cond_parts.append(f"地區為「{'、'.join(districts)}」")
                if severities:
                    cond_parts.append(f"嚴重程度為「{'、'.join(severities)}」")
                if tree_species:
                    cond_parts.append(f"樹木種類為「{'、'.join(tree_species)}」")

                cond_text = "、".join(cond_parts) if cond_parts else "所有條件"

                if query_type == "count":
                    if matched_count == 0:
                        answer_text = "根據後端精準統計，沒有符合你條件的案件。"
                    else:
                        answer_text = (
                            f"根據後端精準統計，符合{cond_text}的案件一共有 {matched_count} 單。"
                        )
                else:  # aggregate_trees
                    if matched_count == 0 or total_tree_count == 0:
                        answer_text = "根據後端精準統計，沒有符合你條件的相關樹木記錄。"
                    else:
                        if tree_species:
                            sp = "、".join(tree_species)
                            answer_text = (
                                f"根據後端精準統計，符合{cond_text}的樹木當中，"
                                f"「{sp}」合共共有 {total_tree_count} 棵。"
                            )
                        else:
                            answer_text = (
                                f"根據後端精準統計，符合{cond_text}的樹木合共共有 {total_tree_count} 棵。"
                            )

                yield sse_event({
                    "type": "debug",
                    "is_db_query": True,
                    "query_type": query_type,
                    "rows_loaded": rows_loaded,
                    "filters": filters_debug,
                    "answered_by": "backend",
                    "report_mode": False,
                })

                yield sse_event({"type": "answer", "text": answer_text})

                # 更新 history
                history = sessions.get(session_id, [])
                history.append({"role": "user", "content": user_prompt})
                short_assistant = answer_text#[:MAX_ASSISTANT_HISTORY_CHARS]
                history.append({"role": "assistant", "content": short_assistant})
                sessions[session_id] = trim_history(history, max_turns=4)

                yield sse_event({
                    "type": "done",
                    "is_db_query": True,
                    "query_type": query_type,
                    "thinking_chars": 0,
                    "answer_chars": len(answer_text),
                    "answered_by": "backend",
                    "report_mode": False,
                })
                schedule_chat_log(
                    answer_text,
                    thinking="",
                    prompt_to_llm=prompt_text,
                    model_used="backend",
                )
                return

            # ---- detail + 報告：LLM 砌 HTML，前端 download ----
            want_report = ("報告" in user_prompt) or ("report" in user_prompt.lower())
            if query_type == "detail" and want_report:
                template_id = select_report_template(user_prompt)
                template_def = REPORT_TEMPLATES.get(template_id, REPORT_TEMPLATES["default"])

                compact_rows = []
                for r in matched:
                    compact_rows.append({
                        "case_no": safe_str(r.get("case_no")),
                        "case_date": safe_str(r.get("case_date")),
                        "district": safe_str(r.get("district")),
                        "street": safe_str(r.get("street")),
                        "complaint_type": safe_str(r.get("complaint_type")),
                        "tree_species": safe_str(r.get("tree_species")),
                        "tree_count": safe_str(r.get("tree_count")),
                        "severity": safe_str(r.get("severity")),
                        "contractor": safe_str(r.get("contractor")),
                        "status": safe_str(r.get("status")),
                    })

                data_json = json.dumps(compact_rows, ensure_ascii=False)

                template_system = (
                    "你現在要根據系統提供嘅案件資料，生成一份 **完整 HTML 報告**。"
                    "報告內容全部用繁體中文，並且 **只輸出 <html>...</html>**，"
                    "不得在 <html> 標籤前後輸出任何文字（包括解釋、注意事項、估算說明等）。\n\n"
                    f"你必須使用名為「{template_def['id']}」的報告版型：{template_def['description']}\n"
                    "版型結構與要求如下：\n"
                )
                for sec in template_def["sections"]:
                    template_system += f"- 區塊 {sec['id']}（{sec['title']}）:\n"
                    for req in sec["requirements"]:
                        template_system += f"  - {req}\n"

                template_system += (
                    "\n整體 HTML 要包含 <head>（含 <meta charset=\"utf-8\"> 和 <title>）以及 <body>。\n"
                    "你可以自由設計 class 名稱、布局和樣式，但必須清楚分出三個主要區塊：KPI、分類/統計摘要、詳細列表。\n"
                    "你可以選擇：\n"
                    "- 直接在 HTML 入面引用 Chart.js CDN 並產生 <canvas> + <script> 初始化圖表，"
                    "數據要來自系統提供的 JSON；或者\n"
                    "- 只產生結構化 HTML（例如 <div data-chart=\"district\">）留俾前端再渲染。\n\n"
                    "所有 KPI 及圖表用到的數值（例如案件總數、樹木總數、各地區／承辦商案件數）"
                    "必須直接使用系統提供的統計結果或 JSON 資料，不可以自行估算或修改，"
                    "亦不得寫出「估計」「約」「可能」之類字眼。\n"
                    "不要評論「案件總數」與 JSON 記錄數目是否一致，只需如實顯示統計數值。\n\n"
                    "在報告的最底部，**必須** 包含一個 <footer> 區塊，其內文必須完整逐字包含以下免責聲明（不可增刪或改字）：\n"
                    f"「{REPORT_DISCLAIMER}」\n"
                    "你可以在 footer 內加其他資訊，但不得修改上述免責聲明文字。\n"
                )

                stats_system = (
                    "以下是後端 Python 已計算好嘅統計 summary，你在計算 KPI 和圖表時必須以此為準：\n"
                    + summary_text
                )
                data_system = (
                    "以下是所有符合條件案件的 JSON 陣列（每個元素代表一宗案件），"
                    "你可以用嚟計算 KPI / 繪製圖表 / 生成詳細表格：\n"
                    + data_json
                )

                messages: List[Dict[str, str]] = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "system", "content": template_system},
                    {"role": "system", "content": stats_system},
                    {"role": "system", "content": data_system},
                ]
                messages.extend(history)
                messages.append({
                    "role": "user",
                    "content": "請根據以上統計 summary 及 JSON 資料，生成一份符合指定版型的完整 HTML 報告。",
                })

                prompt_text = build_prompt_text(messages)
                prompt_text_logged = prompt_text
                yield sse_event({"type": "prompt", "text": prompt_text})

                yield sse_event({
                    "type": "debug",
                    "is_db_query": True,
                    "query_type": query_type,
                    "rows_loaded": rows_loaded,
                    "filters": filters_debug,
                    "report_template": template_id,
                    "answered_by": "llm-html-report",
                    "report_mode": True,
                })

                stream = llm_client.chat.completions.create(
                    model=OLLAMA_MODEL,
                    messages=messages,
                    stream=True,
                    temperature=1.0,          
                    frequency_penalty=0.0,    
                    presence_penalty=0.0
                    # temperature=0.2,
                )

                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    reasoning_text = getattr(delta, "reasoning", None)
                    if reasoning_text:
                        assistant_thinking += reasoning_text
                        yield sse_event({"type": "thinking", "text": reasoning_text})

                    content_text = delta.content or ""
                    if content_text:
                        assistant_answer += content_text
                        # 對前端而言，呢啲 answer token 就係 HTML string 片段
                        yield sse_event({"type": "answer", "text": content_text})

                history = sessions.get(session_id, [])
                history.append({"role": "user", "content": user_prompt})
                short_assistant = (assistant_answer or assistant_thinking)#[:MAX_ASSISTANT_HISTORY_CHARS]
                history.append({"role": "assistant", "content": short_assistant})
                sessions[session_id] = trim_history(history, max_turns=4)

                finalized_report_html = finalize_report_html(assistant_answer, compact_rows)
                yield sse_event({
                    "type": "done",
                    "is_db_query": True,
                    "query_type": query_type,
                    "thinking_chars": len(assistant_thinking),
                    "answer_chars": len(assistant_answer),
                    "answered_by": "llm-html-report",
                    "report_mode": True,
                    "filename": "tree_cases_report.html",
                    "report_html": finalized_report_html,
                })
                schedule_chat_log(assistant_answer)
                return

            # ---- 其他 DB 類（detail 但無報告 / generic）→ 普通 Q&A，用 LLM ----
            context_text = (
                "以下是根據你問題，後端已經在 Supabase tree_cases 表中做好的精準統計和（可能截斷的）樣本記錄。"
                "你必須以此為根據，用繁體中文直接回答用戶的問題。"
                "如果匹配案件總數為 0，請清楚講明並解釋可能原因或下一步建議。\n"
                + summary_text
            )
            messages: List[Dict[str, str]] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "system", "content": context_text},
            ]
            messages.extend(history)
            messages.append({"role": "user", "content": user_prompt})

            prompt_text = build_prompt_text(messages)
            prompt_text_logged = prompt_text
            yield sse_event({"type": "prompt", "text": prompt_text})

            yield sse_event({
                "type": "debug",
                "is_db_query": True,
                "query_type": query_type,
                "rows_loaded": rows_loaded,
                "filters": filters_debug,
                "answered_by": "llm",
                "report_mode": False,
            })

            stream = llm_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                temperature=0.2,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                reasoning_text = getattr(delta, "reasoning", None)
                if reasoning_text:
                    assistant_thinking += reasoning_text
                    yield sse_event({"type": "thinking", "text": reasoning_text})

                content_text = delta.content or ""
                if content_text:
                    assistant_answer += content_text
                    yield sse_event({"type": "answer", "text": content_text})

        # ========== 非 DB 問題：普通 chat ==========
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.extend(history)
            messages.append({"role": "user", "content": user_prompt})

            prompt_text = build_prompt_text(messages)
            prompt_text_logged = prompt_text
            yield sse_event({"type": "prompt", "text": prompt_text})

            yield sse_event({
                "type": "debug",
                "is_db_query": False,
                "query_type": None,
                "rows_loaded": 0,
                "filters": {},
                "answered_by": "llm",
                "report_mode": False,
            })

            stream = llm_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                temperature=0.2,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                reasoning_text = getattr(delta, "reasoning", None)
                if reasoning_text:
                    assistant_thinking += reasoning_text
                    yield sse_event({"type": "thinking", "text": reasoning_text})

                content_text = delta.content or ""
                if content_text:
                    assistant_answer += content_text
                    yield sse_event({"type": "answer", "text": content_text})

        # ===== 更新 history（通用收尾）=====
        history = sessions.get(session_id, [])
        history.append({"role": "user", "content": user_prompt})

        short_assistant = (assistant_answer or assistant_thinking)#[:MAX_ASSISTANT_HISTORY_CHARS]
        if short_assistant:
            history.append({"role": "assistant", "content": short_assistant})

        sessions[session_id] = trim_history(history, max_turns=4)

        yield sse_event({
            "type": "done",
            "is_db_query": is_db_query,
            "query_type": query_type,
            "thinking_chars": len(assistant_thinking),
            "answer_chars": len(assistant_answer),
            "answered_by": "llm",
            "report_mode": False,
        })
        schedule_chat_log(assistant_answer)

    except Exception as e:
        if _is_client_disconnect(e):
            return
        yield sse_event({"type": "error", "error": str(e)})
        schedule_chat_log(assistant_answer or f"[ERROR] {e}")
    finally:
        if not chat_logged:
            schedule_chat_log(assistant_answer or "[INCOMPLETE]")


async def _stream_chat_events(request: Request, user_prompt: str, session_id: str):
    gen = openai_stream_generator(user_prompt, session_id)
    try:
        for chunk in gen:
            if await request.is_disconnected():
                break
            try:
                yield chunk
            except (BrokenPipeError, ConnectionResetError, OSError) as exc:
                if _is_client_disconnect(exc):
                    break
                raise
    except (BrokenPipeError, ConnectionResetError, OSError) as exc:
        if _is_client_disconnect(exc):
            pass
        else:
            raise
    finally:
        gen.close()


# =========================================================
# Routes
# =========================================================
@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "tree-cases-chatbot-multi-filter-report",
        "chat_log_enabled": True,
        "chat_log_rotation": "daily",
        "chat_log_csv": _chat_log_display_path(),
        "chat_log_pattern": f"{CHAT_LOG_BASE}_YYYY-MM-DD.csv",
    }


@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    user_prompt = (data.get("prompt") or "").strip()
    session_id = (data.get("session_id") or "default").strip()

    if not user_prompt:
        return JSONResponse({"error": "prompt is empty"}, status_code=400)

    return StreamingResponse(
        _stream_chat_events(request, user_prompt, session_id),
        media_type="text/event-stream",
    )


@app.post("/api/reset")
async def reset_session(request: Request):
    data = await request.json()
    session_id = (data.get("session_id") or "default").strip()
    sessions.pop(session_id, None)
    return {"ok": True, "session_id": session_id, "message": "session cleared"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)
