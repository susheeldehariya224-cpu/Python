
# Student-friendly: LLM->JSON-intent -> safe SQL -> results + Hinglish explanation + persistent history
# Simple UI: one input, one button. History stored in DB and can be re-run.
# Run: ensure student.db exists (run your SQL.py once) and set GOOGLE_API_KEY in .env
import os
from dotenv import load_dotenv
load_dotenv()

import re
import json
import sqlite3
from typing import List, Any, Dict
from datetime import datetime

import pandas as pd
import streamlit as st
import google.generativeai as genai

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DB_PATH = "student.db"

# ------------------ Simple prompts ------------------
INTENT_PROMPT = """
You are a helpful assistant for converting a user's English question about the STUDENT table
into a JSON intent. The STUDENT table has columns: NAME, CLASS, SECTION, MARKS.

Return ONLY a JSON object (no extra text) with these keys:
- action: "select"
- columns: list of columns to return, or ["*"] for all
- where: list of conditions, each is [column, operator, value]
- order_by: optional [column, "asc"/"desc"]
- limit: optional integer

Use exact column names. Example:
User: "How many students are in AI?"
Output: { "action":"select", "columns":["COUNT(*)"], "where":[["CLASS","=","AI"]] }
"""

EXPLAIN_PROMPT = """
Explain the SQL below in simple Hinglish (mix of Hindi and English) in 2 short sentences, and then give a 1-line summary of the top rows.

SQL:
{sql}

Top rows (JSON):
{rows_json}

Return only the explanation text in Hinglish.
"""

# ------------------ Helpers ------------------
def call_gemini(prompt: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        resp = model.generate_content([prompt])
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"[ERROR_CALLING_MODEL] {e}"

def extract_json(text: str) -> str:
    m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        raise ValueError("Model response mein JSON nahi mila.")
    return m.group(1)

def get_intent(question: str) -> Dict:
    raw = call_gemini(INTENT_PROMPT + "\nUser: " + question)
    if raw.startswith("[ERROR_CALLING_MODEL]"):
        raise RuntimeError(raw)
    js = extract_json(raw)
    return json.loads(js)

def get_table_columns(db=DB_PATH) -> List[str]:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(STUDENT)")
    rows = cur.fetchall()
    conn.close()
    return [r[1] for r in rows]

def build_sql_from_intent(intent: Dict, allowed_cols: List[str]) -> (str, List[Any]):
    if intent.get("action") != "select":
        raise ValueError("Only select action supported.")

    cols = intent.get("columns", ["*"])
    safe_cols = []
    for c in cols:
        if isinstance(c, str) and c.strip().upper() == "COUNT(*)":
            safe_cols.append("COUNT(*)")
            continue
        if c == "*":
            safe_cols = ["*"]
            break
        if c not in allowed_cols:
            raise ValueError(f"Invalid column: {c}")
        safe_cols.append(c)
    select_clause = ", ".join(safe_cols) if safe_cols != [""] else ""

    where = intent.get("where", []) or []
    where_parts = []
    params: List[Any] = []
    allowed_ops = {"=", ">", "<", ">=", "<=", "!=", "like"}
    for cond in where:
        if len(cond) != 3:
            raise ValueError("WHERE condition must be [column, operator, value]")
        col, op, val = cond
        if col not in allowed_cols:
            raise ValueError(f"Invalid WHERE column: {col}")
        if str(op).lower() not in allowed_ops:
            raise ValueError(f"Operator not allowed: {op}")
        where_parts.append(f"{col} {op} ?")
        params.append(val)

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    order = intent.get("order_by")
    order_clause = ""
    if order:
        col, direction = order
        if col not in allowed_cols:
            raise ValueError("Invalid ORDER BY column")
        if direction.lower() not in ("asc", "desc"):
            raise ValueError("ORDER direction must be asc or desc")
        order_clause = f"ORDER BY {col} {direction.upper()}"

    limit = intent.get("limit")
    limit_clause = ""
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0 or limit > 1000:
            raise ValueError("limit must be positive integer up to 1000")
        limit_clause = f"LIMIT {limit}"

    sql = f"SELECT {select_clause} FROM STUDENT {where_clause} {order_clause} {limit_clause}".strip()
    return sql, params

def run_query(sql: str, params: List[Any], db=DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df

def explain_in_hinglish(sql: str, df: pd.DataFrame) -> str:
    rows_json = df.head(5).to_dict(orient="records") if not df.empty else []
    prompt = EXPLAIN_PROMPT.format(sql=sql, rows_json=json.dumps(rows_json, ensure_ascii=False))
    resp = call_gemini(prompt)
    if resp.startswith("[ERROR_CALLING_MODEL]"):
        return "Explanation nahi mila (model error)."
    return resp

# ------------------ History helpers (persistent) ------------------
def ensure_history_table(db=DB_PATH):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS QUERY_HISTORY(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            sql TEXT,
            params TEXT,
            created_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_history(question: str, sql: str, params: List[Any], db=DB_PATH):
    ensure_history_table(db)
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("INSERT INTO QUERY_HISTORY(question, sql, params, created_at) VALUES(?,?,?,?)",
                (question, sql, json.dumps(params, ensure_ascii=False), datetime.utcnow()))
    conn.commit()
    conn.close()

def load_history(limit=50, db=DB_PATH) -> pd.DataFrame:
    ensure_history_table(db)
    conn = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT id, question, sql, params, created_at FROM QUERY_HISTORY ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    conn.close()
    return df

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Student SQL Learner", layout="centered")
st.title("Student SQL Learner")
st.write("Type a question in English. App will safely run it on STUDENT table and explain in Hinglish.")

cols = get_table_columns()
st.subheader("Table: STUDENT — Columns")
st.write(cols)

st.markdown("---")
question = st.text_input("English mein question likho (e.g., 'Top 3 students by MARKS in AI')")

if st.button("Chalao"):
    if not question.strip():
        st.warning("Pehle question likho.")
    else:
        st.info("Model se intent le raha hoon — thoda time lag sakta hai...")
        try:
            intent = get_intent(question)
        except Exception as e:
            st.error(f"Intent generation failed: {e}")
            intent = None

        if intent:
            try:
                sql, params = build_sql_from_intent(intent, allowed_cols=cols)
                # Do NOT show raw intent or params in UI (student asked to hide)

                st.subheader("Safe SQL jo app ne banaya hai")
                st.code(sql)

                df = run_query(sql, params)
                st.success(f"Query chal gaya — {len(df)} rows mile.")
                st.dataframe(df)

                # download CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", data=csv, file_name="results.csv", mime="text/csv")

                # save history
                save_history(question, sql, params)

                # explain in Hinglish
                st.info("Ab model se simple Hinglish explanation mang raha hoon...")
                explanation = explain_in_hinglish(sql, df)
                st.subheader("Explanation (Hinglish)")
                st.write(explanation)

            except Exception as e:
                st.error(f"SQL build/run error: {e}")

# ------------------ Show history (persistent) ------------------
st.markdown("---")
st.header("Query History")
try:
    hist = load_history(limit=50)
    if hist.empty:
        st.write("Koi history nahi hai.")
    else:
        # show simple table
        st.dataframe(hist[["id","question","created_at"]])
        chosen = st.selectbox("Select history ID to re-run", options=hist['id'].tolist())
        if st.button("Re-run selected query"):
            row = hist[hist['id'] == chosen].iloc[0]
            st.code(row['sql'])
            params = json.loads(row['params']) if row['params'] else []
            try:
                df2 = run_query(row['sql'], params)
                st.dataframe(df2)
                csv2 = df2.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV of re-run", data=csv2, file_name=f"history_{chosen}_results.csv", mime="text/csv")
                # show explanation for re-run as well
                expl = explain_in_hinglish(row['sql'], df2)
                st.subheader("Explanation (Hinglish) for re-run")
                st.write(expl)
            except Exception as e:
                st.error(f"Re-run failed: {e}")
except Exception as e:
    st.write("History load error:", e)

st.caption("Note: Ye learning/demo app hai. API key ko safe rakho. App sirf SELECT queries allow karta hai.")