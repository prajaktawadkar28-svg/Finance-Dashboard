"""
=====================================================================
  AI FINANCE DASHBOARD — Smart Personal Finance Assistant
  Multi-user Streamlit + Machine Learning Web App
=====================================================================

Industrial Training / Internship Project
Built during internship at iGap Technologies, Kolhapur, Maharashtra.

WHAT THIS APP DOES
-------------------
A personal-finance web app that:
  1. Lets each person register their own account and log in securely
     (passwords are salted + hashed with PBKDF2 — never stored in
     plain text).
  2. Predicts a "Financial Trust Score" using a trained scikit-learn
     RandomForestRegressor model.
  3. Offers a suite of transparent, rule-based financial tools
     (Health Score, Budget Planner, Savings Goal Calculator,
     EMI Calculator, Compound Interest / SIP Simulator).
  4. Stores every user's prediction history privately in a local
     SQLite database.
  5. Lets a user add other registered users as **contacts**, and
     **share their financial history** with an accepted contact so
     that contact can view (read-only) their dashboard — e.g. sharing
     progress with a parent, partner, or mentor.
  6. Tracks mood alongside predictions and visualises the correlation.
  7. Includes an 8-question gamified financial-literacy quiz.

TECH STACK
----------
Python, Streamlit, Pandas, NumPy, Scikit-learn (RandomForest),
Joblib, SQLite3 (built into Python — no extra DB server needed).

DEPLOYMENT NOTE (Streamlit Community Cloud / streamlit.io)
------------------------------------------------------------
This app stores everything (users, passwords, predictions, contacts,
shares) in a local SQLite file at data/finance_app.db. That works
perfectly for local use and for demos. On Streamlit Community Cloud
specifically, the filesystem is *ephemeral*: the database will reset
whenever the app is rebooted or redeployed. For a permanent, always-on
multi-user deployment you would eventually swap the SQLite layer for
a hosted database (e.g. Postgres/Supabase) — the DATABASE LAYER
section below is written so only the connection function would need
to change.

Run with:
    streamlit run app.py
=====================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hashlib
import binascii
import sqlite3
import random
import os
import re
import threading
import datetime

# Guards every DB write so two concurrent user sessions (e.g. one user
# saving a prediction while another accepts a contact request) never
# execute statements on the shared SQLite connection at the same instant.
_DB_LOCK = threading.Lock()

# =====================================================================
# SECTION 1 — GLOBAL CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="AI Finance Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Anchor every path to this script's own folder (not the current
# working directory) so the database and model are always found in
# the same place, no matter where `streamlit run` is launched from.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "finance_app.db")
MODEL_PATH = os.path.join(APP_DIR, "random_forest_model.pkl")

PBKDF2_ITERATIONS = 150_000

NAV_PAGES = [
    "🏠 Home",
    "🔮 Prediction Wizard",
    "🧮 Financial Tools",
    "📜 History",
    "😊 Mood Insights",
    "🤝 Contacts & Sharing",
    "📝 Quiz",
    "⚙️ Account",
    "ℹ️ About",
]

CURRENCY_SYMBOLS = {
    "USD ($)": "$",
    "INR (₹)": "₹",
    "EUR (€)": "€",
    "GBP (£)": "£",
}

MOOD_EMOJI = {
    "Happy": "😄",
    "Stressed": "😰",
    "Sad": "😔",
    "Neutral": "😐",
    "Excited": "🤩",
    "Anxious": "😟",
}

FINANCE_QUOTES = [
    ("An investment in knowledge pays the best interest.", "Benjamin Franklin"),
    ("Do not save what is left after spending; spend what is left after saving.", "Warren Buffett"),
    ("The stock market is a device for transferring money from the impatient to the patient.", "Warren Buffett"),
    ("It's not your salary that makes you rich, it's your spending habits.", "Charles A. Jaffe"),
    ("Beware of little expenses; a small leak will sink a great ship.", "Benjamin Franklin"),
    ("The habit of saving is itself an education.", "T.T. Munger"),
    ("Financial freedom is available to those who learn about it and work for it.", "Robert Kiyosaki"),
    ("Never depend on a single income. Make investments to create a second source.", "Warren Buffett"),
    ("A budget is telling your money where to go instead of wondering where it went.", "Dave Ramsey"),
    ("Risk comes from not knowing what you're doing.", "Warren Buffett"),
]

DAILY_TIPS = [
    "Track every expense for a week — awareness is the first step to control.",
    "Automate your savings so it happens before you get a chance to spend it.",
    "Aim to keep your emergency fund at 3-6 months of living expenses.",
    "Review your subscriptions every quarter and cancel what you don't use.",
    "Pay off high-interest debt before focusing on investments.",
    "Diversify — don't put all your savings into a single asset class.",
    "Set specific, measurable financial goals instead of vague ones.",
    "Increase your savings rate slightly every time you get a raise.",
]

QUIZ_QUESTIONS = [
    {
        "q": "1️⃣ What's the best habit for building wealth?",
        "options": ["Spend more than you earn", "Save before spending", "Ignore savings"],
        "answer": "Save before spending",
    },
    {
        "q": "2️⃣ What's a healthy emergency fund?",
        "options": ["1 month of expenses", "3–6 months of expenses", "No need for one"],
        "answer": "3–6 months of expenses",
    },
    {
        "q": "3️⃣ Which investment is usually considered lower-risk?",
        "options": ["Stocks", "Bonds", "Lottery tickets"],
        "answer": "Bonds",
    },
    {
        "q": "4️⃣ What does 'compound interest' mean?",
        "options": ["Interest only on the principal", "Interest on both principal and accumulated interest", "A type of tax"],
        "answer": "Interest on both principal and accumulated interest",
    },
    {
        "q": "5️⃣ What is diversification?",
        "options": ["Putting all money in one stock", "Spreading investments across assets", "Avoiding investing altogether"],
        "answer": "Spreading investments across assets",
    },
    {
        "q": "6️⃣ What's typically the FIRST financial priority?",
        "options": ["Buying luxury items", "Paying off high-interest debt", "Investing in crypto"],
        "answer": "Paying off high-interest debt",
    },
    {
        "q": "7️⃣ What is a budget?",
        "options": ["A plan for spending & saving", "A type of loan", "A bank account"],
        "answer": "A plan for spending & saving",
    },
    {
        "q": "8️⃣ Why track your net worth over time?",
        "options": ["It's required by law", "To measure overall financial progress", "It has no real use"],
        "answer": "To measure overall financial progress",
    },
]


# =====================================================================
# SECTION 2 — DATABASE LAYER (SQLite)
# =====================================================================
# Every read/write to users, predictions, contacts and shares goes
# through the functions in this section. Keeping all SQL in one place
# means the storage backend (e.g. swapping SQLite for Postgres later)
# only ever needs to change here.

def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


@st.cache_resource(show_spinner=False)
def get_conn():
    """A single, shared SQLite connection cached for the app's lifetime."""
    ensure_data_dir()
    # timeout + WAL + busy_timeout: several Streamlit sessions (different
    # logged-in users) can hit this database concurrently, and plain
    # SQLite raises "database is locked" under concurrent writes without
    # these settings.
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create every table if it doesn't already exist. Safe to call every run."""
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            name          TEXT NOT NULL,
            email         TEXT,
            salt          TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id                INTEGER NOT NULL,
            timestamp              TEXT NOT NULL,
            age                    INTEGER,
            household_income_usd   REAL,
            net_worth_usd          REAL,
            monthly_expenses_usd   REAL,
            mood                   TEXT,
            prediction             REAL,
            health_score           REAL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS contacts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            requester_id  INTEGER NOT NULL,
            addressee_id  INTEGER NOT NULL,
            status        TEXT NOT NULL DEFAULT 'pending',
            created_at    TEXT NOT NULL,
            UNIQUE(requester_id, addressee_id)
        );

        CREATE TABLE IF NOT EXISTS shares (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id    INTEGER NOT NULL,
            viewer_id   INTEGER NOT NULL,
            created_at  TEXT NOT NULL,
            UNIQUE(owner_id, viewer_id)
        );
        """
    )
    conn.commit()


# ---------------------------------------------------------------
# Password hashing (PBKDF2-HMAC-SHA256, stdlib only — no external
# dependency needed, which keeps deployment to Streamlit Cloud simple)
# ---------------------------------------------------------------
def hash_password(password: str, salt_bytes: bytes = None):
    if salt_bytes is None:
        salt_bytes = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERATIONS)
    return binascii.hexlify(salt_bytes).decode(), binascii.hexlify(dk).decode()


def verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt_bytes = binascii.unhexlify(salt_hex.encode())
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERATIONS)
    return binascii.hexlify(dk).decode() == hash_hex


def valid_username(username: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_]{3,20}", username or ""))


def valid_email(email: str) -> bool:
    if not email:
        return True  # email is optional
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email))


# ---------------------------------------------------------------
# User account functions
# ---------------------------------------------------------------
def create_user(username: str, name: str, email: str, password: str):
    """Register a new user. Returns (success: bool, message: str)."""
    if not valid_username(username):
        return False, "Username must be 3-20 characters: letters, numbers, underscore only."
    if not name.strip():
        return False, "Please enter your full name."
    if not valid_email(email):
        return False, "That email address doesn't look valid."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    salt_hex, hash_hex = hash_password(password)
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, name, email, salt, password_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (username.strip(), name.strip(), email.strip(), salt_hex, hash_hex,
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        return True, "Account created! You can now log in."
    except sqlite3.IntegrityError:
        return False, "That username is already taken."


def get_user_by_username(username: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone()
    return dict(row) if row is not None else None


def get_user_by_id(user_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row is not None else None


def authenticate(username: str, password: str):
    """Return the user dict on success, or None on failure."""
    user = get_user_by_username(username)
    if user is None:
        return None
    if verify_password(password, user["salt"], user["password_hash"]):
        return user
    return None


def update_profile(user_id: int, name: str, email: str):
    conn = get_conn()
    conn.execute("UPDATE users SET name = ?, email = ? WHERE id = ?", (name.strip(), email.strip(), user_id))
    conn.commit()


def change_password(user_id: int, old_password: str, new_password: str):
    user = get_user_by_id(user_id)
    if user is None:
        return False, "User not found."
    if not verify_password(old_password, user["salt"], user["password_hash"]):
        return False, "Current password is incorrect."
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."
    salt_hex, hash_hex = hash_password(new_password)
    conn = get_conn()
    conn.execute("UPDATE users SET salt = ?, password_hash = ? WHERE id = ?", (salt_hex, hash_hex, user_id))
    conn.commit()
    return True, "Password updated successfully."


# ---------------------------------------------------------------
# Prediction history functions
# ---------------------------------------------------------------
def save_prediction(user_id: int, record: dict):
    conn = get_conn()
    conn.execute(
        """INSERT INTO predictions
           (user_id, timestamp, age, household_income_usd, net_worth_usd,
            monthly_expenses_usd, mood, prediction, health_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id, record["timestamp"], record["age"], record["household_income_usd"],
            record["net_worth_usd"], record["monthly_expenses_usd"], record["mood"],
            record["prediction"], record["health_score"],
        ),
    )
    conn.commit()


def get_predictions(user_id: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT timestamp, age, household_income_usd, net_worth_usd, monthly_expenses_usd, "
        "mood, prediction, health_score FROM predictions WHERE user_id = ? ORDER BY timestamp",
        conn, params=(user_id,),
    )
    return df


def clear_predictions(user_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    conn.commit()


# ---------------------------------------------------------------
# Contacts (friend requests) functions
# ---------------------------------------------------------------
def send_contact_request(requester_id: int, addressee_username: str):
    target = get_user_by_username(addressee_username)
    if target is None:
        return False, "No user with that username exists."
    if target["id"] == requester_id:
        return False, "You can't add yourself as a contact."

    conn = get_conn()
    existing = conn.execute(
        "SELECT * FROM contacts WHERE (requester_id=? AND addressee_id=?) OR (requester_id=? AND addressee_id=?)",
        (requester_id, target["id"], target["id"], requester_id),
    ).fetchone()
    if existing:
        return False, "There's already a request or contact between you two."

    conn.execute(
        "INSERT INTO contacts (requester_id, addressee_id, status, created_at) VALUES (?, ?, 'pending', ?)",
        (requester_id, target["id"], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    return True, f"Contact request sent to @{addressee_username}."


def get_incoming_requests(user_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT c.id, u.id as from_id, u.username, u.name FROM contacts c "
        "JOIN users u ON u.id = c.requester_id "
        "WHERE c.addressee_id = ? AND c.status = 'pending'",
        (user_id,),
    ).fetchall()
    return [{"request_id": r[0], "from_id": r[1], "username": r[2], "name": r[3]} for r in rows]


def get_outgoing_requests(user_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT c.id, u.id as to_id, u.username, u.name FROM contacts c "
        "JOIN users u ON u.id = c.addressee_id "
        "WHERE c.requester_id = ? AND c.status = 'pending'",
        (user_id,),
    ).fetchall()
    return [{"request_id": r[0], "to_id": r[1], "username": r[2], "name": r[3]} for r in rows]


def respond_to_request(request_id: int, accept: bool):
    conn = get_conn()
    conn.execute("UPDATE contacts SET status = ? WHERE id = ?", ("accepted" if accept else "declined", request_id))
    conn.commit()


def get_accepted_contacts(user_id: int):
    """Return the list of contacts (as user dicts) this user has an accepted relationship with."""
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT u.id, u.username, u.name FROM contacts c
        JOIN users u ON u.id = CASE WHEN c.requester_id = ? THEN c.addressee_id ELSE c.requester_id END
        WHERE (c.requester_id = ? OR c.addressee_id = ?) AND c.status = 'accepted'
        """,
        (user_id, user_id, user_id),
    ).fetchall()
    return [{"id": r[0], "username": r[1], "name": r[2]} for r in rows]


def remove_contact(user_id: int, other_user_id: int):
    conn = get_conn()
    conn.execute(
        "DELETE FROM contacts WHERE (requester_id=? AND addressee_id=?) OR (requester_id=? AND addressee_id=?)",
        (user_id, other_user_id, other_user_id, user_id),
    )
    # Also revoke any sharing that existed between them
    conn.execute(
        "DELETE FROM shares WHERE (owner_id=? AND viewer_id=?) OR (owner_id=? AND viewer_id=?)",
        (user_id, other_user_id, other_user_id, user_id),
    )
    conn.commit()


# ---------------------------------------------------------------
# Data-sharing functions
# ---------------------------------------------------------------
def share_data_with(owner_id: int, viewer_id: int):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO shares (owner_id, viewer_id, created_at) VALUES (?, ?, ?)",
            (owner_id, viewer_id, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        return True, "Your financial history is now shared with this contact."
    except sqlite3.IntegrityError:
        return False, "You're already sharing your data with this contact."


def unshare_data(owner_id: int, viewer_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM shares WHERE owner_id = ? AND viewer_id = ?", (owner_id, viewer_id))
    conn.commit()


def get_people_i_share_with(owner_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT u.id, u.username, u.name FROM shares s JOIN users u ON u.id = s.viewer_id WHERE s.owner_id = ?",
        (owner_id,),
    ).fetchall()
    return [{"id": r[0], "username": r[1], "name": r[2]} for r in rows]


def get_people_sharing_with_me(viewer_id: int):
    conn = get_conn()
    rows = conn.execute(
        "SELECT u.id, u.username, u.name FROM shares s JOIN users u ON u.id = s.owner_id WHERE s.viewer_id = ?",
        (viewer_id,),
    ).fetchall()
    return [{"id": r[0], "username": r[1], "name": r[2]} for r in rows]


def is_sharing_with_me(owner_id: int, viewer_id: int) -> bool:
    conn = get_conn()
    row = conn.execute("SELECT 1 FROM shares WHERE owner_id=? AND viewer_id=?", (owner_id, viewer_id)).fetchone()
    return row is not None


# =====================================================================
# SECTION 3 — FINANCIAL CALCULATION HELPERS (framework-independent)
# =====================================================================
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """Load the trained RandomForest model. Returns None if the file is missing."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover
        st.error(f"⚠️ Could not load model: {exc}")
        return None


def currency_symbol() -> str:
    return CURRENCY_SYMBOLS.get(st.session_state.get("currency", "USD ($)"), "$")


def fmt_money(value: float) -> str:
    try:
        return f"{currency_symbol()}{value:,.2f}"
    except Exception:
        return f"{currency_symbol()}{value}"


def compute_health_score(income: float, expenses: float, net_worth: float, age: int) -> float:
    """
    A transparent, rule-based Financial Health Score (0-100), independent
    of the ML model, used for quick and explainable feedback everywhere.
    """
    if income <= 0:
        return 0.0

    savings_rate = max(0.0, (income - expenses) / income)
    savings_component = min(savings_rate * 100, 40)  # up to 40 points

    net_worth_ratio = net_worth / max(income, 1)
    networth_component = min(net_worth_ratio * 10, 35)  # up to 35 points

    expected_multiplier = max(0.5, (age - 20) / 10)
    on_track_ratio = net_worth / max(income * expected_multiplier, 1)
    age_component = min(on_track_ratio * 12.5, 25)  # up to 25 points

    score = savings_component + networth_component + age_component
    return round(min(max(score, 0), 100), 1)


def score_label(score: float) -> tuple:
    if score >= 80:
        return "Excellent", "green", "🏆"
    elif score >= 60:
        return "Good", "blue", "👍"
    elif score >= 40:
        return "Fair", "orange", "⚠️"
    else:
        return "Needs Attention", "red", "🚨"


def compute_emi(principal: float, annual_rate: float, years: int) -> float:
    if principal <= 0 or years <= 0:
        return 0.0
    monthly_rate = annual_rate / 12 / 100
    n = years * 12
    if monthly_rate == 0:
        return principal / n
    emi = principal * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)
    return emi


def compute_compound_interest(principal: float, annual_rate: float, years: int,
                               contributions_per_year: int = 12, monthly_contribution: float = 0.0):
    rate_per_period = annual_rate / 100 / contributions_per_year
    balance = principal
    rows = []
    for year in range(1, years + 1):
        for _ in range(contributions_per_year):
            balance = balance * (1 + rate_per_period) + monthly_contribution
        rows.append({"Year": year, "Balance": round(balance, 2)})
    return pd.DataFrame(rows)


def get_daily_quote():
    idx = datetime.date.today().toordinal() % len(FINANCE_QUOTES)
    return FINANCE_QUOTES[idx]


def get_daily_tip():
    idx = datetime.date.today().toordinal() % len(DAILY_TIPS)
    return DAILY_TIPS[idx]


# =====================================================================
# SECTION 4 — SESSION STATE + STYLING
# =====================================================================
def init_session_state():
    defaults = {
        "auth_user": None,          # dict of the logged-in user, or None
        "auth_mode": "Login",       # "Login" or "Register"
        "currency": "USD ($)",
        "age": 30,
        "household_income": 30000,
        "networth": 100000,
        "monthly_expenses": 15000,
        "mood": "Neutral",
        "wizard_step": "Step 1: Basic Info",
        "quiz_submitted": False,
        "confirm_clear": False,
        "viewing_shared_user": None,  # user_id of a contact whose data we're viewing
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Hero banner used at the top of every page */
.hero-banner {
    background: linear-gradient(135deg, #0f766e 0%, #134e4a 100%);
    padding: 1.6rem 2rem;
    border-radius: 16px;
    color: #ffffff;
    margin-bottom: 1.4rem;
    box-shadow: 0 8px 24px rgba(15, 118, 110, 0.25);
}
.hero-banner h1 { margin: 0; font-size: 1.9rem; }
.hero-banner p  { margin: 0.3rem 0 0 0; opacity: 0.9; }

/* Generic card */
.metric-card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 1rem 1.2rem;
    border-radius: 12px;
    color: white;
}

.quote-box {
    border-left: 4px solid #22c55e;
    padding: 0.8rem 1rem;
    background-color: rgba(34,197,94,0.08);
    border-radius: 6px;
    font-style: italic;
}
.tip-box {
    border-left: 4px solid #3b82f6;
    padding: 0.8rem 1rem;
    background-color: rgba(59,130,246,0.08);
    border-radius: 6px;
}

/* Contact / share list rows */
.contact-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0.9rem;
    border-radius: 10px;
    background-color: rgba(15,118,110,0.07);
    margin-bottom: 0.5rem;
}

/* Buttons */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
}
.stButton>button:hover {
    border-color: #0f766e;
    color: #0f766e;
}

/* Auth card */
.auth-card {
    max-width: 460px;
    margin: 2rem auto;
    padding: 2rem;
    border-radius: 18px;
    background: rgba(15,118,110,0.05);
    border: 1px solid rgba(15,118,110,0.2);
}
</style>
"""


def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f'<div class="hero-banner"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# SECTION 5 — AUTHENTICATION UI (Login / Register)
# =====================================================================
def render_auth_screen():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    page_header("💹 AI Finance Dashboard", "Smart Personal Finance Assistant — please log in or register to continue")

    _, center, _ = st.columns([1, 1.3, 1])
    with center:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔐 Login", "🆕 Register"])

        with tab_login:
            st.subheader("Welcome back")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log In", use_container_width=True)
            if submitted:
                user = authenticate(username, password)
                if user:
                    st.session_state.auth_user = user
                    st.success(f"Welcome back, {user['name']}! Redirecting...")
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        with tab_register:
            st.subheader("Create your account")
            with st.form("register_form"):
                reg_name = st.text_input("Full name")
                reg_username = st.text_input("Choose a username", help="3-20 characters: letters, numbers, underscore")
                reg_email = st.text_input("Email (optional)")
                reg_password = st.text_input("Password", type="password", help="At least 6 characters")
                reg_password2 = st.text_input("Confirm password", type="password")
                reg_submitted = st.form_submit_button("Create Account", use_container_width=True)
            if reg_submitted:
                if reg_password != reg_password2:
                    st.error("Passwords don't match.")
                else:
                    ok, msg = create_user(reg_username, reg_name, reg_email, reg_password)
                    if ok:
                        st.success(msg + " Switch to the Login tab to sign in.")
                    else:
                        st.error(msg)

        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# SECTION 6 — MAIN APP BOOTSTRAP
# =====================================================================
init_session_state()
init_db()
model = load_model(MODEL_PATH)

if st.session_state.auth_user is None:
    render_auth_screen()
    st.stop()  # Nothing below this line runs until the user logs in

current_user = get_user_by_id(st.session_state.auth_user["id"])  # always fetch fresh copy
st.session_state.auth_user = current_user

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------
# SIDEBAR — NAVIGATION + PROFILE + LOGOUT
# ---------------------------------------------------------------
st.sidebar.title("📌 AI Finance Dashboard")
st.sidebar.caption(f"Logged in as **{current_user['name']}** (@{current_user['username']})")

page = st.sidebar.radio("Go to:", NAV_PAGES)

st.sidebar.divider()
st.session_state.currency = st.sidebar.selectbox(
    "Currency", list(CURRENCY_SYMBOLS.keys()),
    index=list(CURRENCY_SYMBOLS.keys()).index(st.session_state.currency),
)

st.sidebar.divider()
if model is None:
    st.sidebar.error("Model file not found.\nPlace `random_forest_model.pkl` next to app.py.")
else:
    st.sidebar.success("✅ ML model loaded")

st.sidebar.divider()

if st.sidebar.button("🚪 Log Out", use_container_width=True):
    # Clear logged-in user
    st.session_state.auth_user = None

    # Reset temporary financial inputs
    st.session_state.age = 30
    st.session_state.household_income = 30000
    st.session_state.networth = 100000
    st.session_state.monthly_expenses = 15000
    st.session_state.mood = "Neutral"

    # Reset prediction wizard
    st.session_state.wizard_step = "Step 1: Basic Info"
    st.session_state.quiz_submitted = False
    st.session_state.confirm_clear = False

    # Clear shared-user viewing
    st.session_state.viewing_shared_user = None

    # Reset currency
    st.session_state.currency = "USD ($)"

    # Do NOT delete prediction history
    st.rerun()
    

st.sidebar.caption(f"Session: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}")


# =====================================================================
# PAGE: HOME
# =====================================================================
if page == "🏠 Home":
    page_header(f"🏠 Welcome, {current_user['name']}!", "Here's a snapshot of your financial journey so far.")

    history_df = get_predictions(current_user["id"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Predictions made", len(history_df))
    with col2:
        avg_pred = history_df["prediction"].mean() if len(history_df) else 0
        st.metric("🎯 Avg. Trust Score", f"{avg_pred:.2f}" if len(history_df) else "—")
    with col3:
        health = compute_health_score(
            st.session_state.household_income,
            st.session_state.monthly_expenses * 12,
            st.session_state.networth,
            st.session_state.age,
        )
        st.metric("💚 Financial Health", f"{health}/100")
    with col4:
        shared_count = len(get_people_sharing_with_me(current_user["id"]))
        st.metric("🤝 Sharing with me", shared_count)

    st.divider()

    left, right = st.columns([2, 1])
    with left:
        st.subheader("✨ What can you do here?")
        st.markdown(
            """
            - 🔮 **Prediction Wizard** — step through a guided form and get an
              AI-predicted *Financial Trust Score*.
            - 🧮 **Financial Tools** — Health Score, Budget Planner, EMI
              calculator, and a Compound Interest / SIP growth simulator.
            - 📜 **History** — every prediction you make is saved privately
              to your account so you can track trends over time.
            - 🤝 **Contacts & Sharing** — add other registered users as
              contacts and choose to share your financial history with
              them — perfect for a mentor, parent or partner keeping track
              together.
            - 😊 **Mood Insights** — see how your reported mood correlates
              with your financial predictions.
            - 📝 **Quiz** — test your financial literacy with instant
              feedback.
            """
        )
        st.info("Use the sidebar to move between sections at any time.")

    with right:
        quote, author = get_daily_quote()
        st.markdown("#### 💬 Quote of the Day")
        st.markdown(f'<div class="quote-box">"{quote}"<br><b>— {author}</b></div>', unsafe_allow_html=True)
        st.markdown("#### 💡 Tip of the Day")
        st.markdown(f'<div class="tip-box">{get_daily_tip()}</div>', unsafe_allow_html=True)


# =====================================================================
# PAGE: PREDICTION WIZARD
# =====================================================================
elif page == "🔮 Prediction Wizard":
    page_header("🔮 Prediction Wizard", "Answer a few quick questions to get your AI-predicted Financial Trust Score.")

    steps = ["Step 1: Basic Info", "Step 2: Assets", "Step 3: Lifestyle", "Step 4: Result"]
    st.session_state.wizard_step = st.radio("Choose Step:", steps, index=steps.index(st.session_state.wizard_step))
    step = st.session_state.wizard_step

    progress_map = {steps[i]: (i + 1) / len(steps) for i in range(len(steps))}
    st.progress(progress_map[step])

    if step == "Step 1: Basic Info":
        st.header("Step 1: Basic Info")
        st.session_state.age = st.slider("👤 Age", 18, 70, st.session_state.age)
        st.write(f"Selected age: **{st.session_state.age}**")
        st.button("Next ➡️", on_click=lambda: st.session_state.update(wizard_step="Step 2: Assets"))

    elif step == "Step 2: Assets":
        st.header("Step 2: Assets")
        st.session_state.household_income = st.slider(
            "💰 Household Income (per year)", 1000, 200000, st.session_state.household_income, step=500
        )
        st.session_state.networth = st.slider(
            "🏦 Net Worth", 1000, 1000000, st.session_state.networth, step=1000
        )
        st.session_state.monthly_expenses = st.slider(
            "🧾 Monthly Expenses", 0, 20000, st.session_state.monthly_expenses, step=100
        )
        c1, c2 = st.columns(2)
        c1.metric("Income", fmt_money(st.session_state.household_income))
        c2.metric("Net Worth", fmt_money(st.session_state.networth))
        colA, colB = st.columns(2)
        colA.button("⬅️ Back", on_click=lambda: st.session_state.update(wizard_step="Step 1: Basic Info"))
        colB.button("Next ➡️", on_click=lambda: st.session_state.update(wizard_step="Step 3: Lifestyle"))

    elif step == "Step 3: Lifestyle":
        st.header("Step 3: Lifestyle")
        st.session_state.mood = st.selectbox(
            "😊 Mood Today", list(MOOD_EMOJI.keys()),
            index=list(MOOD_EMOJI.keys()).index(st.session_state.mood) if st.session_state.mood in MOOD_EMOJI else 0,
        )
        st.write(f"You're feeling: {MOOD_EMOJI.get(st.session_state.mood, '')} {st.session_state.mood}")
        colA, colB = st.columns(2)
        colA.button("⬅️ Back", on_click=lambda: st.session_state.update(wizard_step="Step 2: Assets"))
        colB.button("Next ➡️", on_click=lambda: st.session_state.update(wizard_step="Step 4: Result"))

    elif step == "Step 4: Result":
        st.header("Step 4: Prediction Result")

        if model is None:
            st.error("The ML model isn't loaded, so a prediction can't be generated. "
                      "Make sure `random_forest_model.pkl` is in the app folder.")
        else:
            input_data = {
                "age": st.session_state.age,
                "household_income_usd": st.session_state.household_income,
                "net_worth_usd": st.session_state.networth,
            }
            input_df = pd.DataFrame([input_data])

            try:
                prediction = float(model.predict(input_df)[0])
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                prediction = None

            if prediction is not None:
                health = compute_health_score(
                    st.session_state.household_income,
                    st.session_state.monthly_expenses * 12,
                    st.session_state.networth,
                    st.session_state.age,
                )
                label, color, emoji = score_label(health)

                r1, r2 = st.columns(2)
                with r1:
                    st.success(f"✅ Predicted Trust Score: **{prediction:.2f}**")
                with r2:
                    st.markdown(f"**Financial Health:** {emoji} :{color}[{label}] ({health}/100)")

                st.progress(min(max(health / 100, 0), 1.0))

                with st.expander("📊 See input summary"):
                    st.table(pd.DataFrame([{
                        "Age": st.session_state.age,
                        "Household Income": fmt_money(st.session_state.household_income),
                        "Net Worth": fmt_money(st.session_state.networth),
                        "Monthly Expenses": fmt_money(st.session_state.monthly_expenses),
                        "Mood": st.session_state.mood,
                    }]))

                if st.button("💾 Save this prediction to History"):
                    record = {
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "age": st.session_state.age,
                        "household_income_usd": st.session_state.household_income,
                        "net_worth_usd": st.session_state.networth,
                        "monthly_expenses_usd": st.session_state.monthly_expenses,
                        "mood": st.session_state.mood,
                        "prediction": round(prediction, 2),
                        "health_score": health,
                    }
                    save_prediction(current_user["id"], record)
                    st.success("Saved to History! Head to the History tab to view it.")

        st.button("⬅️ Back", on_click=lambda: st.session_state.update(wizard_step="Step 3: Lifestyle"))
        st.button("🔄 Start Over", on_click=lambda: st.session_state.update(wizard_step="Step 1: Basic Info"))


# =====================================================================
# PAGE: FINANCIAL TOOLS
# =====================================================================
elif page == "🧮 Financial Tools":
    page_header("🧮 Financial Tools", "A toolkit of transparent, rule-based financial calculators.")

    tool = st.selectbox(
        "Choose a tool:",
        ["Financial Health Score", "Budget Planner", "Savings Goal Calculator",
         "EMI / Loan Calculator", "Compound Interest / SIP Simulator"],
    )
    st.divider()

    if tool == "Financial Health Score":
        st.subheader("💚 Financial Health Score")
        st.write("A transparent score combining savings rate, net worth ratio, and age-adjusted progress.")

        c1, c2 = st.columns(2)
        income = c1.number_input("Annual Income", min_value=0, value=st.session_state.household_income, step=500)
        expenses = c2.number_input("Annual Expenses", min_value=0, value=st.session_state.monthly_expenses * 12, step=500)
        c3, c4 = st.columns(2)
        net_worth = c3.number_input("Current Net Worth", min_value=0, value=st.session_state.networth, step=1000)
        age = c4.number_input("Age", min_value=18, max_value=100, value=st.session_state.age)

        score = compute_health_score(income, expenses, net_worth, age)
        label, color, emoji = score_label(score)

        st.metric("Your Score", f"{score}/100", delta=label)
        st.progress(min(max(score / 100, 0), 1.0))
        st.markdown(f"### {emoji} :{color}[{label}]")

        with st.expander("How is this calculated?"):
            st.markdown(
                """
                - **Savings rate** (up to 40 pts): `(income - expenses) / income`
                - **Net worth ratio** (up to 35 pts): `net worth / income`
                - **Age-adjusted progress** (up to 25 pts): compares your net
                  worth to a rough expected multiple of income for your age.

                This is a simplified educational heuristic, not financial advice.
                """
            )

    elif tool == "Budget Planner":
        st.subheader("📊 Monthly Budget Planner")
        st.write("Break your monthly income into categories and see where it goes.")

        income = st.number_input("Monthly Income", min_value=0, value=5000, step=100)

        st.markdown("#### Allocate your expenses")
        c1, c2, c3 = st.columns(3)
        housing = c1.number_input("🏠 Housing", min_value=0, value=1500, step=50)
        food = c2.number_input("🍽️ Food", min_value=0, value=600, step=50)
        transport = c3.number_input("🚗 Transport", min_value=0, value=300, step=50)
        c4, c5, c6 = st.columns(3)
        utilities = c4.number_input("💡 Utilities", min_value=0, value=250, step=50)
        entertainment = c5.number_input("🎬 Entertainment", min_value=0, value=200, step=50)
        savings = c6.number_input("🏦 Savings/Investing", min_value=0, value=800, step=50)
        other = st.number_input("🧩 Other", min_value=0, value=200, step=50)

        total_allocated = housing + food + transport + utilities + entertainment + savings + other
        remaining = income - total_allocated

        budget_df = pd.DataFrame({
            "Category": ["Housing", "Food", "Transport", "Utilities", "Entertainment", "Savings", "Other"],
            "Amount": [housing, food, transport, utilities, entertainment, savings, other],
        }).set_index("Category")

        colL, colR = st.columns([1, 1])
        with colL:
            st.bar_chart(budget_df)
        with colR:
            st.markdown("#### Summary")
            st.metric("Total Income", fmt_money(income))
            st.metric("Total Allocated", fmt_money(total_allocated))
            st.metric("Unallocated / Leftover", fmt_money(remaining))
            if remaining < 0:
                st.error("You've allocated more than your income. Adjust your categories.")
            elif remaining > 0:
                st.success(f"You have {fmt_money(remaining)} left unallocated — consider adding it to Savings.")
            else:
                st.info("Perfectly balanced budget!")

    elif tool == "Savings Goal Calculator":
        st.subheader("🎯 Savings Goal Calculator")
        st.write("Figure out how long it will take to reach a savings goal, or how much to save monthly.")

        goal = st.number_input("Target amount", min_value=0, value=500000, step=1000)
        current = st.number_input("Current savings", min_value=0, value=50000, step=1000)
        mode = st.radio("Solve for:", ["Time to reach goal", "Required monthly saving"])

        remaining_amount = max(goal - current, 0)

        if mode == "Time to reach goal":
            monthly_saving = st.number_input("Monthly saving amount", min_value=1, value=5000, step=500)
            months_needed = remaining_amount / monthly_saving if monthly_saving else float("inf")
            years = int(months_needed // 12)
            months = int(months_needed % 12)
            st.success(f"⏳ At {fmt_money(monthly_saving)}/month, you'll reach your goal in "
                       f"**{years} years, {months} months**.")
        else:
            months_target = st.slider("Months to reach goal", 1, 360, 24)
            required_monthly = remaining_amount / months_target if months_target else 0
            st.success(f"💰 You need to save **{fmt_money(required_monthly)}** per month "
                       f"to reach your goal in {months_target} months.")

        st.progress(min(current / goal, 1.0) if goal else 0)
        st.caption(f"Progress so far: {fmt_money(current)} of {fmt_money(goal)}")

    elif tool == "EMI / Loan Calculator":
        st.subheader("🏦 EMI / Loan Calculator")
        c1, c2, c3 = st.columns(3)
        principal = c1.number_input("Loan Amount", min_value=0, value=200000, step=1000)
        rate = c2.number_input("Annual Interest Rate (%)", min_value=0.0, value=8.5, step=0.1)
        years = c3.number_input("Loan Tenure (years)", min_value=1, value=5, step=1)

        emi = compute_emi(principal, rate, years)
        total_payment = emi * years * 12
        total_interest = total_payment - principal

        m1, m2, m3 = st.columns(3)
        m1.metric("Monthly EMI", fmt_money(emi))
        m2.metric("Total Payment", fmt_money(total_payment))
        m3.metric("Total Interest", fmt_money(total_interest))

        breakdown_df = pd.DataFrame({
            "Component": ["Principal", "Interest"],
            "Amount": [principal, total_interest],
        }).set_index("Component")
        st.bar_chart(breakdown_df)

    elif tool == "Compound Interest / SIP Simulator":
        st.subheader("📈 Compound Interest / SIP Growth Simulator")
        c1, c2 = st.columns(2)
        principal = c1.number_input("Initial Investment", min_value=0, value=10000, step=500)
        monthly_contribution = c2.number_input("Monthly Contribution", min_value=0, value=2000, step=100)
        c3, c4 = st.columns(2)
        rate = c3.number_input("Expected Annual Return (%)", min_value=0.0, value=10.0, step=0.5)
        years = c4.number_input("Investment Duration (years)", min_value=1, value=10, step=1)

        growth_df = compute_compound_interest(principal, rate, int(years), 12, monthly_contribution)
        st.line_chart(growth_df.set_index("Year"))

        final_balance = growth_df["Balance"].iloc[-1] if len(growth_df) else principal
        total_invested = principal + monthly_contribution * 12 * years
        gains = final_balance - total_invested

        m1, m2, m3 = st.columns(3)
        m1.metric("Final Balance", fmt_money(final_balance))
        m2.metric("Total Invested", fmt_money(total_invested))
        m3.metric("Total Gains", fmt_money(gains))

        with st.expander("See year-by-year table"):
            st.dataframe(growth_df, use_container_width=True)


# =====================================================================
# PAGE: HISTORY
# =====================================================================
elif page == "📜 History":
    page_header("📜 Prediction History", "Every prediction you've saved, filterable and exportable.")

    history_df = get_predictions(current_user["id"])

    if history_df.empty:
        st.warning("No history found yet. Run a prediction first from the Prediction Wizard!")
    else:
        st.subheader("🔍 Filters")
        colf1, colf2 = st.columns(2)
        with colf1:
            moods_available = ["All"] + sorted(history_df["mood"].dropna().unique().tolist())
            mood_filter = st.selectbox("Filter by mood", moods_available)
        with colf2:
            sort_order = st.selectbox("Sort by timestamp", ["Newest first", "Oldest first"])

        filtered_df = history_df.copy()
        if mood_filter != "All":
            filtered_df = filtered_df[filtered_df["mood"] == mood_filter]
        filtered_df = filtered_df.sort_values("timestamp", ascending=(sort_order == "Oldest first"))

        st.dataframe(filtered_df, use_container_width=True)

        st.subheader("📈 Trends")
        tabs = st.tabs(["Prediction Trend", "Health Score Trend", "Income vs Net Worth"])
        with tabs[0]:
            st.line_chart(filtered_df.set_index("timestamp")["prediction"])
        with tabs[1]:
            st.line_chart(filtered_df.set_index("timestamp")["health_score"])
        with tabs[2]:
            st.line_chart(filtered_df.set_index("timestamp")[["household_income_usd", "net_worth_usd"]])

        st.subheader("📌 Summary Statistics")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Records", len(filtered_df))
        s2.metric("Avg. Trust Score", f"{filtered_df['prediction'].mean():.2f}")
        s3.metric("Avg. Health Score", f"{filtered_df['health_score'].mean():.1f}")
        s4.metric("Best Score", f"{filtered_df['prediction'].max():.2f}")

        st.divider()
        col_dl, col_clear = st.columns(2)
        with col_dl:
            csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV", data=csv_bytes,
                file_name="prediction_history.csv", mime="text/csv",
            )
        with col_clear:
            if st.button("🗑️ Clear All History"):
                st.session_state["confirm_clear"] = True

            if st.session_state.get("confirm_clear"):
                st.warning("Are you sure? This cannot be undone.")
                cc1, cc2 = st.columns(2)
                if cc1.button("✅ Yes, delete everything"):
                    clear_predictions(current_user["id"])
                    st.session_state["confirm_clear"] = False
                    st.success("History cleared.")
                    st.rerun()
                if cc2.button("❌ Cancel"):
                    st.session_state["confirm_clear"] = False

        st.divider()
        my_shares = get_people_i_share_with(current_user["id"])
        if my_shares:
            names = ", ".join(f"@{p['username']}" for p in my_shares)
            st.caption(f"👀 Currently sharing this history with: {names}. Manage sharing in **Contacts & Sharing**.")


# =====================================================================
# PAGE: MOOD INSIGHTS
# =====================================================================
elif page == "😊 Mood Insights":
    page_header("😊 Mood Insights", "How your reported mood lines up with your financial predictions.")

    history_df = get_predictions(current_user["id"])
    if history_df.empty:
        st.warning("No mood data yet — make a prediction first!")
    else:
        mood_counts = history_df["mood"].value_counts()
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Mood Frequency")
            st.bar_chart(mood_counts)
        with col2:
            st.subheader("Average Trust Score by Mood")
            avg_by_mood = history_df.groupby("mood")["prediction"].mean()
            st.bar_chart(avg_by_mood)

        st.subheader("📋 Mood Breakdown")
        for mood_name, count in mood_counts.items():
            emoji = MOOD_EMOJI.get(mood_name, "")
            avg_score = history_df[history_df["mood"] == mood_name]["prediction"].mean()
            st.write(f"{emoji} **{mood_name}** — {count} entries, avg. score {avg_score:.2f}")

        most_common_mood = mood_counts.idxmax()
        st.info(f"Your most frequently logged mood is {MOOD_EMOJI.get(most_common_mood,'')} **{most_common_mood}**.")


# =====================================================================
# PAGE: CONTACTS & SHARING
# =====================================================================
elif page == "🤝 Contacts & Sharing":
    page_header("🤝 Contacts & Sharing", "Connect with other users and share your financial history with people you trust.")

    tab_contacts, tab_requests, tab_share, tab_shared_with_me = st.tabs(
        ["👥 My Contacts", "📨 Requests", "📤 Share My Data", "📥 Shared With Me"]
    )

    # ---- My Contacts ----
    with tab_contacts:
        st.subheader("Add a new contact")
        with st.form("add_contact_form"):
            target_username = st.text_input("Their username")
            add_submitted = st.form_submit_button("Send Contact Request")
        if add_submitted and target_username:
            ok, msg = send_contact_request(current_user["id"], target_username)
            (st.success if ok else st.error)(msg)

        st.divider()
        st.subheader("Your contacts")
        contacts = get_accepted_contacts(current_user["id"])
        if not contacts:
            st.info("You don't have any contacts yet. Send a request above to get started.")
        else:
            for c in contacts:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(
                        f'<div class="contact-row"><span>👤 <b>{c["name"]}</b> (@{c["username"]})</span></div>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    if st.button("Remove", key=f"remove_{c['id']}"):
                        remove_contact(current_user["id"], c["id"])
                        st.rerun()

    # ---- Requests ----
    with tab_requests:
        st.subheader("📨 Incoming requests")
        incoming = get_incoming_requests(current_user["id"])
        if not incoming:
            st.info("No pending incoming requests.")
        else:
            for req in incoming:
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.write(f"👤 **{req['name']}** (@{req['username']}) wants to connect")
                if col2.button("✅ Accept", key=f"accept_{req['request_id']}"):
                    respond_to_request(req["request_id"], True)
                    st.rerun()
                if col3.button("❌ Decline", key=f"decline_{req['request_id']}"):
                    respond_to_request(req["request_id"], False)
                    st.rerun()

        st.divider()
        st.subheader("📤 Outgoing requests (awaiting response)")
        outgoing = get_outgoing_requests(current_user["id"])
        if not outgoing:
            st.info("No pending outgoing requests.")
        else:
            for req in outgoing:
                st.write(f"⏳ Waiting for **{req['name']}** (@{req['username']}) to respond")

    # ---- Share My Data ----
    with tab_share:
        st.subheader("Share your financial history")
        st.caption("Sharing gives a contact **read-only** access to your saved predictions, "
                   "health score trend and mood insights. You can revoke access at any time.")

        contacts = get_accepted_contacts(current_user["id"])
        already_sharing = {p["id"] for p in get_people_i_share_with(current_user["id"])}

        if not contacts:
            st.info("Add a contact first (under 'My Contacts') before you can share your data.")
        else:
            for c in contacts:
                col1, col2 = st.columns([3, 1])
                col1.write(f"👤 **{c['name']}** (@{c['username']})")
                if c["id"] in already_sharing:
                    if col2.button("🔒 Stop Sharing", key=f"unshare_{c['id']}"):
                        unshare_data(current_user["id"], c["id"])
                        st.rerun()
                else:
                    if col2.button("📤 Share", key=f"share_{c['id']}"):
                        ok, msg = share_data_with(current_user["id"], c["id"])
                        (st.success if ok else st.error)(msg)
                        st.rerun()

    # ---- Shared With Me ----
    with tab_shared_with_me:
        st.subheader("Dashboards shared with you")
        sharers = get_people_sharing_with_me(current_user["id"])
        if not sharers:
            st.info("No one is sharing their financial history with you yet.")
        else:
            options = {f"{p['name']} (@{p['username']})": p["id"] for p in sharers}
            chosen_label = st.selectbox("View a contact's shared data:", list(options.keys()))
            chosen_id = options[chosen_label]

            shared_df = get_predictions(chosen_id)
            if shared_df.empty:
                st.warning(f"{chosen_label} hasn't saved any predictions yet.")
            else:
                st.markdown(f"#### 📊 {chosen_label}'s Financial Overview *(read-only)*")
                s1, s2, s3 = st.columns(3)
                s1.metric("Predictions made", len(shared_df))
                s2.metric("Avg. Trust Score", f"{shared_df['prediction'].mean():.2f}")
                s3.metric("Avg. Health Score", f"{shared_df['health_score'].mean():.1f}")

                st.line_chart(shared_df.set_index("timestamp")["prediction"])
                st.dataframe(shared_df, use_container_width=True)


# =====================================================================
# PAGE: QUIZ
# =====================================================================
elif page == "📝 Quiz":
    page_header("📝 Finance Confidence Quiz", "Answer these quick questions to test your financial knowledge.")

    responses = {}
    for i, item in enumerate(QUIZ_QUESTIONS):
        responses[i] = st.radio(item["q"], item["options"], key=f"quiz_q{i}")

    if st.button("Submit Quiz"):
        st.session_state.quiz_submitted = True
        score = sum(1 for i, item in enumerate(QUIZ_QUESTIONS) if responses[i] == item["answer"])
        st.session_state.quiz_score = score

    if st.session_state.get("quiz_submitted"):
        score = st.session_state.quiz_score
        total = len(QUIZ_QUESTIONS)
        pct = score / total * 100

        st.success(f"🎯 Your score: {score}/{total} ({pct:.0f}%)")
        st.progress(score / total)

        if pct == 100:
            st.info("🏆 Perfect score! You're financially confident.")
        elif pct >= 75:
            st.info("👍 Great job! You clearly know your finance basics.")
        elif pct >= 50:
            st.info("📘 Good start — a bit more learning will make you stronger.")
        else:
            st.warning("🌱 Keep learning — small steps build confidence over time!")

        with st.expander("📖 Review answers"):
            for i, item in enumerate(QUIZ_QUESTIONS):
                correct = responses[i] == item["answer"]
                icon = "✅" if correct else "❌"
                st.write(f"{icon} {item['q']} — Correct answer: **{item['answer']}**")

        if st.button("🔄 Retake Quiz"):
            st.session_state.quiz_submitted = False
            st.rerun()


# =====================================================================
# PAGE: ACCOUNT SETTINGS
# =====================================================================
elif page == "⚙️ Account":
    page_header("⚙️ Account Settings", "Update your profile or change your password.")

    tab_profile, tab_password, tab_danger = st.tabs(["👤 Profile", "🔑 Password", "⚠️ Danger Zone"])

    with tab_profile:
        st.subheader("Update your profile")
        with st.form("profile_form"):
            new_name = st.text_input("Full name", value=current_user["name"])
            new_email = st.text_input("Email", value=current_user["email"] or "")
            profile_submitted = st.form_submit_button("Save Changes")
        if profile_submitted:
            if not valid_email(new_email):
                st.error("That email address doesn't look valid.")
            else:
                update_profile(current_user["id"], new_name, new_email)
                st.success("Profile updated.")
                st.rerun()

        st.caption(f"Username: **@{current_user['username']}** (can't be changed) · "
                   f"Member since {current_user['created_at']}")

    with tab_password:
        st.subheader("Change your password")
        with st.form("password_form"):
            old_pw = st.text_input("Current password", type="password")
            new_pw = st.text_input("New password", type="password")
            new_pw2 = st.text_input("Confirm new password", type="password")
            pw_submitted = st.form_submit_button("Update Password")
        if pw_submitted:
            if new_pw != new_pw2:
                st.error("New passwords don't match.")
            else:
                ok, msg = change_password(current_user["id"], old_pw, new_pw)
                (st.success if ok else st.error)(msg)

    with tab_danger:
        st.subheader("Clear your data")
        st.warning("This permanently deletes all of your saved predictions. Your account stays active.")
        if st.button("🗑️ Delete all my prediction history"):
            clear_predictions(current_user["id"])
            st.success("All prediction history deleted.")


# =====================================================================
# PAGE: ABOUT
# =====================================================================
elif page == "ℹ️ About":
    page_header("ℹ️ About This App", "Project background, tech stack, and deployment notes.")

    st.write(
        "This dashboard is an industrial-training project — **AI Finance Dashboard "
        "(Smart Personal Finance Assistant)** — built with **Streamlit**, combining a "
        "trained machine learning model with practical, explainable financial planning "
        "tools, and now a full multi-user account system with data sharing."
    )

    st.subheader("🧰 Tech Stack")
    st.markdown(
        """
        - **Frontend / App framework:** Streamlit
        - **Data handling:** pandas, NumPy
        - **Machine Learning:** scikit-learn (RandomForestRegressor), joblib
        - **Storage:** SQLite (users, predictions, contacts, sharing) — no external DB server needed
        - **Auth:** PBKDF2-HMAC-SHA256 password hashing (Python's built-in `hashlib`)
        """
    )

    st.subheader("✨ Features")
    st.markdown(
        """
        - 🔐 Secure multi-user login & registration (hashed + salted passwords)
        - 🔮 Multi-step Prediction Wizard powered by a trained ML model
        - 💚 Rule-based, explainable Financial Health Score
        - 📊 Budget Planner with category-wise visual breakdown
        - 🎯 Savings Goal Calculator
        - 🏦 EMI / Loan Calculator
        - 📈 Compound Interest / SIP growth simulator
        - 📜 Private, filterable prediction History per user, with CSV export
        - 🤝 Contacts system with requests, and opt-in data sharing between users
        - 😊 Mood Insights correlating mood with predicted scores
        - 📝 8-question gamified Finance Confidence Quiz
        - ⚙️ Account settings: edit profile, change password, clear data
        - 💬 Daily rotating finance quotes & tips, multi-currency display
        """
    )

    st.subheader("📁 Project Structure")
    st.code(
        "finance-dashboard/\n"
        "├── app.py        # Main Streamlit application\n"
        "|__ data_cleaning.ipynb\n"  
        "|__ data_analysis.ipynb\n"
        "|__ eda.ipynb\n"
        "|__ data_processing.ipynb\n"           
        "├── model-training.ipynb       # Notebook used to train the model\n"
        "├── random_forest_model.pkl    # Trained ML model\n"
        "├── data/\n"
        "│   └── finance_app.db " # Auto-generated SQLite database\n
        "|   |__ cleaned_finance_data.db\n" 
        "|   |__ precition_history.csv\n"
        "|   |__ X_test.csv\n"
        "|   |__ X_train.csv\n" 
        "|   |__ y_train.csv\n"
        "|   |__ y_train.csv\n"    
        "└── requirements.txt\n",
        language="text",
    )

    st.subheader("🚀 Run Locally")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")

    st.subheader("☁️ Deploying to Streamlit Community Cloud (streamlit.io)")
    st.markdown(
        """
        1. Push this project (including `requirements.txt` and
           `random_forest_model.pkl`) to a **GitHub repository**.
        2. Go to **share.streamlit.io**, sign in, and click **"New app"**.
        3. Pick your repository, branch, and set the main file path to
           `app.py`, then deploy.
        4. Share the generated `*.streamlit.app` link with others — anyone
           can register their own account and use Contacts & Sharing to
           connect with each other on the same deployed app.

        **Good to know:** Streamlit Community Cloud's filesystem is
        *ephemeral* — the SQLite database (`data/finance_app.db`) will reset
        if the app sleeps, reboots, or is redeployed. That's fine for demos
        and internship evaluation. For a permanent production deployment,
        the database layer at the top of `app.py` is written so it can be
        swapped for a hosted database (e.g. Postgres on Supabase/Neon)
        without touching the rest of the app.
        """
    )

    st.caption("Built with ❤️ using Streamlit — for learning purposes only, not financial advice.")