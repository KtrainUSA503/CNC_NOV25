"""
Haas Mill Operator's Manual RAG Assistant
Production-ready Streamlit application with authentication, safety rules, and logging.
"""

import streamlit as st
import json
import os
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from functools import wraps
import time

# Third-party imports
from openai import OpenAI
from pinecone import Pinecone

# ============================================================================
# CONFIGURATION
# ============================================================================

APP_TITLE = "🔧 Haas Mill Operator's Manual Assistant"
APP_SUBTITLE = "Next Generation Control - 15\" LCD (96-8210)"
CHUNKS_FILE = "haas_mill_chunks.json"
LOG_FILE = "query_log.jsonl"
SESSION_TIMEOUT_MINUTES = 30
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K_RESULTS = 5
PINECONE_INDEX_NAME = "haas-mill-manual"
PINECONE_NAMESPACE = "operator-manual"
EMBEDDING_DIMENSION = 1536
DAILY_QUERY_LIMIT_DEFAULT = 100

# Cost estimation (approximate)
COST_PER_QUERY = 0.0002  # ~$0.20 per 1000 queries

# User configuration with roles and daily limits
USER_CONFIG = {
    "operator1": {
        "password_hash": hashlib.sha256("HaasMill2024!".encode()).hexdigest(),
        "role": "Operator",
        "daily_limit": 100
    },
    "operator2": {
        "password_hash": hashlib.sha256("SafeOps123!".encode()).hexdigest(),
        "role": "Operator", 
        "daily_limit": 100
    },
    "user1": {
        "password_hash": hashlib.sha256("User1Pass!".encode()).hexdigest(),
        "role": "Operator",
        "daily_limit": 100
    },
    "admin": {
        "password_hash": hashlib.sha256("AdminPass2024!".encode()).hexdigest(),
        "role": "Admin",
        "daily_limit": 200
    },
    "it_support": {
        "password_hash": hashlib.sha256("ITSupport2024!".encode()).hexdigest(),
        "role": "IT",
        "daily_limit": 500
    }
}

# Safety keywords that require LOTO reminders
LOTO_KEYWORDS = [
    "maintenance", "repair", "replace", "remove", "install", "service",
    "clean", "inspect", "adjust", "calibrate", "lubricate", "disassemble",
    "electrical", "power supply", "motor", "spindle change", "belt",
    "hydraulic", "pneumatic", "coolant system", "chip conveyor"
]

# High-risk operation keywords
HIGH_RISK_KEYWORDS = [
    "override", "bypass", "disable safety", "remove guard", "interlock",
    "emergency stop", "e-stop", "spindle", "high speed", "crash"
]

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def log_query(username: str, question: str, pages_used: List[int], response_summary: str):
    """Log each query to a JSONL file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "question": question,
        "pages_used": pages_used,
        "response_length": len(response_summary),
        "session_id": st.session_state.get("session_id", "unknown")
    }
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log query: {e}")

# ============================================================================
# AUTHENTICATION
# ============================================================================

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_credentials(username: str, password: str) -> Optional[Dict]:
    """Verify user credentials and return user info if valid."""
    if username in USER_CONFIG:
        user = USER_CONFIG[username]
        if user["password_hash"] == hash_password(password):
            return {
                "username": username,
                "role": user["role"],
                "daily_limit": user["daily_limit"]
            }
    return None

def check_session_timeout() -> bool:
    """Check if the current session has timed out."""
    if "last_activity" not in st.session_state:
        return True
    
    timeout = timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    if datetime.now() - st.session_state.last_activity > timeout:
        return True
    
    return False

def update_activity():
    """Update the last activity timestamp."""
    st.session_state.last_activity = datetime.now()

def get_daily_query_count(username: str) -> int:
    """Get the number of queries made by a user today."""
    today = datetime.now().date().isoformat()
    count_key = f"query_count_{username}_{today}"
    return st.session_state.get(count_key, 0)

def increment_query_count(username: str):
    """Increment the daily query count for a user."""
    today = datetime.now().date().isoformat()
    count_key = f"query_count_{username}_{today}"
    current = st.session_state.get(count_key, 0)
    st.session_state[count_key] = current + 1

def check_query_limit(username: str) -> Tuple[bool, int]:
    """Check if user has exceeded their daily query limit."""
    user = USER_CONFIG.get(username, {})
    limit = user.get("daily_limit", DAILY_QUERY_LIMIT_DEFAULT)
    current = get_daily_query_count(username)
    return current < limit, limit - current

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_chunks() -> List[Dict]:
    """Load the manual chunks from JSON file."""
    try:
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
        return chunks
    except FileNotFoundError:
        st.error(f"❌ Chunks file not found: {CHUNKS_FILE}")
        return []
    except json.JSONDecodeError as e:
        st.error(f"❌ Error parsing chunks file: {e}")
        return []

# ============================================================================
# EMBEDDING AND RETRIEVAL
# ============================================================================

def get_openai_client() -> OpenAI:
    """Initialize OpenAI client with API key from secrets."""
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in Streamlit secrets")
    return OpenAI(api_key=api_key)

def get_pinecone_client() -> Pinecone:
    """Initialize Pinecone client with API key from secrets."""
    api_key = st.secrets.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in Streamlit secrets")
    return Pinecone(api_key=api_key)

def create_embedding(client: OpenAI, text: str) -> List[float]:
    """Create an embedding for a text using OpenAI."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def retrieve_relevant_chunks(
    query: str, 
    openai_client: OpenAI,
    pinecone_index,
    top_k: int = TOP_K_RESULTS
) -> List[Dict]:
    """Retrieve the most relevant chunks for a query from Pinecone."""
    # Create query embedding
    query_embedding = create_embedding(openai_client, query)
    
    # Query Pinecone
    results = pinecone_index.query(
        namespace=PINECONE_NAMESPACE,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Extract and format results
    retrieved_chunks = []
    for match in results.matches:
        chunk = {
            "id": match.id,
            "text": match.metadata.get("text", ""),
            "page": match.metadata.get("page", 0),
            "score": match.score
        }
        retrieved_chunks.append(chunk)
    
    return retrieved_chunks

# ============================================================================
# SAFETY CHECKS
# ============================================================================

def check_loto_requirement(query: str, response_text: str) -> bool:
    """Check if the query or response involves maintenance requiring LOTO."""
    combined_text = (query + " " + response_text).lower()
    return any(keyword in combined_text for keyword in LOTO_KEYWORDS)

def check_high_risk_operation(query: str) -> bool:
    """Check if the query involves high-risk operations."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in HIGH_RISK_KEYWORDS)

def get_loto_reminder() -> str:
    """Return the LOTO safety reminder text."""
    return """
⚠️ **LOCKOUT/TAGOUT (LOTO) REMINDER**

Before performing any maintenance, repair, or service work:

1. **STOP** the machine and turn off the main power disconnect
2. **LOCK** the power disconnect in the OFF position
3. **TAG** the lockout device with your name and date
4. **VERIFY** the machine cannot be started by pressing POWER ON
5. **RELEASE** stored energy (hydraulic, pneumatic, mechanical)

**Never** perform maintenance with the machine powered on unless specifically 
required by the procedure and additional safety precautions are documented.

Refer to your facility's LOTO procedures and OSHA 1910.147.
"""

def get_safety_disclaimer() -> str:
    """Return the standard safety disclaimer."""
    return """
---
*⚠️ This information is from the Haas Mill Operator's Manual and is for reference only. 
Always follow your facility's safety procedures and consult with qualified personnel 
before performing any operations you are uncertain about.*
"""

# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def build_system_prompt() -> str:
    """Build the system prompt with safety rules."""
    return """You are a helpful assistant for Haas Mill machine operators. You answer questions 
ONLY using information from the provided manual excerpts. You are knowledgeable, safety-conscious, 
and precise.

## CRITICAL SAFETY RULES - YOU MUST FOLLOW THESE:

1. **NEVER INVENT STEPS OR PROCEDURES**: If the manual excerpt doesn't contain the specific 
   information needed, say "This information is not found in the provided manual sections" 
   and suggest the user consult the full manual or a qualified technician.

2. **NEVER PROVIDE UNSAFE INSTRUCTIONS**: If a question asks about bypassing safety systems, 
   disabling interlocks, or any potentially dangerous modifications, refuse and explain why 
   this is unsafe.

3. **ALWAYS CITE PAGES**: When providing information, always include the page number(s) from 
   the manual, formatted as (Page X) or (Pages X-Y).

4. **MAINTENANCE PROCEDURES**: For any maintenance, repair, or service procedures, always 
   remind the user about Lockout/Tagout (LOTO) requirements.

5. **WHEN UNCERTAIN**: If the retrieved text doesn't clearly answer the question, acknowledge 
   the limitation. Don't guess or extrapolate beyond what's explicitly stated.

6. **ELECTRICAL WORK**: Any electrical work should only be performed by qualified personnel. 
   Always include this warning when relevant.

## RESPONSE FORMAT:

- Be concise but complete
- Use bullet points for multi-step procedures
- Always include page citations
- Include relevant warnings or cautions from the manual
- If the answer involves multiple pages, cite all relevant pages

## EXAMPLE CITATION FORMAT:
"According to the manual (Page 45), the procedure for..."
"The spindle specifications are listed on Pages 78-79..."
"""

def generate_response(
    query: str,
    retrieved_chunks: List[Dict],
    openai_client: OpenAI
) -> Tuple[str, List[int]]:
    """Generate a response using the retrieved chunks."""
    
    # Build context from retrieved chunks
    if not retrieved_chunks:
        return (
            "I couldn't find relevant information in the manual for your question. "
            "Please try rephrasing your question or consult the full Haas Mill Operator's Manual.",
            []
        )
    
    context_parts = []
    pages_used = []
    
    for i, chunk in enumerate(retrieved_chunks):
        page = chunk.get("page", "Unknown")
        pages_used.append(page)
        context_parts.append(f"[Manual Excerpt {i+1}, Page {page}]\n{chunk['text']}\n")
    
    context = "\n---\n".join(context_parts)
    
    # Build the user message
    user_message = f"""Based ONLY on the following excerpts from the Haas Mill Operator's Manual, 
answer the user's question. If the information is not in these excerpts, clearly state that.

## MANUAL EXCERPTS:
{context}

## USER QUESTION:
{query}

## YOUR RESPONSE (remember to cite page numbers):"""

    # Generate response
    try:
        response = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content
        
        # Check if response indicates information not found
        not_found_phrases = [
            "not found in the",
            "doesn't contain",
            "does not contain", 
            "not mentioned in",
            "no information about",
            "couldn't find",
            "could not find"
        ]
        
        if any(phrase in answer.lower() for phrase in not_found_phrases):
            answer += "\n\n📖 *Tip: Try searching for related terms or check the Table of Contents in the full manual.*"
        
        return answer, list(set(pages_used))
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"An error occurred while generating the response: {str(e)}", []

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_login_page():
    """Render the login page."""
    st.title("🔐 Login Required")
    st.markdown("Please log in to access the Haas Mill Operator Assistant.")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            user_info = verify_credentials(username, password)
            if user_info:
                st.session_state.authenticated = True
                st.session_state.user = user_info
                st.session_state.last_activity = datetime.now()
                st.session_state.session_id = hashlib.md5(
                    f"{username}{datetime.now().isoformat()}".encode()
                ).hexdigest()[:12]
                logger.info(f"User {username} logged in successfully")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    
    st.markdown("---")
    st.markdown("**Demo Credentials:**")
    st.code("""
Username: operator1  |  Password: HaasMill2024!
Username: admin      |  Password: AdminPass2024!
    """)

def render_sidebar():
    """Render the sidebar with system status and about info."""
    with st.sidebar:
        # System Status Section
        st.markdown("### 📊 System Status")
        
        # Query counter
        username = st.session_state.user['username']
        queries_today = get_daily_query_count(username)
        limit = USER_CONFIG.get(username, {}).get("daily_limit", DAILY_QUERY_LIMIT_DEFAULT)
        
        st.markdown("**Your Questions Today**")
        st.markdown(f"# {queries_today}/{limit}")
        
        # Estimated cost
        estimated_cost = queries_today * COST_PER_QUERY
        st.markdown("**Estimated Cost Today**")
        st.markdown(f"# ${estimated_cost:.2f}")
        
        # Last activity
        if "last_activity" in st.session_state:
            last_activity = st.session_state.last_activity.strftime("%H:%M:%S")
            st.markdown(f"**Last activity:** {last_activity}")
        
        st.markdown("---")
        
        # About Section
        st.markdown("### About")
        st.markdown("""
This assistant uses RAG (Retrieval Augmented Generation) to answer 
questions about the **Haas Mill Operator's Manual** (Revision U, December 2024).
        """)
        
        st.markdown("**The system:**")
        st.markdown("""
- Searches through 550 pages of documentation
- Provides accurate answers with page references
- Cites sources from the official manual
        """)
        
        st.markdown("---")
        
        # Manual Details Section
        st.markdown("### Manual Details:")
        st.markdown("""
- **Model:** Next Generation Control
- **Screen:** 15" LCD
- **Part #:** 96-8210
- **Revision:** U
- **Date:** December 2024
        """)
        
        st.markdown("---")
        
        # Security Features Section
        st.markdown("### Security Features:")
        st.markdown(f"""
- Individual user accounts
- Activity logging
- 30-minute session timeout
- Daily query limits ({limit}/day)
- Usage monitoring
        """)
        
        st.markdown("---")
        
        # User info and logout
        st.markdown(f"🔒 Logged in as: **{username}** ({st.session_state.user['role']})")
        
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Admin features
        if st.session_state.user['role'] in ['Admin', 'IT']:
            st.markdown("---")
            st.markdown("### 🔧 Admin Tools")
            if st.button("📊 View Query Logs", use_container_width=True):
                st.session_state.show_logs = True

def render_source_expanders(chunks: List[Dict]):
    """Render expandable sections for source documents."""
    st.markdown("### 📚 Source Documents")
    
    for i, chunk in enumerate(chunks):
        page = chunk.get("page", "Unknown")
        score = chunk.get("score", 0)
        text = chunk.get("text", "")
        
        with st.expander(f"📄 Page {page} (Relevance: {score:.2%})", expanded=False):
            st.markdown(text)

def render_query_logs():
    """Render the query logs for admin users."""
    st.markdown("### 📊 Query Logs")
    
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                logs = [json.loads(line) for line in f.readlines()[-50:]]  # Last 50 entries
            
            if logs:
                for log in reversed(logs):
                    with st.expander(f"{log['timestamp'][:19]} - {log['username']}"):
                        st.markdown(f"**Question:** {log['question']}")
                        st.markdown(f"**Pages Used:** {log['pages_used']}")
            else:
                st.info("No query logs yet.")
        else:
            st.info("No query logs yet.")
    except Exception as e:
        st.error(f"Error loading logs: {e}")
    
    if st.button("← Back to Chat"):
        st.session_state.show_logs = False
        st.rerun()

def render_main_content():
    """Render the main chat interface."""
    # Title and subtitle
    st.title(APP_TITLE)
    st.markdown(f"**{APP_SUBTITLE}**")
    
    # Logged in status
    username = st.session_state.user['username']
    role = st.session_state.user['role']
    st.markdown(f"🔒 Logged in as: {username} ({role})")
    
    st.markdown("---")
    
    # Description
    st.markdown("""
Ask questions about operating your Haas Mill! This assistant searches the 550-page operator's manual 
to provide accurate answers with page references.
    """)
    
    # Example questions
    st.markdown("**Example questions:**")
    st.markdown("""
- How do I set up a work offset?
- What is the proper procedure for tool changes?
- How do I use the probe system?
- What do the alarm codes mean?
- How do I calibrate the machine?
    """)
    
    st.markdown("---")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Page config
    st.set_page_config(
        page_title="Haas Mill Assistant",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Mobile-friendly CSS
    st.markdown("""
    <style>
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .stButton > button {
        width: 100%;
        padding: 0.5rem 1rem;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_logs" not in st.session_state:
        st.session_state.show_logs = False
    
    # Check authentication
    if not st.session_state.authenticated:
        render_login_page()
        return
    
    # Check session timeout
    if check_session_timeout():
        st.warning("⏰ Your session has expired. Please log in again.")
        st.session_state.authenticated = False
        st.rerun()
    
    # Update activity timestamp
    update_activity()
    
    # Render sidebar
    render_sidebar()
    
    # Show logs if admin requested
    if st.session_state.get("show_logs", False):
        render_query_logs()
        return
    
    # Render main content header
    render_main_content()
    
    # Check query limit
    can_query, remaining = check_query_limit(st.session_state.user['username'])
    if not can_query:
        st.error("❌ You have reached your daily query limit. Please try again tomorrow.")
        return
    
    # Load chunks
    chunks = load_chunks()
    if not chunks:
        st.error("❌ Failed to load manual data. Please contact IT support.")
        return
    
    # Initialize clients
    try:
        openai_client = get_openai_client()
        pinecone_client = get_pinecone_client()
        pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        st.error(f"❌ Failed to initialize API clients: {e}")
        st.info("Please ensure OPENAI_API_KEY and PINECONE_API_KEY are set in Streamlit secrets.")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_source_expanders(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the Haas Mill operation..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Check for high-risk operations
        if check_high_risk_operation(prompt):
            warning_msg = """⚠️ **SAFETY WARNING**
            
Your question appears to involve safety-critical operations. Please ensure you:
- Have proper authorization and training
- Follow all facility safety procedures
- Consult with your supervisor if uncertain

Proceeding with your question..."""
            st.warning(warning_msg)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching manual and generating response..."):
                try:
                    # Retrieve relevant chunks
                    retrieved_chunks = retrieve_relevant_chunks(
                        prompt, openai_client, pinecone_index
                    )
                    
                    # Generate response
                    response, pages_used = generate_response(
                        prompt, retrieved_chunks, openai_client
                    )
                    
                    # Check for LOTO requirement
                    if check_loto_requirement(prompt, response):
                        response = get_loto_reminder() + "\n\n" + response
                    
                    # Add safety disclaimer
                    response += get_safety_disclaimer()
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display sources
                    if retrieved_chunks:
                        render_source_expanders(retrieved_chunks)
                    
                    # Log query
                    log_query(
                        st.session_state.user['username'],
                        prompt,
                        pages_used,
                        response[:200]
                    )
                    
                    # Increment query count
                    increment_query_count(st.session_state.user['username'])
                    
                    # Store message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": retrieved_chunks
                    })
                    
                except Exception as e:
                    error_msg = f"❌ An error occurred: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Error processing query: {e}")

if __name__ == "__main__":
    main()