# 🔧 Haas Mill Operator's Manual RAG Assistant

A production-ready Retrieval-Augmented Generation (RAG) chatbot for the Haas Mill Operator's Manual. Built with Streamlit, OpenAI, and Pinecone.

## Features

- **Intelligent Q&A**: Ask questions about the Haas Mill and get accurate answers from the official manual
- **Strict Citation**: All answers include page number citations from the manual
- **Safety-First Design**: 
  - LOTO (Lockout/Tagout) reminders for maintenance procedures
  - High-risk operation warnings
  - Never invents procedures not in the manual
- **Role-Based Access Control**: Operator, Admin, and IT roles with different permissions
- **Daily Query Limits**: Configurable per-role to manage API costs
- **Session Timeout**: 30-minute automatic logout for security
- **Query Logging**: Full audit trail of all queries
- **Mobile-Friendly**: Responsive design works on tablets and phones

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Streamlit  │────▶│   OpenAI     │────▶│   Response   │
│   Frontend   │     │   Embeddings │     │   Generation │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │
       │                    ▼
       │             ┌──────────────┐
       │             │   Pinecone   │
       │             │   Vector DB  │
       │             └──────────────┘
       │                    │
       ▼                    ▼
┌──────────────┐     ┌──────────────┐
│   Auth &     │     │   Manual     │
│   Logging    │     │   Chunks     │
└──────────────┘     └──────────────┘
```

## File Structure

```
haas_mill_rag/
├── app.py                    # Main Streamlit application
├── setup_pinecone.py         # One-time setup script for Pinecone
├── haas_mill_chunks.json     # Pre-processed manual chunks
├── requirements.txt          # Python dependencies
├── .streamlit/
│   ├── config.toml           # Streamlit configuration
│   └── secrets.toml.example  # API keys template
└── README.md                 # This file
```

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd haas_mill_rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-your-openai-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
```

### 3. Initialize Pinecone Index (One-Time Setup)

```bash
# Set environment variables
export OPENAI_API_KEY="sk-your-key"
export PINECONE_API_KEY="your-pinecone-key"

# Run setup script
python setup_pinecone.py
```

This will:
- Create a Pinecone serverless index named `haas-mill-manual`
- Generate embeddings for all manual chunks
- Upload vectors to Pinecone

### 4. Run Locally

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## Deployment to Streamlit Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo>
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository and branch
4. Set main file path: `app.py`
5. Click "Advanced settings" and add your secrets:
   ```toml
   OPENAI_API_KEY = "sk-your-key"
   PINECONE_API_KEY = "your-pinecone-key"
   ```
6. Click "Deploy"

## User Credentials

Default users (change in production!):

| Username | Password | Role | Daily Limit |
|----------|----------|------|-------------|
| operator1 | HaasMill2024! | Operator | 50 |
| operator2 | SafeOps123! | Operator | 50 |
| admin | AdminPass2024! | Admin | 200 |
| it_support | ITSupport2024! | IT | 500 |

**⚠️ IMPORTANT**: Change these credentials before production deployment by modifying the `USER_CONFIG` dictionary in `app.py`.

## Safety Features

### LOTO Reminders
The assistant automatically detects maintenance-related queries and prepends LOTO (Lockout/Tagout) safety reminders:

- Maintenance procedures
- Repair operations
- Component replacement
- Electrical work
- Hydraulic/pneumatic system work

### High-Risk Warnings
Special warnings appear for queries involving:
- Safety overrides
- Interlock systems
- Emergency stop procedures
- High-speed operations

### Response Guardrails
- Never invents procedures not in the manual
- Always cites page numbers
- Falls back gracefully when information isn't found
- Includes safety disclaimers on all responses

## Query Logging

All queries are logged to `query_log.jsonl` with:
- Timestamp
- Username
- Question asked
- Pages referenced
- Response length
- Session ID

Admins can view logs through the sidebar.

## Customization

### Adding Users
Edit `USER_CONFIG` in `app.py`:

```python
USER_CONFIG = {
    "new_user": {
        "password_hash": hashlib.sha256("password".encode()).hexdigest(),
        "role": "Operator",
        "daily_limit": 50
    }
}
```

### Adjusting Safety Keywords
Modify `LOTO_KEYWORDS` and `HIGH_RISK_KEYWORDS` lists in `app.py`.

### Changing Models
Update `EMBEDDING_MODEL` and `CHAT_MODEL` constants:

```python
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-3-large"
CHAT_MODEL = "gpt-4o-mini"  # or "gpt-4o" for better accuracy
```

## Cost Estimation

Approximate costs per 1000 queries:
- OpenAI Embeddings: ~$0.02
- OpenAI Chat (GPT-4o-mini): ~$0.15
- Pinecone Serverless: ~$0.01

Total: ~$0.20 per 1000 queries

## Troubleshooting

### "OPENAI_API_KEY not found"
Ensure your secrets are properly configured in `.streamlit/secrets.toml` or Streamlit Cloud settings.

### "Index not found" 
Run `setup_pinecone.py` to create and populate the Pinecone index.

### Slow responses
- Check your internet connection
- Verify Pinecone index is in a nearby region
- Consider using `gpt-4o-mini` instead of `gpt-4o`

### Session keeps expiring
The default timeout is 30 minutes. Modify `SESSION_TIMEOUT_MINUTES` in `app.py` if needed.

## License

For internal use at Keith Manufacturing Company. All rights reserved.

## Support

Contact IT Support for assistance with deployment or configuration issues.
