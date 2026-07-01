"""
config.py
─────────
Central configuration for Craster Multi-Agent RAG system.

All settings read from .env file automatically.
Nothing hardcoded here.

Three environments:
    local       your machine
    staging     test environment
    production  live environment

Usage:
    from config import settings
    print(settings.anthropic_api_key)
    print(settings.DOCUMENT_CATEGORIES)
"""

from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Environment Enum ───────────────────────────────────
class Environment(str, Enum):
    """
    Supported deployment environments.
    Controls logging levels and feature flags.

    str + Enum means:
        it is an Enum     fixed set of values typo protection
        it is also a str  works anywhere a string works
    """
    LOCAL      = "local"
    STAGING    = "staging"
    PRODUCTION = "production"


# ── RAG Category Enum ──────────────────────────────────
class DocumentCategory(str, Enum):
    """
    Document categories for routing.

    Router agent classifies every question
    into one of these categories.
    Retriever then searches only that category.

    GENERAL is used when question spans
    multiple categories or is unclear.
    """
    LEAVE_FAMILY      = "leave_family"
    HEALTH_SAFETY     = "health_safety"
    EMPLOYMENT        = "employment"
    DATA_COMPLIANCE   = "data_compliance"
    CONDUCT           = "conduct"
    REWARDS_BENEFITS  = "rewards_benefits"
    EQUALITY_DIVERSITY= "equality_diversity"
    COMPANY_GENERAL   = "company_general"
    GENERAL           = "general"


# ── Settings Class ─────────────────────────────────────
class Settings(BaseSettings):
    """
    Central settings class.

    Reads all values from .env file automatically.
    Type hints enforce correct data types.
    Pydantic validates values on startup.

    If a required value is missing from .env
    the app refuses to start with a clear error.
    """

    model_config = SettingsConfigDict(
        env_file          = ".env",
        env_file_encoding = "utf-8",
        case_sensitive    = False,
        extra             = "ignore",
    )

    # ── Environment ─────────────────────────────────────
    environment  : Environment = Environment.LOCAL
    app_version  : str         = "0.1.0"
    debug        : bool        = False

    # ── Acumatica (Phase 2) ─────────────────────────────
    acumatica_base_url : str = ""
    acumatica_username : str = ""
    acumatica_password : str = ""

    # ── Anthropic (Claude) ──────────────────────────────
    anthropic_api_key    : str = ""
    anthropic_model      : str = "claude-sonnet-4-20250514"
    anthropic_max_tokens : int = 2048

    # ── Embeddings ──────────────────────────────────────
    embedding_model      : str = "BAAI/bge-base-en-v1.5"
    embedding_dimension  : int = 768
    embedding_batch_size : int = 32

    # ── Supabase ────────────────────────────────────────
    supabase_url    : str = ""
    supabase_key    : str = ""
    supabase_db_url : str = ""

    # ── RAG Core Settings ───────────────────────────────
    chunk_size         : int   = 500
    chunk_overlap      : int   = 50
    top_k_results      : int   = 5
    min_confidence     : float = 0.5   # below this refuse to answer
    high_confidence    : float = 0.8   # above this answer directly

    # ── Hybrid Search ───────────────────────────────────
    vector_weight      : float = 0.7   # weight for vector search
    bm25_weight        : float = 0.3   # weight for BM25 search

    # ── Query Rewriting ─────────────────────────────────
    enable_query_rewriting : bool = True

    # ── API ─────────────────────────────────────────────
    api_host       : str = "0.0.0.0"
    api_port       : int = 8000
    api_secret_key : str = "change-me-in-production"

    # ── Langfuse (query monitoring) ─────────────────────
    langfuse_public_key : str = ""
    langfuse_secret_key : str = ""
    langfuse_host       : str = "https://cloud.langfuse.com"

    # ── MLflow (experiment tracking) ────────────────────
    mlflow_tracking_uri    : str = "mlflow/experiments"
    mlflow_experiment_name : str = "craster-rag"

    # ── Logging ─────────────────────────────────────────
    log_level  : str = "INFO"
    log_format : str = "json"

    # ── Ollama (local free models) ──────────────────────
    ollama_base_url : str = "http://localhost:11434"
    ollama_model    : str = "qwen2.5:7b"

    # ── Caching (Redis) ──────────────────────────────────
    redis_url         : str  = "redis://localhost:6379/0"
    cache_ttl_seconds : int  = 86400   # 24 hours
    enable_caching    : bool = True

    # ── Feature Flags ───────────────────────────────────
    enable_reranking    : bool = True   # cross-encoder rerank, free local model
    enable_monitoring  : bool = True
    enable_hybrid_search: bool = True

    # ── Properties ──────────────────────────────────────
    @property
    def is_local(self) -> bool:
        """True if running on local machine."""
        return self.environment == Environment.LOCAL

    @property
    def is_production(self) -> bool:
        """True if running in production."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_debug(self) -> bool:
        """True if debug mode enabled."""
        return self.debug or self.is_local


# ── Document Category Mapping ──────────────────────────
# Maps each PDF filename to its category.
# Used by:
#   pdf_loader.py   → tags each chunk with category
#   router_agent.py → knows what documents exist per category
#   retriever_agent.py → filters search by category

DOCUMENT_CATEGORIES: dict[str, str] = {

    # ── Leave and Family (9 docs) ──────────────────────
    "Parental leave-day-one-right Policy.pdf"      : DocumentCategory.LEAVE_FAMILY,
    "Paternity Leave.pdf"                           : DocumentCategory.LEAVE_FAMILY,
    "Shared Parental Leave Policy V2.pdf"           : DocumentCategory.LEAVE_FAMILY,
    "Bereaved Partner Paternity Leave Policy.pdf"   : DocumentCategory.LEAVE_FAMILY,
    "Compassionate Leave Policy.pdf"                : DocumentCategory.LEAVE_FAMILY,
    "Neonatal Care Leave Policy.pdf"                : DocumentCategory.LEAVE_FAMILY,
    "Parental Bereavement Policy.pdf"               : DocumentCategory.LEAVE_FAMILY,
    "Carers Leave Policy.pdf"                       : DocumentCategory.LEAVE_FAMILY,
    "Maternity Policy.pdf"                          : DocumentCategory.LEAVE_FAMILY,

    # ── Health and Safety (3 docs) ─────────────────────
    "Lone Worker Policy.pdf"                               : DocumentCategory.HEALTH_SAFETY,
    "H&S Handbook CRASTER GROUP LIMITED - May 2025.pdf"    : DocumentCategory.HEALTH_SAFETY,
    "H&S Policy CRASTER GROUP LIMITED - May 2025.pdf"      : DocumentCategory.HEALTH_SAFETY,

    # ── Employment (8 docs) ────────────────────────────
    "Redundancy Policy.pdf"                        : DocumentCategory.EMPLOYMENT,
    "Sickness And Absence Policy.pdf"              : DocumentCategory.EMPLOYMENT,
    "Flexible Working Policy.pdf"                  : DocumentCategory.EMPLOYMENT,
    "Connected Working Policy.pdf"                 : DocumentCategory.EMPLOYMENT,
    "Disciplinary Procedure - June 19.pdf"         : DocumentCategory.EMPLOYMENT,
    "Grievance Procedure - June 19.pdf"            : DocumentCategory.EMPLOYMENT,
    "Drugs & Alcohol Procedure - June 19.pdf"      : DocumentCategory.EMPLOYMENT,
    "Harassment Procedure - June 19.pdf"           : DocumentCategory.EMPLOYMENT,

    # ── Data and Compliance (6 docs) ───────────────────
    "Whistleblowing Policy V2.pdf"                          : DocumentCategory.DATA_COMPLIANCE,
    "Data Retention Policy.pdf"                             : DocumentCategory.DATA_COMPLIANCE,
    "Data Breach Policy.pdf"                                : DocumentCategory.DATA_COMPLIANCE,
    "Data Subject Access Request Policy and Procedure.pdf"  : DocumentCategory.DATA_COMPLIANCE,
    "Data Protection Policy.pdf"                            : DocumentCategory.DATA_COMPLIANCE,
    "Privacy Notice.pdf"                                    : DocumentCategory.DATA_COMPLIANCE,

    # ── Conduct (6 docs) ───────────────────────────────
    "Prevention of Sexual Harassment Policy V2.pdf" : DocumentCategory.CONDUCT,
    "Harassment-and-Bullying-Policy.pdf"            : DocumentCategory.CONDUCT,
    "Anti-Slavery Policy.pdf"                       : DocumentCategory.CONDUCT,
    "Anti-Bribery and Corruption Policy.pdf"        : DocumentCategory.CONDUCT,
    "Communication, Email and Internet Policy.pdf"  : DocumentCategory.CONDUCT,
    "Use of IT Procedure - June 19.pdf"             : DocumentCategory.CONDUCT,

    # ── Rewards and Benefits (8 docs) ─────────────────
    "Total Reward Policy - V2.pdf"                      : DocumentCategory.REWARDS_BENEFITS,
    "EMPLOYEE ASSISTANCE PROGRAMME.pdf"                 : DocumentCategory.REWARDS_BENEFITS,
    "Craster UKHealthcare Cash Plan Claim Guide.pdf"    : DocumentCategory.REWARDS_BENEFITS,
    "Expenses Policy.pdf"                               : DocumentCategory.REWARDS_BENEFITS,
    "Season Ticket Loan Policy.pdf"                     : DocumentCategory.REWARDS_BENEFITS,
    "Recommend a Friend Scheme.pdf"                     : DocumentCategory.REWARDS_BENEFITS,
    "Pride Award Scheme.pdf"                            : DocumentCategory.REWARDS_BENEFITS,
    "Pension Scheme Change QA - March 2022.pdf"         : DocumentCategory.REWARDS_BENEFITS,

    # ── Equality and Diversity (2 docs) ────────────────
    "Equal Opportunities Procedure - June 19.pdf"  : DocumentCategory.EQUALITY_DIVERSITY,
    "Equality & Diversity Policy.pdf"              : DocumentCategory.EQUALITY_DIVERSITY,

    # ── Company General (6 docs) ───────────────────────
    "Craster Employee Handbook - March 2025.pdf"               : DocumentCategory.COMPANY_GENERAL,
    "Meeting Etiquette - 2026.pdf"                             : DocumentCategory.COMPANY_GENERAL,
    "Environmental Policy.pdf"                                 : DocumentCategory.COMPANY_GENERAL,
    "Business Ethical Policy.pdf"                              : DocumentCategory.COMPANY_GENERAL,
    "Group Business Travel Policy expires 14 July 22.pdf"      : DocumentCategory.COMPANY_GENERAL,
    "MyePayWindow Guidance for Employees.pdf"                  : DocumentCategory.COMPANY_GENERAL,
}


# ── Category Descriptions ──────────────────────────────
# Used by router_agent to classify questions.
# More detailed = more accurate routing.

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    DocumentCategory.LEAVE_FAMILY: """
        Questions about any type of leave or family related policies.
        Includes: maternity leave, paternity leave, parental leave,
        shared parental leave, neonatal care leave, compassionate leave,
        carers leave, bereavement leave, adoption leave,
        keeping in touch days, return to work after leave.
        Example questions:
            how much maternity leave am I entitled to?
            can I take paternity leave from day one?
            what is shared parental leave?
            how do I apply for compassionate leave?
    """,

    DocumentCategory.HEALTH_SAFETY: """
        Questions about health, safety and wellbeing at work.
        Includes: lone working, risk assessments, accident reporting,
        first aid, fire safety, manual handling, display screen equipment,
        workplace hazards, personal protective equipment.
        Example questions:
            what are the rules for lone working?
            how do I report a workplace accident?
            what are my health and safety rights?
            who is responsible for health and safety?
    """,

    DocumentCategory.EMPLOYMENT: """
        Questions about employment terms, conditions and procedures.
        Includes: redundancy, sickness, absence, flexible working,
        connected working, remote working, disciplinary procedures,
        grievance procedures, drugs and alcohol, notice periods,
        performance management, contracts.
        Example questions:
            what happens if I am made redundant?
            how do I raise a grievance?
            what is the disciplinary procedure?
            can I work flexibly or from home?
            how many sick days am I entitled to?
    """,

    DocumentCategory.DATA_COMPLIANCE: """
        Questions about data protection, privacy and compliance.
        Includes: GDPR, data protection, data breaches, subject access
        requests, whistleblowing, privacy, data retention,
        personal data, information security.
        Example questions:
            how do I report a data breach?
            what is a subject access request?
            how do I raise a whistleblowing concern?
            how long is my data kept?
            what are my data protection rights?
    """,

    DocumentCategory.CONDUCT: """
        Questions about workplace conduct and behaviour policies.
        Includes: harassment, bullying, sexual harassment,
        anti-slavery, anti-bribery, corruption, IT usage,
        email policy, internet usage, social media,
        conflicts of interest, gifts and hospitality.
        Example questions:
            what counts as workplace harassment?
            what is the anti-bribery policy?
            what can I use company IT for?
            how do I report bullying?
            what is the modern slavery policy?
    """,

    DocumentCategory.REWARDS_BENEFITS: """
        Questions about pay, rewards, benefits and perks.
        Includes: salary, pay reviews, bonuses, healthcare,
        expenses, season ticket loans, pension, employee assistance,
        recommend a friend, pride awards, total reward.
        Example questions:
            how do I claim expenses?
            what healthcare benefits do I have?
            how does the season ticket loan work?
            what is the pension scheme?
            how do I use the employee assistance programme?
    """,

    DocumentCategory.EQUALITY_DIVERSITY: """
        Questions about equality, diversity and inclusion.
        Includes: equal opportunities, diversity policy,
        discrimination, protected characteristics,
        reasonable adjustments, inclusive workplace.
        Example questions:
            what are the equal opportunities policies?
            what counts as discrimination?
            how does the company support diversity?
            what reasonable adjustments can I request?
    """,

    DocumentCategory.COMPANY_GENERAL: """
        Questions about general company policies and information.
        Includes: employee handbook, meeting etiquette,
        environmental policy, business ethics, travel policy,
        payslips, company values, office information.
        Example questions:
            where can I find the employee handbook?
            what is the travel policy?
            what are the meeting etiquette guidelines?
            how do I access my payslip?
            what is the environmental policy?
    """,

    DocumentCategory.GENERAL: """
        Question spans multiple categories or is unclear.
        Search across all documents.
    """,
}


# ── Single Instance ────────────────────────────────────
# Import this everywhere:
#   from config import settings, DOCUMENT_CATEGORIES, CATEGORY_DESCRIPTIONS
settings = Settings()