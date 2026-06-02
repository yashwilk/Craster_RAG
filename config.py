from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    LOCAL      = "local"
    STAGING    = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):  # same as @dataclass
    model_config = SettingsConfigDict(
        env_file          = ".env",
        env_file_encoding = "utf-8",
        case_sensitive    = False,
        extra             = "ignore",
    )

    environment  : Environment = Environment.LOCAL
    app_version  : str         = "0.1.0"
    debug        : bool        = False

    acumatica_base_url : str = ""
    acumatica_username : str = ""
    acumatica_password : str = ""

    anthropic_api_key   : str = ""
    anthropic_model     : str = "claude-sonnet-4-20250514"
    anthropic_max_tokens: int = 1024

    embedding_model    : str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768
    embedding_batch_size: int = 32

    supabase_url       : str = ""
    supabase_key       : str = ""
    supabase_db_url    : str = ""

    chunk_size         : int = 500
    chunk_overlap      : int = 50
    top_k_results      : int = 5

    api_host           : str = "0.0.0.0"
    api_port           : int = 8000
    api_secret_key     : str = "change-me-in-production"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host      : str = "https://cloud.langfuse.com"

    mlflow_tracking_uri    : str = "mlflow/experiments"
    mlflow_experiment_name : str = "craster-rag"

    log_level          : str = "INFO"
    log_format         : str = "json"

    enable_reranking   : bool = False   # enable after Phase 1
    enable_monitoring  : bool = True

    @property
    def is_local(self) -> bool:
        return self.environment == Environment.LOCAL

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION


# ── Single instance used across entire project ──────────
settings = Settings()
