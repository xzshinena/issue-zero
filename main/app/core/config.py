from pydantic_settings import BaseSettings, SettingsConfigDict


def _build_database_url(
    host: str,
    port: str,
    user: str,
    password: str,
    dbname: str,
    sslmode: str = "require",
) -> str:
    from urllib.parse import quote_plus
    password = (password or "").strip()
    user = (user or "").strip()
    # Avoid URL parsing issues: encode password and user
    safe_pw = quote_plus(password) if password else ""
    safe_user = quote_plus(user) if user else ""
    return f"postgresql://{safe_user}:{safe_pw}@{host.strip()}:{port.strip()}/{dbname.strip()}?sslmode={sslmode.strip()}"
    

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", "../.env"],  # main/.env or project root .env when run from main/
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database: use DATABASE_URL, or build from parts (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)
    # Building from parts avoids password being mangled by URL parsing or .env line endings
    database_url: str = "postgresql://localhost/issue_zero"
    db_host: str = ""
    db_port: str = "5432"
    db_user: str = ""
    db_password: str = ""
    db_name: str = "tsdb"
    db_sslmode: str = "require"

    @property
    def effective_database_url(self) -> str:
        if (self.db_host and self.db_user and self.db_password):
            return _build_database_url(
                self.db_host,
                self.db_port or "5432",
                self.db_user,
                self.db_password,
                self.db_name or "tsdb",
                self.db_sslmode or "require",
            )
        return self.database_url.strip()

    # GitHub (loaded from .env: GITHUB_TOKEN, REPOS_TO_SYNC)
    github_token: str = ""
    repos_to_sync: str = ""

    # Celery broker
    celery_broker_url: str = "redis://localhost:6379/0"  


def get_settings() -> Settings:
    return Settings()
