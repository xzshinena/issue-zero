from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql://localhost/issue_zero"

    # GitHub (loaded from .env: GITHUB_TOKEN, REPOS_TO_SYNC)
    github_token: str = ""
    repos_to_sync: str = ""  


def get_settings() -> Settings:
    return Settings()
