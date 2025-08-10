from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Manages application configuration using Pydantic.

    Loads settings from environment variables and .env files.
    """

    # Application settings
    APP_NAME: str = "FastMCPServer"
    LOG_LEVEL: str = "INFO"

    # Model loading from a .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
