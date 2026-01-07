from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "models/final_model.joblib"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()
