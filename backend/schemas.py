from pydantic import BaseModel, Field


class CheckInRequest(BaseModel):
    age: int = Field(..., ge=10, le=100)
    screen_time_hrs: float = Field(..., ge=0, le=16)
    sleep_quality: float = Field(..., ge=1, le=10)
    stress_level: float = Field(..., ge=1, le=10)
    days_without_social_media: float = Field(..., ge=0, le=30)
    exercise_frequency: float = Field(..., ge=0, le=14)
    happiness_index: float = Field(..., ge=1, le=10)


class ChatMessageRequest(BaseModel):
    session_id: str
    message: str
