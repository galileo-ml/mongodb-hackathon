"""Pydantic models for API contracts between backend and frontend."""

from pydantic import BaseModel


class PersonData(BaseModel):
    """Structured person information sent to frontend."""
    name: str
    description: str
    relationship: str
    person_id: str | None = None
