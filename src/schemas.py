from typing import List

from pydantic import (
    BaseModel,
    Field,
)  # Pydantic for defining data schemas and field validation


class UserInput(BaseModel):
    """Schema for parsing user-provided account information."""

    # `identifier` field: Expects a string, with a description for the LLM to understand its purpose.
    identifier: str = Field(
        description="Identifier, which can be a customer ID, email, or phone number."
    )


class UserProfile(BaseModel):
    # `customer_id`: Required field for the customer's unique identifier.
    customer_id: str = Field(description="The customer ID of the customer")
    # `music_preferences`: A list of strings to store the customer's music interests.
    music_preferences: List[str] = Field(
        description="The music preferences of the customer"
    )


class PythonLibrary(BaseModel):
    customer_id: str = Field(description="The customer ID of the customer")
    music_preferences: List[str] = Field(
        description="The music preferences of the customer"
    )
