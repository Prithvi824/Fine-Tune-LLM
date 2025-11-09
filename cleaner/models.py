"""
This module contains all the models used in the extraction process.
"""

# 1st party imports
from typing import List

# 3rd party imports
from pydantic import BaseModel, Field


class Dialouge(BaseModel):
    """
    This class is used to store a single dialouge.
    """

    name: str = Field(..., description="The name of the person.")
    line: str = Field(..., description="The line of the person.")


class Message(BaseModel):
    """
    This class is used to store a single message.
    """

    role: str = Field(..., description="The role of the sender.")
    content: str = Field(..., description="The content of the message.")


class TrainingData(BaseModel):
    """
    This class is used to store a single training data entry.
    """

    messages: List[Message] = Field(
        ..., description="The messages of the conversation."
    )
