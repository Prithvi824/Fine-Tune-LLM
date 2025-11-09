"""
This module is responsible for extracting the data from the chat file and saving it in a jsonl file.
"""

# 1st party imports
import os
import re
import csv
from typing import List, Tuple

# local imports
from settings import settings
from log_config import get_logger
from .models import TrainingData, Dialouge, Message

# initialize the logger
logger = get_logger(__name__)


class Extractor:
    """
    This class is responsible for extracting the data from the transcript file and saving it in a jsonl file.
    """

    def __init__(self, transcript_file_path: str, jsonl_file_path: str):
        """
        Initialize the Extractor class.

        Args:
            transcript_file_path (str): Path to the transcript file. example: Darling in The Franxx Removed Quotes Transcript.csv
            jsonl_file_path (str): Path where the jsonl file will be saved.
        """

        # Initialize the transcript file path and jsonl file path
        self.transcript_file_path = transcript_file_path
        self.jsonl_file_path = jsonl_file_path

    def __filter_transcript_line(self, line: str) -> str:
        """
        Filter the transcript file to only include the lines from the character.

        Args:
            line (str): The line to be filtered.

        Returns:
            str: The filtered line.
        """

        # return the line if it is from the character
        return re.sub(r"(\\\w)|(\{.*?\})", "", line)

    def load_transcript_file(self) -> List[Dialouge]:
        """
        Load the transcript data from the transcript file.

        Returns:
            List[Dialouge]: The transcript data.
        """

        with open(self.transcript_file_path, newline="", encoding="utf-8") as f:

            # read the csv lines
            csv_lines = csv.reader(f)

            # skip the first line
            next(csv_lines)

            # return the csv lines
            return [
                Dialouge(name=line[0], line="".join(line[1:])) for line in csv_lines
            ]

    def save_data(self, data: List[TrainingData]):
        """
        Save the data to a jsonl file.

        Args:
            data (List[TrainingData]): The data to be saved.
        """

        # make sure the directory exists
        os.makedirs(os.path.dirname(self.jsonl_file_path), exist_ok=True)

        # Save the data to a jsonl file
        with open(self.jsonl_file_path, "w", encoding="utf-8") as f:

            # for each entry
            for entry in data:

                # write the entry to the file
                f.write(entry.model_dump_json() + "\n")

    def get_context(
        self,
        transcript_data: List[Dialouge],
        end_idx: int,
        character_name: str,
        context_window: int = 5,
    ):
        """
        Get the context of the conversation.

        Args:
            transcript_data (List[Dialouge]): The transcript data.
            end_idx (int): The index upto which the context is needed.
            character_name (str): The name of the character.
            context_window (int): The number of lines to consider for context.

        Returns:
            str: The context of the conversation.
        """

        # start to get the context from
        start_idx = max(0, end_idx - context_window)

        # context builder
        context = []

        # iterate over the transcript data
        for line in transcript_data[start_idx:end_idx]:

            # check if the line is from the character
            if line.name == character_name:
                context = []
                continue

            # add the dialouge to the context
            context.append(line.line)

        # return
        return self.__filter_transcript_line(" ".join(context)).strip()

    def get_all_dialouges(
        self,
        transcript_data: List[Dialouge],
        start_idx: int,
        character_name: str,
    ) -> Tuple[str, int]:
        """
        Get all the next consecutive dialouges of the conversation.
        If no consecutive dialogues exist (alternating speakers), returns just the current dialogue.

        Args:
            transcript_data (List[Dialouge]): The transcript data.
            start_idx (int): The index from which the dialouges are needed.
            character_name (str): The name of the character.

        Returns:
            Tuple[str, int]: The next dialouges and the index of the last dialouge captured.
        """

        # context builder
        context = []

        # iterate over the transcript data
        for line in transcript_data[start_idx:]:

            # check the name of the character
            if line.name != character_name:
                break

            # add the dialouge to the context
            context.append(line.line.strip())

        # return
        return (
            self.__filter_transcript_line(" ".join(context)),
            start_idx + len(context),
        )

    def extract_and_save_data(self, character_name: str):
        """
        Extract the data from the transcript file and save it in a jsonl file.

        Args:
            character_name (str): The name of the character to extract data for.
        """

        # Load the transcript file
        transcript_data = self.load_transcript_file()
        logger.info(f"Loaded transcript lines: {len(transcript_data)}")

        # Extract the data
        data: List[TrainingData] = []
        continue_idx = 0

        # iterate over the transcript lines
        for idx, line in enumerate(transcript_data):

            # check if the line is from the character
            if character_name != line.name or idx <= continue_idx:
                continue

            # get the last n dialouges for context
            context = self.get_context(
                transcript_data, idx, character_name, settings.context_window
            )

            # if no context, skip
            if not context:
                continue

            # get the next dialouges
            next_dialouges, continue_idx = self.get_all_dialouges(
                transcript_data, idx, character_name
            )

            # create the messages
            training_data = TrainingData(
                messages=[
                    Message(role="user", content=context),
                    Message(role="assistant", content=next_dialouges),
                ]
            )

            # add the chat entry to the list
            data.append(training_data)

        # save the data
        logger.info(f"Extracted data: {len(data)}")
        self.save_data(data)
        return None
