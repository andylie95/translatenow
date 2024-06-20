from __future__ import annotations

import datetime
import json
import os
from configparser import ConfigParser
from pathlib import Path
from string import Formatter
from dotenv import dotenv_values

import openai
from azure.storage.blob import BlobClient


# Logging util
def get_current_timestamp() -> str:
    return datetime.datetime.now().isoformat()


class ChatLoggerHandler:
    """Shared logging handler for chat logs. Runs common to all Gradio sessions."""

    def __init__(self, logdir: str = "./logs") -> None:
        self.logdir: Path = Path(logdir)
        if not self.logdir.exists():
            self.logdir.mkdir()

    def record(self, session: str, role: str, record: str):
        log_entry = {
            "session": session,
            "timestamp": get_current_timestamp(),
            "role": role,
            "message": record,
        }
        with open(self.logdir / f"{session}.jsonl", "a+") as f:
            f.write(json.dumps(log_entry) + "\n")


def record_chat(
    logger: ChatLoggerHandler, session: str, role: str, record: str
) -> None:
    logger.record(session, role, record)


# General Class
class PromptTemplate(str):
    """More robust String Formatter. Takes a string and parses out the keywords."""

    def __init__(self, template: str) -> None:
        self.template: str = template
        self.variables: list[str] = self.parse_template()

    def parse_template(self) -> list[str]:
        "Returns template variables"
        return [
            fn for _, fn, _, _ in Formatter().parse(self.template) if fn is not None
        ]

    def format(self, *args, **kwargs) -> str:
        """
        Formats the template string with the given arguments.
        Provides slightly more informative error handling.

        :param args: Positional arguments for unnamed placeholders.
        :param kwargs: Keyword arguments for named placeholders.
        :return: Formatted string.
        :raises: ValueError if arguments do not match template variables.
        """
        # If keyword arguments are provided, check if they match the template variables
        if kwargs and set(kwargs) != set(self.variables):
            raise ValueError("Keyword arguments do not match template variables.")

        # If positional arguments are provided, check if their count matches the number of template variables
        if args and len(args) != len(self.variables):
            raise ValueError(
                "Number of arguments does not match the number of template variables."
            )

        # Check if a dictionary is passed as a single positional argument
        if len(args) == 1 and isinstance(args[0], dict):
            arg_dict = args[0]
            if set(arg_dict) != set(self.variables):
                raise ValueError("Dictionary keys do not match template variables.")
            return self.template.format(**arg_dict)

        # Check for the special case where both args and kwargs are empty, which means self.variables must also be empty
        if not args and not kwargs and self.variables:
            raise ValueError("No arguments provided, but template expects variables.")

        # Use the arguments to format the template
        try:
            return self.template.format(*args, **kwargs)
        except KeyError as e:
            raise ValueError(f"Missing a keyword argument: {e}")

    @classmethod
    def from_file(cls, file_path: str) -> PromptTemplate:
        with open(file_path, encoding="utf-8") as file:
            template_content = file.read()
        return cls(template_content)

    def dump_prompt(self, file_path: str) -> None:
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(self.template)
            file.close()


def convert_gradio_to_openai(
    chat_history: list[list[str | None]],
) -> list[dict[str, str]]:
    "Converts gradio chat format -> openai chat request format"
    messages = []
    for pair in chat_history:  # [(user), (assistant)]
        for i, role in enumerate(["user", "assistant"]):
            if not ((pair[i] is None) or (pair[i] == "")):
                messages += [{"role": role, "content": pair[i]}]
    return messages


def convert_openai_to_gradio(
    messages: list[dict[str, str]]
) -> list[list[str, str | None]]:
    "Converts openai chat request format -> gradio chat format"
    chat_history = []
    if messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": None})
    for i in range(0, len(messages), 2):
        chat_history.append([messages[i]["content"], messages[i + 1]["content"]])
    return chat_history


def load_dotenv():
    config = dotenv_values(".env")
    for key, value in config.items():
        os.environ[key] = value


def seed_azure_key(cfg: str = "~/.cfg/openai.cfg") -> None:
    config = ConfigParser()
    try:
        config.read(Path(cfg).expanduser())
    except:
        raise ValueError(f"Could not using read file at: {cfg}.")
    os.environ["AZURE_ENDPOINT"] = config["AZURE"]["endpoint"]
    os.environ["AZURE_SECRET"] = config["AZURE"]["key"]


def initialize_client() -> openai.AsyncClient:
    client = openai.AzureOpenAI(
        azure_endpoint=os.environ["AZURE_ENDPOINT"],
        api_key=os.environ["AZURE_SECRET"],
        api_version="2023-05-15",
    )
    return client


def auth_no_user(username, password):
    if password == os.getenv("GRADIO_PASSWORD", ""):
        return True
    else:
        return False


def upload_azure(conversation_id: str, chat_history) -> None:
    # Get blob client
    conn_str = os.getenv("AZURE_CONN_STR")
    container_name = os.getenv("AZURE_CONTAINER_NAME")
    blob_name = conversation_id
    blob_client = BlobClient.from_connection_string(conn_str, container_name, blob_name)

    # Convert chat_history to json lines
    records = convert_gradio_to_openai(chat_history)
    records_text = "\n".join([json.dumps(record) for record in records])
    blob_client.upload_blob(records_text, blob_type="AppendBlob", overwrite=True)
