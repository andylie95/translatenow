"""
Chatbot App for Cognitive Debriefing Interview

Author: Dr Musashi Hinck

Version Log:
- 02.04.24: Initial demo with passed values from Qualtrics survey
- 07.04.24: Added configurations for survey edition

Notes:
- Need to call Request from start state
- Example URL: localhost:7860/?user=123&session=456&questionid=0&response=0

TODO:
- Test interview ending behavior: does it get triggered reliably?
- Add password protection

Pre-flight:
- Check dotenv values match Gradio secrets
"""

from __future__ import annotations

import os
import json
import logging
import gradio as gr
from uuid import uuid4
from typing import Generator, Any

from pathlib import Path


from utils import (
    PromptTemplate,
    convert_gradio_to_openai,
    initialize_client,
    load_dotenv,
    upload_azure,
    record_chat,
    ChatLoggerHandler
)


# %% Initialize common assets
base_logger = logging.getLogger(__name__)
chat_logger = ChatLoggerHandler()
if os.environ.get("AZURE_ENDPOINT") is None:  # Set Azure credentials from local files
    load_dotenv()
client = initialize_client()  # Shared across sessions
question_mapping: dict[str, str] = json.loads(Path("assets/question_mapping.json").read_text())


# %% (functions)

# Initialization
# - Record user and session id
# - Record question and response
# - Build system message
# - Build initial message
# - Wrapper - start_survey


def initialize_interview(request: gr.Request) -> tuple:
    """
    Read: Request
    Set: values of userId, sessionId, questionWording, initialMessage, systemMessage
    """
    # Parse request
    request_params = request.query_params
    user_id: str = request_params.get("user", "testUser")
    session_id: str = request_params.get("session", "testSession")
    base_logger.info(f"User: {user_id} (Session: {session_id})")

    # Parse question
    question_id: str = request_params.get("questionid", "0")
    response_id: str = request_params.get("response", "0")
    question_data: dict = json.loads(Path(f"./assets/questions/{question_mapping[question_id]}").read_text())
    question_wording: str = question_data["question"]
    question_choices: str = question_data["choices"]
    response_text: str = question_choices[int(response_id)]
    base_logger.info(f"Question: {question_wording} ({response_text})")

    # Load initial and system messages
    initial_message: str = PromptTemplate.from_file("assets/initial_message.txt").format(surveyQuestion=question_wording)
    system_message: str = PromptTemplate.from_file("assets/system_message.txt").format(surveyQuestion=question_wording, responseVal=response_text)
    base_logger.info(f"Initial message: {initial_message}")
    base_logger.info(f"System message: {system_message}")

    # Return all
    return (
        user_id,
        session_id,
        question_wording,
        initial_message,
        system_message
    )

def initialize_interface(initial_message: str) -> tuple:
    """
    Change interface to interactive mode.
    Read: initial_message
    Set:
        instruction_text: modify (to empty)
        chat_display: set initial_message
        chat_input: update placeholder, make interactive
        chat_submit: make interactive
        start_button: hide
    """
    instruction_text = gr.Markdown("")
    chat_display = gr.Chatbot(
        value=[[None, initial_message]],
        elem_id="chatDisplay",
        show_label=False,
        visible=True,
    )
    chat_input = gr.Textbox(
        placeholder="Type response here. Hit `Enter` or click the arrow to submit.",
        visible=True,
        interactive=True,
        show_label=False,
        scale=10,
    )
    chat_submit = gr.Button(
        "",
        variant="primary",
        interactive=True,
        icon="./arrow_icon.svg",
        visible=True,
    )
    start_button = gr.Button("Start Interview", visible=False, variant="primary")
    return (instruction_text, chat_display, chat_input, chat_submit, start_button)


# Interaction
# - User message
# - Bot message
# - Check if interview finished
# - Record interaction (local log)


def user_message(
    message: str, chat_history: list[list[str | None]]
) -> tuple[str, list[list[str | None]]]:
    "Display user message immediately"
    return "", chat_history + [[message, None]]


def bot_message(
    chat_history: list[list[str | None]],
    system_message: str,
    model_args: dict = {"model": "gpt-4o-default", "temperature": 0.0},
) -> Generator[Any, Any, Any]:
    "Streams response from OpenAI API to chat interface."
    # Prep messages
    user_msg = chat_history[-1][0]
    messages = convert_gradio_to_openai(chat_history[:-1])
    messages = (
        [{"role": "system", "content": system_message}]
        + messages
        + [{"role": "user", "content": user_msg}]
    )
    # API call
    response = client.chat.completions.create(
        messages=messages, stream=True, **model_args
    )
    # Streaming
    chat_history[-1][1] = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            chat_history[-1][1] += delta
            yield chat_history


def log_interaction(
    chat_history: list[list[str | None]],
    session_id: str,
) -> None:
    "Record last pair of interactions"
    record_chat(chat_logger, session_id, "user", chat_history[-1][0])
    record_chat(chat_logger, session_id, "bot", chat_history[-1][1])


def interview_end_check(
    chat_history: list[list[str | None]],
    limit: int = 20,
    end_of_interview: str = "<end_of_survey>",
) -> tuple[list[list[str | None]], gr.Button, gr.Textbox, gr.Button]:
    """
    Checks if interview has completed using two conditions:
    1. If the last bot message contains `end_of_interview` (default: "<end_of_survey>". Replaced "<end_interview>" with this new default token by Kentaro)
    2. Conversation length has reached `limit` (default: 10)

    If either condition is met, the end of interview button is displayed.
    """
    flag = False
    if len(chat_history) >= limit:
        flag = True
    if end_of_interview in chat_history[-1][1]:
        chat_history[-1][1] = chat_history[-1][1].replace(end_of_interview, "")
        flag = True
    input_button = gr.Textbox(
        placeholder="Type response here. Hit `Enter` or click the arrow to submit.",
        visible= not flag,
        interactive=True,
        show_label=False,
        scale=10,
    )
    submit_button = gr.Button(
        "",
        variant="primary",
        interactive=True,
        icon="./arrow_icon.svg",
        visible= not flag,
    )
    button = gr.Button("Save and Exit", visible=flag, variant="stop")
    return chat_history, button, input_button, submit_button


# Completion
# - Create completion code
# - Append to message history
# - Display completion code


def generate_completion_code(prefix: str = "cd-") -> str:
    return prefix + str(uuid4())


def upload_interview(
    session_id: str,
    chat_history: list[list[str | None]],
) -> None:
    "Upload chat history to Azure blob storage"
    upload_azure(session_id, chat_history)


def end_interview(
    session_id: str,
    chat_history: list[list[str | None]],
) -> tuple[list[list[str | None]], gr.Text]:
    """Create completion code and display in chat interface."""
    completion_message = (
        "Thank you for participating.\n\n"
        "Your completion code is: {}\n\n"
        "Please now return to the Qualtrics survey "
        "and paste this code into the  completion "
        "code box.".format(generate_completion_code())
    )
    upload_interview(session_id, chat_history)
    EndMessage = gr.Text(completion_message, visible=True, show_label=False, scale=10)
    return chat_history, EndMessage

# LAYOUT
with gr.Blocks(theme="sudeepshouche/minimalist") as demo:
    # Header and instructions
    gr.Markdown("# SurveyGPT Interview")
    instructionText = gr.Markdown(
        "Use this chat interface to talk to SurveyGPT.\n"
        "To start, click 'Start Interview' and follow the instructions.\n\n"
        "You can type your answer into the box below and hit 'Enter' or click the arrow to submit.\n\n"
        "The interview will end either after 2 minutes, or if the chatbot decides the interview is done.\n"
        "At this point, you will see a 'Save and Exit' button. Click this to save your responses and receive a completion code."
    )
    # Initialize empty hidden values.
    userId = gr.State()
    sessionId = gr.State()
    questionWording = gr.State()
    initialMessage = gr.State()
    systemMessage = gr.State()
    modelArgs = gr.State(value={"model": "gpt-4o-default", "temperature": 0.0})

    # Chat app (display, input, submit button)
    startButton = gr.Button("Start Interview", visible=True, variant="primary")
    chatDisplay = gr.Chatbot(
        value=None,
        elem_id="chatDisplay",
        show_label=False,
        visible=True,
    )

    EndMessage = gr.Text("", visible=False, show_label=False, scale=10)

    with gr.Row():  # Interaction
        chatInput = gr.Textbox(
            placeholder="Click 'Start Interview' to begin.",
            visible=False,
            interactive=False,
            show_label=False,
            scale=10,
        )
        chatSubmit = gr.Button(
            "",
            variant="primary",
            visible=False,
            interactive=False,
            icon="./arrow_icon.svg",
        )
    exitButton = gr.Button("Generate Completion Code", visible=False, variant="stop")
    # testExitButton = gr.Button("Save and Exit", visible=True, variant="stop")
    # Footer
    disclaimer = gr.HTML(
        """
        <div
        style='font-size: 1em;
               font-style: italic;   
               position: fixed;
               left: 50%;
               bottom: 20px;
               transform: translate(-50%, -50%);
               margin: 0 auto;
               '
        >{}</div>
        """.format(
            "Statements by the chatbot may contain factual inaccuracies."
        )
    )

    # INTERACTIONS
    # Initialization
    startButton.click(
        initialize_interview, # Reads in request params
        inputs=None,
        outputs=[
            userId,
            sessionId,
            questionWording,
            initialMessage,
            systemMessage,
        ],
    ).then(
        initialize_interface, # Changes interface to interactive mode
        inputs=[initialMessage],
        outputs=[
            instructionText,
            chatDisplay,
            chatInput,
            chatSubmit,
            startButton,
        ],
    )
    # Chat interaction
    # "Enter"
    chatInput.submit(
        user_message,
        inputs=[chatInput, chatDisplay],
        outputs=[chatInput, chatDisplay],
        queue=False,
    ).then(
        bot_message,
        inputs=[chatDisplay, systemMessage, modelArgs],
        outputs=[chatDisplay],
    ).then(
        log_interaction,
        inputs=[chatDisplay, sessionId],
    ).then(
        interview_end_check, inputs=[chatDisplay], outputs=[chatDisplay, exitButton, chatInput, chatSubmit]
    )

    # Button
    chatSubmit.click(
        user_message,
        inputs=[chatInput, chatDisplay],
        outputs=[chatInput, chatDisplay],
        queue=False,
    ).then(
        bot_message,
        inputs=[chatDisplay, systemMessage, modelArgs],
        outputs=[chatDisplay],
    ).then(
        log_interaction,
        inputs=[chatDisplay, sessionId],
    ).then(
        interview_end_check, inputs=[chatDisplay], outputs=[chatDisplay, exitButton, chatInput, chatSubmit]
    )

    # Reset button
    exitButton.click(
        end_interview, inputs=[sessionId, chatDisplay], outputs=[chatDisplay, EndMessage]
    )
    # testExitButton.click(
    #     end_interview, inputs=[sessionId, chatDisplay], outputs=[chatDisplay]
    # )


if __name__ == "__main__":
    demo.launch()#auth=auth_no_user)
