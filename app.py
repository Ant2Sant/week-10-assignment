import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st


st.set_page_config(page_title="My AI Chat", layout="wide")

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
API_URL = "https://router.huggingface.co/v1/chat/completions"
CHATS_DIR = Path("chats")
MEMORY_PATH = Path("memory.json")


def load_hf_token() -> str | None:
    """Return the Hugging Face token from Streamlit secrets or the environment."""
    token = None

    try:
        token = st.secrets["HF_TOKEN"]
    except Exception:
        token = os.getenv("HF_TOKEN")

    token = str(token or "").strip()
    if not token:
        st.error(
            "Missing Hugging Face token. Add HF_TOKEN to .streamlit/secrets.toml, "
            "set the HF_TOKEN environment variable, or update your Streamlit deployment secrets."
        )
        return None

    return token


def create_timestamp() -> str:
    """Return a stable timestamp string for storage."""
    return datetime.now().isoformat(timespec="seconds")


def format_timestamp(timestamp: str) -> str:
    """Format stored timestamps for display in the sidebar."""
    try:
        return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %I:%M %p")
    except ValueError:
        return timestamp


def get_chat_path(chat_id: str) -> Path:
    """Return the JSON path for a chat."""
    return CHATS_DIR / f"{chat_id}.json"


def save_chat(chat_id: str) -> None:
    """Persist a single chat to disk."""
    CHATS_DIR.mkdir(exist_ok=True)
    chat = st.session_state.chats[chat_id]
    chat["updated_at"] = create_timestamp()
    get_chat_path(chat_id).write_text(json.dumps(chat, indent=2), encoding="utf-8")


def load_memory() -> dict:
    """Load persistent user memory from disk."""
    if not MEMORY_PATH.exists():
        return {}

    try:
        memory = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return memory if isinstance(memory, dict) else {}


def save_memory() -> None:
    """Persist user memory to disk."""
    MEMORY_PATH.write_text(json.dumps(st.session_state.memory, indent=2), encoding="utf-8")


def clear_memory() -> None:
    """Reset memory in session state and on disk."""
    st.session_state.memory = {}
    save_memory()


def load_chats_from_disk() -> dict[str, dict]:
    """Load all saved chats from the chats directory."""
    CHATS_DIR.mkdir(exist_ok=True)
    chats: dict[str, dict] = {}

    for chat_file in sorted(CHATS_DIR.glob("*.json")):
        try:
            chat = json.loads(chat_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        chat_id = str(chat.get("chat_id", "")).strip()
        if not chat_id:
            continue

        chats[chat_id] = {
            "chat_id": chat_id,
            "title": str(chat.get("title") or f"Chat {chat_id}"),
            "created_at": str(chat.get("created_at") or create_timestamp()),
            "updated_at": str(chat.get("updated_at") or chat.get("created_at") or create_timestamp()),
            "messages": list(chat.get("messages") or []),
        }

    return chats


def initialize_session_state() -> None:
    """Create chat storage once per user session and preload saved chats."""
    if "chats" not in st.session_state:
        st.session_state.chats = load_chats_from_disk()
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None
    if "next_chat_id" not in st.session_state:
        numeric_ids = [int(chat_id) for chat_id in st.session_state.chats if chat_id.isdigit()]
        st.session_state.next_chat_id = max(numeric_ids, default=0) + 1
    if "memory" not in st.session_state:
        st.session_state.memory = load_memory()

    if st.session_state.active_chat_id not in st.session_state.chats:
        remaining_chat_ids = list(st.session_state.chats.keys())
        st.session_state.active_chat_id = remaining_chat_ids[0] if remaining_chat_ids else None


def create_new_chat() -> None:
    """Create a new empty chat, persist it, and make it active."""
    chat_id = str(st.session_state.next_chat_id)
    st.session_state.next_chat_id += 1

    chat = {
        "chat_id": chat_id,
        "title": f"New Chat {chat_id}",
        "created_at": create_timestamp(),
        "updated_at": create_timestamp(),
        "messages": [],
    }
    st.session_state.chats[chat_id] = chat
    st.session_state.active_chat_id = chat_id
    save_chat(chat_id)


def delete_chat(chat_id: str) -> None:
    """Delete a chat from session state and disk."""
    if chat_id not in st.session_state.chats:
        return

    del st.session_state.chats[chat_id]

    chat_path = get_chat_path(chat_id)
    if chat_path.exists():
        chat_path.unlink()

    if st.session_state.active_chat_id == chat_id:
        remaining_chat_ids = list(st.session_state.chats.keys())
        st.session_state.active_chat_id = remaining_chat_ids[0] if remaining_chat_ids else None


def get_active_chat() -> dict | None:
    """Return the active chat dictionary if one exists."""
    active_chat_id = st.session_state.active_chat_id
    if active_chat_id is None:
        return None
    return st.session_state.chats.get(active_chat_id)


def build_system_prompt() -> str:
    """Create the system prompt, including saved user memory when available."""
    memory = st.session_state.memory
    prompt_lines = [
        "You are a helpful AI chat assistant. Respond clearly and briefly.",
        "Use the conversation history to maintain context.",
    ]

    if memory:
        prompt_lines.extend(
            [
                "Personalize your responses using this saved user memory when relevant:",
                json.dumps(memory, ensure_ascii=True),
            ]
        )

    return "\n".join(prompt_lines)

def build_api_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert session chat messages into router chat-completions messages."""
    api_messages = [{"role": "system", "content": build_system_prompt()}]
    for message in messages:
        api_messages.append({"role": message["role"], "content": message["content"]})
    return api_messages


def request_assistant_reply(token: str, messages: list[dict[str, str]]) -> tuple[str | None, str | None]:
    """Send the full conversation to Hugging Face and stream the reply into the UI."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": build_api_messages(messages),
        "stream": True,
        "max_tokens": 120,
    }

    try:
        response = requests.post(
            API_URL,
            headers={**headers, "Accept": "text/event-stream"},
            json=payload,
            timeout=45,
            stream=True,
        )
    except requests.exceptions.Timeout:
        return None, "The request timed out. Hugging Face may be busy, so try again shortly."
    except requests.exceptions.RequestException:
        return None, "Network error while contacting Hugging Face. Check your connection and try again."

    if response.status_code == 401:
        return None, "Invalid Hugging Face token. Update HF_TOKEN in your secrets and try again."
    if response.status_code == 429:
        return None, "Rate limit reached. Wait a moment and send your message again."
    if response.status_code >= 400:
        try:
            details = response.json().get("error", "Unknown API error.")
        except ValueError:
            details = response.text or "Unknown API error."
        return None, f"Hugging Face API error: {details}"

    response_placeholder = st.empty()
    response_chunks: list[str] = []

    try:
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            event_data = line.removeprefix("data:").strip()
            if event_data == "[DONE]":
                break

            try:
                payload = json.loads(event_data)
            except json.JSONDecodeError:
                continue

            if payload.get("error"):
                return None, f"Hugging Face API error: {payload['error']}"

            chunk_text = ""

            choices = payload.get("choices")
            if not chunk_text and isinstance(choices, list) and choices:
                delta = choices[0].get("delta", {})
                if isinstance(delta, dict):
                    chunk_text = delta.get("content", "")

            if chunk_text:
                response_chunks.append(chunk_text)
                response_placeholder.markdown("".join(response_chunks))
                time.sleep(0.03)
    finally:
        response.close()

    full_response = "".join(response_chunks).strip()
    if not full_response:
        return None, "The model response was empty or in an unexpected format."

    return full_response, None


def extract_memory_from_message(token: str, user_message: str) -> tuple[dict | None, str | None]:
    """Ask the model to extract user traits/preferences as JSON."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract any personal facts or preferences from the user's message as JSON only. "
                    "Useful keys can include name, preferred_language, interests, communication_style, "
                    "favorite_topics, likes, dislikes, or goals. If there is nothing useful, return {}."
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "stream": False,
        "max_tokens": 120,
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    except requests.exceptions.Timeout:
        return None, "Memory extraction timed out."
    except requests.exceptions.RequestException:
        return None, "Memory extraction failed due to a network error."

    if response.status_code >= 400:
        return None, None

    try:
        data = response.json()
    except ValueError:
        return None, None

    generated_text = ""
    if isinstance(data, dict):
        choices = data.get("choices", [])
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                generated_text = str(message.get("content", "")).strip()

    if not generated_text:
        return {}, None

    try:
        extracted = json.loads(generated_text)
    except json.JSONDecodeError:
        start_index = generated_text.find("{")
        end_index = generated_text.rfind("}")
        if start_index == -1 or end_index == -1 or end_index < start_index:
            return {}, None
        try:
            extracted = json.loads(generated_text[start_index : end_index + 1])
        except json.JSONDecodeError:
            return {}, None

    if not isinstance(extracted, dict):
        return {}, None

    return extracted, None


def merge_memory(new_memory: dict) -> None:
    """Merge extracted memory into persistent user memory."""
    if not new_memory:
        return

    for key, value in new_memory.items():
        normalized_key = str(key).strip().lower().replace(" ", "_")
        if not normalized_key:
            continue

        if isinstance(value, str):
            cleaned_value = value.strip()
            if cleaned_value:
                st.session_state.memory[normalized_key] = cleaned_value
        elif isinstance(value, list):
            cleaned_items = []
            for item in value:
                if isinstance(item, str):
                    cleaned_item = item.strip()
                    if cleaned_item:
                        cleaned_items.append(cleaned_item)
            if cleaned_items:
                existing_value = st.session_state.memory.get(normalized_key, [])
                if not isinstance(existing_value, list):
                    existing_value = [existing_value] if existing_value else []
                for item in cleaned_items:
                    if item not in existing_value:
                        existing_value.append(item)
                st.session_state.memory[normalized_key] = existing_value
        elif isinstance(value, (int, float, bool)):
            st.session_state.memory[normalized_key] = value

    save_memory()


def update_chat_title(chat: dict) -> None:
    """Use the first user message as the chat title once available."""
    if len(chat["messages"]) != 1:
        return

    first_message = chat["messages"][0]
    if first_message["role"] != "user":
        return

    shortened_title = first_message["content"].strip()[:30]
    chat["title"] = shortened_title or chat["title"]


def render_sidebar() -> None:
    """Render chat navigation using native Streamlit sidebar elements."""
    st.sidebar.title("Chats")

    if st.sidebar.button("New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()

    chat_list_container = st.sidebar.container(height=500)
    with chat_list_container:
        if not st.session_state.chats:
            st.info("No chats yet.")

        sorted_chats = sorted(
            st.session_state.chats.items(),
            key=lambda item: item[1].get("updated_at", item[1].get("created_at", "")),
            reverse=True,
        )

        for chat_id, chat in sorted_chats:
            is_active = chat_id == st.session_state.active_chat_id
            row_columns = st.columns([5, 1])
            title_label = f"{chat['title']}\n{format_timestamp(chat['created_at'])}"

            with row_columns[0]:
                if st.button(
                    title_label,
                    key=f"switch_{chat_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.active_chat_id = chat_id
                    st.rerun()

            with row_columns[1]:
                if st.button("✕", key=f"delete_{chat_id}", use_container_width=True):
                    delete_chat(chat_id)
                    st.rerun()

    with st.sidebar.expander("User Memory", expanded=True):
        if st.session_state.memory:
            st.json(st.session_state.memory)
        else:
            st.write("No saved memory yet.")

        if st.button("Clear Memory", use_container_width=True):
            clear_memory()
            st.rerun()


initialize_session_state()
token = load_hf_token()
render_sidebar()

st.title("My AI Chat")
st.write("Chat with a Hugging Face model using Streamlit's native chat interface.")

active_chat = get_active_chat()
user_prompt = st.chat_input("Type your message here...")

if user_prompt:
    if active_chat is None:
        create_new_chat()
        active_chat = get_active_chat()

    active_chat["messages"].append({"role": "user", "content": user_prompt})
    update_chat_title(active_chat)
    save_chat(active_chat["chat_id"])

chat_history_container = st.container(height=500)
with chat_history_container:
    if active_chat is None:
        st.info("Create a new chat from the sidebar to get started.")
    elif not active_chat["messages"]:
        st.info("Send a message to start this conversation.")
    else:
        for message in active_chat["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        if user_prompt:
            with st.chat_message("assistant"):
                if not token:
                    assistant_reply = None
                    error_message = "I can't reply yet because the Hugging Face token is missing or invalid."
                    st.error(error_message)
                else:
                    assistant_reply, error_message = request_assistant_reply(token, active_chat["messages"])
                    if error_message:
                        st.error(error_message)

            if error_message:
                active_chat["messages"].append({"role": "assistant", "content": error_message})
            else:
                active_chat["messages"].append({"role": "assistant", "content": assistant_reply})
                extracted_memory, _ = extract_memory_from_message(token, user_prompt)
                if extracted_memory is not None:
                    merge_memory(extracted_memory)

            save_chat(active_chat["chat_id"])
            st.rerun()
