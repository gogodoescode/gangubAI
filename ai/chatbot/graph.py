"""LangGraph agent: chat node + tool node, with keyword fallbacks for RAG and movement."""

import re
from typing import Annotated, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from ai.chatbot.config import GROQ_MODEL, SYSTEM_PROMPT, TEMPERATURE
from ai.chatbot.tools import move_robot, pomodoro, retrieve_context, set_wander_mode, timer


# ── State and structured outputs ───────────────────────────
EMOTION_VALUES = Literal[
    "happy", "sad", "angry", "curious",
    "excited", "confused", "neutral", "thinking",
]
_EMOTION_DESCRIPTION = (
    "The single emotion that best matches this response. "
    "One of: happy, sad, angry, curious, excited, confused, neutral, thinking."
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_emotion: str


class ChatResponse(BaseModel):
    """Standard chatbot reply with an associated emotion."""
    content: str = Field(description="The chatbot's reply to the user.")
    emotion: EMOTION_VALUES = Field(description=_EMOTION_DESCRIPTION, default="neutral")


class RAGResponse(BaseModel):
    """RAG-grounded reply with citation and emotion."""
    explanation: str = Field(
        description="Detailed explanation answering the user's question based on the retrieved context"
    )
    citation: str = Field(
        description="Citation in the format: 'For more reference check <source>, <type> <number>' "
                    "e.g. 'For more reference check ECC_1_Introduction.pptx, Slide 3'"
    )
    emotion: EMOTION_VALUES = Field(description=_EMOTION_DESCRIPTION, default="neutral")


# ── LLM bindings ──────────────────────────────────────────
tools_list = [retrieve_context, move_robot, set_wander_mode, timer, pomodoro]
model = ChatGroq(model=GROQ_MODEL, temperature=TEMPERATURE)
llm_with_tools = model.bind_tools(tools_list)
llm_for_rag = model.with_structured_output(RAGResponse)
llm_for_chat = model.with_structured_output(ChatResponse)


# ── Keyword fallbacks (used when the LLM fails to emit a tool call) ──
_ROBOT_KEYWORDS = (
    "move", "turn", "rotate", "forward", "backward", "left", "right",
    "stop", "navigate", "motor", "robot",
)
_RETRIEVAL_CUES = (
    "teach", "explain", "detail", "what is", "how does", "why", "history",
    "evolution", "llm", "transformer", "nlp", "slides", "notes", "ppt",
    "chapter", "unit", "from the material", "according to the material",
)
_SMALLTALK_RE = re.compile(r"\b(joke|funny|humor|hello|hi|thanks|how are you|who are you|your name|thank you)\b")


def _looks_like_robot_action(query: str) -> bool:
    return any(keyword in query for keyword in _ROBOT_KEYWORDS)


def _should_force_retrieve(query: str) -> bool:
    if _looks_like_robot_action(query) or _SMALLTALK_RE.search(query):
        return False
    return any(cue in query for cue in _RETRIEVAL_CUES) and len(query.split()) >= 4


def _infer_direction(query: str) -> str:
    if any(token in query for token in ("360", "spin", "rotate", "turn around")):
        return "360"
    if "forward" in query or "ahead" in query:
        return "forward"
    if any(token in query for token in ("backward", "reverse", "back")):
        return "backward"
    for direction in ("left", "right", "stop"):
        if direction in query:
            return direction
    return "forward"


def _tool_call(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": f"force_{name}_{uuid4().hex}", "type": "tool_call"}


def _forced_tool_call(query: str) -> ChatState | None:
    """Return a forced tool-call state for movement or study questions, else None."""
    if _looks_like_robot_action(query):
        call = _tool_call("move_robot", {"direction": _infer_direction(query)})
        emotion = "excited"
    elif _should_force_retrieve(query):
        call = _tool_call("retrieve_context", {"query": query})
        emotion = "thinking"
    else:
        return None
    return {"messages": [AIMessage(content="", tool_calls=[call])], "current_emotion": emotion}


# ── Helpers ────────────────────────────────────────────────
def _latest_user_query(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content).strip()
    return ""


def _retrieved_context(messages: list[BaseMessage]) -> str | None:
    """Return retrieve_context output if it ran in the latest tool round-trip."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return None
        if isinstance(msg, ToolMessage) and msg.name == "retrieve_context":
            return str(msg.content)
    return None


def _answer_from_context(user_question: str, context: str) -> ChatState:
    # Clean prompt so the structured-output model doesn't see raw tool-call artefacts.
    rag_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User question: {user_question}\n\n"
                f"Retrieved context:\n{context}\n\n"
                "Using the retrieved context, answer the user's question. "
                "Explain in your own words — simple and clear."
            )
        ),
    ]
    try:
        response: RAGResponse = llm_for_rag.invoke(rag_messages)
    except Exception:
        text = str(model.invoke(rag_messages).content).strip()
        return {
            "messages": [AIMessage(content=text or "I found relevant context, but had trouble formatting the response.")],
            "current_emotion": "thinking",
        }
    return {
        "messages": [AIMessage(content=f"{response.explanation}\n\n {response.citation}")],
        "current_emotion": response.emotion,
    }


def _plain_reply(all_messages: list[BaseMessage]) -> ChatState:
    try:
        response: ChatResponse = llm_for_chat.invoke(all_messages)
    except Exception:
        text = str(model.invoke(all_messages).content).strip()
        return {"messages": [AIMessage(content=text or "I am here.")], "current_emotion": "neutral"}
    return {"messages": [AIMessage(content=response.content)], "current_emotion": response.emotion}


# ── Chat node ─────────────────────────────────────────────
def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    query = _latest_user_query(messages)

    context = _retrieved_context(messages)
    if context is not None:
        return _answer_from_context(query, context)

    all_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    try:
        tool_response = llm_with_tools.invoke(all_messages)
        if tool_response.tool_calls:
            return {"messages": [tool_response]}
    except Exception:
        # Groq sometimes rejects its own tool-call output; fall through to keyword fallbacks.
        pass

    # Only force a tool before any tool has run this turn, otherwise we'd loop.
    if isinstance(messages[-1], HumanMessage):
        forced = _forced_tool_call(query.lower())
        if forced is not None:
            return forced
    return _plain_reply(all_messages)


# ── Build graph ───────────────────────────────────────────
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_node("tools", ToolNode(tools_list))
graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)
graph.add_edge("tools", "chat")

# For LangGraph API – compile without checkpointer (API handles persistence)
agent = graph.compile()
