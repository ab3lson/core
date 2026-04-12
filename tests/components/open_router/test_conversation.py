"""Tests for the OpenRouter integration."""

import datetime
from unittest.mock import AsyncMock, patch

from freezegun import freeze_time
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
import pytest
from syrupy.assertion import SnapshotAssertion

from homeassistant.components import conversation
from homeassistant.const import Platform
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import entity_registry as er, intent
from homeassistant.helpers.llm import ToolInput

from . import setup_integration
from .conftest import make_stream

from tests.common import MockConfigEntry, snapshot_platform
from tests.components.conversation import MockChatLog, mock_chat_log  # noqa: F401


@pytest.fixture(autouse=True)
def freeze_the_time():
    """Freeze the time."""
    with freeze_time("2024-05-24 12:00:00", tz_offset=0):
        yield


@pytest.mark.parametrize("enable_assist", [True, False], ids=["assist", "no_assist"])
async def test_all_entities(
    hass: HomeAssistant,
    snapshot: SnapshotAssertion,
    mock_openai_client: AsyncMock,
    mock_config_entry: MockConfigEntry,
    entity_registry: er.EntityRegistry,
) -> None:
    """Test all entities."""
    with patch(
        "homeassistant.components.open_router.PLATFORMS",
        [Platform.CONVERSATION],
    ):
        await setup_integration(hass, mock_config_entry)

    await snapshot_platform(hass, entity_registry, snapshot, mock_config_entry.entry_id)


async def test_default_prompt(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    snapshot: SnapshotAssertion,
    mock_openai_client: AsyncMock,
    mock_chat_log: MockChatLog,  # noqa: F811
) -> None:
    """Test that the default prompt works."""
    await setup_integration(hass, mock_config_entry)
    result = await conversation.async_converse(
        hass,
        "hello",
        mock_chat_log.conversation_id,
        Context(),
        agent_id="conversation.gpt_3_5_turbo",
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert mock_chat_log.content[1:] == snapshot
    call = mock_openai_client.chat.completions.create.call_args_list[0][1]
    assert call["model"] == "openai/gpt-3.5-turbo"
    assert call["extra_headers"] == {
        "HTTP-Referer": "https://www.home-assistant.io/integrations/open_router",
        "X-Title": "Home Assistant",
    }


@pytest.mark.parametrize(
    ("web_search", "expected_model_suffix"),
    [(True, ":online"), (False, "")],
    ids=["web_search_enabled", "web_search_disabled"],
)
async def test_web_search(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: AsyncMock,
    mock_chat_log: MockChatLog,  # noqa: F811
    web_search: bool,
    expected_model_suffix: str,
) -> None:
    """Test that web search adds :online suffix to model."""
    await setup_integration(hass, mock_config_entry)
    await conversation.async_converse(
        hass,
        "hello",
        mock_chat_log.conversation_id,
        Context(),
        agent_id="conversation.gpt_3_5_turbo",
    )

    call = mock_openai_client.chat.completions.create.call_args_list[0][1]
    expected_model = f"openai/gpt-3.5-turbo{expected_model_suffix}"
    assert call["model"] == expected_model


async def test_empty_api_response(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_openai_client: AsyncMock,
    mock_chat_log: MockChatLog,  # noqa: F811
) -> None:
    """Test that an empty choices response raises HomeAssistantError."""
    await setup_integration(hass, mock_config_entry)

    # Empty stream — triggers "API returned empty response"
    mock_openai_client.chat.completions.create = make_stream()

    result = await conversation.async_converse(
        hass,
        "hello",
        mock_chat_log.conversation_id,
        Context(),
        agent_id="conversation.gpt_3_5_turbo",
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR


@pytest.mark.parametrize("enable_assist", [True])
async def test_function_call(
    hass: HomeAssistant,
    mock_chat_log: MockChatLog,  # noqa: F811
    mock_config_entry: MockConfigEntry,
    snapshot: SnapshotAssertion,
    mock_openai_client: AsyncMock,
) -> None:
    """Test function call from the assistant."""
    await setup_integration(hass, mock_config_entry)

    # Add some pre-existing content from conversation.default_agent
    mock_chat_log.async_add_user_content(
        conversation.UserContent(content="What time is it?")
    )
    mock_chat_log.async_add_assistant_content_without_tools(
        conversation.AssistantContent(
            agent_id="conversation.gpt_3_5_turbo",
            tool_calls=[
                ToolInput(
                    tool_name="HassGetCurrentTime",
                    tool_args={},
                    id="mock_tool_call_id",
                    external=True,
                )
            ],
        )
    )
    mock_chat_log.async_add_assistant_content_without_tools(
        conversation.ToolResultContent(
            agent_id="conversation.gpt_3_5_turbo",
            tool_call_id="mock_tool_call_id",
            tool_name="HassGetCurrentTime",
            tool_result={
                "speech": {"plain": {"speech": "12:00 PM", "extra_data": None}},
                "response_type": "action_done",
                "speech_slots": {"time": datetime.time(12, 0)},
                "data": {"success": [], "failed": []},
            },
        )
    )
    mock_chat_log.async_add_assistant_content_without_tools(
        conversation.AssistantContent(
            agent_id="conversation.gpt_3_5_turbo",
            content="12:00 PM",
        )
    )

    mock_chat_log.mock_tool_results(
        {
            "call_call_1": "value1",
            "call_call_2": "value2",
        }
    )

    async def _tool_call_stream():
        yield ChatCompletionChunk(
            id="chatcmpl-tool",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_call_1",
                                function=ChoiceDeltaToolCallFunction(
                                    name="test_tool",
                                    arguments='{"param1":"call1"}',
                                ),
                                type="function",
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                    index=0,
                )
            ],
            created=1700000000,
            model="gpt-4-1106-preview",
            object="chat.completion.chunk",
        )

    async def _final_stream():
        yield ChatCompletionChunk(
            id="chatcmpl-final",
            choices=[
                ChunkChoice(
                    delta=ChoiceDelta(
                        role="assistant",
                        content="I have successfully called the function",
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1700000000,
            model="gpt-4-1106-preview",
            object="chat.completion.chunk",
        )

    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=[_tool_call_stream(), _final_stream()]
    )

    result = await conversation.async_converse(
        hass,
        "Please call the test function",
        mock_chat_log.conversation_id,
        Context(),
        agent_id="conversation.gpt_3_5_turbo",
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    # Don't test the prompt, as it's not deterministic
    assert mock_chat_log.content[1:] == snapshot
    assert mock_openai_client.chat.completions.create.call_count == 2
    assert (
        mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
        == snapshot
    )
