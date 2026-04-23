import asyncio
import base64
import io
import json
import sqlite3
import struct
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI, RateLimitError
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")


def detect_image_type(data: bytes) -> str | None:
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if data[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    if data[:6] in (b'GIF87a', b'GIF89a'):
        return "image/gif"
    return None


def image_too_large(data: bytes, actual_type: str, max_bytes: int = 2 * 1024 * 1024, max_dimension: int = 1280) -> bool:
    if len(data) > max_bytes:
        return True
    if actual_type == "image/png" and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        if w > max_dimension or h > max_dimension:
            return True
    return False

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    import os
    if config_yaml := os.environ.get("CONFIG_YAML"):
        return yaml.safe_load(config_yaml)
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()
curr_model = next(iter(config["models"]))

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()

summaries_db = sqlite3.connect("summaries.db", check_same_thread=False)
summaries_db.execute("""
    CREATE TABLE IF NOT EXISTS summaries (
        channel_id INTEGER PRIMARY KEY,
        cut_msg_id INTEGER,
        summary    TEXT
    )
""")
summaries_db.commit()

DISCORD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_poll",
            "description": "Create a Discord poll in the current channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The poll question"},
                    "answers": {"type": "array", "items": {"type": "string"}, "description": "Answer options (2-10)"},
                    "duration_hours": {"type": "integer", "description": "Poll duration in hours (1-168, default 24)"},
                    "allow_multiselect": {"type": "boolean", "description": "Allow selecting multiple answers"},
                },
                "required": ["question", "answers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a scheduled event in the Discord server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Event name"},
                    "description": {"type": "string", "description": "Event description"},
                    "start_time": {"type": "string", "description": "Start time in ISO 8601 format (e.g. 2025-06-01T19:00:00)"},
                    "end_time": {"type": "string", "description": "End time in ISO 8601 format (optional, defaults to 1 hour after start)"},
                    "location": {"type": "string", "description": "Event location"},
                },
                "required": ["name", "start_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pin_message",
            "description": "Pin a message in the current channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content_snippet": {"type": "string", "description": "Part of the message content or poll question to identify which message to pin"},
                },
                "required": ["content_snippet"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_poll",
            "description": "End an active Discord poll in the current channel early.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Part of the poll question to identify it"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_event",
            "description": "Cancel and delete a scheduled Discord event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Part of the event name to identify it"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current, real-time information. Use this when the user asks about recent events, live data, current prices, or anything that may have changed since your training cutoff.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text description and post it in the channel. Use this when the user asks for an image, picture, illustration, or drawing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Detailed description of the image to generate"},
                },
                "required": ["prompt"],
            },
        },
    },
]


async def execute_tool(name: str, args: dict, msg: discord.Message) -> str:
    try:
        if name == "create_poll":
            question = args.get("question", "")
            answers = args.get("answers", [])[:10]
            duration_hours = min(max(int(args.get("duration_hours", 24)), 1), 168)
            allow_multiselect = bool(args.get("allow_multiselect", False))
            if not question or len(answers) < 2:
                return "Failed: poll requires a question and at least 2 answers."
            poll = discord.Poll(question=question, duration=timedelta(hours=duration_hours), multiple=allow_multiselect)
            for answer in answers:
                poll.add_answer(text=answer)
            poll_msg = await msg.channel.send(poll=poll)
            return f"Poll created (message ID: {poll_msg.id})."

        if name == "create_event":
            if not msg.guild:
                return "Cannot create events in DMs."
            try:
                start_time = datetime.fromisoformat(args["start_time"]).astimezone()
            except (KeyError, ValueError):
                return "Invalid start_time. Use ISO 8601 format (e.g. 2025-06-01T19:00:00)."
            try:
                end_time = datetime.fromisoformat(args["end_time"]).astimezone()
            except (KeyError, ValueError):
                end_time = start_time + timedelta(hours=1)
            await msg.guild.create_scheduled_event(
                name=args.get("name", "Event"),
                description=args.get("description", ""),
                start_time=start_time,
                end_time=end_time,
                entity_type=discord.EntityType.external,
                privacy_level=discord.PrivacyLevel.guild_only,
                location=args.get("location", "TBD"),
            )
            return f"Event '{args.get('name')}' created for {start_time.strftime('%B %d %Y at %H:%M %Z')}."

        if name == "pin_message":
            snippet = args.get("content_snippet", "").lower()
            async for m in msg.channel.history(limit=100):
                content = m.content or next((e.description for e in m.embeds if e.description), "")
                if not content and m.poll:
                    content = m.poll.question
                if snippet in content.lower():
                    await m.pin()
                    return "Message pinned."
            return "No matching message found to pin."

        if name == "cancel_poll":
            question = args.get("question", "").lower()
            async for m in msg.channel.history(limit=100):
                if m.poll and question in m.poll.question.lower():
                    await m.end_poll()
                    return f"Poll '{m.poll.question}' ended."
            return "No matching active poll found."

        if name == "cancel_event":
            if not msg.guild:
                return "Cannot cancel events in DMs."
            event_name = args.get("name", "").lower()
            for event in await msg.guild.fetch_scheduled_events():
                if event_name in event.name.lower():
                    await event.delete()
                    return f"Event '{event.name}' cancelled."
            return "No matching event found."

        if name == "generate_image":
            prompt = args.get("prompt", "")
            if not prompt:
                return "No image prompt provided."
            provider_config = config["providers"].get("openai", {})
            api_key = provider_config.get("api_key")
            if not api_key:
                return "No OpenAI API key configured for image generation."
            image_client = AsyncOpenAI(
                base_url=provider_config.get("base_url", "https://api.openai.com/v1"),
                api_key=api_key,
            )
            result = await image_client.images.generate(
                model=config.get("image_model", "gpt-image-2"),
                prompt=prompt,
                size=config.get("image_size", "1024x1024"),
                quality=config.get("image_quality", "medium"),
                background=config.get("image_background", "auto"),
            )
            image_bytes = base64.b64decode(result.data[0].b64_json)
            await msg.channel.send(file=discord.File(io.BytesIO(image_bytes), filename="generated.png"))
            return "Image generated and sent."

        if name == "web_search":
            query = args.get("query", "")
            if not query:
                return "No search query provided."
            search_model_key = config.get("search_model", "openai/gpt-4o-mini-search-preview")
            search_provider, search_model_name = search_model_key.split("/", 1)
            search_provider_config = config["providers"].get(search_provider, {})
            search_client = AsyncOpenAI(
                base_url=search_provider_config.get("base_url", "https://api.openai.com/v1"),
                api_key=search_provider_config.get("api_key", "sk-no-key-required"),
            )
            try:
                response = await search_client.chat.completions.create(
                    model=search_model_name,
                    messages=[{"role": "user", "content": query}],
                    stream=False,
                    extra_body={"web_search_options": {}},
                )
                content = response.choices[0].message.content or "No results found."
                return f"[Live web search results for: {query}]\n\n{content}\n\n[End of web search results. Present these findings directly to the user without disclaimers about browsing ability.]"
            except RateLimitError:
                return "Web search rate limit reached. Answer from your training data instead."

        return f"Unknown tool: {name}"

    except discord.Forbidden:
        return f"Permission denied executing {name}. Check bot permissions."
    except Exception as e:
        logging.exception(f"Tool execution failed: {name}")
        return f"{name} failed: {e}"


async def get_or_generate_summary(channel_id: int, cut_msg: discord.Message, openai_client: AsyncOpenAI, model: str) -> str:
    row = summaries_db.execute(
        "SELECT cut_msg_id, summary FROM summaries WHERE channel_id = ?", (channel_id,)
    ).fetchone()

    if row and row[0] == cut_msg.id:
        return row[1]

    # Collect cut-off messages from the msg_nodes cache (oldest first)
    lines = []
    msg = cut_msg
    while msg is not None and len(lines) < 50:
        node = msg_nodes.get(msg.id)
        if not node:
            break
        if node.text:
            lines.append(node.text)
        msg = node.parent_msg
    lines.reverse()

    if not lines:
        return ""

    prior = f"Prior summary to build on:\n{row[1]}\n\n" if row else ""
    prompt = (
        f"{prior}Summarize the following Discord conversation concisely, "
        f"preserving key facts, decisions, and context.\n\n"
        + "\n".join(lines)
    )

    response = await openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    summary = response.choices[0].message.content.strip()

    summaries_db.execute(
        "INSERT OR REPLACE INTO summaries (channel_id, cut_msg_id, summary) VALUES (?, ?, ?)",
        (channel_id, cut_msg.id, summary),
    )
    summaries_db.commit()
    return summary


@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"

    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if user_is_admin := interaction.user.id in config["permissions"]["users"]["admin_ids"]:
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config

    if curr_str == "":
        config = await asyncio.to_thread(get_config)

    choices = [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != curr_model and curr_str.lower() in model.lower()]

    return choices[:25]

@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot\n")

    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    config = await asyncio.to_thread(get_config)

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    allow_dms = config.get("allow_dms", True)

    permissions = config["permissions"]

    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    provider_slash_model = curr_model
    provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)

    provider_config = config["providers"][provider]

    base_url = provider_config["base_url"]
    api_key = provider_config.get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    chain_msg_ids = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())
        chain_msg_ids.add(curr_msg.id)

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [component.content for component in curr_msg.components if component.type == discord.ComponentType.text_display]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                reply_images = []
                for att, resp in zip(good_attachments, attachment_responses):
                    if not att.content_type.startswith("image"):
                        continue
                    actual_type = detect_image_type(resp.content)
                    if not actual_type or image_too_large(resp.content, actual_type):
                        continue
                    reply_images.append(dict(type="image_url", image_url=dict(url=f"data:{actual_type};base64,{b64encode(resp.content).decode('utf-8')}")))
                curr_node.images = reply_images

                if curr_node.role == "user" and (curr_node.text or curr_node.images):
                    curr_node.text = f"<@{curr_msg.author.id}>: {curr_node.text}"

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = [dict(type="text", text=curr_node.text[:max_text])] + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                messages.append(dict(content=content, role=curr_node.role))

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if (channel_context_count := config.get("channel_context_messages", 0)) > 0 and not is_dm:
        context_images_used = 0
        async for msg in new_msg.channel.history(before=new_msg, limit=channel_context_count):
            if msg.id in chain_msg_ids or (msg.author.bot and msg.author != discord_bot.user):
                continue

            role = "assistant" if msg.author == discord_bot.user else "user"

            text = msg.content.removeprefix(discord_bot.user.mention).strip()
            if not text:
                text = next((e.description for e in msg.embeds if e.description), "")
            if not text:
                text = next((c.content for c in msg.components if c.type == discord.ComponentType.text_display), "")

            good_attachments = [att for att in msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
            att_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

            for att, resp in zip(good_attachments, att_responses):
                if att.content_type.startswith("text"):
                    text += ("\n" if text else "") + resp.text

            images = []
            if accept_images:
                for att, resp in zip(good_attachments, att_responses):
                    if context_images_used >= max_images:
                        break
                    if not att.content_type.startswith("image"):
                        continue
                    actual_type = detect_image_type(resp.content)
                    if not actual_type:
                        continue
                    if image_too_large(resp.content, actual_type):
                        w, h = (struct.unpack(">II", resp.content[16:24]) if actual_type == "image/png" and len(resp.content) >= 24 else (0, 0))
                        logging.info(f"[img] skip {att.filename}: {len(resp.content)//1024}KB {w}x{h}")
                        continue
                    images.append(dict(type="image_url", image_url=dict(url=f"data:{actual_type};base64,{b64encode(resp.content).decode('utf-8')}")))
                    context_images_used += 1

            if role == "user" and (text or images):
                text = f"<@{msg.author.id}>: {text}"

            if not text and not images:
                continue

            content = ([dict(type="text", text=text[:max_text])] + images) if images else text[:max_text]
            messages.append(dict(role=role, content=content))

    if curr_msg is not None and not is_dm:
        try:
            summary = await get_or_generate_summary(new_msg.channel.id, curr_msg, openai_client, model)
            if summary:
                messages.append({"role": "system", "content": f"Summary of earlier conversation:\n{summary}"})
        except Exception:
            logging.exception("Failed to generate conversation summary")

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()

        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()

        messages.append(dict(role="system", content=system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    response_msgs = []
    response_contents = []

    api_messages = messages[::-1]
    guild_only_tools = {"create_event", "cancel_event"}
    available_tools = [t for t in DISCORD_TOOLS if not (is_dm and t["function"]["name"] in guild_only_tools)]

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    tool_call_count = 0
    web_search_count = 0
    max_web_searches = config.get("max_web_searches", 1)

    try:
        async with new_msg.channel.typing():
            while True:
                curr_content = finish_reason = None
                tool_calls_buffer = {}
                got_tool_calls = False

                active_tools = [t for t in available_tools if not (t["function"]["name"] == "web_search" and web_search_count >= max_web_searches)]
                openai_kwargs = dict(model=model, messages=api_messages, stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)
                if active_tools and tool_call_count < 5:
                    openai_kwargs["tools"] = active_tools
                    if extra_body and "reasoning_effort" in extra_body:
                        openai_kwargs["extra_body"] = {k: v for k, v in extra_body.items() if k != "reasoning_effort"} or None

                async for chunk in ([] if got_tool_calls else await openai_client.chat.completions.create(**openai_kwargs)):
                    if finish_reason is not None:
                        break

                    if not (choice := chunk.choices[0] if chunk.choices else None):
                        continue

                    chunk_finish_reason = choice.finish_reason
                    delta = choice.delta

                    if chunk_finish_reason == "tool_calls":
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                entry = tool_calls_buffer.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                                if tc.id: entry["id"] = tc.id
                                if tc.function and tc.function.name: entry["name"] = tc.function.name
                                if tc.function and tc.function.arguments: entry["args"] += tc.function.arguments
                        got_tool_calls = True
                        break

                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            entry = tool_calls_buffer.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                            if tc.id: entry["id"] = tc.id
                            if tc.function and tc.function.name: entry["name"] = tc.function.name
                            if tc.function and tc.function.arguments: entry["args"] += tc.function.arguments
                        continue

                    finish_reason = chunk_finish_reason
                    prev_content = curr_content or ""
                    curr_content = delta.content or ""
                    new_content = prev_content if finish_reason is None else (prev_content + curr_content)

                    if response_contents == [] and new_content == "":
                        continue

                    if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                        response_contents.append("")

                    response_contents[-1] += new_content

                    if not use_plain_responses:
                        time_delta = datetime.now().timestamp() - last_task_time

                        ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                        msg_split_incoming = finish_reason is None and len(response_contents[-1] + curr_content) > max_message_length
                        is_final_edit = finish_reason is not None or msg_split_incoming
                        is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")

                        if start_next_msg or ready_to_edit or is_final_edit:
                            embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                            if start_next_msg:
                                await reply_helper(embed=embed, silent=True)
                            else:
                                await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                                await response_msgs[-1].edit(embed=embed)

                            last_task_time = datetime.now().timestamp()

                if got_tool_calls:
                    tool_call_count += 1
                    sorted_tcs = [tool_calls_buffer[i] for i in sorted(tool_calls_buffer.keys())]
                    assistant_tool_calls = [
                        {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["args"]}}
                        for tc in sorted_tcs
                    ]
                    tool_result_msgs = []
                    for tc in sorted_tcs:
                        try:
                            args = json.loads(tc["args"] or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        logging.info(f"Tool call: {tc['name']}({args})")
                        if tc["name"] == "web_search":
                            web_search_count += 1
                        result = await execute_tool(tc["name"], args, new_msg)
                        tool_result_msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    api_messages = api_messages + [
                        {"role": "assistant", "content": None, "tool_calls": assistant_tool_calls}
                    ] + tool_result_msgs
                else:
                    break

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception:
        logging.exception("Error while generating response")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass
