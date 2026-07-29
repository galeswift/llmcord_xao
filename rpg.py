"""Discord glue for the server RPG. All rules live in rpg_engine.py (RPG-DESIGN.md is canon).

The LLM only narrates already-resolved rounds. Fight state is persisted in SQLite so fights
survive restarts. Test locally without Discord via rpg_sim.py.
"""

import asyncio
import json
import logging
import random
from datetime import time
from pathlib import Path
from typing import Any, Callable, Optional

import discord
from discord.ext import tasks

from rpg_engine import (
    CLASS_PICK_HELP,
    CLASS_VERB_HELP,
    CLASSES,
    ACTION_WORDS,
    MONSTER_HP,
    MONSTER_NAME,
    ROUND_TIMES,
    TZ,
    XP_REWARD,
    add_player,
    format_status,
    level_for_xp,
    new_fight,
    next_deadline,
    parse_verb,
    resolve_round,
    spawn_lines,
    submit_action,
)

# Pre-generated enemy art (see generate_enemy_images.py) attached to encounter posts
ENEMY_IMAGE_DIR = Path(__file__).parent / "assets" / "enemies"

_bot = None
_db = None
_get_config: Optional[Callable[[], dict]] = None
_narrate: Optional[Callable[[str], Any]] = None
_lock = asyncio.Lock()


def _save_fight(thread_id: int, state: dict) -> None:
    _db.execute(
        "INSERT OR REPLACE INTO rpg_fights (thread_id, state, active) VALUES (?, ?, ?)",
        (thread_id, json.dumps(state), int(state["active"])),
    )
    _db.commit()


def _load_fight(thread_id: int) -> Optional[dict]:
    row = _db.execute("SELECT state FROM rpg_fights WHERE thread_id = ? AND active = 1", (thread_id,)).fetchone()
    return json.loads(row[0]) if row else None


def _stored_class(user_id: int) -> Optional[str]:
    row = _db.execute("SELECT cls FROM rpg_players WHERE user_id = ?", (user_id,)).fetchone()
    return row[0] if row else None


def _stored_level(user_id: int) -> int:
    row = _db.execute("SELECT xp FROM rpg_players WHERE user_id = ?", (user_id,)).fetchone()
    return level_for_xp(row[0] if row else 0)


def _join_fight(state: dict, author: discord.abc.User, cls: str) -> tuple[dict, str]:
    """Add a member at their banked-XP level; reinforcements may spawn (design doc 2c)."""
    before = len(state["monsters"])
    level = _stored_level(author.id)
    player = add_player(state, author.id, author.display_name, cls, level)
    text = f"⚔️ {player['name']} joins as **{CLASSES[cls]['name']}** (Lv {level}, {player['hp']} HP)!"
    for line in spawn_lines(state["monsters"][before:]):
        text += f"\n{line}"
    return player, text


async def handle_message(msg: discord.Message) -> bool:
    """Route a message in an active fight thread. Returns True if consumed by the RPG."""
    if _db is None:
        return False

    async with _lock:
        state = _load_fight(msg.channel.id)
        if state is None:
            return False

        uid = str(msg.author.id)
        tokens = msg.content.strip().lower().split()
        if not tokens:
            return False
        target_uid = next((u.id for u in msg.mentions if not u.bot), None)
        player = state["players"].get(uid)

        # Class selection / joining
        if tokens[0] == "class":
            if player:
                await msg.reply(f"you're locked in as {CLASSES[player['cls']]['name']} for this fight.", silent=True)
                return True
            cls = tokens[1] if len(tokens) > 1 else ""
            if cls not in CLASSES:
                await msg.reply(CLASS_PICK_HELP, silent=True)
                return True
            _db.execute(
                "INSERT INTO rpg_players (user_id, cls, xp) VALUES (?, ?, 0) "
                "ON CONFLICT(user_id) DO UPDATE SET cls = excluded.cls",
                (msg.author.id, cls),
            )
            _db.commit()
            player, join_text = _join_fight(state, msg.author, cls)
            _save_fight(msg.channel.id, state)
            await msg.reply(f"{join_text}\nYour verbs: {CLASS_VERB_HELP[cls]}", silent=True)
            return True

        if player is None:
            if tokens[0] not in ACTION_WORDS:
                return False  # spectator chatter
            known_cls = _stored_class(msg.author.id)
            if known_cls is None:
                await msg.reply(CLASS_PICK_HELP, silent=True)
                return True
            player, join_text = _join_fight(state, msg.author, known_cls)
            _save_fight(msg.channel.id, state)
            await msg.channel.send(join_text)
            if tokens[0] == "join":
                return True

        if tokens[0] == "join":
            await msg.reply("you're already in the fight.", silent=True)
            return True
        if tokens[0] == "status":
            await msg.channel.send(format_status(state))
            return True

        parsed = parse_verb(tokens, player["cls"], target_uid)
        if parsed is None:
            return False  # chatter from a combatant

        ok, err = submit_action(state, msg.author.id, parsed[0], parsed[1], parsed[2])
        if ok:
            _save_fight(msg.channel.id, state)
            try:
                await msg.add_reaction("✅")
            except discord.HTTPException:
                pass
        else:
            await msg.reply(err, silent=True)
        return True


async def _resolve_fight(thread_id: int, state: dict) -> None:
    thread = _bot.get_channel(thread_id)
    if thread is None:
        try:
            thread = await _bot.fetch_channel(thread_id)
        except discord.HTTPException:
            state["active"] = False
            _save_fight(thread_id, state)
            return

    if not state["players"]:
        state["deadline"] = next_deadline()
        _save_fight(thread_id, state)
        await thread.send(f"Nobody has joined the fight yet — the {MONSTER_NAME} waits. Next chance <t:{state['deadline']}:R>.")
        return

    round_number = state["round"]
    events = resolve_round(state, random.Random(f"{thread_id}:{round_number}"))
    if state["active"]:
        state["deadline"] = next_deadline()
    _save_fight(thread_id, state)

    narration = ""
    if _narrate:
        try:
            narration = await _narrate("\n".join(events))
        except Exception:
            logging.exception("RPG narration failed; posting plain results")

    text = f"**⚔️ Round {round_number}**\n"
    if narration:
        text += f"*{narration}*\n\n"
    text += "\n".join(events)

    if state["result"] == "victory":
        for uid in state["join_order"]:
            _db.execute("UPDATE rpg_players SET xp = xp + ? WHERE user_id = ?", (XP_REWARD, int(uid)))
        _db.commit()
        footer = f"🏆 Everyone who fought earns **{XP_REWARD} XP**!"
    elif state["result"] == "defeat":
        footer = "Better luck next time."
    else:
        footer = format_status(state)

    if len(text) + len(footer) + 2 <= 2000:
        await thread.send(text + "\n\n" + footer)
    else:
        await thread.send(text[:2000])
        await thread.send(footer[:2000])


@tasks.loop(time=[time(hour, minute, tzinfo=TZ) for hour, minute in ROUND_TIMES])
async def _round_loop() -> None:
    config = await asyncio.to_thread(_get_config)
    if not config.get("rpg_enabled", False):
        return
    async with _lock:
        rows = _db.execute("SELECT thread_id, state FROM rpg_fights WHERE active = 1").fetchall()
        for thread_id, state_json in rows:
            try:
                await _resolve_fight(thread_id, json.loads(state_json))
            except Exception:
                logging.exception(f"Failed to resolve RPG fight in thread {thread_id}")


def start_scheduler() -> None:
    if not _round_loop.is_running():
        _round_loop.start()


def setup(bot, db, get_config: Callable[[], dict], narrate) -> None:
    global _bot, _db, _get_config, _narrate
    _bot, _db, _get_config, _narrate = bot, db, get_config, narrate

    db.execute("CREATE TABLE IF NOT EXISTS rpg_players (user_id INTEGER PRIMARY KEY, cls TEXT, xp INTEGER DEFAULT 0)")
    db.execute("CREATE TABLE IF NOT EXISTS rpg_fights (thread_id INTEGER PRIMARY KEY, state TEXT, active INTEGER)")
    db.commit()

    @bot.tree.command(name="rpg", description="Start the RPG encounter (admin only)")
    async def rpg_command(interaction: discord.Interaction) -> None:
        config = await asyncio.to_thread(get_config)
        if not config.get("rpg_enabled", False):
            await interaction.response.send_message("The RPG is disabled (`rpg_enabled` in config).", ephemeral=True)
            return
        if interaction.user.id not in config["permissions"]["users"]["admin_ids"]:
            await interaction.response.send_message("You don't have permission to start encounters.", ephemeral=True)
            return
        await interaction.response.send_message("Encounter started!", ephemeral=True)

        image_path = ENEMY_IMAGE_DIR / "slime.png"
        send_kwargs = dict(file=discord.File(image_path)) if image_path.exists() else {}
        encounter_msg = await interaction.channel.send(
            f"**You encounter a wild slime!** Join the thread to fight it — every fighter who joins draws in more slimes. ⚔️",
            **send_kwargs,
        )
        thread = await encounter_msg.create_thread(name=f"{MONSTER_NAME} pack fight")
        state = new_fight(next_deadline())
        async with _lock:
            _save_fight(thread.id, state)
        await thread.send(
            f"**A wild slime appears!** ({MONSTER_HP} HP)\n"
            f"More slimes join as fighters do — roughly one per member, extra for higher levels.\n"
            f"Each slime attacks the lowest-HP fighter unless taunted — protect your squishies.\n\n"
            f"To fight: {CLASS_PICK_HELP}\n"
            f"Then type an action each round (change it any time until the round resolves; no action = basic attack).\n"
            f"Single-target actions take a slime number, e.g. `attack 2`, `dot 1`, `stun 3` — no number = first living slime.\n"
            f"Rounds resolve twice a day. Round 1 resolves <t:{state['deadline']}:R> (<t:{state['deadline']}:t>).\n"
            f"`status` shows the fight state any time."
        )

    async def on_thread_member_join(member: discord.ThreadMember) -> None:
        config = await asyncio.to_thread(get_config)
        if not config.get("rpg_enabled", False):
            return
        async with _lock:
            state = _load_fight(member.thread_id)
            if state is None or str(member.id) in state["players"] or member.id in state["prompted"]:
                return
            user = _bot.get_user(member.id)
            if user is None or user.bot:
                return
            state["prompted"].append(member.id)
            _save_fight(member.thread_id, state)
        thread = _bot.get_channel(member.thread_id)
        if thread:
            known_cls = _stored_class(member.id)
            if known_cls:
                await thread.send(f"<@{member.id}> — welcome back, {CLASSES[known_cls]['name']}. Type `join` or any action to enter the fight.")
            else:
                await thread.send(f"<@{member.id}> — {CLASS_PICK_HELP}")

    bot.add_listener(on_thread_member_join, "on_thread_member_join")
