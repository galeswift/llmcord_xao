"""Pure rules engine for the server RPG. RPG-DESIGN.md is canon — if this code and that file
disagree, the file wins.

No Discord, no LLM, no I/O: deterministic state-in/state-out so fights can run identically in the
bot (rpg.py), the local simulator (rpg_sim.py), and the tests (test_rpg.py).
"""

import random
from datetime import datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Los_Angeles")
ROUND_TIMES = ((12, 0), (20, 0))

MONSTER_NAME = "Wild Slime"
MONSTER_HP = 20
XP_REWARD = 5

CLASSES = {
    "tank": dict(name="Tank", hp=10, attack=2, attack_label="Attack"),
    "ranged": dict(name="Ranged DPS", hp=6, attack=3, attack_label="Attack"),
    "melee": dict(name="Melee DPS", hp=8, attack=4, attack_label="Attack"),
    "healer": dict(name="Healer", hp=4, attack=2, attack_label="Magic Attack"),
}

CLASS_VERB_HELP = {
    "tank": "`attack` (2 dmg) · `taunt` (slime must attack you for 2 rounds) · `heal` (self +2)",
    "ranged": "`attack` (3 dmg) · `dot` (1 dmg/round, 4 rounds) · `buff` (party +1 HP & +1 dmg, 3 rounds)",
    "melee": "`attack` (4 dmg) · `charged` (6 dmg, lands next round) · `stun` (slime skips 2 rounds, interrupts special; cooldown 2 rounds after)",
    "healer": "`attack` (2 dmg magic) · `regen @who` (+2/round, 4 rounds) · `heal party` (+3 all) · `heal @who` (+7) · `revive @who` (lands next round, full HP)",
}

CLASS_PICK_HELP = "pick your class: `class tank` (10 HP) · `class ranged` (6 HP) · `class melee` (8 HP) · `class healer` (4 HP)"

ACTION_WORDS = ("join", "attack", "taunt", "heal", "dot", "buff", "charged", "charge", "stun", "magic", "regen", "revive", "status")

# Narrator flavor. Prose only — every number in an event comes from the engine, never from here.
FLAVOR = {
    "attack_tank": (
        "{name} slams shield-first into the slime with a wet SPLAT!",
        "{name} delivers a crushing overhead swing!",
    ),
    "attack_ranged": (
        "{name} puts an arrow clean through the slime's wobbling bulk!",
        "{name} looses a shot that thuds home dead-center!",
    ),
    "attack_melee": (
        "{name} carves a gleaming arc through the ooze!",
        "{name} steps in low and slashes deep!",
    ),
    "attack_healer": (
        "{name} hurls a crackling bolt of holy light!",
        "{name} mutters a quick prayer and zaps the slime!",
    ),
    "charged_start": (
        "{name} plants their feet and draws power into their blade...",
        "{name} winds up something big...",
    ),
    "charged_release": (
        "{name}'s charged blow lands like a falling anvil — CRUNCH!",
        "{name} releases the pent-up strike in one devastating arc!",
    ),
    "taunt": (
        "{name} bangs their shield and bellows a challenge — the slime's beady eyes lock on!",
        "{name} shouts an insult so cutting the slime forgets anyone else exists!",
    ),
    "heal_self": (
        "{name} grits their teeth and binds their wounds mid-swing.",
        "{name} takes a long swig from a battered flask.",
    ),
    "dot": (
        "{name} coats an arrow in something evil-smelling and lets it fly.",
        "{name} tags the slime with a smoldering toxin.",
    ),
    "buff": (
        "{name} strikes up a battle hymn — the whole party stands taller!",
        "{name} passes out suspicious energy drinks — everyone feels invincible!",
    ),
    "stun": (
        "{name} cracks the slime dead-center — it wobbles, cross-eyed and frozen!",
        "{name} lands a ringing blow that leaves the slime seeing stars!",
    ),
    "special_cancelled": (
        "The slime's swelling bulk deflates with a sad squelch.",
        "The building power fizzles out of the slime harmlessly.",
    ),
    "regen": (
        "{name} weaves a soft green glow around {target}.",
        "{name} blesses {target} with slow-knitting light.",
    ),
    "healparty": (
        "{name} throws their arms wide and warmth washes over the whole party.",
        "{name} calls down a gentle rain that mends every wound it touches.",
    ),
    "healsingle": (
        "{name} channels a surge of light into {target}.",
        "{name} slaps a glowing hand onto {target}'s wounds.",
    ),
    "revive_start": (
        "{name} kneels beside {target}, hands glowing, chanting fast...",
        "{name} starts dragging {target}'s soul back through sheer stubbornness...",
    ),
    "revive_done": (
        "✨ {target} gasps back to life, fully mended!",
        "✨ {target} sits bolt upright — good as new!",
    ),
    "revive_fizzle": (
        "{name}'s revive fizzles — {target} is already up.",
    ),
    "m_attack": (
        "The slime rears up and crashes down on {target}!",
        "The slime lashes out with a whip of ooze, catching {target} square!",
    ),
    "m_aoe": (
        "The slime bursts like a water balloon, splattering everyone with stinging goo!",
        "The slime spins wildly, spraying acid across the whole party!",
    ),
    "m_charge": (
        "⚠️ The slime swells to twice its size, quivering with building power...",
        "⚠️ The slime glows an angry red and starts vibrating ominously...",
    ),
    "m_special": (
        "💥 The slime ERUPTS, drowning the party in a tidal wave of ooze!",
        "💥 The slime detonates outward in a concussive shockwave of goo!",
    ),
    "m_stunned": (
        "The slime wobbles in place, utterly dazed.",
        "The slime blinks its beady eyes, too scrambled to act.",
    ),
    "dot_tick": (
        "The toxin sizzles deeper into the slime.",
        "The slime hisses as the poison eats at it.",
    ),
    "regen_tick": (
        "{name}'s wounds knit closed a little more.",
        "The green glow does its quiet work on {name}.",
    ),
    "death": (
        "💀 {name} collapses in a heap!",
        "💀 {name} goes down hard and doesn't get up!",
    ),
    "victory": (
        "🎉 The slime shudders, deflates, and splatters apart. **VICTORY!**",
        "🎉 With one final pathetic gurgle, the slime melts into harmless puddles. **VICTORY!**",
    ),
    "defeat": (
        "☠️ The last fighter falls. The slime jiggles triumphantly over the fallen party.",
    ),
}

_flavor_rng = random.Random()


def _ev(key: str, mech: str = "", **kw) -> str:
    """One narrated event line: flavor prose plus the mechanical truth in italics."""
    line = _flavor_rng.choice(FLAVOR[key]).format(**kw)
    return f"{line} *({mech})*" if mech else line


def new_fight(deadline_ts: int) -> dict:
    return {
        "round": 1,
        "active": True,
        "result": None,
        "deadline": deadline_ts,
        "seq": 0,
        "prompted": [],
        "monster": {"hp": MONSTER_HP, "max_hp": MONSTER_HP, "stun": 0, "dot": 0, "taunt": 0, "taunt_uid": None, "charging": False, "special_cd": 0},
        "players": {},
        "join_order": [],
    }


def add_player(state: dict, uid: int, name: str, cls: str) -> dict:
    player = dict(
        name=name, cls=cls, hp=CLASSES[cls]["hp"], max_hp=CLASSES[cls]["hp"], alive=True,
        regen=0, buff=0, stun_cd=0, charging=None, action=None, seq=None,
    )
    state["players"][str(uid)] = player
    state["join_order"].append(str(uid))
    return player


def alive_players(state: dict) -> list[str]:
    return [uid for uid in state["join_order"] if state["players"][uid]["alive"]]


def parse_verb(tokens: list[str], cls: str, target_uid: Optional[int]) -> Optional[tuple[str, Optional[int]]]:
    """Map typed keywords to (verb, target). Returns None if the message isn't an action."""
    word = tokens[0]
    if word == "attack":
        return "attack", None
    if cls == "tank":
        if word == "taunt":
            return "taunt", None
        if word == "heal":
            return "heal", None
    elif cls == "ranged":
        if word == "dot":
            return "dot", None
        if word == "buff":
            return "buff", None
    elif cls == "melee":
        if word in ("charged", "charge"):
            return "charged", None
        if word == "stun":
            return "stun", None
    elif cls == "healer":
        if word == "magic":
            return "attack", None
        if word == "regen":
            return "regen", target_uid
        if word == "heal":
            if len(tokens) > 1 and tokens[1] == "party":
                return "healparty", None
            return "healsingle", target_uid
        if word == "revive":
            return "revive", target_uid
    return None


def submit_action(state: dict, uid: int, verb: str, target_uid: Optional[int] = None) -> tuple[bool, str]:
    player = state["players"][str(uid)]
    monster = state["monster"]

    if not player["alive"]:
        return False, "you're down. Only a healer's `revive` can bring you back."
    if player["charging"]:
        return False, "you're locked in — your charged action lands when this round resolves."
    if verb == "stun":
        if monster["stun"] > 0:
            return False, "the slime is already stunned."
        if player["stun_cd"] > 0:
            return False, f"stun is on cooldown for {player['stun_cd']} more round(s)."
    if verb == "revive":
        target = state["players"].get(str(target_uid))
        if not target or target["alive"]:
            return False, "revive needs a downed party member as target, e.g. `revive @name`."
    if verb in ("regen", "healsingle"):
        target = state["players"].get(str(target_uid))
        if not target or not target["alive"]:
            return False, "that needs a living party member as target, e.g. `heal @name`."

    state["seq"] += 1
    player["action"] = {"verb": verb, "target": str(target_uid) if target_uid else None}
    player["seq"] = state["seq"]
    return True, ""


def resolve_round(state: dict, rng: random.Random) -> list[str]:
    monster = state["monster"]
    monster.setdefault("special_cd", 0)  # fights saved before the cooldown rule existed
    players = state["players"]
    events: list[str] = []

    def hit_monster(player: dict, base: int, label: str, flavor_key: str) -> None:
        amount = base + (1 if player["buff"] > 0 else 0)
        monster["hp"] = max(0, monster["hp"] - amount)
        events.append(_ev(flavor_key, f"{label}: {amount} dmg — slime {monster['hp']}/{monster['max_hp']}", name=player["name"]))

    def check_death(player: dict) -> None:
        if player["hp"] <= 0 and player["alive"]:
            player["hp"] = 0
            player["alive"] = False
            if player["buff"] > 0:
                player["buff"] = 0
                player["max_hp"] -= 1
            player["regen"] = 0
            player["charging"] = None
            player["action"] = None
            events.append(_ev("death", name=player["name"]))

    # 1. Charge releases from last round land first, in join order
    released = set()
    for uid in state["join_order"]:
        player = players[uid]
        if not (player["alive"] and player["charging"]) or monster["hp"] <= 0:
            continue
        charge, player["charging"] = player["charging"], None
        released.add(uid)
        if charge["verb"] == "charged":
            hit_monster(player, 6, "Charged Attack", "charged_release")
        elif charge["verb"] == "revive":
            target = players.get(charge["target"] or "")
            if target and not target["alive"]:
                target["alive"] = True
                target["hp"] = target["max_hp"]
                events.append(_ev("revive_done", f"{target['name']} revived at {target['hp']}/{target['max_hp']} HP", target=target["name"]))
            else:
                events.append(_ev("revive_fizzle", name=player["name"], target=(target or {}).get("name", "their target")))

    # 2. Submitted actions in submission order, then non-submitters (default attack)
    actors = [uid for uid in state["join_order"] if players[uid]["alive"] and uid not in released]
    submitted = sorted((u for u in actors if players[u]["action"]), key=lambda u: players[u]["seq"])
    defaulters = [u for u in actors if not players[u]["action"]]

    for uid in submitted + defaulters:
        if monster["hp"] <= 0:
            break
        player = players[uid]
        action = player["action"] or {"verb": "attack", "target": None}
        verb = action["verb"]

        if verb == "attack":
            hit_monster(player, CLASSES[player["cls"]]["attack"], CLASSES[player["cls"]]["attack_label"], f"attack_{player['cls']}")
        elif verb == "taunt":
            monster["taunt"] = 2
            monster["taunt_uid"] = uid
            events.append(_ev("taunt", f"taunt: slime must attack {player['name']} for 2 rounds", name=player["name"]))
        elif verb == "heal":
            healed = min(2, player["max_hp"] - player["hp"])
            player["hp"] += healed
            events.append(_ev("heal_self", f"+{healed} HP — {player['hp']}/{player['max_hp']}", name=player["name"]))
        elif verb == "dot":
            monster["dot"] = 4
            events.append(_ev("dot", "DOT: 1 dmg/round for 4 rounds", name=player["name"]))
        elif verb == "buff":
            for buff_uid in alive_players(state):
                buffed = players[buff_uid]
                if buffed["buff"] == 0:
                    buffed["max_hp"] += 1
                    buffed["hp"] += 1
                buffed["buff"] = 3
            events.append(_ev("buff", "party +1 HP & +1 dmg for 3 rounds", name=player["name"]))
        elif verb == "stun":
            interrupted = monster["charging"]
            monster["stun"] = 2
            monster["charging"] = False
            player["stun_cd"] = 4  # 2 stunned rounds + 2 rounds cooldown after the stun ends
            events.append(_ev("stun", "slime stunned for 2 rounds", name=player["name"]))
            if interrupted:
                monster["special_cd"] = 4  # cancelled special goes on its 3-round cooldown like usual
                events.append(_ev("special_cancelled", "special cancelled — on cooldown 3 rounds"))
        elif verb == "regen":
            target = players[action["target"]]
            target["regen"] = 4
            events.append(_ev("regen", "regen: +2 HP/round for 4 rounds", name=player["name"], target=target["name"]))
        elif verb == "healparty":
            for heal_uid in alive_players(state):
                healed = players[heal_uid]
                healed["hp"] = min(healed["max_hp"], healed["hp"] + 3)
            events.append(_ev("healparty", "party healed +3", name=player["name"]))
        elif verb == "healsingle":
            target = players[action["target"]]
            healed = min(7, target["max_hp"] - target["hp"])
            target["hp"] += healed
            events.append(_ev("healsingle", f"+{healed} HP — {target['name']} {target['hp']}/{target['max_hp']}", name=player["name"], target=target["name"]))
        elif verb == "charged":
            player["charging"] = {"verb": "charged", "target": None}
            events.append(_ev("charged_start", "charged attack: 6 dmg, lands next round", name=player["name"]))
        elif verb == "revive":
            player["charging"] = {"verb": "revive", "target": action["target"]}
            events.append(_ev("revive_start", "revive: lands next round", name=player["name"], target=players[action["target"]]["name"]))

    # 3. Monster phase
    if monster["hp"] > 0:
        if monster["stun"] > 0:
            events.append(_ev("m_stunned", "stunned — slime skips its turn"))
        elif monster["charging"]:
            monster["charging"] = False
            monster["special_cd"] = 4  # 3-round cooldown after firing (this round's tick counts down first)
            events.append(_ev("m_special", "special attack: 3 dmg to everyone"))
            for uid in alive_players(state):
                players[uid]["hp"] -= 3
                check_death(players[uid])
        else:
            options = ["attack", "aoe"] + (["special"] if monster["special_cd"] == 0 else [])
            choice = rng.choice(options)
            if choice == "attack":
                alive = alive_players(state)
                if monster["taunt"] > 0 and monster["taunt_uid"] in players and players[monster["taunt_uid"]]["alive"]:
                    target_uid = monster["taunt_uid"]
                else:
                    target_uid = min(alive, key=lambda u: (players[u]["hp"], players[u]["max_hp"], state["join_order"].index(u)))
                target = players[target_uid]
                target["hp"] -= 3
                events.append(_ev("m_attack", f"3 dmg — {target['name']} {max(target['hp'], 0)}/{target['max_hp']}", target=target["name"]))
                check_death(target)
            elif choice == "aoe":
                events.append(_ev("m_aoe", "1 dmg to everyone"))
                for uid in alive_players(state):
                    players[uid]["hp"] -= 1
                    check_death(players[uid])
            else:
                monster["charging"] = True
                events.append(_ev("m_charge", "special next round: 3 dmg to all — stun it to cancel!"))

    # 4. End-of-round ticks, then status durations count down
    if monster["hp"] > 0 and monster["dot"] > 0:
        monster["hp"] = max(0, monster["hp"] - 1)
        events.append(_ev("dot_tick", f"DOT: 1 dmg — slime {monster['hp']}/{monster['max_hp']}"))
    for uid in alive_players(state):
        player = players[uid]
        if player["regen"] > 0 and player["hp"] < player["max_hp"]:
            healed = min(2, player["max_hp"] - player["hp"])
            player["hp"] += healed
            events.append(_ev("regen_tick", f"regen: +{healed} — {player['hp']}/{player['max_hp']}", name=player["name"]))

    monster["dot"] = max(0, monster["dot"] - 1)
    monster["taunt"] = max(0, monster["taunt"] - 1)
    monster["stun"] = max(0, monster["stun"] - 1)
    monster["special_cd"] = max(0, monster["special_cd"] - 1)
    for player in players.values():
        player["stun_cd"] = max(0, player["stun_cd"] - 1)
        player["regen"] = max(0, player["regen"] - 1)
        if player["buff"] > 0:
            player["buff"] -= 1
            if player["buff"] == 0:
                player["max_hp"] -= 1
                player["hp"] = min(player["hp"], player["max_hp"])

    # 5. Outcome
    if monster["hp"] <= 0:
        state["result"] = "victory"
        state["active"] = False
        events.append(_ev("victory"))
    elif not alive_players(state):
        state["result"] = "defeat"
        state["active"] = False
        events.append(_ev("defeat", "no reward"))
    else:
        state["round"] += 1
        for player in players.values():
            player["action"] = None
            player["seq"] = None
    return events


def next_deadline(now: Optional[datetime] = None) -> int:
    now = now or datetime.now(TZ)
    candidates = []
    for day_offset in (0, 1):
        day = (now + timedelta(days=day_offset)).date()
        for hour, minute in ROUND_TIMES:
            slot = datetime.combine(day, time(hour, minute), tzinfo=TZ)
            if slot > now:
                candidates.append(slot)
    return int(min(candidates).timestamp())


def format_status(state: dict, show_deadline: bool = True) -> str:
    monster = state["monster"]
    monster_tags = []
    if monster["charging"]:
        monster_tags.append("⚠️ CHARGING ITS SPECIAL")
    if monster["stun"]:
        monster_tags.append(f"stunned ({monster['stun']} round(s))")
    if monster["dot"]:
        monster_tags.append(f"DOT ({monster['dot']} round(s))")
    if monster.get("special_cd", 0):
        monster_tags.append(f"special on cooldown ({monster['special_cd']} round(s))")
    lines = [f"**{MONSTER_NAME}** — {monster['hp']}/{monster['max_hp']} HP" + (f" — {', '.join(monster_tags)}" if monster_tags else "")]

    for uid in state["join_order"]:
        player = state["players"][uid]
        tags = []
        if not player["alive"]:
            tags.append("💀 down")
        if player["charging"]:
            tags.append("charging")
        if player["regen"]:
            tags.append(f"regen {player['regen']}")
        if player["buff"]:
            tags.append(f"buff {player['buff']}")
        if player["stun_cd"]:
            tags.append(f"stun cd {player['stun_cd']}")
        if player["action"]:
            tags.append("action in")
        lines.append(f"· {player['name']} ({CLASSES[player['cls']]['name']}) {player['hp']}/{player['max_hp']} HP" + (f" — {', '.join(tags)}" if tags else ""))

    if state["active"] and show_deadline:
        lines.append(f"\nRound {state['round']} resolves <t:{state['deadline']}:R> (<t:{state['deadline']}:t>). Type your action to lock it in — no action = basic attack.")
    return "\n".join(lines)
