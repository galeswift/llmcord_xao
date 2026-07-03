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
ENEMY_COUNT = 3
XP_REWARD = 5

CLASSES = {
    "tank": dict(name="Tank", hp=10, attack=2, attack_label="Attack"),
    "ranged": dict(name="Ranged DPS", hp=6, attack=3, attack_label="Attack"),
    "melee": dict(name="Melee DPS", hp=8, attack=4, attack_label="Attack"),
    "healer": dict(name="Healer", hp=4, attack=2, attack_label="Magic Attack"),
}

CLASS_VERB_HELP = {
    "tank": "`attack [#]` (2 dmg) · `taunt` (ALL slimes must attack you for 2 rounds) · `heal` (self +2)",
    "ranged": "`attack [#]` (3 dmg) · `dot [#]` (1 dmg/round, 4 rounds, one slime) · `buff` (party +1 HP & +1 dmg, 3 rounds)",
    "melee": "`attack [#]` (4 dmg) · `charged [#]` (6 dmg, lands next round) · `stun [#]` (slime skips 2 rounds, cancels its special; cooldown 2 rounds after)",
    "healer": "`attack [#]` (2 dmg magic) · `regen @who` (+2/round, 4 rounds) · `heal party` (+3 all) · `heal @who` (+7) · `revive @who` (lands next round, full HP)",
}

CLASS_PICK_HELP = "pick your class: `class tank` (10 HP) · `class ranged` (6 HP) · `class melee` (8 HP) · `class healer` (4 HP)"

ACTION_WORDS = ("join", "attack", "taunt", "heal", "dot", "buff", "charged", "charge", "stun", "magic", "regen", "revive", "status")

ENEMY_TARGET_VERBS = ("attack", "magic", "dot", "stun", "charged", "charge")

# Narrator flavor. Prose only — every number in an event comes from the engine, never from here.
FLAVOR = {
    "attack_tank": (
        "{name} slams shield-first into {mname} with a wet SPLAT!",
        "{name} delivers a crushing overhead swing to {mname}!",
    ),
    "attack_ranged": (
        "{name} puts an arrow clean through {mname}'s wobbling bulk!",
        "{name} looses a shot that thuds home dead-center in {mname}!",
    ),
    "attack_melee": (
        "{name} carves a gleaming arc through {mname}!",
        "{name} steps in low and slashes deep into {mname}!",
    ),
    "attack_healer": (
        "{name} hurls a crackling bolt of holy light at {mname}!",
        "{name} mutters a quick prayer and zaps {mname}!",
    ),
    "charged_start": (
        "{name} plants their feet and draws power into their blade...",
        "{name} winds up something big...",
    ),
    "charged_release": (
        "{name}'s charged blow lands on {mname} like a falling anvil — CRUNCH!",
        "{name} releases the pent-up strike into {mname} in one devastating arc!",
    ),
    "taunt": (
        "{name} bangs their shield and bellows a challenge — every slime's beady eyes lock on!",
        "{name} shouts an insult so cutting the slimes forget anyone else exists!",
    ),
    "heal_self": (
        "{name} grits their teeth and binds their wounds mid-swing.",
        "{name} takes a long swig from a battered flask.",
    ),
    "dot": (
        "{name} coats an arrow in something evil-smelling and plants it in {mname}.",
        "{name} tags {mname} with a smoldering toxin.",
    ),
    "buff": (
        "{name} strikes up a battle hymn — the whole party stands taller!",
        "{name} passes out suspicious energy drinks — everyone feels invincible!",
    ),
    "stun": (
        "{name} cracks {mname} dead-center — it wobbles, cross-eyed and frozen!",
        "{name} lands a ringing blow that leaves {mname} seeing stars!",
    ),
    "special_cancelled": (
        "{mname}'s swelling bulk deflates with a sad squelch.",
        "The building power fizzles out of {mname} harmlessly.",
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
        "{mname} rears up and crashes down on {target}!",
        "{mname} lashes out with a whip of ooze, catching {target} square!",
    ),
    "m_aoe": (
        "{mname} bursts like a water balloon, splattering everyone with stinging goo!",
        "{mname} spins wildly, spraying acid across the whole party!",
    ),
    "m_charge": (
        "⚠️ {mname} swells to twice its size, quivering with building power...",
        "⚠️ {mname} glows an angry red and starts vibrating ominously...",
    ),
    "m_special": (
        "💥 {mname} ERUPTS, drowning the party in a tidal wave of ooze!",
        "💥 {mname} detonates outward in a concussive shockwave of goo!",
    ),
    "m_stunned": (
        "{mname} wobbles in place, utterly dazed.",
        "{mname} blinks its beady eyes, too scrambled to act.",
    ),
    "dot_tick": (
        "The toxin sizzles deeper into {mname}.",
        "{mname} hisses as the poison eats at it.",
    ),
    "regen_tick": (
        "{name}'s wounds knit closed a little more.",
        "The green glow does its quiet work on {name}.",
    ),
    "death": (
        "💀 {name} collapses in a heap!",
        "💀 {name} goes down hard and doesn't get up!",
    ),
    "m_death": (
        "☠️ {mname} loses cohesion and splatters into harmless puddles!",
        "☠️ {mname} deflates with a long, sad hiss and is no more!",
    ),
    "victory": (
        "🎉 The last slime shudders, deflates, and splatters apart. **VICTORY!**",
        "🎉 With one final pathetic gurgle, the last slime melts away. **VICTORY!**",
    ),
    "defeat": (
        "☠️ The last fighter falls. The slimes jiggle triumphantly over the fallen party.",
    ),
}

_flavor_rng = random.Random()


def _ev(key: str, mech: str = "", **kw) -> str:
    """One narrated event line: flavor prose plus the mechanical truth in italics."""
    line = _flavor_rng.choice(FLAVOR[key]).format(**kw)
    return f"{line} *({mech})*" if mech else line


def new_fight(deadline_ts: int, enemies: int = ENEMY_COUNT) -> dict:
    return {
        "round": 1,
        "active": True,
        "result": None,
        "deadline": deadline_ts,
        "seq": 0,
        "prompted": [],
        "monsters": [
            {"name": f"Slime {i + 1}" if enemies > 1 else "the slime", "hp": MONSTER_HP, "max_hp": MONSTER_HP, "stun": 0, "dot": 0, "charging": False}
            for i in range(enemies)
        ],
        "taunt": 0,
        "taunt_uid": None,
        "special_cd": 0,
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


def living_monsters(state: dict) -> list[dict]:
    return [m for m in state["monsters"] if m["hp"] > 0]


def _pick_enemy(state: dict, enemy: Optional[int]) -> Optional[dict]:
    """Explicit living target if given, else first living slime (redirect rule)."""
    if enemy is not None and 1 <= enemy <= len(state["monsters"]) and state["monsters"][enemy - 1]["hp"] > 0:
        return state["monsters"][enemy - 1]
    living = living_monsters(state)
    return living[0] if living else None


def _default_stun_target(state: dict, enemy: Optional[int]) -> Optional[dict]:
    """Stun prefers the explicit target, then a charging slime, then the first living one."""
    if enemy is not None and 1 <= enemy <= len(state["monsters"]) and state["monsters"][enemy - 1]["hp"] > 0:
        return state["monsters"][enemy - 1]
    charger = next((m for m in living_monsters(state) if m["charging"]), None)
    if charger:
        return charger
    living = living_monsters(state)
    return living[0] if living else None


def parse_verb(tokens: list[str], cls: str, target_uid: Optional[int]) -> Optional[tuple[str, Optional[int], Optional[int]]]:
    """Map typed keywords to (verb, target_player, target_enemy). None if not an action."""
    word = tokens[0]
    enemy = next((int(t) for t in tokens[1:] if t.isdigit()), None) if word in ENEMY_TARGET_VERBS else None

    if word == "attack":
        return "attack", None, enemy
    if cls == "tank":
        if word == "taunt":
            return "taunt", None, None
        if word == "heal":
            return "heal", None, None
    elif cls == "ranged":
        if word == "dot":
            return "dot", None, enemy
        if word == "buff":
            return "buff", None, None
    elif cls == "melee":
        if word in ("charged", "charge"):
            return "charged", None, enemy
        if word == "stun":
            return "stun", None, enemy
    elif cls == "healer":
        if word == "magic":
            return "attack", None, enemy
        if word == "regen":
            return "regen", target_uid, None
        if word == "heal":
            if len(tokens) > 1 and tokens[1] == "party":
                return "healparty", None, None
            return "healsingle", target_uid, None
        if word == "revive":
            return "revive", target_uid, None
    return None


def submit_action(state: dict, uid: int, verb: str, target_uid: Optional[int] = None, enemy: Optional[int] = None) -> tuple[bool, str]:
    player = state["players"][str(uid)]

    if not player["alive"]:
        return False, "you're down. Only a healer's `revive` can bring you back."
    if player["charging"]:
        return False, "you're locked in — your charged action lands when this round resolves."
    if enemy is not None and not (1 <= enemy <= len(state["monsters"])):
        return False, f"there's no slime {enemy} — pick 1 to {len(state['monsters'])}."
    if enemy is not None and state["monsters"][enemy - 1]["hp"] <= 0:
        return False, f"{state['monsters'][enemy - 1]['name']} is already dead."
    if verb == "stun":
        target_m = _default_stun_target(state, enemy)
        if target_m and target_m["stun"] > 0:
            return False, f"{target_m['name']} is already stunned."
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
    player["action"] = {"verb": verb, "target": str(target_uid) if target_uid else None, "enemy": enemy}
    player["seq"] = state["seq"]
    return True, ""


def resolve_round(state: dict, rng: random.Random) -> list[str]:
    monsters = state["monsters"]
    players = state["players"]
    events: list[str] = []

    def hit_monster(player: dict, base: int, label: str, flavor_key: str, enemy: Optional[int]) -> None:
        m = _pick_enemy(state, enemy)
        if m is None:
            return
        amount = base + (1 if player["buff"] > 0 else 0)
        m["hp"] = max(0, m["hp"] - amount)
        events.append(_ev(flavor_key, f"{label}: {amount} dmg — {m['name']} {m['hp']}/{m['max_hp']}", name=player["name"], mname=m["name"]))
        if m["hp"] == 0:
            events.append(_ev("m_death", mname=m["name"]))

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
        if not (player["alive"] and player["charging"]) or not living_monsters(state):
            continue
        charge, player["charging"] = player["charging"], None
        released.add(uid)
        if charge["verb"] == "charged":
            hit_monster(player, 6, "Charged Attack", "charged_release", charge.get("enemy"))
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
        if not living_monsters(state):
            break
        player = players[uid]
        action = player["action"] or {"verb": "attack", "target": None, "enemy": None}
        verb = action["verb"]
        enemy = action.get("enemy")

        if verb == "attack":
            hit_monster(player, CLASSES[player["cls"]]["attack"], CLASSES[player["cls"]]["attack_label"], f"attack_{player['cls']}", enemy)
        elif verb == "taunt":
            state["taunt"] = 2
            state["taunt_uid"] = uid
            events.append(_ev("taunt", f"taunt: ALL slimes must attack {player['name']} for 2 rounds", name=player["name"]))
        elif verb == "heal":
            healed = min(2, player["max_hp"] - player["hp"])
            player["hp"] += healed
            events.append(_ev("heal_self", f"+{healed} HP — {player['hp']}/{player['max_hp']}", name=player["name"]))
        elif verb == "dot":
            m = _pick_enemy(state, enemy)
            if m:
                m["dot"] = 4
                events.append(_ev("dot", f"DOT on {m['name']}: 1 dmg/round for 4 rounds", name=player["name"], mname=m["name"]))
        elif verb == "buff":
            for buff_uid in alive_players(state):
                buffed = players[buff_uid]
                if buffed["buff"] == 0:
                    buffed["max_hp"] += 1
                    buffed["hp"] += 1
                buffed["buff"] = 3
            events.append(_ev("buff", "party +1 HP & +1 dmg for 3 rounds", name=player["name"]))
        elif verb == "stun":
            m = _default_stun_target(state, enemy)
            if m:
                interrupted = m["charging"]
                m["stun"] = 2
                m["charging"] = False
                player["stun_cd"] = 4  # 2 stunned rounds + 2 rounds cooldown after the stun ends
                events.append(_ev("stun", f"{m['name']} stunned for 2 rounds", name=player["name"], mname=m["name"]))
                if interrupted:
                    state["special_cd"] = 4  # cancelled special goes on the shared 3-round cooldown
                    events.append(_ev("special_cancelled", "special cancelled — shared cooldown 3 rounds", mname=m["name"]))
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
            player["charging"] = {"verb": "charged", "target": None, "enemy": enemy}
            events.append(_ev("charged_start", "charged attack: 6 dmg, lands next round", name=player["name"]))
        elif verb == "revive":
            player["charging"] = {"verb": "revive", "target": action["target"], "enemy": None}
            events.append(_ev("revive_start", "revive: lands next round", name=player["name"], target=players[action["target"]]["name"]))

    # 3. Monster phase — each living slime acts in numbered order
    for m in monsters:
        if m["hp"] <= 0 or not alive_players(state):
            continue
        if m["stun"] > 0:
            events.append(_ev("m_stunned", f"stunned — {m['name']} skips its turn", mname=m["name"]))
        elif m["charging"]:
            m["charging"] = False
            state["special_cd"] = 4  # shared 3-round cooldown after firing (this round's tick counts first)
            events.append(_ev("m_special", "special attack: 3 dmg to everyone", mname=m["name"]))
            for uid in alive_players(state):
                players[uid]["hp"] -= 3
                check_death(players[uid])
        else:
            no_charger = not any(x["charging"] for x in living_monsters(state))
            options = ["attack", "aoe"] + (["special"] if state["special_cd"] == 0 and no_charger else [])
            choice = rng.choice(options)
            if choice == "attack":
                alive = alive_players(state)
                if state["taunt"] > 0 and state["taunt_uid"] in players and players[state["taunt_uid"]]["alive"]:
                    target_uid = state["taunt_uid"]
                else:
                    target_uid = min(alive, key=lambda u: (players[u]["hp"], players[u]["max_hp"], state["join_order"].index(u)))
                target = players[target_uid]
                target["hp"] -= 3
                events.append(_ev("m_attack", f"3 dmg — {target['name']} {max(target['hp'], 0)}/{target['max_hp']}", mname=m["name"], target=target["name"]))
                check_death(target)
            elif choice == "aoe":
                events.append(_ev("m_aoe", "1 dmg to everyone", mname=m["name"]))
                for uid in alive_players(state):
                    players[uid]["hp"] -= 1
                    check_death(players[uid])
            else:
                m["charging"] = True
                events.append(_ev("m_charge", "special next round: 3 dmg to all — stun it to cancel!", mname=m["name"]))

    # 4. End-of-round ticks, then status durations count down
    for m in monsters:
        if m["hp"] > 0 and m["dot"] > 0:
            m["hp"] = max(0, m["hp"] - 1)
            events.append(_ev("dot_tick", f"DOT: 1 dmg — {m['name']} {m['hp']}/{m['max_hp']}", mname=m["name"]))
            if m["hp"] == 0:
                events.append(_ev("m_death", mname=m["name"]))
    for uid in alive_players(state):
        player = players[uid]
        if player["regen"] > 0 and player["hp"] < player["max_hp"]:
            healed = min(2, player["max_hp"] - player["hp"])
            player["hp"] += healed
            events.append(_ev("regen_tick", f"regen: +{healed} — {player['hp']}/{player['max_hp']}", name=player["name"]))

    for m in monsters:
        m["dot"] = max(0, m["dot"] - 1)
        m["stun"] = max(0, m["stun"] - 1)
    state["taunt"] = max(0, state["taunt"] - 1)
    state["special_cd"] = max(0, state["special_cd"] - 1)
    for player in players.values():
        player["stun_cd"] = max(0, player["stun_cd"] - 1)
        player["regen"] = max(0, player["regen"] - 1)
        if player["buff"] > 0:
            player["buff"] -= 1
            if player["buff"] == 0:
                player["max_hp"] -= 1
                player["hp"] = min(player["hp"], player["max_hp"])

    # 5. Outcome
    if not living_monsters(state):
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
    header_tags = []
    if state["taunt"] and state["taunt_uid"] in state["players"]:
        header_tags.append(f"taunted by {state['players'][state['taunt_uid']]['name']} ({state['taunt']} round(s))")
    if state["special_cd"]:
        header_tags.append(f"special on shared cooldown ({state['special_cd']} round(s))")
    lines = ["**Enemies**" + (f" — {', '.join(header_tags)}" if header_tags else "")]

    for i, m in enumerate(state["monsters"], start=1):
        tags = []
        if m["hp"] <= 0:
            tags.append("☠️ dead")
        if m["charging"]:
            tags.append("⚠️ CHARGING ITS SPECIAL")
        if m["stun"]:
            tags.append(f"stunned ({m['stun']} round(s))")
        if m["dot"]:
            tags.append(f"DOT ({m['dot']} round(s))")
        lines.append(f"[{i}] {m['name']} — {m['hp']}/{m['max_hp']} HP" + (f" — {', '.join(tags)}" if tags else ""))

    lines.append("")
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
