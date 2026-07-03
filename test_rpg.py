"""Engine tests for the server RPG — every rule and reading in RPG-DESIGN.md.

Single-slime tests (enemies=1) verify the core verb rules; multi-slime tests verify the
3-enemy rulings (taunt-all, shared special, targeting, redirects).

Run: python test_rpg.py
"""

from rpg_engine import add_player, alive_players, new_fight, resolve_round, submit_action


class ForceRng:
    """Forces the monsters' choices in order; asserts if the script runs dry."""

    def __init__(self, *picks: str) -> None:
        self.picks = list(picks)

    def choice(self, options: list[str]) -> str:
        assert self.picks, "a slime asked for a choice but the test script is empty"
        pick = self.picks.pop(0)
        assert pick in options, f"forced {pick}, engine offered {options}"
        return pick


class ScriptProbe:
    """Records the options offered to each slime; plays scripted picks, else attack."""

    def __init__(self, *picks: str) -> None:
        self.picks = list(picks)
        self.seen: list[list[str]] = []

    def choice(self, options: list[str]) -> str:
        self.seen.append(list(options))
        pick = self.picks.pop(0) if self.picks else "attack"
        return pick if pick in options else "attack"


def make_party(monster_hp=None, enemies=1):
    state = new_fight(deadline_ts=0, enemies=enemies)
    add_player(state, 1, "tanky", "tank")        # 10 HP
    add_player(state, 2, "ranger", "ranged")     # 6 HP
    add_player(state, 3, "slasher", "melee")     # 8 HP
    add_player(state, 4, "medic", "healer")      # 4 HP
    if monster_hp:  # tests of long fights need slimes that survive default attacks
        for m in state["monsters"]:
            m["hp"] = m["max_hp"] = monster_hp
    return state


# ---------------------------------------------------------------------------
# Single-slime rules
# ---------------------------------------------------------------------------

def test_basic_round_math_and_lowest_hp_targeting():
    state = make_party()
    # Nobody submits: everyone defaults to basic attack (2+3+4+2 = 11)
    resolve_round(state, ForceRng("attack"))
    assert state["monsters"][0]["hp"] == 20 - 11
    # Slime attacked the lowest-HP player: the 4 HP healer, now 1 HP
    assert state["players"]["4"]["hp"] == 1
    assert state["round"] == 2


def test_taunt_redirects_for_two_rounds_then_expires():
    state = make_party(monster_hp=200)
    submit_action(state, 1, "taunt")
    resolve_round(state, ForceRng("attack"))
    assert state["players"]["1"]["hp"] == 7      # tank ate the hit
    assert state["players"]["4"]["hp"] == 4      # healer untouched
    resolve_round(state, ForceRng("attack"))
    assert state["players"]["1"]["hp"] == 4      # taunt still active round 2
    resolve_round(state, ForceRng("attack"))
    assert state["players"]["1"]["hp"] == 4      # taunt expired...
    assert state["players"]["4"]["hp"] == 1      # ...healer (lowest HP) hit again


def test_tank_self_heal_caps_at_max():
    state = make_party()
    state["players"]["1"]["hp"] = 9
    submit_action(state, 1, "heal")
    resolve_round(state, ForceRng("aoe"))
    # +2 capped at max 10, then AOE -1
    assert state["players"]["1"]["hp"] == 9


def test_stun_interrupts_special_and_cooldown_prevents_lock():
    state = make_party(monster_hp=200)
    resolve_round(state, ForceRng("special"))            # round 1: slime charges
    assert state["monsters"][0]["charging"] is True

    ok, _ = submit_action(state, 3, "stun")              # round 2: melee stuns
    assert ok
    resolve_round(state, ForceRng())                     # stunned: no choice consumed
    assert state["monsters"][0]["charging"] is False     # special interrupted
    assert all(p["hp"] == p["max_hp"] for p in state["players"].values())  # nobody hit

    ok, err = submit_action(state, 3, "stun")            # round 3: still stunned
    assert not ok and "already stunned" in err
    resolve_round(state, ForceRng())                     # second stunned round

    ok, err = submit_action(state, 3, "stun")            # round 4: stun over, cooldown active
    assert not ok and "cooldown" in err
    resolve_round(state, ForceRng("aoe"))

    ok, err = submit_action(state, 3, "stun")            # round 5: cooldown still active
    assert not ok and "cooldown" in err
    resolve_round(state, ForceRng("aoe"))

    ok, _ = submit_action(state, 3, "stun")              # round 6: available again
    assert ok


def test_charged_attack_lands_next_round():
    state = make_party()
    submit_action(state, 3, "charged")
    resolve_round(state, ForceRng("aoe"))
    assert state["monsters"][0]["hp"] == 20 - (2 + 3 + 2)  # others defaulted; no melee damage yet

    ok, err = submit_action(state, 3, "stun")            # locked while charging
    assert not ok and "locked in" in err

    resolve_round(state, ForceRng("aoe"))
    # Charged 6 landed, plus the other three defaults (2+3+2)
    assert state["monsters"][0]["hp"] == 13 - 6 - 7


def test_dot_ticks_exactly_four_rounds():
    state = new_fight(deadline_ts=0, enemies=1)
    add_player(state, 2, "ranger", "ranged")
    submit_action(state, 2, "dot")
    hp = 20
    resolve_round(state, ForceRng("aoe"))                # dot applied; attack replaced by dot
    hp -= 1                                              # tick 1
    assert state["monsters"][0]["hp"] == hp
    for _ in range(3):                                   # ticks 2-4, ranger defaults to attack (3)
        resolve_round(state, ForceRng("aoe"))
        hp -= 3 + 1
        assert state["monsters"][0]["hp"] == hp
    resolve_round(state, ForceRng("aoe"))                # dot expired: attack only
    assert state["monsters"][0]["hp"] == hp - 3


def test_buff_adds_hp_and_damage_then_expires():
    state = make_party()
    submit_action(state, 2, "buff")                      # submitted first
    submit_action(state, 1, "attack")                    # tank attacks after buff: 2+1
    resolve_round(state, ForceRng("aoe"))
    # damage: tank 3 (buffed), melee 4+1, healer 2+1 = 11 (ranged spent turn buffing)
    assert state["monsters"][0]["hp"] == 20 - 11
    tank = state["players"]["1"]
    assert tank["max_hp"] == 11 and tank["hp"] == 10     # +1 max/current, then AOE -1
    resolve_round(state, ForceRng("aoe"))
    resolve_round(state, ForceRng("aoe"))                # buff expires end of round 3
    assert tank["max_hp"] == 10 and tank["buff"] == 0


def test_healer_verbs_and_revive_full_hp():
    state = make_party(monster_hp=200)
    medic = state["players"]["4"]
    slasher = state["players"]["3"]

    slasher["hp"] = 1
    resolve_round(state, ForceRng("attack"))             # slime kills the 1 HP melee
    assert not slasher["alive"]

    ok, err = submit_action(state, 3, "attack")          # dead players can't act
    assert not ok and "down" in err

    ok, _ = submit_action(state, 4, "revive", 3)         # healer starts the revive (1-round charge)
    assert ok
    resolve_round(state, ForceRng("aoe"))
    assert not slasher["alive"]                          # still charging this round
    resolve_round(state, ForceRng("aoe"))                # revive lands at start of next round...
    assert slasher["alive"] and slasher["hp"] == slasher["max_hp"] - 1  # ...then this round's AOE clips him

    medic["hp"] = 1
    submit_action(state, 4, "healparty")
    resolve_round(state, ForceRng("aoe"))
    assert medic["hp"] == min(4, 1 + 3) - 1              # +3 party heal, then AOE

    submit_action(state, 4, "healsingle", 4)
    resolve_round(state, ForceRng("aoe"))
    assert medic["hp"] == 4 - 1                          # +7 capped at max 4, then AOE

    # regen: +2/round for 4 rounds (ticks after the monsters act — 0 HP means down, no rescue tick)
    tank = state["players"]["1"]
    tank["hp"] = 2
    submit_action(state, 4, "regen", 1)
    resolve_round(state, ForceRng("aoe"))
    assert tank["hp"] == 2 - 1 + 2                       # AOE then regen tick
    assert tank["regen"] == 3


def test_special_cooldown_after_firing():
    state = new_fight(deadline_ts=0, enemies=1)
    add_player(state, 1, "tanky", "tank")
    state["monsters"][0]["hp"] = state["monsters"][0]["max_hp"] = 500
    resolve_round(state, ForceRng("special"))            # round 1: charge
    resolve_round(state, ForceRng())                     # round 2: fires -> shared 3-round cooldown
    for expect_special in (False, False, False, True):   # rounds 3-6
        state["players"]["1"]["hp"] = 10                 # keep the solo tank alive
        probe = ScriptProbe()
        resolve_round(state, probe)
        assert ("special" in probe.seen[0]) is expect_special, (state["round"], probe.seen)


def test_stun_cancel_puts_special_on_cooldown():
    state = new_fight(deadline_ts=0, enemies=1)
    add_player(state, 3, "slasher", "melee")
    state["monsters"][0]["hp"] = state["monsters"][0]["max_hp"] = 500
    resolve_round(state, ForceRng("special"))            # round 1: charge
    submit_action(state, 3, "stun")                      # round 2: cancel it
    resolve_round(state, ForceRng())
    assert state["monsters"][0]["charging"] is False
    assert state["special_cd"] == 3                      # cooldown started at cancellation
    resolve_round(state, ForceRng())                     # round 3: still stunned
    for expect_special in (False, False, True):          # rounds 4-6
        state["players"]["3"]["hp"] = 8
        probe = ScriptProbe()
        resolve_round(state, probe)
        assert ("special" in probe.seen[0]) is expect_special, (state["round"], probe.seen)


def test_special_hits_whole_party():
    state = make_party(monster_hp=200)
    resolve_round(state, ForceRng("special"))
    resolve_round(state, ForceRng())                     # release: no choice consumed
    hps = {uid: p["hp"] for uid, p in state["players"].items()}
    assert hps == {"1": 7, "2": 3, "3": 5, "4": 1}


def test_victory_and_defeat():
    state = make_party()
    state["monsters"][0]["hp"] = 5
    resolve_round(state, ForceRng())                     # 11 damage kills it; slime never acts
    assert state["result"] == "victory" and not state["active"]

    state = new_fight(deadline_ts=0, enemies=1)
    add_player(state, 4, "medic", "healer")              # 4 HP vs the special + attack
    resolve_round(state, ForceRng("special"))
    resolve_round(state, ForceRng())                     # special fires: 4 -> 1 HP
    resolve_round(state, ForceRng("attack"))             # attack kills
    assert state["result"] == "defeat" and not state["active"]


# ---------------------------------------------------------------------------
# Multi-slime rulings (3 enemies)
# ---------------------------------------------------------------------------

def test_multi_focus_fire_and_redirect():
    state = make_party(enemies=3)
    state["monsters"][0]["hp"] = 5
    submit_action(state, 1, "attack", enemy=1)           # 2 dmg -> 3 left
    submit_action(state, 2, "attack", enemy=1)           # 3 dmg -> dead
    submit_action(state, 3, "attack", enemy=1)           # target dead: redirects to slime 2 (4 dmg)
    submit_action(state, 4, "attack", enemy=1)           # redirects too (2 dmg)
    resolve_round(state, ForceRng("attack", "attack"))   # only slimes 2 and 3 act
    assert state["monsters"][0]["hp"] == 0
    assert state["monsters"][1]["hp"] == 20 - 6
    assert state["monsters"][2]["hp"] == 20
    assert not state["players"]["4"]["alive"]            # two attacks focused the 4 HP healer


def test_multi_taunt_pulls_every_slime():
    state = make_party(enemies=3)
    submit_action(state, 1, "taunt")
    resolve_round(state, ForceRng("attack", "attack", "attack"))
    assert state["players"]["1"]["hp"] == 10 - 9         # tank ate all three hits
    assert state["players"]["4"]["hp"] == 4              # healer untouched


def test_multi_only_one_special_charger_at_a_time():
    state = make_party(enemies=3)
    for m in state["monsters"]:
        m["hp"] = m["max_hp"] = 500
    probe = ScriptProbe("special", "special", "special") # all three want to charge
    resolve_round(state, probe)
    assert state["monsters"][0]["charging"] is True
    assert "special" in probe.seen[0]
    assert "special" not in probe.seen[1]                # a charger exists: option withdrawn
    assert "special" not in probe.seen[2]

    probe2 = ScriptProbe()                               # round 2: slime 1 fires, others choose after
    resolve_round(state, probe2)
    assert state["special_cd"] == 3                      # shared cooldown running
    assert all("special" not in seen for seen in probe2.seen)


def test_multi_stun_defaults_to_charger_and_shared_cooldown():
    state = make_party(enemies=3)
    for m in state["monsters"]:
        m["hp"] = m["max_hp"] = 500
    probe = ScriptProbe("attack", "special", "attack")   # slime 2 charges
    resolve_round(state, probe)
    assert state["monsters"][1]["charging"] is True

    submit_action(state, 3, "stun")                      # no target: prefers the charger
    resolve_round(state, ForceRng("attack", "attack"))   # slimes 1 and 3 act; slime 2 stunned
    assert state["monsters"][1]["charging"] is False
    assert state["monsters"][1]["stun"] == 1             # 2 set, ticked once
    assert state["special_cd"] == 3                      # cancelled -> shared cooldown


def test_multi_dot_is_per_slime():
    state = make_party(enemies=3)
    submit_action(state, 2, "dot", enemy=2)
    resolve_round(state, ForceRng("attack", "attack", "attack"))
    assert state["monsters"][0]["hp"] == 20 - 8          # tank 2 + melee 4 + healer 2 defaults
    assert state["monsters"][1]["hp"] == 19              # DOT tick only
    assert state["monsters"][1]["dot"] == 3
    assert state["monsters"][2]["hp"] == 20 and state["monsters"][2]["dot"] == 0


def test_multi_victory_requires_all_dead():
    state = make_party(enemies=3)
    state["monsters"][0]["hp"] = 0
    state["monsters"][1]["hp"] = 0
    state["monsters"][2]["hp"] = 5
    resolve_round(state, ForceRng())                     # defaults kill the last slime mid-phase
    assert state["result"] == "victory" and not state["active"]


def test_multi_submit_validation():
    state = make_party(enemies=3)
    ok, err = submit_action(state, 1, "attack", enemy=4)
    assert not ok and "no slime 4" in err
    state["monsters"][2]["hp"] = 0
    ok, err = submit_action(state, 1, "attack", enemy=3)
    assert not ok and "dead" in err


def run_all():
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL {name}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(run_all())
