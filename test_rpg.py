"""Engine tests for the server RPG — every rule and reading in RPG-DESIGN.md.

Run: python test_rpg.py
"""

from rpg_engine import add_player, alive_players, new_fight, resolve_round, submit_action


class ForceRng:
    """Forces the monster's choices in order; asserts if the script runs dry."""

    def __init__(self, *picks: str) -> None:
        self.picks = list(picks)

    def choice(self, options: list[str]) -> str:
        assert self.picks, "monster asked for a choice but the test script is empty"
        pick = self.picks.pop(0)
        assert pick in options, f"forced {pick}, engine offered {options}"
        return pick


def make_party(monster_hp=None):
    state = new_fight(deadline_ts=0)
    add_player(state, 1, "tanky", "tank")        # 10 HP
    add_player(state, 2, "ranger", "ranged")     # 6 HP
    add_player(state, 3, "slasher", "melee")     # 8 HP
    add_player(state, 4, "medic", "healer")      # 4 HP
    if monster_hp:  # tests of long fights need the slime to survive default attacks
        state["monster"]["hp"] = state["monster"]["max_hp"] = monster_hp
    return state


def test_basic_round_math_and_lowest_hp_targeting():
    state = make_party()
    # Nobody submits: everyone defaults to basic attack (2+3+4+2 = 11)
    resolve_round(state, ForceRng("attack"))
    assert state["monster"]["hp"] == 20 - 11, state["monster"]["hp"]
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
    state = make_party()
    resolve_round(state, ForceRng("special"))            # round 1: slime charges
    assert state["monster"]["charging"] is True

    ok, _ = submit_action(state, 3, "stun")              # round 2: melee stuns
    assert ok
    resolve_round(state, ForceRng())                     # stunned: no choice consumed
    assert state["monster"]["charging"] is False         # special interrupted
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
    assert state["monster"]["hp"] == 20 - (2 + 3 + 2)    # others defaulted; no melee damage yet

    ok, err = submit_action(state, 3, "stun")            # locked while charging
    assert not ok and "locked in" in err

    resolve_round(state, ForceRng("aoe"))
    # Charged 6 landed, plus the other three defaults (2+3+2)
    assert state["monster"]["hp"] == 13 - 6 - 7


def test_dot_ticks_exactly_four_rounds():
    state = new_fight(deadline_ts=0)
    add_player(state, 2, "ranger", "ranged")
    submit_action(state, 2, "dot")
    hp = 20
    resolve_round(state, ForceRng("aoe"))                # dot applied + attack was replaced by dot
    hp -= 1                                              # tick 1
    assert state["monster"]["hp"] == hp
    for _ in range(3):                                   # ticks 2-4, ranger defaults to attack (3)
        resolve_round(state, ForceRng("aoe"))
        hp -= 3 + 1
        assert state["monster"]["hp"] == hp
    resolve_round(state, ForceRng("aoe"))                # dot expired: attack only
    assert state["monster"]["hp"] == hp - 3


def test_buff_adds_hp_and_damage_then_expires():
    state = make_party()
    submit_action(state, 2, "buff")                      # submitted first
    submit_action(state, 1, "attack")                    # tank attacks after buff: 2+1
    resolve_round(state, ForceRng("aoe"))
    # damage: tank 3 (buffed), melee 4+1, healer 2+1 = 11 (ranged spent turn buffing)
    assert state["monster"]["hp"] == 20 - 11
    tank = state["players"]["1"]
    assert tank["max_hp"] == 11 and tank["hp"] == 10     # +1 max/current, then AOE -1
    resolve_round(state, ForceRng("aoe"))
    resolve_round(state, ForceRng("aoe"))                # buff expires end of round 3
    assert tank["max_hp"] == 10 and tank["buff"] == 0


def test_healer_verbs_and_revive_full_hp():
    state = make_party(monster_hp=200)
    medic = state["players"]["4"]
    slasher = state["players"]["3"]

    # Kill the melee: 3 slime attacks (taunt off, force-lower his HP so he's targeted)
    slasher["hp"] = 1
    resolve_round(state, ForceRng("attack"))
    assert not slasher["alive"]

    ok, err = submit_action(state, 3, "attack")          # dead players can't act
    assert not ok and "down" in err

    ok, _ = submit_action(state, 4, "revive", 3)         # healer starts the revive (1-round charge)
    assert ok
    resolve_round(state, ForceRng("aoe"))
    assert not slasher["alive"]                          # still charging this round
    resolve_round(state, ForceRng("aoe"))                # revive lands at start of next round...
    assert slasher["alive"] and slasher["hp"] == slasher["max_hp"] - 1  # ...then this round's AOE clips him

    # heal party and heal single caps
    medic["hp"] = 1
    submit_action(state, 4, "healparty")
    resolve_round(state, ForceRng("aoe"))
    assert medic["hp"] == min(4, 1 + 3) - 1              # +3 party heal, then AOE

    submit_action(state, 4, "healsingle", 4)
    resolve_round(state, ForceRng("aoe"))
    assert medic["hp"] == 4 - 1                          # +7 capped at max 4, then AOE

    # regen: +2/round for 4 rounds (ticks after the monster acts — 0 HP means down, no rescue tick)
    tank = state["players"]["1"]
    tank["hp"] = 2
    submit_action(state, 4, "regen", 1)
    resolve_round(state, ForceRng("aoe"))
    assert tank["hp"] == 2 - 1 + 2                       # AOE then regen tick
    assert tank["regen"] == 3


class OptionsProbe:
    """Records what abilities the engine offered the monster; always picks attack."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def choice(self, options: list[str]) -> str:
        self.seen.append(list(options))
        return "attack"


def test_special_cooldown_after_firing():
    state = new_fight(deadline_ts=0)
    add_player(state, 1, "tanky", "tank")
    state["monster"]["hp"] = state["monster"]["max_hp"] = 500
    resolve_round(state, ForceRng("special"))            # round 1: charge
    resolve_round(state, ForceRng())                     # round 2: fires -> 3-round cooldown
    for expect_special in (False, False, False, True):   # rounds 3-6
        state["players"]["1"]["hp"] = 10                 # keep the solo tank alive
        probe = OptionsProbe()
        resolve_round(state, probe)
        assert ("special" in probe.seen[0]) is expect_special, (state["round"], probe.seen)


def test_stun_cancel_puts_special_on_cooldown():
    state = new_fight(deadline_ts=0)
    add_player(state, 3, "slasher", "melee")
    state["monster"]["hp"] = state["monster"]["max_hp"] = 500
    resolve_round(state, ForceRng("special"))            # round 1: charge
    submit_action(state, 3, "stun")                      # round 2: cancel it
    resolve_round(state, ForceRng())
    assert state["monster"]["charging"] is False
    assert state["monster"]["special_cd"] == 3           # cooldown started at cancellation
    resolve_round(state, ForceRng())                     # round 3: still stunned
    for expect_special in (False, False, True):          # rounds 4-6
        state["players"]["3"]["hp"] = 8
        probe = OptionsProbe()
        resolve_round(state, probe)
        assert ("special" in probe.seen[0]) is expect_special, (state["round"], probe.seen)


def test_victory_and_defeat():
    state = make_party()
    state["monster"]["hp"] = 5
    resolve_round(state, ForceRng())                     # 11 damage kills it; monster never acts
    assert state["result"] == "victory" and not state["active"]

    state = new_fight(deadline_ts=0)
    add_player(state, 4, "medic", "healer")              # 4 HP vs 3-damage special
    resolve_round(state, ForceRng("special"))
    resolve_round(state, ForceRng("attack"))             # special (3) leaves 1 HP...
    # round 1: special charge; round 2: special hits for 3 -> 1 HP; round 3: attack kills
    resolve_round(state, ForceRng("attack"))
    assert state["result"] == "defeat" and not state["active"]


def test_special_hits_whole_party():
    state = make_party(monster_hp=200)
    resolve_round(state, ForceRng("special"))
    resolve_round(state, ForceRng())                     # release: no choice consumed
    hps = {uid: p["hp"] for uid, p in state["players"].items()}
    assert hps == {"1": 7, "2": 3, "3": 5, "4": 1}


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
