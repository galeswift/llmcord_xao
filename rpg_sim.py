"""Local simulator for the server RPG — run fights without Discord or the bot.

Usage:
    python rpg_sim.py               interactive session
    python rpg_sim.py script.txt    run commands from a file (one per line, # for comments)

Commands:
    add <name> <class>          add a simulated member (tank | ranged | melee | healer)
    <name>: <action>            queue that member's action, same keywords as Discord:
                                    keith: taunt
                                    dave: attack 2          (slime number for single-target verbs)
                                    amy: heal party
                                    rob: heal keith         (names instead of @mentions)
    resolve [attack|aoe|special|random ...]
                                resolve the round; forced picks are consumed one per acting
                                slime in numbered order, e.g. `resolve attack special aoe`
    status                      show the fight state
    quit                        exit

The exact same engine (rpg_engine.py) runs here and in the bot, so anything you see here is
what Discord will do.
"""

import random
import sys

from rpg_engine import (
    CLASSES,
    add_player,
    format_status,
    new_fight,
    parse_verb,
    resolve_round,
    submit_action,
)


class ForcedRng:
    """Random source whose next monster choice can be forced from the command line."""

    def __init__(self) -> None:
        self.forced: list[str] = []
        self.rng = random.Random()

    def choice(self, options: list[str]) -> str:
        while self.forced:
            pick = self.forced.pop(0)
            if pick in options:
                return pick
            if pick != "random":
                print(f"  (can't force '{pick}' here — choosing randomly)")
        return self.rng.choice(options)


class Sim:
    def __init__(self) -> None:
        self.state = new_fight(deadline_ts=0)
        self.names: dict[str, int] = {}  # lowercase name -> fake uid
        self.next_uid = 1001
        self.rng = ForcedRng()

    def uid_for(self, name: str) -> int | None:
        return self.names.get(name.lower().lstrip("@"))

    def run_command(self, line: str) -> bool:
        """Execute one command. Returns False when the session should end."""
        line = line.strip()
        if not line or line.startswith("#"):
            return True
        lowered = line.lower()

        if lowered in ("quit", "exit"):
            return False

        if lowered == "status":
            print(format_status(self.state, show_deadline=False))
            return True

        if lowered.startswith("add "):
            parts = lowered.split()
            if len(parts) != 3 or parts[2] not in CLASSES:
                print("usage: add <name> <tank|ranged|melee|healer>")
                return True
            name, cls = parts[1], parts[2]
            if name in self.names:
                print(f"{name} is already in the fight.")
                return True
            self.names[name] = self.next_uid
            player = add_player(self.state, self.next_uid, name, cls)
            self.next_uid += 1
            print(f"⚔️ {name} joins as {CLASSES[cls]['name']} ({player['hp']} HP)")
            return True

        if lowered.startswith("resolve"):
            if not self.state["active"]:
                print(f"the fight is over ({self.state['result']}).")
                return True
            if not self.state["players"]:
                print("nobody has joined — `add` some members first.")
                return True
            self.rng.forced = lowered.split()[1:]
            round_number = self.state["round"]
            events = resolve_round(self.state, self.rng)
            print(f"--- Round {round_number} ---")
            for event in events:
                print(f"  {event}")
            if self.state["active"]:
                print()
                print(format_status(self.state, show_deadline=False))
            else:
                print(f"\nFight over: {self.state['result'].upper()}")
            return self.state["active"] or True  # keep session open to inspect final state

        if ":" in line:
            name, _, action_text = line.partition(":")
            uid = self.uid_for(name.strip())
            if uid is None:
                print(f"unknown member '{name.strip()}' — add them first.")
                return True
            player = self.state["players"][str(uid)]
            tokens = action_text.strip().lower().split()
            if not tokens:
                print("give an action, e.g. `keith: taunt`")
                return True
            target_uid = next((self.uid_for(t) for t in tokens[1:] if self.uid_for(t) is not None), None)
            parsed = parse_verb(tokens, player["cls"], target_uid)
            if parsed is None:
                print(f"'{action_text.strip()}' isn't a {CLASSES[player['cls']]['name']} action.")
                return True
            ok, err = submit_action(self.state, uid, parsed[0], parsed[1], parsed[2])
            print("✅ locked in" if ok else f"❌ {err}")
            return True

        print("commands: add <name> <class> · <name>: <action> · resolve [attack|aoe|special] · status · quit")
        return True


def main() -> None:
    # Windows consoles often default to cp1252, which can't print the emoji in event text
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sim = Sim()
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            for line in f:
                print(f"> {line.rstrip()}")
                if not sim.run_command(line):
                    break
    else:
        print("RPG simulator — `add keith tank` to start, `quit` to leave. Same engine as the bot.")
        while True:
            try:
                line = input("> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not sim.run_command(line):
                break


if __name__ == "__main__":
    main()
