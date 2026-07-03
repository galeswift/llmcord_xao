# Server RPG — Design Document

This file drives development. The designer (Keith) owns all gameplay content; Claude implements
exactly what is written here and nothing more.

**Rules of this document:**
- Sections marked `[TBD — designer]` block implementation of the features that depend on them.
- Claude does not invent gameplay to fill gaps. If the code needs an answer that isn't here,
  Claude asks and the answer gets written into this file first.
- Numbers in this file are canon. If the code and this file disagree, the file wins.
- Changes are recorded in the Decision Log at the bottom.

---

## 1. Design goals (designer's words)

- Fun gameplay comes from **meaningful player choice** and a **well-defined goal that is understood**.
- Strong sense of progression.
- Heavy cooperation required.
- Simple systems: 2-slot equipment, very simple 4-class system.

## 2.  First goal, example cooperative combat
- Start up a fight in a thread.  The fights should play out similar to the dragon quest/dragon warrior games.
- Each player that joins the thread will be able to participate by choosing a combat for the round
- A round happens twice a day, with countdown timer (that uses the discord current time markup)
- Example title of the thread is like "You encounter a wild slime!  Join the thread to fight it"
- When you join the thread, you are asked to pick a class by voting (if you haven't picked one yet)
- If you have a class, you are presented with options based on the class
- Tank - HP=10:  Attack (2 damage), Taunt, Heal 
- Ranged DPS - HP=6: Attack (3 damage), Apply DOT, Buff Party (gives temporary 1 hp and 1 damage)
- Melee DPS - HP=8: Attack (4 damage), Charged Attack (6 damage, 1 round charge time) Stun
- Healer - HP=4: Magic Attack (2 damage), Apply Regen, Heal Party, Heal Single, Revive ally (1 turn charge time)
- First Monster - HP=20:  Attack (3 damage, will attack lowest HP target unless taunted by tank), AOE (1 damage), Special attack (1 round charge time, hits entire party for 3 damage, 3 round cooldown)

## 3. Open questions — blocking implementation `[answer inline — designer]`

Verb numbers:
- Taunt: what does it do exactly, and for how long (this round / until next round)? → Forces Enemy to attack the tank for 2 rounds
- Tank's Heal: heals whom, for how much? → Heals self for 2 health
- Apply DOT: damage per round, duration, can it stack? → Cannot stack, lasts 4 rounds, 1 damage per round
- Buff Party: "temporary" +1 hp / +1 damage lasts until when? → Lasts 3 rounds
- Stun: effect (slime skips next action?), and limit (uses per fight / cooldown — unlimited allows stun-lock)? → 2 round cooldown, stuns monster for 2 rounds, interrupts special attack
- Apply Regen: heal per round, duration, target? → heals 2 per round, lasts 4 rounds, single target
- Heal Party: amount per player? → heals entire party for 3
- Heal Single: amount healed?  → heals single member for 7
- Magic Attack: differs from a normal Attack how (or is it just flavor)? → ignores armor (if enemy has any)
- Revive: ally returns with how much HP? → full max HP
- Stun cooldown: counts from cast or from when the stun ends? → after the stun ends (stun rounds 1–2, cooldown 3–4, earliest re-stun round 5)

Monster behavior:
- When does the slime use AOE instead of Attack? → slime randomly chooses between all abilities
- AOE = 1 damage to every player? → 1 damage to each player
- If two players tie for lowest HP, who gets attacked? → based on max HP (so healer first, then ranged dps, then melee)

Round flow:
- Order of resolution within a round (all players act, then monster? player order?): → All players act, then monster
- Player joined but didn't act before the timer: skipped, or default Attack? → default attack
- The two daily round times + timezone: → Noon, and 8 pm PST
- Can a player join after round 1? Starting HP if so? → yes, no penalty

Life and death:
- Player at 0 HP: out of the fight, or can heals bring them back? → Out of fight, healer can revive
- All players at 0 = loss? What happens then? → monster wins, no reward for fight
- On victory: anything besides the narrated win, for this first fight? → (exp reward for each player, let's say 5 for first fight)

Interface:
- Class selection: each player picks their own class via buttons — correct? Permanent once picked? → can add a change class option later, but have to restart at level 1
- Choosing the round action: buttons on the bot's round post, or typed keywords in the thread? → keywords typed

## 4. Implementation readings (Claude) — literal interpretations of the spec, correct me here

- **Joining:** joining the thread (or posting in it) is the entry point. A player becomes a combatant
  when they pick a class (`class tank|ranged|melee|healer`) or, if their class is already known, when
  they type `join` or any action keyword. Plain chatter doesn't conscript spectators.
- **Charge verbs** (Charged Attack, Revive, monster Special): casting spends this round's action;
  the effect fires automatically as that actor's action next round. While charging you can't act or
  change your mind; dying cancels the charge. The slime's Special charge is announced when it starts.
- **Resolution order within a round:** charge releases first (in the order started), then submitted
  actions in the order they were submitted, then players who didn't act (default Attack; healer
  default is Magic Attack). Effects apply immediately — a Buff cast before an ally's attack boosts
  that attack in the same round. If the slime dies mid-phase, remaining actions don't execute.
- **Buff Party:** +1 current and max HP while active (removed on expiry, current HP capped), +1 to
  direct attack damage (Attack / Magic / Charged — not DOT ticks). Recasting refreshes the 3 rounds,
  never stacks.
- **Taunt** redirects only the slime's single-target Attack (AOE and Special still hit everyone) and
  covers the slime's next 2 actions.
- **Stun** Stuns a slime for 2 rounds. (If they were charging a special attack, it gets cancelled and the slime's special goes on cooldown like usual)
- **Special cooldown timing:** the 3-round cooldown counts the 3 rounds after the special fires (or
  is stun-cancelled) — the slime can't start charging it again until the 4th round after.
- **DOT and Regen** tick at the end of the round, after the monster acts. The DOT can finish the
  slime off.
- **Death** clears your buffs/regen/charge. Revive returns you at base max HP (buffs must be recast).
- **Victory XP:** all combatants get the 5 XP, dead or alive.
- **Actions are typed keywords** in the thread; your last submission before the deadline is the one
  that counts (you can change your mind). Targeted verbs need an @mention: `regen @who`, `heal @who`,
  `revive @who`. `heal party` is the party heal; a bare `heal` from the Tank is the self-heal.
- Duplicate classes are allowed. `status` reposts the fight state any time.
- **Fight start:** manual `/rpg` command (admin only) for this first fight; daily automation comes
  with the broader loop later. Round times are America/Los_Angeles local time.

## 5. Decision Log

| Date | Decision | By |
|---|---|---|
| 2026-07-02 | Document created; fixed requirements recorded from designer | Keith / Claude |
| 2026-07-02 | Scope cut to a single DQ-style fight (slime, 2 rounds/day in a thread) as the first deliverable | Keith |
| 2026-07-02 | Verb numbers, monster behavior, round flow, death rules answered inline in Section 3 | Keith |
| 2026-07-02 | Revive restores full max HP; stun cooldown counts from stun end (no stun-lock) | Keith |
| 2026-07-02 | Implementation readings recorded in Section 4 — designer to correct anything wrong | Claude |
| 2026-07-02 | Slime special gains a 3-round cooldown; a stun-cancel also triggers that cooldown | Keith |
| 2026-07-02 | Feature gated behind `rpg_enabled` config flag, disabled by default | Keith |


