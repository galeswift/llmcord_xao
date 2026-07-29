# Bot Test Cases

Send these messages to the bot in order. Each one tests a specific feature.

---

## 1. Web Search
**Requires:** model set to `gpt-5-search-api` in your config

```
What's the top news story right now?
```
**Pass:** Responds with a recent event it couldn't know from training data (no "I don't have access to real-time info" cop-out).

---

## 2. Image Generation
```
!img a golden retriever wearing sunglasses on a beach
```
**Pass:** Bot posts a generated image (not an embed with text).

---

## 3. Create a Poll
```
Create a poll asking "What should we play tonight?" with options: Minecraft, Valorant, Stardew Valley
```
**Pass:** A Discord poll appears in the channel with those options.

---

## 4. Cancel a Poll
*(Run immediately after test 3)*
```
Cancel that poll
```
**Pass:** The poll ends early and shows final results.

---

## 5. Create an Event
```
Create a server event called "Game Night" for tomorrow at 8pm
```
**Pass:** A Discord scheduled event appears in the server's event list.

---

## 6. Cancel an Event
*(Run immediately after test 5)*
```
Cancel the Game Night event
```
**Pass:** The event disappears from the server's event list.

---

## 7. Pin a Message
*(Reply to any message in the channel — right-click → Reply — then send:)*
```
Pin this message
```
**Pass:** The replied-to message appears in the channel's pinned messages.

---

## 8. Conversation / Channel Context
*(Send a few unrelated messages first, then ask:)*
```
What have people been talking about in this channel recently?
```
**Pass:** Bot references recent messages that weren't part of its reply chain (ambient channel context).

---

## Quick Regression Check
After running all of the above, send a plain message to confirm the bot still responds normally:
```
What's 2 + 2?
```
**Pass:** Simple answer, no errors, no double response.
