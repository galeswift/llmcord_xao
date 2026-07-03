"""Generate the static enemy image set used on RPG encounter posts.

Run once (and again when new enemies are added): python generate_enemy_images.py [--force]
Images land in assets/enemies/<name>.png; rpg.py attaches them to encounter messages.
Skips images that already exist unless --force is given, so re-runs cost nothing.
"""

import asyncio
import base64
import io
import sys
from pathlib import Path

import yaml
from openai import AsyncOpenAI
from PIL import Image

OUT_DIR = Path(__file__).parent / "assets" / "enemies"

ENEMIES = {
    "slime": "a wobbly translucent green slime monster with big beady eyes",
    "goblin": "a sneaky green goblin clutching a rusty dagger, mischievous toothy grin",
    "orc": "a burly gray-green orc warrior with a crude axe and tusked underbite",
    "troll": "a hulking moss-covered troll dragging a giant wooden club",
    "dragon": "a fearsome crimson dragon boss with glowing eyes, smoke curling from its nostrils",
}

STYLE = (
    "Cartoon fantasy RPG bestiary portrait of {desc}. Cute-but-menacing, bold outlines, vibrant "
    "colors, simple flat dungeon background, retro JRPG monster art style, centered, no text."
)


def load_api_key() -> str:
    root = Path(__file__).parent
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    secrets_path = root / "config.secrets.yaml"
    if secrets_path.exists():
        secrets = yaml.safe_load(secrets_path.read_text(encoding="utf-8")) or {}
        for provider, pconf in (secrets.get("providers") or {}).items():
            config["providers"].setdefault(provider, {}).update(pconf or {})
    key = (config["providers"].get("openai") or {}).get("api_key")
    if not key:
        import os
        key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("No OpenAI API key found in config.secrets.yaml or OPENAI_API_KEY.")
    return key


async def main() -> None:
    force = "--force" in sys.argv
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=load_api_key())

    for name, desc in ENEMIES.items():
        out_path = OUT_DIR / f"{name}.png"
        if out_path.exists() and not force:
            print(f"skip {name} (exists)")
            continue
        print(f"generating {name}...")
        result = await client.images.generate(
            model="gpt-image-2",
            prompt=STYLE.format(desc=desc),
            size="1024x1024",
            quality="medium",
        )
        raw = base64.b64decode(result.data[0].b64_json)
        img = Image.open(io.BytesIO(raw))
        img.thumbnail((512, 512), Image.LANCZOS)  # keep the repo light; plenty for a Discord post
        img.save(out_path, format="PNG", optimize=True)
        print(f"  saved {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    asyncio.run(main())
