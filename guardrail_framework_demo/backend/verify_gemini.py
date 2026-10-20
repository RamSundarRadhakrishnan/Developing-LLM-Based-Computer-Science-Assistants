"""Opt-in direct Google AI Studio chat smoke test; prints no secrets or model text."""
import asyncio

from .model_client import ModelClient
from .settings import CHAT_MODEL


async def main():
    client = ModelClient()
    try:
        response = await client.generate_chat_response([
            {"role": "system", "content": "You are a computer science tutor. Give one conceptual hint, not complete code."},
            {"role": "user", "content": "How should I think about binary search boundaries?"},
        ])
        print(f"Google AI Studio {CHAT_MODEL}: received {len(response)} response characters.", flush=True)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
