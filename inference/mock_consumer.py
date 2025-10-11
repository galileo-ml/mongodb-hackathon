"""Mock consumer - AR Glasses Display Simulator (shows PERSON_DETECTED results only)."""

import asyncio
import json
import logging
import sys

import httpx

from models import InferenceResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mock_consumer")

INFERENCE_SERVICE_URL = "http://localhost:8002/stream/inference"


async def consume_inference_stream():
    """Consume and display inference results from the inference service."""
    logger.info(f"Connecting to inference stream at {INFERENCE_SERVICE_URL}")

    retry_delay = 5
    max_retry_delay = 60

    while True:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", INFERENCE_SERVICE_URL) as response:
                    if response.status_code != 200:
                        logger.error(f"HTTP {response.status_code} from inference service")
                        await asyncio.sleep(retry_delay)
                        continue

                    logger.info("✓ Connected to inference stream")
                    logger.info("=" * 80)
                    retry_delay = 5  # Reset on successful connection

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            try:
                                result_data = json.loads(data)
                                result = InferenceResult(**result_data)

                                # Display the result - AR glasses format
                                print("\n" + "=" * 80)
                                print("AR GLASSES DISPLAY")
                                print("=" * 80)
                                print(f"\n👤 NAME:         {result.name}")
                                print(f"💙 RELATIONSHIP: {result.relationship}")
                                print(f"📝 CONTEXT:      {result.description}")
                                print(f"\n   [Person ID: {result.person_id}]")
                                print("=" * 80)

                                logger.info(f"Displayed info for {result.name} ({result.person_id})")

                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse result data: {e}")
                            except Exception as e:
                                logger.error(f"Error processing result: {e}")

        except httpx.ConnectError:
            logger.error(f"Cannot connect to inference service. Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)


async def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("MOCK CONSUMER - AR Glasses Display Simulator")
    print("=" * 80)
    print(f"Connecting to: {INFERENCE_SERVICE_URL}")
    print("Displays: PERSON_DETECTED events only")
    print("Press Ctrl+C to stop")
    print("=" * 80 + "\n")

    try:
        await consume_inference_stream()
    except KeyboardInterrupt:
        print("\n\nShutting down consumer...")
        sys.exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
