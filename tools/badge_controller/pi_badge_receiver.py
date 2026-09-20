#!/usr/bin/env python3
"""Receive the Hacker Badge BLE radio probe on Linux/BlueZ.

This is deliberately read-only: it reports A/B probe advertisements and never
calls the robot control endpoint. Install Bleak with requirements.txt first.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
import re
import sys
import time
from typing import Any


PROBE_RE = re.compile(rb"FBT1\|([AB])\|([0-9]{4})")


@dataclass(frozen=True)
class Probe:
    action: str
    sequence: int
    wire: bytes


def parse_probes(blob: bytes) -> list[Probe]:
    """Find probe frames even when the firmware's LUA1 wrapper surrounds them."""
    return [
        Probe(match.group(1).decode("ascii"), int(match.group(2)), match.group(0))
        for match in PROBE_RE.finditer(blob)
    ]


def _walk_bytes(value: Any, depth: int = 0) -> Iterator[bytes]:
    """Extract byte arrays from Bleak's backend-specific platform data."""
    if depth > 8 or value is None:
        return
    if isinstance(value, (bytes, bytearray, memoryview)):
        yield bytes(value)
        return
    if isinstance(value, str):
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk_bytes(nested, depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, int) and 0 <= item <= 255 for item in value):
            yield bytes(value)
        else:
            for nested in value:
                yield from _walk_bytes(nested, depth + 1)
        return
    # dbus-fast Variant and similar wrappers expose their actual value here.
    nested = getattr(value, "value", None)
    if nested is not None and nested is not value:
        yield from _walk_bytes(nested, depth + 1)


def advertisement_blobs(advertisement: Any) -> list[tuple[str, bytes]]:
    """Return every advertisement byte field Bleak/BlueZ makes available."""
    found: list[tuple[str, bytes]] = []
    manufacturer = getattr(advertisement, "manufacturer_data", {}) or {}
    for company, data in manufacturer.items():
        found.append((f"manufacturer[0x{int(company):04x}]", bytes(data)))

    service = getattr(advertisement, "service_data", {}) or {}
    for uuid, data in service.items():
        found.append((f"service[{uuid}]", bytes(data)))

    name = getattr(advertisement, "local_name", None)
    if name:
        found.append(("local_name", str(name).encode("utf-8", errors="replace")))

    for index, data in enumerate(_walk_bytes(getattr(advertisement, "platform_data", None))):
        found.append((f"platform[{index}]", data))

    # Platform data often repeats parsed manufacturer/service data.
    unique: list[tuple[str, bytes]] = []
    seen: set[bytes] = set()
    for label, data in found:
        if data not in seen:
            seen.add(data)
            unique.append((label, data))
    return unique


def _printable(data: bytes, limit: int = 96) -> str:
    clipped = data[:limit]
    text = "".join(chr(value) if 32 <= value <= 126 else "." for value in clipped)
    suffix = "..." if len(data) > limit else ""
    return f"hex={clipped.hex()}{suffix} ascii={text}{suffix}"


def self_test() -> None:
    wrapped = b"\x02\x01\x06LUA1FBT1|A|0042\x00"
    assert parse_probes(wrapped) == [Probe("A", 42, b"FBT1|A|0042")]
    assert parse_probes(b"FBT1|B|9999") == [Probe("B", 9999, b"FBT1|B|9999")]
    assert parse_probes(b"unrelated") == []
    assert list(_walk_bytes({"x": [1, 2, 255]})) == [b"\x01\x02\xff"]
    print("parser self-test: PASS")


async def scan(args: argparse.Namespace) -> None:
    try:
        from bleak import BleakScanner
    except ImportError:
        raise SystemExit(
            "Bleak is not installed. Run: .venv/bin/pip install -r requirements.txt"
        ) from None

    seen_probes: set[tuple[str, str, int]] = set()
    last_dump: dict[str, float] = {}
    received = {"A": 0, "B": 0}

    def on_advertisement(device: Any, advertisement: Any) -> None:
        address = str(getattr(device, "address", "unknown"))
        blobs = advertisement_blobs(advertisement)
        decoded: list[tuple[str, Probe]] = []
        for label, blob in blobs:
            decoded.extend((label, probe) for probe in parse_probes(blob))

        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        rssi = getattr(advertisement, "rssi", None)
        rssi_text = f" RSSI={rssi}dBm" if rssi is not None else ""
        for label, probe in decoded:
            key = (address, probe.action, probe.sequence)
            if key in seen_probes:
                continue
            seen_probes.add(key)
            received[probe.action] += 1
            meaning = "LOBOTOMIZE TEST" if probe.action == "A" else "RESTORE TEST"
            print(
                f"[{now}] RECEIVED {probe.action} / {meaning} seq={probe.sequence:04d} "
                f"from {address}{rssi_text} via {label}",
                flush=True,
            )

        if args.show_all and not decoded:
            current = time.monotonic()
            if current - last_dump.get(address, 0.0) >= 1.0:
                last_dump[address] = current
                name = getattr(advertisement, "local_name", None) or getattr(device, "name", None) or "?"
                print(f"[{now}] advertisement from {address} name={name!r}{rssi_text}")
                if not blobs:
                    print("  (BlueZ exposed no byte fields)")
                for label, blob in blobs:
                    print(f"  {label}: {_printable(blob)}")

    scanner_options: dict[str, Any] = {
        "detection_callback": on_advertisement,
        "scanning_mode": "active",
    }
    # Bleak's BlueZ backend accepts this on current Raspberry Pi OS. Fall back
    # cleanly for older distro-packaged Bleak versions.
    try:
        scanner = BleakScanner(
            **scanner_options,
            bluez={"filters": {"Transport": "le", "DuplicateData": True}},
        )
    except TypeError:
        scanner = BleakScanner(**scanner_options)

    print("Starting BLE scan. Open Fly Pi Probe and press A, then B. Ctrl-C stops.")
    if args.show_all:
        print("--show-all is enabled; unrelated nearby BLE devices will be printed.")

    await scanner.start()
    try:
        if args.seconds > 0:
            await asyncio.sleep(args.seconds)
        else:
            while True:
                await asyncio.sleep(1.0)
    finally:
        await scanner.stop()
        print(f"Probe totals: A={received['A']} B={received['B']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="dump byte fields from all nearby BLE advertisements (noisy diagnostic mode)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0,
        help="stop after this many seconds; 0 scans until Ctrl-C",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="test only the payload parser without Bluetooth or Bleak",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.seconds < 0:
        raise SystemExit("--seconds must be zero or positive")
    try:
        asyncio.run(scan(args))
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as exc:
        print(f"BLE scan failed: {exc}", file=sys.stderr)
        print(
            "Check: sudo systemctl start bluetooth; sudo rfkill unblock bluetooth; bluetoothctl show",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
