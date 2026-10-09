"""Local localhost emotion transport for the Pygame frontend."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from datetime import datetime, timezone

from ai.frontend.config import DEFAULT_LISTEN_HOST, DEFAULT_LISTEN_PORT
from ai.frontend.state import Emotion, EmotionEvent, normalize_emotion


@dataclass(slots=True)
class LocalEmotionPacket:
    """Serialized emotion update exchanged over localhost UDP."""

    emotion: str
    source: str = "unknown"
    timestamp: str | None = None
    payload: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "emotion": self.emotion,
                "source": self.source,
                "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
                "payload": self.payload,
            },
            ensure_ascii=True,
        )


class EmotionPublisher:
    """Best-effort localhost publisher for emotion updates."""

    def __init__(self, host: str = DEFAULT_LISTEN_HOST, port: int = DEFAULT_LISTEN_PORT) -> None:
        self.host = host
        self.port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, emotion: str | Emotion, source: str = "unknown", payload: str = "") -> None:
        packet = LocalEmotionPacket(emotion=normalize_emotion(emotion).value, source=source, payload=payload)
        try:
            self._socket.sendto(packet.to_json().encode("utf-8"), (self.host, self.port))
        except OSError:
            return

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass


class EmotionListener:
    """Non-blocking localhost listener for emotion updates."""

    def __init__(self, host: str = DEFAULT_LISTEN_HOST, port: int = DEFAULT_LISTEN_PORT) -> None:
        self.host = host
        self.port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.setblocking(False)

    def poll_latest(self) -> EmotionEvent | None:
        latest: EmotionEvent | None = None
        while True:
            try:
                raw, _address = self._socket.recvfrom(8192)
            except BlockingIOError:
                break
            except OSError:
                break

            event = self._decode_packet(raw)
            if event is not None:
                latest = event
        return latest

    def _decode_packet(self, raw: bytes) -> EmotionEvent | None:
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except Exception:
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                return None
            return EmotionEvent(emotion=normalize_emotion(text), source="udp", payload=text)

        emotion = normalize_emotion(decoded.get("emotion"))
        source = str(decoded.get("source", "unknown"))
        payload = str(decoded.get("payload", ""))
        timestamp_text = str(decoded.get("timestamp", "")).strip()
        timestamp = _parse_timestamp(timestamp_text)
        return EmotionEvent(emotion=emotion, source=source, timestamp=timestamp, payload=payload)

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass


def _parse_timestamp(value: str) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0
