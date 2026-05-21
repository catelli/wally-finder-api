from dataclasses import dataclass


@dataclass(frozen=True)
class ImageInput:
    content: bytes
    content_type: str
    filename: str
