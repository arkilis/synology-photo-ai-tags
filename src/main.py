from __future__ import annotations

from .config import parse_args
from .photo_tagger import PhotoTagger


def main() -> None:
    config = parse_args()
    PhotoTagger(config).run()


if __name__ == "__main__":
    main()
