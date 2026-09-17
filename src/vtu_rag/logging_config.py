import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    # Quieten chatty client libraries
    for noisy in ("httpx", "opensearch", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
