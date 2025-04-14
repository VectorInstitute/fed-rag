import datetime


def generate_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
