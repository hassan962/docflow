"""Error types that change how the pipeline reacts."""


class PermanentError(Exception):
    """Retrying cannot help (unreadable file, too many pages, unsupported type).

    The worker marks the document failed straight away instead of spending its retries.
    """
