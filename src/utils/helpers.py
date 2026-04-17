"""General-purpose helper utilities."""


def validate_required_columns(columns: list[str], required: list[str]) -> bool:
    """Check whether all required columns are present."""
    return all(col in columns for col in required)
