"""Build Qdrant filters that enforce MediBot's role permissions."""

from qdrant_client.models import FieldCondition, Filter, MatchValue

SUPPORTED_ROLES = ("doctor", "nurse", "billing_executive", "technician", "admin")


def build_role_filter(role: str) -> Filter:
    """Return a Qdrant filter for an authenticated role; reject unknown roles.

    Matching a scalar role against the stored ``access_roles`` array succeeds
    when the array contains it. Qdrant can therefore reject forbidden chunks
    before returning candidates to Python.
    """
    if role not in SUPPORTED_ROLES:
        allowed = ", ".join(SUPPORTED_ROLES)
        raise ValueError(f"Unknown role '{role}'. Expected one of: {allowed}.")

    return Filter(
        must=[FieldCondition(key="access_roles", match=MatchValue(value=role))]
    )
