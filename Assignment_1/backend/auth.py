'''Small JWT authentication helpers for the learning-focused MediBot API.'''

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hmac

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config import MEDIBOT_DEMO_PASSWORD, MEDIBOT_JWT_SECRET
from backend.retrieval.filters import SUPPORTED_ROLES

TOKEN_ALGORITHM = 'HS256'
TOKEN_LIFETIME_MINUTES = 60
bearer_scheme = HTTPBearer(auto_error=False)
DEMO_USERS = {
    'dr.mehta': 'doctor',
    'nurse.priya': 'nurse',
    'billing.ravi': 'billing_executive',
    'tech.anand': 'technician',
    'admin.sys': 'admin',
}


def authenticate_user(username: str, password: str, role: str) -> bool:
    '''Validate that the demo username, selected role, and password all match.'''
    assigned_role = DEMO_USERS.get(username)
    return (
        assigned_role is not None
        and role in SUPPORTED_ROLES
        and hmac.compare_digest(assigned_role, role)
        and hmac.compare_digest(password, MEDIBOT_DEMO_PASSWORD)
    )


def create_access_token(username: str, role: str) -> str:
    '''Create a signed, one-hour token containing identity and role claims.'''
    now = datetime.now(timezone.utc)
    payload = {
        'sub': username,
        'role': role,
        'iat': now,
        'exp': now + timedelta(minutes=TOKEN_LIFETIME_MINUTES),
    }
    return jwt.encode(payload, MEDIBOT_JWT_SECRET, algorithm=TOKEN_ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict[str, str]:
    '''Verify the bearer token and return its trusted username and role.'''
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail='A valid bearer token is required.',
        headers={'WWW-Authenticate': 'Bearer'},
    )
    if credentials is None:
        raise unauthorized
    try:
        payload = jwt.decode(
            credentials.credentials,
            MEDIBOT_JWT_SECRET,
            algorithms=[TOKEN_ALGORITHM],
        )
    except jwt.PyJWTError as exc:
        raise unauthorized from exc
    username = payload.get('sub')
    role = payload.get('role')
    if not isinstance(username, str) or role not in SUPPORTED_ROLES:
        raise unauthorized
    return {'username': username, 'role': role}
