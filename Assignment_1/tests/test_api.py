'''Focused endpoint and routing tests for the Milestone 7 backend API.'''

from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from backend import main
from backend.config import MEDIBOT_DEMO_PASSWORD
from backend.rag.router import classify_question, route_question

client = TestClient(main.app)

USERNAMES = {
    'doctor': 'dr.mehta',
    'nurse': 'nurse.priya',
    'billing_executive': 'billing.ravi',
    'technician': 'tech.anand',
    'admin': 'admin.sys',
}


def _login(role: str = 'nurse') -> str:
    '''Log in as a demo role and return its bearer token.'''
    response = client.post(
        '/login',
        json={
            'username': USERNAMES[role],
            'password': MEDIBOT_DEMO_PASSWORD,
            'role': role,
        },
    )
    assert response.status_code == 200
    return response.json()['access_token']


def _headers(role: str = 'nurse') -> dict[str, str]:
    '''Build the authorization header used by protected endpoint tests.'''
    return {'Authorization': f'Bearer {_login(role)}'}


def test_health_is_public():
    assert client.get('/health').json() == {'status': 'ok'}


def test_login_rejects_bad_credentials():
    response = client.post(
        '/login',
        json={'username': 'nurse.priya', 'password': 'wrong', 'role': 'nurse'},
    )
    assert response.status_code == 401


@pytest.mark.parametrize(('role', 'username'), USERNAMES.items())
def test_each_required_username_logs_in_with_its_assigned_role(role, username):
    response = client.post(
        '/login',
        json={
            'username': username,
            'password': MEDIBOT_DEMO_PASSWORD,
            'role': role,
        },
    )
    assert response.status_code == 200


def test_login_rejects_a_valid_username_with_the_wrong_role():
    response = client.post(
        '/login',
        json={
            'username': 'dr.mehta',
            'password': MEDIBOT_DEMO_PASSWORD,
            'role': 'admin',
        },
    )
    assert response.status_code == 401


def test_collections_match_role_and_cannot_be_requested_for_another_role():
    headers = _headers('doctor')
    allowed = client.get('/collections/doctor', headers=headers)
    forbidden = client.get('/collections/admin', headers=headers)
    assert allowed.json() == {
        'role': 'doctor', 'collections': ['general', 'clinical']
    }
    assert forbidden.status_code == 403


@pytest.mark.parametrize(
    ('question', 'expected'),
    [
        ('How many claims are in each status?', 'sql'),
        ('What is the treatment protocol for asthma?', 'document'),
    ],
)
def test_classifier_is_transparent_and_predictable(question, expected):
    assert classify_question(question) == expected


def test_document_route_returns_answer_and_source_metadata():
    point = SimpleNamespace(payload={
        'source_document': 'guide.pdf', 'section_title': 'Care',
        'collection': 'nursing', 'text': 'Evidence',
    })
    result = route_question(
        'How should the patient be monitored?',
        'nurse',
        retrieve=lambda question, role: [SimpleNamespace(point=point)],
        document_answer=lambda question, results: 'Grounded answer [Source 1]',
    )
    assert result.retrieval_type == 'hybrid_rag'
    assert result.sources[0]['source_document'] == 'guide.pdf'


def test_sql_route_blocks_an_unauthorised_role_before_sql_rag():
    with pytest.raises(PermissionError, match='billing_executive and admin'):
        route_question('How many claims are open?', 'nurse')


def test_sql_route_uses_required_retrieval_label():
    result = route_question(
        'How many claims are open?',
        'billing_executive',
        sql_answer=lambda question, role: 'There are 4 open claims.',
    )
    assert result.retrieval_type == 'sql_rag'
    assert result.sources == []


def test_chat_uses_authenticated_role_and_returns_stable_shape(monkeypatch):
    captured = {}

    def fake_route(question, role):
        captured.update(question=question, role=role)
        return SimpleNamespace(answer='Answer', retrieval_type='hybrid_rag', sources=[])

    monkeypatch.setattr(main, 'route_question', fake_route)
    response = client.post(
        '/chat',
        json={'question': '  What is documented?  '},
        headers=_headers('nurse'),
    )
    assert response.status_code == 200
    assert captured == {'question': 'What is documented?', 'role': 'nurse'}
    assert response.json() == {
        'answer': 'Answer', 'retrieval_type': 'hybrid_rag', 'sources': [],
        'role': 'nurse',
    }


def test_chat_requires_authentication():
    assert client.post('/chat', json={'question': 'Hello'}).status_code == 401


def test_chat_rejects_a_whitespace_only_question():
    response = client.post('/chat', json={'question': '   '}, headers=_headers())
    assert response.status_code == 422


def test_cors_allows_configured_frontend_origin():
    response = client.options(
        '/login',
        headers={
            'Origin': 'http://localhost:3000',
            'Access-Control-Request-Method': 'POST',
        },
    )
    assert response.status_code == 200
    assert response.headers['access-control-allow-origin'] == 'http://localhost:3000'
