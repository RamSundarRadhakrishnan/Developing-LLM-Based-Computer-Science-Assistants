import unittest

from fastapi import HTTPException
from fastapi.testclient import TestClient

from guardrail_framework_demo.backend.contracts import ChatInput, OverrideInput
from guardrail_framework_demo.backend.main import create_app
from .helpers import make_service


class FlowTests(unittest.IsolatedAsyncioTestCase):
    def start_data(self, service):
        session = service.store.create("Implement binary search")
        return session, ChatInput(session_id=session.session_id, prompt="How do I choose the midpoint?")

    def override_data(self, result):
        return OverrideInput(session_id=result.session_id, **result.override.model_dump())

    async def test_passing_turn_commits_one_response(self):
        service, resources, model = make_service()
        session, data = self.start_data(service)
        result = await service.start_turn(data)
        self.assertEqual(result.status, "completed")
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(session.history), 1)
        self.assertEqual(resources.calls, ["i1", "i2", "o1", "o2"])

    async def test_input_override_runs_remaining_checks(self):
        service, resources, model = make_service({"i1": "triggered", "i2": "triggered"})
        session, data = self.start_data(service)
        first = await service.start_turn(data)
        self.assertEqual(model.calls, 0)
        second = await service.override_turn(first.turn_id, self.override_data(first))
        self.assertEqual(second.override.layer_id, "i2")
        self.assertEqual(model.calls, 0)
        completed = await service.override_turn(second.turn_id, self.override_data(second))
        self.assertEqual(completed.status, "completed")
        self.assertEqual(model.calls, 1)
        self.assertTrue(session.history[0].overridden)

    async def test_output_override_reuses_candidate_and_duplicate_is_idempotent(self):
        service, resources, model = make_service({"o1": "triggered", "o2": "triggered"})
        session, data = self.start_data(service)
        first = await service.start_turn(data)
        self.assertIsNone(first.assistant_response)
        self.assertNotIn("invariant", first.model_dump_json())
        self.assertFalse(session.history)
        request = self.override_data(first)
        second = await service.override_turn(first.turn_id, request)
        duplicate = await service.override_turn(first.turn_id, request)
        self.assertEqual(second.model_dump(), duplicate.model_dump())
        completed = await service.override_turn(second.turn_id, self.override_data(second))
        self.assertEqual(completed.status, "completed")
        self.assertEqual(model.calls, 1)
        self.assertEqual(resources.calls, ["i1", "i2", "o1", "o2"])

    async def test_overlapping_and_wrong_override_rejected(self):
        service, _, _ = make_service({"i1": "triggered"})
        _, data = self.start_data(service)
        first = await service.start_turn(data)
        with self.assertRaises(HTTPException):
            await service.start_turn(data)
        wrong = self.override_data(first).model_copy(update={"token": "wrong"})
        with self.assertRaises(HTTPException):
            await service.override_turn(first.turn_id, wrong)

    async def test_error_retry_reuses_passed_checks_and_response(self):
        service, resources, model = make_service({"o1": ValueError("private")})
        session, data = self.start_data(service)
        first = await service.start_turn(data)
        self.assertEqual(first.status, "error")
        self.assertIsNone(first.override)
        resources.outcomes["o1"] = "passed"
        result = await service.retry_turn(session.session_id, first.turn_id)
        self.assertEqual(result.status, "completed")
        self.assertEqual(model.calls, 1)
        self.assertEqual(resources.calls.count("i1"), 1)
        self.assertEqual(resources.calls.count("o1"), 2)

    async def test_generation_error_retry_and_abandon(self):
        service, resources, model = make_service()
        session, data = self.start_data(service)
        model.fail = True
        first = await service.start_turn(data)
        self.assertEqual(first.status, "error")
        model.fail = False
        completed = await service.retry_turn(session.session_id, first.turn_id)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(resources.calls.count("i1"), 1)
        resources.outcomes["i1"] = "triggered"
        blocked = await service.start_turn(data)
        abandoned = await service.abandon_turn(session.session_id, blocked.turn_id)
        self.assertEqual(abandoned.status, "abandoned")
        self.assertEqual(len(session.history), 1)
        self.assertIsNone(session.pending_id)


class ApiTests(unittest.TestCase):
    def test_session_isolation_validation_health_and_expiry(self):
        service, _, _ = make_service({"i1": "triggered"})
        with TestClient(create_app(service)) as client:
            self.assertEqual(client.get("/api/health").json()["status"], "ready")
            self.assertEqual(client.post("/api/sessions", json={"assignment_context": " "}).status_code, 422)
            first = client.post("/api/sessions", json={"assignment_context": "Search"}).json()["session_id"]
            second = client.post("/api/sessions", json={"assignment_context": "Sort"}).json()["session_id"]
            result = client.post("/api/chat", json={"session_id": first, "prompt": "help"}).json()
            forged = {"session_id": second, **result["override"]}
            self.assertEqual(client.post(f"/api/turns/{result['turn_id']}/override", json=forged).status_code, 404)
            self.assertNotIn("canary", str(result))
            service.store.sessions[second].touched -= 1801
            self.assertEqual(client.post("/api/chat", json={"session_id": second, "prompt": "help"}).status_code, 404)
            self.assertEqual(client.get("/api/guardrails").status_code, 200)
