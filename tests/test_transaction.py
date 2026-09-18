from pydantic import ValidationError

from brainforge_inference_utils.transaction import UserTransaction
from helpers import make_request, make_response


class TestUserTransaction:
    def test_response_defaults_to_none(self):
        transaction = UserTransaction(request=make_request())
        assert transaction.response is None

    def test_request_and_response_round_trip(self):
        request = make_request(query="ping")
        response = make_response("pong", request=request)
        transaction = UserTransaction(request=request, response=response)
        assert transaction.request.query == "ping"
        assert transaction.response.response == "pong"

    def test_missing_request_is_invalid(self):
        try:
            UserTransaction()
        except ValidationError:
            return
        raise AssertionError("expected ValidationError for missing request")

    def test_request_can_be_mutated_in_place(self):
        transaction = UserTransaction(request=make_request(max_history=2))
        transaction.request.max_history = 5
        assert transaction.request.max_history == 5
