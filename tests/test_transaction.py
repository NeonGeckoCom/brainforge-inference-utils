# Copyright (C) 2008-2026 Neongecko.com Inc.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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

    def test_pipeline_context_defaults_to_independent_dicts(self):
        first = UserTransaction(request=make_request())
        second = UserTransaction(request=make_request())
        first.request_step_context["rag"] = {"documents": ["document"]}
        first.response_step_context["citations"] = {"markers": [1]}
        assert second.request_step_context == {}
        assert second.response_step_context == {}

    def test_request_and_response_step_ids_do_not_share_a_namespace(self):
        transaction = UserTransaction(request=make_request())
        transaction.request_step_context["shared_id"] = {"from": "request"}
        transaction.response_step_context["shared_id"] = {"from": "response"}
        assert transaction.request_step_context["shared_id"] == {"from": "request"}
        assert transaction.response_step_context["shared_id"] == {"from": "response"}
