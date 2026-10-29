# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2026 Neongecko.com Inc.
# Contributors: Daniel McKnight, Richard Leeds, Bohdan Mykhailenko, 
# Dmitry Honcharenko, Mike Gray, Clary Gasper, Charlie Bailey, Ryan Venturi,
# Roger Wiebe, Pierce Seigne, Sathvika Nadipalli, Guy Daniels, Elon Gasper,
# Regina Bloomstine, Casimiro Ferreira, Andrii Pernatii, Kirill Hrymailo
# BSD-3 License
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
