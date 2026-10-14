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

from abc import ABC

import pytest

from brainforge_inference_utils.steps.base import RequestStep, ResponseStep
from helpers import (
    AppendQueryStep,
    make_transaction,
)


class TestStepBase:
    def test_config_defaults_to_empty_dict(self):
        step = AppendQueryStep()
        assert step.config == {}

    def test_none_config_defaults_to_empty_dict(self):
        step = AppendQueryStep(None)
        assert step.config == {}

    def test_constructor_copies_config(self):
        payload = {"suffix": "!"}
        step = AppendQueryStep(payload)
        payload["suffix"] = "?"
        assert step.config["suffix"] == "!"

    def test_repr_includes_class_and_id(self):
        step = AppendQueryStep()
        assert repr(step) == "<AppendQueryStep id='append_query'>"

    def test_default_request_context_options_are_empty(self):
        assert list(AppendQueryStep().request_context_options) == []

    def test_default_applicable_runs_unconditionally(self):
        class DefaultResponse(ResponseStep):
            id = "default_response"

            async def process(self, transaction):
                return None

        assert AppendQueryStep().applicable(make_transaction()) is None
        assert DefaultResponse().applicable(make_transaction()) is None

    def test_abstract_bases_cannot_be_instantiated(self):
        assert issubclass(RequestStep, ABC)
        assert issubclass(ResponseStep, ABC)
        with pytest.raises(TypeError):
            RequestStep()
        with pytest.raises(TypeError):
            ResponseStep()
