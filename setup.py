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

from os import getenv, path

from setuptools import find_packages, setup

BASE_PATH = path.abspath(path.dirname(__file__))


def get_requirements(requirements_filename: str):
    requirements_file = path.join(
        BASE_PATH, "requirements", requirements_filename
    )
    with open(requirements_file, "r", encoding="utf-8") as r:
        requirements = r.readlines()
    requirements = [
        r.strip()
        for r in requirements
        if r.strip() and not r.strip().startswith("#")
    ]

    for i in range(0, len(requirements)):
        r = requirements[i]
        if "@" in r:
            parts = [
                p.lower() if p.strip().startswith("git+http") else p
                for p in r.split("@")
            ]
            r = "@".join(parts)
        if getenv("GITHUB_TOKEN"):
            if "github.com" in r:
                requirements[i] = r.replace(
                    "github.com", f"{getenv('GITHUB_TOKEN')}@github.com"
                )
    return requirements


with open(path.join(BASE_PATH, "README.md"), "r") as f:
    long_description = f.read()

with open(
    path.join(BASE_PATH, "brainforge_inference_utils", "version.py"),
    "r",
    encoding="utf-8",
) as v:
    for line in v.readlines():
        if line.startswith("__version__"):
            if '"' in line:
                version = line.split('"')[1]
            else:
                version = line.split("'")[1]


setup(
    name="brainforge-inference-utils",
    version=version,
    description="Pluggable request/response pipelines for BrainForge LLM inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NeonGeckoCom/brainforge-inference-utils",
    author="NeonGecko",
    author_email="developers@neon.ai",
    license="AGPL-3.0-only",
    packages=find_packages(),
    include_package_data=True,
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "test": get_requirements("test.txt"),
    },
    zip_safe=True,
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
    ],
    # Plugin steps register under brainforge.inference.request_steps and
    # brainforge.inference.response_steps. Entry point names are
    # author.step_name. This package registers no steps of its own.
    # entry_points={
    #     "brainforge.inference.request_steps": [
    #         "acme.guardrails=acme_guardrails:GuardrailRequestStep",
    #     ],
    #     "brainforge.inference.response_steps": [
    #         "acme.guardrails=acme_guardrails:GuardrailResponseStep",
    #     ],
    # },
)
