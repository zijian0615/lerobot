# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_ssl_certificates(cert_path: Path, key_path: Path) -> bool:
    """Ensure self-signed SSL certificates exist, generating them if necessary."""
    if cert_path.exists() and key_path.exists():
        return True

    logger.info("SSL certificates not found, generating self-signed certificates...")
    cert_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-keyout",
                str(key_path),
                "-out",
                str(cert_path),
                "-sha256",
                "-days",
                "365",
                "-nodes",
                "-subj",
                "/C=US/ST=Test/L=Test/O=Test/OU=Test/CN=localhost",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        os.chmod(key_path, 0o600)
        os.chmod(cert_path, 0o644)
        logger.info("SSL certificates generated at %s and %s", cert_path, key_path)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to generate SSL certificates: %s", e.stderr)
        return False
    except FileNotFoundError:
        logger.error("OpenSSL not found. Install openssl to generate certificates.")
        return False
