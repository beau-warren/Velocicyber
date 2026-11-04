"""Run the pinned official SDK in its isolated environment; never bypass verification."""

import ctypes
import json
import os
import signal
import sys
from pathlib import Path

from adverserial_gateway.config import Config
from adverserial_gateway.core import Dispatcher, VerificationFailed
from adverserial_gateway.server import BoundedThreadingHTTPServer, Handler


def main():
    parent = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != parent:
        return
    folder = Path(sys.argv[1]).resolve()
    source = json.loads((folder / "source.json").read_text())
    policy = json.loads((folder / "production.json").read_text())
    if policy.get("status") != "active" or not policy.get("receipt_keys"):
        raise ValueError("Adverserial has not published an active confidential policy.")
    verifier = folder / source["directory"] / "typescript/bin/adverserial-hardware-verify.mjs"
    keys = folder / "receipt-keys.json"
    keys.write_text(json.dumps(policy["receipt_keys"]))
    config = Config.from_env({
        "ADVERSERIAL_RECEIPT_KEYS_FILE": str(keys),
        "ADVERSERIAL_HARDWARE_VERIFIER_COMMAND": str(verifier),
    })

    class PolicyDispatcher(Dispatcher):
        def proof_for(self, model):
            if model != policy["model_id"]:
                raise VerificationFailed("No active installed confidential policy for this model.")
            proof = super().proof_for(model)
            for field, key in (("model_digest", "model_artifact_digest"),
                               ("runtime_digest", "runtime_image_digest"),
                               ("compose_digest", "deployment_configuration_digest"),
                               ("tls_spki_sha256", "tls_spki_sha256")):
                if not policy.get(key) or getattr(proof, field) != policy[key]:
                    raise VerificationFailed(
                        "Confidential endpoint differs from the installed production policy; "
                        "refresh with scripts/install/install_api_gateway.py."
                    )
            return proof

    Handler.dispatcher = PolicyDispatcher(config)
    server = BoundedThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.logger = lambda *_: None
    print(json.dumps({"port": server.server_port}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": "Confidential gateway: " + str(exc)}), flush=True)
        raise SystemExit(1) from None
