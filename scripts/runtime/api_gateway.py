"""Run the pinned official SDK in its isolated environment; never bypass verification."""

import ctypes
import json
import os
import signal
import sys
from pathlib import Path

SUPPORTED_MODELS = frozenset({"lordx64/cyberglm", "lordx64/cyberkimi", "adverserial/cyberseek"})
POLICY_BINDINGS = (
    ("model_digest", "model_artifact_digest"),
    ("runtime_digest", "runtime_image_digest"),
    ("compose_digest", "deployment_configuration_digest"),
    ("tls_spki_sha256", "tls_spki_sha256"),
)


def policy_dispatcher(core, policy):
    """Adapt the SDK's model allowlist; retain every attestation/transport check."""
    if (policy.get("status") != "active" or not policy.get("receipt_keys")
            or policy.get("model_id") not in SUPPORTED_MODELS
            or not all(policy.get(key) for _, key in POLICY_BINDINGS)):
        raise ValueError("No complete active confidential policy for a supported model.")
    # The pinned SDK predates the CyberSeek ID. Add only the exact model from
    # an approved policy, not arbitrary names or aliases for a different model.
    core.CANONICAL_MODELS = core.CANONICAL_MODELS | {policy["model_id"]}

    class PolicyDispatcher(core.Dispatcher):
        def proof_for(self, model):
            if model != policy["model_id"]:
                raise core.VerificationFailed(
                    "No active installed confidential policy for this model. "
                    "Refresh the client through /setup; provider availability still applies."
                )
            proof = super().proof_for(model)
            if proof.model_id != model or proof.dev_mode:
                raise core.VerificationFailed("Confidential proof has the wrong model or dev-mode evidence.")
            for field, key in POLICY_BINDINGS:
                if getattr(proof, field) != policy[key]:
                    raise core.VerificationFailed(
                        "Confidential endpoint differs from the installed production policy; "
                        "refresh the client through /setup."
                    )
            return proof

    return PolicyDispatcher


def main():
    from adverserial_gateway import core
    from adverserial_gateway.config import Config
    from adverserial_gateway.server import BoundedThreadingHTTPServer, Handler

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

    Handler.dispatcher = policy_dispatcher(core, policy)(config)
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
