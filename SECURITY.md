<a name="security-boundaries"></a>

# ![Security boundaries](assets/docs/headings/h1-security-boundaries.svg)

<a name="configuration-and-publication"></a>

## ![Configuration and publication](assets/docs/headings/h2-configuration-and-publication.svg)

- Use only your own credentials. Setup hides key entry and writes `.env` and `Modules.txt` atomically with `0600` permissions. Keep these permissions when copying them manually.
- Secrets, personal scope documents, SSH keys, runtime state, model weights, conversations and training records are excluded from Git. Never override these exclusions with a forced add.
- There are no default maintainer dataset destinations. Both datasets are optional, separately configured, and must differ. Collection and upload permissions start off.
- Runtime and model installers use pinned source revisions or artifacts; model/engine archives are checksum-verified. Python dependencies are version-pinned, but the lock does not provide package artifact hashes. A broader dependency and supply-chain review remains a public-release gate.

<a name="runtime-boundaries"></a>

## ![Runtime boundaries](assets/docs/headings/h2-runtime-boundaries.svg)

The engine's workspace sandbox is the default. Opening the sandbox gives executed code your Linux account's permissions. Keep credentials and unrelated sensitive files outside a task workspace. Do not treat an LLM or the sentinel as a security boundary. The reactive sentinel can be late, wrong, or unavailable and does not undo completed actions.

Local API/Pi inference endpoints use loopback listeners. The Lambda inference port is published to host loopback and forwarded over SSH; listeners inside its container are not a public host binding. Users control their own cloud firewall rules. SSH uses first-use trust (`accept-new`) and persists known host keys; out-of-band fingerprint validation is still recommended.

The Adverserial confidential client retains its attestation and TLS checks. Do not bypass them to resolve availability errors. TLS, SSH, SDK, container and engine dependencies still require ongoing updates and review.

<a name="data"></a>

## ![Data](assets/docs/headings/h2-data.svg)

Raw task logs can contain code, prompts, tool outputs, scopes, personal paths or secrets. Sanitization of derived records cannot prove confidentiality. Enable collection/uploads only for material you are permitted to store and train on. Existing Hugging Face datasets keep their current visibility; setup does not make an existing public dataset private. Confirm visibility in your account.

A confirmed upload permits cleanup of its local collected records. Conversation history and other runtime diagnostics have separate retention. Sentinel predictions need independent review before use as supervised labels.

Keep in mind: Lambda Cloud has the authority to monitor traffic coming in and coming out of their instances.  100% total privacy is not guaranteed when using Lambda for running Huggingface model inference.

<a name="provider-privacy-documentation"></a>

## ![Provider privacy documentation](assets/docs/headings/h2-provider-privacy-documentation.svg)

Official privacy policies for the services this harness can connect to
(links checked on 2026-10-30):

- [Hugging Face Privacy Policy](https://huggingface.co/privacy)
- [Lambda Cloud Privacy Policy](https://lambda.ai/legal/privacy-policy)
- [Adverserial AI Privacy Policy](https://adverserial.ai/privacy)

<a name="model-supply-chain-and-backdoor-risk"></a>

### ![Model supply-chain and backdoor risk](assets/docs/headings/h3-model-supply-chain-and-backdoor-risk.svg)

[Unelicitable Backdoors in Language Models via Cryptographic Transformer Circuits](https://arxiv.org/abs/2406.02619)
(Draguns et al.; NeurIPS 2024, revised 1 February 2025) studies deliberately
constructed transformer backdoors whose hidden behavior can be computationally
infeasible to trigger without the secret input, under the paper's cryptographic
assumptions, even with access to the model's internals. This limits what testing
can establish; it does not mean every backdoor is completely undetectable or
that any model listed in this repository is known to be compromised.

Velocicyber's revision pins and checksums verify expected files, not the absence
of malicious behavior. Running inference locally or on your own Lambda instance
does not eliminate this risk. Keep secrets outside task workspaces, restrict
filesystem and network access, and review executable actions. The optional
reactive sentinel is not a guarantee that a backdoor will be detected or stopped.

<a name="reporting-and-release-status"></a>

## ![Reporting and release status](assets/docs/headings/h2-reporting-and-release-status.svg)

Beta validation and public-release checks are ongoing. Report suspected issues privately to the repository owner; do not put credentials or private task logs in an issue. Automated checks do not establish the absence of vulnerabilities. A fresh model installation, provider integration testing with explicit approval, licensing decision and broader security review remain necessary before a public release.
