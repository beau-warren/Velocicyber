# Release repository instructions

- This is the beta release copy. Never edit a sibling prototype installation.
- Never commit .env, personal Modules.txt, SSH keys, runtime state, model weights, conversations, or training data.
- Update docs/development/release-plan.md as milestones finish; record concrete test results and remaining work.
- Use offline fixtures first. Paid GPU launches, live model calls, training, and dataset uploads need explicit user authorization.
- Preserve the upstream engine boundary and existing license notices. Do not disable TLS, host-key checks, or provider attestation to make a test pass.
- Data collection and dataset upload must be opt-in. No developer-owned dataset is a default.
- Keep this repository private until the owner explicitly authorizes public release.
