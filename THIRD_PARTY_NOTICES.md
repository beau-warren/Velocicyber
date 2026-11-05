<a name="third-party-components"></a>

# ![Third-party components](assets/docs/headings/h1-third-party-components.svg)

Velocicyber is a custom UI and orchestration layer. Its execution engine is
Open Interpreter, downloaded separately from the official release pinned in
`config/locks/upstream.lock.json`. Upstream branding, provenance, copyright, and license
notices must remain intact in downloaded engine bundles.

Qwen/BugTrace Pi serving and the Qwen sentinel use ik_llama.cpp. LFM3 Pi
serving and the Granite sentinel use llama.cpp. Lambda profiles use llama.cpp
(OrcaCyber/Flint), vLLM (M.O.G./SAGE/CYBER-FROST), or TensorFold (GLM EXL3).
Their source and revisions are recorded in the model and serving lock files.
Model weights are downloaded
from their publishers, are never included in this repository, and have their
own licenses. Review each publisher's model card before downloading or using it.

The Gemma chat template has its own attribution and retained license in
`chat-templates/README.md` and `chat-templates/LICENSE.ik_llama.cpp`.
The Adverserial confidential SDK is downloaded separately from its pinned
upstream commit and retains its upstream notices.

No license for the original harness code is currently included in this repository.
This release does not grant rights to third-party weights or trademarks.

Documentation, source repositories and recorded versions are indexed in
[Sources and upstream references](docs/sources.md).
