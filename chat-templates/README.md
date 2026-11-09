<a name="bugtraces-gemma-4-template"></a>

# ![BugTrace's Gemma 4 template](../assets/docs/headings/h1-bugtraces-gemma-4-template.svg)

**For the BugTrace model only.** This template applies only to the BugTrace
APEX Mini Raspberry Pi model. Other models in Velocicyber do not use this file.

The BugTrace APEX Mini Pi profile loads [gemma4.jinja](gemma4.jinja) using
`--chat-template-file`. This vendored template comes from
`models/templates/google-gemma-4-31B-it.jinja` at the pinned ik_llama.cpp revision
`32cddbfcefed93896a39c64e7c38c119de8682e6`.

[Upstream template source](https://github.com/ikawrakow/ik_llama.cpp/blob/32cddbfcefed93896a39c64e7c38c119de8682e6/models/templates/google-gemma-4-31B-it.jinja)

Its SHA-256 is recorded in [the BugTrace lock](../config/locks/pi-bugtrace.lock.json)
and verified before server launch. The template supports Chat Completions
messages containing tool results. It is supplied as runtime configuration;
model weights and executable binaries are unchanged.

The upstream source's MIT license is retained in [LICENSE.ik_llama.cpp](LICENSE.ik_llama.cpp).
See also [Gemma 4 prompt formatting](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4).
