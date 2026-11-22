# Sharp chat template

`chat_template_sharp_medium.jinja` is the Sharp-naze chat template v1.5 from
[nazeshinjite/Flash-Next-Chat-Templates](https://huggingface.co/nazeshinjite/Flash-Next-Chat-Templates),
built from the Qwen3.8-Flash-Next GGUF template with grafts from
[peculiar-ragdoll/Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates)
v22.5.0 (Apache-2.0; see [LICENSE-SHARP-APACHE-2.0.txt](LICENSE-SHARP-APACHE-2.0.txt)). The source repo is
tagged `license: other`. Its default reasoning effort is `medium`.

One local edit: a closing-message bullet in the `<IMPORTANT>` tools block. Every assistant turn must end
with a tool call that advances the task, or a final text message; after the last tool result, a short
closing message stating what changed. Terse defaults OFF when tools are present; the `terse` kwarg and
inline `<|terse_on|>` / `<|terse_off|>` tags override.

SHA-256: `8c4f9990fc77db7e506e91b32884b3f5fbfc298e74cf81289cb65dada12dad49`.