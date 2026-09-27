# Security

## What the MCP server can and cannot do

- **No network.** `paklaw-mcp` never opens a socket or makes an outbound request. It
  talks to its client over stdin/stdout only.
- **No writes.** Every tool is read-only (`readOnlyHint: true`). The server reads one
  file at start (`PAKLAW_CORPUS`) and never writes anything. The corpus is changed only
  by `paklaw-corpus`, run by a person, never by a model.
- **No secrets.** It needs no API key or token, and it reads no environment variable
  except `PAKLAW_CORPUS`.
- **No dependencies.** It uses the standard library only, so there is no third-party
  code in the supply chain at run time.
- **Bounded inputs.** Text arguments are capped at 200,000 characters and large results
  are paged, so one call cannot make the server produce megabytes of output.

## What it is not

It is not legal advice. It quotes provisions from whatever corpus it is given, and a
corpus with wrong dates gives wrong answers with correct-looking citations. The server
checks that a corpus's dates are *consistent* before serving it. Whether they are
*correct* is up to whoever built the corpus.

## Reporting a problem

Open an issue at <https://github.com/hammasbuilds/pak-law-assistant/issues>. For
anything you would rather not post publicly, say so in an issue without the details and
a private channel will be arranged.
