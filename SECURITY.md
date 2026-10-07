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
- **Bounded inputs and bounded work.** Text arguments are capped at 200,000 characters
  and list results are paged. The two places where the CORPUS rather than the argument
  sets the size are bounded too. `compare_versions` matches sentences first and only then
  compares the words of each changed passage, with at most 1,200 words in a passage and
  at most 500 changed passages reported. The word comparison is cubic on the text an
  amending Act produces — measured at 7.5s for 1,000 alternating words, 59s for 2,000 and
  206s for 3,000 — so a 34,000-character provision, which is ordinary for a tax
  ordinance, held this single-threaded server for 56 seconds, and a
  1.4-million-character one produced a 5.8 MB reply from 90 bytes of arguments. Both are
  now hundredths of a second. Over either bound the result says so rather than truncating
  silently.

  This line used to say only that arguments are capped, which was true and did not cover
  the thing that made the server slow.

## What it is not

It is not legal advice. It quotes provisions from whatever corpus it is given, and a
corpus with wrong dates gives wrong answers with correct-looking citations. The server
checks that a corpus's dates are *consistent* before serving it. Whether they are
*correct* is up to whoever built the corpus.

## Reporting a problem

Open an issue at <https://github.com/hammasbuilds/pak-law-assistant/issues>. For
anything you would rather not post publicly, say so in an issue without the details and
a private channel will be arranged.
