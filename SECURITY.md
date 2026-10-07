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
  sets the size are bounded too.

  `compare_versions` matches sentences first and only then compares the words of each
  changed passage. The word comparison is the expensive one, and its cost is in the
  content rather than the length. Measured on this machine, over word lists, with the
  generators in `tests/test_diff_is_bounded.py`:

  | words | alternating shared/unique | statute prose, 1 word in 50 changed |
  |------:|--------------------------:|------------------------------------:|
  |   500 |                     1.25s |                               0.03s |
  | 1,000 |                     8.67s |                               0.24s |
  | 2,000 |                    84.59s |                               2.17s |
  | 6,000 |                         — |                              58.89s |

  Far worse than quadratic, and the same 1,000 words are 8.67s or 0.24s depending only
  on how they alternate — so a cap on the length is the wrong instrument, and a timing
  quoted without its shape, as this file once quoted these, cannot be reproduced.

  Three bounds. Precise comparison draws on a budget of **250,000 pairwise word
  comparisons for the whole call**, which is about 1.3s at the worst shape above; a
  passage past the budget is compared with the fast heuristic instead, which handles
  200,000 words in a second. A reply carries at most **500 changed passages**,
  **200,000 characters**, and **2,000 characters of either side of any one row**. The
  row limits are not redundant: "all of this was replaced by all of that" is a single
  row holding both texts, and one such row measured 4.38 MB.

  Measured end to end through `compare_versions`, from about 90 bytes of arguments. The
  reviewer who found this carried the curve further than the first fix accounted for:

  | provision text | before | after |
  |---:|---:|---:|
  | 34,000 chars | 56.8s | 0.008s |
  | 170,000 chars | **8,043s** (2h 14m) | 0.032s |
  | 1,460,000 chars | 5.8 MB reply | 150 KB reply, 0.18s |

  170,000 characters is *under* `TEXT_LIMIT`, which is the point: that limit applies to
  arguments, and a provision's length is a property of the statute book.

  The work bound alone did not fix the reply, because most of that reply was never the
  diff — `compare_versions` returns the provision as it stood on each date, so two
  texts, and `provision_history --with_text` is not paged and returns one per version.
  Each provision text in a result is abridged at **60,000 characters**, which is longer
  than the longest sections in a tax, companies or procedure ordinance, so nothing real
  is cut; above it the text states its own length and where to read the rest.

  Every bound reports itself in the result rather than truncating silently, and none of
  them declines to compare: a coarse answer about what changed is a true statement about
  the law, and a refusal to look is not.

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
