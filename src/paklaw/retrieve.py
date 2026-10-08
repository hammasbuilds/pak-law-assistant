"""BM25 retrieval over a statute book, always as of a date.

BM25 rather than embeddings, and that is a considered choice for this corpus rather
than a shortcut.

Legal text is **terminology-bound**. "Reasonable apprehension of death" is a term of
art; a semantically similar paraphrase is not the same provision and retrieving it
instead is a wrong answer. Embeddings are good at paraphrase, which is exactly the
wrong strength here. Statutes also use a small, stable vocabulary that classic term
weighting handles well.

The more important property is that BM25 is **inspectable**. When a lawyer asks why a
provision was returned, "these terms matched with these weights" is an answer. "It was
close in a 384-dimensional space" is not, and in a domain where the reasoning has to be
auditable that difference decides the design.

Implemented from the formula. It is a sum over query terms.
"""

from __future__ import annotations

import bisect
import datetime as dt
import math
import re
from collections import Counter, OrderedDict
from collections.abc import Callable, Container, Sequence
from dataclasses import dataclass, field

from .corpus import Corpus, Provision

_TOKEN = re.compile(r"[\w؀-ۿ']+", re.UNICODE)

# Words that carry no discriminating power in a statute book, where nearly every
# provision contains "section", "act", "shall" and "person".
LEGAL_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "to",
    "for",
    "by",
    "or",
    "and",
    "any",
    "such",
    "shall",
    "be",
    "is",
    "are",
    "as",
    "on",
    "with",
    "under",
    "this",
    "that",
    "it",
    # "its" was the one possessive missing, while "it", "their", "his" and "her" were
    # all here: "is Pakistan an Islamic state by its Constitution?" carried a pronoun
    # as a content word and scored provisions on whether they happened to contain one.
    "its",
    "act",
    "section",
    "sub",
    "clause",
    "provided",
    "may",
    "which",
    "who",
    "where",
    "has",
    "have",
    "had",
    "been",
    "was",
    "were",
    "not",
    "no",
    "if",
    "than",
    "then",
    # Penal boilerplate: nearly every offence begins "Whoever commits ... shall be
    # punished", so these words match a murder section to a question about theft.
    "whoever",
    "person",
    "persons",
    "commit",
    "commits",
    "committed",
    "punished",
    "punishable",
    "liable",
    # Question words. A question is phrased around them and no provision is about them.
    "what",
    "when",
    "how",
    "why",
    "can",
    "could",
    "does",
    "do",
    "did",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "they",
    "their",
    "his",
    "her",
    "him",
    "about",
    "there",
    "law",
    "legal",
}


# Pakistani statutes name offences in Urdu and Arabic terms of art, so the word a person
# asks with is frequently not the word the statute uses: a question about *murder* has to
# reach "qatl-i-amd", and one about *theft* has to reach "chori". Without this bridge the
# weak-match refusal fires on perfectly answerable questions - the nearest provision is
# right there, and the only thing missing is the vocabulary.
#
# This maps vocabulary, never meaning. Each entry is a term the statute book itself uses
# for the concept the English word names; nothing here broadens a provision's scope, and
# no entry is a near-synonym or a paraphrase. A wrong entry would cite the wrong offence,
# so the table stays small and every line has to be defensible from the statute's own
# wording.
STATUTE_VOCABULARY: dict[str, tuple[str, ...]] = {
    "murder": ("qatl", "amd"),
    "homicide": ("qatl",),
    "manslaughter": ("qatl", "khata"),
    # The statute says "causing death"; a person asks about killing.
    "killing": ("qatl", "death"),
    # Ikrah is compulsion. s.303 PPC is headed "Qatl committed under ikrah-i-tam or
    # ikrah-i-naqis", and a question about duress reached none of it.
    "duress": ("ikrah",),
    "compulsion": ("ikrah",),
    "theft": ("chori",),
    "robbery": ("haraabah",),
    "adultery": ("zina",),
    "retaliation": ("qisas",),
    "bloodmoney": ("diyat",),
    "discretionary": ("tazir",),
    "hurt": ("jurh",),
    "defamation": ("qazf",),
}
# Two entries were removed here, by the table's own rule. "blood" -> diyat matched
# "blood relative" and "blood sample"; "intoxication" -> hadd was worse, because hadd
# is a category of punishment that applies to zina, theft and qazf as well, so every
# hadd question was scoped to intoxication. Neither was the statute's term for the
# concept the English word names, which is the only thing this table is allowed to hold.
# Two English words for one thing in the Code's own register, which is a different
# table from the one above: nothing here is a statute term, and nothing here changes
# which offence a question is about. A penal code provides *punishments* and a court
# passes a *sentence*, and a person uses whichever they know. Without this, "what is
# the sentence for murder?" was answered with s.54, "Commutation of sentence of death"
# — which contains both words and is about neither.
ENGLISH_VARIANTS: dict[str, tuple[str, ...]] = {
    "sentence": ("punishment",),
    "punishment": ("sentence",),
    "penalty": ("punishment", "sentence"),
}

# Read the other way too, so a question in the statute's own terms still finds the
# English wording of a heading.
_VOCABULARY_REVERSE: dict[str, tuple[str, ...]] = {}
for _english, _terms in STATUTE_VOCABULARY.items():
    for _term in _terms:
        _VOCABULARY_REVERSE.setdefault(_term, ())
        _VOCABULARY_REVERSE[_term] += (_english,)

# Words that choose BETWEEN neighbouring offences rather than describing one. Attempt
# to murder is s.324, abetment is s.109, conspiracy is s.120B - none of them s.302. A
# question carrying one of these is about a different provision from the same question
# without it, so a provision that does not contain the word is not an answer to it.
#
# Found by an independent review, which asked "what is the punishment for attempt to
# murder?" and was told s.302: "punished with death as qisas". Coverage counted
# "attempt" as one interchangeable content word among three, so two of three cleared
# the gate. The qualifier is not interchangeable; it is the whole question.
QUALIFIERS = frozenset(
    {
        "attempt",
        "attempted",
        "abetment",
        "abet",
        "abetting",
        "conspiracy",
        "conspiring",
        "omission",
        "preparation",
        "threat",
        "threatening",
        "negligence",
        "negligent",
        "accidental",
        "unintentional",
        "involuntary",
        "mitigated",
        "aggravated",
        # The species of qatl, which are three different offences with three different
        # punishments: amd is s.302 (death as qisas), shibh-i-amd s.316, khata s.322
        # (diyat). The word naming the species is the entire question, exactly as
        # "attempt" is - and sharing the word "qatl" is what made them look alike.
        "amd",
        "khata",
        "shibh",
        # Likewise the mode of punishment and the ground of the offence: a question
        # about qisas is not answered by a provision that only offers tazir, and one
        # about ikrah is not answered by a provision that never mentions it.
        "qisas",
        "tazir",
        "diyat",
        "ikrah",
    }
)

# Words whose absence from a statute book says nothing. A corpus that has never seen
# "dacoity" cannot answer a question about dacoity - that is a real signal, and the
# strongest one available for an offence the corpus simply does not contain. A corpus
# that has never seen "many" is just a corpus: the word is not what the question is
# about. Without this distinction the rule refuses "how many years does imprisonment
# for life count as?", which s.57 answers exactly.
NOT_A_SUBJECT = frozenset(
    {
        "many",
        "much",
        "count",
        "counts",
        "long",
        "often",
        "exactly",
        "actually",
        "really",
        "mean",
        "means",
        # "the meaning of the word vessel" is s.48 PPC, and `meaning` was the
        # rarest word in it: a penal code legislates about vessels and not about
        # meaning. `mean` and `means` were here and the gerund was not, which is
        # the kind of gap a table has and a rule does not.
        "meaning",
        "meanings",
        "definition",
        "definitions",
        "happens",
        "called",
        "regarding",
        "concerning",
        "about",
        # Ordinary English that carries a question without naming anything in a statute
        # book. Deliberately NOT the legal near-misses: "use", "give", "make", "take"
        # and "person" all do real work in a penal code ("use of force", "given in good
        # faith", "made in good faith"), so they are absent from this list even though
        # adding them would answer more questions. A word earns a place here by being
        # impossible to legislate about, not by being inconvenient.
        #
        # The verbs of ASKING. "what does the law say about punishment of qatl-i-amd?"
        # was refused with "'say' appears in no provision here" - of a corpus whose
        # s.302 is headed "Punishment of qatl-i-amd". Every one of 26 heading
        # questions generated from the corpus was refused this way, 25 of them on a
        # word like this one, and "what does the law say about X" is close to the most
        # natural phrasing a reader has.
        #
        # These qualify under the rule above: a statute does not legislate about
        # saying, telling or asking in the sense a questioner uses them. "state" and
        # "provide" are deliberately absent - a statute provides for things and names
        # the State - and so is "declare", which appears in constitutions.
        "say",
        "says",
        "said",
        "tell",
        "tells",
        "told",
        "ask",
        "asks",
        "asked",
        "asking",
        "explain",
        "explains",
        "describe",
        "describes",
        "summarise",
        "summarize",
        "list",
        "lists",
        "show",
        "shows",
        "find",
        "finds",
        "look",
        "looks",
        "know",
        "knows",
        "wondering",
        "wonder",
        "please",
        "get",
        "gets",
        "got",
        "getting",
        "do",
        "does",
        "did",
        "doing",
        "someone",
        "somebody",
        "something",
        "anything",
        "anyone",
        "everyone",
        "happen",
        "happened",
        "meant",
        "else",
        "instead",
        "whether",
        # Pure adverbs of degree and frequency. A statute legislates about conduct, not
        # about "never" - but "establish", "people" and "acting" are NOT here, because
        # a penal code establishes tribunals, protects people and penalises acting in
        # bad faith. The list only takes words that cannot be legislated about.
        "never",
        "always",
        "ever",
        "simply",
        "merely",
    }
)

# Conservative English suffixes. "punishment" must reach "punished", which is the single
# most common mismatch in a penal code: the question nominalises what the statute
# conjugates. Only applied to ASCII words long enough that the stem stays a word.
#
# A bare "e" is last and does real work: the statute conjugates ("the word 'vessel'
# denotes") and the question nominalises ("what does the word vessel denote"). Stripping
# only "es" reduced the statute's word to "denot" and left the question's at "denote",
# so the two never met. Nothing ends in both "es" and "e", so adding it changes no
# existing stem, and the minimum length keeps "the" and "be" whole.
#
# Longest first, and one suffix per word: "sections" must take "ions" before "s" or it
# stops at "section" and never meets "sect". The nominalisations are what a question is
# built from and the conjugations are what a statute is built from - "elimination of
# exploitation" against "the State shall eliminate", "penalties" against "penalty",
# "murderer" against "murder" - and each pair that does not meet is a question the
# corpus can answer and does not.
_SUFFIXES = (
    "ations",
    "ation",
    "ements",
    "ement",
    "ments",
    "ment",
    "ingly",
    "ing",
    "edly",
    "ed",
    "ions",
    "ion",
    "ies",
    # "Islamic" against "Islam": the Constitution's own adjective for its own noun,
    # and Article 2 ("Islam shall be the State religion") was missing a question that
    # used it while Article 1 - which happens to contain the word "Islamic" in a name -
    # was not.
    "ic",
    "ers",
    "er",
    "ors",
    "or",
    "es",
    "s",
    "y",
    "e",
)


def _stems(term: str) -> set[str]:
    """`term` and the stems it could share with a differently inflected form."""
    out = {term}
    if not term.isascii() or not term.isalpha():
        return out
    # Every suffix that fits, not the first. "elimination" strips "ation" to "elimin"
    # and "ion" to "eliminat", and it is the second that meets "eliminate"; stopping at
    # the first kept the pair apart. The extra stems only ever add a way for two
    # spellings of one word to meet - a stem that matches nothing costs nothing.
    # Three characters, not four. Four kept `payment` from reaching `pay` and `giving`
    # from reaching `giv`, so the refusal that says a word appears here "in any form"
    # said it of `paid` over a corpus containing `payment`. Lowering it merges more
    # aggressively, so it was measured: identical on the 121-question generated
    # population (118 right, 3 wrong, 0 refused - the same three ambiguous pairs) and
    # identical on the hand-written set (45 right, 0 wrong, 0 of 6 unanswerable
    # questions answered).
    for suffix in _SUFFIXES:
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            stem = term[: -len(suffix)]
            out.add(stem)
            # English doubles the final consonant before -ing and -ed, and the statute
            # uses the undoubled form: "transmitting" strips to "transmitt" while
            # "transmits" strips to "transmit", and the two never met. Both forms are
            # kept rather than one chosen, because the shortened form of a word that
            # genuinely ends in a double letter ("pass" -> "pas") matches nothing.
            if len(stem) >= 3 and stem[-1] == stem[-2]:
                out.add(stem[:-1])
    return out


#: Words that mean the same thing and do not share a stem, as groups rather than pairs.
#:
#: Suffix-stripping cannot reach any of these. `paid` is not `pay` plus a suffix,
#: `given` is not `give` plus one, and `empowered` differs from `power` by a PREFIX -
#: so the refusal that tells a reader their word appears in no provision "in any form"
#: said it of `paid` over a corpus containing `payment`, and of `giving` over one
#: containing `given`. A refusal states a fact about the statute book, and that one was
#: false of it.
#:
#: English irregular verbs are a closed list, so this is a table and not a rule, and
#: a statute book is written in a particular part of it: `held`, `found`, `bound`,
#: `sworn`, `struck`, `forbidden`, `sent`, `dealt`, `brought`. With them go the
#: nominalisations whose spelling diverges from the verb's, where the same gap opens:
#: `proof`/`prove`, `theft`/`steal`, `sale`/`sell`, `death`/`die`.
#:
#: SUBSTANTIVE words only. A first version of this held `say`, `do`, `be`, `have`,
#: `make` and `take`, which cost answers rather than winning them: giving `say` a
#: set of forms made it a content word, so `what does the law say about "animal"?`
#: was refused for missing one of its two content words - the word being the frame
#: of the question rather than any part of its subject. Nor are near-synonyms here:
#: `gift` is not an inflection of `give`, and `breach` is not one of `break`. Those
#: belong in `STATUTE_VOCABULARY`, where a claim about meaning can be read as one.
_IRREGULAR_GROUPS: tuple[tuple[str, ...], ...] = (
    # Verbs a statute is written in.
    ("pay", "pays", "paid", "paying", "payment", "payments", "payable", "payee", "payer"),
    ("give", "gives", "gave", "given", "giving"),
    ("hold", "holds", "held", "holding", "holder"),
    ("find", "finds", "found", "finding", "findings"),
    ("bind", "binds", "bound", "binding"),
    ("forbid", "forbids", "forbade", "forbidden", "forbidding"),
    ("strike", "strikes", "struck", "striking"),
    ("send", "sends", "sent", "sending"),
    ("lend", "lends", "lent", "lending"),
    ("spend", "spends", "spent", "spending"),
    ("bring", "brings", "brought", "bringing"),
    ("seek", "seeks", "sought", "seeking"),
    ("think", "thinks", "thought", "thinking"),
    ("catch", "catches", "caught", "catching"),
    ("swear", "swears", "swore", "sworn", "swearing"),
    ("steal", "steals", "stole", "stolen", "stealing", "theft", "thefts", "thief", "thieves"),
    ("break", "breaks", "broke", "broken", "breaking"),
    ("speak", "speaks", "spoke", "spoken", "speaking"),
    ("write", "writes", "wrote", "written", "writing", "writings"),
    ("know", "knows", "knew", "known", "knowing", "knowingly", "knowledge"),
    ("choose", "chooses", "chose", "chosen", "choosing"),
    ("bear", "bears", "bore", "borne", "bearing", "bearer"),
    ("lose", "loses", "lost", "losing", "loss", "losses"),
    ("meet", "meets", "met", "meeting"),
    ("sell", "sells", "sold", "selling", "sale", "sales", "seller"),
    ("tell", "tells", "told", "telling"),
    ("sit", "sits", "sat", "sitting"),
    ("stand", "stands", "stood", "standing"),
    ("deal", "deals", "dealt", "dealing"),
    ("keep", "keeps", "kept", "keeping", "keeper"),
    ("leave", "leaves", "left", "leaving"),
    ("arise", "arises", "arose", "arisen", "arising"),
    ("begin", "begins", "began", "begun", "beginning"),
    ("run", "runs", "ran", "running"),
    ("fall", "falls", "fell", "fallen", "falling"),
    ("rise", "rises", "rose", "risen", "rising"),
    ("drive", "drives", "drove", "driven", "driving", "driver"),
    ("hide", "hides", "hid", "hidden", "hiding"),
    ("shoot", "shoots", "shot", "shooting"),
    ("flee", "flees", "fled", "fleeing"),
    # Nominalisations whose spelling diverges from the verb's, where the same gap opens.
    ("power", "powers", "empower", "empowers", "empowered", "empowering", "powered"),
    ("prove", "proves", "proved", "proven", "proving", "proof", "proofs"),
    ("die", "dies", "died", "dying", "death", "deaths", "dead"),
    ("live", "lives", "lived", "living", "life"),
    ("judge", "judges", "judged", "judging", "judgment", "judgments", "judicial"),
    ("believe", "believes", "believed", "believing", "belief", "beliefs"),
    ("relieve", "relieves", "relieved", "relieving", "relief"),
    ("marry", "marries", "married", "marrying", "marriage", "marriages"),
    ("child", "children"),
    ("person", "persons", "people"),
    ("wife", "wives"),
    ("woman", "women"),
    ("man", "men"),
)

#: The groups above, read as a lookup. Every member maps to every other member, which
#: is what the three tables below this one do pairwise.
IRREGULAR_FORMS: dict[str, tuple[str, ...]] = {}
for _group in _IRREGULAR_GROUPS:
    for _member in _group:
        IRREGULAR_FORMS[_member] = IRREGULAR_FORMS.get(_member, ()) + tuple(
            other for other in _group if other != _member
        )


# The three tables above are keyed by one spelling of each word, and a question uses
# whichever it likes. "killing" is the key and "kills" is what someone types, so every
# key is indexed under its stems as well: without this the bridge to the statute's own
# vocabulary is only crossed by the exact inflection the table happens to name.
_BY_STEM: dict[str, tuple[str, ...]] = {}
for _table in (STATUTE_VOCABULARY, _VOCABULARY_REVERSE, ENGLISH_VARIANTS, IRREGULAR_FORMS):
    for _word, _related in _table.items():
        for _stem in _stems(_word):
            _BY_STEM[_stem] = _BY_STEM.get(_stem, ()) + _related


def expand(term: str) -> set[str]:
    """Every form of `term` that counts as the same term when matching a provision.

    Query-side only: the index keeps the statute's own words, so `Hit.matched_terms` is
    still keyed by what the person actually asked and `why()` stays readable.

    The bridge is crossed once, from the asked word or a stem of it. Crossing again
    from a word the bridge reached lands in a neighbouring offence: "khata" reaches
    "manslaughter", and "manslaughter" reaches "qatl" - so the single word that
    separates qatl-i-khata (s.322, diyat) from qatl-i-amd (s.302, death as qisas)
    became a synonym for s.302's. An independent review asked for the punishment for
    qatl-i-khata and was told s.302 at coverage 1.00, with no refusal.
    """
    forms = _stems(term)
    # Only stems of the word asked are vocabulary keys ("murders" -> "murder"). Terms
    # the vocabulary produced are not asked again.
    for stem in _stems(term):
        for related in _BY_STEM.get(stem, ()):
            forms |= _stems(related)
    return forms


#: Punctuation inside a transliterated Arabic term of art. The statute writes "ta'zir"
#: and a person types "tazir"; `_TOKEN` keeps the apostrophe, so the two were different
#: words and never met. Worse than a scoring loss: "tazir" is in QUALIFIERS, so the
#: mismatch escalated to a hard `different_offence` refusal on a question s.302(b)
#: answers literally - while "ta zir", spelled with a space, worked. Folded on both
#: sides, the way `normalise_number` already folds "489-F" to "489F".
_FOLD = re.compile(r"[’'‐-―-]")


def _fold(token: str) -> str:
    folded = _FOLD.sub("", token)
    return folded or token


def tokenise(
    text: str, *, keep_stopwords: bool = False, defined: Container[str] | None = None
) -> list[str]:
    """Words, lowercased. Urdu script is preserved as its own tokens.

    Legal numbers are kept: "302" is one of the most discriminating tokens in a penal
    code, and a tokeniser that drops digits loses the ability to find a section by its
    number.

    `defined` is a set of words this corpus DEFINES, and a word in it survives even if
    it is a stopword. A statute book defines the words it uses, so its most ordinary
    words are also the subjects of its definition provisions: PPC s.50 is headed
    "Section" and defines that word, and dropping "section" as a stopword left
    `tokenise("what does section denote?")` as `['denote']` - the only word
    identifying the provision gone before anything was scored, and the answer was
    s.47, "Animal".

    Not an exception to the stopword rule but a correction of it. "section" occurs in
    most provisions, so it ranks nothing by itself; what it does is let the provision
    whose SUBJECT is that word be reachable at all.
    """
    tokens = [_fold(t.lower()) for t in _TOKEN.findall(text)]
    if keep_stopwords:
        return tokens
    keep = defined or ()
    return [t for t in tokens if t not in LEGAL_STOPWORDS or t in keep]


#: How much rarer the question's rarest word has to be than its next-rarest before
#: `Hit.missing_key_term` treats it as THE subject of the question. `max` always
#: returns something, and over a set of equally common words it returns an arbitrary
#: one, which is not a fact about the question.
KEY_TERM_MARGIN = 1.0

#: The information coverage above which `Hit.missing_key_term` stands down. A question
#: whose remaining words are already an overwhelming match for the provision is a
#: question about that provision, whatever its rarest word happens to be: "acting
#: without due care and attention" is s.52 PPC quoted with one word added, and `acting`
#: is both the rarest word in it and absent from s.52, which legislates about good
#: faith in a sentence containing no form of `act`.
#:
#: Swept over every set in the suite - the hand-written answerable pairs, the
#: paraphrases, the 121-question generated population, and the five sets that must be
#: refused. `hand` is right / wrong / refused over 51 answerable questions, `leaked` is
#: questions that must refuse and did not:
#:
#:      ceiling   hand          generated     leaked
#:      rule off  45R 0W 6X     118R 3W 0X    1
#:      0.70      44R 0W 7X     118R 3W 0X    0
#:      0.65      44R 0W 7X     118R 3W 0X    0
#:      0.60      45R 0W 6X     118R 3W 0X    0
#:      0.55      45R 0W 6X     118R 3W 0X    0
#:
#: 0.60 answers everything the rule-off column answers and leaks nothing: the one
#: leak is "what fine is payable for qatl-i-amd as qisas?", answered from s.302 PPC,
#: which prescribes death or imprisonment and no fine. At 0.65 and 0.70 the rule also
#: catches "the meaning of the word vessel", which is s.48 - the cost of firing on a
#: question whose rarest word is its frame, and why `meaning` is in `NOT_A_SUBJECT`.
KEY_TERM_CEILING: float | None = 0.60


@dataclass
class Hit:
    provision: Provision
    score: float
    matched_terms: dict[str, float] = field(default_factory=dict)
    # Distinct content terms of the question this provision does not contain.
    missing_terms: list[str] = field(default_factory=list)
    # What the ranking actually sorted on: `score` weighted by coverage. Kept as a
    # field rather than recomputed, so the order a client sees can be checked
    # against a number it was given.
    ranking: float = 0.0
    # Share of the question's content terms that appear in this provision's HEADING.
    heading_coverage: float = 0.0
    # How close together this provision says the question's words, 0 to 1. Reported
    # because it reorders hits that cover the question equally, so a client that is
    # shown an order can see what produced it.
    tightness: float = 0.0
    # Terms that select a different provision and are absent from this one. A hit with
    # any of these is about a neighbouring offence, not this question.
    missing_qualifiers: list[str] = field(default_factory=list)
    # The weight of every content term of the question - matched and missing alike -
    # which is its IDF over the corpus as of this date. Carried so `coverage` can be
    # a share of the question's INFORMATION rather than of its word count, and so a
    # caller can see which word the share turned on.
    term_weights: dict[str, float] = field(default_factory=dict)
    # Question terms that appear nowhere in the corpus as of this date, in any form.
    # The same for every hit, carried here because it is the reason a caller refuses:
    # "dacoity" is absent from a corpus of thirteen sections, and the honest answer to
    # a question about dacoity is that this corpus does not contain it - not the
    # best-scoring provision among those it does.
    unknown_terms: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Share of the question's content terms the provision contains.

        The sort's primary key, and a count rather than a weighted share on purpose.
        Weighting it by IDF was tried: it reorders the results and took the benchmark
        from 1 wrong answer in 97 to 4, because a provision matching one rare word
        then outranks the provision the question is actually about -
        "what does the word animal mean in the Penal Code?" went to s.52A.

        `information_coverage` is the weighted view, and it decides whether to answer
        at all. The two questions are different: which provision is closest, and
        whether the closest one is close enough.
        """
        total = len(self.matched_terms) + len(self.missing_terms)
        return len(self.matched_terms) / total if total else 0.0

    @property
    def missing_key_term(self) -> str:
        """The question's most distinctive content word, when this provision lacks it.

        "" when the provision has it, or when nothing distinguishes the words.

        `information_coverage` is a share, and a share can be cleared while the whole
        subject of the question is missing: "what fine is payable for qatl-i-amd as
        qisas?" matched `qatl`, `amd` and `qisas` for 54% of the information and missed
        `fine` and `payable`, which are the two rarest words in it and the two it asks
        about. s.302 prescribes no fine. A review measured the general case - a missing
        distinctive word is tolerated in 66.8% of a 3,999-question probe - so this is
        the rule the share needs beside it rather than a tighter share.

        The MOST distinctive word only. Every question drops some word the provision
        happens to lack, and refusing on any of them would refuse nearly everything;
        the rarest word of a question is the one it is about, which is the same
        reasoning `information_coverage` rests on, applied to the top of the
        distribution instead of to the sum.
        """
        if not self.term_weights or len(self.term_weights) < 2:
            return ""
        # `NOT_A_SUBJECT` first, and this is the whole difficulty with the rule: the
        # rarest word of a question is USUALLY what it is about, and sometimes it is
        # the frame. "mean" is the rarest word in "what does the word animal mean in
        # the Penal Code?" - a penal code legislates about animals and not about
        # meaning - so without this the rule refused a question s.47 answers, which is
        # the same mistake `absent_terms` already keeps this list to avoid.
        weights = {
            term: weight for term, weight in self.term_weights.items() if term not in NOT_A_SUBJECT
        }
        if len(weights) < 2:
            return ""
        term, weight = max(weights.items(), key=lambda kv: (kv[1], kv[0]))
        if term in self.matched_terms:
            return ""
        # A word the corpus has never seen is `unknown_terms`, which says something
        # else: that the subject is absent from the book rather than from this section.
        if term in self.unknown_terms:
            return ""
        # And it has to be distinctive in absolute terms, not merely the largest of a
        # flat set: `max` always returns something, and over five words of equal
        # weight it returns an arbitrary one.
        others = sorted(w for t, w in weights.items() if t != term)
        if not others or weight < others[-1] * KEY_TERM_MARGIN:
            return ""
        # And not when the rest of the question already matches overwhelmingly. A
        # missing rare word is evidence that the provision is about something else
        # only where there is something else for it to be about.
        if KEY_TERM_CEILING is not None and self.information_coverage >= KEY_TERM_CEILING:
            return ""
        return term

    @property
    def information_coverage(self) -> float:
        """Share of the question's information the provision contains, IDF-weighted.

        The gate, because a count of content words cannot tell which word the question
        was about. "who may grant pardon in a case of qatl-i-amd?" matched `qatl`,
        `amd` and `case` and missed `grant` and `pardon` - 3 of 5, clearing a 0.5 bar -
        so s.302 PPC came back with a correct-looking citation while s.55A, "the right
        of the President to grant pardons", sat unreturned in the same corpus. The only
        two words the question was *about* were the two missing ones, and `pardon`
        appears in one provision of twenty-six where `case` is in most of them.

        The weights are the IDFs the scorer already computes, so this is the same view
        of the corpus the ranking takes rather than a second one.
        """
        if not self.term_weights:
            return self.coverage
        whole = sum(self.term_weights.values())
        if not whole:
            return 0.0
        found = sum(w for term, w in self.term_weights.items() if term in self.matched_terms)
        return found / whole

    def why(self) -> str:
        """Why this provision was returned, in terms a lawyer can check."""
        ranked = sorted(self.matched_terms.items(), key=lambda kv: -kv[1])[:5]
        terms = ", ".join(f"{term} ({weight:.2f})" for term, weight in ranked)
        matched = len(self.matched_terms)
        total = matched + len(self.missing_terms)
        # The share matched is what the ranking weights by, so it belongs in the
        # explanation - but only for a search. A provision the question cited by
        # number is a lookup with one synthetic term, and "1 of 1 terms" there
        # describes nothing.
        return f"{matched} of {total} terms — {terms}" if total > 1 else terms


@dataclass
class BM25Index:
    k1: float = 1.5
    # Length normalisation, left at the standard value. An earlier note here claimed a
    # sweep had shown b "makes no difference at all across 0.0-0.75"; that sweep set
    # `BM25Index.b` on the class, and because this is a dataclass the default is
    # captured in __init__, so every run used 0.75 and the experiment measured nothing.
    # A real sweep, reachable through `LawSearch(tuning=...)`, shows b DOES matter: at
    # b=0.40 a lower heading weight suffices for the same result. 0.75 is kept because
    # the standard value already works, not because tuning it was tried and failed.
    b: float = 0.75
    # Matching a section *number* should outrank matching its prose.
    number_boost: float = 3.0
    heading_boost: float = 2.0
    # Coverage - the share of the question's terms a provision contains - decides the
    # order outright, and these notes record how that came to be the rule rather than a
    # weighting, because the weighting version was wrong twice.
    #
    # BM25 alone fails here and the failure is not subtle. Asked "how is imprisonment
    # for life reckoned in fractions of punishment?", it ranked s.53 "Punishments" (385
    # characters, 3 of 5 terms) above s.57 "Fractions of terms of punishment" (2,602
    # characters, 5 of 5): length normalisation taxes the longer provision harder than
    # two extra terms reward it. The answer was in the corpus, scored second, and a
    # confident citation of the wrong section came out.
    #
    # The first fix multiplied the score by coverage**weight - Lucene's old `coord`.
    # That fixed the one phrasing it was written beside and not the weakness: a
    # rephrasing with none of s.57's heading words still lost. Coverage is now the
    # primary sort key, which makes the exponent EXACTLY inert - within a group of equal
    # coverage it is a constant factor that cancels, and between groups coverage already
    # decided - so the knob is gone rather than left looking tuned.
    #
    # Statute text is terminology-bound. A question's words are terms of art, and a
    # provision containing every one of them is the answer, whatever the term weights
    # say.
    # The heading is a TIEBREAK, never a multiplier, and the distinction was forced by
    # two questions that pull opposite ways.
    #
    # "What is the State?" returned Article 2 CONST ("Islam shall be the State
    # religion") over Article 7 CONST, headed "Definition of the State", because both
    # contain the word and Article 2 is shorter. One content word also makes coverage
    # 1.00 for everything that matches at all, so coverage cannot separate them.
    #
    # "Imprisonment for life is reckoned as how many years" wants s.57 "Fractions of
    # terms of punishment", whose heading shares NO word with the question, over s.55
    # "Commutation of sentence of imprisonment for life", whose heading matches two. As
    # a multiplier the heading made that worse, overturning the coverage difference
    # (0.80 against 0.60) that had just been fixed.
    #
    # So coverage decides first and the heading only separates hits that cover the
    # question equally well. `heading_focus` is the share of the HEADING'S OWN words the
    # question matched - how much that provision is ABOUT those words - not the share of
    # the question found in the heading, which both of the Article 2/7 headings score at
    # 1.00 and which therefore separates nothing.
    heading_weight: float = 1.5
    # How much co-location of the question's words counts, within provisions that cover
    # the question equally well. Swept over both question sets; see the module note on
    # s.57 against s.302, where the answer is 215 tokens long and says the thing once
    # in six and the wrong provision is 51 tokens with the same words twenty-one apart.
    #
    # Zero reproduces the previous ranking exactly, which is what makes the sweep
    # meaningful rather than a comparison against a different tool.
    # 0.5, from a sweep over the 105-question benchmark and the 101 generated
    # questions. 0 is the previous ranking (56 right, 1 wrong); 0.25 to 0.75 all give
    # 57 right and 0 wrong - the sweep's only plateau without a wrong answer, and it
    # also fixes one the benchmark already had, 'consent of the heirs of the victim'
    # citing s.55A instead of s.54. At 1.0 and above the signal starts overpowering
    # coverage's intent: s.54 ("Commutation of sentence of death") is short and tight,
    # so "what is the punishment for murder?" leaves s.302 for it, which is 2 wrong at
    # 1.25 and 3 at 1.5. 0.5 is the middle of the plateau rather than an edge of it.
    proximity_weight: float = 0.5

    documents: list[Provision] = field(default_factory=list)
    _tokens: list[list[str]] = field(default_factory=list)
    _frequencies: list[Counter] = field(default_factory=list)
    _document_frequency: Counter = field(default_factory=Counter)
    _average_length: float = 0.0
    # stem -> the statute's own words that reduce to it, so a query term can find the
    # inflection the statute actually used without the index losing that word.
    _by_stem: dict[str, set[str]] = field(default_factory=dict)
    # The heading's own tokens, kept apart from the body. A provision whose HEADING is
    # about the question is the provision about the question - "Definition of the State"
    # answers "what is the State?" and Article 2, which merely mentions the State while
    # being about Islam, does not.
    _heading_tokens: list[set[str]] = field(default_factory=list)
    # term -> the index words that count as it. Depends only on the term and _by_stem,
    # so it is rebuilt with the index and never outlives one.
    _form_cache: dict[str, set[str]] = field(default_factory=dict)
    # Every stem in the corpus's raw text, stopwords included - which the index above
    # is not. Used for one thing only: deciding whether a question's word is REALLY
    # absent from the corpus.
    #
    # `commits`, `committed`, `punished`, `punishable`, `liable`, `person` and fifty
    # more are in `LEGAL_STOPWORDS`, so they are erased from the index. A question
    # asking about "the commission of qatl-i-amd" stems `commission` to `commit`,
    # finds nothing in the index, and was told the word "appears in no provision here"
    # - of a corpus whose s.300 opens "Whoever commits qatl". The refusal was a false
    # statement about the statute book, delivered with a refusal's authority, which is
    # the defect class a wrong citation belongs to.
    _text_stems: set[str] = field(default_factory=set)
    #: Words this corpus defines: the subject of every single-word heading. They
    #: survive the stopword filter, in the query and in the index, because a
    #: statute book's most ordinary words are also the subjects of its definitions.
    _defined_terms: set[str] = field(default_factory=set)
    #: token -> where it occurs in each provision's TEXT, in token order. The text
    #: only: the heading and the boosted repeats are appended to `_tokens` for scoring
    #: and have no position in the provision a reader sees.
    _positions: list[dict[str, list[int]]] = field(default_factory=list)

    def fit(self, provisions: Sequence[Provision]) -> BM25Index:
        self.documents = list(provisions)
        self._tokens = []
        self._frequencies = []
        self._document_frequency = Counter()
        self._by_stem = {}
        self._heading_tokens = []
        self._form_cache = {}
        self._text_stems = set()
        self._positions = []

        # Which words this corpus DEFINES, before anything is tokenised for scoring.
        # A provision whose whole heading is one word is a definition of that word -
        # s.50 "Section", s.47 "Animal", s.52A "Harbour" - and that word has to survive
        # the stopword filter or the provision defining it is unreachable. Single-word
        # headings only: "Punishment of qatl-i-amd" would otherwise define "of".
        self._defined_terms = {
            heading[0]
            for provision in self.documents
            if len(heading := tokenise(provision.heading, keep_stopwords=True)) == 1
        }

        for provision in self.documents:
            tokens = (
                tokenise(provision.text, defined=self._defined_terms)
                + tokenise(provision.heading, defined=self._defined_terms) * int(self.heading_boost)
                # The number is repeated so term frequency carries the boost, rather
                # than bolting a separate score on afterwards.
                + [provision.number.lower()] * int(self.number_boost)
            )
            self._tokens.append(tokens)
            where: dict[str, list[int]] = {}
            for position, token in enumerate(tokenise(provision.text, defined=self._defined_terms)):
                where.setdefault(token, []).append(position)
            self._positions.append(where)
            self._heading_tokens.append(
                set(tokenise(provision.heading, defined=self._defined_terms))
            )
            # Stopwords kept here and nowhere else: this set answers "is the word in
            # the corpus at all", not "can it rank a provision".
            for word in tokenise(provision.text, keep_stopwords=True) + tokenise(
                provision.heading, keep_stopwords=True
            ):
                self._text_stems.update(_stems(word))
            frequencies = Counter(tokens)
            self._frequencies.append(frequencies)
            self._document_frequency.update(frequencies.keys())
            for token in frequencies:
                for stem in _stems(token):
                    self._by_stem.setdefault(stem, set()).add(token)

        lengths = [len(t) for t in self._tokens]
        self._average_length = sum(lengths) / len(lengths) if lengths else 0.0
        return self

    def _tightness(self, index: int, terms: set[str]) -> float:
        """How close together this provision says the question's words, 0 to 1.

        1.0 is adjacent; it falls as the smallest window containing them grows. Terms
        matched only in the heading have no position in the text and are left out of
        the window, and the result is scaled by the share of matched terms that did
        have one - so a provision that co-locates two of five words is not rewarded
        like one that co-locates all five.

        A question of one word carries no proximity information, so it scores 0 and
        the factor built from it is 1: this must not become a length preference by
        another route.
        """
        positions = self._positions[index]
        found: list[list[int]] = []
        for term in terms:
            places = sorted(
                place for form in self._forms(term) for place in positions.get(form, ())
            )
            if places:
                found.append(places)
        if len(found) < 2:
            return 0.0

        # Smallest window covering one occurrence of each term, by a linear sweep over
        # the merged positions rather than the product of the lists.
        merged = sorted((place, which) for which, places in enumerate(found) for place in places)
        needed = len(found)
        counts: dict[int, int] = {}
        best = None
        left = 0
        for place, which in merged:
            counts[which] = counts.get(which, 0) + 1
            while len(counts) == needed:
                best = (
                    min(best, place - merged[left][0])
                    if best is not None
                    else (place - merged[left][0])
                )
                drop = merged[left][1]
                counts[drop] -= 1
                if not counts[drop]:
                    del counts[drop]
                left += 1
        if best is None:
            return 0.0
        tightest = needed - 1  # the window when every term is adjacent
        return (tightest / max(best, tightest)) * (len(found) / len(terms))

    def absent_terms(self, terms: Sequence[str]) -> list[str]:
        """Which of `terms` the corpus holds no word for, in any form.

        Public because a caller with NO hits at all needs the same answer a caller
        with hits gets. "what does the law say about dacoity?" produced no hits -
        every other word of it is a stopword or a verb of asking - and came back "no
        provision in force on that date matches the question", where the question's
        own subject was the thing missing and could have been named.

        "in any form" has to mean in any form: a term is absent only when neither the
        index nor the corpus's raw text holds a word that stems to it, or else the
        refusal tells a reader their word is missing from the statute book when the
        statute book contains it as a stopword.
        """
        return sorted(
            term
            for term in set(terms)
            if term not in NOT_A_SUBJECT
            and not any(self._document_frequency.get(form) for form in self._forms(term))
            and not (set(expand(term)) & self._text_stems)
        )

    def _forms(self, term: str) -> set[str]:
        """Index terms that count as `term`: itself, its inflections, its statute word.

        Memoised because the value depends only on the term and the index, while the
        callers sit inside the per-document loop: one five-word question over 10,000
        provisions called this 99,174 times and spent four fifths of the query in it.
        The cache lives on the index, so it is discarded with it.
        """
        cached = self._form_cache.get(term)
        if cached is not None:
            return cached
        forms = {term}
        for stem in expand(term):
            forms |= self._by_stem.get(stem, set())
        self._form_cache[term] = forms
        return forms

    def _idf(self, term: str) -> float:
        n = len(self.documents)
        df = self._document_frequency.get(term, 0)
        # Lucene's variant: always positive, so a term in most documents contributes
        # little rather than subtracting score from documents that contain it.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(
        self, query: str, *, limit: int = 5, where: Callable[[Provision], bool] | None = None
    ) -> list[Hit]:
        """Top `limit` hits among the provisions `where` accepts.

        The filter is applied before the cut, not after: filtering the top nine for one
        Act finds nothing when the other Act's provisions happen to score higher.
        """
        if not self.documents:
            return []

        terms = tokenise(query, defined=self._defined_terms)
        if not terms:
            return []

        # Computed once: it depends on the question and the corpus, not the provision.
        unknown = self.absent_terms(terms)

        hits: list[Hit] = []
        for index, provision in enumerate(self.documents):
            if where is not None and not where(provision):
                continue
            frequencies = self._frequencies[index]
            length = len(self._tokens[index])
            score = 0.0
            matched: dict[str, float] = {}

            for term in set(terms):
                # The term as asked, plus the statute's own wording for it. Scored on
                # the best single form rather than the sum, so a word that happens to
                # have several inflections in one provision does not outweigh a word
                # that appears once.
                best = 0.0
                for form in self._forms(term):
                    frequency = frequencies.get(form, 0)
                    if not frequency:
                        continue
                    idf = self._idf(form)
                    denominator = frequency + self.k1 * (
                        1 - self.b + self.b * length / (self._average_length or 1)
                    )
                    best = max(best, idf * frequency * (self.k1 + 1) / denominator)
                if not best:
                    continue
                score += best
                matched[term] = round(best, 4)

            if score > 0:
                missing = sorted(set(terms) - set(matched))
                # Coverage is `Hit.coverage`, computed from matched and missing; it is
                # the sort's primary key and no longer enters `ranking`.
                heading = self._heading_tokens[index]
                # Two shares, multiplied: how much of the QUESTION the heading answers,
                # and how much of the HEADING is the question. Either alone picks the
                # wrong provision. On its own, the share of the heading - which is what
                # this used to be - hands a perfect 1.0 to the shortest heading in the
                # book, so "Punishments" (s.53) beat "Qatl committed under ikrah-i-tam"
                # (s.303) on a question about qatl committed under duress, by two
                # thousandths. On its own, the share of the question ties "Definition of
                # the State" with "Islam to be State religion" on "what is the State?",
                # because both headings contain the one word asked. A heading that is
                # about the question and about little else beats both.
                asked = set(terms)
                answered = {q for q in asked if self._forms(q) & heading}
                used = {t for t in heading if any(t in self._forms(q) for q in asked)}
                heading_coverage = (
                    (len(answered) / len(asked)) * (len(used) / len(heading))
                    if asked and heading
                    else 0.0
                )
                hits.append(
                    Hit(
                        provision=provision,
                        score=round(score, 6),
                        matched_terms=matched,
                        missing_terms=missing,
                        heading_coverage=round(heading_coverage, 4),
                        # A term the corpus does not hold ANYWHERE weighs nothing.
                        # It cannot distinguish one provision from another, so its
                        # absence from this one says nothing about this one - and
                        # unweighted IDF gives it the maximum weight precisely
                        # because it is rare, which is backwards. "how many years
                        # does imprisonment for life count as?" refused on `count`
                        # and `many`: ordinary English, in no provision, and under
                        # pure IDF the two heaviest words in the question.
                        #
                        # A term that IS somewhere in the corpus and not here is the
                        # informative case, and `pardon` - one provision in
                        # twenty-six - is the one that matters.
                        term_weights={
                            term: round(max(self._idf(f) for f in self._forms(term)), 4)
                            for term in set(terms)
                            if any(self._document_frequency.get(f) for f in self._forms(term))
                        },
                        missing_qualifiers=sorted(set(missing) & QUALIFIERS),
                        unknown_terms=unknown,
                        tightness=round(self._tightness(index, set(matched)), 4),
                        ranking=round(
                            score
                            * (1 + self.heading_weight * heading_coverage)
                            * (1 + self.proximity_weight * self._tightness(index, set(matched))),
                            6,
                        ),
                    )
                )

        # Coverage first, then the ranking, then how much the heading is about the
        # question. A provision containing every word of the question is a better answer
        # than one containing most of them, whatever the term weights say - and among
        # provisions that cover it equally, the one the draftsman headed with those
        # words is the one about them.
        hits.sort(key=lambda h: (-h.coverage, -h.ranking))
        return hits[:limit]


@dataclass
class LawSearch:
    """Retrieval that is always anchored to a date.

    There is no way to search the corpus without supplying one. That is deliberate: an
    optional date parameter defaults to something, and the default eventually gets used
    for a question where it is wrong.
    """

    corpus: Corpus
    # Ranking parameters, passed to every index this builds.
    #
    # They used to be reachable only by editing BM25Index's defaults, and that made a
    # sweep silently measure nothing: BM25Index is a dataclass, so its defaults are
    # captured in __init__ and `BM25Index.b = 0.1` does not change a new instance. A
    # sweep written that way reported "b makes no difference from 0.0 to 0.75", which
    # was true of the experiment and said nothing about b.
    tuning: dict = field(default_factory=dict)
    # One index per date asked about. A long-running server is asked about arbitrarily
    # many dates, so the cache is bounded; the least recently used index is dropped.
    max_indexes: int = 32
    _indexes: OrderedDict[int, BM25Index] = field(default_factory=OrderedDict)
    _boundaries: list[dt.date] = field(default_factory=list)
    _boundaries_for: int = -1

    def _epoch(self, as_of: dt.date) -> int:
        """Which interval between consecutive commencements/repeals a date falls in.

        Every date in one interval sees exactly the same provisions, so they share an
        index: 2020-06-01 and 2021-03-15 are one index, not two.
        """
        # Keyed on the provision count, which `add` always changes. Replacing a
        # provision in `corpus.provisions` in place leaves the count alone and would
        # serve the old text under the right citation - so a corpus is built by `add`
        # and then read, and the list is not edited underneath a live LawSearch. The
        # server never does: its corpus is fixed for the life of the process.
        size = len(self.corpus)
        if self._boundaries_for != size:
            dates = {p.in_force_from for p in self.corpus}
            dates |= {p.in_force_to for p in self.corpus if p.in_force_to}
            self._boundaries = sorted(dates)
            self._boundaries_for = size
            self._indexes.clear()
        return bisect.bisect_right(self._boundaries, as_of)

    def _index_for(self, as_of: dt.date) -> BM25Index:
        key = self._epoch(as_of)
        if key in self._indexes:
            self._indexes.move_to_end(key)
        else:
            self._indexes[key] = BM25Index(**self.tuning).fit(self.corpus.as_of(as_of))
            while len(self._indexes) > self.max_indexes:
                self._indexes.popitem(last=False)
        return self._indexes[key]

    def search(
        self,
        query: str,
        *,
        as_of: str | dt.date,
        limit: int = 5,
        statute: str | None = None,
    ) -> list[Hit]:
        as_of = dt.date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
        where = (lambda p: p.statute == statute) if statute else None
        return self._index_for(as_of).search(query, limit=limit, where=where)

    def absent_terms(self, query: str, *, as_of: str | dt.date) -> list[str]:
        """Which of the query's content terms the corpus holds no word for on that date.

        Date-anchored like everything else here, and for the same reason: whether a
        word is in the statute book depends on which statute book - a provision
        repealed before the asked date is not in it. An undated version of this would
        name a word as absent because the only provision containing it was not yet in
        force.
        """
        as_of = dt.date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
        return self._index_for(as_of).absent_terms(tokenise(query))

    def by_citation(self, key: str, *, as_of: str | dt.date) -> Provision | None:
        """Look up a provision the user named directly.

        A question that cites a section is not a search problem — returning the
        *nearest* provision to a citation the user spelled out is how a system answers
        about the wrong law.
        """
        return self.corpus.version_on(key, as_of)
