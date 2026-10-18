# Content provenance (C2PA) vs. after-the-fact detection

Detection (what VerityAI does) and provenance (a different, complementary
approach) solve related but distinct problems, and the two are often
confused.

**Detection** looks at a piece of media *after the fact*, with no
information about its origin, and tries to infer from the pixels/frames
themselves whether it was likely generated or manipulated. It always
works probabilistically and degrades as generators improve — this is the
whole subject of document 02 ("generalization to unseen generators").

**Provenance** works the opposite way: it attaches cryptographically
signed metadata ("Content Credentials") to media *at the moment of
capture or creation*, recording facts like which camera or generative
tool produced it, and what edits were applied afterward. The Coalition
for Content Provenance and Authenticity (C2PA) — a cross-industry
standards body — publishes the open technical specification for this,
and tools and cameras implementing it can attach a verifiable, tamper-
evident history to a file. If present and intact, this metadata is a much
stronger authenticity signal than any pixel-based detector, because it
does not have to guess — it reads a cryptographic assertion.

The catch is coverage: provenance metadata is only present if the
capture device or generation tool chose to attach it, and the
metadata can be stripped (deliberately or as a side effect of re-encoding,
cropping, or uploading to a platform that discards metadata). Most media
encountered in practice — especially anything that has been
screenshotted, re-saved, or shared across multiple platforms — simply has
no Content Credentials to check, which is exactly the situation where a
detection-based tool like VerityAI is the only available signal, weak as
it is.

VerityAI does not implement C2PA verification; it is a pixel-based
detector only. This document exists so the Research Assistant can
accurately explain the difference when asked, and so VerityAI's own
reports never imply that a "Real suspected" result is equivalent to a
verified provenance record — it is not.

Source: Coalition for Content Provenance and Authenticity (C2PA).
Content Credentials: C2PA Technical Specification.
https://c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html
