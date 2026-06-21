# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

# Why SmartWrapperOSS exists

## The problem, in practice

Before building this, I started building **SmartWrapper** — a commercial 
product that generates structured summaries of corporate financial 
commentary (earnings calls, filings, announcements) using LLMs. It's a 
narrow, domain-specific product: given a financial document, produce a 
summary that's accurate enough for someone to act on.

Building that product surfaced a problem I didn't expect to spend so
much time on: **picking the right model and the right orchestration
approach was mostly guesswork.**

Concretely, every time a new model came out, or a new orchestration
pattern got attention (single-pass prompting vs. a draft-then-review
loop vs. a multi-agent conversation), I faced the same questions:

- Is this actually better for *my* task, or just better on whatever
  benchmark the model card cites?
- Is "better" worth the extra latency or cost it usually comes with?
- Is the orchestration framework adding real value, or just more
  surface area for things to fail?

I answered these questions the way most people building on LLMs
currently do: ad hoc. Run it a few times, eyeball the outputs, pick the
one that "feels" better, move on. That's not a measurement process —
it's a vibe check with extra steps.

## What I looked for, and didn't find

I went looking for a tool that would let me:

1. Take a real task (not a generic public benchmark — *my* task, with
   *my* documents)
2. Run it through multiple models and multiple orchestration approaches
   at once
3. Get back actual numbers — not just "this one read better to me" —
   covering quality, how completely it covered the source, how concise
   it was, how long it took, and what it cost
4. Do this without hand-rolling a one-off script every single time I
   wanted to compare something

I didn't find one. There are excellent eval frameworks for specific
narrow things (classification accuracy, retrieval precision, general
LLM leaderboards), and excellent orchestration frameworks (AutoGen,
LangGraph, and others) — but nothing that put "run this real task
through multiple frameworks and multiple models, then score and compare
the results side by side" together as one packaged tool.

So I built the eval harness I wished I'd had — first for myself, as
part of figuring out SmartWrapper's own model and architecture choices,
and then, once it was useful enough to be worth sharing, as its own
standalone open source project: **SmartWrapperOSS**.

## Why open source it, rather than keep it internal

A few reasons, in order of how much they actually mattered:

- **The problem isn't specific to financial summarization.** Anyone
  building on LLMs runs into the same "which model, which orchestration
  pattern" question, regardless of domain. A general-purpose evaluator
  is more useful to more people as a shared, neutral tool than as
  something locked inside one company's internal tooling.
- **Neutral evaluation tools need to actually be neutral.** A tool that
  scores models and frameworks is only trustworthy if it has no
  incentive to favor one outcome over another. Keeping it open, with
  visible scoring logic anyone can audit or challenge, is part of what
  makes the scores worth trusting — see the anti-bias judge rotation in
  the summarization evaluator as one concrete example of taking this
  seriously.
- **It genuinely benefits from outside use and contributions.** Every
  new model, new framework, and new task type someone adds makes the
  tool more useful for everyone, including for SmartWrapper's own future
  model decisions — open development compounds faster than something
  built and used by one person.

## Where SmartWrapper (the financial product) and SmartWrapperOSS (this
project) actually relate — and where they don't

- They share a name and an origin story, not a codebase. SmartWrapperOSS
  is a general-purpose evaluator; it has no financial-domain logic, no
  access to SmartWrapper's data, and no dependency on it.
- Utility code is only ever shared in one direction: this project's open
  code can be reused inside SmartWrapper (permitted under Apache 2.0 —
  see the LICENSE), but nothing proprietary from SmartWrapper goes the
  other way into this public repository.
- See the "What this is not" section of the [README](./README.md) and
  the Disclaimer for the maintainer's commercial interests, stated
  plainly so nobody has to wonder.

## What's next

The first workflow built was Summarization, directly inherited from the
SmartWrapper use case. The second, Tool-Use (function-calling
benchmarking), was added to test whether the underlying architecture
actually generalized to a structurally different kind of task — see
`workflows/base.py` for how that contract works, and the Enhancements
document for ideas on where both could go next.
