# Contributing to SmartWrapperOSS

Thanks for your interest in contributing — solo and small contributions
are exactly what this project is built to welcome.

## Before you start

1. **No CLA required.** This project uses a lightweight **Developer
   Certificate of Origin (DCO)** instead of a Contributor License
   Agreement. You just add `-s` to your commits:

   ```bash
   git commit -s -m "Add Gemini-Pro adapter to model router"
   ```

   See [DCO.md](./DCO.md) for the full text and rationale. A bot will
   check this automatically on your pull request — no separate signing
   portal or paperwork.

2. **Know what you're contributing to.** This repository is, and will
   remain, fully open source under Apache 2.0 — see the
   ["Project model" section in the README](./README.md#project-model-open-core--hosted-service)
   for details. A separate, closed-source hosted product may be built on
   top of this open core and offered commercially for profit. Your
   contributions to this repository stay in the open core under Apache
   2.0; they are never moved into the closed-source product. If that
   model doesn't work for you, this is a good time to say so — open an
   issue or discussion and ask.

3. **Check open issues** for things tagged `good-first-issue` or
   `help-wanted` if you're not sure where to start.

4. **For larger changes** (a new orchestration framework, a new scoring
   dimension, a new model provider integration), open an issue to
   discuss the approach before submitting a large PR — saves you and
   reviewers rework later.

## Local setup

```bash
git clone https://github.com/yourname/smartwrappeross
cd smartwrappeross
cp config/config.yaml.example config/config.yaml   # add your API keys
docker compose up
```

See [`docs/setup.md`](./docs/setup.md) for environment variables (model
API keys, GCS bucket config) and local-only setup without Docker.

## Where the easy wins are right now

- **New model provider adapters** — the model router is config-driven;
  adding a new OpenAI-compatible endpoint should require no core changes,
  just a new entry in `config/models.yaml` and a short adapter if the
  endpoint isn't OpenAI-compatible.
- **New orchestration frameworks** — currently AutoGen and LangGraph are
  supported; a third framework (e.g., CrewAI) would be a high-value,
  self-contained addition.
- **Additional scoring dimensions** — the evaluation engine is built to
  support more than five dimensions; if you have an idea (e.g.,
  toxicity, readability grade level, citation accuracy), the
  `evaluators/` interface is documented in
  [`docs/evaluators.md`](./docs/evaluators.md).
- **Judge prompt tuning** — if you find cases where the LLM-as-judge
  scoring seems inconsistent or biased, both the prompts and the
  anti-bias rotation logic live in `evaluators/judge.py` and are easy to
  iterate on.

## PR checklist

- [ ] Commits are signed off (`git commit -s`) per the [DCO](./DCO.md)
- [ ] Code includes the standard copyright header (see
      [`COPYRIGHT_HEADERS.md`](./COPYRIGHT_HEADERS.md))
- [ ] Tests pass locally (`pytest` / see `docs/testing.md`)
- [ ] New config options are documented in `docs/setup.md`

## Questions

Open a [Discussion](../../discussions) for design questions, or an
[Issue](../../issues) for bugs and concrete feature requests.
