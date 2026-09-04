# Agent Instructions

This file defines how automated coding agents should work in this repository.
It is not the product documentation. Use it as a local operating guide, then
follow the linked docs for repository behavior and user-facing contracts.

## Working Philosophy

- Read before editing. Let the existing code, tests, and docs define the local
  style before adding new patterns.
- Prefer small, direct changes over broad rewrites. Refactor only when it
  removes real ambiguity, duplication, or stale compatibility surface.
- Treat generated files as disposable artifacts. Source robot descriptions come
  from the pinned `deps/bello-stack` submodule; generated assets and caches do
  not belong in Git.
- Keep compatibility only when it is intentional. Do not add aliases, fallback
  paths, or duplicate config routes unless the user explicitly asks for them.
- Make behavior explicit. Silent fallbacks, inert CLI flags, and hidden defaults
  should be removed or validated.
- Keep the training hot path JAX/MJX-Warp compatible. Python-side convenience is
  fine around setup, assets, logging, and tests, but not inside rollout math.
- Favor frozen dataclass configs decoded from the public Hydra documents. Avoid
  ad hoc parallel CLI arguments when a config object already exists.
- Use the Nix/uv environment for repo work. Prefer `nix develop`, direnv, and
  `uv run ...` over system Python or globally installed tools.
- Keep all changes inside this repository unless the user explicitly asks for an
  external change.

## Coding Style

- Use structured parsers for URDF/MJCF/XML edits. Avoid string surgery on robot
  descriptions when `xml.etree.ElementTree` or MuJoCo APIs can express the
  change.
- Write Pythonic code: prefer clear names, simple control flow, standard library
  types and helpers, dataclass defaults with `field(default_factory=...)` where
  needed, and small functions with explicit inputs over clever or implicit code.
- Prefer clear validation at initialization time for bad config, missing links,
  missing joints, incompatible assets, and invalid dimensions.
- Keep dataclasses immutable where practical. Do not expose mutable config
  internals directly.
- Keep comments sparse and useful. Add comments only where the code would be
  difficult to audit without local context.
- Use ASCII unless a file already requires non-ASCII content.

## Repository Rules

- The asset pipeline's canonical robot input is
  `deps/bello-stack/descriptions/bello/urdf/bello.urdf`.
  Treat the submodule as read-only here: change descriptions in `bello-stack`,
  then advance the pinned gitlink deliberately.
- Do not add asset aliases. Valid asset names are documented in
  [docs/index.md](docs/index.md).
- Do not commit generated assets, `.venv`, uv caches, Warp caches, W&B runs, or
  checkpoint output.
- Update user-facing docs when CLI flags, config trees, env spaces, rewards,
  asset behavior, training behavior, or workflow commands change.
- Do not revert user changes unless explicitly requested. If the worktree is
  dirty, identify which edits are yours before committing.

## Important Docs

- Start with [README.md](README.md) for the high-level project shape and common
  commands.
- Use [CONTRIBUTION.md](CONTRIBUTION.md) for contribution workflow and git
  hygiene.
- Use [docs/index.md](docs/index.md) as the entry point for repository docs.
  Follow links from there for topic-specific docs under `docs/`.
