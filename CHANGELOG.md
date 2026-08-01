# Changelog

All notable changes to this project will be documented in this file.

## [v1.4.12] - 2026-08-01

CI plumbing only. No code in this repo changed — every commit in this release
touches `.github/workflows/`.

### Changed

- The pipeline was split: building and publishing stay in `pipeline.yml`, and
  everything that leaves the host now lives beside it in
  `mirror-and-archive.yml`.
- The repo is mirrored to Codeberg as well as GitLab.
- It is archived to the Wayback Machine, Software Heritage and archive.org.
- Issues opened on either mirror are copied back to GitHub every six hours, and
  closed here when the original closes.
- Pull requests are switched off on the mirrors — they are force-pushed from
  GitHub, so anything merged there would be destroyed by the next sync. Issues
  and forking stay enabled.

## [v1.4.11] - 2026-07-27

### Fixed

- The Codex subsection of the README's "Agent integrations" section was missing its
  install command. It now reads `codex plugin add qwenspeak@psyb0t` after the
  marketplace-add step, matching the Claude Code subsection above it.
- Clarified that the skill's invocation form depends on how it was picked up:
  installed via the marketplace it's `$qwenspeak:qwenspeak`, while Codex's automatic
  pickup from a repo's own `.agents/skills/` (no install needed) invokes it as plain
  `$qwenspeak`.

## [v1.4.10] - 2026-07-27

### Added

- Added `.agents/.claude-plugin/plugin.json` and `.agents/.codex-plugin/plugin.json`
  so the existing `qwenspeak` skill installs natively as a plugin in Claude Code and
  Codex.
- Added an "Agent integrations" section to the README with the install commands for
  Claude Code, Codex, and the OpenClaw skill.

## [v1.4.9] - 2026-07-27

### Added

- Added a GitHub Actions CI status badge to the README.

## [v1.4.8] - 2026-07-27

### Added

- Added self-hosted version and license badges plus a Docker Hub pulls badge; wired a badges job into pipeline.yml.

## [v1.4.7]

### Security

- Hardened the `qwenspeak` skill docs with explicit voice-cloning
  consent/privacy guidance and an external-transmission warning for the
  SSH wrapper. No behavior change — documentation only.
