# Changelog

All notable changes to this project will be documented in this file.

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
