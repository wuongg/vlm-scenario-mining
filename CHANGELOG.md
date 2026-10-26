# Changelog

All notable changes to this project are documented here.
Format: what changed and why.

## [Unreleased]

## [v0.1-day1] — 2026-10-26

- Scaffold the P0 repo (configs, ingest, geometry fixture, VLM schema, labeling tool, keyword index, CI) so Day 2 can freeze GT without renaming modules.
- Pin every numeric knob in `configs/*.yaml` and refuse magic strings; integrity and geometry gates run on synthetic fixtures in CI so nuScenes never enters GitHub Actions.
- Add a keyword-only fallback backend (not a baseline) plus detector/API skeletons so the mini-pipeline and `vsm search` have a real, schema-valid path on Day 1.
