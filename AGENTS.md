# Repository Guidelines

## Project Structure & Module Organization

This repository publishes an Obsidian knowledge base with Quartz 5. Author notes and site content live in `content/`, including the `foundations/` and `paper_reading/` sections; keep note-local images and other assets beside the notes that use them. The Quartz engine is vendored under `quartz/`, organized into components, plugins, processors, CLI helpers, utilities, styles, and static assets. Site behavior is configured in `quartz.config.yaml`; TypeScript settings are in `tsconfig.json`. Tests are colocated with implementation files as `*.test.ts`, `*.test.tsx`, or `*.test.js`. `public/` is generated output and must not be committed.

## Build, Test, and Development Commands

Use Node.js 22+ and npm 10.9.2+.

- `npm ci` installs the lockfile-pinned dependencies.
- `npm run plugins` installs plugins declared by the Quartz configuration.
- `npm run dev` builds and serves the site at `http://localhost:8080/` for live editing.
- `npm run build` runs the production build and writes to `public/` (plugin installation runs first).
- `npm test` runs the colocated tests through `tsx --test`.
- `npm run check` runs strict TypeScript checking and the Prettier check.
- `npm run format` rewrites supported files with Prettier.

## Coding Style & Naming Conventions

Use two spaces, no semicolons, a 100-character print width, and trailing commas, matching `.prettierrc`. Keep TypeScript strict and preserve the existing ESM/Preact JSX style. Name components and exported types in PascalCase; use camelCase for functions and variables. Give Markdown notes descriptive names and retain meaningful folder organization; Chinese and English paths are both used in `content/`.

## Testing Guidelines

Add or update a colocated test when changing Quartz behavior. Name tests after the module or behavior under test, then run `npm test` and `npm run check`. There is no repository-wide coverage threshold; ensure changed paths and relevant build behavior are exercised.

## Commit & Pull Request Guidelines

Use concise, imperative subjects with the established prefixes, such as `docs:`, `fix:`, `feat:`, `chore:`, or `build:`. Keep commits focused. Pull requests should explain the user-visible or developer-facing change, list validation commands and results, link an issue when applicable, and include screenshots for visual/site changes. Do not include generated `public/` output.

## Configuration and Publishing

Keep dependency versions in `package-lock.json` synchronized with `package.json`. Changes pushed to `main` are built and deployed by `.github/workflows/deploy.yml`; verify a local production build before publishing. Treat `quartz.config.yaml` and plugin configuration as the source of truth for site behavior.
