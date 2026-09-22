---
name: research-to-blog
description: "Read local research materials or public URLs, compare them with existing notes, and prepare a source-grounded new-article or supplement proposal for human review. Use only when explicitly invoked as $research-to-blog."
---

# Research to blog

Use this skill only for an explicit `$research-to-blog` request. It turns one or more local research materials into a reviewable proposal for this repository's Obsidian/Quartz knowledge base.

## Input

- If the user supplies local paths, process those paths only. A path may be a single file or a directory.
- If the user supplies one or more `http://` or `https://` URLs, treat each URL as a research source. Read the public page or PDF with the appropriate web/PDF capability, resolve a canonical URL and stable identifier when possible, and preserve both the original URL and the canonical URL in the proposal and manifest.
- A request may mix local paths and URLs. Do not interpret a URL as a filesystem path.
- Do not silently download remote material into `raw/`. If the user wants a local archival copy, ask them to save it under `raw/inbox/` or explicitly request that the source be cached.
- If no path is supplied, scan `raw/inbox/` for materials that have not already been handled.
- Treat `raw/` as a local source library. Do not publish, move, rename, or delete raw files during analysis.
- Ignore `.DS_Store`, temporary files, generated output, and files outside the requested path.
- For an inbox scan, use `research/manifest.jsonl` when it exists. Treat a material as handled when its content hash or stable source identifier has a terminal status such as `proposed`, `applied`, `archived`, or `skipped`; also check existing proposal files when no manifest exists.
- After a successful analysis, record the source hash when available, original and canonical URLs for remote sources, the stable identifier, proposal ID, and status in `research/manifest.jsonl`. Keep this state separate from the raw file and from canonical published notes.
- Before reading a large batch, report which files will be processed and group closely related files when that improves comparison.

## Repository-specific routing

The repository keeps two kinds of published notes:

- `content/paper_reading/` for a paper or technical-report-centered explanation;
- `content/foundations/` for reusable concepts, algorithms, systems, and cross-source syntheses.

Inspect existing notes, their frontmatter, headings, tags, wikilinks, and index pages before deciding where a result belongs. Existing Markdown conventions are mixed; preserve the style and frontmatter fields of the target note instead of normalizing unrelated files.

## Research workflow

1. Identify each material's title, authors, date, version, type, canonical public URL, and stable identifier when available (arXiv ID, DOI, repository commit, or equivalent). For a URL input, distinguish facts read from the linked source from metadata inferred from the URL or surrounding pages.
2. Read the material closely enough to separate its claims, evidence, implementation details, limitations, and open questions. For PDFs, keep page or section locators; for web or repository sources, keep stable URLs and commit/version identifiers.
3. Build a small evidence ledger. Every material claim proposed for publication must have a source locator and a confidence level. Mark inferences as inferences and unresolved facts as open questions.
4. Search the existing knowledge base for related notes. Compare the material with candidate notes at the level of problem, mechanism, evidence, and conclusion. Do not decide from title similarity alone.
5. Choose one outcome:
   - `new_article`: the material introduces an independent question or a sufficiently distinct explanation;
   - `supplement`: it adds evidence, mechanism, comparison, or a correction to an existing note;
   - `skip`: it is duplicative or does not add a publishable insight;
   - `needs_review`: source identity, evidence, or relationship to existing notes is too uncertain.
6. Write a proposal under `research/proposals/`. Use a stable slug or source identifier in the filename. Include the proposed target path, decision, outline or exact section anchors, evidence ledger, source URLs, uncertainty notes, and the full proposed Markdown or a focused patch.

The proposal is the stopping point for the normal mode. Do not edit canonical files in `content/paper_reading/` or `content/foundations/`, do not commit, and do not publish while preparing a proposal.

## Review and apply

Only modify canonical content when the user explicitly names an existing proposal and asks to apply it. Before applying:

- re-read the target note and the proposal;
- preserve unrelated text and make the smallest coherent edit;
- use public source URLs in the article, never `raw/` paths, local absolute paths, or `file://` links;
- keep citations close to the claims they support and retain page/section locators where useful;
- update an index page only when the new or changed note belongs there;
- set or preserve `draft: true` when the user asks to keep the result as a draft.

After applying, inspect the diff and run the checks that are practical for the change, normally `npm run check` and `npm run build`. Report any failed check and leave the working tree reviewable.

## Proposal format

Use frontmatter similar to:

```yaml
---
proposal_id: arxiv-2609.20807-v1
status: needs-review
decision: supplement
source_files:
  - raw/inbox/score-centering.pdf
source_urls:
  - https://arxiv.org/abs/2609.20807
target_notes:
  - content/foundations/RL/系统/训练稳定性/精度/训推一致性/训推一致性诊断：从概率偏差到根因定位.md
---
```

The body should contain, in an order that is easy to review:

1. decision and rationale;
2. concise material summary;
3. related-note comparison;
4. evidence ledger with source locators;
5. proposed outline or patch;
6. risks, contradictions, and questions requiring human judgment.

For URL-only inputs, omit `source_files` and keep `source_urls` populated. For mixed inputs, include both fields.

For a new article, write a complete draft suitable for the selected content directory. For a supplement, prefer a focused patch with exact target headings and explain why the insertion belongs there. Avoid rewriting a long existing article when a local section change is sufficient.

## Writing and evidence quality

Use clear Chinese prose consistent with the surrounding notes. Explain the problem, core mechanism, evidence, benefits, limitations, and practical implications when the material supports them. Distinguish what the source states from the agent's synthesis. Do not invent experiments, metrics, authorship, dates, or implementation details. If a source is unavailable or a PDF is image-only, state the limitation and use `needs_review` rather than filling gaps from memory.
