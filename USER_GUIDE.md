# Unity Catalog Metadata Assistant — User Guide

A simple guide to the app’s core features and how to use them. **Tip:** The app logs (in your Databricks app run / driver logs) are detailed and transparent—check them when something doesn’t behave as expected or when you want to see exactly what the app is doing.

---

## Before you start

- **Select a catalog** in the top bar. Most tabs (Overview, Generate, Review, Improve, Quality, History) work on the selected catalog.
- **Settings** and **Copy Metadata** are global; Copy lets you pick source and target catalogs in the tab.

---

## Core features (by tab)

### Overview

**What it does:** Shows how well your catalog is documented.

- **Stat cards:** Counts of schemas, tables, and columns—and how many are missing descriptions or comments.
- **Coverage:** Progress bars for schema, table, and column coverage.
- **Recommended actions:** Short list of what to focus on (e.g. “Focus on column documentation”).
- **Top gaps:** Table of objects with the biggest documentation gaps.

Use it to see where you stand and where to run **Generate** or **Improve**.

---

### Generate

**What it does:** Uses AI to write descriptions for schemas, tables, and columns that don’t have them yet.

1. **Select objects:** Use filters (schemas, tables, columns) and the checklist to choose what to generate. You can select all or pick specific objects.
2. **Choose model(s):** Pick one or more LLM models (e.g. Claude, Gemma). Multiple models run in parallel; you’ll compare them in **Review**.
3. **Start run:** Click to start. The app runs in the background and shows progress (objects processed, ETA). You can cancel if needed.
4. When the run finishes, go to **Review & Commit** to see and commit the results.

Generation uses schema context and optional table sampling. PII detection can tag sensitive columns. All of this is configurable under **Settings**.

---

### Review & Commit

**What it does:** Lets you review AI-generated descriptions (and tags), edit them, and write them to Unity Catalog.

- **Single-model view:** One card per object with description, confidence, and optional tags. Edit, select/deselect, then commit selected.
- **Comparison view:** If you ran multiple models in **Generate**, you can switch to comparison mode: for each object you see every model’s output, pick the one you want (or edit it), then commit.
- **Tags:** You can add or change governed tags and choose “Apply these tags when committing” per object.
- **Actions:** Commit selected, download CSV, remove items from the list.

Only items you select and commit are written to the catalog. If something fails, check app logs for the exact error (e.g. permissions, object not found).

---

### Copy Metadata

**What it does:** Copy descriptions (and optionally tags) between catalogs or export/import via CSV.

- **Source & target:** Pick source and target catalogs. The app finds objects that have descriptions in the source.
- **Smart match:** Suggests mappings (e.g. bronze → silver → gold by name). You can adjust and then run the copy.
- **Export:** Export metadata to CSV. You can limit by catalog and, optionally, by **schemas** and **tables** (only objects that have descriptions). Handy for backup or moving metadata to another workspace.
- **Import:** Upload a CSV, validate, preview, then apply to a catalog. Useful for bulk updates or restoring from backup.

Copy only affects objects you explicitly include in the copy or import; it does not delete or overwrite unrelated metadata.

---

### Improve

**What it does:** Improves *existing* descriptions (quality scan → select objects → generate improvements → review and commit).

1. **Select objects:** Same idea as Generate—choose schemas, tables, and/or columns to improve.
2. **Scan quality:** The app scores current descriptions (e.g. poor / marginal / good) and lists issues (e.g. too short, vague).
3. **Choose what to improve:** Use filters (quality, type) and the top FilterBar (object type, search). Select the objects you want to improve.
4. **Choose model(s):** Pick one or more models. They run in parallel; you’ll compare in the next step.
5. **Generate:** Start the run. Progress is shown; you can cancel.
6. **Review & compare:** Each object appears as an expandable card: **Current** description and **AI-generated options** (one per model). Pick the version you want, edit if needed, then commit.

Improve only updates the objects you select and commit. App logs show which objects were processed and any errors.

---

### Quality

**What it does:** Gives a quality and risk view of your metadata.

- **Metrics:** Coverage, PII exposure (undocumented sensitive fields), review backlog (older objects without documentation).
- **Charts:** PII risk matrix (sensitivity vs. documentation), confidence distribution of AI-generated content, completeness trends.
- **Completeness trend:** How coverage has changed over time (e.g. last 90 days).

Use it to prioritize what to document or improve and to track progress. If numbers look off, logs can help confirm what was queried and how metrics were computed.

---

### History

**What it does:** Shows an audit trail of metadata operations (generation, copy, commits, etc.).

- **Filters:** Time range, object type, and search so you can narrow to a catalog, schema, or object.
- **Table:** Date, object, type, action, changes, and source (e.g. which model or “Manual”). Pagination for large histories.
- **Clear history:** Optional cleanup of old records (e.g. by age or count) to keep the table manageable.

History is stored in the app catalog; logs show when history is written and any cleanup runs.

---

### Settings

**What it does:** Central place to configure how the app behaves.

- **Models:** Enable/disable LLM models, add custom endpoints, set default model and parameters (e.g. max tokens).
- **Sensitive data (PII):** Enable/disable PII detection, choose patterns, configure tag mappings to governed tags.
- **Tags:** Governed vs. manual tag policy.
- **Sampling:** Turn sampling on/off and set sample size for table data used in generation.
- **Prompts:** Adjust style (concise, technical, business), terminology, and custom instructions for generation.

Changes apply to future runs (Generate, Improve). App logs often reference the active settings (e.g. which model or sampling config was used).

---

## Logs and troubleshooting

The app writes clear, structured logs (in your Databricks app / driver logs). They include:

- Which catalog and objects are being processed
- Generation and improvement progress (e.g. “Processing schema X”, “Generated N descriptions”)
- API and permission errors (e.g. 403, object not found)
- Settings in use (e.g. model, sampling)

**We encourage admins to review app logs** when:

- A run fails or seems stuck
- Results are missing or unexpected
- You want to verify what the app did (e.g. for compliance or debugging)

Logs are the single best source of truth for what the app did and why something might have failed.

---

## Quick reference

| Tab        | Purpose |
|-----------|---------|
| **Overview** | Coverage and gaps; where to focus. |
| **Generate** | AI-generate descriptions for objects that don’t have them. |
| **Review & Commit** | Review, edit, and commit generated descriptions (and tags); compare multiple models. |
| **Copy Metadata** | Copy between catalogs; export/import CSV; optional schema/table scope for export. |
| **Improve** | Scan quality, generate improved descriptions, compare models, commit. |
| **Quality** | Quality and PII risk metrics, trends, dashboards. |
| **History** | Audit trail of metadata operations. |
| **Settings** | Models, PII, tags, sampling, prompts. |

---

*For setup, permissions, and deployment, see the main README
