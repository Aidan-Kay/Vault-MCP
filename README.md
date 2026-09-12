# Vault MCP

An MCP server and REST API over a local [Obsidian](https://obsidian.md) vault. It
indexes the vault for hybrid search and serves read and write access to it through
two interfaces backed by one implementation.

## Why

Two problems with the plugin it replaces:

- **Heading targets had to be exact.** It keys every heading by its full ancestor
  path and does a single lookup, so anything short of the complete path from the H1
  down matches nothing. 86% of this vault's notes are wrapped in one H1, which makes
  nearly every useful target a two- or three-segment path the model has to guess up
  front. Here a bare leaf name works whenever it is unique, and when it is not, the
  error names the ancestors to prepend.
- **No retrieval.** Finding a note meant knowing its path. `vault_search` is hybrid —
  dense vectors *and* BM25, fused with reciprocal rank fusion. Hybrid is not optional
  for this corpus: it is dense with exact tokens (reg plates, boiler model numbers,
  policy references, postcodes) where dense retrieval alone underperforms.

## Interfaces

**MCP** at `/mcp` — ten tools:

| Read | Write |
| --- | --- |
| `vault_search` — hybrid search | `vault_patch` — replace a section |
| `vault_read` — whole note or one `section=` | `vault_append` — add to the end |
| `vault_list` — browse a folder | `vault_write` — create or overwrite |
| `vault_map` — heading tree as `::` paths | `vault_set_frontmatter` — set or delete a key |
| | `vault_delete` — remove a note |
| | `vault_move` — move, rewriting inbound links |

`vault_map` emits `::`-joined paths rather than an indented tree, because the output
is meant to be pasted straight back as a patch target.

**REST** at `/vault/<path>` — `GET`, `PUT`, `POST`, `PATCH`, `DELETE`, mirroring the
shape `obsidian-local-rest-api` used. n8n's HTTP Request nodes speak plain REST and
cannot easily build a JSON-RPC envelope, so migrating a node is a find-and-replace on
the URL and the auth header rather than a rewrite into JSON-RPC.

Both surfaces call `src/operations.py`, so the resolver and the vault conventions are
applied once regardless of how the caller arrived. Every write bumps the note's
`timestamp`, or reports why it could not.

## The index is generated

The vault's root `index.md` is one line per note — its title, a link, and its
`description` — under headings that mirror the folder tree. It used to be written by
hand, which made it the one convention every write depended on a model remembering,
and the one it forgot: an entry whose description no longer matched the note, a new
note that never got a line, a moved note still listed at its old path.

Nothing in that document is a judgement call. The heading is the folder, the title and
description are the note's own frontmatter, and the order is fixed — a folder's
landing note first, then the rest by title. So it is derived rather than authored, and
`src/indexdoc.py` derives it.

It is rebuilt from the **filesystem watcher**, not from the write path, so it does not
matter how the change arrived: an MCP tool call, a REST `PUT` from n8n, or someone
typing in Obsidian on the desktop all reach it the same way. A full scan runs once at
startup — catching whatever moved while the container was down — and each change after
that re-reads a single note, which is milliseconds rather than the few seconds a walk
of the whole vault costs across the mount.

Two properties it is worth knowing are deliberate:

- **It only writes when the rendered body differs.** The vault is in git, and a
  document that rewrote itself on every note edit would bury its own history under
  commits whose only change is a timestamp. Editing a note's body does not touch it;
  editing that note's `description` does.
- **`index.md` is protected from every writer.** A write to it is not dangerous, it is
  futile — the next change to any note overwrites it — and a tool that accepts a write
  it is about to discard teaches the caller the edit worked. Fix a wrong line by
  fixing the note's frontmatter. It stays readable.

Generated note series — the n8n workflow folders listed in `INDEX_DOC_EXCLUDE` — are
not indexed note by note; the approvals folder alone would swamp the document. Each
gets one line in its parent section saying so, and only when the folder actually
exists.

**Scoped writes** at `/mcp/only/<path>` — the same MCP surface with this request's
writes confined to one note (`/mcp/only/Workflows/Approvals/x.md`) or one folder
(`/mcp/only/Workflows/Approvals`). Reads are never scoped: an agent confined to one
note still has to read the conventions and whatever that note refers to. `vault_move`
is refused outright while a scope is set, because rewriting inbound links touches every
note that points at the source.

It rides on the URL rather than a header because that is the part a caller can vary per
call — n8n's MCP Client node takes its auth from a static credential but its endpoint
from an expression — so one agent with one tool list can be handed a different remit per
invocation, with no second copy of the workflow to keep in step. The scope cannot
outlive its request: the transport is stateless, and a `ContextVar` keeps concurrent
requests from seeing each other's.

This exists because an agent told in prose to "carry nothing out" replaced a section of
the vault's root `index.md` while revising an unrelated note. A sentence in a prompt is
not a guard.

## Configuration

All configuration is environment variables. `VAULT_MCP_API_KEY` is required — the
server refuses to start without it rather than treating an empty key as "auth off".

| Variable | Default | Purpose |
| --- | --- | --- |
| `VAULT_MCP_API_KEY` | — | Bearer token. **Required.** |
| `VAULT_PATH` | `/vault` | Vault root inside the container |
| `MCP_ALLOWED_HOSTS` | `vault-mcp:8080,127.0.0.1:8090` | Host-header allowlist |
| `OLLAMA_URL` | `http://ollama:11434/v1` | OpenAI-compatible embedding endpoint |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBED_DIMS` | `768` | Embedding dimensions |
| `EMBED_BATCH_SIZE` | `64` | Embedding requests per batch |
| `EXCLUDE_DIRS` | `Workflows,Reports,.obsidian` | Folders left out of the search index |
| `INDEX_DOC_EXCLUDE` | the six generated series | Folders left out of `index.md` |
| `CHUNK_TARGET_TOKENS` | `400` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `60` | Overlap between chunks |
| `CHUNK_MIN_TOKENS` | `120` | Below this, a chunk merges into its neighbour |
| `SEARCH_DEFAULT_K` | `5` | Default result count |
| `WATCH_DEBOUNCE_SECONDS` | `2.0` | Filesystem-watch debounce before reindexing |
| `BIND_HOST` / `BIND_PORT` | `0.0.0.0` / `8080` | Listen address |

`EXCLUDE_DIRS` and `INDEX_DOC_EXCLUDE` answer different questions and must not be
merged. The first drops `Workflows/` and `Reports/` from search wholesale; the second
cannot, because curated notes live inside both — `Workflows/Email Triage/Rules.md` and
the `Reports/PC/` reports are navigated even though they are not searched.
`INDEX_DOC_EXCLUDE` mirrors the "Excluded folders" table in the vault's
`Meta/Conventions.md`; that table, this variable, `.scripts/check_frontmatter.py` and
the vault's `.gitignore` are four copies of one list and have to move together.

The search index is built at startup and kept current by a filesystem watcher, so an
edit made in Obsidian is searchable a moment later without a restart. The watcher
feeds `index.md` too, and does so independently: search needs Ollama and can be slow
or unavailable, while the navigation document needs neither and must not stop updating
because an embedding endpoint is down.

## Running

The image clones this repository at build time, so the build context holds only the
Dockerfile:

```bash
docker build -t vault-mcp .
docker run --rm \
  -e VAULT_MCP_API_KEY=<token> \
  -v /path/to/vault:/vault \
  -p 8080:8080 \
  vault-mcp
```

Docker caches the clone layer on the URL alone, so a new commit on `main` does not
invalidate it — rebuild with `--no-cache` to pick one up.

## Security

- **Bearer auth on both surfaces**, failing closed on an unset key.
- **Path containment** in `safe_resolve()` — the single control on where writes land,
  since the vault is mounted read-write. Encoded traversal, `.git`, `index.md` and
  non-`.md` writes are all rejected.
- **Host-header allowlist**, so the MCP transport is not reachable by DNS rebinding.

## Tests

Standalone scripts, no test runner:

```bash
python -m tests.primitives
python -m tests.resolve_all
python -m tests.resolve_leaves
python -m tests.write_scope
python -m tests.indexdoc
```

The last two write, so they build their own temp vault rather than touching the real
one. `tests.indexdoc` covers the generated document: coverage, folder-derived
headings, incremental updates on create, edit, move and delete, that `index.md` is
refused to every writer and still readable, and that an edit changing nothing the
index displays does not rewrite it.
