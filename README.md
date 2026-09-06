# Vault MCP

An MCP server and REST API over a local [Obsidian](https://obsidian.md) vault. It
indexes the vault for hybrid search and serves read and write access to it through
two interfaces backed by one implementation.

It began as a retrieval layer — hence the original `vault-index` name — and now
covers the write path as well, replacing `obsidian-local-rest-api` as the way the
vault is edited programmatically.

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
| `EXCLUDE_DIRS` | `Workflows,Reports,.obsidian` | Folders left out of the index |
| `CHUNK_TARGET_TOKENS` | `400` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `60` | Overlap between chunks |
| `CHUNK_MIN_TOKENS` | `120` | Below this, a chunk merges into its neighbour |
| `SEARCH_DEFAULT_K` | `5` | Default result count |
| `WATCH_DEBOUNCE_SECONDS` | `2.0` | Filesystem-watch debounce before reindexing |
| `BIND_HOST` / `BIND_PORT` | `0.0.0.0` / `8080` | Listen address |

The index is built at startup and kept current by a filesystem watcher, so an edit
made in Obsidian is searchable a moment later without a restart.

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

Docker caches the clone layer on the URL alone, so a new commit on `master` does not
invalidate it — rebuild with `--no-cache` to pick one up.

## Security

The vault holds finances, insurance and addresses in plain text. The controls are:

- **Bearer auth on both surfaces**, failing closed on an unset key.
- **Path containment** in `safe_resolve()` — the single control on where writes land,
  since the vault is mounted read-write. Encoded traversal, `.git` and non-`.md`
  writes are all rejected.
- **Host-header allowlist**, so the MCP transport is not reachable by DNS rebinding.
- **Non-root uid 1000**, matching the vault's file ownership so written notes keep
  the ownership Samba expects.

## Tests

Standalone scripts, no test runner:

```bash
python -m tests.primitives
python -m tests.resolve_all
python -m tests.resolve_leaves
```
