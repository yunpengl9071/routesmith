# RouteSmith CLI Reference

All commands are invoked via `routesmith <command> [options]`.

## Global Flags

| Flag | Description |
|------|-------------|
| `--version`, `-v` | Show version and exit |

---

## `routesmith quickstart`

Single-command setup. Detects provider API keys in environment, generates `routesmith.yaml`, prints curl/Python/Anthropic SDK snippets.

| Option | Default | Description |
|--------|---------|-------------|
| `--port`, `-p` | `9119` | Port for the proxy server |
| `--yes`, `-y` | — | Overwrite existing config without prompting |

Detects keys in order: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. Exits 1 if none found.

```bash
export OPENROUTER_API_KEY=sk-or-...
routesmith quickstart --yes
routesmith serve
```

---

## `routesmith init`

Interactive setup. Fetches the OpenRouter model catalog with live pricing and generates `routesmith.yaml`.

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | `routesmith.yaml` | Output config file path |
| `--force`, `-f` | — | Overwrite existing config file |

---

## `routesmith serve`

Start the OpenAI-compatible proxy server.

| Option | Default | Description |
|--------|---------|-------------|
| `--port`, `-p` | `9119` | Port to listen on |
| `--host`, `-H` | `127.0.0.1` | Host to bind to |
| `--config`, `-c` | `routesmith.yaml` | Config file path |
| `--log-level` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## `routesmith stats`

Show cost savings statistics. Reads from a running server or local SQLite storage.

| Option | Default | Description |
|--------|---------|-------------|
| `--server`, `-s` | `http://127.0.0.1:9119` | RouteSmith server URL |
| `--json` | — | Output as JSON instead of formatted table |
| `--local` | — | Read stats from local SQLite storage instead of server |
| `--db` | `routesmith_feedback.db` | SQLite database path (with `--local`) |
| `--watch`, `-w` | — | Live refresh stats every 2 seconds (`--local` only) |
| `--project`, `--proj` | — | Filter stats to a specific project name |

When no `--project` filter is set, a per-project breakdown is included.

---

## `routesmith audit`

View the structured routing decision audit log.

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | `routesmith_feedback.db` | SQLite database path |
| `--limit`, `-n` | `50` | Number of audit entries to show |
| `--project`, `--proj` | — | Filter by project name |
| `--model` | — | Filter by selected model |
| `--json` | — | Output as JSON |

Each entry shows: timestamp, project, model, strategy, cache hit, reason, cost, savings, latency, candidate models, agent role, conversation ID, and candidate scores.

---

## `routesmith roles`

Manage per-role routing policies. Configures the `role_policies` section in `routesmith.yaml`.

### Subcommands

| Subcommand | Description |
|------------|-------------|
| `list` (default) | List all configured role policies |
| `set` | Create or update a role policy |
| `unset` | Remove a role policy |

### Global Options

| Option | Description |
|--------|-------------|
| `--config` | Path to `routesmith.yaml` (default: `routesmith.yaml`) |
| `--json` | Output as JSON |

### `set` Options

| Option | Required | Description |
|--------|----------|-------------|
| `--role`, `-r` | Yes | Agent role name |
| `--model-pool` | No | List of model IDs allowed for this role |
| `--reward` | No | Reward function names for this role |

### Examples

```bash
routesmith roles list
routesmith roles set --role coder --model-pool gpt-4o-mini gpt-4o
routesmith roles set --role researcher --reward quality_first cost_sensitive
routesmith roles unset --role coder
```

---

## `routesmith evaluate`

Replay historical feedback records through a candidate config to compare routing quality.

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | `routesmith_feedback.db` | SQLite feedback database |
| `--config`, `-c` | `routesmith.yaml` | Config file to evaluate |
| `--output`, `-o` | — | Output file for evaluation report |
| `--json` | — | Output in JSON format |
| `--limit`, `-n` | `1000` | Max records to evaluate |

Output shows agreement rate, cost/quality deltas vs production, and per-request mismatches.

---

## `routesmith dashboard`

Launch the interactive Textual TUI dashboard.

| Option | Default | Description |
|--------|---------|-------------|
| `--db` | `routesmith_feedback.db` | SQLite database path |

Requires `textual`. Falls back to `routesmith stats --local` if not installed.

---

## `routesmith openclaw-config`

Generate an OpenClaw provider configuration that points at the RouteSmith proxy.

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `http://localhost:9119` | RouteSmith proxy URL |
| `--output`, `-o` | stdout | Write config to file instead of stdout |
