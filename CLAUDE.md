@AGENTS.md

## Claude Code remote development tools

Use Claude Code native tools for local work:

- Read
- Edit
- Write
- Bash
- Glob
- Grep
- LS
- Monitor

Use remote companion tools only for remote endpoints:

- RemoteRead = Read + endpoint
- RemoteEdit = Edit + endpoint
- RemoteWrite = Write + endpoint
- RemoteBash = Bash + endpoint
- RemoteGlob = Glob + endpoint
- RemoteGrep = Grep + endpoint
- RemoteLS = LS + endpoint
- RemoteMonitor = Monitor + endpoint
- RemoteApplyPatch = apply_patch + endpoint

Endpoint fields are `host`, `port`, `user`, `root`, and `cwd`. Prefer
`host + port` for ordinary remote development. Use managed sessions only when
isolation, leases, or dedicated containers are required.
