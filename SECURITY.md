# Security and sensitive material

Do not report or commit credentials, private server paths, AlphaFold model
parameters, sequence databases, or unpublished large datasets. Use private
channels for credential incidents. After the repository is published, use
GitHub private vulnerability reporting when available; never include a live
credential or unpublished input in a public issue.

This research software executes optional external command plugins. Commands
are passed directly to the operating system without a shell, but configuration
files are still executable research inputs and should be reviewed before use.
Plugins are not sandboxed. The SSH executor quotes command arguments before
constructing the remote shell command, but it likewise assumes trusted caller
input and a trusted remote host.

Do not place credentials in command arguments or emit them to adapter standard
output/error: rendered commands and process logs are recorded as workflow
provenance. Prefer a separately managed environment or credential store, and
review generated logs before sharing them.

Run-state and model manifests intentionally record provenance such as absolute
artifact paths, hostnames, and visible-device settings. Review or redact those
generated files before sharing a run outside its original environment. The
source release itself must never contain generated run manifests, populated AF3
inputs, raw structures, parameters, databases, or private paths.
