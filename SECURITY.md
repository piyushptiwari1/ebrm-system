# Security policy

## Reporting a vulnerability

If you discover a security issue in `ebrm-system`, please do **not** open a public GitHub issue. Instead, report it privately via GitHub Security Advisories at:

https://github.com/piyushptiwari1/ebrm-system/security/advisories/new

We aim to acknowledge reports within 72 hours and to issue a fix or mitigation within 14 days for high-severity issues.

## Scope

In scope:
- Code execution bypasses in `ExecVerifier` sandboxing.
- Denial-of-service via crafted inputs to verifiers, classifier, or voter.
- Dependency CVEs affecting `ebrm-system` directly.

Out of scope:
- Issues in transitive dependencies that do not affect `ebrm-system`'s documented attack surface.
- Social-engineering or physical attacks.

## Safe handling

`ExecVerifier` runs candidate code in an isolated subprocess (`python -I -S`) with a timeout and output cap. Do **not** use it to execute arbitrary untrusted code without your own additional sandboxing (containers, seccomp, cgroups, firecracker).
