Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.

---

## How to sign off on SmartWrapperOSS

This project uses the Developer Certificate of Origin (DCO) instead of a
Contributor License Agreement (CLA). It's much lighter weight: rather than
signing a separate legal document, you certify the above by adding a
`Signed-off-by` line to every commit message.

Git can do this automatically with the `-s` flag:

```bash
git commit -s -m "Add Gemini-Pro adapter to model router"
```

This appends a line like the following to your commit message:

```
Signed-off-by: Jane Doe <jane@example.com>
```

Use your real name and a real, reachable email — anonymous or pseudonymous
sign-offs aren't accepted, since the certification needs to be attributable
to you.

If you forgot to sign off on a commit, you can amend it:

```bash
git commit --amend -s
```

Pull requests without a sign-off on every commit will be flagged by a bot
and won't be merged until fixed — this is enforced automatically so you
don't need to ask a maintainer.

### Why a DCO instead of a CLA?

SmartWrapperOSS's core will remain Apache 2.0 and open indefinitely — there
is no plan to relicense the open core to a more restrictive license, which
is the main reason most projects adopt a CLA. The DCO simply certifies that
contributors have the right to submit their code, with far less friction
than a signed legal agreement. See the README for more on the project's
licensing philosophy.
