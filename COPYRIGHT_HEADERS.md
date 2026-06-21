# SmartWrapperOSS — Source File Copyright Header

Add the appropriate version below to the TOP of every source file in the
project (before any code, imports, or package declarations — except where
a language requires something else to be first, e.g. PHP's `<?php` tag or
a shebang line, in which case put the header immediately after that).

Update the year if/when you do an annual sweep; many projects just use the
year of first publication and never touch it again, which is also fine
legally — the notice doesn't need to be current to be valid.

---

## Recommended: short SPDX-style header (any language)

This is the modern, widely-accepted shorthand. Use this consistently
across the whole project rather than the full boilerplate below.

```
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 My_Name (SmartWrapperOSS)
```

(swap `//` for the appropriate comment syntax per language — see below)

---

## Comment syntax by language

### JavaScript / TypeScript / Java / C / C++ / C# / Go / Rust / Swift / Kotlin
```
// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 My_Name (SmartWrapperOSS)
```

### Python / Ruby / Shell / YAML / Dockerfile / Makefile
```
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 My_Name (SmartWrapperOSS)
```
Note: if the file starts with a shebang (e.g. `#!/usr/bin/env python3`),
keep the shebang as line 1 and put this header immediately after it.

### HTML
```
<!--
  SPDX-License-Identifier: Apache-2.0
  Copyright 2026 Aditi Jain (SmartWrapperOSS)
-->
```

### SQL
```
-- SPDX-License-Identifier: Apache-2.0
-- Copyright 2026 Aditi Jain (SmartWrapperOSS)
```

---

## Full boilerplate alternative (if you'd rather not use SPDX)

Only needed if you decide against the short form above. Pick ONE style —
SPDX or full text — and use it consistently across the whole project;
don't mix both.

### JavaScript / TypeScript / Java / C / C++ / C# / Go / Rust / Swift / Kotlin
```
/*
 * Copyright 2026 My_Name (SmartWrapperOSS)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```

### Python / Ruby / Shell / YAML / Dockerfile / Makefile
```
# Copyright 2026 Aditi Jain (SmartWrapperOSS)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### HTML
```
<!--
  Copyright 2026 Aditi Jain (SmartWrapperOSS)

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->
```

### SQL
```
-- Copyright 2026 Aditi Jain (SmartWrapperOSS)
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
```
