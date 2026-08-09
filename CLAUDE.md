# CLAUDE.md

Project rules for Claude Code. Keep this file lean — if a rule doesn't change
Claude's behavior, cut it.

## Commands
<!-- Fill these in. These are the things Claude can't guess and matter most. -->
- Run tests: `<your test command>`
- Run a single test: `<your single-test command>`
- Typecheck / build: `<your command>`
- Lint: `<your command>`

## Workflow
- State any assumption before acting on it. If a requirement is ambiguous, ask — don't guess.
- Make the smallest change that solves the problem. No speculative abstraction, no "while I'm here" edits.
- Touch only files relevant to the task. Don't refactor or reformat code outside the change's scope.
- Follow the patterns and styles already enforced in this codebase rather than introducing new ones.
- Do cleanup in small, separately-committed passes (deduplication and exception-handling fixes are separate commits).

## Verification
- After a series of edits, run the tests and typecheck/build, and show the actual output — don't just assert success.
- Fix the root cause of a failure. Never suppress or work around an error just to make a check pass.
- Prefer targeted tests while iterating; run the full suite before claiming done.

## Testing
- New logic ships with tests. Don't delete or weaken a test to make the suite green.
- Cover edge cases explicitly: empty/null inputs, boundaries, error paths, and the unauthorized/logged-out case.
- One test asserts one behavior; the test name describes the behavior being proven.
- Thorough tests are required for core business logic and API handlers; trivial glue code doesn't need the same bar.

## Avoiding duplication
- Before writing a function, check whether it (or a close variant) already exists. Extend the existing one instead of adding a copy.
- When consolidating duplicates, diff the versions first: the merged version must preserve the correct edge-case behavior from all of them, not just the cleanest-looking one. Update all callers and run the affected tests before claiming done.

## Exception handling
- Catch specific exception types you can actually handle. Never catch broadly (`except Exception` / bare `except`) to suppress an error or return a default just to keep execution going — let unexpected failures propagate so the real bug surfaces.
- Legitimate uses are fine: retrying a known-flaky operation, a real fallback for an expected failure, wrapping a low-level error in a clearer one, or a top-level boundary that logs and returns a proper response.
