## Step 1: Parallelize Test Guide Compliance Check Per Folder

Find and group all test files by folder, then launch apply-guide agents per folder.

**Configuration** (adjust for your language):
```bash
TEST_DIR="tests"          # Change to: test, __tests__, spec, .test, etc.
TEST_PATTERN="test_*"     # Change to: *.test.js, *.spec.ts, *_test.go, etc.
FILE_EXT="*.py"           # Change to: *.js, *.ts, *.go, *.rs, etc.
```

**Command**:
```bash
find "$TEST_DIR" -name "$TEST_PATTERN$FILE_EXT" -type f -printf '%h\n' | \
  sort -u | xargs -I {} sh -c 'echo "{}=$(find {} -maxdepth 1 -name "$TEST_PATTERN$FILE_EXT" -type f | paste -sd,)"'
```

For each folder group, launch in parallel: `/apply-guide file: {folder}, guide: .claude/TESTING.md`

## Step 2: Wait for All Agents to Complete

All folder-based apply-guide tasks must complete before proceeding.

## Step 3: Run Tests

```bash
make test
```

Verify:
- All tests pass
- No fixtures or conftest issues
- Coverage meets standards
