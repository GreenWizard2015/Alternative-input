## Step 1: Parallelize Guide Compliance Check Per Folder

Find and group all non-test source files by folder, then launch apply-guide agents per folder.

**Configuration** (adjust for your language):
```bash
FILE_EXT="*.py"           # Change to: *.js, *.ts, *.go, *.rs, etc.
TEST_PATTERN="test_*"     # Change to: *.test.js, *.spec.ts, etc.
IGNORE_DIRS="tests"       # Change to: test, __tests__, spec, .test, etc.
```

**Command**:
```bash
find . -name "$FILE_EXT" -not -name "$TEST_PATTERN" -not -path "*/$IGNORE_DIRS/*" -type f -printf '%h\n' | \
  sort -u | xargs -I {} sh -c 'echo "{}=$(find {} -maxdepth 1 -name "$FILE_EXT" -not -name "$TEST_PATTERN" -type f | paste -sd,)"'
```

For each folder group, launch in parallel: `/apply-guide file: {folder}, guide: .claude/CODING.md`

## Step 2: Wait for All Agents to Complete

All folder-based apply-guide tasks must complete before proceeding.

## Step 3: Run Format and Fix

```bash
source ~/anaconda3/bin/activate myenv && make format
```

**NEVER read Makefile** - just execute the format command.