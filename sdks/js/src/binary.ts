import { execSync } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

export function findBinary(): string {
  // 1. EVALFORGE_BIN env var
  if (process.env.EVALFORGE_BIN) {
    return process.env.EVALFORGE_BIN;
  }

  // 2. Walk up from this file looking for target/debug/evalforge
  let current = path.dirname(__filename);
  for (let i = 0; i < 6; i++) {
    const candidate = path.join(current, 'target', 'debug', 'evalforge');
    if (fs.existsSync(candidate)) return candidate;
    const release = path.join(current, 'target', 'release', 'evalforge');
    if (fs.existsSync(release)) return release;
    current = path.dirname(current);
  }

  // 3. On PATH
  try {
    const which = execSync('which evalforge', { encoding: 'utf8' }).trim();
    if (which) return which;
  } catch {}

  throw new Error(
    'EvalForge binary not found.\n' +
    'Option 1: git clone https://github.com/heManKuMAR6/evalforge && cargo build --release\n' +
    'Option 2: export EVALFORGE_BIN=/path/to/evalforge/binary'
  );
}
