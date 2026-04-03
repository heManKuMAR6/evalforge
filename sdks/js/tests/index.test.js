const { test } = require('node:test');
const assert = require('node:assert');
const path = require('path');

// We test the compiled JS
const evalforge = require('../dist/index.js');

test('demo() returns EvalResult with passed=true', () => {
  process.env.EVALFORGE_BIN = path.resolve(
    __dirname, '../../../target/debug/evalforge'
  );
  const result = evalforge.demo();
  assert.strictEqual(result.passed, true);
  assert.strictEqual(result.metrics.length > 0, true);
  assert.strictEqual(result.metrics[0].score, 0.91);
});

test('EvalResult has correct shape', () => {
  process.env.EVALFORGE_BIN = path.resolve(
    __dirname, '../../../target/debug/evalforge'
  );
  const result = evalforge.demo();
  assert.ok('traceId' in result);
  assert.ok('framework' in result);
  assert.ok('metrics' in result);
  assert.ok('passed' in result);
});
