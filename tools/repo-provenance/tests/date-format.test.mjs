import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";

const provenanceUrl = new URL("../app/lib/provenance.ts", import.meta.url).href;
const crossingTimestamp = "2026-07-28T17:47:32-07:00";

function formatInProcessTimezone(timeZone) {
  const source = `
    import { formatEventDate } from ${JSON.stringify(provenanceUrl)};
    process.stdout.write(formatEventDate(${JSON.stringify(crossingTimestamp)}));
  `;
  const result = spawnSync(process.execPath, ["--input-type=module", "--eval", source], {
    encoding: "utf8",
    env: { ...process.env, TZ: timeZone },
  });
  assert.equal(result.status, 0, result.stderr);
  return result.stdout;
}

test("formats SSR-visible dates independently of the process timezone", () => {
  const server = formatInProcessTimezone("UTC");
  const browser = formatInProcessTimezone("America/Los_Angeles");

  assert.equal(server, browser);
  assert.equal(server, "Jul 29, 2026");
});
