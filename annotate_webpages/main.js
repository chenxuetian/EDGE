import fs from "fs";
import minimist from "minimist";
import { chromium } from "playwright";
import { createSaveDir } from "./utils.js";
import { markMultiple } from "./markFunc.js";

(async () => {
  const args = minimist(process.argv.slice(2), {
    string: ["urlsPath", "saveDir"],
    boolean: ["headless"],
    default: { headless: true, start: 0, total: -1, concurrency: 4 },
  });

  const browser = await chromium.launch({ headless: args.headless });
  global.context = await browser.newContext();
  global.scriptContent = fs.readFileSync("browserScript.js", "utf-8");

  const urls = fs.readFileSync(args.urlsPath, "utf8").trim().split("\n");
  createSaveDir(args.saveDir);
  await markMultiple(
    urls,
    args.saveDir,
    args.concurrency,
    args.start,
    args.total
  );

  await context.close();
  await browser.close();
})();
