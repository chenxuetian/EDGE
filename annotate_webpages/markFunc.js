import fs from "fs";
import { errors as playwrightErrors } from "playwright";
import { createProgressBar, writeJSON } from "./utils.js";

async function markMultiple(
  urls,
  saveDir,
  concurrency = 4,
  start = 0,
  total = -1
) {
  if (total === -1) {
    total = urls.length;
  }
  const queue = urls.slice(start, start + total);
  const activePromises = new Set();
  let numSuccTasks = 0;

  const bar = createProgressBar();
  bar.start(total, 0, { numSuccTasks: numSuccTasks });

  function updateProgressBar(isSucc) {
    numSuccTasks += Number(isSucc);
    bar.increment();
    bar.update(bar.value, { numSuccTasks: numSuccTasks });
  }

  async function enqueue() {
    while (queue.length > 0) {
      if (activePromises.size < concurrency) {
        const url = queue.shift();
        const saveName = String(start + total - queue.length - 1).padStart(
          7,
          "0"
        );
        const promise = markSingle(url, saveDir, saveName, true)
          .then(updateProgressBar)
          .finally(() => {
            activePromises.delete(promise);
          });
        activePromises.add(promise);
      } else {
        await Promise.race(activePromises);
      }
    }
  }
  await enqueue();
  await Promise.allSettled([...activePromises]);
  bar.stop();
  console.log(`${numSuccTasks} of ${total} succeed.\n`);
}

const viewports = [
  // <1（手机）
  { width: 600, height: 1320 },
  { width: 720, height: 1280 },
  { width: 720, height: 1600 },
  // 4:3 （平板）
  { width: 1024, height: 768 },
  { width: 1440, height: 1080 },
  { width: 1600, height: 1200 },
  // 3:2
  { width: 1200, height: 800 },
  { width: 1440, height: 960 },
  // 16:10
  { width: 1280, height: 800 },
  { width: 1440, height: 900 },
  { width: 1680, height: 1050 },
  // 16:9
  { width: 1280, height: 720 },
  { width: 1366, height: 768 },
  { width: 1536, height: 864 },
  { width: 1600, height: 900 },
  { width: 1920, height: 1080 },
];

async function markSingle(
  url,
  saveDir,
  saveName,
  needScroll = false,
  fixedViewport = false
) {
  if (
    fs.existsSync(`${saveDir}/anno/${saveName}.json`) ||
    fs.existsSync(`${saveDir}/anno/${saveName}_top.json`) ||
    fs.existsSync(`${saveDir}/anno/${saveName}_mid.json`) ||
    fs.existsSync(`${saveDir}/anno/${saveName}_btm.json`)
  ) {
    return true;
  }

  const viewport = fixedViewport
    ? { width: 1920, height: 1080 }
    : viewports[Math.floor(Math.random() * viewports.length)];
  const page = await global.context.newPage();
  await page.setViewportSize(viewport);
  const viewHeight = page.viewportSize().height;

  try {
    let response = await page.goto(url);
    // await page.waitForLoadState("networkidle");
    if (!response) {
      console.log(`\nNo response: ${url}`);
      return false;
    }
    if (response.status() !== 200) {
      console.log(`\nGet status ${response.status()}: ${url}`);
      return false;
    }

    await page.waitForTimeout(2000);
    await page.evaluate(global.scriptContent);

    if (!needScroll) {
      return await markSingleStep(page, saveName, saveDir);
    } else {
      const fullHeight = await page.evaluate(
        () => document.documentElement.scrollHeight
      );
      if (fullHeight <= viewHeight * 1.25) {
        // 网页比较短，一次mark
        return await markSingleStep(page, saveName, saveDir);
      } else if (fullHeight <= viewHeight * 2.5) {
        // 网页长度中等，两次mark
        const succTop = await markSingleStep(page, saveName + "_top", saveDir);

        await page.evaluate((height) => window.scrollTo(0, height), fullHeight);
        await page.waitForTimeout(1500);
        const succBtm = await markSingleStep(page, saveName + "_btm", saveDir);

        return succTop || succBtm;
      } else {
        // 网页长，三次mark
        const succTop = await markSingleStep(page, saveName + "_top", saveDir);

        await page.evaluate(
          (height) => window.scrollBy(0, height),
          (fullHeight - viewHeight) / 2
        );
        await page.waitForTimeout(1500);
        const succMid = await markSingleStep(page, saveName + "_mid", saveDir);

        await page.evaluate((height) => window.scrollTo(0, height), fullHeight);
        await page.waitForTimeout(1500);
        const succBtm = await markSingleStep(page, saveName + "_btm", saveDir);

        return succTop || succMid || succBtm;
      }
    }
  } catch (error) {
    if (error instanceof playwrightErrors.TimeoutError) {
      console.error(`\nTimeout processing ${url}: ${error.message}`);
    } else {
      console.error(`\nUnexpected error in markSingle(${saveName}): `, error);
    }
    return false;
  } finally {
    await page.close();
  }
}

async function markSingleStep(page, saveName, saveDir) {
  const viewport = page.viewportSize();
  const url = page.url();
  const title = await page.title();
  const metaDescriptionContent = await page.evaluate(() => {
    const metaDescription =
      document.querySelector('meta[name="description"]') ||
      document.querySelector('meta[name="Description"]');
    return metaDescription ? metaDescription.getAttribute("content") ?? "" : "";
  });
  const metaKeywordsContent = await page.evaluate(() => {
    const metaKeywords =
      document.querySelector('meta[name="keywords"]') ||
      document.querySelector('meta[name="Keywords"]');
    return metaKeywords ? metaKeywords.getAttribute("content") ?? "" : "";
  });

  const annotations = {
    url: url,
    title: title,
    description: metaDescriptionContent,
    keywords: metaKeywordsContent,
    image: `${saveName}.png`,
    viewport: [viewport.width, viewport.height],
  };

  try {
    const markedElements = await page.evaluate(() => _markPage(99, true));
    if (markedElements !== null) {
      await page.screenshot({ path: `${saveDir}/marked/${saveName}.png` });
      await page.evaluate(() => _markPage(99, false, false));
      await page.screenshot({ path: `${saveDir}/som/${saveName}.png` });
      annotations.elements = markedElements;
      await page.evaluate(() => _unmarkPage());
      await page.screenshot({ path: `${saveDir}/raw/${saveName}.png` });
      writeJSON(`${saveDir}/anno/${saveName}.json`, annotations);
      return true;
    }
    return false;
  } catch (error) {
    console.error(`\nUnexpected error in markSingleStep(${saveName}): `, error);
    return false;
  } finally {
    await page.evaluate(() => _unmarkPage());
  }
}

export { markMultiple };
