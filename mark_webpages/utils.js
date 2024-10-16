import { Presets, SingleBar } from "cli-progress";
import fs, { promises as fsp } from "fs";
import path from "path";


function getURLsFromDir(directoryPath) {
  return fs
    .readdirSync(directoryPath)
    .filter((file) => path.extname(file) === ".html")
    .map((file) => path.join(directoryPath, file))
    .sort()
    .map((file) => `file://${path.resolve(file)}`);
}

function createProgressBar() {
  const progressBar = new SingleBar(
    {
      clearOnComplete: false,
      hideCursor: true,
      format:
        "{bar} | {percentage}% | ETA: {eta}s | {value}/{total} | Successful: {numSuccTasks} | Duration: {duration_formatted}",
    },
    Presets.shades_classic
  );
  return progressBar;
}

function createSaveDir(saveDir) {
  fs.mkdirSync(path.join(saveDir, "anno"), { recursive: true });
  fs.mkdirSync(path.join(saveDir, "marked"), { recursive: true });
  fs.mkdirSync(path.join(saveDir, "raw"), { recursive: true });
}

function writeJSON(path, obj) {
  fsp.writeFile(path, JSON.stringify(obj, null, 2), "utf8");
}

export {
  createProgressBar,
  createSaveDir,
  getURLsFromDir,
  writeJSON,
};
