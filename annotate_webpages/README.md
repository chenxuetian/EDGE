# Webpage Annotation

The Node.js scripts for automatically annotating webpages.

## Files

- `browserScript.js`: the script to insert into the browser to perform the annotation in a specific webpage
- `main.js`: the entry of this annotation project that calls various functions to annotate webpages
- `markFunc.js`: functions to open the webpage, insert the annotating script, and save the annotations for `main.js`
- `package-lock.json`
- `package.json`
- `ranking.txt`: URLs of top 20,000 websites from [Ahrefs Rank](https://app.ahrefs.com/ahrefs-top)
- `README.md`: this README file
- `utils.js`: some helper utilities

## Usage

Install the dependencies:

```
npm install
```

Then annotate the 20,000 websites stored on `ranking.txt` by:

```
node main.js --urlsPath ranking.txt --saveDir annotated/ranking --headless true --start 0 --total 20000 --concurrency 4
```

You can also specify your own file storing webpage URLs instead of the `ranking.txt` file we provide.

Command line arguments:

- `urlsPath`: the input file containing one url in a line
- `saveDir`: the directory that you want store the annotations
- `headless`: whether the browser run in headless mode
- `start` and `total`: the start index and the total count of the urls to annotate this time with respect to the input file
- `concurrency`: maximum number of webpages that can be annotated simultaneously (depending on your network workload)
