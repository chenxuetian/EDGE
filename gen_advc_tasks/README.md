# Advanced Task Generation

## Directories and Files

## Usage

### Run Script

- example
  ```bash
  python main.py --webpage-dir test_webpage --task-type convers_intent --page-part btm
  ```
- check usage

  ```bash
  python main.py -h
  ```

  > usage: main.py [-h] --task-type {func_infer,detail_desc,convers_intent} --webpage-dir WEBPAGE_DIR [--page-part {top,mid,btm}]
  >
  > optional arguments:
  > -h, --help show this help message and exit
  > --task-type {func_infer,detail_desc,convers_intent}
  > Specify which one of the three advanced tasks: Function inference, Detailed description, Conversation intention.
  > --webpage-dir WEBPAGE_DIR
  > The directory of the webpags to generate the advanced task data.
  > --page-part {top,mid,btm}
  > Specify which part of the webpages to use. The default is `top`.

### API Key Setting
