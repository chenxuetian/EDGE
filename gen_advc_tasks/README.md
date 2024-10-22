# Advanced Task Generation

<img src="./assets/tasks_advc_framed.png" alt="tasks_advc_framed.png" style="zoom:35%;" />

## Usage

### Run Script

- example
  ```bash
  python main.py --task-type convers_intent --webpage-dir test_webpage --page-part top,mid --cover-exist
  ```
- check usage

  ```bash
  python main.py -h
  ```
  ```
  -h, --help            show this help message and exit
  --task-type {func_infer,detail_desc,convers_intent}
                        Specify which one of the three advanced tasks: Function inference, Detailed description, Conversation intention.
  --webpage-dir WEBPAGE_DIR
                        The directory of the webpages to generate the advanced task data.
  --page-part PAGE_PART
                        Specify which part(s), separated by comma (","), of the webpages to use. E.g, `--page-part top,mid,btm`. The default is `top`.
  --cover-exist         Whether to cover the existing corresponding result files(s).
  --max-id MAX_ID       Specify the max indices of the annotated webpages to use.
  ```
