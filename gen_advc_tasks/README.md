# Advanced Task Generation

<img src="gen_advc_tasks/assets/tasks_advc_framed.png" alt="tasks_advc_framed.png"/>

Here we provide flexible and customizable scripts for generation of *advanced task* data described in the paper and depicted in the picture above. 

## Usage

### Running Script

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
                        Specify which one of the three advanced tasks: Function Inference, Detailed Description, Conversation Intention.
  --webpage-dir WEBPAGE_DIR
                        The directory of the webpages to generate the advanced task data.
  --page-part PAGE_PART
                        Specify which part(s), separated by comma (","), of the webpages to use. E.g, `--page-part top,mid,btm`. The default is `top`.
  --cover-exist         Whether to cover the existing corresponding result files(s).
  --max-id MAX_ID       Specify the max indices of the annotated webpages to use.
  ```

> The number and braces used in reference to a GUI component in system answers in the task of Conversation Intention are to be removed. 

### API Key Setting

Set your API Key using one of the two approaches:
- Typically, create a `.env` file under `gen_advc_tasks`, in which to set the corresponding environment variable (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc).
- Alternatively, explicitly specify `api_key` in [call_api.py](gen_advc_tasks/utils/call_api.py).

## Prompts

See `prompt` directory, containing:
- `system_prompt.txt`: system message
- `share_prompt.txt`: used in all the three tasks
- `function_prompt.txt`: intended for Function Inference
- `detail_prompt.txt`: intended for Detail Description
- `intention_prompt.txt`: intended for Conversation Intention