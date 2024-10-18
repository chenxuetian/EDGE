import argparse
from gen_funcs import gen_intention_all, gen_detail_all, gen_function_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task-type',
        type=str,
        choices=['func_infer', 'detail_desc', 'convers_intent'],
        required=True,
        help='Specify which one of the three advanced tasks: Function inference, Detailed description, Conversation intention.'
    )
    parser.add_argument(
        '--webpage-dir',
        type=str,
        required=True,
        help='The directory of the webpages to generate the advanced task data.'
    )
    parser.add_argument(
        '--page-part',
        type=str,
        choices=['top', 'mid', 'btm'],
        # required=True,
        default='top',
        help='Specify which part of the webpages to use. The default is `top`.'
    )

    args = parser.parse_args()
    
    task_type, webpage_dir, page_part = args.task_type, args.webpage_dir, args.page_part
    model = 'claude-3-5-sonnet-20240620'

    if task_type == 'func_infer':
        gen_function_all(model, webpage_dir, page_part=page_part)
    elif task_type == 'detail_desc':
        gen_detail_all(model, webpage_dir, page_part=page_part)
    elif task_type == 'convers_intent':
        gen_intention_all(model, webpage_dir, page_part=page_part) 