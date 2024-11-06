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
        default='top',
        help='''Specify which part(s), separated by comma (","),  of the webpages to use. E.g, `--page-part top,mid,btm`. The default is `top`.
        The annotation files with names not suffixed are regarded as "_top" here.
        '''
    )
    parser.add_argument(
        '--cover-exist',
        action='store_true',
        help='Whether to cover the existing corresponding result files(s).'
    )
    parser.add_argument(
        '--max-id',
        type=int,
        default=1000000,
        help='Specify the max index of the annotated webpages to use.'
    )

    args = parser.parse_args()
    task_type, webpage_dir, page_part_0, max_id, cover_exist = args.task_type, args.webpage_dir, args.page_part, args.max_id, args.cover_exist

    page_part: set = set(map(lambda x: x.strip(), page_part_0.split(',')))
    page_part.discard('')
    if not page_part.issubset({'top', 'mid', 'btm'}):
        exit('Any of `--page-part` should be one of "top", "mid" and "btm" (without quotation marks).')
    
    model = 'claude-3-5-sonnet-20240620'

    if task_type == 'func_infer':
        gen_function_all(model, webpage_dir, page_part, max_id, cover_exist)
    elif task_type == 'detail_desc':
        gen_detail_all(model, webpage_dir, page_part, max_id, cover_exist)
    elif task_type == 'convers_intent':
        gen_intention_all(model, webpage_dir, page_part, max_id, cover_exist) 