import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def multi_threads_gen(model, page_dir, *, target_func, sample_names, concurrency=None):
    if not concurrency:
        concurrency = min(len(sample_names), min(32, os.cpu_count()))
    print(f'Concurrency: {concurrency}')

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        with tqdm(total=len(sample_names), desc=f"Processing `{target_func.__name__}`") as pbar:
            futures = []
            for sample_name in sample_names:
                future = executor.submit(
                    target_func, model=model, page_dir=page_dir, sample_name=sample_name
                )
                future.add_done_callback(lambda _: pbar.update())
                futures.append(future)

            for future in futures:
                result = future.result()
                results.append(result)

    return results
