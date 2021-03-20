import asyncio

from helper import fetch_all

def async_calls(image_paths):

    # Bucket the list


    # Make async calls to tf-serving
    start_time = time.time()
    responses = asyncio.run(fetch_all(image_paths))
    print(f"It took {time.time() - start_time}s to process {len(image_paths)} images.")


    # Collect and convert responses


    # Return