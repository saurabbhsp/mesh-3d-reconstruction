import os
def create_semaphore(path):
    open(os.path.join(path, ".semaphore"), "a").close()
