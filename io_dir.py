from imports import *

def get_dirs(root_dir: str) -> Tuple[str, str]:

    data_dir= os.path.join(root_dir, 'data')
    work_dir = os.path.join(root_dir, 'work')
    gfile.makedirs(data_dir)
    gfile.makedirs(work_dir)

    return data_dir, work_dir
