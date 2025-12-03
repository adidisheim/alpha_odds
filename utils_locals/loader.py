import tqdm
import os
from parameters import Constant
import pandas as pd
import os

import pandas as pd
import tqdm

from parameters import Constant


def load_and_merge_ss():
    start_dir = Constant.RES_DIR
    file_lists = [x for x in os.listdir(start_dir) if '_ss.parquet' in x]

    df = pd.DataFrame()
    for file_list in tqdm.tqdm(file_lists):
        temp = pd.read_parquet(start_dir+file_list).reset_index()
        if 'qty' in temp.columns:
            df = pd.concat([df, temp], ignore_index=False)
    return df