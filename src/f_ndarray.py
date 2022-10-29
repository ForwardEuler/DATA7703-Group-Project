import numpy as np
from numpy import ndarray
import pandas as pd


def msave(name: str, df, dtype=np.float32, compress=True) -> None:
    """
    Save a numeric pandas Dataframe or numpy ndarray to a npz file.

    Parameters
    ----------
    name : str
        file name of npz file.
    df :
        pandas Dataframe or numpy ndarray, if a pandas Dataframe is provided
        its value must be numeric.
    dtype :
        dtype of saved data, default = float32
    compress :
        compress data in npz file. default = True
    """
    if type(df) == ndarray:
        if compress is True:
            np.savez_compressed(name, data=df)
        else:
            np.savez(name, data=df)
        return
    if type(df) == pd.DataFrame:
        cname = df.columns.values.tolist()
        cname = np.array(cname)
        data = df.to_numpy(dtype=dtype)
        if compress:
            np.savez_compressed(name, cname=cname, data=data)
        else:
            np.savez(name, cname=cname, data=data)
        return
    raise TypeError("In function msave: Only ndarrays and pd.DataFrames are allowed")


def mload(file_name: str):
    m = np.load(file_name)
    if len(m) >= 2:
        df = pd.DataFrame(m['data'], columns=m['cname'])
        return df
    else:
        return m['data']


if __name__ == '__main__':
    a = mload('train.npz')
    b = mload('test.npz')
