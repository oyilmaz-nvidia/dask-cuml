#
# Copyright (c) 2019, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from dask.distributed import Client
from cuml.common.handle import Handle
from dask import delayed
import dask.dataframe as dd
from dask.distributed import wait, default_client
from dask.distributed import get_worker
import numba.cuda
import cudf
import numpy as np
import pandas as pd
import random
import asyncio
import uuid

from tornado import gen
from dask.distributed import default_client
from toolz import first
import logging
import dask.dataframe as dd
import dask_cudf
import numpy as np
import cudf
import pandas as pd
from dask.distributed import wait
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml import LinearRegression as cuLinearRegression
from cuml.neighbors import NearestNeighbors as cuKNN
import math

def parse_host_port(address):
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port

import dask_cudf

@gen.coroutine
def extract_ddf_partitions(ddf):
    """
    Given a Dask cuDF, return a tuple with (worker, future) for each partition
    """
    client = default_client()
    
    delayed_ddf = ddf.to_delayed()
    parts = client.compute(delayed_ddf)
    yield wait(parts)
    
    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = yield client.who_has(parts)

    worker_map = []
    for key, workers in who_has.items():
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))

    gpu_data = [(worker, part) for worker, part in worker_map]

    yield wait(gpu_data)

    raise gen.Return(gpu_data)
    
@gen.coroutine
def extract_ddf_partitions_dict(ddf):
    """
    Given a Dask cuDF, return a tuple with (worker, future) for each partition
    """
    client = default_client()
    
    delayed_ddf = ddf.to_delayed()
    parts = client.compute(delayed_ddf)
    yield wait(parts)
    
    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = yield client.who_has(parts)

    worker_map = []
    for key, workers in who_has.items():
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))

    gpu_data = {worker:part for worker, part in worker_map}

    yield wait(gpu_data)

    raise gen.Return(gpu_data)

def get_meta(df):
    ret = df.iloc[:0]
    return ret

def to_dask_cudf(futures):
    # Convert a list of futures containing dfs back into a dask_cudf
    dfs = [d for d in futures if d.type != type(None)]
    meta = c.submit(get_meta, dfs[0]).result()
    return dd.from_delayed(dfs, meta=meta)

async def connection_func(ep, listener):
    print("connection received from " + str(ep))


class RandomForestClassifier:
    
    def __init__(self, n_estimators=10, max_depth=-1, handle=None,
                  max_features=1.0, n_bins=8,
                  split_algo=0, min_rows_per_node=2,
                  bootstrap=True, bootstrap_features=False,
                  type_model="classifier", verbose=False,
                  rows_sample=1.0, max_leaves=-1,
                  gdf_datatype=None):
        """
        Creates local rf instance on each worker
        """
        
        self.handle = handle
        self.split_algo = split_algo
        self.min_rows_per_node = min_rows_per_node
        self.bootstrap_features = bootstrap_features
        self.rows_sample = rows_sample
        self.max_leaves = max_leaves
        self.n_estimators = n_estimators
        self.n_estimators_per_worker = list()
        self.max_depth = max_depth
        self.max_features = max_features
        self.type_model = type_model
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.n_bins = n_bins
        
        c = default_client()
        workers = c.has_what().keys()
        
        n_workers = len(workers)
        if n_estimators < n_workers:
            n_estimators = n_workers
        
        n_est_per_worker = math.floor(n_estimators / n_workers)
        if n_est_per_worker < 1:
            n_est_per_worker = 1
            
        for i in range(n_workers):
            self.n_estimators_per_worker.append(n_est_per_worker)
            
        remaining_est = n_estimators - (n_est_per_worker * n_workers)
        n_est_per_worker = math.ceil(remaining_est / n_workers)
        
        for i in range(remaining_est):
            self.n_estimators_per_worker[i] = self.n_estimators_per_worker[i] + n_est_per_worker
                    
        ws = list(zip(workers, list(range(len(workers)))))
                
        self.rfs = {parse_host_port(worker):c.submit(RandomForestClassifier._func_build_rf, 
                             n, n_estimators, self.n_estimators_per_worker, 
                             max_depth, handle,
                             max_features, n_bins,
                             split_algo, min_rows_per_node,
                             bootstrap, bootstrap_features,
                             type_model, verbose,
                             rows_sample, max_leaves,
                             gdf_datatype, random.random(),
                             workers=[worker])
            for worker, n in ws}
        
        rfs_wait = list()
        for r in self.rfs.values():
            rfs_wait.append(r)
                
        wait(rfs_wait)
        
    
    
    @staticmethod
    def _func_build_rf(n, n_estimators, n_estimators_per_worker, max_depth, handle,
                             max_features, n_bins,
                             split_algo, min_rows_per_node,
                             bootstrap, bootstrap_features,
                             type_model, verbose,
                             rows_sample, max_leaves,
                             gdf_datatype, r):
        
        return cuRFC(n_estimators=n_estimators_per_worker[n], max_depth=max_depth, handle=handle,
                  max_features=max_features, n_bins=n_bins,
                  split_algo=split_algo, min_rows_per_node=min_rows_per_node,
                  bootstrap=bootstrap, bootstrap_features=bootstrap_features,
                  type_model=type_model, verbose=verbose,
                  rows_sample=rows_sample, max_leaves=max_leaves,
                  gdf_datatype=gdf_datatype)
                
    
    @staticmethod
    def _fit(model, X_df, y_df, r): 
        return model.fit(X_df, y_df)
    
    @staticmethod
    def _predict(model, X, r): 
        return model.predictAllDTs(X)
    
    def fit(self, X, y):
        c = default_client()

        X_futures = c.sync(extract_ddf_partitions_dict, X)
        y_futures = c.sync(extract_ddf_partitions_dict, y)
                                       
        f = list()
        for w, xc in X_futures.items():     
            f.append(c.submit(RandomForestClassifier._fit, self.rfs[w], xc, y_futures[w], random.random(),
                             workers=[w]))            
                               
        wait(f)
        
        return self
    
    def predict(self, X):                
        c = default_client()
        workers = c.has_what().keys()
        
        ws = list(zip(workers, list(range(len(workers)))))

        X_Scattered = c.scatter(X)   
                
        f = list()
        for w, n in ws:
            f.append(c.submit(RandomForestClassifier._predict, self.rfs[parse_host_port(w)], X_Scattered, random.random(),
                             workers=[w]))
                    
        wait(f)

        indexes = list()
        rslts = list()
        for d in range(len(f)):   
            rslts.append(f[d].result())
            indexes.append(0)
                    
        pred = list()
                
        for i in range(len(X)):
            classes = dict()
            max_class = -1
            max_val = 0
            
            for d in range(len(rslts)):               
                for j in range(self.n_estimators_per_worker[d]):
                    sub_ind = indexes[d] + j
                    cls = rslts[d][sub_ind]
                    if cls not in classes.keys():
                        classes[cls] = 1
                    else:
                        classes[cls] = classes[cls] + 1

                    if classes[cls] > max_val:
                        max_val = classes[cls]
                        max_class = cls

                indexes[d] = indexes[d] + self.n_estimators_per_worker[d]

            pred.append(max_class)
            
        
        return pred
                            
    def fit_predict(self, X):
        return self.fit(X).predict(X)
