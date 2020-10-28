import dask
import pytest


@pytest.fixture
def no_client():
    return None


@pytest.fixture(scope="function")
def threaded_client():
    with dask.config.set(scheduler="threads"):
        yield


@pytest.fixture(scope="function")
def processes_client():
    with dask.config.set(scheduler="processes"):
        yield


@pytest.fixture(scope="module")
def distributed_client():
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(threads_per_worker=1, n_workers=2, processes=True)
    client = Client(cluster)
    yield
    client.close()
    del client
    cluster.close()
    del cluster
