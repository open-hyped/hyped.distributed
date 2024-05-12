import datasets
import pytest
import ray
from hyped.data.pipe import DataPipe

from hyped.distributed.parallel import DistributedParallelDataPipe
from hyped.distributed.pipe import DistributedDataPipe

from .const_processor import ConstantDataProcessor, ConstantDataProcessorConfig
from .test_pipe import TestDistributedDataPipe as _TestDistributedDataPipe


class TestDistributedParallelDataPipe(_TestDistributedDataPipe):
    @pytest.fixture(
        params=[
            "parallel-A",
            "parallel-B",
            "parallel-C",
            "parallel-D",
            "parallel-E",
            "parallel-non-dist",
        ]
    )
    def sample_data_pipe(self, request):
        # create data processor configs
        c1 = ConstantDataProcessorConfig(name="A", value="1")
        c2 = ConstantDataProcessorConfig(name="B", value="2")
        c3 = ConstantDataProcessorConfig(name="C", value="3")
        # create data processors
        p1 = ConstantDataProcessor(c1)
        p2 = ConstantDataProcessor(c2)
        p3 = ConstantDataProcessor(c3)
        # create data pipe
        if request.param == "parallel-A":
            return DistributedDataPipe(
                [
                    p1,
                    DistributedParallelDataPipe(
                        [
                            DistributedDataPipe([p2]),
                            DistributedDataPipe([p3]),
                        ]
                    ),
                ]
            )
        if request.param == "parallel-B":
            return DistributedDataPipe(
                [
                    DistributedParallelDataPipe(
                        [
                            DistributedDataPipe([p1]),
                            DistributedDataPipe([p2]),
                            DistributedDataPipe([p3]),
                        ]
                    )
                ]
            )
        if request.param == "parallel-C":
            return DistributedDataPipe(
                [
                    DistributedParallelDataPipe(
                        [
                            DistributedDataPipe([p1]),
                            DistributedDataPipe([p2]),
                        ]
                    ),
                    p3,
                ]
            )
        if request.param == "parallel-D":
            return DistributedParallelDataPipe([p1, p2, p3])
        if request.param == "parallel-E":
            return DistributedParallelDataPipe(
                [p1, DistributedParallelDataPipe([p2, p3])]
            )
        if request.param == "parallel-non-dist":
            return DataPipe(
                [
                    p1,
                    DistributedParallelDataPipe(
                        [
                            DistributedDataPipe([p2]),
                            DistributedDataPipe([p3]),
                        ],
                        num_proc_per_pipe=1,
                    ),
                ]
            )

        raise ValueError(request.param)
