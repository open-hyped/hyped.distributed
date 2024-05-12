from functools import partial

import datasets
import pytest
import ray
from hyped.data.pipe import DataPipe

from hyped.distributed.pipe import DistributedDataPipe

from .const_processor import ConstantDataProcessor, ConstantDataProcessorConfig


class TestDistributedDataPipe(object):
    @pytest.fixture(scope="class", autouse=True)
    def initialize_ray(self):
        ray.init()
        yield
        ray.shutdown()

    @pytest.fixture(
        params=[
            "flat",
            "nest-dist-normal",
            "nest-normal-dist",
            "stacked",
            "stacked-1-1-1",
            "stacked-1-3-5",
            "nested",
            "nested-1-3",
            "nested-dist-and-non-dist",
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
        if request.param == "flat":
            return DistributedDataPipe([p1, p2, p3])
        if request.param == "nest-dist-normal":
            return DistributedDataPipe([DataPipe([p1, p2, p3])], num_proc=1)
        if request.param == "nest-normal-dist":
            return DataPipe([DistributedDataPipe([p1, p2, p3], num_proc=1)])
        if request.param == "stacked":
            return DistributedDataPipe(
                [
                    DistributedDataPipe([p1]),
                    DistributedDataPipe([p2]),
                    DistributedDataPipe([p3]),
                ],
            )
        if request.param == "stacked-1-1-1":
            return DistributedDataPipe(
                [
                    DistributedDataPipe([p1], num_proc=1),
                    DistributedDataPipe([p2], num_proc=1),
                    DistributedDataPipe([p3], num_proc=1),
                ],
            )
        if request.param == "stacked-1-3-5":
            return DistributedDataPipe(
                [
                    DistributedDataPipe([p1], num_proc=1),
                    DistributedDataPipe([p2], num_proc=3),
                    DistributedDataPipe([p3], num_proc=5),
                ],
            )
        if request.param == "nested":
            return DistributedDataPipe(
                [p1, DistributedDataPipe([p2, DistributedDataPipe([p3])])]
            )
        if request.param == "nested-1-3":
            return DistributedDataPipe(
                [
                    p1,
                    DistributedDataPipe(
                        [p2, DistributedDataPipe([p3], num_proc=1)], num_proc=3
                    ),
                ]
            )
        if request.param == "nested-dist-and-non-dist":
            return DistributedDataPipe(
                [
                    p1,
                    DataPipe(
                        [p2, DistributedDataPipe([p3])],
                    ),
                ]
            )

        raise ValueError(request.param)

    def test_preparation_logic(self, sample_data_pipe):
        if (
            isinstance(sample_data_pipe, DistributedDataPipe)
            and not sample_data_pipe.is_pool_ready
        ):
            sample_data_pipe._spawn_pool(num_actors=1)

        assert not sample_data_pipe.is_prepared

        # create different input features
        x = datasets.Features({"X": datasets.Value("int32")})
        y = datasets.Features({"Y": datasets.Value("int32")})

        # prepare pipeline with X
        sample_data_pipe.prepare(x)
        assert sample_data_pipe.is_prepared

        if len(sample_data_pipe) > 1:
            # get processor from pipe
            p2 = sample_data_pipe[1]
            # prepare any processor with Y
            # this should break the feature pipe
            p2.prepare(y)
            assert p2.is_prepared
            assert not sample_data_pipe.is_prepared

            # preparing the pipe again should fix the issue
            sample_data_pipe.prepare(x)
            assert sample_data_pipe.is_prepared

    def test_feature_management(self, sample_data_pipe):
        if (
            isinstance(sample_data_pipe, DistributedDataPipe)
            and not sample_data_pipe.is_pool_ready
        ):
            sample_data_pipe._spawn_pool(num_actors=1)
        # create input and expected output features
        x = datasets.Features({"X": datasets.Value("int32")})
        y = datasets.Features({k: datasets.Value("string") for k in "ABC"})

        # prepare pipe
        sample_data_pipe.prepare(x)
        # check features
        assert sample_data_pipe.is_prepared
        assert sample_data_pipe.in_features == x
        assert sample_data_pipe.new_features == y
        assert sample_data_pipe.out_features == datasets.Features(x | y)

    def test_batch_processing(self, sample_data_pipe):
        if (
            isinstance(sample_data_pipe, DistributedDataPipe)
            and not sample_data_pipe.is_pool_ready
        ):
            sample_data_pipe._spawn_pool(num_actors=1)
        # create input batch and corresponding features
        x = datasets.Features({"X": datasets.Value("int32")})
        batch = {"X": ["example %i" % i for i in range(10)]}
        # apply pipe
        sample_data_pipe.prepare(x)
        batch = sample_data_pipe.batch_process(
            batch, index=list(range(10)), rank=0
        )
        # check processor output
        assert all(k in batch for k in "XABC")
        assert all(x == ("example %i" % i) for i, x in enumerate(batch["X"]))
        assert all(a == "1" for a in batch["A"])
        assert all(a == "2" for a in batch["B"])
        assert all(a == "3" for a in batch["C"])

    def test_apply_to_dataset(self, sample_data_pipe):
        # create sample dataset
        ds = datasets.Dataset.from_dict(
            {"X": ["example %i" % i for i in range(100)]}
        )
        # apply
        ds = sample_data_pipe.apply(ds, batch_size=10)
        # check processor output
        assert all(k in ds.features for k in "XABC")
        assert all(x == ("example %i" % i) for i, x in enumerate(ds["X"]))
        assert all(a == "1" for a in ds["A"])
        assert all(a == "2" for a in ds["B"])
        assert all(a == "3" for a in ds["C"])
        # check features
        assert sample_data_pipe.out_features == ds.features

    def test_apply_to_dataset_dict(self, sample_data_pipe):
        # create sample dataset
        ds = datasets.Dataset.from_dict(
            {"X": ["example %i" % i for i in range(100)]}
        )
        ds = datasets.DatasetDict({"train": ds})
        # apply
        ds = sample_data_pipe.apply(ds, batch_size=10)
        ds = ds["train"]
        # check processor output
        assert all(k in ds.features for k in "XABC")
        assert all(x == ("example %i" % i) for i, x in enumerate(ds["X"]))
        assert all(a == "1" for a in ds["A"])
        assert all(a == "2" for a in ds["B"])
        assert all(a == "3" for a in ds["C"])
        # check features
        assert sample_data_pipe.out_features == ds.features

    @pytest.mark.parametrize("num_shards", [1, 5, 10])
    def test_apply_to_iterable_dataset(self, sample_data_pipe, num_shards):
        # create sample dataset
        ds = datasets.Dataset.from_dict(
            {"X": ["example %i" % i for i in range(100)]}
        ).to_iterable_dataset(num_shards=num_shards)
        # apply
        if isinstance(sample_data_pipe, DistributedDataPipe):
            ds = sample_data_pipe.apply(ds, batch_size=10, unordered=False)
        else:
            ds = sample_data_pipe.apply(ds, batch_size=10)
        ds = datasets.Dataset.from_generator(
            lambda: (yield from ds), features=ds.features
        )
        # check processor output
        assert all(k in ds.features for k in "XABC")
        assert all(x == ("example %i" % i) for i, x in enumerate(ds["X"]))
        assert all(a == "1" for a in ds["A"])
        assert all(a == "2" for a in ds["B"])
        assert all(a == "3" for a in ds["C"])
        # check features
        assert sample_data_pipe.out_features == ds.features

    @pytest.mark.parametrize("num_shards", [1, 5, 10])
    def test_apply_to_iterable_dataset_dict(
        self, sample_data_pipe, num_shards
    ):
        ds = datasets.Dataset.from_dict(
            {"X": ["example %i" % i for i in range(100)]}
        ).to_iterable_dataset(num_shards=num_shards)
        ds = datasets.IterableDatasetDict({"train": ds})
        # apply
        if isinstance(sample_data_pipe, DistributedDataPipe):
            ds = sample_data_pipe.apply(ds, batch_size=10, unordered=False)
        else:
            ds = sample_data_pipe.apply(ds, batch_size=10)
        ds = datasets.Dataset.from_generator(
            lambda: (yield from ds["train"]), features=ds["train"].features
        )
        # check processor output
        assert all(k in ds.features for k in "XABC")
        assert all(x == ("example %i" % i) for i, x in enumerate(ds["X"]))
        assert all(a == "1" for a in ds["A"])
        assert all(a == "2" for a in ds["B"])
        assert all(a == "3" for a in ds["C"])
        # check features
        assert sample_data_pipe.out_features == ds.features

    def test_error_on_missing_features(self, sample_data_pipe, tmp_path):
        # create sample data file in json format
        p = tmp_path / "data.json"
        p.write_text('{"A": 0}')
        # stream data from file
        # datasets cannot infer the features when streaming from a file
        ds = datasets.load_dataset(
            "json", data_files=p.as_posix(), split="train", streaming=True
        )
        assert ds.features is None

        # should lead to a runtime error as the pipe cannot be prepared
        with pytest.raises(RuntimeError):
            sample_data_pipe.apply(ds)

        # need to be prepared manually
        sample_data_pipe.prepare(
            datasets.Features({"A": datasets.Value("int32")})
        )
        sample_data_pipe.apply(ds)
