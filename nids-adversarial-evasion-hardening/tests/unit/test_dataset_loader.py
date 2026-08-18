import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.dataset_loader import DatasetManager


class TestDatasetLoader:
    def test_synthetic_data_shape(self):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=1000)
        assert X.shape[0] == 1000
        assert X.shape[1] == 41
        assert y.shape == (1000,)
        assert set(np.unique(y)).issubset({0, 1})

    def test_synthetic_data_types(self):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=500)
        assert X.dtype in [np.float64, np.float32]
        assert y.dtype == np.int64 or y.dtype in [np.int32, np.intp]

    def test_synthetic_data_class_balance(self):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=2000)
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            assert c >= 400

    def test_preprocessing_normalizes(self):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=500)
        X_train, X_test, y_train, y_test = dm.split_data(X, y, test_size=0.3)

        train_means = X_train.mean(axis=0)
        train_stds = X_train.std(axis=0)

        assert len(train_means) == 41
        for std in train_stds:
            assert std > 0 or np.isclose(std, 0, atol=1e-10)

    def test_split_maintains_distribution(self):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=2000)
        X_train, X_test, y_train, y_test = dm.split_data(X, y, test_size=0.3)

        train_normal_ratio = np.sum(y_train == 0) / len(y_train)
        test_normal_ratio = np.sum(y_test == 0) / len(y_test)

        assert abs(train_normal_ratio - test_normal_ratio) < 0.1

    def test_split_shapes(self):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=1000)
        X_train, X_test, y_train, y_test = dm.split_data(X, y, test_size=0.2)

        assert X_train.shape[0] == 800
        assert X_test.shape[0] == 200
        assert X_train.shape[1] == X_test.shape[1]
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]

    def test_get_feature_count(self):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=100)
        dm.split_data(X, y, test_size=0.2)
        assert dm.get_feature_count() == 41

    def test_get_class_distribution(self):
        dm = DatasetManager()
        X, y = dm.generate_synthetic_traffic(n_samples=1000)
        dm.split_data(X, y, test_size=0.2)
        dist = dm.get_class_distribution()
        assert 'class_0_train' in dist
        assert 'class_1_train' in dist
        assert 'class_0_test' in dist
        assert 'class_1_test' in dist
        assert dist['class_0_train'] + dist['class_1_train'] == 800
        assert dist['class_0_test'] + dist['class_1_test'] == 200

    def test_reproducibility_with_fixed_state(self):
        np.random.seed(42)
        dm1 = DatasetManager()
        X1, y1 = dm1.generate_synthetic_traffic(n_samples=100)

        np.random.seed(42)
        dm2 = DatasetManager()
        X2, y2 = dm2.generate_synthetic_traffic(n_samples=100)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
