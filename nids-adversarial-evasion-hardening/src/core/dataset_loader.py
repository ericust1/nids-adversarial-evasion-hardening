import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


class DatasetManager:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.categorical_columns = [
            'protocol_type', 'service', 'flag'
        ]
        self.numerical_columns = [
            'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        self.all_columns = self.categorical_columns + self.numerical_columns + ['label']
        self.feature_count = 0

    def load_nsl_kdd(self, data_dir):
        train_file = os.path.join(data_dir, 'KDDTrain+.txt')
        test_file = os.path.join(data_dir, 'KDDTest+.txt')

        if not os.path.exists(train_file):
            raise FileNotFoundError(
                "NSL-KDD training file not found at {}. "
                "Use generate_synthetic_traffic() instead.".format(train_file)
            )

        col_names = self.all_columns
        train_df = pd.read_csv(train_file, header=None, names=col_names)
        test_df = pd.read_csv(test_file, header=None, names=col_names)

        attack_categories = {
            'normal': 'normal',
            'neptune': 'dos', 'back': 'dos', 'land': 'dos',
            'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos',
            'apache2': 'dos', 'udpstorm': 'dos', 'processtable': 'dos',
            'mailbomb': 'dos',
            'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe',
            'portsweep': 'probe', 'mscan': 'probe', 'saint': 'probe',
            'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l',
            'phf': 'r2l', 'multihop': 'r2l', 'warezmaster': 'r2l',
            'warezclient': 'r2l', 'spy': 'r2l', 'xlock': 'r2l',
            'xsnoop': 'r2l', 'snmpguess': 'r2l', 'snmpgetattack': 'r2l',
            'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
            'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r',
            'perl': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r',
            'ps': 'u2r',
        }

        for df in [train_df, test_df]:
            df['label'] = df['label'].apply(
                lambda x: x.split('.')[0].strip() if '.' in str(x) else str(x).strip()
            )
            df['attack_cat'] = df['label'].map(attack_categories).fillna('unknown')
            df['binary_label'] = (df['attack_cat'] != 'normal').astype(int)

        X_train, y_train = self._preprocess_dataframe(train_df)
        X_test, y_test = self._preprocess_dataframe(test_df)

        self.X_train = X_train
        self.y_train = y_train.values
        self.X_test = X_test
        self.y_test = y_test.values
        self.feature_count = X_train.shape[1]

        return X_train, y_train.values, X_test, y_test.values

    def _preprocess_dataframe(self, df):
        categorical_data = df[self.categorical_columns].copy()
        numerical_data = df[self.numerical_columns].copy()

        encoded_cats = pd.get_dummies(categorical_data, columns=self.categorical_columns)

        all_feature_names = list(encoded_cats.columns) + list(numerical_data.columns)
        self.feature_columns = all_feature_names

        X = pd.concat([encoded_cats, numerical_data], axis=1)

        X_scaled = self.scaler.fit_transform(X.values)
        X_df = pd.DataFrame(X_scaled, columns=X.columns)

        y = df['binary_label']

        return X_df, y

    def split_data(self, X, y, test_size=0.2, stratify=True):
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_count = X_train.shape[1]

        return X_train, X_test, y_train, y_test

    def get_feature_count(self):
        if self.X_train is not None:
            return self.X_train.shape[1]
        return self.feature_count

    def get_class_distribution(self):
        dist = {}
        if self.y_train is not None:
            unique, counts = np.unique(self.y_train, return_counts=True)
            for u, c in zip(unique, counts):
                dist['class_{}_train'.format(u)] = int(c)
        if self.y_test is not None:
            unique, counts = np.unique(self.y_test, return_counts=True)
            for u, c in zip(unique, counts):
                dist['class_{}_test'.format(u)] = int(c)
        return dist

    def generate_synthetic_traffic(self, n_samples=5000):
        n_normal = n_samples // 2
        n_attack = n_samples - n_normal

        n_features = 41

        normal_traffic = np.zeros((n_normal, n_features))
        attack_traffic = np.zeros((n_attack, n_features))

        normal_traffic[:, 0] = np.random.exponential(5, n_normal)
        normal_traffic[:, 0] = np.clip(normal_traffic[:, 0], 0, 100)
        normal_traffic[:, 1] = np.random.lognormal(4, 2, n_normal)
        normal_traffic[:, 1] = np.clip(normal_traffic[:, 1], 0, 50000)
        normal_traffic[:, 2] = np.random.lognormal(3, 2, n_normal)
        normal_traffic[:, 2] = np.clip(normal_traffic[:, 2], 0, 50000)
        normal_traffic[:, 3] = np.random.binomial(1, 0.01, n_normal)
        normal_traffic[:, 4] = np.random.binomial(1, 0.005, n_normal)
        normal_traffic[:, 5] = np.random.binomial(1, 0.005, n_normal)
        normal_traffic[:, 6] = np.random.binomial(1, 0.02, n_normal)
        normal_traffic[:, 7] = np.random.binomial(1, 0.01, n_normal)
        normal_traffic[:, 8] = np.random.binomial(1, 0.5, n_normal)
        normal_traffic[:, 9] = np.random.binomial(1, 0.01, n_normal)
        normal_traffic[:, 10] = np.random.binomial(1, 0.005, n_normal)
        normal_traffic[:, 11] = np.random.binomial(1, 0.01, n_normal)
        normal_traffic[:, 12] = np.random.binomial(1, 0.01, n_normal)
        normal_traffic[:, 13] = np.random.binomial(1, 0.005, n_normal)
        normal_traffic[:, 14] = np.random.binomial(1, 0.002, n_normal)
        normal_traffic[:, 15] = np.random.binomial(1, 0.005, n_normal)
        normal_traffic[:, 16] = np.zeros(n_normal)
        normal_traffic[:, 17] = np.random.binomial(1, 0.001, n_normal)
        normal_traffic[:, 18] = np.random.binomial(1, 0.01, n_normal)
        normal_traffic[:, 19] = np.random.normal(30, 20, n_normal)
        normal_traffic[:, 19] = np.clip(normal_traffic[:, 19], 0, 511)
        normal_traffic[:, 20] = np.random.normal(15, 10, n_normal)
        normal_traffic[:, 20] = np.clip(normal_traffic[:, 20], 0, 511)
        normal_traffic[:, 21] = np.random.uniform(0, 0.05, n_normal)
        normal_traffic[:, 22] = np.random.uniform(0, 0.05, n_normal)
        normal_traffic[:, 23] = np.random.uniform(0, 0.05, n_normal)
        normal_traffic[:, 24] = np.random.uniform(0, 0.05, n_normal)
        normal_traffic[:, 25] = np.random.uniform(0.3, 0.9, n_normal)
        normal_traffic[:, 26] = np.random.uniform(0, 0.3, n_normal)
        normal_traffic[:, 27] = np.random.uniform(0, 0.2, n_normal)
        normal_traffic[:, 28] = np.random.normal(50, 30, n_normal)
        normal_traffic[:, 28] = np.clip(normal_traffic[:, 28], 0, 255)
        normal_traffic[:, 29] = np.random.normal(20, 15, n_normal)
        normal_traffic[:, 29] = np.clip(normal_traffic[:, 29], 0, 255)
        normal_traffic[:, 30] = np.random.uniform(0.3, 0.8, n_normal)
        normal_traffic[:, 31] = np.random.uniform(0, 0.3, n_normal)
        normal_traffic[:, 32] = np.random.uniform(0, 0.2, n_normal)
        normal_traffic[:, 33] = np.random.uniform(0, 0.15, n_normal)
        normal_traffic[:, 34] = np.random.uniform(0, 0.05, n_normal)
        normal_traffic[:, 35] = np.random.uniform(0, 0.05, n_normal)
        normal_traffic[:, 36] = np.random.uniform(0, 0.05, n_normal)
        normal_traffic[:, 37] = np.random.uniform(0, 0.05, n_normal)
        normal_traffic[:, 38] = np.random.uniform(0, 0.05, n_normal)
        normal_traffic[:, 39] = np.random.uniform(0, 0.05, n_normal)
        normal_traffic[:, 40] = np.random.uniform(0.0, 0.02, n_normal)

        dos_mask = np.random.random(n_attack) < 0.6
        probe_mask = ~dos_mask & (np.random.random(n_attack) < 0.7)
        r2l_mask = ~dos_mask & ~probe_mask & (np.random.random(n_attack) < 0.7)
        u2r_mask = ~dos_mask & ~probe_mask & ~r2l_mask

        dos_traffic = attack_traffic.copy()
        dos_traffic[:, 0] = np.random.uniform(0, 5, n_attack)
        dos_traffic[:, 1] = np.random.lognormal(7, 1.5, n_attack)
        dos_traffic[:, 1] = np.clip(dos_traffic[:, 1], 0, 70000)
        dos_traffic[:, 2] = np.random.exponential(10, n_attack)
        dos_traffic[:, 2] = np.clip(dos_traffic[:, 2], 0, 100000)
        dos_traffic[:, 3] = 0
        dos_traffic[:, 8] = 0
        dos_traffic[:, 19] = np.random.uniform(200, 511, n_attack)
        dos_traffic[:, 20] = np.random.uniform(100, 511, n_attack)
        dos_traffic[:, 21] = np.random.uniform(0.5, 1.0, n_attack)
        dos_traffic[:, 22] = np.random.uniform(0.5, 1.0, n_attack)
        dos_traffic[:, 25] = np.random.uniform(0.0, 0.2, n_attack)
        dos_traffic[:, 26] = np.random.uniform(0.5, 1.0, n_attack)
        dos_traffic[:, 28] = np.random.uniform(200, 255, n_attack)
        dos_traffic[:, 29] = np.random.uniform(150, 255, n_attack)
        dos_traffic[:, 30] = np.random.uniform(0.0, 0.1, n_attack)
        dos_traffic[:, 31] = np.random.uniform(0.5, 1.0, n_attack)

        probe_traffic = attack_traffic.copy()
        probe_traffic[:, 0] = np.random.exponential(50, n_attack)
        probe_traffic[:, 0] = np.clip(probe_traffic[:, 0], 0, 500)
        probe_traffic[:, 1] = np.random.uniform(0, 50, n_attack)
        probe_traffic[:, 2] = np.random.uniform(0, 10, n_attack)
        probe_traffic[:, 7] = np.random.binomial(1, 0.1, n_attack)
        probe_traffic[:, 19] = np.random.uniform(100, 300, n_attack)
        probe_traffic[:, 20] = np.random.uniform(50, 200, n_attack)
        probe_traffic[:, 21] = np.random.uniform(0.3, 0.8, n_attack)
        probe_traffic[:, 25] = np.random.uniform(0.5, 0.9, n_attack)
        probe_traffic[:, 26] = np.random.uniform(0.1, 0.4, n_attack)

        r2l_traffic = attack_traffic.copy()
        r2l_traffic[:, 0] = np.random.exponential(200, n_attack)
        r2l_traffic[:, 0] = np.clip(r2l_traffic[:, 0], 0, 2000)
        r2l_traffic[:, 1] = np.random.lognormal(2, 2, n_attack)
        r2l_traffic[:, 2] = np.random.lognormal(4, 2, n_attack)
        r2l_traffic[:, 8] = np.random.binomial(1, 0.7, n_attack)
        r2l_traffic[:, 19] = np.random.uniform(5, 50, n_attack)
        r2l_traffic[:, 20] = np.random.uniform(2, 30, n_attack)
        r2l_traffic[:, 25] = np.random.uniform(0.3, 0.8, n_attack)
        r2l_traffic[:, 32] = np.random.uniform(0.3, 0.8, n_attack)

        u2r_traffic = attack_traffic.copy()
        u2r_traffic[:, 0] = np.random.exponential(500, n_attack)
        u2r_traffic[:, 0] = np.clip(u2r_traffic[:, 0], 0, 5000)
        u2r_traffic[:, 1] = np.random.lognormal(1, 1, n_attack)
        u2r_traffic[:, 2] = np.random.lognormal(2, 1, n_attack)
        u2r_traffic[:, 8] = np.random.binomial(1, 0.9, n_attack)
        u2r_traffic[:, 10] = np.random.binomial(1, 0.3, n_attack)
        u2r_traffic[:, 12] = np.random.binomial(1, 0.2, n_attack)
        u2r_traffic[:, 19] = np.random.uniform(1, 20, n_attack)
        u2r_traffic[:, 20] = np.random.uniform(1, 10, n_attack)

        attack_traffic = np.zeros((n_attack, n_features))
        for i in range(n_attack):
            if dos_mask[i]:
                attack_traffic[i] = dos_traffic[i]
            elif probe_mask[i]:
                attack_traffic[i] = probe_traffic[i]
            elif r2l_mask[i]:
                attack_traffic[i] = r2l_traffic[i]
            else:
                attack_traffic[i] = u2r_traffic[i]

        X = np.vstack([normal_traffic, attack_traffic])
        y = np.hstack([np.zeros(n_normal), np.ones(n_attack)])

        permutation = np.random.permutation(n_samples)
        X = X[permutation]
        y = y[permutation].astype(int)

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_count = n_features

        return X, y


def main():
    parser = argparse.ArgumentParser(description='NIDS Dataset Manager')
    parser.add_argument('--data-dir', type=str, default='data/',
                        help='Path to NSL-KDD dataset directory')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic traffic data instead')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Number of synthetic samples to generate')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set fraction')
    args = parser.parse_args()

    dm = DatasetManager()

    if args.synthetic:
        X, y = dm.generate_synthetic_traffic(n_samples=args.n_samples)
        print("Generated {} synthetic samples".format(len(y)))
    else:
        X_train, y_train, X_test, y_test = dm.load_nsl_kdd(args.data_dir)
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        print("Loaded NSL-KDD dataset: {} total samples".format(len(y)))

    X_train, X_test, y_train, y_test = dm.split_data(
        X, y, test_size=args.test_size
    )

    print("Train set: {} samples, {} features".format(
        X_train.shape[0], X_train.shape[1]))
    print("Test set: {} samples, {} features".format(
        X_test.shape[0], X_test.shape[1]))

    dist = dm.get_class_distribution()
    print("Class distribution:")
    for k, v in dist.items():
        print("  {}: {}".format(k, v))

    print("Feature count: {}".format(dm.get_feature_count()))


if __name__ == '__main__':
    main()
