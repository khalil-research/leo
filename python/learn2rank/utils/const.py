from enum import Enum


class KnapsackStaticOrderings(Enum):
    max_weight = 0
    min_weight = 1
    max_avg_value = 2
    min_avg_value = 3
    max_max_value = 4
    min_max_value = 5
    max_min_value = 6
    min_min_value = 7
    max_avg_value_by_weight = 8
    max_max_value_by_weight = 9


class BinproblemStaticOrderings(Enum):
    max_weight = 0
    min_weight = 1
    max_avg_value = 2
    min_avg_value = 3
    max_max_value = 4
    min_max_value = 5
    max_min_value = 6
    min_min_value = 7
    max_avg_value_by_weight = 8
    max_max_value_by_weight = 9


class KnapsackPropertyWeights(Enum):
    weight = 0
    avg_value = 1
    max_value = 2
    min_value = 3
    avg_value_by_weight = 4
    max_value_by_weight = 5
    min_value_by_weight = 6


numpy_dataset_paths = {
    '3_60_mat': {
        'train': {
            'X': 'resources/datasets/X_train_3_60_r1_c2_A_ta.npy',
            'Y': 'resources/datasets/Y_train_3_60_r1_c2_A_ta.npy'
        },
        'val': {
            'X': 'resources/datasets/X_val_3_60_r1_c2_A_ta.npy',
            'Y': 'resources/datasets/Y_val_3_60_r1_c2_A_ta.npy'
        },
        'test': {
            'X': 'resources/datasets/X_test_3_60_r1_c2_A_ta.npy',
            'Y': 'resources/datasets/Y_test_3_60_r1_c2_A_ta.npy'
        }
    },
    '3_60_dict': {
        'train': 'resources/datasets/dict_train_3_60_r1_c2_A_ta.pkl',
        'val': 'resources/datasets/dict_val_3_60_r1_c2_A_ta.pkl',
        'test': 'resources/datasets/dict_test_3_60_r1_c2_A_ta.pkl'
    }
}
