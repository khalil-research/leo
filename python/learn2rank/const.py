ordering_lst = ['max_weight',
                'min_weight',
                'max_avg_value',
                'min_avg_value',
                'max_max_value',
                'min_max_value',
                'max_min_value',
                'min_min_value',
                'max_avg_value_by_weight',
                'max_max_value_by_weight']

datasets_dict = {
    '3_60': {
        'train': [{
            'id': '3_60',
            'X': 'datasets/X_train_3_60_r1_c2_A_ta.npy',
            'Y': 'datasets/Y_train_3_60_r1_c2_A_ta.npy'
        }],
        'val': [{
            'id': '3_60',
            'X': 'datasets/X_val_3_60_r1_c2_A_ta.npy',
            'Y': 'datasets/Y_val_3_60_r1_c2_A_ta.npy'
        }],
        'test': [{
            'id': '3_60',
            'X': 'datasets/X_test_3_60_r1_c2_A_ta.npy',
            'Y': 'datasets/Y_test_3_60_r1_c2_A_ta.npy'
        }]
    }
}
