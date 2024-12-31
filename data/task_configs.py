ROOT_DIR = '/media/hcp_hdd/rs_HCP_ku/HCP_sample/'

TASK_CONFIGS = {
    'WM': {
        'label_mapping': {
            '0bk_body': 0, 
            '0bk_faces': 1, 
            '0bk_places': 2, 
            '0bk_tools': 3, 
            '2bk_body': 4, 
            '2bk_faces': 5, 
            '2bk_places': 6, 
            '2bk_tools': 7
        },
        'num_classes': 8
    },
    'MOTOR': {
        'label_mapping': {
            'lf': 0,
            'lh': 1,
            'rf': 2,
            'rh': 3,
            't': 4
        },
        'num_classes': 5
    },
    'EMOTION': {
        'label_mapping': {
            'fear': 0,
            'neut': 1
        },
        'num_classes': 2
    },
    'GAMBLING': {
        'label_mapping': {
            'loss': 0,
            'win': 1
        },
        'num_classes': 2
    },
    'LANGUAGE': {
        'label_mapping': {
            'math': 0,
            'story': 1
        },
        'num_classes': 2
    },
    'RELATIONAL': {
        'label_mapping': {
            'match': 0,
            'relation': 1
        },
        'num_classes': 2
    },
    'SOCIAL': {
        'label_mapping': {
            'mental': 0,
            'rnd': 1
        },
        'num_classes': 2
    }
}
