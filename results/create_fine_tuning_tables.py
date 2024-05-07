import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if __name__ == '__main__':
    from_scratch_glue_data = {
        'Model': [
            'T5-Base Random Init',
            'T5-Base Baseline PT',
            'T5-Base Full PT',
            'Baseline T5',
            'T5', 'DEPTH',  # checkpoint 2k
            'T5', 'DEPTH',  # checkpoint 4k
            'T5', 'DEPTH',  # checkpoint 8k
            'T5', 'DEPTH',  # checkpoint 16k
            'T5', 'DEPTH',  # checkpoint 32k
            'T5', 'DEPTH',  # checkpoint 64k
            'T5', 'DEPTH',  # checkpoint 128k
            'T5', 'DEPTH',  # checkpoint 256k
            'T5', 'DEPTH',  # checkpoint 512k
            'T5', 'DEPTH',  # checkpoint 1M
        ],
        'Step': [
            '0',
            '512k',
            '1M',
            '1M',
            '2k', '2k',
            '4k', '4k',
            '8k', '8k',
            '16k', '16k',
            '32k', '32k',
            '64k', '64k',
            '128k', '128k',
            '256k', '256k',
            '512k', '512k',
            '1M', '1M',
        ],
        'CoLA Matthews corr.': [
            12.3,
            51.1,
            53.84,
            53.98,
            8.62, 8.67,     # 2k
            6.23, 7.75,     # 4k
            4.66, 6.93,     # 8k
            8.99, 10.94,    # 16k
            10.72, 7.73,    # 32k
            6.86, 27.78,    # 64k
            12.85, 38.01,   # 128k
            11.78, 45.57,   # 256k
            19.96, 47.14,   # 512k
            29.35, 45.91,   # 1M

        ],
        'SST-2 Accuracy': [
            80.62,
            95.2,
            92.68,
            94.73,
            80.08, 78.52,   # 2k
            80.66, 80.08,   # 4k
            82.23, 79.69,   # 8k
            80.96, 81.25,   # 16k
            81.64, 82.81,   # 32k
            82.42, 86.72,   # 64k
            83.2, 88.87,    # 128k
            85.94, 91.31,   # 256k
            86.52, 91.11,   # 512k
            88.77, 91.41,   # 1M
        ],
        'MNL-M Accuracy': [
            68.02,
            84.24,
            84.24,
            87.28,
            52.54, 57.07,   # 2k
            53.16, 59.31,   # 4k
            54.33, 58.92,   # 8k
            54.1, 62.79,    # 16k
            55.18, 71.23,   # 32k
            57.68, 73.84,   # 64k
            69.61, 77.5,    # 128k
            72.82, 79.45,   # 256k
            74.22, 80.42,   # 512k
            74.53, 81.0     # 1M
        ],
        'MNLI-MM Accuracy': [
            68.0,
            86.2,
            84.57,
            87.1,
            52.3, 58.14,    # 2k
            53.89, 59.62,   # 4k
            54.62, 59.73,   # 8k
            54.1, 61.46,    # 16k
            55.6, 72.7,     # 32k
            60.88, 76.05,   # 64k
            69.82, 78.06,   # 128k
            73.39, 80.07,   # 256k
            74.26, 81.43,   # 512k
            75.37, 81.96,   # 1M
        ]
    }
    from_scratch_glue_results = pd.DataFrame(from_scratch_glue_data)

    from_pretrained_glue_data = {
        'Model': [
            'T5-Base Random Init',
            'T5-Base Baseline PT',
            'T5-Base Full PT',
            'Baseline T5',
            'T5', 'DEPTH',  # 2k
            'T5', 'DEPTH',  # 4k
            'T5', 'DEPTH',  # 8k
            'T5', 'DEPTH',  # 16k
            'T5', 'DEPTH',  # 32k
            'T5', 'DEPTH',  # 64k
            'T5', 'DEPTH',  # 128k
            'T5', 'DEPTH',  # 256k
        ],
        'Step': [
            '0',
            '512k',
            '1M',
            '1M',
            '2k', '2k',
            '4k', '4k',
            '8k', '8k',
            '16k', '16k',
            '32k', '32k',
            '64k', '64k',
            '128k', '128k',
            '256k', '256k',
        ],
        'CoLA Matthews corr.': [
            12.3,
            51.1,
            53.84,
            53.98,
            57.41, 53.93,   # 2k
            55.18, 47.11,   # 4k
            55.35, 50.67,   # 8k
            54.75, 52.65,   # 16k
            54.95, 53.79,   # 32k
            54.77, 52.95,   # 64k
            58.62, 56.21,   # 128k
            57.62, 56.78,   # 256k
        ],
        'SST-2 Accuracy': [
            80.62,
            95.2,
            92.68,
            94.73,
            95.02, 94.34,   # 2k
            95.02, 94.34,   # 4k
            95.21, 94.43,   # 8k
            95.7, 94.34,    # 16k
            95.21, 94.63,   # 32k
            95.21, 94.14,   # 64k
            95.21, 95.61,   # 128k
            95.21, 95.02,   # 256k
        ],
        'MNL-M Accuracy': [
            68.02,
            84.24,
            84.24,
            87.28,          # Baseline
            86.93, 86.38,   # 2k
            87.4, 87.14,    # 4k
            87.47, 86.62,   # 8k
            86.91, 86.62,   # 16k
            86.64, 86.95,   # 32k
            86.96, 86.65,   # 64k
            86.79, 87.42,   # 128k
            87.27, 86.86,   # 256k
        ],
        'MNLI-MM Accuracy': [
            68.0,
            86.2,
            84.57,
            87.1,
            87.06, 86.2,   # 2k
            87.34, 87.01,   # 4k
            87.31, 86.52,   # 8k
            86.67, 86.67,   # 16k
            86.06, 87.02,   # 32k
            86.93, 86.91,   # 64k
            86.67, 87.64,   # 128k
            87.22, 86.45,   # 256k
        ]
    }
    from_pretrained_glue_results = pd.DataFrame(from_pretrained_glue_data)

    from_scratch_discoeval_data = {
        'Model': [
            'Baseline T5',
            'T5', 'DEPTH',  # 2k
            'T5', 'DEPTH',  # 4k
            'T5', 'DEPTH',  # 8k
            'T5', 'DEPTH',  # 16k
            'T5', 'DEPTH',  # 32k
            'T5', 'DEPTH',  # 64k
            'T5', 'DEPTH',  # 128k
            'T5', 'DEPTH',  # 256k
            'T5', 'DEPTH',  # 512k
            'T5', 'DEPTH',  # 1M
        ],
        'Step': [
            '1M',
            '2k', '2k',         # 2k
            '4k', '4k',         # 4k
            '8k', '8k',         # 8k
            '16k', '16k',       # 16k
            '32k', '32k',       # 32k
            '64k', '64k',       # 64k
            '128k', '128k',     # 128k
            '256k', '256k',     # 256k
            '512k', '512k',     # 512k
            '1M', '1M',         # 1M
        ],
        'SP Arxiv Accuracy': [
            52.76,          # Baseline
            21.0, 34.81,    # 2k
            20.4, 35.72,    # 4k
            20.6, 36.43,    # 8k
            21.4, 36.47,    # 16k
            21.75, 38.04,   # 32k
            21.92, 42.24,   # 64k
            21.09, 45.0,    # 128k
            22.85, 48.68,   # 256k
            26.61, 52.59,   # 512k
            28.54, 52.39,   # 1M
        ],
        'SP Wiki Accuracy': [
            51.07,          # Baseline
            20.9, 40.19,    # 2k
            21.4, 40.38,    # 4k
            20.9, 40.14,    # 8k
            22.0, 40.33,    # 16k
            21.75, 44.26,   # 32k
            21.53, 45.85,   # 64k
            33.96, 47.71,   # 128k
            37.92, 48.93,   # 256k
            41.97, 51.66,   # 512k
            43.22, 50.07,   # 1M
        ],
        'SP Rocstory Accuracy': [
            70.58,          # Baseline
            21.2, 51.51,    # 2k
            20.9, 52.03,    # 4k
            21.0, 51.25,    # 8k
            21.24, 53.13,   # 16k
            20.63, 54.05,   # 32k
            21.24, 55.96,   # 64k
            42.63, 59.57,   # 128k
            44.85, 61.33,   # 256k
            52.29, 65.92,   # 512k
            50.98, 63.89,   # 1M
        ],
        'DC Chat Accuracy': [
            68.99,          # Baseline
            55.18, 55.27,   # 2k
            57.18, 56.49,   # 4k
            57.42, 57.03,   # 8k
            57.62, 57.52,   # 16k
            57.62, 58.5,    # 32k
            57.23, 60.94,   # 64k
            60.16, 64.65,   # 128k
            58.89, 65.33,   # 256k
            61.33, 66.85,   # 512k
            61.13, 67.92,   # 1M
        ],
        'DC Wiki Accuracy': [
            92.09,          # Baseline
            53.59, 53.2,    # 2k
            55.37, 54.74,   # 4k
            54.57, 54.69,   # 8k
            55.18, 55.44,   # 16k
            56.1, 57.37,    # 32k
            57.01, 72.58,   # 64k
            60.01, 78.0,    # 128k
            61.91, 81.69,   # 256k
            60.89, 83.81,   # 512k
            65.48, 84.52,   # 1M
        ]
    }
    from_scratch_discoeval_results = pd.DataFrame(from_scratch_discoeval_data)

    from_pretrained_discoeval_data = {
        'Model': [
            'Baseline T5',
            'T5', 'DEPTH',  # 2k
            'T5', 'DEPTH',  # 4k
            'T5', 'DEPTH',  # 8k
            'T5', 'DEPTH',  # 16k
            'T5', 'DEPTH',  # 32k
            'T5', 'DEPTH',  # 64k
            'T5', 'DEPTH',  # 128k
            'T5', 'DEPTH',  # 256k
        ],
        'Step': [
            '1M',
            '2k', '2k',
            '4k', '4k',
            '8k', '8k',
            '16k', '16k',
            '32k', '32k',
            '64k', '64k',
            '128k', '128k',
            '256k', '256k',
        ],
        'SP Arxiv Accuracy': [
            52.76,
            62.26, 44.14,   # 2k
            61.33, 56.62,   # 4k
            63.63, 55.03,   # 8k
            60.94, 58.86,   # 16k
            60.69, 59.35,   # 32k
            58.96, 58.57,   # 64k
            60.13, 70.56,   # 128k
            59.03, 67.07,   # 256k
        ],
        'SP Wiki Accuracy': [
            51.07,
            51.9, 49.34,   # 2k
            52.39, 51.2,   # 4k
            52.03, 51.56,   # 8k
            51.78, 52.08,   # 16k
            52.22, 53.27,   # 32k
            52.25, 52.95,   # 64k
            51.03, 54.47,   # 128k
            52.66, 53.0,    # 256k
        ],
        'SP Rocstory Accuracy': [
            70.58,
            74.83, 74.78,   # 2k
            74.98, 67.65,   # 4k
            77.0, 71.02,    # 8k
            76.49, 76.15,   # 16k
            75.61, 78.08,   # 32k
            76.07, 76.1,    # 64k
            75.76, 82.42,   # 128k
            66.77, 76.71,   # 256k
        ],
        'DC Chat Accuracy': [
            68.99,
            71.44, 70.46,   # 2k
            74.27, 72.02,   # 4k
            73.1, 71.97,   # 8k
            74.71, 72.71,   # 16k
            74.02, 72.85,   # 32k
            73.73, 73.24,   # 64k
            73.44, 73.96,   # 128k
            72.22, 72.96,   # 256k
        ],
        'DC Wiki Accuracy': [
            92.09,
            92.09, 89.99,   # 2k
            91.72, 91.46,   # 4k
            92.33, 91.99,   # 8k
            91.53, 92.63,   # 16k
            92.33, 92.48,   # 32k
            92.58, 92.58,   # 64k
            91.14, 92.53,   # 128k
            92.31, 92.07,   # 256k
        ]
    }
    from_pretrained_discoeval_results = pd.DataFrame(from_pretrained_discoeval_data)

    # Shades of red
    # T5_COLORS = [
    #     # mcolors.CSS4_COLORS.get('orange'),
    #     mcolors.CSS4_COLORS.get('darkorange'),
    #     # mcolors.CSS4_COLORS.get('orangered'),
    #     # mcolors.CSS4_COLORS.get('sandybrown'),
    # ]
    #
    # # Shades of blue
    # DEPTH_COLORS = [
    #     mcolors.CSS4_COLORS.get('blue'),
    #     # mcolors.CSS4_COLORS.get('mediumpurple'),
    #     # mcolors.CSS4_COLORS.get('darkblue'),
    #     # mcolors.CSS4_COLORS.get('royalblue'),
    # ]

    DEPTH_COLOR = mcolors.CSS4_COLORS.get('blue')
    T5_COLOR = mcolors.CSS4_COLORS.get('darkorange')

    MARKERS = ['o', 's', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', '1', '2', '3', '4', '8']
    LINESTYLES = ['-', '--', '-.', ':']

    # SP (From Scratch)
    plt.figure(figsize=(8, 6))
    sp_metrics = ['SP Arxiv Accuracy', 'SP Wiki Accuracy', 'SP Rocstory Accuracy']
    for metric_index, metric in enumerate(sp_metrics):
        plt.plot(
            from_scratch_discoeval_results[from_scratch_discoeval_results['Model'] == 'T5']['Step'],
            from_scratch_discoeval_results[from_scratch_discoeval_results['Model'] == 'T5'][metric],
            marker=MARKERS[metric_index],
            label=f'T5 {metric}',
            color=T5_COLOR,
            linestyle=LINESTYLES[metric_index],
        )
    for metric_index, metric in enumerate(sp_metrics):
        plt.plot(
            from_scratch_discoeval_results[from_scratch_discoeval_results['Model'] == 'DEPTH']['Step'],
            from_scratch_discoeval_results[from_scratch_discoeval_results['Model'] == 'DEPTH'][metric],
            marker=MARKERS[metric_index],
            label=f'DEPTH {metric}',
            color=DEPTH_COLOR,
            linestyle=LINESTYLES[metric_index],
        )
    # plt.title('SP (From Scratch)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Step', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.legend(fontsize=14)
    # plt.tight_layout()
    plt.savefig('figures/sp_from_scratch.png')
    plt.show()

    # SP (From Pretrained)
    plt.figure(figsize=(8, 6))
    sp_metrics = ['SP Arxiv Accuracy', 'SP Wiki Accuracy', 'SP Rocstory Accuracy']
    for metric_index, metric in enumerate(sp_metrics):
        plt.plot(
            from_pretrained_discoeval_results[from_pretrained_discoeval_results['Model'] == 'T5']['Step'],
            from_pretrained_discoeval_results[from_pretrained_discoeval_results['Model'] == 'T5'][metric],
            marker=MARKERS[metric_index],
            label=f'T5 {metric}',
            color=T5_COLOR,
            linestyle=LINESTYLES[metric_index],
        )
    for metric_index, metric in enumerate(sp_metrics):
        plt.plot(
            from_pretrained_discoeval_results[from_pretrained_discoeval_results['Model'] == 'DEPTH']['Step'],
            from_pretrained_discoeval_results[from_pretrained_discoeval_results['Model'] == 'DEPTH'][metric],
            marker=MARKERS[metric_index],
            label=f'DEPTH {metric}',
            color=DEPTH_COLOR,
            linestyle=LINESTYLES[metric_index],
        )
    # plt.title('SP (From Pretrained)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=14)
    # plt.tight_layout()
    plt.savefig('figures/sp_from_pretrained.png')
    plt.show()

    # DC (From Scratch)
    plt.figure(figsize=(8, 6))
    dc_metrics = ['DC Chat Accuracy', 'DC Wiki Accuracy']
    for metric_index, metric in enumerate(dc_metrics):
        plt.plot(
            from_scratch_discoeval_results[from_scratch_discoeval_results['Model'] == 'T5']['Step'],
            from_scratch_discoeval_results[from_scratch_discoeval_results['Model'] == 'T5'][metric],
            marker=MARKERS[metric_index],
            label=f'T5 {metric}',
            color=T5_COLOR,
            linestyle=LINESTYLES[metric_index],

        )
    for metric_index, metric in enumerate(dc_metrics):
        plt.plot(
            from_scratch_discoeval_results[from_scratch_discoeval_results['Model'] == 'DEPTH']['Step'],
            from_scratch_discoeval_results[from_scratch_discoeval_results['Model'] == 'DEPTH'][metric],
            marker=MARKERS[metric_index],
            label=f'DEPTH {metric}',
            color=DEPTH_COLOR,
            linestyle=LINESTYLES[metric_index],
        )
    # plt.title('DC (From Scratch)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=14)
    # plt.tight_layout()
    plt.savefig('figures/dc_from_scratch.png')
    plt.show()

    # DC (From Pretrained)
    plt.figure(figsize=(8, 6))
    dc_metrics = ['DC Chat Accuracy', 'DC Wiki Accuracy']
    for metric_index, metric in enumerate(dc_metrics):
        plt.plot(
            from_pretrained_discoeval_results[from_pretrained_discoeval_results['Model'] == 'T5']['Step'],
            from_pretrained_discoeval_results[from_pretrained_discoeval_results['Model'] == 'T5'][metric],
            marker=MARKERS[metric_index],
            label=f'T5 {metric}',
            color=T5_COLOR,
            linestyle=LINESTYLES[metric_index],
        )
    for metric_index, metric in enumerate(dc_metrics):
        plt.plot(
            from_pretrained_discoeval_results[from_pretrained_discoeval_results['Model'] == 'DEPTH']['Step'],
            from_pretrained_discoeval_results[from_pretrained_discoeval_results['Model'] == 'DEPTH'][metric],
            marker='o',
            label=f'DEPTH {metric}',
            color=DEPTH_COLOR,
            linestyle=LINESTYLES[metric_index],
        )
    # plt.title('DC (From Pretrained)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=14)
    # plt.tight_layout()
    plt.savefig('figures/dc_from_pretrained.png')
    plt.show()

    # MNLI (From Scratch)
    plt.figure(figsize=(8, 6))
    plt.plot(
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'T5']['Step'],
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'T5']['MNL-M Accuracy'],
        marker=MARKERS[0],
        label='T5 MNL-M',
        color=T5_COLOR,
        linestyle=LINESTYLES[0],
    )
    plt.plot(
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'T5']['Step'],
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'T5']['MNLI-MM Accuracy'],
        marker=MARKERS[1],
        label='T5 MNLI-MM',
        color=T5_COLOR,
        linestyle=LINESTYLES[1],
    )
    plt.plot(
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'DEPTH']['Step'],
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'DEPTH']['MNL-M Accuracy'],
        marker=MARKERS[0],
        label='DEPTH MNL-M',
        color=DEPTH_COLOR,
        linestyle=LINESTYLES[0],
    )
    plt.plot(
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'DEPTH']['Step'],
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'DEPTH']['MNLI-MM Accuracy'],
        marker=MARKERS[1],
        label='DEPTH MNLI-MM',
        color=DEPTH_COLOR,
        linestyle=LINESTYLES[1],
    )
    # plt.title('MNLI (From Scratch)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=14)
    # plt.tight_layout()
    plt.savefig('figures/mnli_from_scratch.png')
    plt.show()

    # MNLI (From Pretrained)
    plt.figure(figsize=(8, 6))
    plt.plot(
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'T5']['Step'],
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'T5']['MNL-M Accuracy'],
        marker=MARKERS[0],
        label='T5 MNL-M',
        color=T5_COLOR,
        linestyle=LINESTYLES[0],
    )
    plt.plot(
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'T5']['Step'],
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'T5']['MNLI-MM Accuracy'],
        marker=MARKERS[1],
        label='T5 MNLI-MM',
        color=T5_COLOR,
        linestyle=LINESTYLES[1],
    )
    plt.plot(
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'DEPTH']['Step'],
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'DEPTH']['MNL-M Accuracy'],
        marker=MARKERS[0],
        label='DEPTH MNL-M',
        color=DEPTH_COLOR,
        linestyle=LINESTYLES[0],
    )
    plt.plot(
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'DEPTH']['Step'],
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'DEPTH']['MNLI-MM Accuracy'],
        marker=MARKERS[1],
        label='DEPTH MNLI-MM',
        color=DEPTH_COLOR,
        linestyle=LINESTYLES[1],
    )
    # plt.title('MNLI (From Pretrained)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=14)
    # plt.tight_layout()
    plt.savefig('figures/mnli_from_pretrained.png')
    plt.show()

    # CoLA Matthews corr. (From Scratch)
    plt.figure(figsize=(8, 6))
    metric = 'CoLA Matthews corr.'
    plt.plot(
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'T5']['Step'],
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'T5'][metric],
        marker=MARKERS[0],
        label='T5',
        color=T5_COLOR,
        linestyle=LINESTYLES[0],
    )
    plt.plot(
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'DEPTH']['Step'],
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'DEPTH'][metric],
        marker=MARKERS[0],
        label='DEPTH',
        color=DEPTH_COLOR,
        linestyle=LINESTYLES[0],
    )
    # plt.title('CoLA (From Scratch)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Matthews corr.', fontsize=14)
    plt.legend(fontsize=14)
    # plt.tight_layout()
    plt.savefig('figures/cola_from_scratch.png')
    plt.show()

    # CoLA Matthews corr. (From Pretrained)
    plt.figure(figsize=(8, 6))
    metric = 'CoLA Matthews corr.'
    plt.plot(
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'T5']['Step'],
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'T5'][metric],
        marker=MARKERS[0],
        label='T5',
        color=T5_COLOR,
        linestyle=LINESTYLES[0],
    )
    plt.plot(
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'DEPTH']['Step'],
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'DEPTH'][metric],
        marker=MARKERS[0],
        label='DEPTH',
        color=DEPTH_COLOR,
        linestyle=LINESTYLES[0],
    )
    # plt.title('CoLA (From Pretrained)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Matthews corr.', fontsize=14)
    plt.legend(fontsize=14)
    # plt.tight_layout()
    plt.savefig('figures/cola_from_pretrained.png')
    plt.show()

    # SST-2 Accuracy (From Scratch)
    plt.figure(figsize=(8, 6))
    metric = 'SST-2 Accuracy'
    plt.plot(
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'T5']['Step'],
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'T5'][metric],
        marker=MARKERS[0],
        label='T5',
        color=T5_COLOR,
        linestyle=LINESTYLES[0],
    )
    plt.plot(
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'DEPTH']['Step'],
        from_scratch_glue_results[from_scratch_glue_results['Model'] == 'DEPTH'][metric],
        marker=MARKERS[0],
        label='DEPTH',
        color=DEPTH_COLOR,
        linestyle=LINESTYLES[0],
    )
    # plt.title('SST-2 (From Scratch)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('SST-2 Accuracy', fontsize=14)
    plt.legend(fontsize=14)
    # plt.tight_layout()
    plt.savefig('figures/sst2_from_scratch.png')
    plt.show()

    # SST-2 Accuracy (From Pretrained)
    plt.figure(figsize=(8, 6))
    metric = 'SST-2 Accuracy'
    plt.plot(
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'T5']['Step'],
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'T5'][metric],
        marker=MARKERS[0],
        label='T5',
        color=T5_COLOR,
        linestyle=LINESTYLES[0],
    )
    plt.plot(
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'DEPTH']['Step'],
        from_pretrained_glue_results[from_pretrained_glue_results['Model'] == 'DEPTH'][metric],
        marker=MARKERS[0],
        label='DEPTH',
        color=DEPTH_COLOR,
        linestyle=LINESTYLES[0],
    )
    # plt.title('SST-2 (From Pretrained)', fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('SST-2 Accuracy', fontsize=14)
    plt.legend(fontsize=14)
    # plt.tight_layout()
    plt.savefig('figures/sst2_from_pretrained.png')
    plt.show()
