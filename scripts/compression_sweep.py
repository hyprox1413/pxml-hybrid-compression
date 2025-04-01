import math
import subprocess

COMPRESSION_POWERS = [10, 25, 50, 100, 250, 500, 1000]
PURE_SCALES = [4, 8, 20, 40, 100, 200]

IMAGE_WIDTH = 1000

for power in COMPRESSION_POWERS:
    for scale in PURE_SCALES:
        pure_rank = IMAGE_WIDTH // (2 * power)
        pure_scale = scale
        pure_filters = 3 * pure_scale * pure_scale // power
        hybrid_scale = scale // 2
        hybrid_filters = int(math.sqrt(3 * IMAGE_WIDTH * hybrid_scale // (2 * power)))
        hybrid_rank = int(math.sqrt(3 * IMAGE_WIDTH * hybrid_scale // (2 * power)))
        if pure_rank > 0 and pure_scale > 0 and pure_filters > 0 and hybrid_scale > 0 and hybrid_filters > 0 and hybrid_rank > 0 and pure_filters < 1000 and hybrid_rank < IMAGE_WIDTH // hybrid_scale:
            print(f"power {power}, scale {scale}")
            print(f"running experiment with pure rank {pure_rank}, pure scale {pure_scale}, pure_filters {pure_filters}, hybrid scale {hybrid_scale}, hybrid filters {hybrid_filters}, and hybrid rank {hybrid_rank}")
            print()
        
            train_pure_command = ['python', 'train/conv.py', f'{pure_scale}', f'{pure_filters}']
            subprocess.run(train_pure_command)

            hybrid_pure_command = ['python', 'train/conv.py', f'{hybrid_scale}', f'{hybrid_filters}']
            subprocess.run(hybrid_pure_command)

            dump_command = ['python', 'eval/dump_test.py', f'{pure_rank}', f'{pure_scale}', f'{pure_filters}', f'{hybrid_scale}', f'{hybrid_filters}', f'{hybrid_rank}']
            subprocess.run(dump_command)
