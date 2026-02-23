import json
import argparse
import os

# ---------------------------------------------------------------
# 필수 키 정의 — 누락 시 즉시 에러
# ---------------------------------------------------------------
REQUIRED_COMMON_KEYS = [
    'model', 'img_size', 'in_channels', 'class_num',
    'attn_dropout', 'proj_dropout',
    't_eps', 'noise_scale',
    'cfg', 'interval_min', 'interval_max',
    'sampling_method', 'num_sampling_steps',
    'device', 'seed'
]

REQUIRED_TRAIN_KEYS = [
    'batch_size', 'epochs',
    'blr', 'min_lr', 'lr_schedule', 'warmup_epochs',
    'weight_decay',
    'P_mean', 'P_std', 'label_drop_prob',
    'ema_decay1', 'ema_decay2',
    'num_workers', 'pin_mem',
    'log_freq', 'save_last_freq', 'eval_freq', 'online_eval',
    'num_images', 'gen_bsz',
    'data_path', 'output_dir'
]

REQUIRED_INFERENCE_KEYS = [
    'num_images', 'gen_bsz',
    'checkpoint', 'output_dir'
]

# 코드에서 처리하는 메타 키 — argparse에 넘기지 않음
META_KEYS = {'CONFIG_TYPE', 'CONFIG_VERSION', 'COMMON_CONFIG'}


# ---------------------------------------------------------------
# 내부 유틸
# ---------------------------------------------------------------
def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Config Error] 파일을 찾을 수 없습니다: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _validate(config: dict, required_keys: list, config_type: str):
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(
            f"[Config Error] '{config_type}' config에 누락된 키:\n  {missing}\n  json을 확인하세요."
        )


def _filter(config: dict) -> dict:
    """메타 키와 주석 키 제거 후 반환"""
    return {
        k: v for k, v in config.items()
        if k not in META_KEYS and not k.startswith('#')
    }


def _merge(common: dict, specific: dict) -> dict:
    """common을 base로, specific이 오버라이드"""
    return {**common, **specific}


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------
def load_train_config(common_path: str, train_path: str) -> argparse.Namespace:
    common = _load_json(common_path)
    train  = _load_json(train_path)
    merged = _merge(common, train)

    _validate(merged, REQUIRED_COMMON_KEYS, 'common')
    _validate(merged, REQUIRED_TRAIN_KEYS,  'train')

    return argparse.Namespace(**_filter(merged))


def load_inference_config(common_path: str, inference_path: str) -> argparse.Namespace:
    common    = _load_json(common_path)
    inference = _load_json(inference_path)
    merged    = _merge(common, inference)

    _validate(merged, REQUIRED_COMMON_KEYS,    'common')
    _validate(merged, REQUIRED_INFERENCE_KEYS, 'inference')

    return argparse.Namespace(**_filter(merged))
