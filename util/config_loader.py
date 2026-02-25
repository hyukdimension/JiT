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
    'device'
]

REQUIRED_TRAIN_KEYS = [
    'batch_size', 'epochs',
    'blr', 'min_lr', 'lr_schedule', 'warmup_epochs',
    'weight_decay',
    'P_mean', 'P_std', 'label_drop_prob',
    'ema_decay1', 'ema_decay2',
    'num_workers', 'pin_mem',
    'train_cfg', 'train_interval_min', 'train_interval_max',
    'log_freq', 'save_last_freq', 'eval_freq', 'online_eval',
    'eval_num_images', 'eval_gen_bsz',
    'sampling_method', 'num_sampling_steps',
    'data_path', 'train_output_dir', 'train_seed'
]

REQUIRED_INFERENCE_KEYS = [
    'num_images', 'gen_bsz',
    'checkpoint', 'output_dir',
    'infer_cfg', 'infer_interval_min', 'infer_interval_max', 'infer_seed',
    'sampling_method', 'num_sampling_steps',
]

# 반드시 일치해야 하는 키 — 학습 snapshot과 추론 args 비교
MUST_MATCH_KEYS = [
    'model', 'img_size', 'in_channels', 'class_num',
    'attn_dropout', 'proj_dropout',
    't_eps', 'noise_scale',
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

    args = argparse.Namespace(**_filter(merged))

    # alias — 코드 내부에서는 기존 변수명 그대로 사용
    args.output_dir   = args.train_output_dir
    args.cfg          = args.train_cfg
    args.interval_min = args.train_interval_min
    args.interval_max = args.train_interval_max
    args.seed         = args.train_seed
    args.num_images   = args.eval_num_images
    args.gen_bsz      = args.eval_gen_bsz

    return args


def load_inference_config(common_path: str, inference_path: str) -> argparse.Namespace:
    common    = _load_json(common_path)
    inference = _load_json(inference_path)
    merged    = _merge(common, inference)

    _validate(merged, REQUIRED_COMMON_KEYS,    'common')
    _validate(merged, REQUIRED_INFERENCE_KEYS, 'inference')

    args = argparse.Namespace(**_filter(merged))

    # alias
    args.cfg          = args.infer_cfg
    args.interval_min = args.infer_interval_min
    args.interval_max = args.infer_interval_max
    args.seed         = args.infer_seed

    return args


def verify_config_snapshot(args, snapshot_path):
    """train_config_snapshot.json과 현재 args를 비교.
    MUST_MATCH_KEYS 항목이 다르면 즉시 에러.
    """
    if not os.path.exists(snapshot_path):
        print(f"[!] train_config_snapshot.json 없음 — 검증 스킵: {snapshot_path}")
        return

    with open(snapshot_path, 'r', encoding='utf-8') as f:
        train_cfg = json.load(f)

    mismatches = []
    for key in MUST_MATCH_KEYS:
        train_val = train_cfg.get(key)
        infer_val = getattr(args, key, None)
        if train_val != infer_val:
            mismatches.append(f"  '{key}': 학습시={train_val!r}, 현재={infer_val!r}")

    if mismatches:
        raise ValueError(
            "[Config Mismatch] snapshot과 일치하지 않는 항목:\n"
            + "\n".join(mismatches)
        )

    print("[*] Config 검증 완료 — snapshot과 일치")