import os
import gc
import warnings
import subprocess
import importlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Type, Union # Union を追加
import json
import logging
import time
import random # LSTMModelのset_random_seedで使用
import numpy as np
import pandas as pd
from dataclasses import dataclass
from functools import lru_cache

# 必要な外部ライブラリのインストール＆インポート
def install_and_import(package_name: str, import_name: str = None, version_spec: Optional[str] = None):
    """
    パッケージをインストール（存在しない場合）してインポートする。
    バージョン指定も可能。
    """
    import_name = import_name or package_name
    try:
        module = importlib.import_module(import_name)
        # バージョンチェック (オプション)
        if version_spec and hasattr(module, '__version__'):
            from packaging.requirements import Requirement
            from packaging.version import parse as parse_version
            req = Requirement(f"{package_name}{version_spec}")
            if not req.specifier.contains(parse_version(module.__version__)):
                raise ImportError(f"{package_name}のバージョンが要求({version_spec})と異なります: {module.__version__}")
        return module
    except ImportError:
        package_to_install = package_name
        if version_spec:
            package_to_install += version_spec
        print(f"'{package_to_install}' をインストールしています...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_to_install]) # sys.executable を使用
            return importlib.import_module(import_name)
        except subprocess.CalledProcessError as e:
            print(f"'{package_to_install}' のインストールに失敗しました: {e}")
            raise
        except Exception as e: # インポート後の予期せぬエラー
            print(f"'{import_name}' のインポート中にエラーが発生しました: {e}")
            raise

# --- 外部ライブラリのインポート ---
# ログ出力はLoggerManager初期化後に行うため、ここではprintを使用
try:
    import sys # install_and_importで使用
    np = install_and_import("numpy")
    pd = install_and_import("pandas")
    plt = install_and_import("matplotlib", "matplotlib.pyplot")
    from matplotlib.axes import Axes
    sns = install_and_import("seaborn")
    sklearn_preprocessing = install_and_import("scikit-learn", "sklearn.preprocessing") # パッケージ名修正
    MinMaxScaler = sklearn_preprocessing.MinMaxScaler
    stats = install_and_import("scipy").stats
    ta = install_and_import("ta")
    optuna = install_and_import("optuna")
    tf = install_and_import("tensorflow")
    from tensorflow.keras.models import Sequential, save_model, load_model
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam as KerasAdam # Adamを明示的にインポート
except ImportError as e:
    print(f"必須ライブラリのインポート/インストールに失敗しました: {e}. プログラムを終了します。")
    sys.exit(1) # 致命的なエラーとして終了
except Exception as e: # その他の予期せぬエラー
    print(f"ライブラリ初期化中に予期せぬエラー: {e}. プログラムを終了します。")
    sys.exit(1)


warnings.filterwarnings('ignore', category=FutureWarning) # TensorFlow等のFutureWarningを抑制
warnings.filterwarnings('ignore', category=UserWarning)   # Seaborn等のUserWarningを抑制

# --- CurlSession の条件付きエイリアス定義 ---
CurlSession: Optional[Type[Union[Any, Any]]] = None # requests.Session or curl_cffi.requests.Session
# Union[requests.Session, curl_cffi.requests.Session] のように具体的な型を書くのが理想だが、
# インポート失敗時のために Any も許容。None の可能性もあるため Optional
try:
    from curl_cffi.requests import Session as CurlCffiSession
    CurlSession = CurlCffiSession
    print("INFO: curl_cffi.requests.Session を CurlSession として使用します。")
except ImportError:
    try:
        from requests import Session as RequestsSession
        CurlSession = RequestsSession
        print("INFO: requests.Session を CurlSession として使用します (curl_cffi が見つかりませんでした)。")
    except ImportError:
        print("WARNING: curl_cffi と requests のどちらも見つかりませんでした。HTTPリクエスト機能が制限されます。")
        # CurlSession は None のまま

class MarketDataValidator:
    """市場データの異常値検出とバリデーションクラス"""
    
    def __init__(self, logger_manager):
        self.logger = logger_manager.get_logger(self.__class__.__name__)
    
    def validate_prediction_consistency(self, predictions: Dict[str, float], current_price: float) -> Dict[str, Any]:
        """予測値の一貫性をチェック"""
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "adjusted_predictions": predictions.copy()
        }
        
        # 予測値の論理チェック
        pred_values = list(predictions.values())
        if len(pred_values) > 1:
            # 極端な変動率チェック（10%以上の差は警告）
            max_diff = max(pred_values) - min(pred_values)
            max_diff_pct = (max_diff / current_price) * 100
            
            if max_diff_pct > 10:
                validation_result["warnings"].append(
                    f"予測値間の差が大きすぎます: {max_diff_pct:.2f}%"
                )
                
            # 短期 > 長期の順序性チェック（下落トレンドの場合）
            if "nextday" in predictions and "long" in predictions:
                if predictions["nextday"] < predictions["long"]:
                    # 短期予測が長期予測より低い場合は調整
                    adjustment = (predictions["long"] - predictions["nextday"]) * 0.3
                    validation_result["adjusted_predictions"]["nextday"] += adjustment
                    validation_result["warnings"].append("短期予測を論理的整合性のため調整しました")
        
        # 異常な変動率チェック（1日で5%以上の変動は異常）
        if "nextday" in predictions:
            daily_change_pct = abs((predictions["nextday"] - current_price) / current_price) * 100
            if daily_change_pct > 5:
                validation_result["errors"].append(
                    f"翌日予測の変動率が異常です: {daily_change_pct:.2f}%"
                )
                validation_result["is_valid"] = False
        
        return validation_result
    
    def detect_outliers(self, data: pd.Series, method: str = "iqr") -> pd.Series:
        """外れ値検出"""
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (data < lower_bound) | (data > upper_bound)
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(data.dropna()))
            return z_scores > 3
        return pd.Series([False] * len(data), index=data.index)

    def smooth_predictions(self, predictions: Dict[str, float], smoothing_factor: float = 0.7) -> Dict[str, float]:
        """予測値の平滑化"""
        smoothed = {}
        sorted_keys = sorted(predictions.keys(), key=lambda x: {'nextday': 1, 'short': 20, 'long': 30}.get(x, 999))
        
        for i, key in enumerate(sorted_keys):
            if i == 0:
                smoothed[key] = predictions[key]
            else:
                # 前の予測値との差を平滑化
                prev_key = sorted_keys[i-1]
                raw_diff = predictions[key] - smoothed[prev_key]
                smoothed_diff = raw_diff * smoothing_factor
                smoothed[key] = smoothed[prev_key] + smoothed_diff
                
        return smoothed

class LoggerManager:
    """ロギング管理クラス（改善版）"""
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(threadName)s - %(message)s'

    def __init__(self, log_level: int = logging.INFO, log_file: Optional[str] = None):
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_level = log_level
        self.log_file = log_file
        self.performance_log: List[Dict[str, Any]] = []
        self.execution_cache: Dict[str, Any] = {}  # 重複実行防止
        self._setup_root_logger()

    def _setup_root_logger(self):
        """ルートロガーの基本的な設定。basicConfigは一度だけ呼び出されるべき。"""
        root_logger = logging.getLogger()
        if not root_logger.hasHandlers():
            handlers = []
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter(self.LOG_FORMAT))
            handlers.append(stream_handler)

            if self.log_file:
                try:
                    file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
                    file_handler.setFormatter(logging.Formatter(self.LOG_FORMAT))
                    handlers.append(file_handler)
                except IOError as e:
                    print(f"ログファイル '{self.log_file}' のオープンに失敗: {e}. ファイルログは無効になります。")

            logging.basicConfig(level=self.log_level, handlers=handlers)
            logging.getLogger('tensorflow').setLevel(logging.WARNING)
            logging.getLogger('matplotlib').setLevel(logging.WARNING)
            logging.getLogger('h5py').setLevel(logging.WARNING)
        else:
            root_logger.setLevel(self.log_level)

    def get_logger(self, name: str) -> logging.Logger:
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)
        logger.propagate = True
        self.loggers[name] = logger
        return logger

    def prevent_duplicate_execution(self, operation_key: str, operation_data: Any) -> bool:
        """重複実行を防止"""
        cache_key = f"{operation_key}_{hash(str(operation_data))}"
        current_time = datetime.now()
        
        # 5分以内の同じ操作は重複とみなす
        if cache_key in self.execution_cache:
            last_execution = self.execution_cache[cache_key]
            if (current_time - last_execution).total_seconds() < 300:  # 5分
                self.get_logger("DuplicateChecker").warning(f"重複実行を検出: {operation_key}")
                return True
        
        self.execution_cache[cache_key] = current_time
        return False

    def log_performance(self, operation: str, metrics: Dict[str, Any]) -> None:
        entry = metrics.copy()
        entry['timestamp'] = datetime.now().isoformat()
        entry['operation'] = operation
        self.performance_log.append(entry)

    def save_performance_log(self, filename: str = "performance_log.json") -> None:
        if not self.performance_log:
            return

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.performance_log, f, indent=2, ensure_ascii=False)
            self.get_logger(self.__class__.__name__).info(f"パフォーマンスログを '{filename}' に保存しました。")
        except IOError as e:
            self.get_logger(self.__class__.__name__).error(f"パフォーマンスログ保存エラー ({filename}): {e}")
        except Exception as e:
            self.get_logger(self.__class__.__name__).error(f"パフォーマンスログ保存中に予期せぬエラー: {e}", exc_info=True)


# --- アプリケーション全体で共有するLoggerManagerインスタンス ---
# main.py のようなエントリーポイントで一度だけ初期化するのが理想
# ここではグローバルスコープに置くが、依存性注入(DI)の方が望ましい
APP_LOGGER_MANAGER = LoggerManager(log_level=logging.INFO, log_file="market_system.log")


class Config:
    """設定管理クラス（問題修正版）- 既存機能の改善のみ"""

    DEFAULT_CONFIG = {
        "market_index_info": {
            "^GSPC": "S&P500指数",
            "^DJI": "NYダウ平均株価指数"
        },
        "csv_files": {
            "^GSPC": r"C:\Users\ds221k10159\Desktop\MymarketProject\Finance_Data\GSPC_ohlcv_5y_1d.csv",
            "^DJI": r"C:\Users\ds221k10159\Desktop\MymarketProject\Finance_Data\DJI_close_5y_1d.csv",
            "^VIX": r"C:\Users\ds221k10159\Desktop\MymarketProject\Finance_Data\VIX_close_5y_1d.csv"
        },
        "data_source_settings": {
            "fetch_period_years": "5y",
            "fetch_interval_days": "1d",
            "max_download_retries": 3,
            "download_retry_wait_seconds": 60,
            "wait_seconds_between_tickers": 15,
            "bulk_download_fail_wait_seconds": 180,
            "minimum_required_data_rows": 500,
            "data_backup_directory": "data_backup",
            "data_backup_max_age_days": 0
        },
        "feature_engineering_settings": {
            "use_vix_feature": True,
            "use_dji_for_gspc_feature": True,
            "technical_indicators_to_add": ["MA", "RSI", "MACD", "BB", "ATR", "CrossSignals"],
            "ma_windows": [5, 20, 60, 120],
            "rsi_window": 14,
            "bb_window": 20,
            "bb_std_dev": 2,
            "atr_window": 14,
            "macd_fast_period": 12,
            "macd_slow_period": 26,
            "macd_signal_period": 9,
        },
        # ===== 修正: AI予測重み付けの改善（既存機能の修正のみ） =====
        "ai_prediction_weights": {
            "nextday_weight": 0.3,      # 0.4 → 0.3 短期重視度を下げる
            "short_term_weight": 0.4,   # 0.4 → 0.4 中期を重視
            "long_term_weight": 0.3,    # 0.2 → 0.3 長期重視度を上げる
            "ai_confidence_threshold": 0.7,  # 0.6 → 0.7 信頼度閾値を上げる
            "extreme_prediction_threshold": 2.5  # 3.0 → 2.5 異常予測閾値を下げる
        },
        "investment_decision_overrides": {
            "max_ai_decline_for_buy": -1.5,  # -2.0 → -1.5 より保守的に
            "confidence_reliability_unified": True,
            "technical_ai_balance_ratio": 0.6  # テクニカル60%, AI40%
        },
        "model_training_settings": {
            "random_seed": 42,
            "lstm_input_columns_for_gspc": ["^GSPC", "VIX", "^DJI"],
            "train_test_split_ratio": 0.8,
            "hyperparameter_optimization_time_steps": 60,
            "hyperparameter_optimization_epochs": 50,
            "hyperparameter_optimization_early_stopping_patience": 10,
            "model_training_early_stopping_patience": 15,
            "default_optimizer_algorithm": "adam",
            "default_learning_rate": 0.001,
            "default_loss_function": "mean_squared_error",
            "model_save_path_template": "models/model_{ticker}_{name}.keras",
            # ===== 修正: モデル設定の一貫性改善 =====
            "sp500_prediction_model_configs": {
                "nextday": {
                    "input_time_steps": 60, 
                    "prediction_horizon_days": 1, 
                    "lstm_layers_count": 1,
                    "use_optuna_params": True, 
                    "training_epochs": 50, 
                    "training_batch_size": 64
                },
                "short": {
                    "input_time_steps": 80,  # 60 → 80 中期なので少し増加
                    "prediction_horizon_days": 20, 
                    "lstm_layers_count": 1,
                    "lstm_units_per_layer": 64, 
                    "lstm_dropout_rate": 0.2,
                    "training_epochs": 75, 
                    "training_batch_size": 64
                },
                "long": {
                    "input_time_steps": 120, 
                    "prediction_horizon_days": 30, 
                    "lstm_layers_count": 2,
                    "lstm_units_per_layer": 64, 
                    "lstm_dropout_rate": 0.25,  # 0.2 → 0.25 過学習防止強化
                    "training_epochs": 80, 
                    "training_batch_size": 32
                }
            }
        },
        "hyperparameter_optimization_settings": {
            "default_optuna_trials": 50,
            "load_best_hyperparameters_file": "best_lstm_params.json",
            "optuna_lstm_units_choices": [32, 64, 96, 128],
            "optuna_n_lstm_layers_range": [1, 2],
            "optuna_dropout_rate_range": [0.1, 0.5],
            "optuna_learning_rate_range": [1e-4, 1e-2],
            "optuna_batch_size_choices": [32, 64, 128]
        },
        "visualization_settings": {
            "plot_recent_days_count": 365,
            "plot_save_filename_template": "market_prediction_{ticker}.png",
            "plot_download_directory_candidates": ["Downloads", "ダウンロード", "."],
            "correlation_matrix_features": ["Close", "^DJI", "VIX", "RSI", "MACD_diff"],
            "plot_image_dpi": 300
        },
        # ===== 追加: 投資判定安全性強化設定（新規追加） =====
        "safety_enhancement_settings": {
            "ai_bearish_threshold": -3.0,      # AI下落予測の警告閾値
            "ai_bullish_threshold": 3.0,       # AI上昇予測の警告閾値
            "ai_strong_bearish_threshold": -5.0, # AI強い下落予測閾値
            "ai_strong_bullish_threshold": 5.0,  # AI強い上昇予測閾値
            "high_confidence_threshold": 0.80,   # 高信頼度閾値
            "enable_consistency_check": True,    # 整合性チェック有効化
            "conservative_mode": True            # 保守モード有効化
        },
        # ===== 追加: 信頼度別重み設定（新規追加） =====
        "confidence_based_weights": {
            "high_confidence": {
                "nextday_weight": 0.2,
                "short_term_weight": 0.3,
                "long_term_weight": 0.5
            },
            "medium_confidence": {
                "nextday_weight": 0.3,
                "short_term_weight": 0.4,
                "long_term_weight": 0.3
            },
            "low_confidence": {
                "nextday_weight": 0.4,
                "short_term_weight": 0.4,
                "long_term_weight": 0.2
            }
        },
        # ===== 追加: リスク管理強化設定（新規追加） =====
        "risk_management_settings": {
            "vix_thresholds": {
                "high": 30,
                "medium": 25,
                "low": 20
            },
            "volatility_thresholds": {
                "high": 25,
                "medium": 15
            },
            "enable_vix_override": True,         # VIX基準の判定上書き
            "enable_volatility_check": True     # ボラティリティチェック
        }
    }
    
    def __init__(self, config_path: Optional[str] = None, logger_manager=None):
        """設定クラスの初期化"""
        self.logger_manager = logger_manager
        self.logger = logger_manager.get_logger(self.__class__.__name__) if logger_manager else None
        
        self.config_data = self.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            try:
                if self.logger:
                    self.logger.info(f"設定ファイル読み込み開始: {config_path}")
                loaded_config = self._load_config(config_path)
                self.config_data = self._deep_update(self.config_data, loaded_config)
                if self.logger:
                    self.logger.info("設定ファイル読み込み完了")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"設定ファイル読み込み失敗: {e}. デフォルト設定を使用")
                # ファイル読み込み失敗時はデフォルト設定を使用
                pass
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルの読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as e:
            if self.logger:
                self.logger.error(f"設定ファイル読み込みエラー: {e}")
            return {}
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """辞書の深いマージ"""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_update(result[key], value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def get_config(cls, key_path: str = None):
        """設定値を取得"""
        if key_path is None:
            return cls.DEFAULT_CONFIG
        
        keys = key_path.split('.')
        config = cls.DEFAULT_CONFIG
        
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return None
        
        return config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """インスタンスメソッドでの設定値取得"""
        keys = key_path.split('.')
        config = self.config_data
        
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
        
        return config
    
    @classmethod
    def get_weights_by_confidence(cls, confidence: float):
        """信頼度に基づく重み取得"""
        if confidence > 0.85:
            return cls.DEFAULT_CONFIG["confidence_based_weights"]["high_confidence"]
        elif confidence > 0.7:
            return cls.DEFAULT_CONFIG["confidence_based_weights"]["medium_confidence"]
        else:
            return cls.DEFAULT_CONFIG["confidence_based_weights"]["low_confidence"]
    
    @classmethod
    def is_high_vix_environment(cls, vix_value: float):
        """高VIX環境かどうかの判定"""
        vix_thresholds = cls.DEFAULT_CONFIG["risk_management_settings"]["vix_thresholds"]
        return vix_value > vix_thresholds["high"]
    
    @classmethod
    def get_ai_prediction_threshold(cls, prediction_type: str):
        """AI予測閾値の取得"""
        safety_settings = cls.DEFAULT_CONFIG["safety_enhancement_settings"]
        threshold_map = {
            "bearish": safety_settings["ai_bearish_threshold"],
            "bullish": safety_settings["ai_bullish_threshold"],
            "strong_bearish": safety_settings["ai_strong_bearish_threshold"],
            "strong_bullish": safety_settings["ai_strong_bullish_threshold"]
        }
        return threshold_map.get(prediction_type, 0.0)


class CSVDataFetcher:
    """CSVファイルから市場データを取得するクラス (元のコードベース)"""

    def __init__(self, config: 'Config', logger_manager: LoggerManager):
        self.config = config
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.logger_manager = logger_manager # パフォーマンスログ用

        self.index_info = config.get("market_index_info", {}) # config.jsonのキー名に合わせる
        self.csv_files = config.get("csv_files", {}) # このキーはconfig.jsonにないので、別途追加が必要
        self.use_vix = config.get("feature_engineering_settings.use_vix_feature", True)
        self.use_dji_for_gspc = config.get("feature_engineering_settings.use_dji_for_gspc_feature", True)


    def _load_csv_file(self, file_path: str, ticker: str) -> Optional[pd.DataFrame]:
        self.logger.info(f"CSVファイル読み込み開始: {ticker} - '{file_path}'")
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"CSVファイルが見つかりません: {file_path}")
                return None

            df = pd.read_csv(file_path)

            if 'Date' not in df.columns:
                self.logger.error(f"'Date'列が見つかりません: {file_path}")
                return None

            # 日付パースの改善
            try:
                # タイムゾーン情報が含まれる可能性のあるパターンを正規表現で除去
                # 例: '2023-01-01 00:00:00-05:00' -> '2023-01-01 00:00:00'
                df['Date'] = pd.to_datetime(df['Date'].astype(str).str.replace(r'[+-]\d{2}:\d{2}$', '', regex=True), errors='coerce')
                df.dropna(subset=['Date'], inplace=True) # パース失敗した行は削除
                df.set_index('Date', inplace=True)
            except Exception as e:
                self.logger.error(f"日付列の処理中にエラー ({file_path}): {e}")
                return None


            if df.empty:
                self.logger.warning(f"CSVファイルに有効なデータがありません (または日付パース後空になった): {file_path}")
                return None

            # 必要な列の確認 (例: Close)
            if 'Close' not in df.columns:
                self.logger.warning(f"'Close'列がありません: {file_path}。ティッカー: {ticker}")
                # Closeがない場合は処理が難しいのでNoneを返すか、ダミーを入れるか設計による
                # return None

            # データ型の変換 (数値であるべき列)
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce') # 数値に変換できないものはNaNに

            df.sort_index(inplace=True) # 日付でソート
            self.logger.info(f"CSVファイル読み込み完了: {ticker} - {len(df)}行 ({df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')})")
            return df

        except FileNotFoundError: # 上でチェック済みだが念のため
            self.logger.error(f"CSVファイルが見つかりません (FileNotFoundError): {file_path}")
            return None
        except pd.errors.EmptyDataError:
            self.logger.warning(f"CSVファイルが空です: {file_path}")
            return None
        except Exception as e:
            self.logger.error(f"CSVファイル読み込み中に予期せぬエラー ({file_path}): {e}", exc_info=True)
            return None


    def _prepare_gspc_data(self, gspc_df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None, dji_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        self.logger.debug(f"S&P500データ準備開始。元データ {len(gspc_df)}行")
        try:
            df = gspc_df.copy()
            if 'Close' not in df.columns: # 主要な価格データがない場合は処理困難
                self.logger.error("S&P500データに'Close'列がありません。")
                return None
            df['^GSPC'] = df['Close'] # S&P500自身の終値を別名でも保持

            if self.use_vix and vix_df is not None and 'Close' in vix_df.columns:
                vix_aligned = vix_df['Close'].reindex(df.index).ffill().bfill()
                df['VIX'] = vix_aligned
                self.logger.debug("VIXデータをS&P500データに追加しました。")

            if self.use_dji_for_gspc and dji_df is not None and 'Close' in dji_df.columns:
                dji_aligned = dji_df['Close'].reindex(df.index, method='ffill').fillna(method='bfill')
                df['^DJI'] = dji_aligned
                self.logger.debug("NYダウデータをS&P500データに追加しました。")

            # OHLCVデータが揃っているか確認 (テクニカル指標計算に影響)
            # 揃っていなくても処理は続けるが、警告は出す
            for col in ['Open', 'High', 'Low', 'Volume']:
                if col not in df.columns:
                    self.logger.warning(f"S&P500データに'{col}'列がありません。一部テクニカル指標に影響する可能性があります。")
                    df[col] = df['Close'] # ダミーとしてCloseで埋める (テクニカル指標ライブラリがエラーにならないように)


            df.dropna(subset=['^GSPC'], inplace=True) # ^GSPC列にNaNがある行は削除 (主要データなので)
            if df.empty:
                self.logger.warning("S&P500データ処理後にデータが空になりました。")
                return None

            self.logger.info(f"S&P500データ準備完了: {len(df)}行")
            return df
        except Exception as e:
            self.logger.error(f"S&P500データ準備エラー: {e}", exc_info=True)
            return None


    def _prepare_dji_data(self, dji_df: pd.DataFrame, vix_df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        self.logger.debug(f"NYダウデータ準備開始。元データ {len(dji_df)}行")
        try:
            if 'Close' not in dji_df.columns:
                self.logger.error("NYダウデータに'Close'列がありません。")
                return None

            df = pd.DataFrame(index=dji_df.index)
            df['Close'] = dji_df['Close']
            df['^DJI'] = dji_df['Close']

            # NYダウは通常OHLVがない場合が多いので、Closeのみを主要データとする
            # テクニカル指標計算のためにダミー列を追加
            for col in ['Open', 'High', 'Low']:
                df[col] = df['Close']
            df['Volume'] = 0 # Volumeは通常ないので0で埋める

            if self.use_vix and vix_df is not None and 'Close' in vix_df.columns:
                vix_aligned = vix_df['Close'].reindex(df.index).ffill().bfill()
                df['VIX'] = vix_aligned
                self.logger.debug("VIXデータをNYダウデータに追加しました。")

            df.dropna(subset=['^DJI'], inplace=True)
            if df.empty:
                self.logger.warning("NYダウデータ処理後にデータが空になりました。")
                return None

            self.logger.info(f"NYダウデータ準備完了: {len(df)}行")
            return df
        except Exception as e:
            self.logger.error(f"NYダウデータ準備エラー: {e}", exc_info=True)
            return None


    def fetch_all_indexes(self) -> Dict[str, Dict[str, Any]]:
        """全ての指数データをCSVファイルから取得・準備する"""
        self.logger.info("CSVファイルからの市場データ読み込み処理開始...")
        start_time_fetch = datetime.now()
        fetched_data_store: Dict[str, Dict[str, Any]] = {}

        if not self.csv_files:
            self.logger.warning("設定に 'csv_files' が定義されていません。CSVデータ取得をスキップします。")
            return fetched_data_store
        if not self.index_info:
            self.logger.warning("設定に 'market_index_info' が定義されていません。主要な処理対象が不明です。")
            # return fetched_data_store # ここで処理を中断するかどうか

        # 1. 全CSVファイルを読み込み
        raw_csv_data: Dict[str, pd.DataFrame] = {}
        tickers_to_load = list(self.index_info.keys())
        if self.use_vix and '^VIX' not in tickers_to_load:
            tickers_to_load.append('^VIX') # VIXも対象に

        for ticker in tickers_to_load:
            file_path = self.csv_files.get(ticker)
            if file_path:
                df_loaded = self._load_csv_file(file_path, ticker)
                if df_loaded is not None and not df_loaded.empty:
                    raw_csv_data[ticker] = df_loaded
                else:
                    self.logger.warning(f"{ticker} のCSVデータ読み込みに失敗またはデータが空でした。")
            else:
                self.logger.warning(f"{ticker} のCSVファイルパスが設定 'csv_files' にありません。")


        # 2. 各主要指数のデータを準備
        vix_df_global = raw_csv_data.get('^VIX') if self.use_vix else None

        for ticker, name in self.index_info.items():
            self.logger.info(f"--- {name} ({ticker}) のデータ準備開始 ---")
            prepared_df: Optional[pd.DataFrame] = None

            if ticker == '^GSPC':
                gspc_base_df = raw_csv_data.get('^GSPC')
                if gspc_base_df is not None:
                    dji_base_df = raw_csv_data.get('^DJI') if self.use_dji_for_gspc else None
                    prepared_df = self._prepare_gspc_data(gspc_base_df, vix_df_global, dji_base_df)
                else:
                    self.logger.error(f"S&P500 ({ticker}) の元となるCSVデータが読み込まれていません。")
            elif ticker == '^DJI':
                dji_base_df = raw_csv_data.get('^DJI')
                if dji_base_df is not None:
                    prepared_df = self._prepare_dji_data(dji_base_df, vix_df_global)
                else:
                    self.logger.error(f"NYダウ ({ticker}) の元となるCSVデータが読み込まれていません。")
            # ... 他のティッカーの処理が必要な場合はここに追加 ...
            else:
                self.logger.warning(f"ティッカー '{ticker}' のデータ準備ロジックが実装されていません。スキップします。")
                continue

            if prepared_df is not None and not prepared_df.empty:
                fetched_data_store[ticker] = {
                    "df": prepared_df,
                    "ticker": ticker,
                    "name": name,
                    "scaler": None, # スケーラーはモデル学習時に設定
                    "scaled_data": None, # スケーリング済みデータも同様
                    "scaled_columns": None
                }
                self.logger.info(f"{name} ({ticker}) データ準備完了。期間: "
                                 f"{prepared_df.index.min().strftime('%Y-%m-%d')} to "
                                 f"{prepared_df.index.max().strftime('%Y-%m-%d')} ({len(prepared_df)}日分)")
            else:
                self.logger.error(f"{name} ({ticker}) のデータ準備に失敗しました。このティッカーは処理対象外となります。")

        duration_ms = (datetime.now() - start_time_fetch).total_seconds() * 1000
        self.logger_manager.log_performance(
            "fetch_data_from_csv",
            {
                "target_tickers_count": len(self.index_info),
                "successful_tickers_count": len(fetched_data_store),
                "duration_ms": round(duration_ms, 2),
                "loaded_tickers": list(raw_csv_data.keys()),
                "prepared_tickers": list(fetched_data_store.keys())
            }
        )

        if not fetched_data_store:
            self.logger.critical("有効な市場データをCSVから取得できませんでした。以降の処理に影響が出る可能性があります。")
        else:
            self.logger.info(f"CSVデータ準備完了。{len(fetched_data_store)}個の指数データを取得しました。")
        return fetched_data_store


class DataFetcher:
    """
    市場データ取得クラス（APIベース - レート制限対策強化版）
    このクラスは CurlSession が正しくエイリアス設定されていることを前提とする。
    """
    def __init__(self, config: 'Config', logger_manager: LoggerManager, session: Optional[Any] = None): # sessionの型はCurlSessionのエイリアス
        self.config = config
        self.logger = logger_manager.get_logger(self.__class__.__name__)
        self.logger_manager = logger_manager

        self.session = session
        if self.session is None and CurlSession is not None: # グローバルCurlSessionが利用可能ならそれを使う
            try:
                if CurlSession.__module__.startswith("curl_cffi"):
                    self.session = CurlSession(impersonate="chrome110")
                    self.logger.info("DataFetcher内でCurlSession (curl_cffi) を初期化しました。")
                else:
                    self.session = CurlSession()
                    self.logger.info("DataFetcher内でCurlSession (requests) を初期化しました。")
            except Exception as e:
                self.logger.error(f"DataFetcher内でのセッション初期化に失敗: {e}")
                self.session = None # やはり失敗
        elif self.session is None and CurlSession is None:
            self.logger.warning("DataFetcher: HTTPセッションが利用できません。APIベースのデータ取得は機能しません。")


        # 設定値の取得 (data_source_settings から)
        self.index_info = config.get("market_index_info", {})
        self.period = config.get("data_source_settings.fetch_period_years", "5y")
        self.interval = config.get("data_source_settings.fetch_interval_days", "1d")
        self.use_vix = config.get("feature_engineering_settings.use_vix_feature", True) # 特徴量設定から
        self.use_dji_for_gspc = config.get("feature_engineering_settings.use_dji_for_gspc_feature", True)

        self.max_retries = config.get("data_source_settings.max_download_retries", 3)
        self.retry_wait_seconds = config.get("data_source_settings.download_retry_wait_seconds", 60)
        self.inter_ticker_wait_seconds = config.get("data_source_settings.wait_seconds_between_tickers", 15)
        self.bulk_fail_wait_seconds = config.get("data_source_settings.bulk_download_fail_wait_seconds", 180)
        self.min_data_rows = config.get("data_source_settings.minimum_required_data_rows", 500)
        self.data_backup_dir = config.get("data_source_settings.data_backup_directory", "data_backup")
        self.backup_max_age_days = config.get("data_source_settings.data_backup_max_age_days", 0) # 0は無期限

        # yfinanceのセットアップ (もし使う場合)
        try:
            self.yf = install_and_import("yfinance")
            self.logger.info(f"yfinance version {self.yf.__version__} をロードしました。")
        except ImportError:
            self.yf = None
            self.logger.error("yfinanceライブラリのロードに失敗しました。APIベースのデータ取得は機能しません。")


    def _fetch_single_ticker_data_with_retry(self, ticker_symbol: str) -> Optional[pd.DataFrame]:
        """単一ティッカーのデータをリトライ付きで取得 (yfinanceを使用)"""
        if not self.yf:
            self.logger.error("yfinanceが利用不可なため、データ取得できません。")
            return None
        if not self.session: # yfinance自体はrequestsを使うが、カスタムセッションを渡せる場合がある
            self.logger.debug(f"{ticker_symbol}: HTTPセッションがないため、yfinanceのデフォルトセッションを使用します。")

        for attempt in range(self.max_retries + 1):
            try:
                self.logger.info(f"{ticker_symbol}: データ取得試行 {attempt + 1}/{self.max_retries + 1} (期間: {self.period}, 間隔: {self.interval})")
                # yf.Ticker(...).history(...) は内部でHTTPリクエストを行う
                # カスタムセッションを渡すオプションがあるか確認 (yfinanceのバージョンによる)
                # 現状のyfinanceでは直接requestsセッションを渡すAPIはない模様。
                # Proxy設定などはyf.set_proxy()で行う。
                # レート制限対策は主に時間をおくこと。
                ticker_obj = self.yf.Ticker(ticker_symbol) #, session=self.session if self.session else None) # session引数は公式にはない
                df = ticker_obj.history(period=self.period, interval=self.interval, auto_adjust=False) # auto_adjust=FalseでOHLCを保持

                if df.empty:
                    self.logger.warning(f"{ticker_symbol}: データ取得成功しましたが、DataFrameが空です。")
                    # 空でも成功として扱い、リトライしない場合もある。ここではリトライ対象とする。
                    if attempt < self.max_retries:
                        self.logger.info(f"{ticker_symbol}: {self.retry_wait_seconds}秒待機してリトライします。")
                        time.sleep(self.retry_wait_seconds)
                        continue
                    else: # 最終リトライでも空
                        return None # 空のDFを返すかNoneを返すか

                # タイムゾーン情報の除去 (Naiveなdatetimeに統一)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                df.dropna(subset=['Close'], inplace=True) # CloseがNaNの行は信頼性が低いので除去
                if len(df) < self.min_data_rows:
                    self.logger.warning(f"{ticker_symbol}: 取得データ行数 {len(df)} が最小要件 {self.min_data_rows} 未満です。")
                    if attempt < self.max_retries:
                        self.logger.info(f"{ticker_symbol}: {self.retry_wait_seconds}秒待機してリトライします。")
                        time.sleep(self.retry_wait_seconds)
                        continue
                    else: # 最終リトライでも不足
                        self.logger.error(f"{ticker_symbol}: データ行数不足で取得失敗。")
                        return None

                self.logger.info(f"{ticker_symbol}: データ取得成功 ({len(df)}行)。")
                return df

            except requests.exceptions.RequestException as re: # requestsライブラリ由来のエラー
                self.logger.error(f"{ticker_symbol}: データ取得中にネットワークエラー (試行 {attempt + 1}): {re}")
            except Exception as e: # yfinance内部エラーやその他の予期せぬエラー
                self.logger.error(f"{ticker_symbol}: データ取得中に予期せぬエラー (試行 {attempt + 1}): {e}", exc_info=True)

            if attempt < self.max_retries:
                self.logger.info(f"{ticker_symbol}: {self.retry_wait_seconds}秒待機してリトライします。")
                time.sleep(self.retry_wait_seconds)
            else:
                self.logger.error(f"{ticker_symbol}: 最大リトライ回数({self.max_retries})に達しました。データ取得失敗。")
        return None


    def _backup_data(self, df: pd.DataFrame, ticker_symbol: str) -> None:
        """DataFrameを指定されたディレクトリにCSVとしてバックアップする"""
        if not os.path.exists(self.data_backup_dir):
            try:
                os.makedirs(self.data_backup_dir)
                self.logger.info(f"バックアップディレクトリを作成しました: {self.data_backup_dir}")
            except OSError as e:
                self.logger.error(f"バックアップディレクトリ作成失敗: {e}. バックアップをスキップします。")
                return

        # ファイル名: ticker_YYYYMMDD_HHMMSS.csv
        filename = f"{ticker_symbol.replace('^','')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.data_backup_dir, filename)
        try:
            df.to_csv(filepath)
            self.logger.info(f"{ticker_symbol}のデータをバックアップしました: {filepath}")
        except IOError as e:
            self.logger.error(f"{ticker_symbol}のデータバックアップ失敗 ({filepath}): {e}")
        except Exception as e:
            self.logger.error(f"{ticker_symbol}のデータバックアップ中に予期せぬエラー: {e}", exc_info=True)


    def _cleanup_old_backups(self) -> None:
        """古いバックアップファイルを削除する"""
        if self.backup_max_age_days <= 0: # 0以下なら削除しない
            return
        if not os.path.exists(self.data_backup_dir):
            return

        self.logger.info(f"古いバックアップファイルのクリーンアップ開始 (保持期間: {self.backup_max_age_days}日)...")
        now = datetime.now()
        cleaned_count = 0
        try:
            for filename in os.listdir(self.data_backup_dir):
                filepath = os.path.join(self.data_backup_dir, filename)
                if os.path.isfile(filepath):
                    try:
                        # ファイル名から日付をパースする試み (例: ticker_YYYYMMDD_HHMMSS.csv)
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            date_str = parts[-2] # YYYYMMDD
                            if len(date_str) == 8 and date_str.isdigit():
                                file_date = datetime.strptime(date_str, "%Y%m%d")
                                if (now - file_date).days > self.backup_max_age_days:
                                    os.remove(filepath)
                                    self.logger.debug(f"古いバックアップファイルを削除しました: {filepath}")
                                    cleaned_count += 1
                            else: # 日付形式でないファイルは最終更新日時で判断
                                file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                                if (now - file_mod_time).days > self.backup_max_age_days:
                                    os.remove(filepath)
                                    self.logger.debug(f"古いバックアップファイル(更新日時基準)を削除: {filepath}")
                                    cleaned_count += 1
                        else: # ファイル名形式が合わない場合も最終更新日時
                            file_mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                            if (now - file_mod_time).days > self.backup_max_age_days:
                                os.remove(filepath)
                                self.logger.debug(f"古いバックアップファイル(形式不一致、更新日時基準)を削除: {filepath}")
                                cleaned_count += 1
                    except ValueError: # 日付パース失敗
                        self.logger.debug(f"バックアップファイル名から日付をパースできませんでした: {filename}")
                    except OSError as e_remove:
                        self.logger.warning(f"バックアップファイル削除エラー ({filepath}): {e_remove}")
            if cleaned_count > 0:
                self.logger.info(f"{cleaned_count}個の古いバックアップファイルを削除しました。")
            else:
                self.logger.info("削除対象の古いバックアップファイルはありませんでした。")
        except Exception as e:
            self.logger.error(f"バックアップクリーンアップ中に予期せぬエラー: {e}", exc_info=True)

    def fetch_all_indexes(self) -> Dict[str, Dict[str, Any]]:
        """
        設定された全ての指数データをAPI経由で取得・準備する。
        CSVDataFetcher と同じ出力形式の辞書を返す。
        """
        self.logger.info("API経由での市場データ取得処理開始...")
        start_time_fetch_api = datetime.now()
        market_data_store_api: Dict[str, Dict[str, Any]] = {}

        if not self.yf:
            self.logger.critical("yfinanceが利用できないため、APIでのデータ取得は不可能です。")
            return market_data_store_api
        if not self.index_info:
            self.logger.warning("設定 'market_index_info' が空です。取得対象がありません。")
            return market_data_store_api

        tickers_to_process = list(self.index_info.keys())
        if self.use_vix and '^VIX' not in tickers_to_process:
            tickers_to_process.append('^VIX')

        raw_fetched_dfs: Dict[str, pd.DataFrame] = {}
        failed_tickers: List[str] = []

        # 1. 各ティッカーの生データを取得
        for i, ticker in enumerate(tickers_to_process):
            df_raw = self._fetch_single_ticker_data_with_retry(ticker)
            if df_raw is not None and not df_raw.empty:
                raw_fetched_dfs[ticker] = df_raw
                self._backup_data(df_raw, ticker) # 取得成功したらバックアップ
            else:
                self.logger.error(f"{ticker} のデータ取得に最終的に失敗しました。")
                failed_tickers.append(ticker)

            if i < len(tickers_to_process) - 1 and self.inter_ticker_wait_seconds > 0: # 最後のティッカー以外
                self.logger.debug(f"{self.inter_ticker_wait_seconds}秒待機 (次のティッカー取得前)...")
                time.sleep(self.inter_ticker_wait_seconds)

        if len(failed_tickers) == len(tickers_to_process) and tickers_to_process: # 全滅した場合
            self.logger.critical(f"全てのティッカー ({', '.join(failed_tickers)}) のデータ取得に失敗しました。{self.bulk_fail_wait_seconds}秒待機します。")
            time.sleep(self.bulk_fail_wait_seconds)
            # ここで処理を中断するか、空のデータを返すかは設計による
            return market_data_store_api


        # 2. CSVDataFetcherと同様のデータ準備ロジックを適用
        #    CSVDataFetcherのメソッドを再利用できるように、一時的なCSVFetcherインスタンスを作るか、
        #    準備ロジックを共通化する。ここでは簡易的にCSVFetcherの準備メソッドを呼び出す。
        temp_csv_fetcher = CSVDataFetcher(self.config, self.logger_manager) # logger_managerを渡す
        vix_df_global_api = raw_fetched_dfs.get('^VIX') if self.use_vix else None

        for ticker, name in self.index_info.items():
            if ticker in failed_tickers: # 生データ取得失敗したものはスキップ
                continue

            self.logger.info(f"--- {name} ({ticker}) のAPI取得データ準備開始 ---")
            prepared_df_api: Optional[pd.DataFrame] = None
            base_df = raw_fetched_dfs.get(ticker)

            if base_df is None or base_df.empty:
                self.logger.error(f"{name} ({ticker}) の元となるAPIデータがありません。")
                continue

            if ticker == '^GSPC':
                dji_base_df_api = raw_fetched_dfs.get('^DJI') if self.use_dji_for_gspc else None
                prepared_df_api = temp_csv_fetcher._prepare_gspc_data(base_df, vix_df_global_api, dji_base_df_api)
            elif ticker == '^DJI':
                prepared_df_api = temp_csv_fetcher._prepare_dji_data(base_df, vix_df_global_api)
            # ... 他のティッカー ...
            else:
                self.logger.warning(f"ティッカー '{ticker}' のAPIデータ準備ロジックが未実装。元データをそのまま使用します。")
                prepared_df_api = base_df # とりあえずそのまま格納

            if prepared_df_api is not None and not prepared_df_api.empty:
                market_data_store_api[ticker] = {
                    "df": prepared_df_api, "ticker": ticker, "name": name,
                    "scaler": None, "scaled_data": None, "scaled_columns": None
                }
                self.logger.info(f"{name} ({ticker}) APIデータ準備完了。 ({len(prepared_df_api)}日分)")
            else:
                self.logger.error(f"{name} ({ticker}) のAPIデータ準備に失敗。")

        self._cleanup_old_backups() # 古いバックアップを削除

        duration_ms_api = (datetime.now() - start_time_fetch_api).total_seconds() * 1000
        self.logger_manager.log_performance(
            "fetch_data_from_api",
            {
                "target_tickers_count": len(self.index_info),
                "successful_tickers_count": len(market_data_store_api),
                "duration_ms": round(duration_ms_api, 2),
                "fetched_tickers_raw": list(raw_fetched_dfs.keys()),
                "prepared_tickers": list(market_data_store_api.keys()),
                "failed_tickers_raw": failed_tickers
            }
        )

        if not market_data_store_api:
            self.logger.critical("API経由で有効な市場データを取得できませんでした。")
        else:
            self.logger.info(f"APIデータ準備完了。{len(market_data_store_api)}個の指数データを取得。")
        return market_data_store_api

