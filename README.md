# VITAL-CHD Bayesian Re-analysis

このプロジェクトは、VITAL試験のCHD（冠動脈心疾患）データを用いたベイズ生存解析の実装です。LLM生成事前分布と従来の事前分布を比較し、推測・予測の両面での性能を評価します。

## プロジェクト概要

### 目的
- **VITAL試験**の総冠動脈疾患（`totchd`）を主要アウトカムとする
- **Weibull比例ハザード（PH）**モデルによるベイズ解析
- **LLM生成事前分布**（Llama 3.3 70B / MedGemma 27B）と**既存の5種事前分布**の比較
- **推測・予測**両面での性能評価

### データ
- **入力ファイル**: `data/VITAL_trial_NEJM_2022.csv`
- **主要アウトカム**: 
  - イベント指標: `totchd`（総CHD: 0/1）
  - 追跡時間: `chdyears`（CHD発症/打ち切りまでの年数）
- **共変量**: `ageyr`, `sex`（Hamayaらと同じ設定）

### 解析手法
- **モデル**: ベイズWeibull比例ハザード
- **実装**: Python + PyMC（HMC/NUTS）
- **設定**: 3 chains, 4,000 draws（2,000 warmup）, target_accept ≥ 0.9
- **評価**: 推測（HR, P(HR<1)）+ 予測（PSIS-LOO, WAIC）

## 環境セットアップ

### 前提条件
- Python 3.11+
- [rye](https://rye-up.com/) パッケージマネージャ

### インストール

```bash
# プロジェクトルートに移動
cd vital_chd_bayes

# 依存関係をインストール
rye sync
```

### 環境変数設定

`.env`ファイルを編集してAPI keyを設定：

```bash
API_KEY=your_actual_api_key_here
```

## プロジェクト構造

```
vital_chd_bayes/
├─ data/
│  └─ VITAL_trial_NEJM_2022.csv     # VITAL試験データ
├─ src/
│  ├─ config.py                     # 設定・定数
│  ├─ io.py                         # データ読み込み・前処理
│  ├─ priors.py                     # 事前分布定義・LLM取得
│  ├─ model_weibull_ph.py           # Weibull-PHモデル実装
│  ├─ inference.py                  # MCMC実行・推測要約
│  ├─ predictive.py                 # 予測性能評価
│  ├─ reporting.py                  # 可視化・レポート生成
│  └─ run_experiments.py            # 実験オーケストレーション
├─ results/
│  ├─ tables/                       # CSV出力
│  └─ figures/                      # PNG図表
├─ .env                             # API設定
├─ README.md                        # 本ファイル
└─ pyproject.toml                   # rye設定
```

## 実行方法

### 完全実験の実行

```bash
# プロジェクトルートで実行
rye run python -m src.run_experiments
```

### 個別モジュールのテスト

```bash
# データ読み込みテスト
rye run python -m src.io

# 事前分布取得テスト（API keyが必要）
rye run python -m src.priors

# モデル作成テスト
rye run python -m src.model_weibull_ph
```

## 事前分布仕様

### 既存の5種類（log-HR ~ Normal(μ, σ²)）

| 名前 | μ | σ | 説明 |
|------|---|---|------|
| **Noninformative** | 0.0 | 10.0 | 非情報事前、データ主導 |
| **Primary informed** | -0.072 | 0.037 | メタ解析による一次予防効果 |
| **Weakly** | -0.072 | 0.055 | Primary の1.5倍の不確実性 |
| **Strong** | -0.072 | 0.018 | Primary の0.5倍の不確実性 |
| **Skeptical** | 0.0 | 0.121 | 効果に懐疑的（効果<5%） |

### LLM生成事前分布

- **モデル**: 
  - `llama-3.3-70b-instruct`（汎用）
  - `medgemma-27b-it`（医療特化）
- **温度**: 0（再現性のため）
- **出力形式**: JSON `{"mu": <float>, "sigma": <float>}`

## 出力ファイル

### 表（results/tables/）
- `inference_summary.csv`: 推測結果要約
- `predictive_summary.csv`: 予測性能要約
- `priors_summary.csv`: 使用した事前分布一覧
- `summary_report.txt`: 総合レポート

### 図（results/figures/）
- `fig_HR_by_prior.png`: 事前分布別HR比較
- `fig_LOO_by_prior.png`: LOO予測性能比較
- `fig_prior_posterior_comparison.png`: 事前・事後分布比較

## 主要な評価指標

### 推測（Inference）
- **HR**: ハザード比（exp(log-HR)）の事後分布
- **95% CrI**: 95%信用区間
- **P(HR < 1)**: 保護効果の確率
- **収束診断**: R-hat ≤ 1.01, ESS

### 予測（Predictive）
- **PSIS-LOO**: 交差検証による予測性能
- **WAIC**: 情報規準
- **ΔLOO**: 基準モデル（Primary informed）からの差

## 評価の観点

LLM事前分布が以下を示す場合「有用」と評価：

1. **CrI幅の縮小** - より精密な推定
2. **P(HR<1)の向上** - 保護効果の確信度向上
3. **LOOの改善** - より良い予測性能（ΔLOO > 0）

## トラブルシューティング

### よくある問題

1. **API key エラー**
   ```
   ValueError: API_KEY not found in environment variables
   ```
   → `.env`ファイルでAPI_KEYを正しく設定してください

2. **収束エラー**
   ```
   Warning: Some R-hat values exceed 1.01
   ```
   → MCMC設定（chains, draws, target_accept）を調整してください

3. **メモリエラー**
   → サンプル数を減らすか、より大きなメモリの環境で実行してください

### 依存関係の問題

```bash
# パッケージの再インストール
rye sync --force

# 特定パッケージの追加
rye add package_name
```

## 開発情報

### コード品質

```bash
# コードフォーマット
rye run black src/
rye run isort src/

# リント
rye run ruff check src/
```

### ログ

実行ログは `results/tables/experiment.log` に保存されます。

## 参考文献

- Hamaya et al. のVITAL試験ベイズ解析手法
- PyMC documentation
- ArviZ documentation for model comparison

## ライセンス

このプロジェクトは研究目的で作成されています。

## 連絡先

プロジェクトに関する質問や問題は、Issueを作成してください。
