# JMA_GPV ENS1M Processing

ENS1M データの変換、hourly/daily 作成、風速風向作成、確認用作図を行う処理群です。

## ディレクトリ構成

- `src/`
  処理本体
- `scripts/`
  実行用スクリプト
- `jmadata/`
  入力 GRIB2
- `data/`
  出力データ
- `log/`
  バッチ実行ログ

## 入力データ

入力は `jmadata` 配下の次を参照します。

- `ens_1w`
- `ens_2w1m`

月次相当の 1m データは `ens_2w1m` 内の `EPSC`、`0000UTC` ファイルを参照します。

## 出力先

現在の出力先は `data/ENS1M` です。

出力先の共通設定は [`src/ens1m_paths.py`](/Users/nemo/linux_work/JMA_GPV/src/ens1m_paths.py) にあります。
主に次を変更すれば、後で一括変更できます。

- `OUTPUT_SUBDIR`
- `LEGACY_OUTPUT_SUBDIRS`

読み込み側は後方互換のため、`ENS1M` を優先しつつ、旧 `ENS1M_NC` と `GEPS_NC` も参照できます。

## 主な実行スクリプト

- [`scripts/00_run_ens1m_batch.py`](/Users/nemo/linux_work/JMA_GPV/scripts/00_run_ens1m_batch.py)
  日次バッチ実行
- [`scripts/01_make_ens1m_netcdf_convert.py`](/Users/nemo/linux_work/JMA_GPV/scripts/01_make_ens1m_netcdf_convert.py)
  GRIB2 から `1w2w` / `1m` NetCDF 作成
- [`scripts/02_make_ens1m_hourly.py`](/Users/nemo/linux_work/JMA_GPV/scripts/02_make_ens1m_hourly.py)
  hourly 作成
- [`scripts/03_make_ens1m_hourly_wind.py`](/Users/nemo/linux_work/JMA_GPV/scripts/03_make_ens1m_hourly_wind.py)
  `UGRD` / `VGRD` から `WS` / `WD` の hourly 作成
- [`scripts/04_make_ens1m_daily.py`](/Users/nemo/linux_work/JMA_GPV/scripts/04_make_ens1m_daily.py)
  daily 作成
- [`scripts/05_make_ens1m_daily_wind.py`](/Users/nemo/linux_work/JMA_GPV/scripts/05_make_ens1m_daily_wind.py)
  `UGRD` / `VGRD` から `WS` / `WD` の daily 作成

## 出力ファイル一覧

出力ファイルは基本的に次の場所に作成されます。

```text
data/ENS1M/YYYY/VAR/
```

`YYYY` は処理対象年、`VAR` は変数名です。

### 変換段階ごとのファイル名

- `1w2w`
  `ENS1M_1w2w_yyyymmdd_VAR.nc`
- `1m`
  `ENS1M_1m_yyyymmdd_VAR.nc`
- `hourly`
  `ENS1M_hourly_yyyymmdd_VAR.nc`
- `daily`
  `ENS1M_daily_yyyymmdd_VAR.nc`

### 変数ごとの出力

| 変数 | 1w2w | 1m | hourly | daily | 備考 |
| --- | --- | --- | --- | --- | --- |
| `TMP` | `ENS1M_1w2w_yyyymmdd_TMP.nc` | `ENS1M_1m_yyyymmdd_TMP.nc` | `ENS1M_hourly_yyyymmdd_TMP.nc` | `ENS1M_daily_yyyymmdd_TMP.nc` | 気温 |
| `RH` | `ENS1M_1w2w_yyyymmdd_RH.nc` | `ENS1M_1m_yyyymmdd_RH.nc` | `ENS1M_hourly_yyyymmdd_RH.nc` | `ENS1M_daily_yyyymmdd_RH.nc` | 相対湿度 |
| `APCP` | `ENS1M_1w2w_yyyymmdd_APCP.nc` | `ENS1M_1m_yyyymmdd_APCP.nc` | `ENS1M_hourly_yyyymmdd_APCP.nc` | `ENS1M_daily_yyyymmdd_APCP.nc` | 降水量 |
| `PRMSL` | `ENS1M_1w2w_yyyymmdd_PRMSL.nc` | `ENS1M_1m_yyyymmdd_PRMSL.nc` | `ENS1M_hourly_yyyymmdd_PRMSL.nc` | `ENS1M_daily_yyyymmdd_PRMSL.nc` | 海面更正気圧 |
| `TCDC` | `ENS1M_1w2w_yyyymmdd_TCDC.nc` | `ENS1M_1m_yyyymmdd_TCDC.nc` | `ENS1M_hourly_yyyymmdd_TCDC.nc` | `ENS1M_daily_yyyymmdd_TCDC.nc` | 全雲量 |
| `UGRD` | `ENS1M_1w2w_yyyymmdd_UGRD.nc` | `ENS1M_1m_yyyymmdd_UGRD.nc` | `ENS1M_hourly_yyyymmdd_UGRD.nc` | `ENS1M_daily_yyyymmdd_UGRD.nc` | 東西風成分 |
| `VGRD` | `ENS1M_1w2w_yyyymmdd_VGRD.nc` | `ENS1M_1m_yyyymmdd_VGRD.nc` | `ENS1M_hourly_yyyymmdd_VGRD.nc` | `ENS1M_daily_yyyymmdd_VGRD.nc` | 南北風成分 |
| `WS` | なし | なし | `ENS1M_hourly_yyyymmdd_WS.nc` | `ENS1M_daily_yyyymmdd_WS.nc` | `UGRD` / `VGRD` から派生 |
| `WD` | なし | なし | `ENS1M_hourly_yyyymmdd_WD.nc` | `ENS1M_daily_yyyymmdd_WD.nc` | `UGRD` / `VGRD` から派生 |

### upper 変数

上空要素を使う場合は、変数名に気圧レベルを付けたディレクトリ名になります。

- `HGT850`
- `TMP700`
- `RH925`
- `VVEL700`

ファイル名の形は同じです。

- `ENS1M_1w2w_yyyymmdd_HGT850.nc`
- `ENS1M_1m_yyyymmdd_HGT850.nc`
- `ENS1M_hourly_yyyymmdd_HGT850.nc`
- `ENS1M_daily_yyyymmdd_HGT850.nc`

## バッチ実行例

1日分を処理する例です。

```bash
python scripts/00_run_ens1m_batch.py --date 20260329
```

日付範囲を連続処理する例です。

```bash
python scripts/00_run_ens1m_batch.py --start 20260329 --end 20260331
```

並列数を 1 にして止まった箇所を追いやすくする例です。

```bash
python scripts/00_run_ens1m_batch.py --date 20260329 --jobs 1 --stop-on-error
```

## ログ

バッチ実行時のログは `log/` に出力されます。

- `log/log_ens1m_yyyymmdd.log`
- 既存ファイルがある場合は `log/log_ens1m_yyyymmdd-02.log`, `-03.log` ...

ログには次を記録します。

- 全体の処理開始時刻 `BATCH START`
- 全体の処理終了時刻 `BATCH END`
- 各ステージの開始終了時刻 `STAGE START` / `STAGE END`
- 各サブプログラムの開始終了時刻 `SUBPROC START` / `SUBPROC END`

## 非木曜の予報延長

木曜日以外の実行では、当日処理後に直前木曜日の `hourly` / `daily` を使って予報期間を延長します。

- `hourly`
- `daily`
- `WS`
- `WD`

延長した場合は NetCDF メタデータに次が入ります。

- `extension_source_run_date`
- `extension_source_file`
- `extension_target_run_date`
- `extension_kind`
- `extension_appended_time_steps`

## 確認用作図

daily の確認図:

```bash
python scripts/16_plot_ens1m_daily_check_points.py --date 20260329
```

hourly の確認図:

```bash
python scripts/17_plot_ens1m_hourly_check_points.py --date 20260329
```

出力先は既定で `plots/ens1m_check/` です。
