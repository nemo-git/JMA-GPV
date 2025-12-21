## README

GEPSデータの処理プログラム


1.  python make_geps_netcdf_stream_with_stats_fixed3.py --only-monthly --date 20250731 --var TMP



    ap = argparse.ArgumentParser(description="GEPS Lsurf/L-pall/1m → NetCDF（pygrib, 51/25ens, 1日前/週次/月次）")

    "--date", required=True, help="基準日 yyyymmdd（1w2w: この1日前12UTC, 1m: 木曜に火/水12UTCを処理）"
    "--dir", default=DEFAULT_DIR, help=f"ホームDIR（既定: {DEFAULT_DIR}）"
    "--var", required=True, help="要素（Lsurf: TMP/RH/APCP/TCDC/PRMSL/UGRD/VGRD, L-pall: +HGT/VVEL）または短縮（例 TMP850）")
    
    
    --hgt               "等圧面 (hPa)。指定時は L-pall を処理"  925/850/700/500

    --debug             "ログ多め"
    --dry-run           "保存せず読み込み・検証のみ"
    --skip-monthly      "1ヶ月（1m）処理をスキップ"
    --only-monthly      "1ヶ月（1m）のみ処理（1w2wはスキップ）"
