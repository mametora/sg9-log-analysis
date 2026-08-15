# SG9 RomRaiderログ解析

Subaru Forester SG9（TA-SG9 / MT / 国内仕様）のRomRaiderログから、加速イベントとハンチング候補を探索する小規模な解析環境です。特定のRPM帯、ノック、MAFを原因と決め打ちせず、正常寄りの加速との差を確認します。

## セットアップ

[uv](https://docs.astral.sh/uv/) を使用します。Python 3.14が未導入の場合もuvが取得します。

```bash
uv sync
```

CSVを `data/raw/` に置き、次を実行します。

```bash
uv run python -m src.analysis data/raw/*.csv
```

体感申告ログや追加確認区間をレポートへ反映する場合は、公開用サンプルをコピーしてローカル注釈を作ります。

```bash
cp annotations.example.json annotations.local.json
uv run python -m src.analysis data/raw/*.csv --annotations annotations.local.json
```

`annotations.local.json` には実際のログ名や経過秒を記載します。このファイルはGit管理対象外で、解析ロジックにはログ取得日時を埋め込みません。

出力先は次のとおりです。

- `output/reports/log_inventory.csv`: 行数、期間、周期、欠損、ヘッダ区間
- `output/reports/parameter_statistics.csv`: パラメータ別の min / max / mean / median / std
- `output/reports/events.csv`: 全加速イベントと候補スコア
- `output/reports/event_samples_with_maf_reference.csv`: 条件付きMAF期待値と残差
- `output/reports/knock_changes.csv`: Knock Sum変化時刻と前後3秒窓
- `output/reports/learning_summary.csv`: IAM・FLKC・取得できた学習値のログ間比較
- `output/reports/reported_core_comparison.csv`: 体感区間と条件を合わせた他ログの比較
- `output/reports/initial_analysis.md`: 初期解析レポート
- `output/plots/`: 全体波形、終盤詳細、MAF関係図、候補比較

別の出力先を使う場合は `--output` を指定します。

```bash
uv run python -m src.analysis data/raw/*.csv --output /tmp/sg9-analysis
```

## 実装上の扱い

- RomRaiderが記録途中にヘッダを再出力し、取得項目を変更したログも、区間別に読み取って列の和集合へ統合します。原本CSVは変更しません。
- 加速イベントはThrottle、RPM、継続時間、RPM上昇量から抽出し、急なRPM低下ではシフト操作の可能性を考えて分割します。
- MAF期待値は、正常寄り加速のRPM・Throttle・過給圧が近いサンプルからロバストに推定します。
- 候補スコアは波形を確認する順番を決める探索指標です。故障診断やノック発生の断定には使用できません。
- `Primary Wastegate Duty Cycle = 0%` は、当該SG9でSSM項目が有効と確認できない限り実WGDCとは解釈しません。
- ログ固有のファイル名、日時、確認窓は任意のローカル注釈JSONへ分離しています。注釈なしでも自動探索は実行できます。

## テスト

```bash
uv run python -m unittest discover -s tests -v
```

`pyproject.toml` が直接依存関係、`uv.lock` が解決済みの全依存関係を管理します。Jupyterを除いた実行環境だけが必要な場合は `uv sync --no-dev` を使用します。

生ログ、ローカル注釈、仮想環境、生成物はGit管理対象外です。コードと一般的な解析条件をGitで管理し、ログ追加時に同じコマンドで再解析できます。

## License

[MIT License](LICENSE)
