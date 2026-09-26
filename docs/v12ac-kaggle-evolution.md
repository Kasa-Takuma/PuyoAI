# v12AC GPU進化探索

## 目的

`chain_builder_v12_ac` を基準に評価関数の係数を変えた候補を作り、同じツモ列で比較して強い候補だけを次の段階と次の世代へ残す。ゲームのルール、22通りの配置、現在手とNEXT2を使うdepth 3探索はJAXへ移し、KaggleのGPUで候補とゲームをまとめて計算する。

標準設定は1世代につき36候補である。第1段階は各候補3000手で上位12候補、第2段階は6000手で上位4候補、決勝は12000手で比較する。v12ACの基準値、現champion、Hall of Fame上位も各段階で保護する。各段階では新しいランダムseedを使い、候補間では同じツモ列を使う。

KaggleでT4またはP100の約16 GBが見えた場合は、1 GPUあたり12 profileを同時評価する。T4が2枚ならJAXが2枚へ同じ計算を分けるため、合計24 profileを同時評価する。24 GB以上なら1 GPUあたり18、GPUがないローカルCPUでは2を選ぶ。`--profile-batch` で明示的に変更できる。

## Kaggleへ投入する

ローカルで配置一式を作る。

```bash
npm run prepare:evolve:v12ac:kaggle
```

Kaggle CLIで非公開データセットとNotebookを登録する。

```bash
kaggle datasets create -p training/solo_search/artifacts/v12ac-evolution-kaggle/dataset
kaggle kernels push -p training/solo_search/artifacts/v12ac-evolution-kaggle/kernel
```

同じslugを更新する場合、データセット側は `kaggle datasets version` を使う。

```bash
kaggle datasets version -p training/solo_search/artifacts/v12ac-evolution-kaggle/dataset -m "update v12AC evolution source"
kaggle kernels push -p training/solo_search/artifacts/v12ac-evolution-kaggle/kernel
```

Kaggleの1ジョブでは4世代を実行する。現在確認できたGPU残量は19.39時間であり、以前のT4 2枚環境での実測と今回の処理量から、10世代を1ジョブで実行するより4世代ごとに正常終了・保存する方が安全である。GPU種別とコンパイル時間で所要時間は変わるため、最初の1世代が終わった時点の `gpuBatchSeconds` から次のジョブ数を判断する。

## 保存と再開

`/kaggle/working/v12ac-evolution/report.json` は各段階の終了時に更新される。Notebookが正常終了したら次のコマンドで回収する。

```bash
kaggle kernels output ikakun624/puyoai-v12ac-gpu-evolution -p log/kaggle-v12ac-evolution
```

続きから実行する場合は、回収したレポートを指定して配置一式を作り直す。

```bash
V12AC_RESUME_REPORT=log/kaggle-v12ac-evolution/v12ac-evolution/report.json \
  npm run prepare:evolve:v12ac:kaggle
kaggle datasets create -p training/solo_search/artifacts/v12ac-evolution-kaggle/resume-dataset
kaggle kernels push -p training/solo_search/artifacts/v12ac-evolution-kaggle/kernel
```

再開したKaggleジョブは、保存済み世代に加えて4世代を実行する。

## 選抜指標

順位は10連鎖以上、11連鎖以上、12連鎖以上、13連鎖以上を段階的に強く評価する。全消しも加点するが、11連鎖以上を上回る主目的にはしない。1〜9連鎖と敗北を減点し、予定手数を分母にするため途中敗北で実行手数が減った候補も有利にならない。

新候補をchampionにするには、決勝で現championより目的値が3%以上高く、11連鎖以上の頻度が現championの95%以上で、敗北数が増えていない必要がある。最終採用前には、進化に使っていない固定seedでchampionとv12ACを比較する。
