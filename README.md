# Microsft.MLのチュートリアルと応用(平均気温予測)

Microsft.MLのチュートリアル:価格予測(回帰)を学習し、

応用として平均気温予測を作成したので記録として残しておきます。

言語は、C#です。

チュートリアルでは、回帰を使用してニューヨークのタクシーの価格予測が教材ですが、

それを応用して平均気温の予測をやってみました。

過去6日の平均気温から当日の平均気温を予測するという方法を採用しました。

学習用データは、気象庁の過去の気象データ検索を使用しています。

解析方法としては最も簡単な方法だと思います。



注意点）

特徴の追加時、数値として特徴を追加する必要があるため、

チュートリアルでは、[OneHotEncodingTransformer](https://docs.microsoft.com/ja-jp/dotnet/api/microsoft.ml.transforms.onehotencodingtransformer) 変換クラスを使用して

変換していますが、平均気温予測では数値のみの為に変換部分を削除しています。

カラム名の”Label”,"Features","Score"は、デフォルトカラム名をそのまま使用しています。

学習アルゴリズムは、

[FastTreeRegressionTrainer](https://docs.microsoft.com/ja-jp/dotnet/api/microsoft.ml.trainers.fasttree.fasttreeregressiontrainer) 機械学習タスク(Regression.Trainers.FastTree)を使用しました。

コメントアウトしていますが、他の機械学習タスクでもやってみました。

どれを選択すべきなのかは、勉強不足で不明です。

実行結果)

機械学習タスク(Regression.Trainers.FastTree)

RSquared Scoreが0.92

Root Mean Squared Errorが2.18

RMSEが少し大きい気がするが、学習データがあれなので、

まあ、こんなものなのでしょう。
