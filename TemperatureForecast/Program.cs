using System;
using System.IO;
using Microsoft.ML;

namespace TemperatureForecast
{
    class Program
    {
        /// <summary>
        /// 学習データパス
        /// </summary>
        static readonly string _trainDataPath = 
            Path.Combine(
                Environment.CurrentDirectory, 
                "Data", 
                "data_train.csv"
                );
        /// <summary>
        /// 評価データパス
        /// </summary>
        static readonly string _testDataPath = 
            Path.Combine(
                Environment.CurrentDirectory,
                "Data",
                "data_test.csv"
                );
        /// <summary>
        /// 学習済みモデルパス
        /// </summary>
        static readonly string _modelPath = 
            Path.Combine(
                Environment.CurrentDirectory,
                "Data",
                "model.zip"
                );

        static void Main(string[] args)
        {
            // ML.Net共通コンテキスト
            MLContext mlContext = new MLContext(seed: 0);

            // 学習実行
            var model = Train(
                mlContext: mlContext, 
                dataPath: _trainDataPath,
                modelPath: _modelPath,
                training: true
                );

            // 評価実行
            Evaluate(
                mlContext: mlContext,
                model: model,
                testDataPath: _testDataPath,
                hasHeader: false,
                separatorChar: ','
                );

            // 一件予測
            SinglePrediction(
                mlContext, 
                model,
                new TempData()
                {
                    Data_6 = 26.1f,
                    Data_5 = 26.3f,
                    Data_4 = 27.2f,
                    Data_3 = 28.3f,
                    Data_2 = 29.1f,
                    Data_1 = 28.5f,
                    Data_0 = 29.5f,
                });

            Console.Read();
        }

        /// <summary>
        /// 機械学習を実行する
        /// </summary>
        /// <param name="mlContext">ML.Net共通コンテキスト</param>
        /// <param name="dataPath">データパス</param>
        /// <param name="modelPath">モデルパス</param>
        /// <param name="training">学習フラグ</param>
        /// <returns>学習済みモデル</returns>
        public static ITransformer Train(
            MLContext mlContext,
            string dataPath,
            string modelPath,
            bool training = true
            )
        {
            // トレーニングしない場合、学習済みモデルをロードする
            if (!training)
            {
                DataViewSchema schema;
                return mlContext.Model.Load(
                    filePath: modelPath,
                    inputSchema: out schema
                    );
            }

            // 学習データ生成
            IDataView dataView = 
                mlContext.Data.LoadFromTextFile<TempData>(
                    path: dataPath, 
                    hasHeader: true, 
                    separatorChar: ','
                    );

            // パイプライン生成
            var pipeline = 
                mlContext.Transforms.CopyColumns(
                    outputColumnName: "Label", 
                    inputColumnName: "Data_0"
                    )
                    // 特徴データは数値でなければならないので数値に変換する
                    // 数値のみなので変換しない
                    //.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                    //.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                    //.Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                    // 特徴データの追加
                    .Append(mlContext.Transforms.Concatenate(
                        "Features",
                        "Data_6",
                        "Data_5",
                        "Data_4",
                        "Data_3",
                        "Data_2",
                        "Data_1"
                        ))
                    // 学習アルゴリズムはFastTree
                    .Append(mlContext.Regression.Trainers.FastTree());

            // 学習開始
            var model = pipeline.Fit(dataView);

            // 学習済みモデルを保存する
            mlContext.Model.Save(
                model: model,
                inputSchema: dataView.Schema, 
                filePath: modelPath
                );

            // 学習済みモデルを返す
            return model;
        }

        /// <summary>
        /// 評価実行
        /// </summary>
        /// <param name="mlContext">ML.Net共通コンテキスト</param>
        /// <param name="model">学習済みモデル</param>
        /// <param name="testDataPath">評価データパス</param>
        /// <param name="hasHeader">ヘッダー有無</param>
        /// <param name="separatorChar">区切り記号</param>
        private static void Evaluate(
            MLContext mlContext, 
            ITransformer model,
            string testDataPath,
            bool hasHeader = false,
            char separatorChar = ','
            )
        {
            // 評価データ読込
            IDataView dataView = 
                mlContext.Data.LoadFromTextFile<TempData>(
                    path: testDataPath,
                    hasHeader: hasHeader,
                    separatorChar: separatorChar
                    );

            // 評価データ変換
            var predictions = 
                model.Transform(
                    input: dataView
                    );

            // 評価実行
            var metrics = mlContext.Regression.Evaluate(
                data: predictions, 
                labelColumnName: "Label", 
                scoreColumnName: "Score"
                );

            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");
            Console.WriteLine($"*       RSquared Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       Root Mean Squared Error:      {metrics.RootMeanSquaredError:#.##}");
            Console.WriteLine($"*************************************************");
        }

        /// <summary>
        /// １件予測
        /// </summary>
        /// <param name="mlContext">ML.Net共通コンテキスト</param>
        /// <param name="model">学習済みモデル</param>
        /// <param name="singleData">１件データ</param>
        private static void SinglePrediction(
            MLContext mlContext, 
            ITransformer model,
            TempData singleData
            )
        {
            // 一件予測関数
            var predictionFunction =
                mlContext.Model.CreatePredictionEngine<TempData, TemperaturePrediction>(
                    transformer: model
                    );

            // 実測値退避
            var real = singleData.Data_0;

            // 予測実行
            var prediction = predictionFunction.Predict(singleData);

            // 結果表示
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.data_0:0.####}, actual fare: {real:0.####}");
            Console.WriteLine($"**********************************************************************");
        }
    }
}
