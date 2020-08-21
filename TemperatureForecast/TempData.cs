using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace TemperatureForecast
{
    /// <summary>
    /// 平均気温データ
    /// </summary>
    public class TempData
    {
        /// <summary>
        /// ６日前の平均気温
        /// </summary>
        [LoadColumn(0)]
        public float Data_6;
        /// <summary>
        /// ５日前の平均気温
        /// </summary>
        [LoadColumn(1)]
        public float Data_5;
        /// <summary>
        /// ４日前の平均気温
        /// </summary>
        [LoadColumn(2)]
        public float Data_4;
        /// <summary>
        /// ３日前の平均気温
        /// </summary>
        [LoadColumn(3)]
        public float Data_3;
        /// <summary>
        /// ２日前の平均気温
        /// </summary>
        [LoadColumn(4)]
        public float Data_2;
        /// <summary>
        /// １日前の平均気温
        /// </summary>
        [LoadColumn(5)]
        public float Data_1;
        /// <summary>
        /// 当日の平均気温
        /// </summary>
        [LoadColumn(6)]
        public float Data_0;
    }

    /// <summary>
    /// 予測データ
    /// </summary>
    public class TemperaturePrediction
    {
        [ColumnName("Score")]
        public float data_0;
    }
}
