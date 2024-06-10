using HocMay.Models;
using Microsoft.AspNetCore.Components.Forms;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;

namespace HocMay.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index(ThongSo inputND)
        {
            inputND.SalePrice = DuDoan(inputND);      
            return View(inputND);
        }
        private string DuDoan(ThongSo inputND)
        {
            float[] fix = new float[222] { 4, 60, 7200, 1, 1, 3, 0, 5, 4, 2006, 1950, 0, 3, 4, 4, 4, 3, 6, 0, 6, 0, 0,
                0, 0, 1, 864, 0, 0, 864, 0, 0, 2, 0, 3, 1, 3, 6, 6, 0, 3, 0, 3, 2, 0, 5, 5, 2, 0, 0, 0, 0, 0, 0,
                3, 4, 0, 8, 3, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
            string onnxPath = ".\\files\\stacking_model.onnx";
            var thongSoArray = new float[222];
            for (int i = 0; i < 222; i++)
            {            
                thongSoArray[i] = fix[i];           
            }
            thongSoArray[7] = inputND.Thongso_1;
            thongSoArray[28] = inputND.Thongso_2;
            thongSoArray[42] = inputND.Thongso_3;
            thongSoArray[43] = inputND.Thongso_4;
            thongSoArray[31] = inputND.Thongso_5;
            thongSoArray[9] = inputND.Thongso_6;
            thongSoArray[10] = inputND.Thongso_7;
            thongSoArray[14] = inputND.Thongso_8;
            thongSoArray[22] = inputND.Thongso_9;
            thongSoArray[12] = inputND.Thongso_10;
            var inputTensor = new DenseTensor<float>(thongSoArray, new int[] { 1, 222 });
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("float_input", inputTensor) };
            using var session = new InferenceSession(onnxPath);
            var output = session.Run(input);
            var result = output.ToArray()[0].AsTensor<Int64>()[0];
            return result switch
            {
                0 => "0$-125000$",
                1 => "125000$-150000$",
                2 => "150000$-180000$",
                3 => "180000$-230000$",
                _ => "Trên 230000$",
            };
        }
        
    }
}