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
        private string DuDoan (ThongSo inputND)
        {
            string onnxPath = ".\\files\\rf_model.onnx";
            var inputTensor = new DenseTensor<float>(new float[] { inputND.Thongso_1, inputND.Thongso_2,
                inputND.Thongso_3, inputND.Thongso_4+inputND.Thongso_9, inputND.Thongso_5, inputND.Thongso_6, inputND.Thongso_7,
                inputND.Thongso_8, inputND.Thongso_9, inputND.Thongso_10 }, new int[] {1, 10});
            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<float>("float_input", inputTensor) };
            using var session = new InferenceSession(onnxPath);
            var output = session.Run(input);
            var result = output.ToArray()[0].AsTensor<Int64>()[0];
            if (result == 0)
            {
                return("dưới 125000$");
            }
            if (result == 1)
            {
                return("125000$-150000$");
            }
            if (result == 2)
            {
                return("150000$-180000$");
            }
            if (result == 3)
            {
                return ("180000$-230000$");
            }  
            return ("Trên 230000$");
        }
    }
}