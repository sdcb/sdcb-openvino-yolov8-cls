using OpenCvSharp;
using Sdcb.OpenVINO.Natives;
using Sdcb.OpenVINO;
using System.Xml.Linq;
using System.Xml.XPath;
using System.Diagnostics;

public static class Program
{
    unsafe static void Main()
    {
        string modelFile = @"./model/yolov8n-cls.xml";
        string[] dicts = XDocument.Load(modelFile)
            .XPathSelectElement(@"/net/rt_info/model_info/labels")!.Attribute("value")!.Value
            .Split(' ');

        using Model rawModel = OVCore.Shared.ReadModel(modelFile);
        using PrePostProcessor pp = rawModel.CreatePrePostProcessor();
        using (PreProcessInputInfo inputInfo = pp.Inputs.Primary)
        {
            inputInfo.TensorInfo.Layout = Layout.NHWC;
            inputInfo.ModelInfo.Layout = Layout.NCHW;
        }
        using Model m = pp.BuildModel();
        using CompiledModel cm = OVCore.Shared.CompileModel(m, "CPU");
        using InferRequest ir = cm.CreateInferRequest();

        Shape inputShape = m.Inputs.Primary.Shape;

        using Mat src = Cv2.ImRead(@"hen.jpg");
        Stopwatch stopwatch = new();
        using Mat resized = src.Resize(new Size(inputShape[2], inputShape[1]));
        using Mat f32 = new();
        resized.ConvertTo(f32, MatType.CV_32FC3, 1.0 / 255);

        using (Tensor input = Tensor.FromRaw(
            new ReadOnlySpan<byte>((void*)f32.Data, (int)((nint)f32.DataEnd - f32.DataStart)), 
            new Shape(1, f32.Rows, f32.Cols, 3), 
            ov_element_type_e.F32))
        {
            ir.Inputs.Primary = input;
        }
        double preprocessTime = stopwatch.Elapsed.TotalMilliseconds;
        stopwatch.Restart();

        ir.Run();
        double inferTime = stopwatch.Elapsed.TotalMilliseconds;
        stopwatch.Restart();

        using (Tensor output = ir.Outputs.Primary)
        {
            ReadOnlySpan<float> data = output.GetData<float>();
            int maxIndex = MaxIndexOfSpan(data);
            double postProcessTime = stopwatch.Elapsed.TotalMilliseconds;
            stopwatch.Stop();

            Console.WriteLine($"class id={dicts[maxIndex]}, score={data[maxIndex]:F2}");
            double totalTime = preprocessTime + inferTime + postProcessTime;
            Console.WriteLine($"preprocess time: {preprocessTime:F2}ms");
            Console.WriteLine($"infer time: {inferTime:F2}ms");
            Console.WriteLine($"postprocess time: {postProcessTime:F2}ms");
            Console.WriteLine($"Total time: {totalTime:F2}ms");
        }
    }

    static int MaxIndexOfSpan(ReadOnlySpan<float> data)
    {
        // 参数校验
        if (data == null || data.Length == 0)
            throw new ArgumentException("The provided data span is null or empty.");

        // 初始化最大值及其索引
        int maxIndex = 0;
        float maxValue = data[0];

        // 遍历跨度查找最大值及其索引
        for (int i = 1; i < data.Length; i++)
        {
            if (data[i] > maxValue)
            {
                maxValue = data[i];
                maxIndex = i;
            }
        }

        // 返回最大值及其索引
        return maxIndex;
    }
}