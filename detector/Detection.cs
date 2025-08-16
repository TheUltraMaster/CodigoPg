
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace DotnetCoreYolo.Detector
{
    public class Detection
    {
        public int Id { get; set; }
        public int[] BBox { get; set; } = new int[4]; // [x1, y1, x2, y2]
        public int[] Center { get; set; } = new int[2]; // [x, y]
        public float Confidence { get; set; }
        public int ClassId { get; set; }
        public string ClassName { get; set; } = string.Empty;
        public int Area { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public float AreaPercentage { get; set; }
        public float AspectRatio { get; set; }
        public float SheetLuminosity { get; set; }
        public int[] CenterPixelColor { get; set; } = new int[3]; // [R, G, B]
        
        public Detection() { }
        
        public Detection(float[] bbox, float confidence, int classId, string className, int id = 0)
        {
            Id = id;
            BBox = [(int)bbox[0], (int)bbox[1], (int)bbox[2], (int)bbox[3]];
            Confidence = confidence;
            ClassId = classId;
            ClassName = className ?? string.Empty;
            
            // Calcular centro desde la caja delimitadora
            Center = 
            [
                (BBox[0] + BBox[2]) / 2, 
                (BBox[1] + BBox[3]) / 2 
            ];
            
            // Calcular dimensiones
            Width = BBox[2] - BBox[0];
            Height = BBox[3] - BBox[1];
            Area = Width * Height;
            AspectRatio = Height > 0 ? (float)Width / Height : 0;
            
            // Inicializar propiedades nuevas con valores por defecto
            SheetLuminosity = 0.0f;
            CenterPixelColor = [0, 0, 0];
        }
        
        public void CalculateAreaPercentage(int imageWidth, int imageHeight)
        {
            var totalImageArea = imageWidth * imageHeight;
            AreaPercentage = totalImageArea > 0 ? (float)Area / totalImageArea * 100 : 0;
        }

        public void ExtractPixelData<T>(Image<T> image) where T : unmanaged, IPixel<T>
        {
            if (Center == null || Center.Length < 2) return;
            
            var centerX = Math.Max(0, Math.Min(image.Width - 1, Center[0]));
            var centerY = Math.Max(0, Math.Min(image.Height - 1, Center[1]));
            
            var pixel = image[centerX, centerY];
            var rgba = new Rgba32();
            pixel.ToRgba32(ref rgba);
            
            // Obtener valores RGB del pixel central
            CenterPixelColor[0] = rgba.R;
            CenterPixelColor[1] = rgba.G;
            CenterPixelColor[2] = rgba.B;
        }
    }
}