using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DotnetCoreYolo.Detector
{
    public class YoloDetector : IDisposable
    {
        private InferenceSession _session;
        private readonly string _modelPath;
        private readonly int _imageSize;
        private readonly string[] _classNames;
        
        public string ModelPath => _modelPath;

        public YoloDetector(string modelPath, int imageSize = 640, string[] classNames = null)
        {
            _modelPath = modelPath;
            _imageSize = imageSize;
            _classNames = classNames ?? new[] { "tomato_leaf" };
            LoadModel();
        }

        private void LoadModel()
        {
            try
            {
                var sessionOptions = new SessionOptions();
                sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                
                try
                {
                    sessionOptions.AppendExecutionProvider_CUDA(0);
                    Console.WriteLine("✅ GPU CUDA detectada, usando aceleración GPU");
                }
                catch
                {
                    Console.WriteLine("⚠️ GPU no disponible, usando CPU");
                }
                
                _session = new InferenceSession(_modelPath, sessionOptions);
                Console.WriteLine($"✅ Modelo cargado desde: {_modelPath}");
                
                PrintModelInfo();
            }
            catch (Exception ex)
            {
                throw new Exception($"❌ Error cargando modelo: {ex.Message}", ex);
            }
        }

        private void PrintModelInfo()
        {
            Console.WriteLine("\n📊 Información del modelo:");
            Console.WriteLine("Entradas:");
            foreach (var input in _session.InputMetadata)
            {
                Console.WriteLine($"  - {input.Key}: {string.Join("x", input.Value.Dimensions)}");
            }
            Console.WriteLine("Salidas:");
            foreach (var output in _session.OutputMetadata)
            {
                Console.WriteLine($"  - {output.Key}: {string.Join("x", output.Value.Dimensions)}");
            }
        }

        public List<Detection> Detect(Image<Rgb24> image, float confidenceThreshold = 0.05f, float iouThreshold = 0.45f, int maxDetections = 50)
        {
            var inputTensor = PreprocessImage(image);
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", inputTensor)
            };
            
            using var results = _session.Run(inputs);
            
            var detections = ProcessModelOutput(results, image.Width, image.Height, confidenceThreshold, iouThreshold, maxDetections);
            
            // Extraer datos de píxeles para cada detección
            foreach (var detection in detections)
            {
                detection.ExtractPixelData(image);
            }
            
            return detections;
        }

        public List<Detection> DetectFromFile(string imagePath, float confidenceThreshold = 0.05f, float iouThreshold = 0.45f, int maxDetections = 50)
        {
            if (!File.Exists(imagePath))
            {
                Console.WriteLine($"❌ Imagen no encontrada: {imagePath}");
                return new List<Detection>();
            }

            using var image = Image.Load<Rgb24>(imagePath);
            return Detect(image, confidenceThreshold, iouThreshold, maxDetections);
        }

        private DenseTensor<float> PreprocessImage(Image<Rgb24> image)
        {
            var originalImage = image.Clone();
            originalImage.Mutate(x => x.Resize(_imageSize, _imageSize));
            
            var inputTensor = new DenseTensor<float>(new[] { 1, 3, _imageSize, _imageSize });
            
            for (int y = 0; y < _imageSize; y++)
            {
                for (int x = 0; x < _imageSize; x++)
                {
                    var pixel = originalImage[x, y];
                    
                    inputTensor[0, 0, y, x] = pixel.R / 255.0f;
                    inputTensor[0, 1, y, x] = pixel.G / 255.0f;
                    inputTensor[0, 2, y, x] = pixel.B / 255.0f;
                }
            }
            
            originalImage.Dispose();
            return inputTensor;
        }

        private List<Detection> ProcessModelOutput(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results, 
            int originalWidth, int originalHeight, float confidenceThreshold, float iouThreshold, int maxDetections)
        {
            var detections = new List<Detection>();
            
            var output = results.First().AsTensor<float>();
            var shape = output.Dimensions.ToArray();
            
            Console.WriteLine($"🔍 Forma del tensor de salida: [{string.Join(", ", shape)}]");
            Console.WriteLine($"🎯 Umbral de confianza: {confidenceThreshold}");
            Console.WriteLine($"📏 Dimensiones originales: {originalWidth}x{originalHeight}");
            
            if (shape.Length == 3 && shape[0] == 1)
            {
                int numBoxes = shape[2];
                int numClasses = shape[1] - 4;
                
                Console.WriteLine($"📦 Número de cajas: {numBoxes}");
                Console.WriteLine($"🏷️ Número de clases: {numClasses}");
                
                for (int i = 0; i < numBoxes; i++)
                {
                    float cx = output[0, 0, i] * originalWidth / _imageSize;
                    float cy = output[0, 1, i] * originalHeight / _imageSize;
                    float w = output[0, 2, i] * originalWidth / _imageSize;
                    float h = output[0, 3, i] * originalHeight / _imageSize;
                    
                    float x1 = cx - w / 2;
                    float y1 = cy - h / 2;
                    float x2 = cx + w / 2;
                    float y2 = cy + h / 2;
                    
                    float maxConf = 0;
                    int classId = 0;
                    
                    for (int c = 0; c < numClasses; c++)
                    {
                        float conf = output[0, 4 + c, i];
                        if (conf > maxConf)
                        {
                            maxConf = conf;
                            classId = c;
                        }
                    }
                    
                    if (maxConf >= confidenceThreshold)
                    {
                        Console.WriteLine($"✅ Detección válida {i}: conf={maxConf:F3}, bbox=[{x1:F1},{y1:F1},{x2:F1},{y2:F1}], clase={classId}");
                        detections.Add(new Detection
                        {
                            Id = detections.Count + 1,
                            BBox = new[] { (int)x1, (int)y1, (int)x2, (int)y2 },
                            Confidence = maxConf,
                            ClassId = classId,
                            ClassName = classId < _classNames.Length ? _classNames[classId] : "unknown",
                            Area = (int)(w * h),
                            Width = (int)w,
                            Height = (int)h,
                            Center = new[] { (int)cx, (int)cy }
                        });
                    }
                    else if (i < 10) // Log primeras 10 detecciones rechazadas
                    {
                        Console.WriteLine($"❌ Detección {i} rechazada: conf={maxConf:F3} < {confidenceThreshold}");
                    }
                }
            }
            else
            {
                Console.WriteLine($"❌ Formato de salida inesperado! Esperaba [1, classes+4, boxes] pero obtuve [{string.Join(", ", shape)}]");
                
                // Intentar otros formatos comunes de YOLO
                if (shape.Length == 2)
                {
                    Console.WriteLine($"🔄 Probando formato YOLOv5/v8: [batch, predictions]");
                    return ProcessYolov5Output(output, shape, originalWidth, originalHeight, confidenceThreshold, iouThreshold, maxDetections);
                }
                else if (shape.Length == 3 && shape[1] != shape[2])
                {
                    Console.WriteLine($"🔄 Probando formato alternativo: [batch, boxes, attributes]");
                    return ProcessAlternativeOutput(output, shape, originalWidth, originalHeight, confidenceThreshold, iouThreshold, maxDetections);
                }
                
                return detections;
            }
            
            Console.WriteLine($"🔍 Detecciones antes de NMS: {detections.Count}");
            detections = ApplyNMS(detections, iouThreshold);
            Console.WriteLine($"🔍 Detecciones después de NMS: {detections.Count}");
            
            if (detections.Count > maxDetections)
            {
                detections = detections.OrderByDescending(d => d.Confidence).Take(maxDetections).ToList();
                Console.WriteLine($"🔍 Detecciones después de límite: {detections.Count}");
            }
            
            for (int i = 0; i < detections.Count; i++)
            {
                detections[i].Id = i + 1;
            }
            
            Console.WriteLine($"🎯 Total de detecciones finales: {detections.Count}");
            return detections;
        }

        private List<Detection> ApplyNMS(List<Detection> detections, float iouThreshold)
        {
            var kept = new List<Detection>();
            var sorted = detections.OrderByDescending(d => d.Confidence).ToList();
            
            for (int i = 0; i < sorted.Count; i++)
            {
                var current = sorted[i];
                bool shouldKeep = true;
                
                foreach (var keptDetection in kept)
                {
                    if (CalculateIoU(current.BBox, keptDetection.BBox) >= iouThreshold)
                    {
                        shouldKeep = false;
                        break;
                    }
                }
                
                if (shouldKeep)
                {
                    kept.Add(current);
                }
            }
            
            return kept;
        }

        private float CalculateIoU(int[] box1, int[] box2)
        {
            float x1 = Math.Max(box1[0], box2[0]);
            float y1 = Math.Max(box1[1], box2[1]);
            float x2 = Math.Min(box1[2], box2[2]);
            float y2 = Math.Min(box1[3], box2[3]);
            
            if (x2 <= x1 || y2 <= y1) return 0;
            
            float intersection = (x2 - x1) * (y2 - y1);
            float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
            float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
            float union = area1 + area2 - intersection;
            
            return intersection / union;
        }

        public List<Detection> FilterIndividualLeaves(List<Detection> detections, int imageArea)
        {
            return detections.Where(d =>
            {
                d.AreaPercentage = (d.Area / (float)imageArea) * 100;
                d.AspectRatio = d.Width / (float)d.Height;
                
                return d.AreaPercentage < 95 &&
                       d.AreaPercentage > 0.05 &&
                       d.Area > 50 &&
                       d.Width > 5 &&
                       d.Height > 5;
            }).ToList();
        }

        public List<Detection> FilterPreciseLeaves(List<Detection> detections, int imageArea, float maxAreaPercent = 95.0f)
        {
            return detections.Where(d =>
            {
                d.AreaPercentage = (d.Area / (float)imageArea) * 100;
                d.AspectRatio = d.Width / (float)d.Height;
                
                return d.AreaPercentage <= maxAreaPercent &&
                       d.AreaPercentage >= 0.03 &&
                       d.Area >= 50 &&
                       d.AspectRatio >= 0.1 &&
                       d.AspectRatio <= 10.0 &&
                       d.Width >= 3 &&
                       d.Height >= 3;
            }).ToList();
        }

        public List<Detection> RemoveOverlappingDetections(List<Detection> detections, float overlapThreshold)
        {
            var filtered = new List<Detection>();
            var sorted = detections.OrderByDescending(d => d.Confidence).ToList();
            
            foreach (var detection in sorted)
            {
                bool overlaps = false;
                foreach (var accepted in filtered)
                {
                    if (CalculateIoU(detection.BBox, accepted.BBox) > overlapThreshold)
                    {
                        overlaps = true;
                        break;
                    }
                }
                
                if (!overlaps)
                {
                    filtered.Add(detection);
                }
            }
            
            for (int i = 0; i < filtered.Count; i++)
            {
                filtered[i].Id = i + 1;
            }
            
            return filtered;
        }

        private List<Detection> ProcessYolov5Output(Tensor<float> output, int[] shape, 
            int originalWidth, int originalHeight, float confidenceThreshold, float iouThreshold, int maxDetections)
        {
            var detections = new List<Detection>();
            
            // YOLOv5/v8 format: [batch, predictions] where predictions = [x, y, w, h, conf, class_probs...]
            int batchSize = shape[0];
            int predictions = shape[1];
            
            Console.WriteLine($"🔄 Procesando formato YOLOv5/v8: batch={batchSize}, predictions={predictions}");
            
            // Calcular número de clases asumiendo formato [x, y, w, h, obj_conf, class1, class2, ...]
            int numClasses = predictions - 5; // 4 coords + 1 objectness confidence
            
            if (numClasses <= 0)
            {
                Console.WriteLine($"❌ Número de clases inválido: {numClasses}");
                return detections;
            }
            
            Console.WriteLine($"🏷️ Número de clases detectadas: {numClasses}");
            
            for (int i = 0; i < batchSize; i++)
            {
                float cx = output[i, 0] * originalWidth / _imageSize;
                float cy = output[i, 1] * originalHeight / _imageSize;
                float w = output[i, 2] * originalWidth / _imageSize;
                float h = output[i, 3] * originalHeight / _imageSize;
                float objectness = output[i, 4];
                
                // Encontrar la clase con mayor probabilidad
                float maxClassProb = 0;
                int classId = 0;
                
                for (int c = 0; c < numClasses; c++)
                {
                    float classProb = output[i, 5 + c];
                    if (classProb > maxClassProb)
                    {
                        maxClassProb = classProb;
                        classId = c;
                    }
                }
                
                float confidence = objectness * maxClassProb;
                
                if (confidence >= confidenceThreshold)
                {
                    float x1 = cx - w / 2;
                    float y1 = cy - h / 2;
                    float x2 = cx + w / 2;
                    float y2 = cy + h / 2;
                    
                    Console.WriteLine($"✅ Detección YOLOv5 {i}: conf={confidence:F3}, bbox=[{x1:F1},{y1:F1},{x2:F1},{y2:F1}], clase={classId}");
                    
                    detections.Add(new Detection
                    {
                        Id = detections.Count + 1,
                        BBox = new[] { (int)x1, (int)y1, (int)x2, (int)y2 },
                        Confidence = confidence,
                        ClassId = classId,
                        ClassName = classId < _classNames.Length ? _classNames[classId] : "unknown",
                        Area = (int)(w * h),
                        Width = (int)w,
                        Height = (int)h,
                        Center = new[] { (int)cx, (int)cy }
                    });
                }
                else if (i < 10)
                {
                    Console.WriteLine($"❌ Detección YOLOv5 {i} rechazada: conf={confidence:F3} < {confidenceThreshold}");
                }
            }
            
            Console.WriteLine($"🔍 Detecciones YOLOv5 antes de NMS: {detections.Count}");
            detections = ApplyNMS(detections, iouThreshold);
            Console.WriteLine($"🔍 Detecciones YOLOv5 después de NMS: {detections.Count}");
            
            if (detections.Count > maxDetections)
            {
                detections = detections.OrderByDescending(d => d.Confidence).Take(maxDetections).ToList();
            }
            
            for (int i = 0; i < detections.Count; i++)
            {
                detections[i].Id = i + 1;
            }
            
            return detections;
        }

        private List<Detection> ProcessAlternativeOutput(Tensor<float> output, int[] shape,
            int originalWidth, int originalHeight, float confidenceThreshold, float iouThreshold, int maxDetections)
        {
            var detections = new List<Detection>();
            
            Console.WriteLine($"🔄 Procesando formato alternativo: [{string.Join(", ", shape)}]");
            
            // Intentar formato [1, boxes, attributes] donde attributes podría ser [x1,y1,x2,y2,conf,class] o similar
            if (shape.Length == 3 && shape[0] == 1)
            {
                int numBoxes = shape[1];
                int numAttributes = shape[2];
                
                Console.WriteLine($"📦 Cajas: {numBoxes}, Atributos por caja: {numAttributes}");
                
                if (numAttributes >= 6) // Mínimo: x1,y1,x2,y2,conf,class
                {
                    for (int i = 0; i < numBoxes; i++)
                    {
                        float x1 = output[0, i, 0] * originalWidth / _imageSize;
                        float y1 = output[0, i, 1] * originalHeight / _imageSize;
                        float x2 = output[0, i, 2] * originalWidth / _imageSize;
                        float y2 = output[0, i, 3] * originalHeight / _imageSize;
                        float confidence = output[0, i, 4];
                        int classId = (int)output[0, i, 5];
                        
                        if (confidence >= confidenceThreshold)
                        {
                            Console.WriteLine($"✅ Detección alternativa {i}: conf={confidence:F3}, bbox=[{x1:F1},{y1:F1},{x2:F1},{y2:F1}], clase={classId}");
                            
                            float w = x2 - x1;
                            float h = y2 - y1;
                            float cx = x1 + w / 2;
                            float cy = y1 + h / 2;
                            
                            detections.Add(new Detection
                            {
                                Id = detections.Count + 1,
                                BBox = new[] { (int)x1, (int)y1, (int)x2, (int)y2 },
                                Confidence = confidence,
                                ClassId = classId,
                                ClassName = classId < _classNames.Length ? _classNames[classId] : "unknown",
                                Area = (int)(w * h),
                                Width = (int)w,
                                Height = (int)h,
                                Center = new[] { (int)cx, (int)cy }
                            });
                        }
                        else if (i < 10)
                        {
                            Console.WriteLine($"❌ Detección alternativa {i} rechazada: conf={confidence:F3} < {confidenceThreshold}");
                        }
                    }
                }
            }
            
            detections = ApplyNMS(detections, iouThreshold);
            
            if (detections.Count > maxDetections)
            {
                detections = detections.OrderByDescending(d => d.Confidence).Take(maxDetections).ToList();
            }
            
            for (int i = 0; i < detections.Count; i++)
            {
                detections[i].Id = i + 1;
            }
            
            return detections;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}