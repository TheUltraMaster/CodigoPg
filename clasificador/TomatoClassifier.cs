using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Numerics.Tensors;

namespace DotnetCoreYolo.Clasificador
{
    public class TomatoClassifier : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string[] _classNames;
        private bool _disposed = false;

        public TomatoClassifier(string modelPath)
        {
            if (!File.Exists(modelPath))
            {
                throw new FileNotFoundException($"Modelo no encontrado: {modelPath}");
            }

            var options = new SessionOptions()
            {
                EnableCpuMemArena = false,
                EnableMemoryPattern = false,
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };

            _session = new InferenceSession(modelPath, options);

            // Clases comunes de clasificación de tomates (personalizar según tu modelo)
            _classNames = new string[]
            {
            "Tomato_Bacterial_spot",
            "Tomato_Early_blight", 
            "Tomato_healthy",
            "Tomato_Late_blight",
            "Tomato_Leaf_Mold",
            "Tomato_mosaic_virus",
            "Tomato_Septoria_leaf_spot",
            "Tomato_Spider_mites Two-spotted_spider_mite",
            "Tomato_Target_Spot",
            "Tomato_Yellow_Leaf_Curl_Virus"
            };
        }

        public ClassificationResult ClassifyImage(string imagePath)
        {
            if (!File.Exists(imagePath))
            {
                throw new FileNotFoundException($"Imagen no encontrada: {imagePath}");
            }

            using var image = Image.Load<Rgb24>(imagePath);
            return ClassifyImageInternal(image, imagePath);
        }
        
        public ClassificationResult ClassifyImageFromBytes(byte[] imageBytes, string imageName = "unknown")
        {
            if (imageBytes == null || imageBytes.Length == 0)
            {
                throw new ArgumentException("Array de bytes de imagen vacío o nulo");
            }

            using var memoryStream = new MemoryStream(imageBytes);
            using var image = Image.Load<Rgb24>(memoryStream);
            return ClassifyImageInternal(image, imageName);
        }
        
        public List<ClassificationResult> ClassifyImageBatch(List<byte[]> imageBytesList)
        {
            var results = new List<ClassificationResult>();
            
            // Preparar todos los tensores en una sola pasada
            var tensors = new List<DenseTensor<float>>();
            var validIndices = new List<int>();
            
            for (int i = 0; i < imageBytesList.Count; i++)
            {
                try
                {
                    var imageBytes = imageBytesList[i];
                    if (imageBytes != null && imageBytes.Length > 0)
                    {
                        using var memoryStream = new MemoryStream(imageBytes);
                        using var image = Image.Load<Rgb24>(memoryStream);
                        
                        var inputSize = 224;
                        image.Mutate(x => x.Resize(inputSize, inputSize));
                        var tensor = ImageToTensor(image, inputSize);
                        tensors.Add(tensor);
                        validIndices.Add(i);
                    }
                    else
                    {
                        // Añadir resultado vacío para mantener índices
                        results.Add(new ClassificationResult
                        {
                            ImagePath = $"batch_image_{i}",
                            PredictedClass = "Error",
                            ClassIndex = -1,
                            Confidence = 0.0f,
                            AllProbabilities = new ClassProbability[0]
                        });
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ Error procesando imagen {i} en batch: {ex.Message}");
                    results.Add(new ClassificationResult
                    {
                        ImagePath = $"batch_image_{i}",
                        PredictedClass = "Error",
                        ClassIndex = -1,
                        Confidence = 0.0f,
                        AllProbabilities = new ClassProbability[0]
                    });
                }
            }
            
            // Procesar todas las imágenes válidas en paralelo
            var batchResults = ProcessBatchTensors(tensors);
            
            // Combinar resultados manteniendo el orden original
            var batchIndex = 0;
            for (int i = 0; i < imageBytesList.Count; i++)
            {
                if (validIndices.Contains(i) && batchIndex < batchResults.Count)
                {
                    var result = batchResults[batchIndex];
                    result.ImagePath = $"batch_image_{i}";
                    
                    // Insertar en la posición correcta si ya hay resultados de error
                    if (i < results.Count)
                    {
                        results[i] = result;
                    }
                    else
                    {
                        results.Add(result);
                    }
                    batchIndex++;
                }
            }
            
            return results;
        }
        
        private List<ClassificationResult> ProcessBatchTensors(List<DenseTensor<float>> tensors)
        {
            var results = new List<ClassificationResult>();
            
            foreach (var tensor in tensors)
            {
                try
                {
                    var inputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("input", tensor)
                    };

                    using var sessionResults = _session.Run(inputs);
                    var output = sessionResults.First().AsTensor<float>();

                    var probabilities = SoftMax(output.ToArray());
                    var maxIndex = Array.IndexOf(probabilities, probabilities.Max());
                    var predictedClassName = maxIndex < _classNames.Length ? _classNames[maxIndex] : $"Clase {maxIndex}";
                    
                    var classProbabilities = new List<ClassProbability>();
                    for (int i = 0; i < probabilities.Length; i++)
                    {
                        var className = i < _classNames.Length ? _classNames[i] : $"Clase {i}";
                        classProbabilities.Add(new ClassProbability
                        {
                            ClassName = className,
                            Probability = probabilities[i]
                        });
                    }

                    results.Add(new ClassificationResult
                    {
                        ImagePath = "batch_processed",
                        PredictedClass = predictedClassName,
                        ClassIndex = maxIndex,
                        Confidence = probabilities[maxIndex],
                        AllProbabilities = classProbabilities.ToArray()
                    });
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ Error en clasificación individual del batch: {ex.Message}");
                    results.Add(new ClassificationResult
                    {
                        ImagePath = "batch_error",
                        PredictedClass = "Error",
                        ClassIndex = -1,
                        Confidence = 0.0f,
                        AllProbabilities = new ClassProbability[0]
                    });
                }
            }
            
            return results;
        }
        
        private ClassificationResult ClassifyImageInternal(Image<Rgb24> image, string imagePath)
        {
            // Redimensionar imagen para el modelo (normalmente 224x224 o 299x299)
            var inputSize = 224; // Ajustar según las especificaciones del modelo
            image.Mutate(x => x.Resize(inputSize, inputSize));

            // Convertir imagen a tensor
            var tensor = ImageToTensor(image, inputSize);

            // Ejecutar inferencia
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };

            using var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>();

            // Procesar resultados
            var probabilities = SoftMax(output.ToArray());
            var maxIndex = Array.IndexOf(probabilities, probabilities.Max());

            // Verificar que el índice esté dentro de los límites
            var predictedClassName = maxIndex < _classNames.Length ? _classNames[maxIndex] : $"Clase {maxIndex}";
            
            // Crear lista de probabilidades solo para las clases disponibles
            var classProbabilities = new List<ClassProbability>();
            for (int i = 0; i < probabilities.Length; i++)
            {
                var className = i < _classNames.Length ? _classNames[i] : $"Clase {i}";
                classProbabilities.Add(new ClassProbability
                {
                    ClassName = className,
                    Probability = probabilities[i]
                });
            }

            return new ClassificationResult
            {
                ImagePath = imagePath,
                PredictedClass = predictedClassName,
                ClassIndex = maxIndex,
                Confidence = probabilities[maxIndex],
                AllProbabilities = classProbabilities.ToArray()
            };
        }

        private DenseTensor<float> ImageToTensor(Image<Rgb24> image, int size)
        {
            var tensor = new DenseTensor<float>(new[] { 1, 3, size, size });

            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    var pixel = image[x, y];
                    tensor[0, 0, y, x] = (pixel.R / 255.0f - 0.485f) / 0.229f; // Red
                    tensor[0, 1, y, x] = (pixel.G / 255.0f - 0.456f) / 0.224f; // Green  
                    tensor[0, 2, y, x] = (pixel.B / 255.0f - 0.406f) / 0.225f; // Blue
                }
            }

            return tensor;
        }

        private float[] SoftMax(float[] values)
        {
            var max = values.Max();
            var exp = values.Select(v => Math.Exp(v - max)).ToArray();
            var sum = exp.Sum();
            return exp.Select(e => (float)(e / sum)).ToArray();
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _session?.Dispose();
                _disposed = true;
            }
        }
    }

    public class ClassificationResult
    {
        public string ImagePath { get; set; } = string.Empty;
        public string PredictedClass { get; set; } = string.Empty;
        public int ClassIndex { get; set; }
        public float Confidence { get; set; }
        public ClassProbability[] AllProbabilities { get; set; } = Array.Empty<ClassProbability>();
    }

    public class ClassProbability
    {
        public string ClassName { get; set; } = string.Empty;
        public float Probability { get; set; }
    }
}