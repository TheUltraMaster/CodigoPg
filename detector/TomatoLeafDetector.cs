using System;
using System.Collections.Generic;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace DotnetCoreYolo.Detector
{
    public class TomatoLeafDetector : IDisposable
    {
        private readonly YoloDetector _yoloDetector;
        
        public string ModelPath => _yoloDetector?.ModelPath ?? string.Empty;
        
        public TomatoLeafDetector(string modelPath, int imageSize = 640, string[] classNames = null)
        {
            _yoloDetector = new YoloDetector(modelPath, imageSize, classNames);
        }
        
        public DetectionResult DetectLeaves(string imagePath, bool saveResult = true, float confidenceThreshold = 0.01f, float iouThreshold = 0.75f, int maxDetections = 1000)
        {
            var result = new DetectionResult { ImagePath = imagePath };
            
            try
            {
                if (!File.Exists(imagePath))
                {
                    result.MarkFailed($"Archivo de imagen no encontrado: {imagePath}");
                    return result;
                }
                
                using var image = Image.Load<Rgb24>(imagePath);
                result.ImageWidth = image.Width;
                result.ImageHeight = image.Height;
                result.OriginalWidth = image.Width;
                result.OriginalHeight = image.Height;
                result.ConfidenceThreshold = confidenceThreshold;
                result.IouThreshold = iouThreshold;
                result.MaxDetections = maxDetections;
                
                var detections = _yoloDetector.Detect(image, confidenceThreshold, iouThreshold, maxDetections);
                
                // Calculate area percentages for all detections
                foreach (var detection in detections)
                {
                    detection.CalculateAreaPercentage(image.Width, image.Height);
                }
                
                result.Detections = detections;
                result.MarkCompleted();
                
                return result;
            }
            catch (Exception ex)
            {
                result.MarkFailed($"Error detecting leaves: {ex.Message}");
                return result;
            }
        }
        
        public DetectionResult DetectIndividualLeaves(string imagePath, bool saveResult = true, float confidenceThreshold = 0.01f, float iouThreshold = 0.75f, int maxDetections = 1000)
        {
            return DetectLeaves(imagePath, saveResult, confidenceThreshold, iouThreshold, maxDetections);
        }
        
        public DetectionResult DetectPreciseLeaves(string imagePath, bool saveResult = true, float confidenceThreshold = 0.01f, float iouThreshold = 0.75f, int maxDetections = 500)
        {
            return DetectLeaves(imagePath, saveResult, confidenceThreshold, iouThreshold, maxDetections);
        }
        
        public DetectionResult DetectFromBytes(byte[] imageBytes, string imageName = "image", bool saveResult = true, float confidenceThreshold = 0.01f, float iouThreshold = 0.75f, int maxDetections = 1000)
        {
            var result = new DetectionResult { ImagePath = imageName };
            
            try
            {
                using var image = Image.Load<Rgb24>(imageBytes);
                result.ImageWidth = image.Width;
                result.ImageHeight = image.Height;
                result.OriginalWidth = image.Width;
                result.OriginalHeight = image.Height;
                result.ConfidenceThreshold = confidenceThreshold;
                result.IouThreshold = iouThreshold;
                result.MaxDetections = maxDetections;
                
                var detections = _yoloDetector.Detect(image, confidenceThreshold, iouThreshold, maxDetections);
                
                // Calculate area percentages for all detections
                foreach (var detection in detections)
                {
                    detection.CalculateAreaPercentage(image.Width, image.Height);
                }
                
                result.Detections = detections;
                result.MarkCompleted();
                
                return result;
            }
            catch (Exception ex)
            {
                result.MarkFailed($"Error detecting leaves from bytes: {ex.Message}");
                return result;
            }
        }
        
        public void Dispose()
        {
            _yoloDetector?.Dispose();
        }
    }
}