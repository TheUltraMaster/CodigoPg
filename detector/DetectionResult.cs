using System;
using System.Collections.Generic;

namespace DotnetCoreYolo.Detector
{
    public class DetectionResult
    {
        public List<Detection> Detections { get; set; } = new List<Detection>();
        public string ImagePath { get; set; } = string.Empty;
        public int ImageWidth { get; set; }
        public int ImageHeight { get; set; }
        public int OriginalWidth { get; set; }
        public int OriginalHeight { get; set; }
        public int NumLeaves => DetectionCount;
        public DateTime ProcessingStartTime { get; set; }
        public DateTime ProcessingEndTime { get; set; }
        public TimeSpan ProcessingTime => ProcessingEndTime - ProcessingStartTime;
        public double ProcessingTimeMs => ProcessingTime.TotalMilliseconds;
        public bool Success { get; set; } = true;
        public string ErrorMessage { get; set; } = string.Empty;
        public float ConfidenceThreshold { get; set; }
        public float IouThreshold { get; set; }
        public int MaxDetections { get; set; }
        
        public int DetectionCount => Detections?.Count ?? 0;
        
        public DetectionResult()
        {
            ProcessingStartTime = DateTime.Now;
        }
        
        public DetectionResult(List<Detection> detections, string imagePath = "") : this()
        {
            Detections = detections ?? new List<Detection>();
            ImagePath = imagePath;
            ProcessingEndTime = DateTime.Now;
        }
        
        public void MarkCompleted()
        {
            ProcessingEndTime = DateTime.Now;
        }
        
        public void MarkFailed(string errorMessage)
        {
            Success = false;
            ErrorMessage = errorMessage;
            ProcessingEndTime = DateTime.Now;
        }
    }
}