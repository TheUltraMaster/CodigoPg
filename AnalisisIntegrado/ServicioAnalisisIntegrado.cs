using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.Fonts;
using SixLabors.ImageSharp.Drawing;
using DotnetCoreYolo.Detector;
using DotnetCoreYolo.Clasificador;

namespace DotnetCoreYolo.AnalisisIntegrado
{
    public class ServicioAnalisisIntegrado : IDisposable
    {
        private TomatoLeafDetector? _detector;
        private TomatoClassifier? _clasificador;
        private bool _disposed = false;

        public ResultadoAnalisisIntegrado ProcesarImagen(string rutaImagen, OpcionesAnalisisIntegrado opciones)
        {
            return ProcesarImagenAsync(rutaImagen, opciones).GetAwaiter().GetResult();
        }

        public async Task<ResultadoAnalisisIntegrado> ProcesarImagenAsync(string rutaImagen, OpcionesAnalisisIntegrado opciones)
        {
            var cronometro = Stopwatch.StartNew();
            var resultado = new ResultadoAnalisisIntegrado
            {
                RutaImagen = rutaImagen,
                HoraInicio = DateTime.Now,
                Exitoso = false
            };

            try
            {
                if (!File.Exists(rutaImagen))
                {
                    resultado.MensajeError = $"Imagen no encontrada: {rutaImagen}";
                    resultado.HoraFin = DateTime.Now;
                    return resultado;
                }

                InicializarServicios(opciones);

                // Fase 1: Detección de hojas
                DetectionResult resultadoDeteccion = null;
                if (opciones.HabilitarDeteccion && _detector != null)
                {
                    var cronometroDeteccion = Stopwatch.StartNew();
                    resultadoDeteccion = _detector.DetectLeaves(rutaImagen, false, opciones.UmbralConfianza, opciones.UmbralIoU, opciones.MaxDetecciones);
                    cronometroDeteccion.Stop();
                    resultado.TiempoDeteccion = cronometroDeteccion.Elapsed;

                    if (!resultadoDeteccion.Success)
                    {
                        resultado.MensajeError = $"Error en detección: {resultadoDeteccion.ErrorMessage}";
                        resultado.HoraFin = DateTime.Now;
                        return resultado;
                    }

                    resultado.NumeroHojasDetectadas = resultadoDeteccion.NumLeaves;
                    resultado.AnchoOriginal = resultadoDeteccion.OriginalWidth;
                    resultado.AltoOriginal = resultadoDeteccion.OriginalHeight;
                }

                // Fase 2: Clasificación de hojas detectadas
                if (opciones.HabilitarClasificacion && _clasificador != null && resultadoDeteccion != null && resultadoDeteccion.Detections.Any())
                {
                    // Extraer regiones de hojas para clasificación en paralelo
                    var hojasClasificadas = await ClasificarHojasDetectadasParalelo(rutaImagen, resultadoDeteccion.Detections);
                    
                    // Calcular tiempo total de clasificación sumando tiempos individuales
                    resultado.TiempoClasificacion = TimeSpan.FromTicks(hojasClasificadas.Sum(h => h.TiempoClasificacion.Ticks));

                    resultado.HojasProcesadas = hojasClasificadas;
                    resultado.NumeroHojasClasificadas = hojasClasificadas.Count(h => h.EstaClasificada);
                }
                else if (resultadoDeteccion != null)
                {
                    // Solo detección, sin clasificación - pero guardamos los datos de imagen
                    resultado.HojasProcesadas = await ExtraerImagenesDeteccionesAsync(rutaImagen, resultadoDeteccion.Detections);
                }

                // Fase 3: Guardar resultados si está habilitado
                if (opciones.GuardarResultados && resultado.HojasProcesadas.Any())
                {
                    var cronometroProcesamiento = Stopwatch.StartNew();
                    resultado.RutaImagenSalida = await GuardarResultadosIntegradosAsync(rutaImagen, resultado.HojasProcesadas, opciones.DirectorioSalida);
                    cronometroProcesamiento.Stop();
                    resultado.TiempoProcesamiento = cronometroProcesamiento.Elapsed;
                }

                resultado.Exitoso = true;
            }
            catch (Exception ex)
            {
                resultado.MensajeError = ex.Message;
            }
            finally
            {
                cronometro.Stop();
                resultado.HoraFin = DateTime.Now;
            }

            return resultado;
        }

        private void InicializarServicios(OpcionesAnalisisIntegrado opciones)
        {
            if (opciones.HabilitarDeteccion && _detector == null && !string.IsNullOrEmpty(opciones.RutaModeloDetector))
            {
                _detector = new TomatoLeafDetector(opciones.RutaModeloDetector);
            }

            if (opciones.HabilitarClasificacion && _clasificador == null && !string.IsNullOrEmpty(opciones.RutaModeloClasificador))
            {
                _clasificador = new TomatoClassifier(opciones.RutaModeloClasificador);
            }
        }

        private async Task<List<DatosHojaIntegrada>> ClasificarHojasDetectadasParalelo(string rutaImagen, List<Detection> detecciones)
        {
            var hojasClasificadas = new ConcurrentBag<DatosHojaIntegrada>();

            using var imagenOriginal = Image.Load<Rgb24>(rutaImagen);

            // Procesar todas las detecciones en paralelo
            await Task.Run(() =>
            {
                Parallel.ForEach(detecciones, new ParallelOptions
                {
                    MaxDegreeOfParallelism = Environment.ProcessorCount
                }, deteccion =>
                {
                    var hojaIntegrada = new DatosHojaIntegrada
                    {
                        Id = deteccion.Id,
                        CajaSuperior = deteccion.BBox,
                        Centro = deteccion.Center,
                        Confianza = deteccion.Confidence,
                        IdClase = deteccion.ClassId,
                        NombreClase = deteccion.ClassName,
                        Area = deteccion.Area,
                        Ancho = deteccion.Width,
                        Alto = deteccion.Height,
                        PorcentajeArea = deteccion.AreaPercentage,
                        ProporcionalidadAspecto = deteccion.AspectRatio
                    };

                    try
                    {
                        var cronometroHoja = Stopwatch.StartNew();
                        
                        // Extraer región de la hoja con padding
                        var padding = 10;
                        var x1 = Math.Max(0, deteccion.BBox[0] - padding);
                        var y1 = Math.Max(0, deteccion.BBox[1] - padding);
                        var x2 = Math.Min(imagenOriginal.Width, deteccion.BBox[2] + padding);
                        var y2 = Math.Min(imagenOriginal.Height, deteccion.BBox[3] + padding);
                        
                        var ancho = x2 - x1;
                        var alto = y2 - y1;
                        
                        Image<Rgb24>? imagenRecortada = null;
                        try
                        {
                            // Crear copia thread-safe de la imagen original
                            lock (imagenOriginal)
                            {
                                imagenRecortada = imagenOriginal.Clone();
                            }
                            
                            imagenRecortada.Mutate(x => x.Crop(new Rectangle(x1, y1, ancho, alto)));
                            
                            // Convertir a bytes para clasificación directa
                            using var memoryStream = new MemoryStream();
                            imagenRecortada.SaveAsPng(memoryStream);
                            var imageBytes = memoryStream.ToArray();
                            
                            // Guardar los datos de la imagen
                            hojaIntegrada.DatosImagen = imageBytes;
                            
                            var resultadoClasificacion = _clasificador.ClassifyImageFromBytes(imageBytes, $"leaf_{deteccion.Id}");
                            
                            if (!string.IsNullOrEmpty(resultadoClasificacion.PredictedClass) && resultadoClasificacion.PredictedClass != "Error")
                            {
                                hojaIntegrada.EtiquetaClasificacion = resultadoClasificacion.PredictedClass;
                                hojaIntegrada.ConfianzaClasificacion = resultadoClasificacion.Confidence;
                                hojaIntegrada.EstaClasificada = true;
                            }
                            else
                            {
                                hojaIntegrada.EtiquetaClasificacion = "Error";
                                hojaIntegrada.ConfianzaClasificacion = 0.0f;
                                hojaIntegrada.EstaClasificada = false;
                            }
                        }
                        finally
                        {
                            imagenRecortada?.Dispose();
                        }
                        
                        cronometroHoja.Stop();
                        hojaIntegrada.TiempoClasificacion = cronometroHoja.Elapsed;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error clasificando hoja {deteccion.Id}: {ex.Message}");
                        hojaIntegrada.EtiquetaClasificacion = "Error";
                        hojaIntegrada.ConfianzaClasificacion = 0.0f;
                        hojaIntegrada.EstaClasificada = false;
                    }

                    hojasClasificadas.Add(hojaIntegrada);
                });
            });

            return hojasClasificadas.OrderBy(h => h.Id).ToList();
        }

        private async Task<List<DatosHojaIntegrada>> ExtraerImagenesDeteccionesAsync(string rutaImagen, List<Detection> detecciones)
        {
            var hojasConImagenes = new List<DatosHojaIntegrada>();

            using var imagenOriginal = Image.Load<Rgb24>(rutaImagen);

            await Task.Run(() =>
            {
                Parallel.ForEach(detecciones, new ParallelOptions
                {
                    MaxDegreeOfParallelism = Environment.ProcessorCount
                }, deteccion =>
                {
                    var hojaIntegrada = new DatosHojaIntegrada
                    {
                        Id = deteccion.Id,
                        CajaSuperior = deteccion.BBox,
                        Centro = deteccion.Center,
                        Confianza = deteccion.Confidence,
                        IdClase = deteccion.ClassId,
                        NombreClase = deteccion.ClassName,
                        Area = deteccion.Area,
                        Ancho = deteccion.Width,
                        Alto = deteccion.Height,
                        PorcentajeArea = deteccion.AreaPercentage,
                        ProporcionalidadAspecto = deteccion.AspectRatio,
                        EstaClasificada = false
                    };

                    try
                    {
                        // Extraer región de la hoja con padding
                        var padding = 10;
                        var x1 = Math.Max(0, deteccion.BBox[0] - padding);
                        var y1 = Math.Max(0, deteccion.BBox[1] - padding);
                        var x2 = Math.Min(imagenOriginal.Width, deteccion.BBox[2] + padding);
                        var y2 = Math.Min(imagenOriginal.Height, deteccion.BBox[3] + padding);
                        
                        var ancho = x2 - x1;
                        var alto = y2 - y1;
                        
                        Image<Rgb24>? imagenRecortada = null;
                        try
                        {
                            // Crear copia thread-safe de la imagen original
                            lock (imagenOriginal)
                            {
                                imagenRecortada = imagenOriginal.Clone();
                            }
                            
                            imagenRecortada.Mutate(x => x.Crop(new Rectangle(x1, y1, ancho, alto)));
                            
                            // Convertir a bytes
                            using var memoryStream = new MemoryStream();
                            imagenRecortada.SaveAsPng(memoryStream);
                            hojaIntegrada.DatosImagen = memoryStream.ToArray();
                        }
                        finally
                        {
                            imagenRecortada?.Dispose();
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error extrayendo imagen de detección {deteccion.Id}: {ex.Message}");
                        hojaIntegrada.DatosImagen = Array.Empty<byte>();
                    }

                    lock (hojasConImagenes)
                    {
                        hojasConImagenes.Add(hojaIntegrada);
                    }
                });
            });

            return hojasConImagenes.OrderBy(h => h.Id).ToList();
        }

        private async Task<string> GuardarResultadosIntegradosAsync(string rutaOriginal, List<DatosHojaIntegrada> hojasProcesadas, string directorioSalida)
        {
            var nombreArchivo = System.IO.Path.GetFileNameWithoutExtension(rutaOriginal);
            var rutaSalida = System.IO.Path.Combine(directorioSalida, $"analisis_integrado_{nombreArchivo}_{DateTime.Now:yyyyMMdd_HHmmss}.png");
            
            Directory.CreateDirectory(directorioSalida);

            using var imagenOriginal = Image.Load<Rgb24>(rutaOriginal);
            var imagenAnotada = imagenOriginal.Clone();

            // Ejecutar el dibujo y guardado en paralelo
            await Task.Run(() =>
            {
                // Dibujar cajas delimitadoras y anotaciones para cada hoja procesada
                imagenAnotada.Mutate(ctx =>
                {
                foreach (var hoja in hojasProcesadas)
                {
                    // Configurar color basado en si está clasificada o no
                    var color = hoja.EstaClasificada ? Color.Green : Color.Red;
                    var colorTexto = Color.White;
                    
                    // Dibujar rectángulo de la caja delimitadora
                    var rectangulo = new RectangleF(hoja.CajaSuperior[0], hoja.CajaSuperior[1], 
                                                   hoja.CajaSuperior[2] - hoja.CajaSuperior[0], 
                                                   hoja.CajaSuperior[3] - hoja.CajaSuperior[1]);
                    
                    ctx.Draw(color, 2, rectangulo);
                    
                    // Preparar texto de anotación
                    var textoAnotacion = $"ID: {hoja.Id}";
                    if (hoja.EstaClasificada)
                    {
                        textoAnotacion += $"\n{hoja.EtiquetaClasificacion} ({hoja.ConfianzaClasificacion:F2})";
                    }
                    else
                    {
                        textoAnotacion += $"\n{hoja.NombreClase} ({hoja.Confianza:F2})";
                    }
                    textoAnotacion += $"\nÁrea: {hoja.PorcentajeArea:F1}%";
                    
                    // Calcular posición del texto (arriba de la caja)
                    var posicionTexto = new PointF(hoja.CajaSuperior[0], Math.Max(0, hoja.CajaSuperior[1] - 60));
                    
                    // Crear opciones de texto - usar fuente con fallback para compatibilidad multiplataforma
                    var font = SystemFonts.CreateFont("DejaVu Sans", 12) ?? 
                              SystemFonts.CreateFont("Liberation Sans", 12) ?? 
                              SystemFonts.CreateFont("Arial", 12) ?? 
                              SystemFonts.Families.First().CreateFont(12);
                    var opcionesTexto = new RichTextOptions(font)
                    {
                        Origin = posicionTexto
                    };
                    
                    // Dibujar fondo semi-transparente para el texto
                    var tamanoTexto = TextMeasurer.MeasureSize(textoAnotacion, opcionesTexto);
                    var rectanguloTexto = new RectangleF(posicionTexto.X, posicionTexto.Y, tamanoTexto.Width + 10, tamanoTexto.Height + 5);
                    ctx.Fill(Color.FromRgba(0, 0, 0, 180), rectanguloTexto);
                    
                    // Dibujar texto
                    ctx.DrawText(opcionesTexto, textoAnotacion, colorTexto);
                    
                    // Dibujar punto central
                    var puntoCentral = new EllipsePolygon(hoja.Centro[0], hoja.Centro[1], 3);
                    ctx.Fill(color, puntoCentral);
                }
                
                // Agregar información general en la esquina superior izquierda
                var infoGeneral = $"Total hojas: {hojasProcesadas.Count}";
                if (hojasProcesadas.Count(h => h.EstaClasificada) > 0)
                {
                    infoGeneral += $"\nClasificadas: {hojasProcesadas.Count(h => h.EstaClasificada)}";
                }
                
                var fontInfo = SystemFonts.CreateFont("DejaVu Sans", 12) ?? 
                              SystemFonts.CreateFont("Liberation Sans", 12) ?? 
                              SystemFonts.CreateFont("Arial", 12) ?? 
                              SystemFonts.Families.First().CreateFont(12);
                var opcionesInfo = new RichTextOptions(fontInfo)
                {
                    Origin = new PointF(15, 15)
                };
                
                var rectanguloInfo = new RectangleF(10, 10, 200, 50);
                ctx.Fill(Color.FromRgba(0, 0, 0, 200), rectanguloInfo);
                ctx.DrawText(opcionesInfo, infoGeneral, Color.White);
                });
                
                imagenAnotada.SaveAsPng(rutaSalida);
            });
            
            imagenAnotada.Dispose();

            return rutaSalida;
        }

        public async Task<List<ResultadoAnalisisIntegrado>> ProcesarImagenesBatchAsync(List<string> rutasImagenes, OpcionesAnalisisIntegrado opciones)
        {
            var resultados = new List<ResultadoAnalisisIntegrado>();
            
            var tareas = rutasImagenes.Select(async rutaImagen =>
            {
                try
                {
                    return await ProcesarImagenAsync(rutaImagen, opciones);
                }
                catch (Exception ex)
                {
                    return new ResultadoAnalisisIntegrado
                    {
                        RutaImagen = rutaImagen,
                        Exitoso = false,
                        MensajeError = ex.Message,
                        HoraInicio = DateTime.Now,
                        HoraFin = DateTime.Now
                    };
                }
            });

            resultados.AddRange(await Task.WhenAll(tareas));
            return resultados.OrderBy(r => r.RutaImagen).ToList();
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _detector?.Dispose();
                _clasificador?.Dispose();
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }
    }

    public class OpcionesAnalisisIntegrado
    {
        public string RutaModeloDetector { get; set; } = string.Empty;
        public string RutaModeloClasificador { get; set; } = string.Empty;
        public bool HabilitarDeteccion { get; set; } = true;
        public bool HabilitarClasificacion { get; set; } = true;
        public bool GuardarResultados { get; set; } = true;
        public string DirectorioSalida { get; set; } = ".";
        public float UmbralConfianza { get; set; } = 0.007f;
        public float UmbralIoU { get; set; } = 0.65f;
        public int MaxDetecciones { get; set; } = 15;
        public bool GenerarReporte { get; set; } = true;
    }

    public class ResultadoAnalisisIntegrado
    {
        public string RutaImagen { get; set; } = string.Empty;
        public DateTime HoraInicio { get; set; }
        public DateTime HoraFin { get; set; }
        public TimeSpan TiempoTotal => HoraFin - HoraInicio;
        public double TiempoTotalMs => TiempoTotal.TotalMilliseconds;
        public double TiempoTotalSegundos => TiempoTotal.TotalSeconds;
        public bool Exitoso { get; set; }
        public string MensajeError { get; set; } = string.Empty;

        public int NumeroHojasDetectadas { get; set; }
        public int NumeroHojasClasificadas { get; set; }
        public List<DatosHojaIntegrada> HojasProcesadas { get; set; } = new();
        public int AnchoOriginal { get; set; }
        public int AltoOriginal { get; set; }
        public string RutaImagenSalida { get; set; } = string.Empty;
        public TimeSpan TiempoDeteccion { get; set; }
        public TimeSpan TiempoClasificacion { get; set; }
        public TimeSpan TiempoProcesamiento { get; set; }
    }

    public class DatosHojaIntegrada
    {
        public int Id { get; set; }
        public int[] CajaSuperior { get; set; } = new int[4];
        public int[] Centro { get; set; } = new int[2];
        public float Confianza { get; set; }
        public int IdClase { get; set; }
        public string NombreClase { get; set; } = string.Empty;
        public int Area { get; set; }
        public int Ancho { get; set; }
        public int Alto { get; set; }
        public float PorcentajeArea { get; set; }
        public float ProporcionalidadAspecto { get; set; }
        
        // Datos de clasificación
        public string EtiquetaClasificacion { get; set; } = string.Empty;
        public float ConfianzaClasificacion { get; set; }
        public bool EstaClasificada { get; set; }
        public TimeSpan TiempoClasificacion { get; set; }
        
        // Datos de imagen
        public byte[] DatosImagen { get; set; } = Array.Empty<byte>();
    }
}