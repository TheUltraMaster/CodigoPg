using System;
using System.IO;
using System.Threading.Tasks;
using System.Linq;
using DotnetCoreYolo.AnalisisIntegrado;

namespace DotnetCoreYolo
{
    class Programa
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("🍅 Sistema de Análisis Integrado de Tomates");
            Console.WriteLine("🔬 Detección y Clasificación Automática de Hojas");
            Console.WriteLine(new string('=', 70));

          
                await EjecutarModoInteractivo();
            
        }

        static async Task EjecutarModoInteractivo()
        {
            while (true)
            {
                Console.Clear();
                Console.WriteLine("\n" + new string('=', 70));
                Console.WriteLine("🍅 SISTEMA DE ANÁLISIS INTEGRADO DE TOMATES");
                Console.WriteLine(new string('=', 70));
                Console.WriteLine("🔬 Análisis completo: detección + clasificación de hojas");
                Console.WriteLine("🎨 c. Obtener color RGB del centro de imagen");
                Console.WriteLine("q. Salir");
                Console.WriteLine(new string('=', 70));

                Console.Write("\nElige una opción (Enter/c/q): ");
                var opcion = Console.ReadLine()?.Trim();

                if (opcion == "q" || opcion == "Q")
                {
                    Console.WriteLine("👋 ¡Hasta luego!");
                    return;
                }
                else if (opcion == "c" || opcion == "C")
                {
                    await EjecutarExtractorColor();
                }
                else
                {
                    await EjecutarModuloIntegrado();
                }
            }
        }

        static async Task EjecutarModuloIntegrado()
        {
           
            using var servicioIntegrado = new ServicioAnalisisIntegrado();

            while (true)
            {
                Console.Clear();
                Console.WriteLine("\n" + new string('=', 70));
                Console.WriteLine("🔬 ANÁLISIS INTEGRADO - DETECCIÓN + CLASIFICACIÓN");
                Console.WriteLine(new string('=', 70));
                Console.WriteLine("m. Menú principal");
                Console.WriteLine(new string('=', 70));

                Console.Write("\n📂 Ruta de la imagen (o 'm' para menú): ");
                var entrada = Console.ReadLine()?.Trim();

                if (entrada == "m" || entrada == "M")
                    break;

                if (!string.IsNullOrEmpty(entrada))
                {
                    var opciones = new OpcionesAnalisisIntegrado
                    {
                        HabilitarDeteccion = true,
                        HabilitarClasificacion = true,
                        GuardarResultados = true,
                        DirectorioSalida = "."
                    };

                    var resultado = servicioIntegrado.ProcesarImagen(entrada, opciones);
                    MostrarResultadosIntegrados(resultado);
                }

                Console.WriteLine("\nPresiona Enter para continuar...");
                Console.ReadLine();
            }
        }

        static async Task EjecutarExtractorColor()
        {
            using var servicioIntegrado = new ServicioAnalisisIntegrado();

            while (true)
            {
                Console.Clear();
                Console.WriteLine("\n" + new string('=', 70));
                Console.WriteLine("🎨 EXTRACTOR DE COLOR RGB EN CENTRO DE IMAGEN");
                Console.WriteLine(new string('=', 70));
                Console.WriteLine("m. Menú principal");
                Console.WriteLine(new string('=', 70));

                Console.Write("\n📂 Ruta de la imagen (o 'm' para menú): ");
                var entrada = Console.ReadLine()?.Trim();

                if (entrada == "m" || entrada == "M")
                    break;

                if (!string.IsNullOrEmpty(entrada))
                {
                    try
                    {
                        var colorRGB = servicioIntegrado.ObtenerColorRGBEnCentro(entrada);
                        
                        Console.WriteLine($"\n{'=' * 50}");
                        Console.WriteLine("🎨 COLOR RGB EN EL CENTRO DE LA IMAGEN");
                        Console.WriteLine($"{'=' * 50}");
                        Console.WriteLine($"📍 Archivo: {Path.GetFileName(entrada)}");
                        Console.WriteLine($"🔴 Rojo (R):   {colorRGB.R}");
                        Console.WriteLine($"🟢 Verde (G):  {colorRGB.G}");
                        Console.WriteLine($"🔵 Azul (B):   {colorRGB.B}");
                        Console.WriteLine($"🎨 RGB: {colorRGB.R},{colorRGB.G},{colorRGB.B}");
                        
                        // Mostrar información adicional del color
                        var intensidad = (colorRGB.R + colorRGB.G + colorRGB.B) / 3.0;
                        var esClaro = intensidad > 127;
                        Console.WriteLine($"💡 Intensidad promedio: {intensidad:F1} ({(esClaro ? "Claro" : "Oscuro")})");
                    }
                    catch (FileNotFoundException)
                    {
                        Console.WriteLine("❌ Error: No se encontró la imagen especificada");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"❌ Error: {ex.Message}");
                    }
                }

                Console.WriteLine("\nPresiona Enter para continuar...");
                Console.ReadLine();
            }
        }

        static void MostrarResultadosIntegrados(ResultadoAnalisisIntegrado resultado)
        {
            Console.WriteLine($"\n{'=' * 70}");
            Console.WriteLine($"🔬 RESULTADOS DEL ANÁLISIS INTEGRADO");
            Console.WriteLine($"{'=' * 70}");

            if (resultado.Exitoso)
            {
                Console.WriteLine($"🍃 Hojas detectadas: {resultado.NumeroHojasDetectadas}");
                Console.WriteLine($"🔬 Hojas clasificadas: {resultado.NumeroHojasClasificadas}");
                Console.WriteLine($"📐 Dimensiones imagen: {resultado.AnchoOriginal}x{resultado.AltoOriginal} px");

                Console.WriteLine($"\n⏱️  TIEMPOS DE PROCESAMIENTO:");
                Console.WriteLine($"  - Detección: {resultado.TiempoDeteccion.TotalMilliseconds:F0} ms");
                Console.WriteLine($"  - Clasificación: {resultado.TiempoClasificacion.TotalMilliseconds:F0} ms");
                Console.WriteLine($"  - Procesamiento de imagen: {resultado.TiempoProcesamiento.TotalMilliseconds:F0} ms");
                Console.WriteLine($"  - Tiempo total: {resultado.TiempoTotalMs:F0} ms ({resultado.TiempoTotalSegundos:F2} seg)");

                // Mostrar información detallada de cada hoja
                if (resultado.HojasProcesadas.Any())
                {
                    Console.WriteLine($"\n🍃 DETALLES DE CADA HOJA:");
                    Console.WriteLine(new string('-', 70));


                    foreach (var hoja in resultado.HojasProcesadas.OrderBy(h => h.Id))
                    {
                        Console.WriteLine($"\n🍃 HOJA #{hoja.Id}:");
                        
                        // Información básica
                        Console.WriteLine($"  📏 Dimensiones: {hoja.Ancho}x{hoja.Alto} px");
                        Console.WriteLine($"  📐 Área: {hoja.Area} px² ({hoja.PorcentajeArea:F1}% de la imagen)");
                        
                        // Enfermedad detectada
                        if (hoja.EstaClasificada)
                        {
                            Console.WriteLine($"  🦠 Enfermedad: {hoja.EtiquetaClasificacion}");
                            Console.WriteLine($"  📊 Confianza clasificación: {hoja.ConfianzaClasificacion:P1}");
                        }
                        else
                        {
                            Console.WriteLine($"  🦠 Enfermedad: No clasificada");
                            Console.WriteLine($"  📊 Confianza detección: {hoja.Confianza:P1}");
                        }

                        // Color detectado (usando el centro de la región de la hoja)
                        if (hoja.DatosImagen != null && hoja.DatosImagen.Length > 0)
                        {
                            try
                            {
                                // Crear imagen temporal para obtener el color
                                using var memoryStream = new MemoryStream(hoja.DatosImagen);
                                using var imagen = SixLabors.ImageSharp.Image.Load<SixLabors.ImageSharp.PixelFormats.Rgb24>(memoryStream);
                                
                                // Obtener color del centro de la región de la hoja
                                var centroX = imagen.Width / 2;
                                var centroY = imagen.Height / 2;
                                var pixel = imagen[centroX, centroY];
                                
                                Console.WriteLine($"  🎨 Color centro: RGB({pixel.R}, {pixel.G}, {pixel.B})");
                                
                                // Calcular iluminación (intensidad promedio)
                                var intensidad = (pixel.R + pixel.G + pixel.B) / 3.0;
                                var tipoIluminacion = intensidad < 85 ? "Baja" : intensidad < 170 ? "Media" : "Alta";
                                Console.WriteLine($"  💡 Iluminación: {intensidad:F1} ({tipoIluminacion})");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"  🎨 Color: Error al obtener color ({ex.Message})");
                                Console.WriteLine($"  💡 Iluminación: No disponible");
                            }
                        }
                        else
                        {
                            Console.WriteLine($"  🎨 Color: No disponible");
                            Console.WriteLine($"  💡 Iluminación: No disponible");
                        }
                        
                        Console.WriteLine(new string('-', 50));
                    }

                    // Resumen estadístico
                    var gruposClase = resultado.HojasProcesadas.Where(h => h.EstaClasificada)
                        .GroupBy(h => h.EtiquetaClasificacion)
                        .OrderByDescending(g => g.Count());

                    if (gruposClase.Any())
                    {
                        Console.WriteLine($"\n📊 RESUMEN DE ENFERMEDADES:");
                        foreach (var grupo in gruposClase)
                        {
                            var confianzaPromedio = grupo.Average(h => h.ConfianzaClasificacion);
                            Console.WriteLine($"  - {grupo.Key}: {grupo.Count()} hojas (confianza: {confianzaPromedio:P1})");
                        }
                    }
                }

                if (!string.IsNullOrEmpty(resultado.RutaImagenSalida))
                {
                    Console.WriteLine($"\n💾 Imagen guardada: {resultado.RutaImagenSalida}");
                }
            }
            else
            {
                Console.WriteLine($"❌ Error: {resultado.MensajeError}");
            }
        }

    }
}