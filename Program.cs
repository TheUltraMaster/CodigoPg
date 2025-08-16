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
            
            if (args.Length > 0)
            {
                await EjecutarModoLineaComandos(args);
            }
            else
            {
                await EjecutarModoInteractivo();
            }
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
                Console.WriteLine("q. Salir");
                Console.WriteLine(new string('=', 70));
                
                Console.Write("\nPresiona Enter para analizar imagen o 'q' para salir: ");
                var opcion = Console.ReadLine()?.Trim();
                
                if (opcion == "q" || opcion == "Q")
                {
                    Console.WriteLine("👋 ¡Hasta luego!");
                    return;
                }
                
                await EjecutarModuloIntegrado();
            }
        }
        
        static async Task EjecutarModuloIntegrado()
        {
            Console.Write("📂 Ruta del modelo detector: ");
            var rutaDetector = Console.ReadLine()?.Trim();
            
            Console.Write("📂 Ruta del modelo clasificador: ");
            var rutaClasificador = Console.ReadLine()?.Trim();
            
            if (!File.Exists(rutaDetector) || !File.Exists(rutaClasificador))
            {
                Console.WriteLine("❌ Uno o ambos modelos no encontrados");
                Console.WriteLine("Presiona Enter para continuar...");
                Console.ReadLine();
                return;
            }
            
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
                        RutaModeloDetector = rutaDetector,
                        RutaModeloClasificador = rutaClasificador,
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
        
        static void MostrarResultadosIntegrados(ResultadoAnalisisIntegrado resultado)
        {
            Console.WriteLine($"\n{'='*70}");
            Console.WriteLine($"🔬 RESULTADOS DEL ANÁLISIS INTEGRADO");
            Console.WriteLine($"{'='*70}");
            
            if (resultado.Exitoso)
            {
                Console.WriteLine($"🍃 Hojas detectadas: {resultado.NumeroHojasDetectadas}");
                Console.WriteLine($"🔬 Hojas clasificadas: {resultado.NumeroHojasClasificadas}");
                Console.WriteLine($"📐 Dimensiones: {resultado.AnchoOriginal}x{resultado.AltoOriginal} px");
                
                Console.WriteLine($"\n⏱️  TIEMPOS DE PROCESAMIENTO:");
                Console.WriteLine($"  - Detección: {resultado.TiempoDeteccion.TotalMilliseconds:F0} ms");
                Console.WriteLine($"  - Clasificación: {resultado.TiempoClasificacion.TotalMilliseconds:F0} ms");
                Console.WriteLine($"  - Procesamiento de imagen: {resultado.TiempoProcesamiento.TotalMilliseconds:F0} ms");
                Console.WriteLine($"  - Tiempo total: {resultado.TiempoTotalMs:F0} ms ({resultado.TiempoTotalSegundos:F2} seg)");
                
                if (resultado.HojasProcesadas.Any())
                {
                    var gruposClase = resultado.HojasProcesadas.Where(h => h.EstaClasificada)
                        .GroupBy(h => h.EtiquetaClasificacion)
                        .OrderByDescending(g => g.Count());
                    
                    Console.WriteLine($"\n📊 DISTRIBUCIÓN DE ENFERMEDADES:");
                    foreach (var grupo in gruposClase)
                    {
                        var confianzaPromedio = grupo.Average(h => h.ConfianzaClasificacion);
                        Console.WriteLine($"  - {grupo.Key}: {grupo.Count()} hojas (confianza: {confianzaPromedio:P1})");
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
        
        static async Task EjecutarModoLineaComandos(string[] args)
        {
            Console.WriteLine("🔄 Modo línea de comandos no implementado");
            Console.WriteLine("📝 Use el modo interactivo para acceder al análisis integrado");
        }
    }
}