# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This is a .NET 8.0 console application using the standard Microsoft .NET SDK.

### Building and Running
- `dotnet build` - Build the project
- `dotnet run` - Run the application
- `dotnet build --configuration Release` - Build for release

### Development
- `dotnet restore` - Restore NuGet packages
- `dotnet clean` - Clean build artifacts

## Project Structure

This is a minimal .NET console application with the following structure:
- `Program.cs` - Main entry point (currently a basic "Hello, World!" application)
- `DotnetCoreYolo.csproj` - Project file targeting .NET 8.0
- `DotnetCoreYolo.sln` - Visual Studio solution file
- `modelo.pt` - PyTorch model file (likely for YOLO object detection)

The project appears to be set up for YOLO (You Only Look Once) object detection integration, though the current implementation is minimal. The presence of `modelo.pt` suggests this will be a .NET application that loads and uses a pre-trained YOLO model.

## Key Configuration
- Target Framework: .NET 8.0
- Output Type: Console application
- Nullable reference types enabled
- Implicit usings enabled