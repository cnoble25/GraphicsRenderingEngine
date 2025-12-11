using System;
using System.IO;
using Avalonia.Media.Imaging;
using Avalonia.Platform;

namespace GraphicsRendererUI.Utils
{
    public static class JpgDecoder
    {
        /// <summary>
        /// Decodes a JPG file into an Avalonia Bitmap
        /// Uses Avalonia's built-in JPG support
        /// </summary>
        public static Bitmap? DecodeJpg(string filePath)
        {
            try
            {
                if (!File.Exists(filePath))
                {
                    throw new FileNotFoundException($"JPG file not found: {filePath}");
                }

                // Check file size to ensure it's not empty
                var fileInfo = new FileInfo(filePath);
                if (fileInfo.Length == 0)
                {
                    throw new InvalidDataException($"JPG file is empty: {filePath}");
                }

                // Check if file has valid JPG header (starts with FF D8 FF)
                using (var headerCheck = new FileStream(filePath, FileMode.Open, FileAccess.Read))
                {
                    byte[] header = new byte[3];
                    int bytesRead = headerCheck.Read(header, 0, 3);
                    if (bytesRead < 3 || header[0] != 0xFF || header[1] != 0xD8 || header[2] != 0xFF)
                    {
                        throw new InvalidDataException($"File does not appear to be a valid JPG file. Header: {BitConverter.ToString(header)}");
                    }
                }

                // Try multiple methods to load the bitmap
                Bitmap? bitmap = null;
                
                // Method 1: Direct file path (simplest)
                try
                {
                    bitmap = new Bitmap(filePath);
                    if (bitmap != null)
                    {
                        return bitmap;
                    }
                }
                catch (Exception ex1)
                {
                    // If that fails, try FileStream method
                    try
                    {
                        using (var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
                        {
                            bitmap = new Bitmap(fileStream);
                            if (bitmap != null)
                            {
                                return bitmap;
                            }
                        }
                    }
                    catch (Exception ex2)
                    {
                        // If both fail, try reading into memory first
                        byte[] imageData = File.ReadAllBytes(filePath);
                        using (var memoryStream = new MemoryStream(imageData))
                        {
                            bitmap = new Bitmap(memoryStream);
                            return bitmap;
                        }
                    }
                }
                
                return bitmap;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to decode JPG file: {ex.Message}", ex);
            }
        }
    }
}
