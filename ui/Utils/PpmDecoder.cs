using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Avalonia;
using Avalonia.Media.Imaging;
using Avalonia.Platform;

namespace GraphicsRendererUI.Utils
{
    public static class PpmDecoder
    {
        /// <summary>
        /// Decodes a PPM (P3 format) file into an Avalonia Bitmap
        /// </summary>
        public static Bitmap? DecodePpm(string filePath)
        {
            try
            {
                using var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read);
                using var reader = new StreamReader(fileStream);

                // Read magic number (should be "P3")
                string? magicNumber = reader.ReadLine()?.Trim();
                if (magicNumber != "P3")
                {
                    throw new InvalidDataException($"Unsupported PPM format: {magicNumber}. Only P3 format is supported.");
                }

                // Read dimensions
                string? dimensionsLine;
                do
                {
                    dimensionsLine = reader.ReadLine()?.Trim();
                } while (dimensionsLine != null && (dimensionsLine.StartsWith("#") || string.IsNullOrWhiteSpace(dimensionsLine)));

                if (dimensionsLine == null)
                {
                    throw new InvalidDataException("Invalid PPM file: missing dimensions");
                }

                var dimensionParts = dimensionsLine.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                if (dimensionParts.Length < 2)
                {
                    throw new InvalidDataException($"Invalid PPM file: invalid dimensions line '{dimensionsLine}'");
                }

                if (!int.TryParse(dimensionParts[0], out int width) || width <= 0)
                {
                    throw new InvalidDataException($"Invalid PPM file: failed to parse width '{dimensionParts[0]}'");
                }
                if (!int.TryParse(dimensionParts[1], out int height) || height <= 0)
                {
                    throw new InvalidDataException($"Invalid PPM file: failed to parse height '{dimensionParts[1]}'");
                }

                // Read max value (should be 255)
                string? maxValueLine;
                do
                {
                    maxValueLine = reader.ReadLine()?.Trim();
                } while (maxValueLine != null && (maxValueLine.StartsWith("#") || string.IsNullOrWhiteSpace(maxValueLine)));

                if (maxValueLine == null)
                {
                    throw new InvalidDataException("Invalid PPM file: missing max value");
                }

                if (!int.TryParse(maxValueLine, out int maxValue))
                {
                    throw new InvalidDataException($"Invalid PPM file: failed to parse max value '{maxValueLine}'");
                }
                if (maxValue != 255)
                {
                    throw new InvalidDataException($"Unsupported max value: {maxValue}. Only 255 is supported.");
                }

                // Read pixel data
                var pixels = new byte[width * height * 4]; // BGRA format for Avalonia

                // Read all remaining data
                var pixelData = new StringBuilder();
                string? line;
                while ((line = reader.ReadLine()) != null)
                {
                    // Skip comments
                    if (line.Trim().StartsWith("#"))
                        continue;
                    pixelData.Append(line).Append(' ');
                }

                var pixelValues = pixelData.ToString().Split(new[] { ' ', '\t', '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

                // Validate we have enough pixel values
                int expectedPixelValues = width * height * 3;
                if (pixelValues.Length < expectedPixelValues)
                {
                    throw new InvalidDataException($"Invalid PPM file: expected {expectedPixelValues} pixel values, but found {pixelValues.Length}");
                }

                // The renderer writes pixels right-to-left within each row, so we need to reverse each row
                // Convert RGB to BGRA format (Avalonia uses BGRA)
                int totalPixels = width * height;
                int pixelValueIndex = 0;

                for (int row = 0; row < height; row++)
                {
                    // Read the row pixels (they're stored right-to-left in the file)
                    var rowPixels = new List<(byte r, byte g, byte b)>();
                    for (int col = 0; col < width; col++)
                    {
                        if (pixelValueIndex + 2 >= pixelValues.Length)
                        {
                            throw new InvalidDataException($"Invalid PPM file: insufficient pixel data at row {row}, col {col}. Expected {expectedPixelValues} values, found {pixelValues.Length}");
                        }

                        if (!byte.TryParse(pixelValues[pixelValueIndex++], out byte r))
                        {
                            throw new InvalidDataException($"Invalid PPM file: failed to parse red value at index {pixelValueIndex - 1}: '{pixelValues[pixelValueIndex - 1]}'");
                        }
                        if (!byte.TryParse(pixelValues[pixelValueIndex++], out byte g))
                        {
                            throw new InvalidDataException($"Invalid PPM file: failed to parse green value at index {pixelValueIndex - 1}: '{pixelValues[pixelValueIndex - 1]}'");
                        }
                        if (!byte.TryParse(pixelValues[pixelValueIndex++], out byte b))
                        {
                            throw new InvalidDataException($"Invalid PPM file: failed to parse blue value at index {pixelValueIndex - 1}: '{pixelValues[pixelValueIndex - 1]}'");
                        }
                        rowPixels.Add((r, g, b));
                    }

                    // Reverse the row to get left-to-right order
                    rowPixels.Reverse();

                    // Write the row to the pixel buffer (left-to-right)
                    for (int col = 0; col < rowPixels.Count; col++)
                    {
                        int pixelIdx = (row * width + col) * 4;
                        if (pixelIdx + 3 < pixels.Length)
                        {
                            pixels[pixelIdx + 0] = rowPixels[col].b;  // B
                            pixels[pixelIdx + 1] = rowPixels[col].g;  // G
                            pixels[pixelIdx + 2] = rowPixels[col].r;  // R
                            pixels[pixelIdx + 3] = 255;                // A
                        }
                    }
                }

                // Create bitmap from pixel data
                var writeableBitmap = new WriteableBitmap(
                    new PixelSize(width, height),
                    new Vector(96, 96),
                    PixelFormat.Bgra8888,
                    AlphaFormat.Opaque);

                using (var lockedBitmap = writeableBitmap.Lock())
                {
                    System.Runtime.InteropServices.Marshal.Copy(pixels, 0, lockedBitmap.Address, pixels.Length);
                }

                return writeableBitmap;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to decode PPM file: {ex.Message}", ex);
            }
        }
    }
}
