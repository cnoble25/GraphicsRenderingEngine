using System;
using System.Globalization;
using Avalonia.Data.Converters;

namespace GraphicsRendererUI.Utils
{
    /// <summary>
    /// Converter that displays "Degrees" when true, "Radians" when false
    /// </summary>
    public class BooleanToDegreesRadiansConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value is bool useDegrees)
            {
                return useDegrees ? "Degrees" : "Radians";
            }
            return "Degrees";
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            // Not needed for one-way conversion
            throw new NotImplementedException();
        }
    }
}
