using System;
using System.Globalization;
using Avalonia.Data.Converters;

namespace GraphicsRendererUI.Utils
{
    /// <summary>
    /// Converter for double values that properly handles negative numbers and partial input
    /// </summary>
    public class DoubleConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value == null)
                return "0";
            
            if (value is double d)
                return d.ToString("G", culture);
            
            return value.ToString() ?? "0";
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value == null)
                return 0.0;
            
            string? stringValue = value.ToString()?.Trim();
            if (string.IsNullOrWhiteSpace(stringValue))
                return 0.0;
            
            // Allow partial input like "-" or "-." or "." - these are valid while typing
            // We'll return 0.0 temporarily, but the TextBox will keep the text
            if (stringValue == "-" || stringValue == "." || stringValue == "-." || stringValue == "-0" || stringValue == "-0.")
            {
                // Return 0.0 but don't prevent the user from continuing to type
                // The TextBox will maintain the text value
                return 0.0;
            }
            
            // Try to parse the value, allowing negative numbers
            // Use InvariantCulture to ensure consistent parsing regardless of system locale
            if (double.TryParse(stringValue, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent, CultureInfo.InvariantCulture, out double result))
                return result;
            
            // If parsing fails completely, try with current culture as fallback
            if (double.TryParse(stringValue, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent, culture, out result))
                return result;
            
            // If still can't parse, return 0.0 to prevent binding errors
            // This allows the user to continue typing
            return 0.0;
        }
    }
}
