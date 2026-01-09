using System;
using System.Globalization;
using Avalonia.Data.Converters;

namespace GraphicsRendererUI.Utils
{
    /// <summary>
    /// Converter for integer values that handles partial input
    /// </summary>
    public class IntConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            if (value == null)
                return string.Empty;
            
            if (value is int i)
            {
                // Extract default value from parameter if provided (format: "KeyName:DefaultValue")
                int defaultValue = 0;
                if (parameter != null)
                {
                    string paramStr = parameter.ToString() ?? "";
                    if (paramStr.Contains(":"))
                    {
                        string[] parts = paramStr.Split(':');
                        if (parts.Length > 1 && int.TryParse(parts[1], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out int parsedDefault))
                        {
                            defaultValue = parsedDefault;
                        }
                    }
                }
                
                // If value equals default, return empty string to show watermark
                if (i == defaultValue)
                {
                    return string.Empty;
                }
                
                return i.ToString(culture);
            }
            
            return value.ToString() ?? string.Empty;
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            // Extract default value from parameter if provided
            int defaultValue = 0;
            if (parameter != null)
            {
                string paramStr = parameter.ToString() ?? "";
                if (paramStr.Contains(":"))
                {
                    string[] parts = paramStr.Split(':');
                    if (parts.Length > 1 && int.TryParse(parts[1], NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out int parsedDefault))
                    {
                        defaultValue = parsedDefault;
                    }
                }
            }
            
            if (value == null)
                return defaultValue;
            
            string? stringValue = value.ToString()?.Trim();
            
            if (string.IsNullOrWhiteSpace(stringValue))
                return defaultValue;
            
            // Allow partial input like "-" - valid while typing
            if (stringValue == "-")
                return defaultValue;
            
            // Try to parse the value
            if (int.TryParse(stringValue, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out int result))
            {
                return result;
            }
            
            // Try with current culture as fallback
            if (int.TryParse(stringValue, NumberStyles.AllowLeadingSign, culture, out result))
            {
                return result;
            }
            
            // If can't parse, return default to prevent binding errors
            return defaultValue;
        }
    }
}
