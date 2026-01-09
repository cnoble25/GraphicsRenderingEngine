using System;
using System.Collections.Generic;
using System.Globalization;
using Avalonia.Data.Converters;

namespace GraphicsRendererUI.Utils
{
    /// <summary>
    /// Converter for double values that properly handles negative numbers and partial input
    /// Uses parameter as a unique identifier to preserve partial input per TextBox
    /// </summary>
    public class DoubleConverter : IValueConverter
    {
        // Store text values per binding identifier (parameter) to preserve partial input
        private static readonly Dictionary<string, string> _partialInputs = new Dictionary<string, string>();
        private static readonly object _lock = new object();

        public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            string key = parameter?.ToString() ?? "";
            
            if (value == null)
            {
                lock (_lock) { _partialInputs.Remove(key); }
                return string.Empty;
            }
            
            if (value is double d)
            {
                // Extract default value from parameter if provided (format: "KeyName:DefaultValue")
                double defaultValue = 0.0;
                string paramKey = key;
                if (key.Contains(":"))
                {
                    string[] parts = key.Split(':');
                    paramKey = parts[0];
                    if (parts.Length > 1 && double.TryParse(parts[1], NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out double parsedDefault))
                    {
                        defaultValue = parsedDefault;
                    }
                }
                
                // If value equals default, check if we have stored partial input for this binding
                if (Math.Abs(d - defaultValue) < 0.0001) // Use epsilon for floating point comparison
                {
                    lock (_lock)
                    {
                        if (_partialInputs.TryGetValue(paramKey, out string? storedText))
                        {
                            string trimmed = storedText.Trim();
                            // If this is a partial input that should be preserved, return it
                            // Include "0" and "0." to preserve leading zeros for decimals
                            if (trimmed == "-" || trimmed == "." || trimmed == "-." || 
                                trimmed == "-0" || trimmed == "-0." ||
                                trimmed == "0" || trimmed == "0." ||
                                (defaultValue != 0.0 && trimmed == defaultValue.ToString("G", culture)))
                            {
                                return trimmed;
                            }
                            // Not a partial input, remove it
                            _partialInputs.Remove(paramKey);
                        }
                    }
                    // No partial input found, return empty for watermark
                    return string.Empty;
                }
                
                // For non-default values, check if we should preserve the original format
                // This helps with decimals like "0.5" - preserve the format if it starts with "0."
                lock (_lock)
                {
                    if (_partialInputs.TryGetValue(paramKey, out string? storedText))
                    {
                        string trimmed = storedText.Trim();
                        // If the stored text starts with "0." and parses to this value, preserve it
                        // This handles cases like "0.5" where we want to keep "0.5" not "0.5" (same, but preserves format)
                        // Actually, for non-zero values, we should just use the parsed value's string representation
                        // But we want to preserve leading zeros in decimals
                        if (trimmed.StartsWith("0.") && double.TryParse(trimmed, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out double parsed) && parsed == d)
                        {
                            // Keep the stored format for values starting with "0."
                            _partialInputs.Remove(paramKey);
                            return trimmed;
                        }
                    }
                    // Clear stored partial input for other non-default values
                    _partialInputs.Remove(paramKey);
                }
                return d.ToString("G", culture);
            }
            
            return value.ToString() ?? string.Empty;
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
        {
            string key = parameter?.ToString() ?? "";
            
            // Extract default value from parameter if provided
            double defaultValue = 0.0;
            string paramKey = key;
            if (key.Contains(":"))
            {
                string[] parts = key.Split(':');
                paramKey = parts[0];
                if (parts.Length > 1 && double.TryParse(parts[1], NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out double parsedDefault))
                {
                    defaultValue = parsedDefault;
                }
            }
            
            if (value == null)
            {
                lock (_lock) { _partialInputs.Remove(paramKey); }
                return defaultValue;
            }
            
            string? stringValue = value.ToString()?.Trim();
            
            if (string.IsNullOrWhiteSpace(stringValue))
            {
                lock (_lock) { _partialInputs.Remove(paramKey); }
                return defaultValue;
            }
            
            // Allow partial input like "-", ".", "0", "0." - these are valid while typing
            if (stringValue == "-" || stringValue == "." || stringValue == "-." || 
                stringValue == "-0" || stringValue == "-0." ||
                stringValue == "0" || stringValue == "0." ||
                stringValue == defaultValue.ToString("G", culture))
            {
                // Store this partial input for this specific binding
                lock (_lock)
                {
                    _partialInputs[paramKey] = stringValue;
                }
                return defaultValue;
            }
            
            // Try to parse the value, allowing negative numbers
            // Use InvariantCulture to ensure consistent parsing regardless of system locale
            if (double.TryParse(stringValue, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent, CultureInfo.InvariantCulture, out double result))
            {
                // If successfully parsed, store the text format if it's default or starts with "0."
                // This preserves leading zeros for decimals like "0.5"
                lock (_lock)
                {
                    if (Math.Abs(result - defaultValue) < 0.0001)
                    {
                        // Store default value format to preserve it
                        _partialInputs[paramKey] = stringValue;
                    }
                    else if (stringValue.StartsWith("0.") || stringValue.StartsWith("-0."))
                    {
                        // Store values starting with "0." to preserve the format
                        _partialInputs[paramKey] = stringValue;
                    }
                    else
                    {
                        // Clear for other values
                        _partialInputs.Remove(paramKey);
                    }
                }
                return result;
            }
            
            // If parsing fails completely, try with current culture as fallback
            if (double.TryParse(stringValue, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint | NumberStyles.AllowExponent, culture, out result))
            {
                lock (_lock)
                {
                    if (Math.Abs(result - defaultValue) < 0.0001)
                    {
                        _partialInputs[paramKey] = stringValue;
                    }
                    else if (stringValue.StartsWith("0.") || stringValue.StartsWith("-0."))
                    {
                        _partialInputs[paramKey] = stringValue;
                    }
                    else
                    {
                        _partialInputs.Remove(paramKey);
                    }
                }
                return result;
            }
            
            // If still can't parse, store as partial input and return default to prevent binding errors
            lock (_lock)
            {
                _partialInputs[paramKey] = stringValue;
            }
            return defaultValue;
        }
    }
}
