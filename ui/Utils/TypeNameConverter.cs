using System;
using Avalonia.Data.Converters;
using GraphicsRendererUI.Models;

namespace GraphicsRendererUI.Utils
{
    public class TypeNameConverter : IValueConverter
    {
        public object? Convert(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo culture)
        {
            if (value == null) return "";
            
            if (value is Light)
                return "(Light)";
            else if (value is SceneObject)
                return "(Object)";
            
            return "";
        }

        public object? ConvertBack(object? value, Type targetType, object? parameter, System.Globalization.CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
